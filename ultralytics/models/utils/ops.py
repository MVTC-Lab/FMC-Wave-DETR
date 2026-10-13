# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ultralytics.utils.metrics import bbox_iou, probiou
from ultralytics.utils.ops import safe_normalize_angle_vector, xywhr2xyxyxyxy, xywh2xyxy, xyxy2xywh


_MIN_OBB_SIZE = 1.0e-4


def _prepare_rboxes_fp32(rboxes: torch.Tensor) -> torch.Tensor:
    """Promote ``xywhr`` boxes to FP32 and protect tiny widths/heights for AMP-safe geometry."""
    if rboxes.shape[-1] != 5:
        raise ValueError(f"rboxes must end with 5 values [cx,cy,w,h,theta], got {tuple(rboxes.shape)}")
    boxes = rboxes.float()
    return torch.cat(
        (
            boxes[..., :2],
            boxes[..., 2:4].clamp_min(_MIN_OBB_SIZE),
            boxes[..., 4:5],
        ),
        dim=-1,
    )


def theta_to_angle_vector(theta: torch.Tensor) -> torch.Tensor:
    """Convert rectangle angle theta (radians) to a pi-periodic [sin(2θ), cos(2θ)] vector."""
    theta = theta.squeeze(-1) if theta.ndim > 1 and theta.shape[-1] == 1 else theta
    theta = theta.float()
    return torch.stack((torch.sin(2.0 * theta), torch.cos(2.0 * theta)), dim=-1)


def angle_vector_to_theta(angle_vector: torch.Tensor) -> torch.Tensor:
    """Convert [sin(2θ), cos(2θ)] to theta in [-pi/2, pi/2)."""
    angle_vector = safe_normalize_angle_vector(angle_vector)
    theta = 0.5 * torch.atan2(angle_vector[..., 0], angle_vector[..., 1])
    return theta


def aspect_ratio_angle_weight(boxes: torch.Tensor, min_weight: float = 0.25) -> torch.Tensor:
    """Return angle-importance weights that down-weight nearly square ``xywh`` boxes."""
    if boxes.shape[-1] < 4:
        raise ValueError(f"boxes must contain xywh values, got {tuple(boxes.shape)}")
    min_weight = float(min_weight)
    if not 0.0 <= min_weight <= 1.0:
        raise ValueError(f"min_weight must be in [0, 1], got {min_weight}")

    boxes = boxes.float()
    width = boxes[..., 2].abs()
    height = boxes[..., 3].abs()
    elongation = (width - height).abs() / (width + height + 1e-6)
    return min_weight + (1.0 - min_weight) * elongation


def aligned_probiou_loss(pred_rboxes: torch.Tensor, gt_rboxes: torch.Tensor) -> torch.Tensor:
    """Return element-wise ProbIoU loss for aligned ``xywhr`` boxes."""
    if pred_rboxes.shape != gt_rboxes.shape or pred_rboxes.shape[-1] != 5:
        raise ValueError(
            f"aligned rboxes must have the same [N,5] shape, got {tuple(pred_rboxes.shape)} "
            f"and {tuple(gt_rboxes.shape)}"
        )
    if pred_rboxes.numel() == 0:
        return pred_rboxes.new_empty((0,))

    pred_rboxes = _prepare_rboxes_fp32(pred_rboxes)
    gt_rboxes = _prepare_rboxes_fp32(gt_rboxes)
    similarity = probiou(pred_rboxes, gt_rboxes).reshape(-1)
    return (1.0 - similarity).clamp_min(0.0)


def aligned_cyclic_corner_loss(
    pred_rboxes: torch.Tensor,
    gt_rboxes: torch.Tensor,
    beta: float = 1.0 / 9.0,
) -> torch.Tensor:
    """Return minimum Smooth-L1 corner error across four cyclic target alignments."""
    if pred_rboxes.shape != gt_rboxes.shape or pred_rboxes.shape[-1] != 5:
        raise ValueError(
            f"aligned rboxes must have the same [N,5] shape, got {tuple(pred_rboxes.shape)} "
            f"and {tuple(gt_rboxes.shape)}"
        )
    if pred_rboxes.numel() == 0:
        return pred_rboxes.new_empty((0,))
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")

    pred_rboxes = _prepare_rboxes_fp32(pred_rboxes)
    gt_rboxes = _prepare_rboxes_fp32(gt_rboxes)
    pred_corners = xywhr2xyxyxyxy(pred_rboxes)
    gt_corners = xywhr2xyxyxyxy(gt_rboxes)
    candidates = []
    for shift in range(4):
        shifted = torch.roll(gt_corners, shifts=shift, dims=-2)
        error = F.smooth_l1_loss(
            pred_corners,
            shifted,
            reduction="none",
            beta=float(beta),
        ).mean(dim=(-1, -2))
        candidates.append(error)
    return torch.stack(candidates, dim=-1).amin(dim=-1)


def _rbox_covariance_elements(rboxes: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return 2x2 covariance elements for xywhr boxes modeled as Gaussians."""
    if rboxes.shape[-1] != 5:
        raise ValueError(f"rboxes must end with 5 values [cx,cy,w,h,theta], got {tuple(rboxes.shape)}")

    rboxes = _prepare_rboxes_fp32(rboxes)
    w = rboxes[..., 2] * 0.5
    h = rboxes[..., 3] * 0.5
    theta = rboxes[..., 4]

    w2 = w.square()
    h2 = h.square()
    c = torch.cos(theta)
    s = torch.sin(theta)
    c2 = c.square()
    s2 = s.square()

    cov_xx = w2 * c2 + h2 * s2
    cov_yy = w2 * s2 + h2 * c2
    cov_xy = (w2 - h2) * c * s
    trace = cov_xx + cov_yy
    det = (w2 * h2).clamp_min(1e-14)
    return cov_xx, cov_yy, cov_xy, trace, det


def pairwise_gwd_cost(pred_rboxes: torch.Tensor, gt_rboxes: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    """Pairwise normalized Gaussian-Wasserstein cost for rotated boxes.

    Args:
        pred_rboxes: [P, 5] in normalized xywhr.
        gt_rboxes: [G, 5] in normalized xywhr.
        tau: Scale of the exponential normalization. The default is suitable for normalized coordinates.
    """
    pred_rboxes = _prepare_rboxes_fp32(pred_rboxes)
    gt_rboxes = _prepare_rboxes_fp32(gt_rboxes)
    if pred_rboxes.numel() == 0 or gt_rboxes.numel() == 0:
        return pred_rboxes.new_zeros((pred_rboxes.shape[0], gt_rboxes.shape[0]))

    pxx, pyy, pxy, ptr, pdet = _rbox_covariance_elements(pred_rboxes)
    gxx, gyy, gxy, gtr, gdet = _rbox_covariance_elements(gt_rboxes)

    center_d2 = torch.cdist(pred_rboxes[:, :2], gt_rboxes[:, :2], p=2).square()
    trace_product = (
        pxx[:, None] * gxx[None, :]
        + pyy[:, None] * gyy[None, :]
        + 2.0 * pxy[:, None] * gxy[None, :]
    )
    inner = trace_product + 2.0 * torch.sqrt((pdet[:, None] * gdet[None, :]).clamp_min(0.0))
    covariance_d2 = ptr[:, None] + gtr[None, :] - 2.0 * torch.sqrt(inner.clamp_min(0.0))
    distance_d2 = (center_d2 + covariance_d2).clamp_min(0.0)

    tau = max(float(tau), 1e-8)
    return 1.0 - torch.exp(-distance_d2 / tau)


def aligned_gwd_loss(pred_rboxes: torch.Tensor, gt_rboxes: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    """Element-wise normalized Gaussian-Wasserstein loss for aligned rotated boxes."""
    if pred_rboxes.shape != gt_rboxes.shape or pred_rboxes.shape[-1] != 5:
        raise ValueError(
            f"aligned rboxes must have the same [N,5] shape, got {tuple(pred_rboxes.shape)} and {tuple(gt_rboxes.shape)}"
        )
    pred_rboxes = _prepare_rboxes_fp32(pred_rboxes)
    gt_rboxes = _prepare_rboxes_fp32(gt_rboxes)
    if pred_rboxes.numel() == 0:
        return pred_rboxes.new_empty((0,))

    pxx, pyy, pxy, ptr, pdet = _rbox_covariance_elements(pred_rboxes)
    gxx, gyy, gxy, gtr, gdet = _rbox_covariance_elements(gt_rboxes)
    center_d2 = (pred_rboxes[:, :2] - gt_rboxes[:, :2]).square().sum(-1)
    trace_product = pxx * gxx + pyy * gyy + 2.0 * pxy * gxy
    inner = trace_product + 2.0 * torch.sqrt((pdet * gdet).clamp_min(0.0))
    covariance_d2 = ptr + gtr - 2.0 * torch.sqrt(inner.clamp_min(0.0))
    distance_d2 = (center_d2 + covariance_d2).clamp_min(0.0)
    tau = max(float(tau), 1e-8)
    return 1.0 - torch.exp(-distance_d2 / tau)


class HungarianMatcher(nn.Module):
    """Hungarian matcher with optional OBB angle and Gaussian-Wasserstein costs."""

    def __init__(
        self,
        cost_gain: dict[str, float] | None = None,
        use_fl: bool = True,
        with_mask: bool = False,
        num_sample_points: int = 12544,
        alpha: float = 0.25,
        gamma: float = 2.0,
        gwd_tau: float = 0.05,
        angle_min_weight: float = 0.25,
        weighted_match_angle: bool = False,
    ):
        super().__init__()
        if cost_gain is None:
            cost_gain = {"class": 1, "bbox": 5, "giou": 2, "angle": 0, "gwd": 0, "mask": 1, "dice": 1}
        self.cost_gain = cost_gain
        self.use_fl = use_fl
        self.with_mask = with_mask
        self.num_sample_points = num_sample_points
        self.alpha = alpha
        self.gamma = gamma
        self.gwd_tau = float(gwd_tau)
        self.angle_min_weight = float(angle_min_weight)
        self.weighted_match_angle = bool(weighted_match_angle)

    @torch.no_grad()
    def forward(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        masks: torch.Tensor | None = None,
        gt_mask: list[torch.Tensor] | None = None,
        pred_angles: torch.Tensor | None = None,
        gt_angles: torch.Tensor | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Compute optimal assignment for HBB or OBB predictions.

        OBB mode is enabled when both ``pred_angles`` and ``gt_angles`` are supplied. ``pred_angles`` must contain
        [sin(2θ), cos(2θ)] vectors while ``gt_angles`` contains theta in radians.
        """
        bs, nq, nc = pred_scores.shape
        total_gt = int(sum(gt_groups))
        if total_gt == 0:
            return [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)) for _ in range(bs)]

        flat_scores = pred_scores.detach().reshape(-1, nc)
        flat_scores = flat_scores.sigmoid() if self.use_fl else flat_scores.softmax(dim=-1)
        flat_boxes = pred_bboxes.detach().reshape(-1, 4)
        gt_bboxes = gt_bboxes.detach().reshape(-1, 4)
        gt_cls = gt_cls.detach().long().view(-1)

        selected_scores = flat_scores[:, gt_cls]
        if self.use_fl:
            neg_cost = (1.0 - self.alpha) * selected_scores.pow(self.gamma) * (-(1.0 - selected_scores + 1e-8).log())
            pos_cost = self.alpha * (1.0 - selected_scores).pow(self.gamma) * (-(selected_scores + 1e-8).log())
            cost_class = pos_cost - neg_cost
        else:
            cost_class = -selected_scores

        cost_bbox = (flat_boxes[:, None, :] - gt_bboxes[None, :, :]).abs().sum(-1)
        C = self.cost_gain.get("class", 0.0) * cost_class + self.cost_gain.get("bbox", 0.0) * cost_bbox

        obb_mode = pred_angles is not None and gt_angles is not None
        if obb_mode:
            flat_angle_vec = safe_normalize_angle_vector(pred_angles.detach().reshape(-1, 2))
            gt_theta = gt_angles.detach().reshape(-1)
            gt_angle_vec = theta_to_angle_vector(gt_theta).to(flat_angle_vec)
            cost_angle = 1.0 - flat_angle_vec @ gt_angle_vec.transpose(0, 1)
            if self.weighted_match_angle:
                target_angle_weight = aspect_ratio_angle_weight(gt_bboxes, self.angle_min_weight)
                cost_angle = cost_angle * target_angle_weight.unsqueeze(0)

            pred_theta = angle_vector_to_theta(flat_angle_vec)
            pred_rboxes = torch.cat((flat_boxes, pred_theta[:, None]), dim=-1)
            gt_rboxes = torch.cat((gt_bboxes, gt_theta[:, None]), dim=-1)
            cost_gwd = pairwise_gwd_cost(pred_rboxes, gt_rboxes, tau=self.gwd_tau)

            C = C + self.cost_gain.get("angle", 0.0) * cost_angle + self.cost_gain.get("gwd", 0.0) * cost_gwd
        else:
            if self.cost_gain.get("giou", 0.0):
                cost_giou = 1.0 - bbox_iou(
                    flat_boxes[:, None, :], gt_bboxes[None, :, :], xywh=True, GIoU=True
                ).squeeze(-1)
                C = C + self.cost_gain.get("giou", 0.0) * cost_giou

        if self.with_mask:
            C = C + self._cost_mask(bs, gt_groups, masks, gt_mask)

        C = torch.nan_to_num(C, nan=0.0, posinf=1e6, neginf=-1e6)
        C = C.view(bs, nq, total_gt).cpu()

        split_costs = C.split(gt_groups, dim=-1)
        assignments = [linear_sum_assignment(cost_per_image[image_idx]) for image_idx, cost_per_image in enumerate(split_costs)]
        offsets = torch.as_tensor([0, *gt_groups[:-1]], dtype=torch.long).cumsum_(0)
        return [
            (
                torch.as_tensor(src, dtype=torch.long),
                torch.as_tensor(dst, dtype=torch.long) + offsets[image_idx],
            )
            for image_idx, (src, dst) in enumerate(assignments)
        ]

    def _cost_mask(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Mask matching is not enabled in this RT-DETR build.")


def get_cdn_group(
    batch: dict[str, Any],
    num_classes: int,
    num_queries: int,
    class_embed: torch.Tensor,
    num_dn: int = 100,
    cls_noise_ratio: float = 0.5,
    box_noise_scale: float = 1.0,
    training: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, dict[str, Any] | None]:
    """Generate contrastive denoising queries.

    In OBB mode the batch may contain [cx,cy,w,h,theta]; only the first four values are used as deformable-attention
    reference boxes. Angle supervision is applied by the OBB loss to the corresponding DN queries.
    """
    if (not training) or num_dn <= 0 or batch is None:
        return None, None, None, None

    gt_groups = [int(x) for x in batch["gt_groups"]]
    total_num = sum(gt_groups)
    max_nums = max(gt_groups) if gt_groups else 0
    if max_nums == 0:
        return None, None, None, None

    num_group = max(num_dn // max_nums, 1)
    bs = len(gt_groups)
    gt_cls = batch["cls"].view(-1).long()
    gt_bbox = batch["bboxes"][..., :4].reshape(-1, 4)
    b_idx = batch["batch_idx"].view(-1).long()

    dn_cls = gt_cls.repeat(2 * num_group)
    dn_bbox = gt_bbox.repeat(2 * num_group, 1)
    dn_b_idx = b_idx.repeat(2 * num_group)
    neg_idx = torch.arange(total_num * num_group, dtype=torch.long, device=gt_bbox.device) + num_group * total_num

    if cls_noise_ratio > 0:
        noise_mask = torch.rand(dn_cls.shape, device=dn_cls.device) < (cls_noise_ratio * 0.5)
        noise_idx = torch.nonzero(noise_mask, as_tuple=False).squeeze(-1)
        if noise_idx.numel():
            dn_cls[noise_idx] = torch.randint(0, num_classes, noise_idx.shape, device=dn_cls.device, dtype=dn_cls.dtype)

    if box_noise_scale > 0:
        known_bbox = xywh2xyxy(dn_bbox)
        diff = (dn_bbox[..., 2:] * 0.5).repeat(1, 2) * box_noise_scale
        rand_sign = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(dn_bbox)
        rand_part[neg_idx] += 1.0
        known_bbox += rand_part * rand_sign * diff
        known_bbox.clamp_(0.0, 1.0)
        dn_bbox = torch.logit(xyxy2xywh(known_bbox), eps=1e-6)
    else:
        dn_bbox = torch.logit(dn_bbox.clamp(1e-6, 1.0 - 1e-6), eps=1e-6)

    num_dn = int(max_nums * 2 * num_group)
    dn_cls_embed = class_embed[dn_cls]
    padding_cls = torch.zeros(bs, num_dn, dn_cls_embed.shape[-1], device=class_embed.device, dtype=dn_cls_embed.dtype)
    padding_bbox = torch.zeros(bs, num_dn, 4, device=class_embed.device, dtype=dn_bbox.dtype)

    map_indices = torch.cat([torch.arange(num, dtype=torch.long) for num in gt_groups])
    pos_idx = torch.stack([map_indices + max_nums * group_idx for group_idx in range(num_group)], dim=0)
    all_map_indices = torch.cat([map_indices + max_nums * group_idx for group_idx in range(2 * num_group)])

    padding_cls[(dn_b_idx.to(class_embed.device), all_map_indices.to(class_embed.device))] = dn_cls_embed.to(class_embed.device)
    padding_bbox[(dn_b_idx.to(class_embed.device), all_map_indices.to(class_embed.device))] = dn_bbox.to(class_embed.device)

    tgt_size = num_dn + num_queries
    attn_mask = torch.zeros((tgt_size, tgt_size), dtype=torch.bool, device=class_embed.device)
    attn_mask[num_dn:, :num_dn] = True
    for group_idx in range(num_group):
        start = max_nums * 2 * group_idx
        end = max_nums * 2 * (group_idx + 1)
        attn_mask[start:end, :start] = True
        attn_mask[start:end, end:num_dn] = True

    dn_meta = {
        "dn_pos_idx": [p.reshape(-1) for p in pos_idx.cpu().split(gt_groups, dim=1)],
        "dn_num_group": num_group,
        "dn_num_split": [num_dn, num_queries],
    }
    return padding_cls, padding_bbox, attn_mask, dn_meta
