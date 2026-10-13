# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import FocalLoss, VarifocalLoss
from ultralytics.utils.metrics import bbox_iou

from .ops import (
    HungarianMatcher,
    aligned_cyclic_corner_loss,
    aligned_gwd_loss,
    aligned_probiou_loss,
    angle_vector_to_theta,
    aspect_ratio_angle_weight,
    safe_normalize_angle_vector,
    theta_to_angle_vector,
)


class DETRLoss(nn.Module):
    """DETR (DEtection TRansformer) Loss class for calculating various loss components.

    This class computes classification loss, bounding box loss, GIoU loss, and optionally auxiliary losses for the DETR
    object detection model.

    Attributes:
        nc (int): Number of classes.
        loss_gain (dict[str, float]): Coefficients for different loss components.
        aux_loss (bool): Whether to compute auxiliary losses.
        use_fl (bool): Whether to use FocalLoss.
        use_vfl (bool): Whether to use VarifocalLoss.
        use_uni_match (bool): Whether to use a fixed layer for auxiliary branch label assignment.
        uni_match_ind (int): Index of fixed layer to use if use_uni_match is True.
        matcher (HungarianMatcher): Object to compute matching cost and indices.
        fl (FocalLoss | None): Focal Loss object if use_fl is True, otherwise None.
        vfl (VarifocalLoss | None): Varifocal Loss object if use_vfl is True, otherwise None.
        device (torch.device): Device on which tensors are stored.
    """

    def __init__(
        self,
        nc: int = 80,
        loss_gain: dict[str, float] | None = None,
        aux_loss: bool = True,
        use_fl: bool = True,
        use_vfl: bool = False,
        use_uni_match: bool = False,
        uni_match_ind: int = 0,
        gamma: float = 1.5,
        alpha: float = 0.25,
    ):
        """Initialize DETR loss function with customizable components and gains.

        Uses default loss_gain if not provided. Initializes HungarianMatcher with preset cost gains. Supports auxiliary
        losses and various loss types.

        Args:
            nc (int): Number of classes.
            loss_gain (dict[str, float], optional): Coefficients for different loss components.
            aux_loss (bool): Whether to use auxiliary losses from each decoder layer.
            use_fl (bool): Whether to use FocalLoss.
            use_vfl (bool): Whether to use VarifocalLoss.
            use_uni_match (bool): Whether to use fixed layer for auxiliary branch label assignment.
            uni_match_ind (int): Index of fixed layer for uni_match.
            gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
            alpha (float): The balancing factor used to address class imbalance.
        """
        super().__init__()

        if loss_gain is None:
            loss_gain = {"class": 1, "bbox": 5, "giou": 2, "no_object": 0.1, "mask": 1, "dice": 1}
        self.nc = nc
        self.matcher = HungarianMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2})
        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        self.fl = FocalLoss(gamma, alpha) if use_fl else None
        self.vfl = VarifocalLoss(gamma, alpha) if use_vfl else None

        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.device = None

    def _get_loss_class(
        self, pred_scores: torch.Tensor, targets: torch.Tensor, gt_scores: torch.Tensor, num_gts: int, postfix: str = ""
    ) -> dict[str, torch.Tensor]:
        """Compute classification loss based on predictions, target values, and ground truth scores.

        Args:
            pred_scores (torch.Tensor): Predicted class scores with shape (B, N, C).
            targets (torch.Tensor): Target class indices with shape (B, N).
            gt_scores (torch.Tensor): Ground truth confidence scores with shape (B, N).
            num_gts (int): Number of ground truth objects.
            postfix (str, optional): String to append to the loss name for identification in multi-loss scenarios.

        Returns:
            (dict[str, torch.Tensor]): Dictionary containing classification loss value.

        Notes:
            The function supports different classification loss types:
            - Varifocal Loss (if self.vfl is True and num_gts > 0)
            - Focal Loss (if self.fl is True)
            - BCE Loss (default fallback)
        """
        # Logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = f"loss_class{postfix}"
        bs, nq = pred_scores.shape[:2]
        # one_hot = F.one_hot(targets, self.nc + 1)[..., :-1]  # (bs, num_queries, num_classes)
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        gt_scores = gt_scores.view(bs, nq, 1) * one_hot

        if self.fl:
            if num_gts and self.vfl:
                loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
            else:
                loss_cls = self.fl(pred_scores, one_hot.float())
            loss_cls /= max(num_gts, 1) / nq
        else:
            loss_cls = nn.BCEWithLogitsLoss(reduction="none")(pred_scores, gt_scores).mean(1).sum()  # YOLO CLS loss

        return {name_class: loss_cls.squeeze() * self.loss_gain["class"]}

    def _get_loss_bbox(
        self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, postfix: str = ""
    ) -> dict[str, torch.Tensor]:
        """Compute bounding box and GIoU losses for predicted and ground truth bounding boxes.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes with shape (N, 4).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape (N, 4).
            postfix (str, optional): String to append to the loss names for identification in multi-loss scenarios.

        Returns:
            (dict[str, torch.Tensor]): Dictionary containing:
                - loss_bbox{postfix}: L1 loss between predicted and ground truth boxes, scaled by the bbox loss gain.
                - loss_giou{postfix}: GIoU loss between predicted and ground truth boxes, scaled by the giou loss gain.

        Notes:
            If no ground truth boxes are provided (empty list), zero-valued tensors are returned for both losses.
        """
        # Boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = f"loss_bbox{postfix}"
        name_giou = f"loss_giou{postfix}"

        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0.0, device=self.device)
            loss[name_giou] = torch.tensor(0.0, device=self.device)
            return loss

        loss[name_bbox] = self.loss_gain["bbox"] * F.l1_loss(pred_bboxes, gt_bboxes, reduction="sum") / len(gt_bboxes)
        loss[name_giou] = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain["giou"] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}

    # This function is for future RT-DETR Segment models
    # def _get_loss_mask(self, masks, gt_mask, match_indices, postfix=''):
    #     # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
    #     name_mask = f'loss_mask{postfix}'
    #     name_dice = f'loss_dice{postfix}'
    #
    #     loss = {}
    #     if sum(len(a) for a in gt_mask) == 0:
    #         loss[name_mask] = torch.tensor(0., device=self.device)
    #         loss[name_dice] = torch.tensor(0., device=self.device)
    #         return loss
    #
    #     num_gts = len(gt_mask)
    #     src_masks, target_masks = self._get_assigned_bboxes(masks, gt_mask, match_indices)
    #     src_masks = F.interpolate(src_masks.unsqueeze(0), size=target_masks.shape[-2:], mode='bilinear')[0]
    #     # TODO: torch does not have `sigmoid_focal_loss`, but it's not urgent since we don't use mask branch for now.
    #     loss[name_mask] = self.loss_gain['mask'] * F.sigmoid_focal_loss(src_masks, target_masks,
    #                                                                     torch.tensor([num_gts], dtype=torch.float32))
    #     loss[name_dice] = self.loss_gain['dice'] * self._dice_loss(src_masks, target_masks, num_gts)
    #     return loss

    # This function is for future RT-DETR Segment models
    # @staticmethod
    # def _dice_loss(inputs, targets, num_gts):
    #     inputs = F.sigmoid(inputs).flatten(1)
    #     targets = targets.flatten(1)
    #     numerator = 2 * (inputs * targets).sum(1)
    #     denominator = inputs.sum(-1) + targets.sum(-1)
    #     loss = 1 - (numerator + 1) / (denominator + 1)
    #     return loss.sum() / num_gts

    def _get_loss_aux(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        match_indices: list[tuple] | None = None,
        postfix: str = "",
        masks: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Get auxiliary losses for intermediate decoder layers.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes from auxiliary layers.
            pred_scores (torch.Tensor): Predicted scores from auxiliary layers.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes.
            gt_cls (torch.Tensor): Ground truth classes.
            gt_groups (list[int]): Number of ground truths per image.
            match_indices (list[tuple], optional): Pre-computed matching indices.
            postfix (str, optional): String to append to loss names.
            masks (torch.Tensor, optional): Predicted masks if using segmentation.
            gt_mask (torch.Tensor, optional): Ground truth masks if using segmentation.

        Returns:
            (dict[str, torch.Tensor]): Dictionary of auxiliary losses.
        """
        # NOTE: loss class, bbox, giou, mask, dice
        loss = torch.zeros(5 if masks is not None else 3, device=pred_bboxes.device)
        if match_indices is None and self.use_uni_match:
            match_indices = self.matcher(
                pred_bboxes[self.uni_match_ind],
                pred_scores[self.uni_match_ind],
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=masks[self.uni_match_ind] if masks is not None else None,
                gt_mask=gt_mask,
            )
        for i, (aux_bboxes, aux_scores) in enumerate(zip(pred_bboxes, pred_scores)):
            aux_masks = masks[i] if masks is not None else None
            loss_ = self._get_loss(
                aux_bboxes,
                aux_scores,
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=aux_masks,
                gt_mask=gt_mask,
                postfix=postfix,
                match_indices=match_indices,
            )
            loss[0] += loss_[f"loss_class{postfix}"]
            loss[1] += loss_[f"loss_bbox{postfix}"]
            loss[2] += loss_[f"loss_giou{postfix}"]
            # if masks is not None and gt_mask is not None:
            #     loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices, postfix)
            #     loss[3] += loss_[f'loss_mask{postfix}']
            #     loss[4] += loss_[f'loss_dice{postfix}']

        loss = {
            f"loss_class_aux{postfix}": loss[0],
            f"loss_bbox_aux{postfix}": loss[1],
            f"loss_giou_aux{postfix}": loss[2],
        }
        # if masks is not None and gt_mask is not None:
        #     loss[f'loss_mask_aux{postfix}'] = loss[3]
        #     loss[f'loss_dice_aux{postfix}'] = loss[4]
        return loss

    @staticmethod
    def _get_index(match_indices: list[tuple]) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Extract batch indices, source indices, and destination indices from match indices.

        Args:
            match_indices (list[tuple]): List of tuples containing matched indices.

        Returns:
            batch_idx (tuple[torch.Tensor, torch.Tensor]): Tuple containing (batch_idx, src_idx).
            dst_idx (torch.Tensor): Destination indices.
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)])
        src_idx = torch.cat([src for (src, _) in match_indices])
        dst_idx = torch.cat([dst for (_, dst) in match_indices])
        return (batch_idx, src_idx), dst_idx

    def _get_assigned_bboxes(
        self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, match_indices: list[tuple]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assign predicted bounding boxes to ground truth bounding boxes based on match indices.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes.
            match_indices (list[tuple]): List of tuples containing matched indices.

        Returns:
            pred_assigned (torch.Tensor): Assigned predicted bounding boxes.
            gt_assigned (torch.Tensor): Assigned ground truth bounding boxes.
        """
        pred_assigned = torch.cat(
            [
                t[i] if len(i) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
                for t, (i, _) in zip(pred_bboxes, match_indices)
            ]
        )
        gt_assigned = torch.cat(
            [
                t[j] if len(j) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
                for t, (_, j) in zip(gt_bboxes, match_indices)
            ]
        )
        return pred_assigned, gt_assigned

    def _get_loss(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        masks: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
        postfix: str = "",
        match_indices: list[tuple] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Calculate losses for a single prediction layer.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes.
            pred_scores (torch.Tensor): Predicted class scores.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes.
            gt_cls (torch.Tensor): Ground truth classes.
            gt_groups (list[int]): Number of ground truths per image.
            masks (torch.Tensor, optional): Predicted masks if using segmentation.
            gt_mask (torch.Tensor, optional): Ground truth masks if using segmentation.
            postfix (str, optional): String to append to loss names.
            match_indices (list[tuple], optional): Pre-computed matching indices.

        Returns:
            (dict[str, torch.Tensor]): Dictionary of losses.
        """
        if match_indices is None:
            match_indices = self.matcher(
                pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=masks, gt_mask=gt_mask
            )

        idx, gt_idx = self._get_index(match_indices)
        pred_bboxes, gt_bboxes = pred_bboxes[idx], gt_bboxes[gt_idx]

        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]

        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if len(gt_bboxes):
            gt_scores[idx] = bbox_iou(pred_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1)

        return {
            **self._get_loss_class(pred_scores, targets, gt_scores, len(gt_bboxes), postfix),
            **self._get_loss_bbox(pred_bboxes, gt_bboxes, postfix),
            # **(self._get_loss_mask(masks, gt_mask, match_indices, postfix) if masks is not None and gt_mask is not None else {})
        }

    def forward(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        batch: dict[str, Any],
        postfix: str = "",
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Calculate loss for predicted bounding boxes and scores.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes, shape (L, B, N, 4).
            pred_scores (torch.Tensor): Predicted class scores, shape (L, B, N, C).
            batch (dict[str, Any]): Batch information containing cls, bboxes, and gt_groups.
            postfix (str, optional): Postfix for loss names.
            **kwargs (Any): Additional arguments, may include 'match_indices'.

        Returns:
            (dict[str, torch.Tensor]): Computed losses, including main and auxiliary (if enabled).

        Notes:
            Uses last elements of pred_bboxes and pred_scores for main loss, and the rest for auxiliary losses if
            self.aux_loss is True.
        """
        self.device = pred_bboxes.device
        match_indices = kwargs.get("match_indices", None)
        gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]

        total_loss = self._get_loss(
            pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls, gt_groups, postfix=postfix, match_indices=match_indices
        )

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    pred_bboxes[:-1], pred_scores[:-1], gt_bboxes, gt_cls, gt_groups, match_indices, postfix
                )
            )

        return total_loss


@torch.no_grad()
def create_gaussian_target_with_offset(
    batch_bboxes: torch.Tensor,
    batch_idx: torch.Tensor,
    mask_shape: tuple[int, int, int, int],
    device: torch.device,
    batch_angles: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    根据归一化 xywh 真值框生成多目标高斯热力图。

    Args:
        batch_bboxes: [N, 4]，归一化 xywh。
        batch_idx: [N]，每个目标所属的 batch 索引。
        mask_shape: (B, 1, H, W)。
        device: 输出设备。
    """
    B, C, H, W = mask_shape
    if C != 1:
        raise ValueError(f"频率掩码必须是单通道，当前 C={C}")

    target = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)
    if batch_bboxes is None or batch_bboxes.numel() == 0:
        return target

    boxes = batch_bboxes.detach().to(device=device, dtype=torch.float32)
    indices = batch_idx.detach().to(device=device, dtype=torch.long).view(-1)

    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"batch_bboxes 应为 [N,4]，当前形状为 {tuple(boxes.shape)}")
    if boxes.shape[0] != indices.numel():
        raise ValueError("batch_bboxes 与 batch_idx 的目标数量不一致")

    angles = None
    if batch_angles is not None:
        angles = batch_angles.detach().to(device=device, dtype=torch.float32).view(-1)
        if angles.numel() != boxes.shape[0]:
            raise ValueError("batch_angles and batch_bboxes must contain the same number of targets")

    for i in range(boxes.shape[0]):
        b = int(indices[i].item())
        if b < 0 or b >= B:
            raise IndexError(f"目标 {i} 的 batch_idx={b} 超出范围 [0, {B - 1}]")

        cx, cy, bw, bh = boxes[i]

        # 输入必须是归一化坐标；这里只做一次尺度映射。
        cx_real = (cx * W).clamp(0.0, max(float(W) - 1e-4, 0.0))
        cy_real = (cy * H).clamp(0.0, max(float(H) - 1e-4, 0.0))
        w_real = (bw * W).clamp(min=1e-4)
        h_real = (bh * H).clamp(min=1e-4)

        cx_int = int(torch.floor(cx_real).item())
        cy_int = int(torch.floor(cy_real).item())
        offset_x = cx_real - cx_int
        offset_y = cy_real - cy_int

        if angles is None:
            radius = max(2, int(torch.ceil(torch.minimum(w_real, h_real) / 2.0).item()))
            sigma = max(radius / 3.0, 1e-3)
            x_grid = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32) - offset_x
            y_grid = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32) - offset_y
            y_mesh, x_mesh = torch.meshgrid(y_grid, x_grid, indexing="ij")
            gaussian = torch.exp(-(x_mesh.square() + y_mesh.square()) / (2.0 * sigma * sigma))
        else:
            sigma_x = max(float(w_real.item()) / 6.0, 2.0 / 3.0)
            sigma_y = max(float(h_real.item()) / 6.0, 2.0 / 3.0)
            radius = max(2, int(math.ceil(3.0 * max(sigma_x, sigma_y))))
            x_grid = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32) - offset_x
            y_grid = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32) - offset_y
            y_mesh, x_mesh = torch.meshgrid(y_grid, x_grid, indexing="ij")
            theta = angles[i]
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_mesh + sin_theta * y_mesh
            y_rot = -sin_theta * x_mesh + cos_theta * y_mesh
            gaussian = torch.exp(-0.5 * ((x_rot / sigma_x).square() + (y_rot / sigma_y).square()))

        top = max(0, cy_int - radius)
        bottom = min(H, cy_int + radius + 1)
        left = max(0, cx_int - radius)
        right = min(W, cx_int + radius + 1)

        if bottom <= top or right <= left:
            continue

        g_top = radius - (cy_int - top)
        g_bottom = g_top + (bottom - top)
        g_left = radius - (cx_int - left)
        g_right = g_left + (right - left)

        target[b, 0, top:bottom, left:right] = torch.maximum(
            target[b, 0, top:bottom, left:right],
            gaussian[g_top:g_bottom, g_left:g_right],
        )

    return target


def calculate_dice_loss(
    pred_prob: torch.Tensor,
    target_mask: torch.Tensor,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """按图像计算 Dice Loss，同时监督含目标图和空目标图。"""
    pred = pred_prob.float().flatten(1)
    target = target_mask.float().flatten(1)

    intersection = (pred * target).sum(dim=1)
    denominator = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def calculate_nwd_elementwise(
    pred_bboxes: torch.Tensor,
    gt_bboxes: torch.Tensor,
    C: float = 0.1,
) -> torch.Tensor:
    """逐元素 NWD Loss；完全重合时严格等于 0。"""
    if pred_bboxes.numel() == 0:
        return pred_bboxes.new_empty((0,))

    pred_cx, pred_cy, pred_w, pred_h = pred_bboxes.unbind(-1)
    gt_cx, gt_cy, gt_w, gt_h = gt_bboxes.unbind(-1)

    delta = torch.stack(
        (
            pred_cx - gt_cx,
            pred_cy - gt_cy,
            pred_w * 0.5 - gt_w * 0.5,
            pred_h * 0.5 - gt_h * 0.5,
        ),
        dim=-1,
    )
    w2_distance = torch.linalg.vector_norm(delta, ord=2, dim=-1)
    return 1.0 - torch.exp(-w2_distance / max(float(C), 1e-6))


class RTDETRDetectionLoss(DETRLoss):
    """
    修正版：
    1. L1 与动态 NWD/GIoU 不重复计入；
    2. 修复 width/height 解包；
    3. 修复高斯标签重复缩放；
    4. 空目标图仍接受背景掩码监督；
    5. P2/P3/P4 分项仅作为 detached 指标，不进入总损失；
    6. DN 匹配同时兼容 list 和扁平 Tensor。
    """

    def __init__(
        self,
        *args,
        nwd_C: float = 0.1,
        scale_factor: float = 1000.0,
        lambda_mask: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.nwd_C = float(nwd_C)
        self.scale_factor = float(scale_factor)
        self.lambda_mask = float(lambda_mask)
        self.scale_weights = {0: 1.0, 1: 0.6, 2: 0.4}

        # 仅用于日志，不参与 loss_sum。
        self.last_box_metrics: dict[str, torch.Tensor] = {}
        self.last_mask_metrics: dict[str, torch.Tensor] = {}

    def loss_boxes(self, pred_bboxes, batch, indices):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = pred_bboxes[idx]

        target_parts = [
            target["bboxes"][target_idx]
            for target, (_, target_idx) in zip(batch, indices)
            if target_idx.numel() > 0
        ]

        if not target_parts:
            zero = pred_bboxes.sum() * 0.0
            self.last_box_metrics = {
                "metric_giou_raw": zero.detach(),
                "metric_nwd_raw": zero.detach(),
                "metric_nwd_alpha": zero.detach(),
            }
            return {"loss_bbox": zero, "loss_giou": zero}

        target_boxes = torch.cat(target_parts, dim=0)
        num_gts = max(int(target_boxes.shape[0]), 1)

        loss_l1 = F.l1_loss(src_boxes, target_boxes, reduction="none").sum() / num_gts

        src_xyxy = xywh2xyxy(src_boxes)
        target_xyxy = xywh2xyxy(target_boxes)
        giou = bbox_iou(src_xyxy, target_xyxy, xywh=False, GIoU=True).reshape(-1)
        loss_giou_element = 1.0 - giou

        loss_nwd_element = calculate_nwd_elementwise(
            src_boxes,
            target_boxes,
            C=self.nwd_C,
        )

        tgt_w = target_boxes[:, 2]
        tgt_h = target_boxes[:, 3]
        tgt_area = tgt_w * tgt_h

        # 小目标 alpha 接近 1，更多使用 NWD；大目标更多使用 GIoU。
        dynamic_alpha = torch.sigmoid(3.0 - self.scale_factor * tgt_area)
        mixed_reg_element = (
            dynamic_alpha * loss_nwd_element
            + (1.0 - dynamic_alpha) * loss_giou_element
        )
        mixed_reg_loss = mixed_reg_element.sum() / num_gts

        self.last_box_metrics = {
            "metric_giou_raw": (loss_giou_element.sum() / num_gts).detach(),
            "metric_nwd_raw": (loss_nwd_element.sum() / num_gts).detach(),
            "metric_nwd_alpha": dynamic_alpha.mean().detach(),
        }

        # 只返回真正参与优化的两项，避免 NWD/GIoU 重复求和。
        return {
            "loss_bbox": self.bbox_loss_coef * loss_l1,
            "loss_giou": self.giou_loss_coef * mixed_reg_loss,
        }

    @staticmethod
    def _zero(device: torch.device) -> torch.Tensor:
        return torch.zeros((), device=device, dtype=torch.float32)

    def get_dn_match_indices(
        self,
        dn_pos_idx,
        dn_num_group,
        gt_groups,
        device: torch.device,
    ):
        dn_num_group = int(dn_num_group)
        match_indices = []

        if dn_pos_idx is None:
            return [
                (
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                )
                for _ in gt_groups
            ]

        # Ultralytics 常见格式：每张图一个 query 索引 Tensor。
        if isinstance(dn_pos_idx, (list, tuple)):
            for batch_i, num_gt in enumerate(gt_groups):
                num_gt = int(num_gt)
                if num_gt <= 0:
                    match_indices.append(
                        (
                            torch.empty(0, dtype=torch.long, device=device),
                            torch.empty(0, dtype=torch.long, device=device),
                        )
                    )
                    continue

                query_idx = dn_pos_idx[batch_i].to(device=device, dtype=torch.long).reshape(-1)
                gt_idx = torch.arange(num_gt, device=device, dtype=torch.long).repeat(dn_num_group)

                if query_idx.numel() != gt_idx.numel():
                    raise RuntimeError(
                        f"DN 匹配数量异常：batch={batch_i}, "
                        f"query={query_idx.numel()}, gt={gt_idx.numel()}"
                    )
                match_indices.append((query_idx, gt_idx))
            return match_indices

        # 兼容扁平 Tensor 格式。
        flat_query_idx = dn_pos_idx.to(device=device, dtype=torch.long).reshape(-1)
        cursor = 0
        for num_gt in gt_groups:
            num_gt = int(num_gt)
            count = num_gt * dn_num_group
            if count > 0:
                query_idx = flat_query_idx[cursor : cursor + count]
                gt_idx = torch.arange(num_gt, device=device, dtype=torch.long).repeat(dn_num_group)
                if query_idx.numel() != count:
                    raise RuntimeError("扁平 dn_pos_idx 长度不足")
                match_indices.append((query_idx, gt_idx))
                cursor += count
            else:
                match_indices.append(
                    (
                        torch.empty(0, dtype=torch.long, device=device),
                        torch.empty(0, dtype=torch.long, device=device),
                    )
                )
        return match_indices

    def _loss_frequency_masks(
        self,
        freq_masks: dict | None,
        batch: dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        if freq_masks and set(freq_masks) == {"acsa", "query"}:
            if not all(isinstance(branch, dict) and set(branch) == {0, 1, 2}
                       for branch in freq_masks.values()):
                raise ValueError("independent mask loss requires two complete P2/P3/P4 branches")
            branch_losses, branch_metrics = [], []
            for branch in ("acsa", "query"):
                branch_losses.append(self._loss_frequency_masks(freq_masks[branch], batch, device))
                branch_metrics.append(dict(self.last_mask_metrics))
            self.last_mask_metrics = {
                key: (branch_metrics[0][key] + branch_metrics[1][key]) / 2
                for key in branch_metrics[0]
            }
            return (branch_losses[0] + branch_losses[1]) / 2
        self.last_mask_metrics = {}

        if not freq_masks:
            return self._zero(device)

        gt_bboxes = batch.get("bboxes")
        batch_idx = batch.get("batch_idx")
        gt_groups = batch.get("gt_groups")

        if gt_bboxes is None:
            gt_bboxes = torch.empty((0, 4), device=device)
        else:
            gt_bboxes = gt_bboxes.to(device=device, dtype=torch.float32)

        if batch_idx is None:
            if gt_groups is None:
                raise KeyError("掩码监督需要 batch_idx 或 gt_groups")
            batch_idx = torch.repeat_interleave(
                torch.arange(len(gt_groups), device=device, dtype=torch.long),
                torch.as_tensor(gt_groups, device=device, dtype=torch.long),
            )
        else:
            batch_idx = batch_idx.to(device=device, dtype=torch.long).view(-1)

        weighted_loss = self._zero(device)
        active_weight_sum = 0.0

        for scale_idx, pred_mask_logits in sorted(freq_masks.items(), key=lambda item: int(item[0])):
            scale_idx = int(scale_idx)
            if pred_mask_logits.ndim != 4 or pred_mask_logits.shape[1] != 1:
                raise ValueError(
                    f"scale {scale_idx} 掩码应为 [B,1,H,W]，"
                    f"当前为 {tuple(pred_mask_logits.shape)}"
                )

            pred_logits = pred_mask_logits.float()

            # 注意：这里直接传归一化 gt_bboxes，不再提前乘 H/W。
            target_mask = create_gaussian_target_with_offset(
                batch_bboxes=gt_bboxes,
                batch_idx=batch_idx,
                mask_shape=tuple(pred_logits.shape),
                device=pred_logits.device,
            )

            positive_mass = target_mask.sum()
            if positive_mass.item() > 0:
                negative_mass = target_mask.numel() - positive_mass
                pos_weight = (negative_mass / positive_mass).clamp(min=1.0, max=20.0)
            else:
                # 空目标批次仍用全零标签做背景监督。
                pos_weight = torch.ones((), device=pred_logits.device, dtype=torch.float32)

            loss_bce = F.binary_cross_entropy_with_logits(
                pred_logits,
                target_mask,
                pos_weight=pos_weight,
            )
            loss_dice = calculate_dice_loss(torch.sigmoid(pred_logits), target_mask)
            scale_loss = 0.7 * loss_bce + 0.3 * loss_dice

            weight = float(self.scale_weights.get(scale_idx, 0.5))
            weighted_loss = weighted_loss + weight * scale_loss
            active_weight_sum += weight

            self.last_mask_metrics[f"metric_mask_P{scale_idx + 2}"] = scale_loss.detach()

        if active_weight_sum <= 0:
            return self._zero(device)

        # 先做尺度加权平均，再由 lambda_mask 控制总权重。
        return self.lambda_mask * (weighted_loss / active_weight_sum)

    def forward(
        self,
        preds: tuple[torch.Tensor, torch.Tensor],
        batch: dict[str, Any],
        dn_bboxes: torch.Tensor | None = None,
        dn_scores: torch.Tensor | None = None,
        dn_meta: dict[str, Any] | None = None,
        freq_masks: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        pred_bboxes, pred_scores = preds
        device = pred_bboxes.device

        total_loss = super().forward(pred_bboxes, pred_scores, batch)

        if (
            dn_meta is not None
            and dn_bboxes is not None
            and dn_scores is not None
        ):
            match_indices = self.get_dn_match_indices(
                dn_pos_idx=dn_meta["dn_pos_idx"],
                dn_num_group=dn_meta["dn_num_group"],
                gt_groups=batch["gt_groups"],
                device=device,
            )
            dn_loss = super().forward(
                dn_bboxes,
                dn_scores,
                batch,
                postfix="_dn",
                match_indices=match_indices,
            )
            total_loss.update(dn_loss)
        else:
            # 保持 key 稳定，但这些零值不产生额外梯度。
            for key in list(total_loss.keys()):
                total_loss[f"{key}_dn"] = self._zero(device)

        # 只加入一个真正参与优化的频率掩码总损失。
        total_loss["loss_fsa_mask"] = self._loss_frequency_masks(
            freq_masks=freq_masks,
            batch=batch,
            device=device,
        )
        return total_loss
class RTDETROBBLoss(nn.Module):
    """FMC-Wave RT-DETR OBB loss with L1, pi-periodic angle, GWD, DN, auxiliary and frequency-mask terms."""

    log_metric_keys = (
        "metric_gwd_raw",
        "metric_riou_raw",
        "metric_corner_raw",
        "metric_angle_raw",
        "metric_angle_weight",
        "metric_mask_P2",
        "metric_mask_P3",
        "metric_mask_P4",
    )

    def __init__(
        self,
        nc: int = 80,
        loss_gain: dict[str, float] | None = None,
        aux_loss: bool = True,
        use_fl: bool = True,
        use_vfl: bool = True,
        gamma: float = 1.5,
        alpha: float = 0.25,
        gwd_tau: float = 0.05,
        lambda_mask: float = 0.5,
        angle_min_weight: float = 0.25,
        weighted_match_angle: bool = False,
        oriented_mask: bool = False,
        mask_scale_weights: tuple[float, float, float] = (1.0, 0.6, 0.4),
    ):
        super().__init__()
        if loss_gain is None:
            loss_gain = {
                "class": 1.0,
                "bbox": 5.0,
                "angle": 1.0,
                "gwd": 2.0,
                "riou": 0.0,
                "corner": 0.0,
            }
        self.nc = int(nc)
        self.loss_gain = {k: float(v) for k, v in loss_gain.items()}
        for name, value in self.loss_gain.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"loss gain {name!r} must be finite and non-negative, got {value}")

        self.aux_loss = bool(aux_loss)
        self.gwd_tau = float(gwd_tau)
        self.lambda_mask = float(lambda_mask)
        self.angle_min_weight = float(angle_min_weight)
        self.oriented_mask = bool(oriented_mask)
        self.matcher = HungarianMatcher(
            cost_gain={"class": 2.0, "bbox": 5.0, "angle": 2.0, "gwd": 2.0},
            use_fl=use_fl,
            alpha=alpha,
            gamma=2.0,
            gwd_tau=self.gwd_tau,
            angle_min_weight=self.angle_min_weight,
            weighted_match_angle=weighted_match_angle,
        )
        self.fl = FocalLoss(gamma, alpha) if use_fl else None
        self.vfl = VarifocalLoss(gamma, alpha) if use_vfl else None
        if not isinstance(mask_scale_weights, (list, tuple)) or len(mask_scale_weights) != 3:
            raise ValueError("mask_scale_weights must contain exactly three values")
        mask_scale_weights = tuple(float(value) for value in mask_scale_weights)
        if not all(math.isfinite(value) and value >= 0.0 for value in mask_scale_weights) or sum(
            mask_scale_weights
        ) <= 0.0:
            raise ValueError("mask_scale_weights must be finite, nonnegative, and have a positive sum")
        self.scale_weights = dict(enumerate(mask_scale_weights))
        self.last_box_metrics: dict[str, torch.Tensor] = {}
        self.last_mask_metrics: dict[str, torch.Tensor] = {}

    @staticmethod
    def _assert_finite_loss_dict(losses: dict[str, torch.Tensor]) -> None:
        """Fail fast on non-finite branch losses using one synchronization per complete forward pass."""
        names = tuple(losses)
        finite = torch.stack([torch.isfinite(losses[name].detach()).all() for name in names])
        if not bool(finite.all()):
            bad = [name for name, valid in zip(names, finite.cpu().tolist()) if not valid]
            raise FloatingPointError(f"non-finite RT-DETR-OBB losses: {', '.join(bad)}")

    @staticmethod
    def _assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
        """Fail at the first non-finite loss input without changing the offending values."""
        if not torch.is_tensor(tensor):
            return
        finite = torch.isfinite(tensor)
        if bool(finite.all()):
            return
        bad = (~finite).nonzero(as_tuple=False)[:10].detach().cpu().tolist()
        safe = torch.nan_to_num(tensor.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        raise FloatingPointError(
            f"[NON-FINITE] {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
            f"bad_indices={bad}, safe_min={safe.min().item():.6g}, "
            f"safe_max={safe.max().item():.6g}, safe_mean={safe.mean().item():.6g}"
        )

    @staticmethod
    def _zero_from(*tensors: torch.Tensor) -> torch.Tensor:
        for tensor in tensors:
            if torch.is_tensor(tensor):
                return tensor.sum() * 0.0
        return torch.tensor(0.0)

    @staticmethod
    def _move_indices(indices, device: torch.device):
        return [(src.to(device=device, dtype=torch.long), dst.to(device=device, dtype=torch.long)) for src, dst in indices]

    @staticmethod
    def _get_index(indices: list[tuple[torch.Tensor, torch.Tensor]], device: torch.device):
        nonempty = [(batch_i, src, dst) for batch_i, (src, dst) in enumerate(indices) if src.numel()]
        if not nonempty:
            empty = torch.empty(0, device=device, dtype=torch.long)
            return (empty, empty), empty
        batch_idx = torch.cat([torch.full_like(src, batch_i, device=device) for batch_i, src, _ in nonempty])
        src_idx = torch.cat([src.to(device) for _, src, _ in nonempty])
        dst_idx = torch.cat([dst.to(device) for _, _, dst in nonempty])
        return (batch_idx, src_idx), dst_idx

    def _classification_loss(
        self,
        pred_scores: torch.Tensor,
        targets: torch.Tensor,
        quality: torch.Tensor,
        num_matches: int,
        postfix: str = "",
    ) -> torch.Tensor:
        pred_scores = pred_scores.float()
        quality = quality.float()
        bs, nq = pred_scores.shape[:2]
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=pred_scores.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        target_scores = quality.view(bs, nq, 1) * one_hot

        if self.fl is not None:
            if num_matches and self.vfl is not None:
                loss = self.vfl(pred_scores, target_scores, one_hot)
            else:
                loss = self.fl(pred_scores, one_hot.float())
            loss = loss / (max(num_matches, 1) / nq)
        else:
            loss = F.binary_cross_entropy_with_logits(pred_scores, target_scores, reduction="none").mean(1).sum()
        return loss.squeeze() * self.loss_gain.get("class", 1.0)

    def _single_layer_loss(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_angles: torch.Tensor,
        batch: dict[str, Any],
        postfix: str = "",
        match_indices: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        update_metrics: bool = False,
    ) -> tuple[dict[str, torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
        gt_cls = batch["cls"].long().view(-1)
        gt_bboxes = batch["bboxes"].float().reshape(-1, 4)
        gt_angles = batch["angles"].float().view(-1)
        gt_groups = [int(x) for x in batch["gt_groups"]]
        device = pred_bboxes.device

        pred_angles = safe_normalize_angle_vector(pred_angles)
        if match_indices is None:
            match_indices = self.matcher(
                pred_bboxes,
                pred_scores,
                gt_bboxes,
                gt_cls,
                gt_groups,
                pred_angles=pred_angles,
                gt_angles=gt_angles,
            )
        match_indices = self._move_indices(match_indices, device)
        idx, gt_idx = self._get_index(match_indices, device)

        bs, nq = pred_scores.shape[:2]
        class_targets = torch.full((bs, nq), self.nc, dtype=gt_cls.dtype, device=device)
        quality = torch.zeros((bs, nq), dtype=torch.float32, device=device)
        if gt_idx.numel():
            class_targets[idx] = gt_cls[gt_idx]

        zero = self._zero_from(pred_bboxes.float(), pred_scores.float(), pred_angles)
        if not gt_idx.numel():
            losses = {
                f"loss_class{postfix}": self._classification_loss(pred_scores, class_targets, quality, 0, postfix),
                f"loss_bbox{postfix}": zero,
                f"loss_angle{postfix}": zero,
                f"loss_gwd{postfix}": zero,
                f"loss_riou{postfix}": zero,
                f"loss_corner{postfix}": zero,
            }
            if update_metrics:
                self.last_box_metrics = {
                    "metric_gwd_raw": zero.detach(),
                    "metric_riou_raw": zero.detach(),
                    "metric_corner_raw": zero.detach(),
                    "metric_angle_raw": zero.detach(),
                    "metric_angle_weight": zero.detach(),
                }
            return losses, match_indices

        src_boxes = pred_bboxes[idx].float()
        tgt_boxes = gt_bboxes[gt_idx].float()
        src_angle_vec = pred_angles[idx]
        tgt_theta = gt_angles[gt_idx].float()
        tgt_angle_vec = theta_to_angle_vector(tgt_theta)
        num_matches = int(gt_idx.numel())

        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="sum") / max(num_matches, 1)
        cosine = (src_angle_vec * tgt_angle_vec).sum(-1).clamp(-1.0, 1.0)
        angle_weight = aspect_ratio_angle_weight(tgt_boxes, self.angle_min_weight).detach()
        loss_angle_element = 1.0 - cosine
        loss_angle = (loss_angle_element * angle_weight).sum() / angle_weight.sum().clamp_min(1.0)

        pred_theta = angle_vector_to_theta(src_angle_vec)
        pred_rboxes = torch.cat((src_boxes, pred_theta[:, None]), dim=-1)
        tgt_rboxes = torch.cat((tgt_boxes, tgt_theta[:, None]), dim=-1)
        loss_gwd_element = aligned_gwd_loss(pred_rboxes, tgt_rboxes, tau=self.gwd_tau)
        loss_gwd = loss_gwd_element.mean()
        loss_riou_element = aligned_probiou_loss(pred_rboxes, tgt_rboxes)
        loss_riou = loss_riou_element.mean()
        loss_corner_element = aligned_cyclic_corner_loss(pred_rboxes, tgt_rboxes)
        loss_corner = (loss_corner_element * angle_weight).sum() / angle_weight.sum().clamp_min(1e-6)
        quality_value = (1.0 - loss_gwd_element.detach()).clamp(0.0, 1.0)
        quality[idx] = quality_value.to(device=quality.device, dtype=quality.dtype)

        losses = {
            f"loss_class{postfix}": self._classification_loss(
                pred_scores, class_targets, quality, num_matches, postfix
            ),
            f"loss_bbox{postfix}": self.loss_gain.get("bbox", 5.0) * loss_bbox,
            f"loss_angle{postfix}": self.loss_gain.get("angle", 1.0) * loss_angle,
            f"loss_gwd{postfix}": self.loss_gain.get("gwd", 2.0) * loss_gwd,
            f"loss_riou{postfix}": self.loss_gain.get("riou", 0.0) * loss_riou,
            f"loss_corner{postfix}": self.loss_gain.get("corner", 0.0) * loss_corner,
        }
        if update_metrics:
            self.last_box_metrics = {
                "metric_gwd_raw": loss_gwd.detach(),
                "metric_riou_raw": loss_riou.detach(),
                "metric_corner_raw": loss_corner.detach(),
                "metric_angle_raw": loss_angle.detach(),
                "metric_angle_weight": angle_weight.mean().detach(),
            }
        return losses, match_indices

    def _stack_loss(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_angles: torch.Tensor,
        batch: dict[str, Any],
        postfix: str = "",
        match_indices: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        update_metrics: bool = False,
    ) -> dict[str, torch.Tensor]:
        main, match_indices = self._single_layer_loss(
            pred_bboxes[-1],
            pred_scores[-1],
            pred_angles[-1],
            batch,
            postfix=postfix,
            match_indices=match_indices,
            update_metrics=update_metrics,
        )
        if self.aux_loss and pred_bboxes.shape[0] > 1:
            aux_keys = ["class", "bbox", "angle", "gwd", "riou", "corner"]
            aux_sums = {key: self._zero_from(pred_bboxes) for key in aux_keys}
            for layer_idx in range(pred_bboxes.shape[0] - 1):
                layer_loss, _ = self._single_layer_loss(
                    pred_bboxes[layer_idx],
                    pred_scores[layer_idx],
                    pred_angles[layer_idx],
                    batch,
                    postfix=postfix,
                    match_indices=match_indices,
                    update_metrics=False,
                )
                for key in aux_keys:
                    aux_sums[key] = aux_sums[key] + layer_loss[f"loss_{key}{postfix}"]
            for key in aux_keys:
                main[f"loss_{key}_aux{postfix}"] = aux_sums[key]
        return main

    @staticmethod
    def _build_dn_match_indices(dn_meta: dict[str, Any], gt_groups: list[int], device: torch.device):
        dn_pos_idx = dn_meta.get("dn_pos_idx")
        num_group = int(dn_meta.get("dn_num_group", 1))
        if dn_pos_idx is None:
            return [
                (torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device))
                for _ in gt_groups
            ]

        offsets = [0]
        for count in gt_groups[:-1]:
            offsets.append(offsets[-1] + int(count))
        indices = []
        if isinstance(dn_pos_idx, (list, tuple)):
            for image_idx, num_gt in enumerate(gt_groups):
                num_gt = int(num_gt)
                query_idx = dn_pos_idx[image_idx].to(device=device, dtype=torch.long).reshape(-1)
                if num_gt:
                    gt_idx = torch.arange(num_gt, device=device).repeat(num_group) + offsets[image_idx]
                else:
                    gt_idx = torch.empty(0, dtype=torch.long, device=device)
                if query_idx.numel() != gt_idx.numel():
                    raise RuntimeError(
                        f"DN OBB match count mismatch at image {image_idx}: query={query_idx.numel()}, gt={gt_idx.numel()}"
                    )
                indices.append((query_idx, gt_idx))
            return indices

        flat = dn_pos_idx.to(device=device, dtype=torch.long).reshape(-1)
        cursor = 0
        for image_idx, num_gt in enumerate(gt_groups):
            count = int(num_gt) * num_group
            query_idx = flat[cursor : cursor + count]
            gt_idx = torch.arange(int(num_gt), device=device).repeat(num_group) + offsets[image_idx]
            indices.append((query_idx, gt_idx))
            cursor += count
        return indices

    def _loss_frequency_masks(
        self,
        freq_masks: dict | None,
        batch: dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        if freq_masks and set(freq_masks) == {"acsa", "query"}:
            if not all(isinstance(branch, dict) and set(branch) == {0, 1, 2}
                       for branch in freq_masks.values()):
                raise ValueError("independent mask loss requires two complete P2/P3/P4 branches")
            branch_losses, branch_metrics = [], []
            for branch in ("acsa", "query"):
                branch_losses.append(self._loss_frequency_masks(freq_masks[branch], batch, device))
                branch_metrics.append(dict(self.last_mask_metrics))
            self.last_mask_metrics = {
                key: (branch_metrics[0][key] + branch_metrics[1][key]) / 2
                for key in branch_metrics[0]
            }
            return (branch_losses[0] + branch_losses[1]) / 2
        self.last_mask_metrics = {}
        if not freq_masks or self.lambda_mask <= 0:
            return torch.zeros((), device=device)

        gt_bboxes = batch.get("bboxes")
        batch_idx = batch.get("batch_idx")
        gt_groups = batch.get("gt_groups")
        gt_angles = batch.get("angles") if self.oriented_mask else None
        if gt_bboxes is None:
            gt_bboxes = torch.empty((0, 4), device=device)
        else:
            gt_bboxes = gt_bboxes[..., :4].to(device=device, dtype=torch.float32)

        if batch_idx is None:
            batch_idx = torch.repeat_interleave(
                torch.arange(len(gt_groups), device=device, dtype=torch.long),
                torch.as_tensor(gt_groups, device=device, dtype=torch.long),
            )
        else:
            batch_idx = batch_idx.to(device=device, dtype=torch.long).view(-1)

        weighted = torch.zeros((), device=device)
        weight_sum = 0.0
        for scale_idx, pred_mask_logits in sorted(freq_masks.items(), key=lambda item: int(item[0])):
            scale_idx = int(scale_idx)
            if pred_mask_logits.ndim != 4 or pred_mask_logits.shape[1] != 1:
                raise ValueError(f"frequency mask at scale {scale_idx} must be [B,1,H,W], got {tuple(pred_mask_logits.shape)}")
            self._assert_finite_tensor(f"frequency_mask_logits_p{scale_idx + 2}", pred_mask_logits)
            pred_logits = pred_mask_logits.float()
            target_mask = create_gaussian_target_with_offset(
                batch_bboxes=gt_bboxes,
                batch_idx=batch_idx,
                mask_shape=tuple(pred_logits.shape),
                device=pred_logits.device,
                batch_angles=gt_angles,
            )
            positive_mass = target_mask.sum()
            if positive_mass.item() > 0:
                pos_weight = ((target_mask.numel() - positive_mass) / positive_mass).clamp(1.0, 20.0)
            else:
                pos_weight = torch.ones((), device=pred_logits.device)
            loss_bce = F.binary_cross_entropy_with_logits(pred_logits, target_mask, pos_weight=pos_weight)
            loss_dice = calculate_dice_loss(torch.sigmoid(pred_logits), target_mask)
            scale_loss = 0.7 * loss_bce + 0.3 * loss_dice
            scale_weight = float(self.scale_weights.get(scale_idx, 0.5))
            weighted = weighted + scale_weight * scale_loss
            weight_sum += scale_weight
            self.last_mask_metrics[f"metric_mask_P{scale_idx + 2}"] = scale_loss.detach()
        return self.lambda_mask * weighted / max(weight_sum, 1e-8)

    def forward(
        self,
        preds: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch: dict[str, Any],
        dn_bboxes: torch.Tensor | None = None,
        dn_scores: torch.Tensor | None = None,
        dn_angles: torch.Tensor | None = None,
        dn_meta: dict[str, Any] | None = None,
        freq_masks: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        pred_bboxes, pred_scores, pred_angles = preds
        self._assert_finite_tensor("decoder_bboxes", pred_bboxes)
        self._assert_finite_tensor("decoder_scores", pred_scores)
        self._assert_finite_tensor("decoder_angles", pred_angles)
        total = self._stack_loss(
            pred_bboxes,
            pred_scores,
            pred_angles,
            batch,
            update_metrics=True,
        )

        if dn_meta is not None and dn_bboxes is not None and dn_scores is not None and dn_angles is not None:
            dn_indices = self._build_dn_match_indices(dn_meta, batch["gt_groups"], pred_bboxes.device)
            total.update(
                self._stack_loss(
                    dn_bboxes,
                    dn_scores,
                    dn_angles,
                    batch,
                    postfix="_dn",
                    match_indices=dn_indices,
                    update_metrics=False,
                )
            )
        else:
            zero = self._zero_from(pred_bboxes)
            for key in ("class", "bbox", "angle", "gwd", "riou", "corner"):
                total[f"loss_{key}_dn"] = zero
                if self.aux_loss:
                    total[f"loss_{key}_aux_dn"] = zero

        total["loss_fsa_mask"] = self._loss_frequency_masks(freq_masks, batch, pred_bboxes.device)
        self._assert_finite_loss_dict(total)
        return total
