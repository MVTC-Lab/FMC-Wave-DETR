# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model head modules."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils import NOT_MACOS14
from ultralytics.utils.ops import safe_normalize_angle_vector
from ultralytics.utils.tal import dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import TORCH_1_11, fuse_conv_and_bn, smart_inference_mode

from .block import DFL, SAVPE, BNContrastiveHead, ContrastiveHead, Proto, Proto26, RealNVP, Residual, SwiGLUFFN
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init

__all__ = "OBB", "Classify", "Detect", "Pose", "RTDETRDecoder", "WaveRTDETRDecoder", "WaveRTDETRDecoderOBB", "Segment", "YOLOEDetect", "YOLOESegment", "v10Detect"


class Detect(nn.Module):
    """YOLO Detect head for object detection models.

    This class implements the detection head used in YOLO models for predicting bounding boxes and class probabilities.
    It supports both training and inference modes, with optional end-to-end detection capabilities.

    Attributes:
        dynamic (bool): Force grid reconstruction.
        export (bool): Export mode flag.
        format (str): Export format.
        end2end (bool): End-to-end detection mode.
        max_det (int): Maximum detections per image.
        shape (tuple): Input shape.
        anchors (torch.Tensor): Anchor points.
        strides (torch.Tensor): Feature map strides.
        legacy (bool): Backward compatibility for v3/v5/v8/v9 models.
        xyxy (bool): Output format, xyxy or xywh.
        nc (int): Number of classes.
        nl (int): Number of detection layers.
        reg_max (int): DFL channels.
        no (int): Number of outputs per anchor.
        stride (torch.Tensor): Strides computed during build.
        cv2 (nn.ModuleList): Convolution layers for box regression.
        cv3 (nn.ModuleList): Convolution layers for classification.
        dfl (nn.Module): Distribution Focal Loss layer.
        one2one_cv2 (nn.ModuleList): One-to-one convolution layers for box regression.
        one2one_cv3 (nn.ModuleList): One-to-one convolution layers for classification.

    Methods:
        forward: Perform forward pass and return predictions.
        forward_end2end: Perform forward pass for end-to-end detection.
        bias_init: Initialize detection head biases.
        decode_bboxes: Decode bounding boxes from predictions.
        postprocess: Post-process model predictions.

    Examples:
        Create a detection head for 80 classes
        >>> detect = Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = detect(x)
    """

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models
    xyxy = False  # xyxy or xywh output

    def __init__(self, nc: int = 80, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize the YOLO detection layer with specified number of classes and channels.

        Args:
            nc (int): Number of classes.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = reg_max  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v5/v5/v8/v9/11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)

    @property
    def end2end(self):
        """Checks if the model has one2one for v5/v5/v8/v9/11 backward compatibility."""
        return hasattr(self, "one2one")

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        bs = x[0].shape[0]  # batch size
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)

    def forward(
        self, x: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.

        Args:
            x (dict[str, torch.Tensor]): List of feature maps from different detection layers.

        Returns:
            (torch.Tensor): Concatenated tensor of decoded bounding boxes and class probabilities.
        """
        # Inference path
        dbox = self._get_decode_boxes(x)
        return torch.cat((dbox, x["scores"].sigmoid()), 1)

    def _get_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get decoded boxes based on anchors and strides."""
        shape = x["feats"][0].shape  # BCHW
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape

        dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
        return dbox

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):  # from
            a[-1].bias.data[:] = 2.0  # box
            b[-1].bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / self.stride[i]) ** 2
            )  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):  # from
                a[-1].bias.data[:] = 2.0  # box
                b[-1].bias.data[: self.nc] = math.log(
                    5 / self.nc / (640 / self.stride[i]) ** 2
                )  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        """Decode bounding boxes from predictions."""
        return dist2bbox(
            bboxes,
            anchors,
            xywh=xywh and not self.end2end and not self.xyxy,
            dim=1,
        )

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        boxes, scores = preds.split([4, self.nc], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        return torch.cat([boxes, scores, conf], dim=-1)

    def get_topk_index(self, scores: torch.Tensor, max_det: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get top-k indices from scores.

        Args:
            scores (torch.Tensor): Scores tensor with shape (batch_size, num_anchors, num_classes).
            max_det (int): Maximum detections per image.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): Top scores, class indices, and filtered indices.
        """
        batch_size, anchors, nc = scores.shape  # i.e. shape(16,8400,84)
        # Use max_det directly during export for TensorRT compatibility (requires k to be constant),
        # otherwise use min(max_det, anchors) for safety with small inputs during Python inference
        k = max_det if self.export else min(max_det, anchors)
        ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
        scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(k)
        idx = ori_index[torch.arange(batch_size)[..., None], index // nc]  # original index
        return scores[..., None], (index % nc)[..., None].float(), idx

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = None


class Segment(Detect):
    """YOLO Segment head for segmentation models.

    This class extends the Detect head to include mask prediction capabilities for instance segmentation tasks.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto): Prototype generation module.
        cv4 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a segmentation head
        >>> segment = Segment(nc=80, nm=32, npr=256, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = segment(x)
    """

    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, mask_head=self.one2one_cv4)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x[0])  # mask protos
        if isinstance(preds, dict):  # training and validating during training
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with mask coefficients."""
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, mask_head: torch.nn.Module
    ) -> torch.Tensor:
        """Concatenates and returns predicted bounding boxes, class probabilities, and mask coefficients."""
        preds = super().forward_head(x, box_head, cls_head)
        if mask_head is not None:
            bs = x[0].shape[0]  # batch size
            preds["mask_coefficient"] = torch.cat([mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + nm) with last dimension
                format [x, y, w, h, class_probs, mask_coefficient].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + nm) and last
                dimension format [x, y, w, h, max_class_prob, class_index, mask_coefficient].
        """
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = self.cv4 = None


class Segment26(Segment):
    """YOLO26 Segment head for segmentation models.

    This class extends the Detect head to include mask prediction capabilities for instance segmentation tasks.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto): Prototype generation module.
        cv4 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a segmentation head
        >>> segment = Segment26(nc=80, nm=32, npr=256, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = segment(x)
    """

    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, nm, npr, reg_max, end2end, ch)
        self.proto = Proto26(ch, self.npr, self.nm, nc)  # protos

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = Detect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x)  # mask protos
        if isinstance(preds, dict):  # training and validating during training
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = (
                    tuple(p.detach() for p in proto) if isinstance(proto, tuple) else proto.detach()
                )
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def fuse(self) -> None:
        """Remove the one2many head and extra part of proto module for inference optimization."""
        super().fuse()
        if hasattr(self.proto, "fuse"):
            self.proto.fuse()


class OBB(Detect):
    """YOLO OBB detection head for detection with rotation models.

    This class extends the Detect head to include oriented bounding box prediction with rotation angles.

    Attributes:
        ne (int): Number of extra parameters.
        cv4 (nn.ModuleList): Convolution layers for angle prediction.
        angle (torch.Tensor): Predicted rotation angles.

    Methods:
        forward: Concatenate and return predicted bounding boxes and class probabilities.
        decode_bboxes: Decode rotated bounding boxes.

    Examples:
        Create an OBB detection head
        >>> obb = OBB(nc=80, ne=1, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = obb(x)
    """

    def __init__(self, nc: int = 80, ne: int = 1, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`.

        Args:
            nc (int): Number of classes.
            ne (int): Number of extra parameters.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, angle_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, angle_head=self.one2one_cv4)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with rotation angles."""
        # For decode_bboxes convenience
        self.angle = x["angle"]  # TODO: need to test obb
        preds = super()._inference(x)
        return torch.cat([preds, x["angle"]], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, angle_head: torch.nn.Module
    ) -> torch.Tensor:
        """Concatenates and returns predicted bounding boxes, class probabilities, and angles."""
        preds = super().forward_head(x, box_head, cls_head)
        if angle_head is not None:
            bs = x[0].shape[0]  # batch size
            angle = torch.cat(
                [angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
            )  # OBB theta logits
            angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
            preds["angle"] = angle
        return preds

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + ne) with last dimension
                format [x, y, w, h, class_probs, angle].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 7) and last
                dimension format [x, y, w, h, max_class_prob, class_index, angle].
        """
        boxes, scores, angle = preds.split([4, self.nc, self.ne], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        angle = angle.gather(dim=1, index=idx.repeat(1, 1, self.ne))
        return torch.cat([boxes, scores, conf, angle], dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = self.cv4 = None


class OBB26(OBB):
    """YOLO26 OBB detection head for detection with rotation models. This class extends the OBB head with modified angle
    processing that outputs raw angle predictions without sigmoid transformation, compared to the original
    OBB class.

    Attributes:
        ne (int): Number of extra parameters.
        cv4 (nn.ModuleList): Convolution layers for angle prediction.
        angle (torch.Tensor): Predicted rotation angles.

    Methods:
        forward_head: Concatenate and return predicted bounding boxes, class probabilities, and raw angles.

    Examples:
        Create an OBB26 detection head
        >>> obb26 = OBB26(nc=80, ne=1, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = obb26(x).
    """

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, angle_head: torch.nn.Module
    ) -> torch.Tensor:
        """Concatenates and returns predicted bounding boxes, class probabilities, and raw angles."""
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if angle_head is not None:
            bs = x[0].shape[0]  # batch size
            angle = torch.cat(
                [angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
            )  # OBB theta logits (raw output without sigmoid transformation)
            preds["angle"] = angle
        return preds


class Pose(Detect):
    """YOLO Pose head for keypoints models.

    This class extends the Detect head to include keypoint prediction capabilities for pose estimation tasks.

    Attributes:
        kpt_shape (tuple): Number of keypoints and dimensions (2 for x,y or 3 for x,y,visible).
        nk (int): Total number of keypoint values.
        cv4 (nn.ModuleList): Convolution layers for keypoint prediction.

    Methods:
        forward: Perform forward pass through YOLO model and return predictions.
        kpts_decode: Decode keypoints from predictions.

    Examples:
        Create a pose detection head
        >>> pose = Pose(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = pose(x)
    """

    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize YOLO network with default parameters and Convolutional Layers.

        Args:
            nc (int): Number of classes.
            kpt_shape (tuple): Number of keypoints, number of dims (2 for x,y or 3 for x,y,visible).
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, pose_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, pose_head=self.one2one_cv4)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with keypoints."""
        preds = super()._inference(x)
        return torch.cat([preds, self.kpts_decode(x["kpts"])], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, pose_head: torch.nn.Module
    ) -> torch.Tensor:
        """Concatenates and returns predicted bounding boxes, class probabilities, and keypoints."""
        preds = super().forward_head(x, box_head, cls_head)
        if pose_head is not None:
            bs = x[0].shape[0]  # batch size
            preds["kpts"] = torch.cat([pose_head[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + nk) with last dimension
                format [x, y, w, h, class_probs, keypoints].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + self.nk) and
                last dimension format [x, y, w, h, max_class_prob, class_index, keypoints].
        """
        boxes, scores, kpts = preds.split([4, self.nc, self.nk], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        kpts = kpts.gather(dim=1, index=idx.repeat(1, 1, self.nk))
        return torch.cat([boxes, scores, conf, kpts], dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = self.cv4 = None

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        """Decode keypoints from predictions."""
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                if NOT_MACOS14:
                    y[:, 2::ndim].sigmoid_()
                else:  # Apple macOS14 MPS bug https://github.com/ultralytics/ultralytics/pull/21878
                    y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Pose26(Pose):
    """YOLO26 Pose head for keypoints models.

    This class extends the Detect head to include keypoint prediction capabilities for pose estimation tasks.

    Attributes:
        kpt_shape (tuple): Number of keypoints and dimensions (2 for x,y or 3 for x,y,visible).
        nk (int): Total number of keypoint values.
        cv4 (nn.ModuleList): Convolution layers for keypoint prediction.

    Methods:
        forward: Perform forward pass through YOLO model and return predictions.
        kpts_decode: Decode keypoints from predictions.

    Examples:
        Create a pose detection head
        >>> pose = Pose(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = pose(x)
    """

    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize YOLO network with default parameters and Convolutional Layers.

        Args:
            nc (int): Number of classes.
            kpt_shape (tuple): Number of keypoints, number of dims (2 for x,y or 3 for x,y,visible).
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, kpt_shape, reg_max, end2end, ch)
        self.flow_model = RealNVP()

        c4 = max(ch[0] // 4, kpt_shape[0] * (kpt_shape[1] + 2))
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3)) for x in ch)

        self.cv4_kpts = nn.ModuleList(nn.Conv2d(c4, self.nk, 1) for _ in ch)
        self.nk_sigma = kpt_shape[0] * 2  # sigma_x, sigma_y for each keypoint
        self.cv4_sigma = nn.ModuleList(nn.Conv2d(c4, self.nk_sigma, 1) for _ in ch)

        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)
            self.one2one_cv4_kpts = copy.deepcopy(self.cv4_kpts)
            self.one2one_cv4_sigma = copy.deepcopy(self.cv4_sigma)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(
            box_head=self.cv2,
            cls_head=self.cv3,
            pose_head=self.cv4,
            kpts_head=self.cv4_kpts,
            kpts_sigma_head=self.cv4_sigma,
        )

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            pose_head=self.one2one_cv4,
            kpts_head=self.one2one_cv4_kpts,
            kpts_sigma_head=self.one2one_cv4_sigma,
        )

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        pose_head: torch.nn.Module,
        kpts_head: torch.nn.Module,
        kpts_sigma_head: torch.nn.Module,
    ) -> torch.Tensor:
        """Concatenates and returns predicted bounding boxes, class probabilities, and keypoints."""
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if pose_head is not None:
            bs = x[0].shape[0]  # batch size
            features = [pose_head[i](x[i]) for i in range(self.nl)]
            preds["kpts"] = torch.cat([kpts_head[i](features[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
            if self.training:
                preds["kpts_sigma"] = torch.cat(
                    [kpts_sigma_head[i](features[i]).view(bs, self.nk_sigma, -1) for i in range(self.nl)], 2
                )
        return preds

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        super().fuse()
        self.cv4_kpts = self.cv4_sigma = self.flow_model = self.one2one_cv4_sigma = None

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        """Decode keypoints from predictions."""
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            # NCNN fix
            a = (y[:, :, :2] + self.anchors) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                if NOT_MACOS14:
                    y[:, 2::ndim].sigmoid_()
                else:  # Apple macOS14 MPS bug https://github.com/ultralytics/ultralytics/pull/21878
                    y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] + self.anchors[0]) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] + self.anchors[1]) * self.strides
            return y


class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2).

    This class implements a classification head that transforms feature maps into class predictions.

    Attributes:
        export (bool): Export mode flag.
        conv (Conv): Convolutional layer for feature transformation.
        pool (nn.AdaptiveAvgPool2d): Global average pooling layer.
        drop (nn.Dropout): Dropout layer for regularization.
        linear (nn.Linear): Linear layer for final classification.

    Methods:
        forward: Perform forward pass of the YOLO model on input image data.

    Examples:
        Create a classification head
        >>> classify = Classify(c1=1024, c2=1000)
        >>> x = torch.randn(1, 1024, 20, 20)
        >>> output = classify(x)
    """

    export = False  # export mode

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output classes.
            k (int, optional): Kernel size.
            s (int, optional): Stride.
            p (int, optional): Padding.
            g (int, optional): Groups.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor | tuple:
        """Perform forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # get final output
        return y if self.export else (y, x)


class WorldDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings.

    This class extends the standard Detect head to incorporate text embeddings for enhanced semantic understanding in
    object detection tasks.

    Attributes:
        cv3 (nn.ModuleList): Convolution layers for embedding features.
        cv4 (nn.ModuleList): Contrastive head layers for text-vision alignment.

    Methods:
        forward: Concatenate and return predicted bounding boxes and class probabilities.
        bias_init: Initialize detection head biases.

    Examples:
        Create a WorldDetect head
        >>> world_detect = WorldDetect(nc=80, embed=512, with_bn=False, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = world_detect(x, text)
    """

    def __init__(
        self,
        nc: int = 80,
        embed: int = 512,
        with_bn: bool = False,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        """Initialize YOLO detection layer with nc classes and layer channels ch.

        Args:
            nc (int): Number of classes.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max=reg_max, end2end=end2end, ch=ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> dict[str, torch.Tensor] | tuple:
        """Concatenate and return predicted bounding boxes and class probabilities."""
        feats = [xi.clone() for xi in x]  # save original features for anchor generation
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        bs = x[0].shape[0]
        x_cat = torch.cat([xi.view(bs, self.no, -1) for xi in x], 2)
        boxes, scores = x_cat.split((self.reg_max * 4, self.nc), 1)
        preds = dict(boxes=boxes, scores=scores, feats=feats)
        if self.training:
            return preds
        y = self._inference(preds)
        return y if self.export else (y, preds)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class LRPCHead(nn.Module):
    """Lightweight Region Proposal and Classification Head for efficient object detection.

    This head combines region proposal filtering with classification to enable efficient detection with dynamic
    vocabulary support.

    Attributes:
        vocab (nn.Module): Vocabulary/classification layer.
        pf (nn.Module): Proposal filter module.
        loc (nn.Module): Localization module.
        enabled (bool): Whether the head is enabled.

    Methods:
        conv2linear: Convert a 1x1 convolutional layer to a linear layer.
        forward: Process classification and localization features to generate detection proposals.

    Examples:
        Create an LRPC head
        >>> vocab = nn.Conv2d(256, 80, 1)
        >>> pf = nn.Conv2d(256, 1, 1)
        >>> loc = nn.Conv2d(256, 4, 1)
        >>> head = LRPCHead(vocab, pf, loc, enabled=True)
    """

    def __init__(self, vocab: nn.Module, pf: nn.Module, loc: nn.Module, enabled: bool = True):
        """Initialize LRPCHead with vocabulary, proposal filter, and localization components.

        Args:
            vocab (nn.Module): Vocabulary/classification module.
            pf (nn.Module): Proposal filter module.
            loc (nn.Module): Localization module.
            enabled (bool): Whether to enable the head functionality.
        """
        super().__init__()
        self.vocab = self.conv2linear(vocab) if enabled else vocab
        self.pf = pf
        self.loc = loc
        self.enabled = enabled

    @staticmethod
    def conv2linear(conv: nn.Conv2d) -> nn.Linear:
        """Convert a 1x1 convolutional layer to a linear layer."""
        assert isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1)
        linear = nn.Linear(conv.in_channels, conv.out_channels)
        linear.weight.data = conv.weight.view(conv.out_channels, -1).data
        linear.bias.data = conv.bias.data
        return linear

    def forward(self, cls_feat: torch.Tensor, loc_feat: torch.Tensor, conf: float) -> tuple[tuple, torch.Tensor]:
        """Process classification and localization features to generate detection proposals."""
        if self.enabled:
            pf_score = self.pf(cls_feat)[0, 0].flatten(0)
            mask = pf_score.sigmoid() > conf
            cls_feat = cls_feat.flatten(2).transpose(-1, -2)
            cls_feat = self.vocab(cls_feat[:, mask] if conf else cls_feat * mask.unsqueeze(-1).int())
            return self.loc(loc_feat), cls_feat.transpose(-1, -2), mask
        else:
            cls_feat = self.vocab(cls_feat)
            loc_feat = self.loc(loc_feat)
            return (
                loc_feat,
                cls_feat.flatten(2),
                torch.ones(cls_feat.shape[2] * cls_feat.shape[3], device=cls_feat.device, dtype=torch.bool),
            )


class YOLOEDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings.

    This class extends the standard Detect head to support text-guided detection with enhanced semantic understanding
    through text embeddings and visual prompt embeddings.

    Attributes:
        is_fused (bool): Whether the model is fused for inference.
        cv3 (nn.ModuleList): Convolution layers for embedding features.
        cv4 (nn.ModuleList): Contrastive head layers for text-vision alignment.
        reprta (Residual): Residual block for text prompt embeddings.
        savpe (SAVPE): Spatial-aware visual prompt embeddings module.
        embed (int): Embedding dimension.

    Methods:
        fuse: Fuse text features with model weights for efficient inference.
        get_tpe: Get text prompt embeddings with normalization.
        get_vpe: Get visual prompt embeddings with spatial awareness.
        forward_lrpc: Process features with fused text embeddings for prompt-free model.
        forward: Process features with class prompt embeddings to generate detections.
        bias_init: Initialize biases for detection heads.

    Examples:
        Create a YOLOEDetect head
        >>> yoloe_detect = YOLOEDetect(nc=80, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> cls_pe = torch.randn(1, 80, 512)
        >>> outputs = yoloe_detect(x, cls_pe)
    """

    is_fused = False

    def __init__(
        self, nc: int = 80, embed: int = 512, with_bn: bool = False, reg_max=16, end2end=False, ch: tuple = ()
    ):
        """Initialize YOLO detection layer with nc classes and layer channels ch.

        Args:
            nc (int): Number of classes.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        c3 = max(ch[0], min(self.nc, 100))
        assert c3 <= embed
        assert with_bn
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, embed, 1),
                )
                for x in ch
            )
        )
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)
        if end2end:
            self.one2one_cv3 = copy.deepcopy(self.cv3)  # overwrite with new cv3
            self.one2one_cv4 = copy.deepcopy(self.cv4)

        self.reprta = Residual(SwiGLUFFN(embed, embed))
        self.savpe = SAVPE(ch, c3, embed)
        self.embed = embed

    @smart_inference_mode()
    def fuse(self, txt_feats: torch.Tensor = None):
        """Fuse text features with model weights for efficient inference."""
        if txt_feats is None:  # means eliminate one2many branch
            self.cv2 = self.cv3 = self.cv4 = None
            return
        if self.is_fused:
            return

        assert not self.training
        txt_feats = txt_feats.to(torch.float32).squeeze(0)
        self._fuse_tp(txt_feats, self.cv3, self.cv4)
        if self.end2end:
            self._fuse_tp(txt_feats, self.one2one_cv3, self.one2one_cv4)
        del self.reprta
        self.reprta = nn.Identity()
        self.is_fused = True

    def _fuse_tp(self, txt_feats: torch.Tensor, cls_head: torch.nn.Module, bn_head: torch.nn.Module) -> None:
        """Fuse text prompt embeddings with model weights for efficient inference."""
        for cls_h, bn_h in zip(cls_head, bn_head):
            assert isinstance(cls_h, nn.Sequential)
            assert isinstance(bn_h, BNContrastiveHead)
            conv = cls_h[-1]
            assert isinstance(conv, nn.Conv2d)
            logit_scale = bn_h.logit_scale
            bias = bn_h.bias
            norm = bn_h.norm

            t = txt_feats * logit_scale.exp()
            conv: nn.Conv2d = fuse_conv_and_bn(conv, norm)

            w = conv.weight.data.squeeze(-1).squeeze(-1)
            b = conv.bias.data

            w = t @ w
            b1 = (t @ b.reshape(-1).unsqueeze(-1)).squeeze(-1)
            b2 = torch.ones_like(b1) * bias

            conv = (
                nn.Conv2d(
                    conv.in_channels,
                    w.shape[0],
                    kernel_size=1,
                )
                .requires_grad_(False)
                .to(conv.weight.device)
            )

            conv.weight.data.copy_(w.unsqueeze(-1).unsqueeze(-1))
            conv.bias.data.copy_(b1 + b2)
            cls_h[-1] = conv

            bn_h.fuse()

    def get_tpe(self, tpe: torch.Tensor | None) -> torch.Tensor | None:
        """Get text prompt embeddings with normalization."""
        return None if tpe is None else F.normalize(self.reprta(tpe), dim=-1, p=2)

    def get_vpe(self, x: list[torch.Tensor], vpe: torch.Tensor) -> torch.Tensor:
        """Get visual prompt embeddings with spatial awareness."""
        if vpe.shape[1] == 0:  # no visual prompt embeddings
            return torch.zeros(x[0].shape[0], 0, self.embed, device=x[0].device)
        if vpe.ndim == 4:  # (B, N, H, W)
            vpe = self.savpe(x, vpe)
        assert vpe.ndim == 3  # (B, N, D)
        return vpe

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Process features with class prompt embeddings to generate detections."""
        if hasattr(self, "lrpc"):  # for prompt-free inference
            return self.forward_lrpc(x[:3])
        return super().forward(x)

    def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Process features with fused text embeddings to generate detections for prompt-free model."""
        boxes, scores, index = [], [], []
        bs = x[0].shape[0]
        cv2 = self.cv2 if not self.end2end else self.one2one_cv2
        cv3 = self.cv3 if not self.end2end else self.one2one_cv2
        for i in range(self.nl):
            cls_feat = cv3[i](x[i])
            loc_feat = cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            box, score, idx = self.lrpc[i](
                cls_feat,
                loc_feat,
                0 if self.export and not self.dynamic else getattr(self, "conf", 0.001),
            )
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        preds = dict(boxes=torch.cat(boxes, 2), scores=torch.cat(scores, 2), feats=x, index=torch.cat(index))
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _get_decode_boxes(self, x):
        """Decode predicted bounding boxes for inference."""
        dbox = super()._get_decode_boxes(x)
        if hasattr(self, "lrpc"):
            dbox = dbox if self.export and not self.dynamic else dbox[..., x["index"]]
        return dbox

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v5/v5/v8/v9/11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, contrastive_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, contrastive_head=self.one2one_cv4)

    def forward_head(self, x, box_head, cls_head, contrastive_head):
        """Concatenates and returns predicted bounding boxes, class probabilities, and text embeddings."""
        assert len(x) == 4, f"Expected 4 features including 3 feature maps and 1 text embeddings, but got {len(x)}."
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        bs = x[0].shape[0]  # batch size
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        self.nc = x[-1].shape[1]
        scores = torch.cat(
            [contrastive_head[i](cls_head[i](x[i]), x[-1]).reshape(bs, self.nc, -1) for i in range(self.nl)], dim=-1
        )
        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        return dict(boxes=boxes, scores=scores, feats=x[:3])

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        for i, (a, b, c) in enumerate(
            zip(self.one2many["box_head"], self.one2many["cls_head"], self.one2many["contrastive_head"])
        ):
            a[-1].bias.data[:] = 2.0  # box
            b[-1].bias.data[:] = 0.0
            c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, (a, b, c) in enumerate(
                zip(self.one2one["box_head"], self.one2one["cls_head"], self.one2one["contrastive_head"])
            ):
                a[-1].bias.data[:] = 2.0  # box
                b[-1].bias.data[:] = 0.0
                c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)


class YOLOESegment(YOLOEDetect):
    """YOLO segmentation head with text embedding capabilities.

    This class extends YOLOEDetect to include mask prediction capabilities for instance segmentation tasks with
    text-guided semantic understanding.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto): Prototype generation module.
        cv5 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a YOLOESegment head
        >>> yoloe_segment = YOLOESegment(nc=80, nm=32, npr=256, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = yoloe_segment(x, text)
    """

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = False,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        """Initialize YOLOESegment with class count, mask parameters, and embedding dimensions.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)

        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v5/v5/v8/v9/11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv5, contrastive_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            mask_head=self.one2one_cv5,
            contrastive_head=self.one2one_cv4,
        )

    def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Process features with fused text embeddings to generate detections for prompt-free model."""
        boxes, scores, index = [], [], []
        bs = x[0].shape[0]
        cv2 = self.cv2 if not self.end2end else self.one2one_cv2
        cv3 = self.cv3 if not self.end2end else self.one2one_cv3
        cv5 = self.cv5 if not self.end2end else self.one2one_cv5
        for i in range(self.nl):
            cls_feat = cv3[i](x[i])
            loc_feat = cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            box, score, idx = self.lrpc[i](
                cls_feat,
                loc_feat,
                0 if self.export and not self.dynamic else getattr(self, "conf", 0.001),
            )
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        mc = torch.cat([cv5[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        index = torch.cat(index)
        preds = dict(
            boxes=torch.cat(boxes, 2),
            scores=torch.cat(scores, 2),
            feats=x,
            index=index,
            mask_coefficient=mc * index.int() if self.export and not self.dynamic else mc[..., index],
        )
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x[0])  # mask protos
        if isinstance(preds, dict):  # training and validating during training
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with mask coefficients."""
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        mask_head: torch.nn.Module,
        contrastive_head: torch.nn.Module,
    ) -> torch.Tensor:
        """Concatenates and returns predicted bounding boxes, class probabilities, and mask coefficients."""
        preds = super().forward_head(x, box_head, cls_head, contrastive_head)
        if mask_head is not None:
            bs = x[0].shape[0]  # batch size
            preds["mask_coefficient"] = torch.cat([mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + nm) with last dimension
                format [x, y, w, h, class_probs, mask_coefficient].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + nm) and last
                dimension format [x, y, w, h, max_class_prob, class_index, mask_coefficient].
        """
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self, txt_feats: torch.Tensor = None):
        """Fuse text features with model weights for efficient inference."""
        super().fuse(txt_feats)
        if txt_feats is None:  # means eliminate one2many branch
            self.cv5 = None
            if hasattr(self.proto, "fuse"):
                self.proto.fuse()
            return


class YOLOESegment26(YOLOESegment):
    """YOLOE-style segmentation head module using Proto26 for mask generation.

    This class extends the YOLOEDetect functionality to include segmentation capabilities by integrating a prototype
    generation module and convolutional layers to predict mask coefficients.

    Args:
        nc (int): Number of classes. Defaults to 80.
        nm (int): Number of masks. Defaults to 32.
        npr (int): Number of prototype channels. Defaults to 256.
        embed (int): Embedding dimensionality. Defaults to 512.
        with_bn (bool): Whether to use Batch Normalization. Defaults to False.
        reg_max (int): Maximum regression value for bounding boxes. Defaults to 16.
        end2end (bool): Whether to use end-to-end detection mode. Defaults to False.
        ch (tuple[int, ...]): Input channels for each scale.

    Attributes:
        nm (int): Number of segmentation masks.
        npr (int): Number of prototype channels.
        proto (Proto26): Prototype generation module for segmentation.
        cv5 (nn.ModuleList): Convolutional layers for generating mask coefficients from features.
        one2one_cv5 (nn.ModuleList, optional): Deep copy of cv5 for end-to-end detection branches.
    """

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = False,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        """Initialize YOLOESegment26 with class count, mask parameters, and embedding dimensions."""
        YOLOEDetect.__init__(self, nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto26(ch, self.npr, self.nm, nc)  # protos

        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = YOLOEDetect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto([xi.detach() for xi in x], return_semseg=False)  # mask protos

        if isinstance(preds, dict):  # training and validating during training
            if self.end2end and not hasattr(self, "lrpc"):  # not prompt-free
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)


class RTDETRDecoder(nn.Module):
    """Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.

    Attributes:
        export (bool): Export mode flag.
        hidden_dim (int): Dimension of hidden layers.
        nhead (int): Number of heads in multi-head attention.
        nl (int): Number of feature levels.
        nc (int): Number of classes.
        num_queries (int): Number of query points.
        num_decoder_layers (int): Number of decoder layers.
        input_proj (nn.ModuleList): Input projection layers for backbone features.
        decoder (DeformableTransformerDecoder): Transformer decoder module.
        denoising_class_embed (nn.Embedding): Class embeddings for denoising.
        num_denoising (int): Number of denoising queries.
        label_noise_ratio (float): Label noise ratio for training.
        box_noise_scale (float): Box noise scale for training.
        learnt_init_query (bool): Whether to learn initial query embeddings.
        tgt_embed (nn.Embedding): Target embeddings for queries.
        query_pos_head (MLP): Query position head.
        enc_output (nn.Sequential): Encoder output layers.
        enc_score_head (nn.Linear): Encoder score prediction head.
        enc_bbox_head (MLP): Encoder bbox prediction head.
        dec_score_head (nn.ModuleList): Decoder score prediction heads.
        dec_bbox_head (nn.ModuleList): Decoder bbox prediction heads.

    Methods:
        forward: Run forward pass and return bounding box and classification scores.

    Examples:
        Create an RTDETRDecoder
        >>> decoder = RTDETRDecoder(nc=80, ch=(512, 1024, 2048), hd=256, nq=300)
        >>> x = [torch.randn(1, 512, 64, 64), torch.randn(1, 1024, 32, 32), torch.randn(1, 2048, 16, 16)]
        >>> outputs = decoder(x)
    """

    export = False  # export mode
    shapes = []
    anchors = torch.empty(0)
    valid_mask = torch.empty(0)
    dynamic = False

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,  # hidden dim
        nq: int = 300,  # num queries
        ndp: int = 4,  # num decoder points
        nh: int = 8,  # num head
        ndl: int = 6,  # num decoder layers
        d_ffn: int = 1024,  # dim of feedforward
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        # Training args
        nd: int = 100,  # num denoising
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
    ):
        """Initialize the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes.
            ch (tuple): Channels in the backbone feature maps.
            hd (int): Dimension of hidden layers.
            nq (int): Number of query points.
            ndp (int): Number of decoder points.
            nh (int): Number of heads in multi-head attention.
            ndl (int): Number of decoder layers.
            d_ffn (int): Dimension of the feed-forward networks.
            dropout (float): Dropout rate.
            act (nn.Module): Activation function.
            eval_idx (int): Evaluation index.
            nd (int): Number of denoising.
            label_noise_ratio (float): Label noise ratio.
            box_noise_scale (float): Box noise scale.
            learnt_init_query (bool): Whether to learn initial query embeddings.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x: list[torch.Tensor], batch: dict | None = None, freq_masks: torch.Tensor | None = None, **kwargs) -> tuple | torch.Tensor:
        """Run the forward pass of the module, returning bounding box and classification scores for the input.

        Args:
            x (list[torch.Tensor]): List of feature maps from the backbone.
            batch (dict, optional): Batch information for training.

        Returns:
            outputs (tuple | torch.Tensor): During training, returns a tuple of bounding boxes, scores, and other
                metadata. During inference, returns a tensor of shape (bs, 300, 4+nc) containing bounding boxes and
                class scores.
        """
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    @staticmethod
    def _generate_anchors(
        shapes: list[list[int]],
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        eps: float = 1e-2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate anchor bounding boxes for given shapes with specific grid size and validate them.

        Args:
            shapes (list): List of feature map shapes.
            grid_size (float, optional): Base size of grid cells.
            dtype (torch.dtype, optional): Data type for tensors.
            device (str, optional): Device to create tensors on.
            eps (float, optional): Small value for numerical stability.

        Returns:
            anchors (torch.Tensor): Generated anchor boxes.
            valid_mask (torch.Tensor): Valid mask for anchors.
        """
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, list[list[int]]]:
        """Process and return encoder inputs by getting projection features from input and concatenating them.

        Args:
            x (list[torch.Tensor]): List of feature maps from the backbone.

        Returns:
            feats (torch.Tensor): Processed features.
            shapes (list): List of feature map shapes.
        """
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(
        self,
        feats: torch.Tensor,
        shapes: list[list[int]],
        dn_embed: torch.Tensor | None = None,
        dn_bbox: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate and prepare the input required for the decoder from the provided features and shapes.

        Args:
            feats (torch.Tensor): Processed features from encoder.
            shapes (list): List of feature map shapes.
            dn_embed (torch.Tensor, optional): Denoising embeddings.
            dn_bbox (torch.Tensor, optional): Denoising bounding boxes.

        Returns:
            embeddings (torch.Tensor): Query embeddings for decoder.
            refer_bbox (torch.Tensor): Reference bounding boxes.
            enc_bboxes (torch.Tensor): Encoded bounding boxes.
            enc_scores (torch.Tensor): Encoded scores.
        """
        bs = feats.shape[0]
        if self.dynamic or self.shapes != shapes:
            self.anchors, self.valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
            self.shapes = shapes

        # Prepare input for decoder
        features = self.enc_output(self.valid_mask * feats)  # bs, h*w, 256
        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs*num_queries,)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs*num_queries,)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = self.anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    def _reset_parameters(self):
        """Initialize or reset the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)

# WaveRTDETRDecoder v1.0
from ultralytics.utils.torch_utils import TORCH_1_11
from ultralytics.nn.modules.transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from ultralytics.nn.modules.utils import bias_init_with_prob, linear_init
from torch.nn.init import constant_, xavier_uniform_

import torch
import torch.nn as nn
import torch.nn.functional as F

SC_SBFG_LOG_METRIC_KEYS = (
    "metric_sc_selected_p2", "metric_sc_selected_p3", "metric_sc_selected_p4",
    "metric_sc_evidence_p2", "metric_sc_evidence_p3", "metric_sc_evidence_p4",
    "metric_sc_mask_raw_mean_p2", "metric_sc_mask_raw_mean_p3", "metric_sc_mask_raw_mean_p4",
    "metric_sc_mask_cal_mean_p2", "metric_sc_mask_cal_mean_p3", "metric_sc_mask_cal_mean_p4",
    "metric_sc_peak_ratio_p2", "metric_sc_peak_ratio_p3", "metric_sc_peak_ratio_p4",
    "metric_sc_query_entropy", "metric_sc_fixed_fallback_rate", "metric_sc_global_fallback_rate",
    "metric_sc_fallback_count",
    "metric_sc_priority_top_mean_p2", "metric_sc_priority_top_mean_p3", "metric_sc_priority_top_mean_p4",
    "metric_sc_selected_std_p2", "metric_sc_selected_std_p3", "metric_sc_selected_std_p4",
    "metric_sc_selected_min_p2", "metric_sc_selected_min_p3", "metric_sc_selected_min_p4",
    "metric_sc_selected_max_p2", "metric_sc_selected_max_p3", "metric_sc_selected_max_p4",
    "metric_sc_selected_p10_p2", "metric_sc_selected_p10_p3", "metric_sc_selected_p10_p4",
    "metric_sc_selected_p90_p2", "metric_sc_selected_p90_p3", "metric_sc_selected_p90_p4",
    "metric_sc_effective_candidates_p2", "metric_sc_effective_candidates_p3", "metric_sc_effective_candidates_p4",
    "metric_sc_dynamic_ratio", "metric_sc_temperature", "metric_sc_evidence_query_kl",
    "metric_sc_floor_hit_rate", "metric_sc_cap_hit_rate", "metric_sc_capacity_relax_rate",
    "metric_sc_alpha_p2", "metric_sc_alpha_p3",
    "metric_cgsc_g43_mean", "metric_cgsc_g43_active",
    "metric_cgsc_g32_mean", "metric_cgsc_g32_active",
    "metric_lrm_p2_pos_ratio", "metric_lrm_p3_pos_ratio", "metric_lrm_p4_pos_ratio",
    "metric_lrm_p2_mean_abs", "metric_lrm_p3_mean_abs", "metric_lrm_p4_mean_abs",
)


class WaveRTDETRDecoder(nn.Module):
    """
    修正版 WaveRTDETRDecoder：
    1. anchors/valid_mask 注册为非持久 buffer；
    2. Top-K 使用显式 batch 索引，不再 flatten 后混合索引；
    3. 无效 anchor 的分类分数置为 -inf；
    4. 严格检查并自动对齐 P2/P3/P4 掩码；
    5. 频率引导采用残差式增强，避免早期错误掩码强抑制真实目标；
    6. batch 显式传入 get_cdn_group，恢复 DN 去噪训练。
    """
    export = False
    dynamic = False

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,
        nq: int = 300,
        ndp: int = 4,
        nh: int = 8,
        ndl: int = 6,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        nd: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
        freq_query_beta: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl
        self.freq_query_beta = float(freq_query_beta)

        self.input_proj = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(x, hd, 1, bias=False),
                nn.BatchNorm2d(hd),
            )
            for x in ch
        )

        decoder_layer = DeformableTransformerDecoderLayer(
            hd,
            nh,
            d_ffn,
            dropout,
            act,
            self.nl,
            ndp,
        )
        self.decoder = DeformableTransformerDecoder(
            hd,
            decoder_layer,
            ndl,
            eval_idx,
        )

        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)

        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        self.enc_output = nn.Sequential(
            nn.Linear(hd, hd),
            nn.LayerNorm(hd),
        )
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        self.dec_score_head = nn.ModuleList(
            [nn.Linear(hd, nc) for _ in range(ndl)]
        )
        self.dec_bbox_head = nn.ModuleList(
            [MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)]
        )

        self.shapes: list[list[int]] = []
        self.register_buffer(
            "anchors",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "valid_mask",
            torch.empty(0, dtype=torch.bool),
            persistent=False,
        )
        # Two scalar parameters are present in every WaveRTDETRDecoder state dict so old/new YAMLs
        # keep an identical layout. They are used only by SC-v2 residual-logit calibration.
        self.sc_alpha2 = nn.Parameter(torch.zeros(()))
        self.sc_alpha3 = nn.Parameter(torch.zeros(()))

        self._reset_parameters()
        self.configure_freq_guidance(None)

    @staticmethod
    def _sc_triplet(name, value, cast=float):
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"{name} must contain exactly three values")
        result = tuple(cast(item) for item in value)
        if cast is float and not all(math.isfinite(item) for item in result):
            raise ValueError(f"{name} must contain finite values")
        return result

    @staticmethod
    def _sc_pair(name, value, cast=float):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{name} must contain exactly two values")
        result = tuple(cast(item) for item in value)
        if cast is float and not all(math.isfinite(item) for item in result):
            raise ValueError(f"{name} must contain finite values")
        return result

    @staticmethod
    def _sc_quad(name, value):
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError(f"{name} must contain exactly four values")
        result = tuple(float(item) for item in value)
        if not all(math.isfinite(item) and item >= 0.0 for item in result) or sum(result) <= 0.0:
            raise ValueError(f"{name} must contain nonnegative finite values with a positive sum")
        return result

    @staticmethod
    def _sc_nonnegative_int(name, value):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        return value

    @classmethod
    def _sc_quota_triplet(cls, name, value):
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"{name} must contain exactly three nonnegative integers")
        return tuple(cls._sc_nonnegative_int(name, item) for item in value)

    def configure_freq_guidance(self, config: dict | None) -> None:
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise TypeError(f"freq_guidance must be a mapping, got {type(config).__name__}")
        semantic = config.get("semantic_calibration", {}) or {}
        query = config.get("query", {}) or {}
        mask_loss = config.get("mask_loss", {}) or {}
        diagnostics = config.get("diagnostics", {}) or {}
        for name, section in (("semantic_calibration", semantic), ("query", query),
                              ("mask_loss", mask_loss), ("diagnostics", diagnostics)):
            if not isinstance(section, dict):
                raise TypeError(f"freq_guidance.{name} must be a mapping")

        self.sc_enabled = bool(config.get("enabled", False))
        self.sc_semantic_enabled = bool(semantic.get("enabled", False))
        self.sc_semantic_mode = str(semantic.get("mode", "fixed_logit"))
        self.sc_lambda34 = float(semantic.get("lambda34", 0.5))
        self.sc_lambda23 = float(semantic.get("lambda23", 0.3))
        self.sc_lambda24 = float(semantic.get("lambda24", 0.2))
        self.sc_detach_anchor = bool(semantic.get("detach_semantic_anchor", True))
        self.sc_semantic_gate = str(semantic.get("semantic_calib_gate", "none")).lower()
        self.sc_semantic_gate_thr = float(semantic.get("semantic_calib_gate_thr", 0.5))
        self.sc_semantic_gate_gamma = float(semantic.get("semantic_calib_gate_gamma", 1.0))
        self.sc_alpha_max = self._sc_pair("alpha_max", semantic.get("alpha_max", (0.75, 0.75)))
        alpha_init = self._sc_pair("alpha_init", semantic.get("alpha_init", (0.0, 0.0)))
        with torch.no_grad():
            self.sc_alpha2.fill_(alpha_init[0])
            self.sc_alpha3.fill_(alpha_init[1])
        self.sc_query_mode = str(query.get("mode", "global_topk"))
        self.sc_centered_mask = bool(query.get("centered_mask", True))
        self.sc_mask_center_mode = str(query.get("mask_center_mode", "absolute")).lower()
        self.sc_local_relative_kernels = self._sc_triplet(
            "local_relative_kernels", query.get("local_relative_kernels", (7, 5, 3)), cast=int
        )
        self.sc_local_relative_std_floor = float(query.get("local_relative_std_floor", 0.05))
        self.sc_local_relative_temperature = float(query.get("local_relative_temperature", 1.0))
        self.sc_level_betas = self._sc_triplet("level_betas", query.get("level_betas", (0.25, 0.5, 0.75)))
        self.sc_boost_min = float(query.get("boost_min", 0.5))
        self.sc_boost_max = float(query.get("boost_max", 1.5))
        self.sc_peak_kernel = int(query.get("peak_kernel", 3))
        self.sc_non_peak_weight = float(query.get("non_peak_weight", 0.6))
        self.sc_base_quotas = self._sc_quota_triplet("base_quotas", query.get("base_quotas", (0, 0, 0)))
        self.sc_max_quotas = self._sc_quota_triplet("max_quotas", query.get("max_quotas", (0, 0, 0)))
        default_fallback = (self.num_queries, 0, 0)
        self.sc_fallback_quotas = self._sc_quota_triplet(
            "fallback_quotas", query.get("fallback_quotas", default_fallback)
        )
        self.sc_extra_queries = self._sc_nonnegative_int(
            "extra_queries", query.get("extra_queries", self.num_queries)
        )
        self.sc_evidence_topr = int(query.get("evidence_topr", 256))
        self.sc_temperature = float(query.get("temperature", 0.7))
        self.sc_evidence_eps = float(query.get("evidence_eps", 1e-4))
        self.sc_min_quotas = self._sc_quota_triplet("min_quotas", query.get("min_quotas", (0, 0, 0)))
        self.sc_prior = self._sc_triplet("prior", query.get("prior", (0.45, 0.32, 0.23)))
        self.sc_evidence_weights = self._sc_quad(
            "evidence_weights", query.get("evidence_weights", (1.0, 0.5, 0.75, 0.5))
        )
        self.sc_active_threshold_std = float(query.get("active_threshold_std", 0.5))
        self.sc_capacity_multiplier = float(query.get("capacity_multiplier", 2.0))
        self.sc_dynamic_ratio_start = float(query.get("dynamic_ratio_start", 0.2))
        self.sc_dynamic_ratio_mid = float(query.get("dynamic_ratio_mid", 0.5))
        self.sc_dynamic_ratio_final = float(query.get("dynamic_ratio_final", 0.8))
        self.sc_dynamic_ratio_mid_epoch = int(query.get("dynamic_ratio_mid_epoch", 10))
        self.sc_dynamic_ratio_final_epoch = int(query.get("dynamic_ratio_final_epoch", 30))
        self.sc_temperature_start = float(query.get("temperature_start", 1.0))
        self.sc_temperature_final = float(query.get("temperature_final", 0.5))
        self.sc_temperature_final_epoch = int(query.get("temperature_final_epoch", 30))
        self.sc_acsa_betas = self._sc_triplet("acsa_betas", config.get("acsa_betas", (1.0, 1.0, 1.0)))
        self.sc_mask_use_calibrated = bool(mask_loss.get("use_calibrated_masks", False))
        self.sc_mask_scale_weights = self._sc_triplet(
            "scale_weights", mask_loss.get("scale_weights", (1.0, 0.6, 0.4))
        )
        self.sc_diagnostics_enabled = bool(diagnostics.get("enabled", False))
        self.sc_store_last_batch = bool(diagnostics.get("store_last_batch", False))
        self.sc_fallback_to_global = bool(diagnostics.get("fallback_to_global_topk", True))
        self.sc_strict_train_masks = bool(diagnostics.get("strict_train_masks", True))
        self.log_metric_keys = SC_SBFG_LOG_METRIC_KEYS
        self.last_query_stats = {}
        self.last_query_debug = {}
        self._sc_calibration_stats_ready = False
        if not hasattr(self, "_sc_epoch"):
            self._sc_epoch = 0
            self._sc_total_epochs = 1
            self._sc_has_train_progress = False

        finite = (self.sc_lambda34, self.sc_lambda23, self.sc_lambda24,
                  self.sc_semantic_gate_thr, self.sc_semantic_gate_gamma,
                  self.sc_local_relative_std_floor, self.sc_local_relative_temperature, self.sc_boost_min,
                  self.sc_boost_max, self.sc_non_peak_weight, self.sc_temperature, self.sc_evidence_eps,
                  *self.sc_mask_scale_weights, *self.sc_acsa_betas)
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("SC-SBFG scalar configuration must be finite")
        if self.sc_query_mode not in {"global_topk", "dynamic_scale_balanced", "scene_conditioned_dynamic"}:
            raise ValueError(f"unsupported query.mode={self.sc_query_mode!r}")
        if self.sc_semantic_mode not in {"fixed_logit", "residual_logit"}:
            raise ValueError(f"unsupported semantic_calibration.mode={self.sc_semantic_mode!r}")
        if self.sc_semantic_gate not in {"none", "positive_conf"}:
            raise ValueError(f"unsupported semantic_calib_gate={self.sc_semantic_gate!r}")
        if not 0.0 <= self.sc_semantic_gate_thr < 1.0:
            raise ValueError("semantic_calib_gate_thr must be within [0, 1)")
        if self.sc_semantic_gate_gamma <= 0.0:
            raise ValueError("semantic_calib_gate_gamma must be positive")
        if self.sc_mask_center_mode not in {"absolute", "local_relative"}:
            raise ValueError(f"unsupported mask_center_mode={self.sc_mask_center_mode!r}")
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in self.sc_local_relative_kernels):
            raise ValueError("local_relative_kernels must contain three positive odd integers")
        if self.sc_local_relative_std_floor <= 0.0 or self.sc_local_relative_temperature <= 0.0:
            raise ValueError("local-relative std floor and temperature must be positive")
        if self.sc_enabled:
            if self.sc_query_mode == "scene_conditioned_dynamic":
                if sum(self.sc_min_quotas) >= self.num_queries:
                    raise ValueError("sum(min_quotas) must be smaller than num_queries")
                if any(cap < floor for floor, cap in zip(self.sc_min_quotas, self.sc_max_quotas)):
                    raise ValueError("each max_quota must be at least its min_quota")
                if sum(self.sc_max_quotas) < self.num_queries:
                    raise ValueError("sum(max_quotas) must cover num_queries")
                if any(value <= 0.0 for value in self.sc_prior) or not math.isclose(sum(self.sc_prior), 1.0, abs_tol=1e-6):
                    raise ValueError("prior must contain positive values summing to 1")
                if not 0.0 <= self.sc_dynamic_ratio_start <= self.sc_dynamic_ratio_mid <= self.sc_dynamic_ratio_final <= 1.0:
                    raise ValueError("dynamic ratio schedule must be monotonic within [0,1]")
                if not 0 <= self.sc_dynamic_ratio_mid_epoch <= self.sc_dynamic_ratio_final_epoch:
                    raise ValueError("dynamic ratio epochs must be monotonic and nonnegative")
                if self.sc_temperature_start <= 0.0 or self.sc_temperature_final <= 0.0:
                    raise ValueError("SC-v2 temperatures must be positive")
                if self.sc_temperature_final_epoch < 0:
                    raise ValueError("temperature_final_epoch must be nonnegative")
                if self.sc_active_threshold_std < 0.0 or self.sc_capacity_multiplier <= 0.0:
                    raise ValueError("active_threshold_std must be nonnegative and capacity_multiplier positive")
                if any(value <= 0.0 for value in self.sc_alpha_max):
                    raise ValueError("alpha_max values must be positive")
            else:
                if sum(self.sc_base_quotas) + self.sc_extra_queries != self.num_queries:
                    raise ValueError("sum(base_quotas) + extra_queries must equal num_queries")
                if any(cap < base for base, cap in zip(self.sc_base_quotas, self.sc_max_quotas)):
                    raise ValueError("each max_quota must be at least its base_quota")
            if sum(self.sc_fallback_quotas) != self.num_queries:
                raise ValueError("sum(fallback_quotas) must equal num_queries")
            if self.sc_peak_kernel != 3:
                raise ValueError("peak_kernel must be 3 in SC-SBFG-v1")
            if not self.sc_centered_mask:
                raise ValueError("centered_mask must be true in SC-SBFG-v1")
            if not 0.0 < self.sc_non_peak_weight <= 1.0:
                raise ValueError("non_peak_weight must be within (0, 1]")
            if self.sc_temperature <= 0.0:
                raise ValueError("temperature must be positive")
            if self.sc_evidence_topr <= 0 or self.sc_evidence_eps < 0.0:
                raise ValueError("evidence_topr must be positive and evidence_eps nonnegative")
            if self.sc_boost_min <= 0.0 or self.sc_boost_max < self.sc_boost_min:
                raise ValueError("boost bounds are invalid")
            if any(weight < 0.0 for weight in self.sc_mask_scale_weights) or sum(self.sc_mask_scale_weights) <= 0.0:
                raise ValueError("scale_weights must be nonnegative with a positive sum")

    def freq_guidance_summary(self) -> str:
        """Return the effective zero-parameter frequency-guidance configuration for build-time verification."""
        return (
            f"enabled={int(self.sc_enabled)}, semantic={self.sc_semantic_mode}, "
            f"cgsc={self.sc_semantic_gate}(thr={self.sc_semantic_gate_thr:g},gamma={self.sc_semantic_gate_gamma:g}), "
            f"query={self.sc_query_mode}, center={self.sc_mask_center_mode}, "
            f"lrm_kernels={list(self.sc_local_relative_kernels)}, "
            f"lrm_std_floor={self.sc_local_relative_std_floor:g}, "
            f"lrm_temperature={self.sc_local_relative_temperature:g}, "
            f"queries={self.num_queries}, decoder_layers={self.num_decoder_layers}"
        )

    @staticmethod
    def _assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
        """Raise at the first non-finite SC/decoder tensor without altering model values."""
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

    def _positive_semantic_gate(self, logits: torch.Tensor) -> torch.Tensor:
        """Return a zero-parameter positive-support confidence gate in FP32."""
        probability = logits.float().sigmoid()
        denominator = max(1.0 - self.sc_semantic_gate_thr, 1e-6)
        gate = ((probability - self.sc_semantic_gate_thr) / denominator).clamp(0.0, 1.0)
        if self.sc_semantic_gate_gamma != 1.0:
            gate = gate.pow(self.sc_semantic_gate_gamma)
        self._assert_finite_tensor("cgsc_positive_support_gate", gate)
        return gate.to(dtype=logits.dtype)

    def _calibration_support(self, semantic_anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply optional CGSC while preserving the exact V2 tensor when disabled."""
        if self.sc_semantic_gate == "none":
            return semantic_anchor, None
        gate = self._positive_semantic_gate(semantic_anchor)
        return gate * semantic_anchor, gate

    def _local_relative_mask(self, mask_logits: torch.Tensor, kernel: int) -> torch.Tensor:
        """Compute local relative mask evidence in FP32 and return it in the input dtype."""
        mask_prob = mask_logits.float().sigmoid()
        padding = kernel // 2
        padded = F.pad(mask_prob, (padding, padding, padding, padding), mode="replicate")
        local_mean = F.avg_pool2d(padded, kernel_size=kernel, stride=1)
        local_second_moment = F.avg_pool2d(
            padded.square(), kernel_size=kernel, stride=1
        )
        local_variance = (local_second_moment - local_mean.square()).clamp_min(0.0)
        local_std = local_variance.sqrt().clamp_min(self.sc_local_relative_std_floor)
        relative = torch.tanh(
            (mask_prob - local_mean) / (local_std * self.sc_local_relative_temperature)
        )
        self._assert_finite_tensor("lrm_relative_mask", relative)
        return relative.to(dtype=mask_logits.dtype)

    def calibrate_freq_masks(self, raw_masks: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        if set(raw_masks) != {0, 1, 2}:
            raise ValueError(f"SC-SBFG requires raw mask keys {{0,1,2}}, got {sorted(raw_masks)}")
        for level, tensor in raw_masks.items():
            if tensor.ndim != 4 or tensor.shape[1] != 1:
                raise ValueError(f"frequency mask {level} must be [B,1,H,W], got {tuple(tensor.shape)}")
            self._assert_finite_tensor(f"raw_freq_mask_p{level + 2}", tensor)
        if not self.sc_enabled or not self.sc_semantic_enabled:
            active = dict(raw_masks)
        else:
            l2, l3, l4 = raw_masks[0], raw_masks[1], raw_masks[2]
            anchor4 = l4.detach() if self.sc_detach_anchor else l4
            if self.sc_semantic_mode == "residual_logit":
                alpha2 = self.sc_alpha_max[0] * torch.tanh(self.sc_alpha2)
                alpha3 = self.sc_alpha_max[1] * torch.tanh(self.sc_alpha3)
                semantic43 = F.interpolate(anchor4, size=l3.shape[-2:], mode="bilinear", align_corners=False)
                support43, gate43 = self._calibration_support(semantic43)
                l3_cal = l3 + alpha3 * support43
                anchor3 = l3_cal.detach() if self.sc_detach_anchor else l3_cal
                semantic32 = F.interpolate(anchor3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
                support32, gate32 = self._calibration_support(semantic32)
                l2_cal = l2 + alpha2 * support32
            else:
                l3_cal = l3 + self.sc_lambda34 * F.interpolate(
                    anchor4, size=l3.shape[-2:], mode="bilinear", align_corners=False
                )
                anchor3 = l3_cal.detach() if self.sc_detach_anchor else l3_cal
                l2_cal = l2 + self.sc_lambda23 * F.interpolate(
                    anchor3, size=l2.shape[-2:], mode="bilinear", align_corners=False
                ) + self.sc_lambda24 * F.interpolate(
                    anchor4, size=l2.shape[-2:], mode="bilinear", align_corners=False
                )
            active = {0: l2_cal, 1: l3_cal, 2: l4}
        for level, tensor in active.items():
            self._assert_finite_tensor(f"active_freq_mask_p{level + 2}", tensor)
        if self.sc_diagnostics_enabled:
            self.last_query_stats = {
                **{f"metric_sc_mask_raw_mean_p{level + 2}": torch.sigmoid(raw_masks[level].detach()).mean()
                   for level in range(3)},
                **{f"metric_sc_mask_cal_mean_p{level + 2}": torch.sigmoid(active[level].detach()).mean()
                   for level in range(3)},
            }
            if (self.sc_enabled and self.sc_semantic_enabled and self.sc_semantic_mode == "residual_logit"
                    and self.sc_semantic_gate == "positive_conf"):
                self.last_query_stats.update({
                    "metric_cgsc_g43_mean": gate43.detach().float().mean(),
                    "metric_cgsc_g43_active": (gate43.detach() > 0).float().mean(),
                    "metric_cgsc_g32_mean": gate32.detach().float().mean(),
                    "metric_cgsc_g32_active": (gate32.detach() > 0).float().mean(),
                })
        self._sc_calibration_stats_ready = self.sc_diagnostics_enabled
        return active

    def set_train_progress(self, epoch: int, total_epochs: int) -> None:
        """Synchronize the scene-conditioned allocation schedule across DDP ranks."""
        epoch, total_epochs = int(epoch), int(total_epochs)
        if epoch < 0 or total_epochs < 1:
            raise ValueError(f"invalid SC-v2 progress: epoch={epoch}, total_epochs={total_epochs}")
        self._sc_epoch = epoch
        self._sc_total_epochs = total_epochs
        self._sc_has_train_progress = True

    def _sc_schedule(self) -> tuple[float, float]:
        """Return current dynamic-prior mix and softmax temperature."""
        if not self.training and not self._sc_has_train_progress:
            return self.sc_dynamic_ratio_final, self.sc_temperature_final
        epoch = max(int(self._sc_epoch), 0)
        mid_epoch = self.sc_dynamic_ratio_mid_epoch
        final_epoch = self.sc_dynamic_ratio_final_epoch
        if epoch <= mid_epoch:
            fraction = 1.0 if mid_epoch == 0 else epoch / mid_epoch
            ratio = self.sc_dynamic_ratio_start + fraction * (
                self.sc_dynamic_ratio_mid - self.sc_dynamic_ratio_start
            )
        elif epoch < final_epoch:
            span = max(final_epoch - mid_epoch, 1)
            fraction = (epoch - mid_epoch) / span
            ratio = self.sc_dynamic_ratio_mid + fraction * (
                self.sc_dynamic_ratio_final - self.sc_dynamic_ratio_mid
            )
        else:
            ratio = self.sc_dynamic_ratio_final
        temperature_fraction = min(epoch / max(self.sc_temperature_final_epoch, 1), 1.0)
        temperature = self.sc_temperature_start + temperature_fraction * (
            self.sc_temperature_final - self.sc_temperature_start
        )
        return float(ratio), float(temperature)

    def _standardize_sc_scores(self, score: torch.Tensor) -> torch.Tensor:
        """Standardize three scale scores independently for every image."""
        if score.ndim != 2 or score.shape[1] != 3:
            raise ValueError(f"SC-v2 score must be [B,3], got {tuple(score.shape)}")
        centered = score - score.mean(dim=1, keepdim=True)
        scale = score.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.sc_evidence_eps)
        return centered / scale

    def forward(
        self,
        x: list[torch.Tensor],
        batch: dict | None = None,
        freq_masks: dict | None = None,
    ) -> tuple | torch.Tensor:
        from ultralytics.models.utils.ops import get_cdn_group

        feats, shapes = self._get_encoder_input(x)

        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(
            feats=feats,
            shapes=shapes,
            dn_embed=dn_embed,
            dn_bbox=dn_bbox,
            freq_masks_dict=freq_masks,
        )

        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )

        x_out = (
            dec_bboxes,
            dec_scores,
            enc_bboxes,
            enc_scores,
            dn_meta,
        )

        if self.training:
            return x_out

        final_bboxes = (
            dec_bboxes[-1]
            if dec_bboxes.ndim == 4
            else dec_bboxes
        )
        final_scores = (
            dec_scores[-1]
            if dec_scores.ndim == 4
            else dec_scores
        )
        y = torch.cat(
            (final_bboxes, final_scores.sigmoid()),
            dim=-1,
        )
        return y if self.export else (y, x_out)

    @staticmethod
    def _generate_anchors(
        shapes: list[list[int]],
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        eps: float = 1e-2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchors = []

        for level_idx, (h, w) in enumerate(shapes):
            sy = torch.arange(h, dtype=dtype, device=device)
            sx = torch.arange(w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(
                sy,
                sx,
                indexing="ij",
            )
            grid_xy = torch.stack(
                [grid_x, grid_y],
                dim=-1,
            )

            valid_wh = torch.tensor(
                [w, h],
                dtype=dtype,
                device=device,
            )
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_wh

            wh = torch.ones_like(grid_xy) * (
                grid_size * (2.0 ** level_idx)
            )
            level_anchor = torch.cat(
                [grid_xy, wh],
                dim=-1,
            ).reshape(1, h * w, 4)
            anchors.append(level_anchor)

        anchors = torch.cat(anchors, dim=1)
        valid_mask = (
            (anchors > eps)
            & (anchors < 1.0 - eps)
        ).all(dim=-1, keepdim=True)

        anchors = torch.log(
            anchors / (1.0 - anchors)
        )
        anchors = anchors.masked_fill(
            ~valid_mask,
            float("inf"),
        )
        return anchors, valid_mask

    def _get_encoder_input(
        self,
        x: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[list[int]]]:
        projected = [
            self.input_proj[i](feat)
            for i, feat in enumerate(x)
        ]

        feats = []
        shapes = []
        for feat in projected:
            h, w = feat.shape[-2:]
            feats.append(
                feat.flatten(2).permute(0, 2, 1)
            )
            shapes.append([h, w])

        return torch.cat(feats, dim=1), shapes

    @staticmethod
    def _align_freq_mask(
        mask_logits: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        target_h, target_w = target_hw
        mask_h, mask_w = mask_logits.shape[-2:]

        if (mask_h, mask_w) == (target_h, target_w):
            return mask_logits

        if mask_h >= target_h and mask_w >= target_w:
            return F.adaptive_avg_pool2d(
                mask_logits,
                output_size=(target_h, target_w),
            )

        return F.interpolate(
            mask_logits,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )

    def _legacy_query_priority(
        self,
        enc_outputs_scores: torch.Tensor,
        shapes: list[list[int]],
        freq_masks_dict: dict | None,
    ) -> torch.Tensor:
        """Return the pre-SC-SBFG ranking priority without changing its arithmetic."""
        priority = enc_outputs_scores.sigmoid().amax(dim=-1)
        has_all = freq_masks_dict is not None and all(level in freq_masks_dict for level in range(len(shapes)))
        if not has_all:
            return priority
        tokens = []
        for level, (h, w) in enumerate(shapes):
            mask = freq_masks_dict[level]
            if mask.ndim != 4 or mask.shape[1] != 1 or mask.shape[0] != priority.shape[0]:
                raise ValueError(f"scale {level} frequency mask must be [B,1,H,W]")
            aligned = self._align_freq_mask(mask, (h, w)).to(dtype=priority.dtype)
            tokens.append(torch.sigmoid(aligned).flatten(2).transpose(1, 2))
        joined = torch.cat(tokens, dim=1).squeeze(-1)
        if joined.shape[1] != priority.shape[1]:
            raise RuntimeError("frequency-mask token count does not match encoder-token count")
        return priority * (1.0 + self.freq_query_beta * joined)

    def _soft_peak_suppress(self, priority: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Downweight non-peak tokens while retaining every candidate."""
        grid = priority.reshape(priority.shape[0], 1, h, w)
        local_max = F.max_pool2d(grid, kernel_size=3, stride=1, padding=1)
        peaks = grid >= (local_max - 1e-6)
        weights = torch.where(peaks, torch.ones_like(grid), torch.full_like(grid, self.sc_non_peak_weight))
        return (grid * weights).flatten(1), peaks.flatten(1)

    @staticmethod
    def _largest_remainder(total: int, weights: torch.Tensor) -> torch.Tensor:
        raw = weights * int(total)
        result = torch.floor(raw).to(torch.long)
        missing = int(total) - int(result.sum().item())
        if missing:
            order = torch.argsort(raw - result.to(raw.dtype), descending=True, stable=True)
            result[order[:missing]] += 1
        return result

    def _fit_quota(self, proposal: torch.Tensor, limits: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        """Fit a three-level proposal to capacities with deterministic evidence-order redistribution."""
        quota = torch.minimum(proposal.to(torch.long), limits.to(torch.long))
        deficit = self.num_queries - int(quota.sum().item())
        order = torch.argsort(torch.nan_to_num(evidence, nan=-float("inf")), descending=True, stable=True)
        while deficit > 0:
            progressed = False
            for level in order.tolist():
                room = int(limits[level].item() - quota[level].item())
                if room > 0:
                    add = min(room, deficit)
                    quota[level] += add
                    deficit -= add
                    progressed = True
                if deficit == 0:
                    break
            if not progressed:
                raise ValueError("SC-SBFG cannot allocate num_queries within valid-token capacities")
        return quota

    def _allocate_sc_quotas(
        self,
        evidence: torch.Tensor,
        valid_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate dynamic, fixed-fallback, or global-fallback quotas for each image."""
        quotas = torch.zeros_like(valid_counts, dtype=torch.long)
        modes = torch.zeros(evidence.shape[0], device=evidence.device, dtype=torch.long)
        for batch_index in range(evidence.shape[0]):
            ev = evidence[batch_index].float()
            if not torch.isfinite(ev).all():
                modes[batch_index] = 2
                continue
            limits = torch.minimum(
                torch.tensor(self.sc_max_quotas, device=ev.device, dtype=torch.long),
                valid_counts[batch_index].to(torch.long),
            )
            if int(limits.sum().item()) < self.num_queries:
                raise ValueError("fewer than num_queries valid encoder tokens within SC-SBFG quota caps")
            if ev.max() <= self.sc_evidence_eps or (ev.max() - ev.min()) <= self.sc_evidence_eps:
                proposal = torch.tensor(self.sc_fallback_quotas, device=ev.device, dtype=torch.long)
                modes[batch_index] = 1
            else:
                base = torch.tensor(self.sc_base_quotas, device=ev.device, dtype=torch.long)
                weights = torch.softmax(ev / self.sc_temperature, dim=0)
                proposal = base + self._largest_remainder(self.sc_extra_queries, weights)
            quotas[batch_index] = self._fit_quota(proposal, limits, ev)
        return quotas, modes

    def _allocate_sc_v2_quotas(
        self,
        score: torch.Tensor,
        valid_counts: torch.Tensor,
        capacities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Allocate per-image SC-v2 quotas with weak priors and capacity-aware redistribution."""
        if score.shape != valid_counts.shape or score.shape != capacities.shape or score.shape[-1] != 3:
            raise ValueError("SC-v2 score, valid_counts and capacities must share shape [B,3]")
        self._assert_finite_tensor("sc_v2_score", score)
        batch_size = score.shape[0]
        quotas = torch.zeros_like(valid_counts, dtype=torch.long)
        modes = torch.zeros(batch_size, device=score.device, dtype=torch.long)
        p_dynamic_rows, p_query_rows = [], []
        floor_hits, cap_hits, capacity_relaxed = [], [], []
        dynamic_ratio, temperature = self._sc_schedule()
        prior = score.new_tensor(self.sc_prior)
        minimum = torch.tensor(self.sc_min_quotas, device=score.device, dtype=torch.long)
        static_max = torch.tensor(self.sc_max_quotas, device=score.device, dtype=torch.long)
        dynamic_budget = self.num_queries - int(minimum.sum().item())

        for batch_index in range(batch_size):
            row = score[batch_index].float()
            valid_limit = torch.minimum(static_max, valid_counts[batch_index].to(torch.long))
            strict_limit = torch.minimum(valid_limit, capacities[batch_index].to(torch.long))
            relaxed = int(strict_limit.sum().item()) < self.num_queries
            limits = valid_limit if relaxed else strict_limit
            if int(limits.sum().item()) < self.num_queries:
                modes[batch_index] = 2
                quotas[batch_index].zero_()
                p_dynamic_rows.append(prior)
                p_query_rows.append(prior)
                floor_hits.append(row.new_zeros(()))
                cap_hits.append(row.new_zeros(()))
                capacity_relaxed.append(row.new_ones(()))
                continue
            if (row.max() - row.min()) <= self.sc_evidence_eps:
                proposal = torch.tensor(self.sc_fallback_quotas, device=row.device, dtype=torch.long)
                modes[batch_index] = 1
                p_dynamic = prior
            else:
                z = self._standardize_sc_scores(row.unsqueeze(0))[0]
                p_dynamic = torch.softmax(z / temperature, dim=0)
                p_blend = (1.0 - dynamic_ratio) * prior + dynamic_ratio * p_dynamic
                proposal = minimum + self._largest_remainder(dynamic_budget, p_blend)
            quota = self._fit_quota(proposal, limits, p_dynamic)
            quotas[batch_index] = quota
            p_query = quota.to(row.dtype) / float(self.num_queries)
            p_dynamic_rows.append(p_dynamic)
            p_query_rows.append(p_query)
            floor_hits.append((quota == minimum).float().mean())
            cap_hits.append((quota == limits).float().mean())
            capacity_relaxed.append(row.new_tensor(float(relaxed)))

        stats = {
            "p_dynamic": torch.stack(p_dynamic_rows),
            "p_query": torch.stack(p_query_rows),
            "floor_hit": torch.stack(floor_hits),
            "cap_hit": torch.stack(cap_hits),
            "capacity_relaxed": torch.stack(capacity_relaxed),
            "dynamic_ratio": score.new_tensor(dynamic_ratio),
            "temperature": score.new_tensor(temperature),
        }
        return quotas, modes, stats

    def _build_sc_v2_evidence(
        self,
        weighted_levels: list[torch.Tensor],
        mask_prob_levels: list[torch.Tensor],
        valid_level_masks: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Build scale-relative evidence and candidate capacities for every image."""
        top_means, effective_counts, peak_contrasts, active_ratios, capacities = [], [], [], [], []
        for level, (weighted, mask_prob, valid) in enumerate(
            zip(weighted_levels, mask_prob_levels, valid_level_masks)
        ):
            self._assert_finite_tensor(f"sc_weighted_priority_p{level + 2}", weighted.masked_select(valid))
            self._assert_finite_tensor(f"sc_mask_probability_p{level + 2}", mask_prob.masked_select(valid))
            valid_count = valid.sum(dim=1).clamp_min(1)
            quality = torch.where(valid, weighted.clamp_min(0.0), torch.zeros_like(weighted))
            top_mean_rows = []
            for batch_index in range(weighted.shape[0]):
                top_r = min(self.sc_evidence_topr, int(valid_count[batch_index].item()))
                top_mean_rows.append(torch.topk(quality[batch_index], top_r).values.mean())
            top_mean = torch.stack(top_mean_rows)
            quality_sum = quality.sum(dim=1)
            effective_count = quality_sum.square() / quality.square().sum(dim=1).clamp_min(self.sc_evidence_eps)
            global_mean = quality_sum / valid_count
            peak_contrast = top_mean / global_mean.clamp_min(self.sc_evidence_eps)

            mask_mean = (mask_prob * valid).sum(dim=1) / valid_count
            centered = (mask_prob - mask_mean[:, None]) * valid
            mask_std = torch.sqrt(centered.square().sum(dim=1) / valid_count)
            threshold = mask_mean + self.sc_active_threshold_std * mask_std
            active_count = ((mask_prob > threshold[:, None]) & valid).sum(dim=1).to(torch.float32)
            active_ratio = active_count / valid_count
            self._assert_finite_tensor(f"sc_top_mean_p{level + 2}", top_mean)
            self._assert_finite_tensor(f"sc_effective_count_p{level + 2}", effective_count)
            self._assert_finite_tensor(f"sc_peak_contrast_p{level + 2}", peak_contrast)
            self._assert_finite_tensor(f"sc_active_ratio_p{level + 2}", active_ratio)
            capacity = torch.ceil(
                torch.maximum(effective_count, active_count) * self.sc_capacity_multiplier
            ).to(torch.long)

            top_means.append(top_mean)
            effective_counts.append(effective_count)
            peak_contrasts.append(peak_contrast)
            active_ratios.append(active_ratio)
            capacities.append(capacity)

        top_mean = torch.stack(top_means, dim=1)
        effective_count = torch.stack(effective_counts, dim=1)
        peak_contrast = torch.stack(peak_contrasts, dim=1)
        active_ratio = torch.stack(active_ratios, dim=1)
        capacity = torch.stack(capacities, dim=1)
        w_top, w_count, w_peak, w_active = self.sc_evidence_weights
        score = (
            w_top * torch.log(top_mean.clamp_min(self.sc_evidence_eps))
            + w_count * torch.log1p(effective_count)
            + w_peak * torch.log(peak_contrast.clamp_min(self.sc_evidence_eps))
            + w_active * torch.log(active_ratio.clamp_min(self.sc_evidence_eps))
        )
        self._assert_finite_tensor("sc_v2_score", score)
        return {
            "score": score,
            "top_mean": top_mean,
            "effective_count": effective_count,
            "peak_contrast": peak_contrast,
            "active_ratio": active_ratio,
            "capacity": capacity,
        }

    def _sc_valid_tokens(self, batch_size: int, num_tokens: int, device: torch.device) -> torch.Tensor:
        """Normalize the runtime anchor validity mask to one boolean row per batch item."""
        if self.valid_mask.numel() == num_tokens:
            valid = self.valid_mask.reshape(-1, num_tokens).to(device=device, dtype=torch.bool)
            if valid.shape[0] == 1:
                return valid.expand(batch_size, -1)
            if valid.shape[0] == batch_size:
                return valid
        return torch.ones(batch_size, num_tokens, device=device, dtype=torch.bool)

    def _consume_calibration_stats(self, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        """Consume only immediately preceding calibration means, preventing prior-batch stat leakage."""
        if not self._sc_calibration_stats_ready:
            return {}
        self._sc_calibration_stats_ready = False
        keys = tuple(
            f"metric_sc_mask_{kind}_mean_p{level + 2}" for kind in ("raw", "cal") for level in range(3)
        ) + (
            "metric_cgsc_g43_mean", "metric_cgsc_g43_active",
            "metric_cgsc_g32_mean", "metric_cgsc_g32_active",
        )
        return {
            key: self.last_query_stats[key].detach().to(device=reference.device)
            for key in keys
            if key in self.last_query_stats and self.last_query_stats[key].ndim == 0
        }

    def _record_global_fallback_stats(
        self,
        reference: torch.Tensor,
        batch_size: int,
        calibration_stats: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Refresh enabled-selection diagnostics for an exact legacy global Top-K fallback."""
        zero = reference.new_zeros((), dtype=torch.float32).detach()
        self.last_query_stats = {key: zero.clone() for key in self.log_metric_keys}
        if calibration_stats:
            self.last_query_stats.update(calibration_stats)
        self.last_query_stats["metric_sc_global_fallback_rate"] = zero.new_ones(())
        self.last_query_stats["metric_sc_fallback_count"] = zero.new_tensor(float(batch_size))
        self.last_query_debug = {}

    def _select_query_indices(
        self,
        enc_outputs_scores: torch.Tensor,
        shapes: list[list[int]],
        freq_masks_dict: dict | None,
    ) -> torch.Tensor:
        """Select legacy global Top-K or SC-SBFG scale-balanced query indices."""
        batch_size, num_tokens, _ = enc_outputs_scores.shape
        if self.num_queries > num_tokens:
            raise ValueError(f"num_queries={self.num_queries} exceeds available token count {num_tokens}")
        if freq_masks_dict is not None:
            for level, tensor in freq_masks_dict.items():
                self._assert_finite_tensor(f"query_freq_mask_p{int(level) + 2}", tensor)
        legacy_priority = self._legacy_query_priority(enc_outputs_scores, shapes, freq_masks_dict)
        if not self.sc_enabled:
            return torch.topk(legacy_priority, self.num_queries, dim=1).indices
        if self.sc_query_mode == "global_topk":
            self._record_global_fallback_stats(
                enc_outputs_scores, batch_size, self._consume_calibration_stats(enc_outputs_scores)
            )
            return torch.topk(legacy_priority, self.num_queries, dim=1).indices

        calibration_stats = self._consume_calibration_stats(enc_outputs_scores)
        self.last_query_stats = {}
        self.last_query_debug = {}

        has_all_masks = freq_masks_dict is not None and all(level in freq_masks_dict for level in range(3))
        if not has_all_masks:
            if self.training and self.sc_strict_train_masks:
                raise ValueError("SC-SBFG dynamic selection requires P2/P3/P4 frequency masks during training")
            if self.sc_fallback_to_global:
                self._record_global_fallback_stats(enc_outputs_scores, batch_size, calibration_stats)
                return torch.topk(legacy_priority, self.num_queries, dim=1).indices
            raise ValueError("SC-SBFG dynamic selection requires P2/P3/P4 frequency masks")
        if len(shapes) != 3:
            raise ValueError(f"SC-SBFG v1 requires exactly three feature levels, got {len(shapes)}")

        valid_tokens = self._sc_valid_tokens(batch_size, num_tokens, enc_outputs_scores.device)
        if int(valid_tokens.sum(dim=1).min().item()) < self.num_queries:
            raise ValueError("fewer than num_queries valid encoder tokens")
        level_sizes = [int(h) * int(w) for h, w in shapes]
        if sum(level_sizes) != num_tokens:
            raise ValueError("runtime feature shapes do not match encoder-token count")
        starts = [0, level_sizes[0], level_sizes[0] + level_sizes[1]]

        with torch.no_grad():
            weighted_levels, peak_levels, mask_means = [], [], []
            mask_prob_levels, valid_level_masks = [], []
            lrm_relative_levels = []
            debug_priority_before = [] if self.sc_store_last_batch else None
            for level, ((h, w), start, size) in enumerate(zip(shapes, starts, level_sizes)):
                mask = freq_masks_dict[level]
                if mask.ndim != 4 or mask.shape[1] != 1 or mask.shape[0] != batch_size:
                    raise ValueError(f"scale {level} frequency mask must be [B,1,H,W]")
                aligned_mask = self._align_freq_mask(mask, (h, w))
                mask_prob = aligned_mask.float().sigmoid().flatten(2).squeeze(1)
                if self.sc_mask_center_mode == "absolute":
                    centered_mask = 2.0 * mask_prob - 1.0
                    relative_mask = None
                else:
                    relative_map = self._local_relative_mask(
                        aligned_mask, self.sc_local_relative_kernels[level]
                    ).float()
                    centered_mask = relative_map.flatten(2).squeeze(1)
                    relative_mask = centered_mask
                boost = (1.0 + self.sc_level_betas[level] * centered_mask).clamp(
                    self.sc_boost_min, self.sc_boost_max
                )
                priority = enc_outputs_scores[:, start : start + size].float().sigmoid().amax(dim=-1) * boost
                priority = priority.masked_fill(~valid_tokens[:, start : start + size], float("-inf"))
                if debug_priority_before is not None:
                    debug_priority_before.append(
                        enc_outputs_scores[:, start : start + size]
                        .float()
                        .sigmoid()
                        .amax(dim=-1)
                        .masked_fill(~valid_tokens[:, start : start + size], float("-inf"))
                        .detach()
                        .cpu()
                    )
                weighted, peaks = self._soft_peak_suppress(priority, h, w)
                weighted_levels.append(weighted)
                peak_levels.append(peaks & valid_tokens[:, start : start + size])
                mask_means.append(mask_prob.mean(dim=1))
                mask_prob_levels.append(mask_prob)
                valid_level_masks.append(valid_tokens[:, start : start + size])
                lrm_relative_levels.append(relative_mask)

            valid_counts = torch.stack(
                [valid_tokens[:, start : start + size].sum(dim=1) for start, size in zip(starts, level_sizes)], dim=1
            )
            allocation_stats = None
            evidence_details = None
            if self.sc_query_mode == "scene_conditioned_dynamic":
                evidence_details = self._build_sc_v2_evidence(
                    weighted_levels, mask_prob_levels, valid_level_masks
                )
                capacities = torch.maximum(
                    evidence_details["capacity"],
                    torch.tensor(self.sc_min_quotas, device=enc_outputs_scores.device, dtype=torch.long),
                )
                quotas, modes, allocation_stats = self._allocate_sc_v2_quotas(
                    evidence_details["score"], valid_counts, capacities
                )
                evidence = allocation_stats["p_dynamic"]
            else:
                evidence = torch.empty(batch_size, 3, device=enc_outputs_scores.device, dtype=torch.float32)
                for level, weighted in enumerate(weighted_levels):
                    for batch_index in range(batch_size):
                        top_r = min(self.sc_evidence_topr, int(valid_counts[batch_index, level].item()))
                        evidence[batch_index, level] = torch.topk(weighted[batch_index], top_r).values.mean()
                quotas, modes = self._allocate_sc_quotas(evidence, valid_counts)
            level_topk = [
                torch.argsort(weighted, dim=1, descending=True, stable=True)[:, : min(int(cap), size)]
                for weighted, size, cap in zip(weighted_levels, level_sizes, self.sc_max_quotas)
            ]
            legacy_indices = torch.topk(legacy_priority, self.num_queries, dim=1).indices
            selected_rows = []
            for batch_index in range(batch_size):
                if int(modes[batch_index].item()) == 2:
                    selected_rows.append(legacy_indices[batch_index])
                else:
                    selected_rows.append(torch.cat([
                        level_topk[level][batch_index, : int(quotas[batch_index, level].item())] + starts[level]
                        for level in range(3)
                    ]))
            selected = torch.stack(selected_rows, dim=0)
            if selected.shape != (batch_size, self.num_queries):
                raise RuntimeError("SC-SBFG selected query shape does not match [B, num_queries]")
            if any(row.unique().numel() != self.num_queries for row in selected):
                raise RuntimeError("SC-SBFG selected duplicate query indices")

            level_ids = torch.empty(num_tokens, device=selected.device, dtype=torch.long)
            for level, (start, size) in enumerate(zip(starts, level_sizes)):
                level_ids[start : start + size] = level
            selected_levels = level_ids[selected]
            selected_counts = torch.stack([(selected_levels == level).sum(dim=1).float() for level in range(3)], dim=1)
            priority_means = torch.stack([
                torch.stack([
                    (weighted_levels[level][batch_index, selected[batch_index, selected_levels[batch_index] == level] - starts[level]].mean()
                     if (selected_levels[batch_index] == level).any() else weighted_levels[level].new_zeros(()))
                    for batch_index in range(batch_size)
                ]) for level in range(3)
            ], dim=1)
            proportions = selected_counts / float(self.num_queries)
            entropy = -(proportions * proportions.clamp_min(1e-12).log()).sum(dim=1) / math.log(3.0)
            zero = enc_outputs_scores.new_zeros((), dtype=torch.float32).detach()
            query_stats = {key: zero.clone() for key in self.log_metric_keys}
            query_stats.update({
                **{f"metric_sc_selected_p{level + 2}": selected_counts[:, level].mean().detach() for level in range(3)},
                **{f"metric_sc_evidence_p{level + 2}": evidence[:, level].mean().detach() for level in range(3)},
                **{f"metric_sc_mask_raw_mean_p{level + 2}": calibration_stats.get(
                    f"metric_sc_mask_raw_mean_p{level + 2}", mask_means[level].mean()
                ).detach() for level in range(3)},
                **{f"metric_sc_mask_cal_mean_p{level + 2}": calibration_stats.get(
                    f"metric_sc_mask_cal_mean_p{level + 2}", mask_means[level].mean()
                ).detach() for level in range(3)},
                **{f"metric_sc_peak_ratio_p{level + 2}": peak_levels[level].float().sum(dim=1).div(
                    valid_counts[:, level].clamp_min(1)
                ).mean().detach() for level in range(3)},
                "metric_sc_query_entropy": entropy.mean().detach(),
                "metric_sc_fixed_fallback_rate": (modes == 1).float().mean().detach(),
                "metric_sc_global_fallback_rate": (modes == 2).float().mean().detach(),
                "metric_sc_fallback_count": (modes != 0).sum().float().detach(),
                **{f"metric_sc_priority_top_mean_p{level + 2}": priority_means[:, level].mean().detach()
                   for level in range(3)},
                **{f"metric_sc_selected_std_p{level + 2}": selected_counts[:, level].std(unbiased=False).detach()
                   for level in range(3)},
                **{f"metric_sc_selected_min_p{level + 2}": selected_counts[:, level].min().detach()
                   for level in range(3)},
                **{f"metric_sc_selected_max_p{level + 2}": selected_counts[:, level].max().detach()
                   for level in range(3)},
                **{f"metric_sc_selected_p10_p{level + 2}": torch.quantile(selected_counts[:, level], 0.10).detach()
                   for level in range(3)},
                **{f"metric_sc_selected_p90_p{level + 2}": torch.quantile(selected_counts[:, level], 0.90).detach()
                   for level in range(3)},
            })
            # Preserve the immediately preceding CGSC diagnostics across Query selection. The default-zero
            # dictionary above intentionally covers disabled/legacy paths, while an enabled CGSC pass overwrites
            # only the gate fields produced for the current batch.
            query_stats.update(calibration_stats)
            if self.sc_mask_center_mode == "local_relative":
                query_stats.update({
                    **{f"metric_lrm_p{level + 2}_pos_ratio": (
                        (lrm_relative_levels[level] > 0) & valid_level_masks[level]
                    ).sum(dim=1).div(valid_counts[:, level].clamp_min(1)).float().mean().detach()
                       for level in range(3)},
                    **{f"metric_lrm_p{level + 2}_mean_abs": torch.where(
                        valid_level_masks[level], lrm_relative_levels[level].abs(), torch.zeros_like(lrm_relative_levels[level])
                    ).sum(dim=1).div(valid_counts[:, level].clamp_min(1)).mean().detach()
                       for level in range(3)},
                })
            if allocation_stats is not None and evidence_details is not None:
                p_evidence = allocation_stats["p_dynamic"].clamp_min(1e-12)
                p_query = proportions.clamp_min(1e-12)
                query_stats.update({
                    **{f"metric_sc_effective_candidates_p{level + 2}":
                       evidence_details["effective_count"][:, level].mean().detach() for level in range(3)},
                    "metric_sc_dynamic_ratio": allocation_stats["dynamic_ratio"].detach(),
                    "metric_sc_temperature": allocation_stats["temperature"].detach(),
                    "metric_sc_evidence_query_kl": (
                        p_evidence * (p_evidence.log() - p_query.log())
                    ).sum(dim=1).mean().detach(),
                    "metric_sc_floor_hit_rate": allocation_stats["floor_hit"].mean().detach(),
                    "metric_sc_cap_hit_rate": allocation_stats["cap_hit"].mean().detach(),
                    "metric_sc_capacity_relax_rate": allocation_stats["capacity_relaxed"].mean().detach(),
                    "metric_sc_alpha_p2": (self.sc_alpha_max[0] * torch.tanh(self.sc_alpha2)).detach(),
                    "metric_sc_alpha_p3": (self.sc_alpha_max[1] * torch.tanh(self.sc_alpha3)).detach(),
                })
            self.last_query_stats = query_stats
            self.last_query_debug = (
                {
                    "shapes": tuple((int(h), int(w)) for h, w in shapes),
                    "priority_before": debug_priority_before,
                    "priority_after": [item.detach().cpu() for item in weighted_levels],
                    "selected_indices": selected.detach().cpu(),
                }
                if self.sc_store_last_batch
                else {}
            )
        return selected

    def _get_decoder_input(
        self,
        feats: torch.Tensor,
        shapes: list[list[int]],
        dn_embed: torch.Tensor | None = None,
        dn_bbox: torch.Tensor | None = None,
        freq_masks_dict: dict | None = None,
    ) -> tuple:
        bs, num_tokens, _ = feats.shape

        need_new_anchors = (
            self.dynamic
            or self.shapes != shapes
            or self.anchors.numel() == 0
            or self.anchors.device != feats.device
            or self.anchors.dtype != feats.dtype
        )
        if need_new_anchors:
            anchors, valid_mask = self._generate_anchors(
                shapes,
                dtype=feats.dtype,
                device=feats.device,
            )
            self.anchors = anchors
            self.valid_mask = valid_mask
            self.shapes = [list(shape) for shape in shapes]

        valid_mask_float = self.valid_mask.to(dtype=feats.dtype)
        features = self.enc_output(
            feats * valid_mask_float
        )
        self._assert_finite_tensor("encoder_features", features)
        enc_outputs_scores = self.enc_score_head(features)
        self._assert_finite_tensor("encoder_scores", enc_outputs_scores)

        # 防止无效 anchor 进入 Top-K。
        enc_outputs_scores = enc_outputs_scores.masked_fill(
            ~self.valid_mask,
            float("-inf"),
        )

        topk_ind = self._select_query_indices(enc_outputs_scores, shapes, freq_masks_dict)

        batch_ind = torch.arange(
            bs,
            device=feats.device,
            dtype=torch.long,
        ).unsqueeze(1)

        top_k_features = features[
            batch_ind,
            topk_ind,
        ]
        # OBB extension reads the same selected encoder features for its angle branch.
        self._last_top_k_features = top_k_features

        expanded_anchors = self.anchors.expand(
            bs,
            -1,
            -1,
        )
        top_k_anchors = expanded_anchors[
            batch_ind,
            topk_ind,
        ]

        refer_bbox = (
            self.enc_bbox_head(top_k_features)
            + top_k_anchors
        )
        enc_bboxes = refer_bbox.sigmoid()

        enc_scores = enc_outputs_scores[
            batch_ind,
            topk_ind,
        ]

        if self.learnt_init_query:
            embeddings = self.tgt_embed.weight.unsqueeze(0).expand(
                bs,
                -1,
                -1,
            )
        else:
            embeddings = top_k_features

        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()

        if dn_bbox is not None:
            refer_bbox = torch.cat(
                [dn_bbox, refer_bbox],
                dim=1,
            )
        if dn_embed is not None:
            embeddings = torch.cat(
                [dn_embed, embeddings],
                dim=1,
            )

        self._assert_finite_tensor("query_embeddings", embeddings)
        self._assert_finite_tensor("query_reference_points", refer_bbox)
        self._assert_finite_tensor("encoder_selected_boxes", enc_bboxes)
        self._assert_finite_tensor("encoder_selected_scores", enc_scores)

        return (
            embeddings,
            refer_bbox,
            enc_bboxes,
            enc_scores,
        )

    def _reset_parameters(self):
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc

        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)

        for cls_head, reg_head in zip(
            self.dec_score_head,
            self.dec_bbox_head,
        ):
            constant_(cls_head.bias, bias_cls)
            constant_(reg_head.layers[-1].weight, 0.0)
            constant_(reg_head.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)

        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)

        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)

        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class _WaveOBBRegressionHead(MLP):
    """Four-dimensional RT-DETR box head with a side-car pi-periodic angle branch.

    The inherited ``layers`` keep the same state-dict names as the original HBB MLP, so HBB decoder
    weights can be partially loaded into the OBB model. The decoder still receives only 4-D box deltas,
    while ``last_angle`` retains [sin(2theta), cos(2theta)] predictions for OBB supervision.
    """

    def __init__(self, hidden_dim: int):
        super().__init__(hidden_dim, hidden_dim, 4, num_layers=3)
        self.angle_head = MLP(hidden_dim, hidden_dim, 2, num_layers=3)
        self.last_angle: torch.Tensor | None = None

    def clear_angle_cache(self) -> None:
        self.last_angle = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angle_raw = self.angle_head(x)
        self.last_angle = safe_normalize_angle_vector(angle_raw)
        return super().forward(x)


class WaveRTDETRDecoderOBB(WaveRTDETRDecoder):
    """OBB extension of WaveRTDETRDecoder using [cx, cy, w, h, theta].

    Deformable attention continues to refine 4-D reference boxes. Each decoder regression head predicts an
    additional [sin(2theta), cos(2theta)] vector from the same query features. This avoids modifying the shared
    transformer implementation and preserves compatibility with HBB pretrained weights.
    """

    is_obb = True
    angle_representation = "sin2cos2"

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,
        nq: int = 300,
        ndp: int = 4,
        nh: int = 8,
        ndl: int = 6,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        nd: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
        freq_query_beta: float = 1.0,
    ):
        super().__init__(
            nc=nc,
            ch=ch,
            hd=hd,
            nq=nq,
            ndp=ndp,
            nh=nh,
            ndl=ndl,
            d_ffn=d_ffn,
            dropout=dropout,
            act=act,
            eval_idx=eval_idx,
            nd=nd,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            learnt_init_query=learnt_init_query,
            freq_query_beta=freq_query_beta,
        )

        # Replace only the decoder regression modules. The inherited ``layers`` keys stay compatible with HBB weights.
        self.dec_bbox_head = nn.ModuleList([_WaveOBBRegressionHead(hd) for _ in range(ndl)])
        self.enc_angle_head = MLP(hd, hd, 2, num_layers=3)
        self._reset_obb_parameters()

    @staticmethod
    def angle_vector_to_theta(angle_vector: torch.Tensor) -> torch.Tensor:
        angle_vector = safe_normalize_angle_vector(angle_vector)
        return 0.5 * torch.atan2(angle_vector[..., 0], angle_vector[..., 1])

    def _reset_obb_parameters(self) -> None:
        # Keep the four-dimensional box branch identical to the HBB initialization.
        for head in self.dec_bbox_head:
            constant_(head.layers[-1].weight, 0.0)
            constant_(head.layers[-1].bias, 0.0)
            constant_(head.angle_head.layers[-1].weight, 0.0)
            constant_(head.angle_head.layers[-1].bias, 0.0)
            # [sin(0), cos(0)] gives a stable zero-degree initial orientation.
            head.angle_head.layers[-1].bias.data[1] = 1.0

        constant_(self.enc_angle_head.layers[-1].weight, 0.0)
        constant_(self.enc_angle_head.layers[-1].bias, 0.0)
        self.enc_angle_head.layers[-1].bias.data[1] = 1.0

    def _collect_decoder_angles(self, dec_bboxes: torch.Tensor) -> torch.Tensor:
        angle_layers = [head.last_angle for head in self.dec_bbox_head if head.last_angle is not None]
        if not angle_layers:
            raise RuntimeError("OBB decoder produced no angle tensors. Check DeformableTransformerDecoder head calls.")

        if dec_bboxes.ndim == 4:
            expected_layers = dec_bboxes.shape[0]
            if len(angle_layers) >= expected_layers:
                return torch.stack(angle_layers[-expected_layers:], dim=0)
            if expected_layers == 1:
                return angle_layers[-1].unsqueeze(0)
            raise RuntimeError(
                f"OBB angle layer count {len(angle_layers)} does not match box layer count {expected_layers}."
            )

        if dec_bboxes.ndim == 3:
            return angle_layers[-1]
        raise ValueError(f"Unexpected decoder box shape for OBB: {tuple(dec_bboxes.shape)}")

    def forward(
        self,
        x: list[torch.Tensor],
        batch: dict | None = None,
        freq_masks: dict | None = None,
    ) -> tuple | torch.Tensor:
        from ultralytics.models.utils.ops import get_cdn_group

        feats, shapes = self._get_encoder_input(x)
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(
            feats=feats,
            shapes=shapes,
            dn_embed=dn_embed,
            dn_bbox=dn_bbox,
            freq_masks_dict=freq_masks,
        )
        if not hasattr(self, "_last_top_k_features"):
            raise RuntimeError("Encoder Top-K features were not cached for the OBB angle branch.")
        enc_angles = safe_normalize_angle_vector(self.enc_angle_head(self._last_top_k_features))
        self._assert_finite_tensor("encoder_selected_angles", enc_angles)

        for head in self.dec_bbox_head:
            head.clear_angle_cache()

        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        dec_angles = self._collect_decoder_angles(dec_bboxes)
        self._assert_finite_tensor("decoder_boxes", dec_bboxes)
        self._assert_finite_tensor("decoder_scores", dec_scores)
        self._assert_finite_tensor("decoder_angles", dec_angles)

        x_out = (
            dec_bboxes,
            dec_scores,
            dec_angles,
            enc_bboxes,
            enc_scores,
            enc_angles,
            dn_meta,
        )
        if self.training:
            return x_out

        final_bboxes = dec_bboxes[-1] if dec_bboxes.ndim == 4 else dec_bboxes
        final_scores = dec_scores[-1] if dec_scores.ndim == 4 else dec_scores
        final_angles = dec_angles[-1] if dec_angles.ndim == 4 else dec_angles
        theta = self.angle_vector_to_theta(final_angles).unsqueeze(-1)
        # Inference format: [cx, cy, w, h, theta, class probabilities...].
        y = torch.cat((final_bboxes, theta, final_scores.sigmoid()), dim=-1)
        return y if self.export else (y, x_out)


class v10Detect(Detect):
    """v10 Detection head from https://arxiv.org/pdf/2405.14458.

    This class implements the YOLOv10 detection head with dual-assignment training and consistent dual predictions for
    improved efficiency and performance.

    Attributes:
        end2end (bool): End-to-end detection mode.
        max_det (int): Maximum number of detections.
        cv3 (nn.ModuleList): Light classification head layers.
        one2one_cv3 (nn.ModuleList): One-to-one classification head layers.

    Methods:
        __init__: Initialize the v10Detect object with specified number of classes and input channels.
        forward: Perform forward pass of the v10Detect module.
        bias_init: Initialize biases of the Detect module.
        fuse: Remove the one2many head for inference optimization.

    Examples:
        Create a v10Detect head
        >>> v10_detect = v10Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = v10_detect(x)
    """

    end2end = True

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """Initialize the v10Detect object with the specified number of classes and input channels.

        Args:
            nc (int): Number of classes.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, end2end=True, ch=ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)

    def fuse(self):
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = None
