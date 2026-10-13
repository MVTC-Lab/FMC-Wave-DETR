# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format, LetterBox, v8_transforms
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.obb.val import OBBValidator
from ultralytics.utils import LOGGER, colorstr, ops
from ultralytics.utils.metrics import OBBMetrics
from ultralytics.utils.torch_utils import unwrap_model

__all__ = ("RTDETRDataset", "RTDETRValidator", "RTDETROBBMetrics", "RTDETROBBValidator")


SODAA_OFFICIAL_PATCH_SIZE = 800
SODAA_OFFICIAL_CONTENT_SIZE = 1200
SODAA_OFFICIAL_INPUT_SIZE = 1216


def _uses_sodaa_official_1200_protocol(img_path: Any, task: str, imgsz: int) -> bool:
    """Return whether this dataset must use official SODA-A 1200-content preprocessing."""
    path_token = "".join(character for character in str(img_path).lower() if character.isalnum())
    return str(task).lower() == "obb" and int(imgsz) == SODAA_OFFICIAL_INPUT_SIZE and "sodaa" in path_token


class RTDETRDataset(YOLODataset):
    """Real-Time DEtection and TRacking (RT-DETR) dataset class extending the base YOLODataset class.

    This specialized dataset class is designed for use with the RT-DETR object detection model and is optimized for
    real-time detection and tracking tasks.

    Attributes:
        augment (bool): Whether to apply data augmentation.
        rect (bool): Whether to use rectangular training.
        use_segments (bool): Whether to use segmentation masks.
        use_keypoints (bool): Whether to use keypoint annotations.
        imgsz (int): Target image size for training.

    Methods:
        load_image: Load one image from dataset index.
        build_transforms: Build transformation pipeline for the dataset.

    Examples:
        Initialize an RT-DETR dataset
        >>> dataset = RTDETRDataset(img_path="path/to/images", imgsz=640)
        >>> image, hw0, hw = dataset.load_image(0)
    """

    def __init__(self, *args, data=None, task: str = "detect", **kwargs):
        """Initialize RT-DETR dataset for HBB detection or OBB detection."""
        img_path = kwargs.get("img_path", args[0] if args else "")
        imgsz = kwargs.get("imgsz", 640)
        self.sodaa_official_1200 = _uses_sodaa_official_1200_protocol(img_path, task, imgsz)
        super().__init__(*args, data=data, task=task, **kwargs)
        if self.sodaa_official_1200:
            LOGGER.info(
                f"{self.prefix}SODA-A official preprocessing: "
                f"{SODAA_OFFICIAL_PATCH_SIZE} patch -> {SODAA_OFFICIAL_CONTENT_SIZE} content -> "
                f"right/bottom pad to {SODAA_OFFICIAL_INPUT_SIZE}."
            )

    def load_image(self, i, rect_mode=False):
        """Load one image from dataset index 'i'.

        Args:
            i (int): Index of the image to load.
            rect_mode (bool, optional): Whether to use rectangular mode for batch inference.

        Returns:
            im (np.ndarray): Loaded image as a NumPy array.
            hw_original (tuple[int, int]): Original image dimensions in (height, width) format.
            hw_resized (tuple[int, int]): Resized image dimensions in (height, width) format.

        Examples:
            Load an image from the dataset
            >>> dataset = RTDETRDataset(img_path="path/to/images")
            >>> image, hw0, hw = dataset.load_image(0)
        """
        if not getattr(self, "sodaa_official_1200", False):
            return super().load_image(i=i, rect_mode=rect_mode)

        padded_size = self.imgsz
        self.imgsz = SODAA_OFFICIAL_CONTENT_SIZE
        try:
            image, original_shape, resized_shape = super().load_image(i=i, rect_mode=False)
        finally:
            self.imgsz = padded_size
        if original_shape != (SODAA_OFFICIAL_PATCH_SIZE, SODAA_OFFICIAL_PATCH_SIZE):
            raise ValueError(
                "Strict SODA-A protocol requires official 800x800 patches, "
                f"got {original_shape} for {self.im_files[i]}."
            )
        if resized_shape != (SODAA_OFFICIAL_CONTENT_SIZE, SODAA_OFFICIAL_CONTENT_SIZE):
            raise RuntimeError(f"SODA-A patch was not resized to 1200x1200: got {resized_shape}.")
        return image, original_shape, resized_shape

    def build_transforms(self, hyp=None):
        """Build transforms while preserving OBB geometry with isotropic letterboxing."""
        official_sodaa = getattr(self, "sodaa_official_1200", False)
        if official_sodaa and self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, SODAA_OFFICIAL_CONTENT_SIZE, hyp)
            transforms.append(
                LetterBox(
                    new_shape=(SODAA_OFFICIAL_INPUT_SIZE, SODAA_OFFICIAL_INPUT_SIZE),
                    auto=False,
                    scale_fill=False,
                    scaleup=False,
                    center=False,
                    padding_value=114,
                )
            )
        elif official_sodaa:
            transforms = Compose(
                [
                    LetterBox(
                        new_shape=(SODAA_OFFICIAL_INPUT_SIZE, SODAA_OFFICIAL_INPUT_SIZE),
                        auto=False,
                        scale_fill=False,
                        scaleup=False,
                        center=False,
                        padding_value=114,
                    )
                ]
            )
        elif self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = (
                Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), auto=False, scale_fill=False, scaleup=False)])
                if self.use_obb
                else Compose([])
            )
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms


class RTDETRValidator(DetectionValidator):
    """RTDETRValidator extends the DetectionValidator class to provide validation capabilities specifically tailored for
    the RT-DETR (Real-Time DETR) object detection model.

    The class allows building of an RTDETR-specific dataset for validation, applies Non-maximum suppression for
    post-processing, and updates evaluation metrics accordingly.

    Attributes:
        args (Namespace): Configuration arguments for validation.
        data (dict): Dataset configuration dictionary.

    Methods:
        build_dataset: Build an RTDETR Dataset for validation.
        postprocess: Apply Non-maximum suppression to prediction outputs.

    Examples:
        Initialize and run RT-DETR validation
        >>> from ultralytics.models.rtdetr import RTDETRValidator
        >>> args = dict(model="rtdetr-l.pt", data="coco8.yaml")
        >>> validator = RTDETRValidator(args=args)
        >>> validator()

    Notes:
        For further details on the attributes and methods, refer to the parent DetectionValidator class.
    """

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build an RTDETR Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str, optional): `train` mode or `val` mode, users are able to customize different augmentations for
                each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (RTDETRDataset): Dataset configured for RT-DETR validation.
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # no augmentation
            hyp=self.args,
            rect=False,  # no rect
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
            task="detect",
        )

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Scales predictions to the original image size."""
        return predn

    def postprocess(
        self, preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        """Apply Non-maximum suppression to prediction outputs.

        Args:
            preds (torch.Tensor | list | tuple): Raw predictions from the model. If tensor, should have shape
                (batch_size, num_predictions, num_classes + 4) where last dimension contains bbox coords and
                class scores.

        Returns:
            (list[dict[str, torch.Tensor]]): List of dictionaries for each image, each containing:
                - 'bboxes': Tensor of shape (N, 4) with bounding box coordinates
                - 'conf': Tensor of shape (N,) with confidence scores
                - 'cls': Tensor of shape (N,) with class indices
        """
        if not isinstance(preds, (list, tuple)):  # list for PyTorch inference but list[0] Tensor for export inference
            preds = [preds, None]

        bs, _, nd = preds[0].shape
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        bboxes *= self.args.imgsz
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1)  # (300, )
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)  # filter
            # Sort by confidence to correctly get internal metrics
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred[score > self.args.conf]

        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5]} for x in outputs]

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """Serialize YOLO predictions to COCO json format.

        Args:
            predn (dict[str, torch.Tensor]): Predictions dictionary containing 'bboxes', 'conf', and 'cls' keys with
                bounding box coordinates, confidence scores, and class predictions.
            pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.
        """
        path = Path(pbatch["im_file"])
        stem = path.stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = predn["bboxes"].clone()
        box[..., [0, 2]] *= pbatch["ori_shape"][1] / self.args.imgsz  # native-space pred
        box[..., [1, 3]] *= pbatch["ori_shape"][0] / self.args.imgsz  # native-space pred
        box = ops.xyxy2xywh(box)  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for b, s, c in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "file_name": path.name,
                    "category_id": self.class_map[int(c)],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(s, 5),
                }
            )

class RTDETROBBMetrics(OBBMetrics):
    """OBB metrics extended with AP75 while preserving the inherited fitness calculation."""

    @property
    def keys(self) -> list[str]:
        return [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP75(B)",
            "metrics/mAP50-95(B)",
        ]

    def mean_results(self) -> list[float]:
        return [self.box.mp, self.box.mr, self.box.map50, self.box.map75, self.box.map]

    def class_result(self, i: int) -> tuple[float, float, float, float, float]:
        """Return per-class precision, recall, AP50, AP75, and AP50-95 in the same order as ``keys``."""
        return self.box.p[i], self.box.r[i], self.box.ap50[i], self.box.all_ap[i, 5], self.box.ap[i]


class RTDETROBBValidator(OBBValidator):
    """OBB validator for normalized RT-DETR outputs [cx,cy,w,h,theta,class_scores...]."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)
        self.metrics = RTDETROBBMetrics()

    def __call__(self, trainer=None, model=None):
        """Synchronize the EMA/core model curriculum before training-time validation."""
        if trainer is not None:
            active_model = trainer.ema.ema or trainer.model
            core = unwrap_model(active_model)
            if hasattr(core, "set_obb_loss_progress"):
                core.set_obb_loss_progress(trainer.epoch, trainer.epochs)
        return super().__call__(trainer=trainer, model=model)

    def build_dataset(self, img_path, mode="val", batch=None):
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
            task="obb",
        )

    def postprocess(
        self, preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        if not isinstance(preds, (list, tuple)):
            preds = [preds, None]
        raw = preds[0]
        if raw.ndim != 3 or raw.shape[-1] <= 5:
            raise ValueError(f"RT-DETR-OBB predictions must be [B,N,5+nc], got {tuple(raw.shape)}")

        bs, _, nd = raw.shape
        rboxes, scores = raw.split((5, nd - 5), dim=-1)
        rboxes = rboxes.clone()
        rboxes[..., :4] *= float(self.args.imgsz)

        outputs = []
        for image_idx in range(bs):
            score, cls = scores[image_idx].max(-1)
            keep = score > self.args.conf
            if self.args.classes is not None:
                allowed = torch.as_tensor(self.args.classes, device=cls.device)
                keep = keep & (cls[:, None] == allowed[None, :]).any(1)
            order = score[keep].argsort(descending=True)
            boxes_i = rboxes[image_idx][keep][order][: self.args.max_det]
            score_i = score[keep][order][: self.args.max_det]
            cls_i = cls[keep][order][: self.args.max_det]
            outputs.append({"bboxes": boxes_i, "conf": score_i, "cls": cls_i})
        return outputs
