# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK, colorstr

from .val import RTDETRDataset, RTDETROBBValidator, RTDETRValidator


class RTDETRTrainer(DetectionTrainer):
    """Trainer shared by RT-DETR HBB and FMC-Wave RT-DETR-OBB."""

    def get_model(self, cfg: dict | None = None, weights: str | None = None, verbose: bool = True):
        model = RTDETRDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            prefix=colorstr(f"{mode}: "),
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            task=self.args.task,
        )

    def preprocess_batch(self, batch):
        """Preprocess a batch and attach deterministic OBB curriculum progress on every DDP rank."""
        batch = super().preprocess_batch(batch)
        if self.args.task == "obb":
            batch["_obb_loss_epoch"] = int(self.epoch)
            batch["_obb_loss_total_epochs"] = int(self.epochs)
        return batch

    def _terminal_loss_names(self) -> tuple[str, ...]:
        """Show only six optimized OBB losses while retaining all CSV diagnostics."""
        names = tuple(self.loss_names)
        return names[:6] if self.args.task == "obb" else names

    def get_validator(self):
        if self.args.task == "obb":
            self.loss_names = (
                "gwd_loss", "cls_loss", "l1_loss", "angle_loss", "riou_loss", "corner_loss",
                "gwd_raw", "riou_raw", "corner_raw", "angle_raw", "angle_weight",
                "mask_P2", "mask_P3", "mask_P4",
                "sc_selected_p2", "sc_selected_p3", "sc_selected_p4",
                "sc_evidence_p2", "sc_evidence_p3", "sc_evidence_p4",
                "sc_mask_raw_mean_p2", "sc_mask_raw_mean_p3", "sc_mask_raw_mean_p4",
                "sc_mask_cal_mean_p2", "sc_mask_cal_mean_p3", "sc_mask_cal_mean_p4",
                "sc_peak_ratio_p2", "sc_peak_ratio_p3", "sc_peak_ratio_p4",
                "sc_query_entropy", "sc_fixed_fallback_rate", "sc_global_fallback_rate", "sc_fallback_count",
                "sc_priority_top_mean_p2", "sc_priority_top_mean_p3", "sc_priority_top_mean_p4",
                "sc_selected_std_p2", "sc_selected_std_p3", "sc_selected_std_p4",
                "sc_selected_min_p2", "sc_selected_min_p3", "sc_selected_min_p4",
                "sc_selected_max_p2", "sc_selected_max_p3", "sc_selected_max_p4",
                "sc_selected_p10_p2", "sc_selected_p10_p3", "sc_selected_p10_p4",
                "sc_selected_p90_p2", "sc_selected_p90_p3", "sc_selected_p90_p4",
                "sc_effective_candidates_p2", "sc_effective_candidates_p3", "sc_effective_candidates_p4",
                "sc_dynamic_ratio", "sc_temperature", "sc_evidence_query_kl",
                "sc_floor_hit_rate", "sc_cap_hit_rate", "sc_capacity_relax_rate",
                "sc_alpha_p2", "sc_alpha_p3",
            )
            return RTDETROBBValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
        self.loss_names = "giou_loss", "cls_loss", "l1_loss"
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
