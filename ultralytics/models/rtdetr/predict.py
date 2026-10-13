# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class RTDETRPredictor(BasePredictor):
    """Predictor for RT-DETR HBB and FMC-Wave RT-DETR-OBB."""

    def _is_obb(self) -> bool:
        return getattr(self.args, "task", "detect") == "obb"

    def postprocess(self, preds, img, orig_imgs):
        if not isinstance(preds, (list, tuple)):
            preds = [preds, None]
        raw = preds[0]

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        if self._is_obb():
            if raw.ndim != 3 or raw.shape[-1] <= 5:
                raise ValueError(f"RT-DETR-OBB predictions must be [B,N,5+nc], got {tuple(raw.shape)}")
            nd = raw.shape[-1]
            rboxes, scores = raw.split((5, nd - 5), dim=-1)
            results = []
            input_hw = tuple(img.shape[2:])
            for rbox, score, orig_img, img_path in zip(rboxes, scores, orig_imgs, self.batch[0]):
                max_score, cls = score.max(-1, keepdim=True)
                keep = max_score.squeeze(-1) > self.args.conf
                if self.args.classes is not None:
                    allowed = torch.as_tensor(self.args.classes, device=cls.device)
                    keep = keep & (cls == allowed[None, :]).any(1)

                rbox = rbox[keep].clone()
                max_score = max_score[keep]
                cls = cls[keep]
                order = max_score.squeeze(-1).argsort(descending=True)[: self.args.max_det]
                rbox, max_score, cls = rbox[order], max_score[order], cls[order]

                # Network output is normalized xywhr. OBB preprocessing uses isotropic letterboxing.
                rbox[:, [0, 2]] *= input_hw[1]
                rbox[:, [1, 3]] *= input_hw[0]
                if rbox.numel():
                    ops.scale_boxes(input_hw, rbox[:, :4], orig_img.shape[:2], xywh=True)
                pred = torch.cat((rbox, max_score, cls.to(rbox.dtype)), dim=-1)
                results.append(Results(orig_img, path=img_path, names=self.model.names, obb=pred))
            return results

        nd = raw.shape[-1]
        bboxes, scores = raw.split((4, nd - 4), dim=-1)
        results = []
        for bbox, score, orig_img, img_path in zip(bboxes, scores, orig_imgs, self.batch[0]):
            bbox = ops.xywh2xyxy(bbox)
            max_score, cls = score.max(-1, keepdim=True)
            keep = max_score.squeeze(-1) > self.args.conf
            if self.args.classes is not None:
                allowed = torch.as_tensor(self.args.classes, device=cls.device)
                keep = keep & (cls == allowed[None, :]).any(1)
            pred = torch.cat((bbox, max_score, cls), dim=-1)[keep]
            pred = pred[pred[:, 4].argsort(descending=True)][: self.args.max_det]
            oh, ow = orig_img.shape[:2]
            pred[..., [0, 2]] *= ow
            pred[..., [1, 3]] *= oh
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def pre_transform(self, im):
        # OBB must not use anisotropic scale_fill because that changes orientation angles.
        letterbox = LetterBox(
            self.imgsz,
            auto=False,
            scale_fill=not self._is_obb(),
            scaleup=True,
        )
        return [letterbox(image=x) for x in im]
