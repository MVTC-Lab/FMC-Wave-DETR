# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""RT-DETR model interface with HBB and OBB task routing."""

from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import RTDETRDetectionModel, guess_model_task, yaml_model_load
from ultralytics.utils.torch_utils import TORCH_1_11

from .predict import RTDETRPredictor
from .train import RTDETRTrainer
from .val import RTDETROBBValidator, RTDETRValidator


class RTDETR(Model):
    """RT-DETR interface supporting standard detect and FMC-Wave OBB YAML/PT models."""

    def __init__(self, model: str = "rtdetr-l.pt") -> None:
        assert TORCH_1_11, "RTDETR requires torch>=1.11"
        if isinstance(model, (str, Path)) and Path(model).suffix in {".yaml", ".yml"}:
            cfg = yaml_model_load(model)
            task = cfg.get("task") or guess_model_task(cfg)
        else:
            task = guess_model_task(model)
        if task not in {"detect", "obb"}:
            raise ValueError(f"RTDETR supports task='detect' or task='obb', got {task!r} for model {model}")
        super().__init__(model=model, task=task)

    @property
    def task_map(self) -> dict:
        return {
            "detect": {
                "predictor": RTDETRPredictor,
                "validator": RTDETRValidator,
                "trainer": RTDETRTrainer,
                "model": RTDETRDetectionModel,
            },
            "obb": {
                "predictor": RTDETRPredictor,
                "validator": RTDETROBBValidator,
                "trainer": RTDETRTrainer,
                "model": RTDETRDetectionModel,
            },
        }
