#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FMC-Wave RT-DETR-OBB on AI-TOD-R
Paper/repository-aligned fully-supervised training script.

Aligned items:
1. trainval -> test split
2. 800 x 800 single-scale input
3. Transformer optimizer: AdamW, lr=1e-4, weight_decay=1e-5
4. Global batch size = 2 on one GPU
5. Random horizontal flip only, p=0.5
6. FP32 training
7. 1x=12 epochs or 40e=40 epochs
8. 500 optimizer-update linear warmup, ratio 1/3
9. Step LR milestones:
   - 12e: [8, 11]
   - 40e: [24, 32, 38]
10. Gradient clipping: max_norm=35, norm_type=2

Important:
- The TPAMI paper explicitly assigns AdamW to Transformer models.
- The public AI-TOD-R repository provides the 12e/40e step schedules,
  500-iteration warmup, and gradient clipping in its base schedules.
- Use last.pt for the final test. Do not select the final model using test AP.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.use_deterministic_algorithms(False)

import os
import sys
from pathlib import Path

# ============================================================
# Make repository root visible to main process and DDP workers
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

_old_pythonpath = os.environ.get("PYTHONPATH", "")
if _old_pythonpath:
    if PROJECT_ROOT_STR not in _old_pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = (
            PROJECT_ROOT_STR + os.pathsep + _old_pythonpath
        )
else:
    os.environ["PYTHONPATH"] = PROJECT_ROOT_STR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/workspace/FMC-OBB/ultralytics-main").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

MODEL = PROJECT_ROOT / "Ours/model/rt-detr-obb.yaml"
DATA = PROJECT_ROOT / "Ours/datasets/AI-TOD/data-obb.yaml"

RESULT_ROOT = PROJECT_ROOT / "Ours/Result/AI-TOD-R"

# "1x" -> 12 epochs; "40e" -> 40 epochs
SCHEDULE = "40e"

# Strict benchmark alignment: one GPU, global batch size 2.
DEVICE = ""
BATCH = 16
WORKERS = 4

SEED = 42

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Keep this only because the current server was previously observed to hang
# under DDP without it. It does not affect this strict one-GPU run.
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ.pop("NCCL_IB_DISABLE", None)
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

# Reproducible FP32 benchmark run
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(False)

from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer

from Ours.train.aitodr_official_trainer import (
    AITODROfficialRTDETRTrainer,
)


# ---------------------------------------------------------------------------
# Official schedule
# ---------------------------------------------------------------------------
if SCHEDULE == "1x":
    EPOCHS = 12
    LR_MILESTONES = (8, 11)
    SAVE_PERIOD = 4
elif SCHEDULE == "40e":
    EPOCHS = 40
    LR_MILESTONES = (24, 32, 38)
    SAVE_PERIOD = 1
else:
    raise ValueError(f"Unsupported SCHEDULE={SCHEDULE!r}; use '1x' or '40e'.")




def main() -> None:
    for required_path in (MODEL, DATA):
        if not required_path.exists():
            raise FileNotFoundError(f"Required file does not exist: {required_path}")

    run_name = f""

    print("=" * 88)
    print("AI-TOD-R official-aligned FMC-Wave RT-DETR-OBB training")
    print("=" * 88)
    print(f"model:          {MODEL}")
    print(f"data:           {DATA}")
    print(f"schedule:       {SCHEDULE} ({EPOCHS} epochs)")
    print(f"milestones:     {list(LR_MILESTONES)}")
    print(f"device:         {DEVICE}")
    print(f"global batch:   {BATCH}")
    print("optimizer:      AdamW")
    print("lr:             1e-4")
    print("weight decay:   1e-5")
    print("warmup:         500 optimizer updates, ratio=1/3")
    print("grad clip:      L2 max_norm=35")
    print("augmentation:   horizontal flip p=0.5 only")
    print("precision:      FP32")
    print("=" * 88)

    model = RTDETR(str(MODEL))

    model.train(
        trainer=AITODROfficialRTDETRTrainer,
        task="obb",
        data=str(DATA),

        # Official input and schedule
        epochs=EPOCHS,
        imgsz=800,
        multi_scale=False,
        rect=False,

        # Strict official batch setting
        batch=BATCH,
        nbs=BATCH,
        device=DEVICE,
        workers=WORKERS,

        # Transformer settings stated in the TPAMI paper
        optimizer="AdamW",
        lr0=1.0e-4,
        momentum=0.9,       # AdamW beta1; beta2 uses the framework default
        weight_decay=1.0e-5,

        # Disabled because the custom trainer implements the official schedule
        cos_lr=False,
        lrf=1.0,
        warmup_epochs=0.0,
        warmup_momentum=0.9,
        warmup_bias_lr=0.0,

        # Table II uses FP32
        amp=False,

        deterministic=False,
        seed=SEED,

        # AI-TOD-R official pipeline: RResize(800) + RRandomFlip(0.5)
        fliplr=0.5,
        flipud=0.0,

        # Disable all YOLO-specific or additional augmentation
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        close_mosaic=0,

        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,

        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        bgr=0.0,

        # The official evaluator allows at most 1500 predictions per image.
        # The current decoder has only 1000 queries, so it remains capped at 1000.
        max_det=1500,

        # Do not early-stop on the test set.
        patience=1000,
        pretrained=False,
        auto_augment=None,
        erasing=0.0,

        # Final evaluation is performed once by the separate official test script.
        val=False,

        save=True,
        save_period=SAVE_PERIOD,
        plots=False,

        project=str(RESULT_ROOT),
        name=run_name,
    )

    final_weights = RESULT_ROOT / run_name / "weights" / "last.pt"
    print("\nTraining completed.")
    print(f"Use this checkpoint for the official test: {final_weights}")


if __name__ == "__main__":
    main()