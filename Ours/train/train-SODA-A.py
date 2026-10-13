import os
import sys
from pathlib import Path

# ============================================================
# 1. 环境变量：必须在 import torch / ultralytics 之前设置
# ============================================================

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

# SODA-A 正图全保留 + 空图平衡采样
os.environ["SODAA_BALANCED_EMPTY_SAMPLER"] = "1"
os.environ["SODAA_REQUIRED_GLOBAL_BATCH"] = "24"
os.environ["SODAA_NEGATIVE_PER_EPOCH"] = "6907"
os.environ["SODAA_NEGATIVE_RATIO"] = "0.25"
os.environ["SODAA_SAMPLER_SEED"] = "42"

PROJECT_ROOT = Path("/workspace/FMC-OBB/ultralytics-main").resolve()

sys.path = [
    path for path in sys.path
    if "/workspace/YOLO26/ultralytics-main" not in str(path)
]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import ultralytics

expected_ultralytics = (PROJECT_ROOT / "ultralytics").resolve()
actual_ultralytics = Path(ultralytics.__file__).resolve()

if expected_ultralytics not in actual_ultralytics.parents:
    raise RuntimeError(
        "Imported the wrong Ultralytics package.\n"
        f"Expected under: {expected_ultralytics}\n"
        f"Actually loaded: {actual_ultralytics}"
    )

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.use_deterministic_algorithms(False)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

from ultralytics import RTDETR

MODEL = (
    PROJECT_ROOT
    / ""
)
DATA = (
    PROJECT_ROOT
    / ""
)

DEVICE = ""
NUM_GPUS = 4
PHYSICAL_GLOBAL_BATCH = 24
LOCAL_BATCH = PHYSICAL_GLOBAL_BATCH // NUM_GPUS
NOMINAL_BATCH = 8
WORKERS_PER_RANK = 4
EPOCHS = 200
IMAGE_SIZE = 1200


def main() -> None:
    if not MODEL.is_file():
        raise FileNotFoundError(f"Model YAML does not exist: {MODEL}")
    if not DATA.is_file():
        raise FileNotFoundError(f"Data YAML does not exist: {DATA}")
    if PHYSICAL_GLOBAL_BATCH != 24:
        raise ValueError(
            f"This configuration requires global batch=24, got {PHYSICAL_GLOBAL_BATCH}."
        )
    if PHYSICAL_GLOBAL_BATCH % NUM_GPUS != 0:
        raise ValueError(
            f"Global batch {PHYSICAL_GLOBAL_BATCH} cannot be divided by {NUM_GPUS} GPUs."
        )
    if LOCAL_BATCH != 6:
        raise ValueError(
            f"Expected local batch=6, got {LOCAL_BATCH}."
        )

    negative_per_global_batch = round(
        PHYSICAL_GLOBAL_BATCH
        * float(os.environ["SODAA_NEGATIVE_RATIO"])
    )
    positive_per_global_batch = (
        PHYSICAL_GLOBAL_BATCH - negative_per_global_batch
    )
    if (positive_per_global_batch, negative_per_global_batch) != (18, 6):
        raise ValueError(
            "Expected each global batch to contain 18 positive "
            f"and 6 empty images, got {positive_per_global_batch} "
            f"positive and {negative_per_global_batch} empty."
        )

    print("=" * 92)
    print(f"Python:                  {sys.executable}")
    print(f"Ultralytics:             {actual_ultralytics}")
    print(f"Model:                   {MODEL}")
    print(f"Data:                    {DATA}")
    print(f"Devices:                 {DEVICE}")
    print(f"Physical global batch:   {PHYSICAL_GLOBAL_BATCH}")
    print(f"Local batch/GPU:         {LOCAL_BATCH}")
    print(
        f"Global batch mixture:    "
        f"{positive_per_global_batch} POS + "
        f"{negative_per_global_batch} EMPTY"
    )
    print(f"Nominal batch:           {NOMINAL_BATCH}")
    print(f"Workers/rank:            {WORKERS_PER_RANK}")
    print(f"Epochs:                  {EPOCHS}")
    print(f"Requested content size:  {IMAGE_SIZE}")
    print("Expected padded tensor:  1216 x 1216")
    print("=" * 92)

    # 当前自定义 RTDETR 构造器不接受 task= 参数
    model = RTDETR(str(MODEL))
    model.task = "obb"
    if hasattr(model, "model"):
        model.model.task = "obb"

    print(f"Wrapper task:            {model.task}")
    print(
        "Inner model task:        "
        f"{getattr(model.model, 'task', 'undefined')}"
    )

    model.train(
        task="obb",
        data=str(DATA),

        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,

        batch=24,
        nbs=24,
        device=DEVICE,
        workers=4,

        optimizer="AdamW",
        lr0=5e-5,
        lrf=0.01,
        momentum=0.937,
        weight_decay=1e-4,
        cos_lr=True,

        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.01,

        amp=True,

        seed=42,
        deterministic=False,

        mosaic=0.0,
        close_mosaic=0,
        mixup=0.0,
        copy_paste=0.0,

        degrees=0.0,
        perspective=0.0,
        shear=0.0,

        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.5,

        max_det=1000,

        val=True,
        plots=False,

        save=True,
        save_period=10,
        pretrained=False,
        patience=200,

        project = "",
        name = ""
    )


if __name__ == "__main__":
    main()