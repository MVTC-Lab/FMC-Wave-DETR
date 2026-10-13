#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Official SODA-A test pipeline for a custom Ultralytics RT-DETR-OBB model.

Stages
------
1) infer:
   - Resize official 800x800 patches to 1200x1200 effective content.
   - Pad only the right and bottom sides to a 1216x1216 model tensor.
   - Run inference without any second resize and map predictions back by 1/1.5.
   - Read patch offsets from patch_manifest.csv.
   - Map patch OBB predictions back to original-image coordinates.
   - Save a patch prediction cache.

2) eval:
   - Merge overlapping patch predictions on each original image with
     class-wise rotated NMS (official default IoU threshold 0.5).
   - Load original SODA-A test annotations.
   - Run the official SODAAeval implementation.
   - Report:
       AP, AP50, AP75, APeS, APrS, APgS, APN
     plus AR metrics and per-class AP.

3) all:
   - Run both stages.

The official evaluator uses:
  IoU thresholds: 0.50:0.05:0.95
  maxDets: 20000
  area ranges:
    Small : [0^2, 32^2]
    eS    : [0^2, 12^2]
    rS    : [12^2, 20^2]
    gS    : [20^2, 32^2]
    Normal: [32^2, 40*50]

Requirements
------------
Inference stage:
    ultralytics, torch, opencv-python, numpy, tqdm

Evaluation stage:
    mmcv with compiled ops:
        from mmcv.ops import nms_rotated, box_iou_rotated
    official SODA-A evaluator file:
        SODA-mmrotate/mmrotate/datasets/sodaa_eval/sodaa_eval.py

Usage
-----
1. Modify the USER CONFIGURATION section near the top of this file.
2. Run directly without command-line arguments:

    python test_sodaa_official_fixed_config.py

Recommended stage values:
    "infer" : patch inference only
    "eval"  : original-image merge/evaluation only
    "all"   : inference + merge + evaluation
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm


def install_numpy_legacy_aliases() -> list[str]:
    """Restore aliases removed by NumPy 1.24/2.x for the legacy evaluator.

    The official SODA-A evaluator still uses names such as ``np.float`` in
    ``accumulate()``.  NumPy 2.x removes these aliases, which otherwise causes
    evaluation to fail after the expensive IoU calculation has completed.

    This function runs at module import time, so it is also executed inside
    multiprocessing ``spawn`` workers.
    """
    aliases = {
        "float": float,
        "int": int,
        "bool": bool,
        "complex": complex,
        "object": object,
        "str": str,
    }
    installed: list[str] = []
    for name, value in aliases.items():
        # Use np.__dict__ instead of hasattr() to avoid NumPy FutureWarnings.
        if name not in np.__dict__:
            setattr(np, name, value)
            installed.append(name)
    return installed


NUMPY_COMPAT_ALIASES = install_numpy_legacy_aliases()


CLASS_NAMES = [
    "airplane",
    "helicopter",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "container",
    "storage-tank",
    "swimming-pool",
    "windmill",
]
NUM_CLASSES = len(CLASS_NAMES)

OFFICIAL_PROTOCOL_ID = "sodaa-official-800-resize1200-pad1216-right-bottom-v1"
PATCH_SOURCE_SIZE = 800
EFFECTIVE_CONTENT_SIZE = 1200
PADDED_INPUT_SIZE = 1216
PATCH_TO_CONTENT_SCALE = EFFECTIVE_CONTENT_SIZE / PATCH_SOURCE_SIZE


# =============================================================================
# USER CONFIGURATION — 修改下面这一段后，直接运行本文件即可
# =============================================================================
# 阶段：
#   "infer" = 只对测试切片推理，生成 patch_predictions.pkl
#   "eval"  = 复用预测缓存，执行原图合并与官方指标评估
#   "all"   = 推理 + 合并 + 官方评估
STAGE = "all"

REUSE_PATCH_CACHE = True
REUSE_MERGED_CACHE = True

# 模型权重。仅 STAGE="infer" 或 "all" 时使用。
MODEL_PATH = (
    "/workspace/FMC-OBB/ultralytics-main/Ours/Result/ablation/result/AI-TOD-R_A1_40e_seed42/weights/best.pt"
)

# 官方测试切片、切片清单、原图标注与官方评估器
PATCH_DIR = (
    "/workspace/FMC-OBB/ultralytics-main/Ours/datasets/SODA-A-OBB-OFF/test/images"
)
MANIFEST_PATH = (
    "/workspace/FMC-OBB/ultralytics-main/Ours/datasets/SODA-A-OBB-OFF/patch_manifest.csv"
)
ORIGINAL_ANN_DIR = (
    "/workspace/FMC-OBB/ultralytics-main/Ours/datasets/SODA-A/Annotations/test"
)
SODAA_EVAL_PATH = (
    "/workspace/FMC-OBB/ultralytics-main/"
    "SODA-mmrotate/mmrotate/datasets/sodaa_eval/sodaa_eval.py"
)

# 输出目录；缓存和最终指标均保存在这里
OUTPUT_DIR = (
    "/workspace/FMC-OBB/ultralytics-main/Ours/Result/ablation/test"
)

# 推理参数。注意：1200 不是 32 的整数倍，Ultralytics 会自动调整为 1216。
# Strict protocol: content is resized explicitly to 1200 before right/bottom padding.
IMAGE_SIZE = PADDED_INPUT_SIZE
BATCH_SIZE = 16
DEVICE = "0"
CONF_THRESHOLD = 0.001
PATCH_NMS_IOU = 0.7
MAX_DETECTIONS = 1000
# Official metric runs default to FP32. Whole-model forced FP16 can overflow
# frequency-mask activations and is not required for benchmark comparability.
USE_HALF = False

# 原图级旋转框合并与官方评估参数
MERGE_IOU = 0.5
EVAL_WORKERS = 4

# 缓存控制：
# 当前已经完成切片推理和原图 NMS 合并，因此 eval 阶段建议均设为 True。
# 想重新推理时将 REUSE_PATCH_CACHE=False。
# 想重新进行原图旋转 NMS 时将 REUSE_MERGED_CACHE=False。
REUSE_PATCH_CACHE = True
REUSE_MERGED_CACHE = True
# =============================================================================


def build_config() -> argparse.Namespace:
    """Build an argparse-compatible configuration from the constants above."""
    stage = str(STAGE).strip().lower()
    if stage not in {"infer", "eval", "all"}:
        raise ValueError(
            f"Invalid STAGE={STAGE!r}; expected 'infer', 'eval', or 'all'."
        )

    if IMAGE_SIZE <= 0:
        raise ValueError(f"IMAGE_SIZE must be positive, got {IMAGE_SIZE}.")
    if int(IMAGE_SIZE) != PADDED_INPUT_SIZE:
        raise ValueError(
            f"Strict official protocol requires IMAGE_SIZE={PADDED_INPUT_SIZE}, got {IMAGE_SIZE}. "
            "The effective 1200 content is prepared explicitly before inference."
        )
    if BATCH_SIZE <= 0:
        raise ValueError(f"BATCH_SIZE must be positive, got {BATCH_SIZE}.")
    if MAX_DETECTIONS <= 0:
        raise ValueError(
            f"MAX_DETECTIONS must be positive, got {MAX_DETECTIONS}."
        )
    if EVAL_WORKERS <= 0:
        raise ValueError(f"EVAL_WORKERS must be positive, got {EVAL_WORKERS}.")
    if not 0.0 <= CONF_THRESHOLD <= 1.0:
        raise ValueError(
            f"CONF_THRESHOLD must be in [0, 1], got {CONF_THRESHOLD}."
        )
    if not 0.0 <= PATCH_NMS_IOU <= 1.0:
        raise ValueError(
            f"PATCH_NMS_IOU must be in [0, 1], got {PATCH_NMS_IOU}."
        )
    if not 0.0 <= MERGE_IOU <= 1.0:
        raise ValueError(f"MERGE_IOU must be in [0, 1], got {MERGE_IOU}.")

    return argparse.Namespace(
        stage=stage,
        model=MODEL_PATH,
        patch_dir=PATCH_DIR,
        manifest=MANIFEST_PATH,
        ori_ann_dir=ORIGINAL_ANN_DIR,
        sodaa_eval=SODAA_EVAL_PATH,
        out=OUTPUT_DIR,
        imgsz=int(IMAGE_SIZE),
        batch=int(BATCH_SIZE),
        device=str(DEVICE),
        conf=float(CONF_THRESHOLD),
        iou=float(PATCH_NMS_IOU),
        max_det=int(MAX_DETECTIONS),
        half=bool(USE_HALF),
        merge_iou=float(MERGE_IOU),
        eval_workers=int(EVAL_WORKERS),
        reuse_patch_cache=bool(REUSE_PATCH_CACHE),
        reuse_merged_cache=bool(REUSE_MERGED_CACHE),
    )


def preprocess_official_patch(image: np.ndarray, patch_id: str = "<unknown>") -> np.ndarray:
    """Resize an official 800 patch to 1200 content and pad only right/bottom to 1216."""
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Patch {patch_id!r} must be an HWC three-channel image, got {getattr(image, 'shape', None)}")
    if image.shape[:2] != (PATCH_SOURCE_SIZE, PATCH_SOURCE_SIZE):
        raise ValueError(
            f"Official SODA-A patch {patch_id!r} must be 800x800, got {image.shape[1]}x{image.shape[0]}."
        )

    content = cv2.resize(
        image,
        (EFFECTIVE_CONTENT_SIZE, EFFECTIVE_CONTENT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    padded = cv2.copyMakeBorder(
        content,
        top=0,
        bottom=PADDED_INPUT_SIZE - EFFECTIVE_CONTENT_SIZE,
        left=0,
        right=PADDED_INPUT_SIZE - EFFECTIVE_CONTENT_SIZE,
        borderType=cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    if padded.shape[:2] != (PADDED_INPUT_SIZE, PADDED_INPUT_SIZE):
        raise RuntimeError(f"Unexpected preprocessed shape for {patch_id!r}: {padded.shape}")
    return padded


def restore_polygon_to_patch(polygon: np.ndarray) -> np.ndarray:
    """Map a polygon from 1200-content coordinates back to the official 800 patch."""
    polygon = np.asarray(polygon, dtype=np.float32)
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        raise ValueError(f"Expected one finite [4,2] polygon, got shape={polygon.shape}")
    return polygon / np.float32(PATCH_TO_CONTENT_SCALE)


def validate_protocol_cache(payload: dict[str, Any], cache_path: Path) -> None:
    """Reject caches produced by the legacy direct-1216 preprocessing path."""
    cached_protocol = payload.get("protocol_id")
    if cached_protocol != OFFICIAL_PROTOCOL_ID:
        raise RuntimeError(
            f"Cache protocol mismatch for {cache_path}: expected {OFFICIAL_PROTOCOL_ID!r}, "
            f"got {cached_protocol!r}. Delete the cache and rerun strict official inference."
        )


def validate_model_class_order(names: dict[int, str] | list[str] | tuple[str, ...]) -> None:
    """Require the model output indices to match the official SODA-A class order exactly."""
    if isinstance(names, dict):
        try:
            ordered_names = [str(names[index]) for index in range(len(names))]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Model class mapping must use contiguous integer indices: {names}") from exc
    else:
        ordered_names = [str(name) for name in names]
    if ordered_names != CLASS_NAMES:
        raise RuntimeError(
            "Model class order does not match the official SODA-A order. "
            f"Expected {CLASS_NAMES}, got {ordered_names}."
        )


def poly2obb_np_le90(poly: np.ndarray) -> np.ndarray | None:
    """
    MMRotate-compatible poly2obb_np(..., version='le90').

    Input:
        [x1,y1,x2,y2,x3,y3,x4,y4] or shape (4,2)
    Output:
        [cx, cy, w, h, angle] with angle in [-pi/2, pi/2).
    """
    pts = np.asarray(poly, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(pts).all():
        return None

    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(pts)
    if w < 2 or h < 2:
        return None

    angle = angle_deg / 180.0 * np.pi
    if w < h:
        w, h = h, w
        angle += np.pi / 2

    while not (-np.pi / 2 <= angle < np.pi / 2):
        if angle >= np.pi / 2:
            angle -= np.pi
        else:
            angle += np.pi

    return np.asarray([cx, cy, w, h, angle], dtype=np.float32)


def load_test_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {path}")

    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"split", "patch_id", "original_id", "x_start", "y_start"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")

        for row in reader:
            if row["split"].strip().lower() != "test":
                continue
            patch_id = row["patch_id"].strip()
            records[patch_id] = {
                "original_id": row["original_id"].strip(),
                "x_start": float(row["x_start"]),
                "y_start": float(row["y_start"]),
            }

    if not records:
        raise RuntimeError(f"No test records found in manifest: {path}")
    return records


def check_patch_files(patch_dir: Path, manifest: dict[str, dict[str, Any]]) -> None:
    if not patch_dir.is_dir():
        raise FileNotFoundError(f"Patch image directory does not exist: {patch_dir}")

    stems = {
        p.stem
        for p in patch_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    }
    missing = sorted(set(manifest) - stems)
    extra = sorted(stems - set(manifest))

    print(f"[DATA] test manifest patches: {len(manifest)}")
    print(f"[DATA] test patch images:     {len(stems)}")
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} manifest patches have no image. Examples: {missing[:20]}"
        )
    if extra:
        print(f"[WARN] {len(extra)} patch images are absent from the test manifest and will be ignored.")


def run_inference(args: argparse.Namespace, patch_cache: Path) -> None:
    if not args.model:
        raise ValueError("--model is required for --stage infer/all")

    model_path = Path(args.model).expanduser().resolve()
    patch_dir = Path(args.patch_dir).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model does not exist: {model_path}")

    manifest = load_test_manifest(manifest_path)
    check_patch_files(patch_dir, manifest)

    try:
        from ultralytics import RTDETR
    except Exception as exc:
        raise RuntimeError(
            "Could not import the custom Ultralytics RTDETR implementation."
        ) from exc

    model = RTDETR(str(model_path))
    validate_model_class_order(model.names)

    image_paths = {
        path.stem: path
        for path in patch_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    }
    ordered_patches = [(patch_id, image_paths[patch_id]) for patch_id in sorted(manifest) if patch_id in image_paths]
    if len(ordered_patches) != len(manifest):
        raise RuntimeError(
            f"Strict inference resolved {len(ordered_patches)} images for {len(manifest)} manifest patches."
        )

    # collector[original_id][class_id] -> list[np.ndarray(N, 6)]
    # Each row is [cx, cy, w, h, angle, score] in original-image coordinates.
    collector: dict[str, dict[int, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )

    start = time.time()
    processed = 0
    detections = 0

    progress = tqdm(total=len(ordered_patches), desc="Strict 1200+Pad32 patch inference")
    try:
        for batch_start in range(0, len(ordered_patches), args.batch):
            batch_entries = ordered_patches[batch_start : batch_start + args.batch]
            prepared_images = []
            for patch_id, image_path in batch_entries:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(f"Could not read patch image: {image_path}")
                prepared_images.append(preprocess_official_patch(image, patch_id=patch_id))

            batch_results = model.predict(
                source=prepared_images,
                imgsz=PADDED_INPUT_SIZE,
                batch=len(prepared_images),
                device=args.device,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                quantize=16 if args.half else 32,
                stream=False,
                save=False,
                verbose=False,
            )
            if len(batch_results) != len(batch_entries):
                raise RuntimeError(
                    f"Prediction batch returned {len(batch_results)} results for {len(batch_entries)} patches."
                )

            for result, (patch_id, _) in zip(batch_results, batch_entries):
                meta = manifest[patch_id]
                processed += 1
                obb = getattr(result, "obb", None)
                if obb is None or len(obb) == 0:
                    continue
                if not hasattr(obb, "xyxyxyxy"):
                    raise AttributeError(
                        "Results.obb does not provide xyxyxyxy. "
                        "Check that the custom predictor returns a true OBB Results object."
                    )

                polygons = obb.xyxyxyxy.detach().cpu().numpy().astype(np.float32)
                scores = obb.conf.detach().cpu().numpy().astype(np.float32)
                classes = obb.cls.detach().cpu().numpy().astype(np.int64)
                if not np.isfinite(polygons).all() or not np.isfinite(scores).all():
                    raise FloatingPointError(f"Non-finite OBB prediction detected for patch {patch_id!r}.")

                for poly, score, class_id in zip(polygons, scores, classes):
                    if not 0 <= int(class_id) < NUM_CLASSES:
                        continue
                    center = poly.mean(axis=0)
                    if not (0.0 <= center[0] < EFFECTIVE_CONTENT_SIZE and 0.0 <= center[1] < EFFECTIVE_CONTENT_SIZE):
                        continue

                    patch_poly = restore_polygon_to_patch(poly)
                    original_poly = patch_poly.copy()
                    original_poly[:, 0] += meta["x_start"]
                    original_poly[:, 1] += meta["y_start"]

                    rbbox = poly2obb_np_le90(original_poly)
                    if rbbox is None:
                        continue
                    row = np.concatenate([rbbox, np.asarray([score], dtype=np.float32)])[None, :]
                    collector[meta["original_id"]][int(class_id)].append(row)
                    detections += 1
            progress.update(len(batch_entries))
    finally:
        progress.close()

    serializable: dict[str, dict[int, np.ndarray]] = {}
    for original_id, class_map in collector.items():
        serializable[original_id] = {}
        for class_id, arrays in class_map.items():
            serializable[original_id][class_id] = (
                np.concatenate(arrays, axis=0)
                if arrays
                else np.zeros((0, 6), dtype=np.float32)
            )

    payload = {
        "protocol_id": OFFICIAL_PROTOCOL_ID,
        "class_names": CLASS_NAMES,
        "model": str(model_path),
        "imgsz": args.imgsz,
        "patch_source_size": PATCH_SOURCE_SIZE,
        "effective_content_size": EFFECTIVE_CONTENT_SIZE,
        "padded_input_size": PADDED_INPUT_SIZE,
        "patch_to_content_scale": PATCH_TO_CONTENT_SCALE,
        "conf": args.conf,
        "max_det": args.max_det,
        "manifest": str(manifest_path),
        "predictions": serializable,
    }
    with patch_cache.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - start
    print(f"[INFER] processed patches: {processed}")
    print(f"[INFER] mapped detections: {detections}")
    print(f"[INFER] elapsed: {elapsed:.1f} s")
    print(f"[INFER] cache: {patch_cache}")


def _load_importable_module_from_file(path: Path):
    """Load a Python file under an importable module name for spawn workers.

    ``multiprocessing`` with the ``spawn`` start method pickles top-level
    functions by ``module_name + function_name``.  Therefore a temporary name
    such as ``official_sodaa_eval`` is unsafe unless every child process can
    import that name.  Registering the evaluator as ``sodaa_eval`` and adding
    its directory to ``sys.path`` makes the module importable in both the
    parent process and spawned workers.
    """
    path = path.expanduser().resolve()
    module_name = path.stem  # Expected to be "sodaa_eval".
    module_dir = str(path.parent)

    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    importlib.invalidate_caches()

    # Reuse only the exact same module file.  Remove a stale same-name module
    # that may have been imported from another MMRotate installation.
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        try:
            same_file = (
                existing_file is not None
                and Path(existing_file).expanduser().resolve() == path
            )
        except (OSError, RuntimeError, TypeError):
            same_file = False

        if same_file:
            return existing
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for: {path}")

    module = importlib.util.module_from_spec(spec)

    # This registration MUST happen before exec_module().  It is required for
    # correct pickling and also supports imports performed while the evaluator
    # module itself is being executed.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return module


def import_official_sodaa_eval(path_arg: str | None):
    candidates: list[Path] = []

    if path_arg:
        candidates.append(Path(path_arg).expanduser().resolve())

    env_path = os.environ.get("SODAA_EVAL_PY")
    if env_path:
        candidates.append(Path(env_path).expanduser().resolve())

    candidates.extend(
        [
            Path(
                "/workspace/FMC-OBB/ultralytics-main/"
                "SODA-mmrotate/mmrotate/datasets/sodaa_eval/sodaa_eval.py"
            ),
            Path(
                "/workspace/FMC-OBB/"
                "SODA-mmrotate/mmrotate/datasets/sodaa_eval/sodaa_eval.py"
            ),
            Path.cwd()
            / "SODA-mmrotate/mmrotate/datasets/sodaa_eval/sodaa_eval.py",
        ]
    )

    # Remove duplicate candidate paths while preserving their priority.
    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)

    for path in unique_candidates:
        if not path.is_file():
            continue

        module = _load_importable_module_from_file(path)
        if not hasattr(module, "SODAAeval"):
            raise AttributeError(f"SODAAeval class not found in: {path}")

        # Fail early with a clear message instead of crashing after IoU tasks
        # have already been prepared.  The official evaluator currently uses
        # this function as the Pool initializer.
        initializer = getattr(module, "init_worker_with_counter", None)
        if initializer is not None:
            try:
                pickle.dumps(initializer)
            except Exception as exc:
                raise RuntimeError(
                    "Official evaluator worker initializer is not pickleable. "
                    f"module={module.__name__!r}, file={path}"
                ) from exc

        print(f"[EVAL] official evaluator: {path}")
        print(f"[EVAL] importable module:  {module.__name__}")
        print(f"[EVAL] module directory:   {path.parent}")
        if initializer is not None:
            print(
                "[EVAL] spawn initializer: "
                f"{initializer.__module__}.{initializer.__name__}"
            )
        return module.SODAAeval

    try:
        from mmrotate.datasets.sodaa_eval.sodaa_eval import SODAAeval

        print("[EVAL] official evaluator imported from installed mmrotate")
        return SODAAeval
    except Exception as exc:
        locations = "\n".join(f"  - {p}" for p in unique_candidates)
        raise FileNotFoundError(
            "Could not locate the official sodaa_eval.py.\n"
            "Clone the official repository, then pass --sodaa-eval.\n"
            f"Tried:\n{locations}"
        ) from exc


def load_original_gt(ann_dir: Path):
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Original test annotation directory does not exist: {ann_dir}")

    annotations_by_id: dict[str, dict[str, np.ndarray]] = {}
    skipped_ignore = 0
    skipped_invalid = 0

    for json_path in sorted(ann_dir.rglob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8-sig"))
        anns = data.get("annotations", [])
        if not isinstance(anns, list):
            raise ValueError(f"'annotations' is not a list: {json_path}")

        bboxes = []
        labels = []

        for ann in anns:
            category_value = ann.get("category_id", ann.get("cat_id"))
            try:
                category_id = int(category_value)
            except (TypeError, ValueError):
                skipped_invalid += 1
                continue

            if category_id == 9:
                skipped_ignore += 1
                continue
            if not 0 <= category_id < NUM_CLASSES:
                skipped_invalid += 1
                continue

            poly = np.asarray(ann.get("poly", []), dtype=np.float32).reshape(-1)
            if poly.size != 8 or not np.isfinite(poly).all():
                skipped_invalid += 1
                continue

            rbbox = poly2obb_np_le90(poly)
            if rbbox is None:
                skipped_invalid += 1
                continue

            bboxes.append(rbbox)
            labels.append(category_id)

        annotations_by_id[json_path.stem] = {
            "bboxes": (
                np.asarray(bboxes, dtype=np.float32).reshape(-1, 5)
                if bboxes
                else np.zeros((0, 5), dtype=np.float32)
            ),
            "labels": np.asarray(labels, dtype=np.int64),
        }

    if not annotations_by_id:
        raise RuntimeError(f"No original test annotations parsed from: {ann_dir}")

    print(f"[GT] original images: {len(annotations_by_id)}")
    print(f"[GT] ignore annotations skipped: {skipped_ignore}")
    print(f"[GT] invalid annotations skipped: {skipped_invalid}")
    return annotations_by_id


def merge_patch_predictions(
    patch_payload: dict[str, Any],
    original_ids: list[str],
    merge_iou: float,
):
    validate_protocol_cache(patch_payload, Path("<in-memory patch cache>"))
    if patch_payload.get("class_names") != CLASS_NAMES:
        raise RuntimeError(
            "Patch cache class order does not match the official SODA-A nine-class order."
        )
    try:
        import torch
        from mmcv.ops import nms_rotated
    except Exception as exc:
        raise RuntimeError(
            "Official merging requires working MMCV compiled ops. "
            "Test with: python -c \"from mmcv.ops import nms_rotated, box_iou_rotated; print('OK')\""
        ) from exc

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    patch_predictions = patch_payload["predictions"]
    merged_results = []

    total_before = 0
    total_after = 0

    for original_id in tqdm(original_ids, desc="Official rotated merge"):
        per_class = []
        source_map = patch_predictions.get(original_id, {})

        for class_id in range(NUM_CLASSES):
            dets = source_map.get(class_id)
            if dets is None or len(dets) == 0:
                per_class.append(np.zeros((0, 6), dtype=np.float32))
                continue

            dets = np.asarray(dets, dtype=np.float32).reshape(-1, 6)
            total_before += len(dets)

            boxes = torch.from_numpy(dets[:, :5]).to(device).contiguous()
            scores = torch.from_numpy(dets[:, 5]).to(device).contiguous()

            nms_output, _ = nms_rotated(boxes, scores, merge_iou)
            merged = nms_output.detach().cpu().numpy().astype(np.float32)
            per_class.append(merged)
            total_after += len(merged)

        merged_results.append(per_class)

    print(f"[MERGE] detections before NMS: {total_before}")
    print(f"[MERGE] detections after NMS:  {total_after}")
    return merged_results


def per_class_metrics(eval_obj) -> list[dict[str, Any]]:
    precision = eval_obj.eval["precision"]
    rows = []

    for class_id, class_name in enumerate(CLASS_NAMES):
        all_ap = precision[:, :, class_id, 0, -1]
        all_ap = all_ap[all_ap > -1]
        ap = float(np.mean(all_ap)) if all_ap.size else float("nan")

        ap50_values = precision[0, :, class_id, 0, -1]
        ap50_values = ap50_values[ap50_values > -1]
        ap50 = (
            float(np.mean(ap50_values))
            if ap50_values.size
            else float("nan")
        )

        ap75_index = int(np.where(np.isclose(eval_obj.params.iouThrs, 0.75))[0][0])
        ap75_values = precision[ap75_index, :, class_id, 0, -1]
        ap75_values = ap75_values[ap75_values > -1]
        ap75 = (
            float(np.mean(ap75_values))
            if ap75_values.size
            else float("nan")
        )

        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "AP": ap,
                "AP50": ap50,
                "AP75": ap75,
            }
        )
    return rows


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def save_metrics(
    out_dir: Path,
    eval_obj,
    original_ids: list[str],
    per_class_rows: list[dict[str, Any]],
) -> None:
    names = [
        "AP",
        "AP50",
        "AP75",
        "APeS",
        "APrS",
        "APgS",
        "APN",
        "AR",
        "AReS",
        "ARrS",
        "ARgS",
        "ARN",
    ]
    stats = [float(x) for x in eval_obj.stats]
    metrics = dict(zip(names, stats))

    payload = {
        "official_metrics": metrics,
        "official_metrics_percent": {
            key: value * 100.0 for key, value in metrics.items()
        },
        "area_ranges": {
            "AP/AP50/AP75": "[0, 32^2]",
            "APeS": "[0, 12^2]",
            "APrS": "[12^2, 20^2]",
            "APgS": "[20^2, 32^2]",
            "APN": "[32^2, 40*50]",
        },
        "num_original_test_images": len(original_ids),
        "class_names": CLASS_NAMES,
        "per_class": per_class_rows,
    }

    (out_dir / "official_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (out_dir / "official_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["metric", "value", "percent"],
        )
        writer.writeheader()
        for key in names:
            writer.writerow(
                {
                    "metric": key,
                    "value": metrics[key],
                    "percent": metrics[key] * 100.0,
                }
            )

    with (out_dir / "per_class_ap.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["class_id", "class_name", "AP", "AP50", "AP75"],
        )
        writer.writeheader()
        writer.writerows(per_class_rows)

    print("\nOfficial SODA-A metrics")
    print("-" * 72)
    print(
        f"{'AP':>8} {'AP50':>8} {'AP75':>8} "
        f"{'APeS':>8} {'APrS':>8} {'APgS':>8} {'APN':>8}"
    )
    print(
        f"{metrics['AP'] * 100:8.3f} "
        f"{metrics['AP50'] * 100:8.3f} "
        f"{metrics['AP75'] * 100:8.3f} "
        f"{metrics['APeS'] * 100:8.3f} "
        f"{metrics['APrS'] * 100:8.3f} "
        f"{metrics['APgS'] * 100:8.3f} "
        f"{metrics['APN'] * 100:8.3f}"
    )
    print("-" * 72)
    print(f"Saved: {out_dir / 'official_metrics.json'}")
    print(f"Saved: {out_dir / 'official_metrics.csv'}")
    print(f"Saved: {out_dir / 'per_class_ap.csv'}")


def run_evaluation(
    args: argparse.Namespace,
    patch_cache: Path,
    merged_cache: Path,
) -> None:
    """Run original-image-level SODA-A evaluation.

    If ``REUSE_MERGED_CACHE=True`` and ``merged_predictions.pkl`` exists,
    evaluation proceeds directly from that cache. In this case,
    ``patch_predictions.pkl`` is not required.

    The patch cache is required only when rotated NMS merging must be rebuilt.
    """
    out_dir = Path(args.out).expanduser().resolve()
    ann_dir = Path(args.ori_ann_dir).expanduser().resolve()

    gt_map = load_original_gt(ann_dir)
    original_ids = sorted(gt_map.keys())
    annotations = [gt_map[image_id] for image_id in original_ids]

    if args.reuse_merged_cache and merged_cache.exists():
        with merged_cache.open("rb") as file:
            merged_payload = pickle.load(file)
        validate_protocol_cache(merged_payload, merged_cache)

        cached_ids = merged_payload.get("original_ids")
        if cached_ids != original_ids:
            raise RuntimeError(
                "Merged cache original-image order does not match current "
                "test annotations. Delete the merged cache or set "
                "REUSE_MERGED_CACHE=False to rebuild it."
            )

        merged_results = merged_payload.get("results")
        if merged_results is None:
            raise KeyError(
                f"Merged cache has no 'results' field: {merged_cache}"
            )

        if len(merged_results) != len(original_ids):
            raise RuntimeError(
                "Merged cache result count does not match the number of "
                f"original test images: {len(merged_results)} != "
                f"{len(original_ids)}."
            )

        cached_iou = merged_payload.get("merge_iou")
        if cached_iou is not None and not np.isclose(
            float(cached_iou), float(args.merge_iou)
        ):
            print(
                "[WARN] Reusing merged cache created with merge_iou="
                f"{cached_iou}, while current MERGE_IOU={args.merge_iou}. "
                "The cached value is the one actually used."
            )

        print(f"[MERGE] reused cache: {merged_cache}")

    else:
        if not patch_cache.exists():
            if args.reuse_merged_cache:
                reason = (
                    f"merged cache does not exist: {merged_cache}"
                )
            else:
                reason = "REUSE_MERGED_CACHE=False"

            raise FileNotFoundError(
                f"Patch prediction cache does not exist: {patch_cache}. "
                f"Rotated merging is required because {reason}. "
                "Set STAGE='infer' or STAGE='all' to regenerate the patch "
                "cache, or restore merged_predictions.pkl and set "
                "REUSE_MERGED_CACHE=True."
            )

        with patch_cache.open("rb") as file:
            patch_payload = pickle.load(file)
        validate_protocol_cache(patch_payload, patch_cache)

        if "predictions" not in patch_payload:
            raise KeyError(
                f"Patch cache has no 'predictions' field: {patch_cache}"
            )

        merged_results = merge_patch_predictions(
            patch_payload,
            original_ids,
            args.merge_iou,
        )
        with merged_cache.open("wb") as file:
            pickle.dump(
                {
                    "protocol_id": OFFICIAL_PROTOCOL_ID,
                    "class_names": CLASS_NAMES,
                    "original_ids": original_ids,
                    "merge_iou": args.merge_iou,
                    "results": merged_results,
                },
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"[MERGE] cache: {merged_cache}")

    SODAAeval = import_official_sodaa_eval(args.sodaa_eval)

    # The official summarizer writes ./work_dirs/evalRes.txt.
    (out_dir / "work_dirs").mkdir(parents=True, exist_ok=True)

    evaluator = SODAAeval(
        annotations=annotations,
        results=merged_results,
        numCats=NUM_CLASSES,
        nproc=args.eval_workers,
    )
    evaluator.params.iouThrs = np.linspace(
        0.50,
        0.95,
        int(round((0.95 - 0.50) / 0.05)) + 1,
        endpoint=True,
    )

    with working_directory(out_dir):
        evaluator.evaluate()

        # Re-assert aliases immediately before the legacy NumPy-based
        # accumulation code. This fixes NumPy 2.x compatibility.
        install_numpy_legacy_aliases()
        evaluator.accumulate()
        evaluator.summarize()

    class_rows = per_class_metrics(evaluator)
    save_metrics(out_dir, evaluator, original_ids, class_rows)

def main() -> None:
    args = build_config()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_cache = out_dir / "patch_predictions.pkl"
    merged_cache = out_dir / "merged_predictions.pkl"

    print("=" * 80)
    print("Official SODA-A original-image-level test")
    print("=" * 80)
    print(f"numpy version:       {np.__version__}")
    if NUMPY_COMPAT_ALIASES:
        print(
            "numpy compatibility: restored "
            + ", ".join(f"np.{name}" for name in NUMPY_COMPAT_ALIASES)
        )
    else:
        print("numpy compatibility: native aliases available")
    print(f"stage:              {args.stage}")
    print(f"model:              {Path(args.model).expanduser().resolve()}")
    print(f"patch dir:          {Path(args.patch_dir).expanduser().resolve()}")
    print(f"manifest:           {Path(args.manifest).expanduser().resolve()}")
    print(f"original GT:        {Path(args.ori_ann_dir).expanduser().resolve()}")
    print(f"official evaluator: {Path(args.sodaa_eval).expanduser().resolve()}")
    print(f"output:             {out_dir}")
    print(f"protocol:           {OFFICIAL_PROTOCOL_ID}")
    print(
        f"preprocess:         {PATCH_SOURCE_SIZE} -> {EFFECTIVE_CONTENT_SIZE} content -> "
        f"right/bottom pad {PADDED_INPUT_SIZE}"
    )
    print(f"imgsz / batch:      {args.imgsz} / {args.batch}")
    print(f"device / half:      {args.device} / {args.half}")
    print(f"conf / max_det:     {args.conf} / {args.max_det}")
    print(f"patch / merge IoU:  {args.iou} / {args.merge_iou}")
    print(f"eval workers:       {args.eval_workers}")
    print(f"reuse patch cache:  {args.reuse_patch_cache}")
    print(f"reuse merged cache: {args.reuse_merged_cache}")

    if args.stage in {"infer", "all"}:
        if args.reuse_patch_cache and patch_cache.exists():
            with patch_cache.open("rb") as file:
                validate_protocol_cache(pickle.load(file), patch_cache)
            print(f"[INFER] reused cache: {patch_cache}")
        else:
            run_inference(args, patch_cache)

    if args.stage in {"eval", "all"}:
        run_evaluation(args, patch_cache, merged_cache)


if __name__ == "__main__":
    main()
