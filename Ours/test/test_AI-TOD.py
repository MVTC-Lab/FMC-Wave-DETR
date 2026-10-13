#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Official AI-TOD-R evaluation for an Ultralytics FMC-Wave RT-DETR-OBB model.

Pipeline:
last.pt inference on 800x800 test images
    -> convert predicted polygons to OpenCV-style OBBs
    -> build the official AI-TOD-R result structure
    -> run the official AITODReval evaluator
    -> save AP/AP50/AP75/APvt/APt/APs/APm/AR and class-wise AP

Dependencies:
- The user's local Ultralytics project.
- OpenCV, NumPy, PyTorch.
- MMCV with mmcv.ops.box_iou_rotated available.
- A cloned official AI-TOD-R repository containing:
  mmrotate/datasets/aitodr_eval/aitodr_eval.py

The official evaluator uses maxDets=1500. The current model has 1000 queries,
so predictions cannot exceed 1000 even though max_det is set to 1500.
"""

from __future__ import annotations

import csv
import gc
import importlib.util
import json
import os
import pickle
import sys
import time

# Reduce allocator fragmentation when predictor input shapes/batches change.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths: edit only this block when your local directories differ
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/workspace/FMC-OBB/ultralytics-main").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "Ours/datasets/AI-TOD"

TEST_IMAGE_DIR = DATA_ROOT / "test/images"

# Official DOTA-style labels:
# x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
TEST_LABEL_DIR = DATA_ROOT / "labels"

WEIGHTS = (
    PROJECT_ROOT
    / "Ours/Result/AI-TOD-R/AI-TOD-R_FMC-OBB_100e_FP32/weights/best.pt"
)

# Clone the official repository once, for example:
# git clone https://github.com/Chasel-Tsui/AI-TOD-R.git \
#   /workspace/FMC-OBB/AI-TOD-R-official
OFFICIAL_REPO = Path("/workspace/FMC-OBB/ultralytics-main").resolve()
OFFICIAL_EVAL_FILE = (
    OFFICIAL_REPO
    / "Ours/test/aitodr_eval_official.py"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "Ours/Predict/AI-TOD-R_FMC-OBB"
)

# "all": inference + evaluation
# "infer": inference only
# "eval": load cached predictions and evaluate only
STAGE = "all"

DEVICE = "0"
INFER_BATCH = 1
IMGSZ = 800
CONF = 0.001
MAX_DET = 1500
EVAL_WORKERS = 4

# Process the large test set in visible, resumable chunks.
CHUNK_SIZE = 64
RUN_SMOKE_TEST = True
RESUME_INFERENCE = True

CLASSES = (
    "airplane",
    "bridge",
    "storage-tank",
    "ship",
    "swimming-pool",
    "vehicle",
    "person",
    "wind-mill",
)
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

PREDICTION_CACHE = OUTPUT_DIR / "predictions_official.pkl"
PARTIAL_CACHE = OUTPUT_DIR / "predictions_partial.pkl"
METRICS_JSON = OUTPUT_DIR / "official_metrics.json"
CLASSWISE_CSV = OUTPUT_DIR / "classwise_ap.csv"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def import_official_evaluator(eval_file: Path):
    """Import the official AI-TOD-R evaluator as a normal Python module."""

    if not eval_file.exists():
        raise FileNotFoundError(
            f"Official evaluator was not found: {eval_file}"
        )

    try:
        from mmcv.ops import box_iou_rotated  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "The official evaluator requires "
            "mmcv.ops.box_iou_rotated."
        ) from exc

    eval_dir = str(eval_file.parent)

    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)

    import importlib

    module = importlib.import_module(
        "aitodr_eval_official"
    )

    evaluator_class = getattr(
        module,
        "AITODReval",
        None,
    )

    if evaluator_class is None:
        raise AttributeError(
            f"{eval_file} does not define AITODReval."
        )

    print(
        "[Evaluator] loaded:",
        evaluator_class,
    )

    return evaluator_class

def poly_to_obb_oc(poly: np.ndarray) -> np.ndarray | None:
    """
    Convert four polygon points to an OpenCV-style rotated box.

    Returns:
        [cx, cy, w, h, angle_radian], angle in [0, pi/2).

    GT and predictions are passed through the same conversion, preventing
    an angle-convention mismatch between Ultralytics and the evaluator.
    """
    points = np.asarray(poly, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(points).all():
        return None

    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(points)

    if w < 1.0e-4 or h < 1.0e-4:
        return None

    # OpenCV versions may return [-90, 0) or [0, 90).
    # Normalize to the AI-TOD-R/MMRotate OpenCV interval [0, 90).
    if angle_deg < 0.0:
        angle_deg += 90.0
        w, h = h, w

    if angle_deg >= 90.0 - 1.0e-6:
        angle_deg = 0.0
        w, h = h, w

    angle_rad = float(np.deg2rad(angle_deg))
    return np.asarray([cx, cy, w, h, angle_rad], dtype=np.float32)


def find_test_images(image_dir: Path) -> list[Path]:
    images = sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No test images were found in {image_dir}")
    return images


def parse_official_annotation(label_file: Path) -> dict[str, np.ndarray]:
    bboxes: list[np.ndarray] = []
    labels: list[int] = []

    if not label_file.exists():
        return {
            "bboxes": np.zeros((0, 5), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
        }

    with label_file.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            parts = raw_line.strip().split()
            if not parts:
                continue
            if len(parts) < 10:
                raise ValueError(
                    f"Invalid annotation at {label_file}:{line_number}; "
                    f"expected at least 10 fields, got {len(parts)}."
                )

            poly = np.asarray(parts[:8], dtype=np.float32)
            class_name = parts[8]
            difficulty = int(float(parts[9]))

            # The official AITODRDataset uses difficulty=100 as the threshold.
            if difficulty > 100:
                continue

            if class_name not in CLASS_TO_ID:
                raise KeyError(
                    f"Unknown class {class_name!r} at "
                    f"{label_file}:{line_number}. Expected {CLASSES}."
                )

            obb = poly_to_obb_oc(poly)
            if obb is None:
                continue

            bboxes.append(obb)
            labels.append(CLASS_TO_ID[class_name])

    return {
        "bboxes": (
            np.stack(bboxes).astype(np.float32)
            if bboxes else np.zeros((0, 5), dtype=np.float32)
        ),
        "labels": np.asarray(labels, dtype=np.int64),
    }


def build_annotations(image_paths: list[Path]) -> list[dict[str, np.ndarray]]:
    annotations: list[dict[str, np.ndarray]] = []
    missing_labels: list[str] = []

    for image_path in image_paths:
        label_path = TEST_LABEL_DIR / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels.append(image_path.name)
        annotations.append(parse_official_annotation(label_path))

    if missing_labels:
        print(
            f"Warning: {len(missing_labels)} test images have no label file. "
            "They will be evaluated as empty-GT images."
        )

    return annotations


def verify_model_classes(model: Any) -> None:
    names = model.names
    if isinstance(names, dict):
        model_classes = tuple(str(names[i]) for i in sorted(names))
    else:
        model_classes = tuple(str(x) for x in names)

    if model_classes != CLASSES:
        raise ValueError(
            "Model class order does not match the official AI-TOD-R order.\n"
            f"model:    {model_classes}\n"
            f"official: {CLASSES}\n"
            "Fix data-obb.yaml before training/testing."
        )


def empty_image_result() -> list[np.ndarray]:
    return [np.zeros((0, 6), dtype=np.float32) for _ in CLASSES]


def force_obb_task(model: Any) -> None:
    """Repair missing task metadata in a custom RT-DETR-OBB checkpoint."""
    model.task = "obb"

    if not hasattr(model, "overrides") or model.overrides is None:
        model.overrides = {}
    model.overrides["task"] = "obb"

    core = getattr(model, "model", None)
    if core is not None:
        try:
            core.task = "obb"
        except Exception:
            pass

        args = getattr(core, "args", None)
        if isinstance(args, dict):
            args["task"] = "obb"
        elif args is not None:
            try:
                setattr(args, "task", "obb")
            except Exception:
                pass

    # Discard an incorrectly cached detect predictor, if present.
    if hasattr(model, "predictor"):
        model.predictor = None


def result_to_official(result: Any) -> list[np.ndarray]:
    """Convert one Ultralytics OBB result to the official per-class format."""
    if not hasattr(result, "obb") or result.obb is None:
        raise RuntimeError(
            "The prediction did not return result.obb. The checkpoint is "
            "still being routed through a detect predictor, or the custom "
            "RT-DETR-OBB predictor/task_map is not registered.\n"
            f"Image: {getattr(result, 'path', '<unknown>')}"
        )

    per_class_lists: list[list[np.ndarray]] = [[] for _ in CLASSES]

    if len(result.obb) > 0:
        polygons = result.obb.xyxyxyxy.detach().cpu().numpy()
        scores = result.obb.conf.detach().cpu().numpy()
        class_ids = result.obb.cls.detach().cpu().numpy().astype(np.int64)

        for poly, score, class_id in zip(polygons, scores, class_ids):
            if class_id < 0 or class_id >= len(CLASSES):
                raise IndexError(
                    f"Invalid predicted class id {class_id} for "
                    f"{Path(result.path).name}"
                )
            if not np.isfinite(score):
                continue

            obb = poly_to_obb_oc(poly)
            if obb is None:
                continue

            det = np.concatenate(
                [obb, np.asarray([score], dtype=np.float32)]
            ).astype(np.float32)
            per_class_lists[class_id].append(det)

    per_class_result: list[np.ndarray] = []
    for detections in per_class_lists:
        if detections:
            array = np.stack(detections).astype(np.float32)
            order = np.argsort(-array[:, -1], kind="mergesort")
            array = array[order]
        else:
            array = np.zeros((0, 6), dtype=np.float32)
        per_class_result.append(array)

    return per_class_result


def prediction_kwargs(batch: int) -> dict[str, Any]:
    """Prediction settings. FP32 is the default; no deprecated half flag."""
    return {
        "task": "obb",
        "imgsz": IMGSZ,
        "batch": batch,
        "device": DEVICE,
        "conf": CONF,
        "iou": 0.7,
        "max_det": MAX_DET,
        "augment": False,
        "save": False,
        "save_txt": False,
        "save_conf": False,
        "verbose": False,
    }


def save_partial_predictions(
    predictions_by_stem: dict[str, list[np.ndarray]],
) -> None:
    """Atomically save progress so interrupted inference can resume."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = PARTIAL_CACHE.with_suffix(".tmp")
    with temp_path.open("wb") as f:
        pickle.dump(
            {
                "weights": str(WEIGHTS.resolve()),
                "imgsz": IMGSZ,
                "conf": CONF,
                "max_det": MAX_DET,
                "predictions_by_stem": predictions_by_stem,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    temp_path.replace(PARTIAL_CACHE)


def load_partial_predictions(
    image_paths: list[Path],
) -> dict[str, list[np.ndarray]]:
    if not RESUME_INFERENCE or not PARTIAL_CACHE.exists():
        return {}

    with PARTIAL_CACHE.open("rb") as f:
        payload = pickle.load(f)

    compatible = (
        payload.get("weights") == str(WEIGHTS.resolve())
        and payload.get("imgsz") == IMGSZ
        and payload.get("conf") == CONF
        and payload.get("max_det") == MAX_DET
    )
    if not compatible:
        print("[Resume] partial cache settings differ; starting fresh.")
        return {}

    valid_stems = {p.stem for p in image_paths}
    loaded = {
        stem: value
        for stem, value in payload.get("predictions_by_stem", {}).items()
        if stem in valid_stems
    }
    print(
        f"[Resume] loaded {len(loaded)}/{len(image_paths)} completed images "
        f"from {PARTIAL_CACHE}"
    )
    return loaded


def run_inference(image_paths: list[Path]) -> list[list[np.ndarray]]:
    from ultralytics import RTDETR

    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {WEIGHTS}")

    print("[Model] loading checkpoint and forcing task=obb...")
    load_start = time.perf_counter()
    try:
        model = RTDETR(str(WEIGHTS), task="obb")
    except TypeError:
        # Compatibility fallback for a local wrapper with an older signature.
        model = RTDETR(str(WEIGHTS))

    force_obb_task(model)
    verify_model_classes(model)

    core = getattr(model, "model", None)
    print(f"[Model] loaded in {time.perf_counter() - load_start:.2f}s")
    print(f"[Model] wrapper task: {getattr(model, 'task', None)}")
    print(f"[Model] core task:    {getattr(core, 'task', None)}")
    print(f"[Model] core class:   {type(core).__name__}")

    # One-image smoke test: fail early instead of appearing stuck on 14,018 images.
    if RUN_SMOKE_TEST:
        print("\n[Smoke Test] one-image OBB inference:", image_paths[0])
        smoke_start = time.perf_counter()
        smoke_results = model.predict(
            source=str(image_paths[0]),
            **prediction_kwargs(batch=1),
        )
        if not smoke_results or len(smoke_results) != 1:
            raise RuntimeError(
                f"Smoke test expected one Results object, got "
                f"{len(smoke_results) if smoke_results is not None else 0}."
            )
        smoke_official = result_to_official(smoke_results[0])
        smoke_count = sum(len(x) for x in smoke_official)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(
            f"[Smoke Test] PASS: detections={smoke_count}, "
            f"time={time.perf_counter() - smoke_start:.2f}s\n"
        )

    predictions_by_stem = load_partial_predictions(image_paths)
    pending_paths = [
        p for p in image_paths if p.stem not in predictions_by_stem
    ]

    total_images = len(image_paths)
    print("=" * 88)
    print("[Inference] chunked inference started")
    print(f"[Inference] total:      {total_images}")
    print(f"[Inference] completed:  {len(predictions_by_stem)}")
    print(f"[Inference] remaining:  {len(pending_paths)}")
    print(f"[Inference] chunk size: {CHUNK_SIZE}")
    print(f"[Inference] batch size: {INFER_BATCH}")
    print("=" * 88)

    run_start = time.perf_counter()
    newly_completed = 0

    for chunk_number, chunk_start in enumerate(
        range(0, len(pending_paths), CHUNK_SIZE),
        start=1,
    ):
        chunk_paths = pending_paths[chunk_start:chunk_start + CHUNK_SIZE]
        chunk_time = time.perf_counter()
        print(
            f"\n[Inference] chunk {chunk_number}: "
            f"{len(predictions_by_stem) + 1}-"
            f"{len(predictions_by_stem) + len(chunk_paths)}/{total_images}"
        )

        # Custom predictors may replace Results.path with synthetic names
        # such as image0, image1, ... when source is a Python list. Results are
        # still returned in the same order as the input paths, so bind them by
        # index instead of trusting result.path.
        result_stream = model.predict(
            source=[str(p) for p in chunk_paths],
            stream=True,
            **prediction_kwargs(batch=INFER_BATCH),
        )

        returned_count = 0
        for result_index, result in enumerate(result_stream):
            if result_index >= len(chunk_paths):
                raise RuntimeError(
                    "Predictor returned more results than input images: "
                    f"inputs={len(chunk_paths)}."
                )

            expected_path = chunk_paths[result_index]
            expected_stem = expected_path.stem

            if expected_stem in predictions_by_stem:
                raise RuntimeError(
                    f"Duplicate prediction assignment: {expected_stem}"
                )

            # Restore the true path for diagnostics and error messages.
            try:
                result.path = str(expected_path)
            except Exception:
                pass

            predictions_by_stem[expected_stem] = result_to_official(result)
            returned_count += 1
            del result

        if returned_count != len(chunk_paths):
            first_missing = [
                p.name for p in chunk_paths[returned_count:returned_count + 10]
            ]
            raise RuntimeError(
                "Chunk result count mismatch: "
                f"expected={len(chunk_paths)}, received={returned_count}. "
                f"First missing inputs: {first_missing}"
            )

        newly_completed += len(chunk_paths)
        save_partial_predictions(predictions_by_stem)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
        else:
            allocated = reserved = 0.0

        elapsed = time.perf_counter() - run_start
        speed = newly_completed / max(elapsed, 1.0e-6)
        remaining = total_images - len(predictions_by_stem)
        eta_min = remaining / max(speed, 1.0e-6) / 60.0
        print(
            f"[Inference] completed={len(predictions_by_stem)}/{total_images} | "
            f"chunk={time.perf_counter() - chunk_time:.1f}s | "
            f"speed={speed:.2f} img/s | ETA={eta_min:.1f} min | "
            f"GPU={allocated:.2f}/{reserved:.2f} GiB"
        )

    missing_predictions = [
        p.name for p in image_paths
        if p.stem not in predictions_by_stem
    ]
    if missing_predictions:
        raise RuntimeError(
            f"Missing predictions for {len(missing_predictions)} images; "
            f"first examples: {missing_predictions[:10]}"
        )

    predictions = [predictions_by_stem[p.stem] for p in image_paths]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_cache = PREDICTION_CACHE.with_suffix(".tmp")
    with temp_cache.open("wb") as f:
        pickle.dump(
            {
                "image_stems": [p.stem for p in image_paths],
                "predictions": predictions,
                "weights": str(WEIGHTS.resolve()),
                "imgsz": IMGSZ,
                "conf": CONF,
                "max_det": MAX_DET,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    temp_cache.replace(PREDICTION_CACHE)

    print(f"\nPrediction cache saved to: {PREDICTION_CACHE}")
    print(f"Partial/resume cache:       {PARTIAL_CACHE}")
    return predictions

def load_cached_predictions(image_paths: list[Path]) -> list[list[np.ndarray]]:
    if not PREDICTION_CACHE.exists():
        raise FileNotFoundError(
            f"Prediction cache does not exist: {PREDICTION_CACHE}"
        )

    with PREDICTION_CACHE.open("rb") as f:
        payload = pickle.load(f)

    expected_stems = [p.stem for p in image_paths]
    cached_stems = payload.get("image_stems")
    if cached_stems != expected_stems:
        raise RuntimeError(
            "Cached image order does not match the current test set. "
            "Delete the cache and rerun STAGE='all'."
        )

    return payload["predictions"]


def evaluate_official(
    annotations: list[dict[str, np.ndarray]],
    predictions: list[list[np.ndarray]],
) -> dict[str, Any]:
    if len(annotations) != len(predictions):
        raise ValueError(
            f"GT/prediction length mismatch: "
            f"{len(annotations)} vs {len(predictions)}"
        )

    AITODReval = import_official_evaluator(OFFICIAL_EVAL_FILE)
    evaluator = AITODReval(
        annotations,
        predictions,
        numCats=len(CLASSES),
        nproc=EVAL_WORKERS,
    )

    evaluator.params.iouThrs = np.linspace(
        0.50,
        0.95,
        int(round((0.95 - 0.50) / 0.05)) + 1,
        endpoint=True,
    )
    evaluator.params.maxDets = [1500]

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    metric_names = (
        "AP",
        "AP50",
        "AP75",
        "APvt",
        "APt",
        "APs",
        "APm",
        "AR1500",
        "ARvt1500",
        "ARt1500",
        "ARs1500",
        "ARm1500",
    )
    metrics = {
        name: float(value)
        for name, value in zip(metric_names, evaluator.stats)
    }

    # Class-wise AP, matching the official dataset implementation.
    precisions = evaluator.eval["precision"]
    classwise_ap: dict[str, float | None] = {}
    for class_id, class_name in enumerate(CLASSES):
        precision = precisions[:, :, class_id, 0, -1]
        valid = precision[precision > -1]
        classwise_ap[class_name] = (
            float(np.mean(valid)) if valid.size else None
        )

    output = {
        "weights": str(WEIGHTS),
        "test_images": str(TEST_IMAGE_DIR),
        "test_labels": str(TEST_LABEL_DIR),
        "num_images": len(annotations),
        "max_dets": 1500,
        "metrics": metrics,
        "classwise_ap": classwise_ap,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_JSON.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    with CLASSWISE_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "AP"])
        for class_name in CLASSES:
            writer.writerow([class_name, classwise_ap[class_name]])

    print("\nOfficial metrics:")
    for name in metric_names:
        print(f"{name:10s}: {metrics[name]:.4f}")

    print("\nClass-wise AP:")
    for class_name, value in classwise_ap.items():
        text = "nan" if value is None else f"{value:.4f}"
        print(f"{class_name:16s}: {text}")

    print(f"\nMetrics JSON: {METRICS_JSON}")
    print(f"Class-wise CSV: {CLASSWISE_CSV}")
    return output


def main() -> None:
    if STAGE not in {"all", "infer", "eval"}:
        raise ValueError("STAGE must be one of: 'all', 'infer', 'eval'.")

    for path in (TEST_IMAGE_DIR, TEST_LABEL_DIR, WEIGHTS, OFFICIAL_EVAL_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required path does not exist: {path}")

    direct_label_count = sum(1 for _ in TEST_LABEL_DIR.glob("*.txt"))
    if direct_label_count == 0:
        raise FileNotFoundError(
            f"TEST_LABEL_DIR contains no direct .txt files: {TEST_LABEL_DIR}\n"
            "It must directly contain official DOTA-style labels:\n"
            "x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty"
        )

    image_paths = find_test_images(TEST_IMAGE_DIR)
    print("=" * 88)
    print("Official AI-TOD-R test")
    print("=" * 88)
    print(f"stage:          {STAGE}")
    print(f"weights:        {WEIGHTS}")
    print(f"test images:    {TEST_IMAGE_DIR}")
    print(f"test labels:    {TEST_LABEL_DIR}")
    print(f"images:         {len(image_paths)}")
    print(f"input:          {IMGSZ} x {IMGSZ}")
    print(f"precision:      FP32")
    print(f"max detections: {MAX_DET}")
    print(f"official eval:  {OFFICIAL_EVAL_FILE}")
    print("=" * 88)

    if STAGE in {"all", "infer"}:
        predictions = run_inference(image_paths)
    else:
        predictions = load_cached_predictions(image_paths)

    if STAGE in {"all", "eval"}:
        annotations = build_annotations(image_paths)
        evaluate_official(annotations, predictions)


if __name__ == "__main__":
    main()