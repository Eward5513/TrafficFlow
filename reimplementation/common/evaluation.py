"""Shared raw-scale evaluation and prediction saving for R-only models.

Training loops stay model-specific. After best-checkpoint selection, STID,
STAEformer, and PDFormer call this helper so test metrics and NPZ archives
are written the same way. Does not change Dataset semantics, scalers, or
model math.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reimplementation.common.data.r_only_npz_dataset import invert_target
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.metrics.traffic_metrics import traffic_metrics
from reimplementation.common.utils.atomic_io import atomic_write_json, atomic_write_npz
from reimplementation.common.utils.checkpoint import load_checkpoint

PREDICTION_KEYS = (
    "pred_raw",
    "y_raw",
    "y_pred_raw",
    "y_true_raw",
    "y_pred_normalized",
    "y_true_normalized",
    "day_index",
    "window_start_slot",
    "target_slot",
    "sample_index",
)


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.no_grad()
def collect_split_predictions(
    loader,
    predict_fn: Callable[[Mapping[str, Any]], torch.Tensor],
    *,
    mean_y: float,
    std_y: float,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run ``predict_fn`` on every batch and invert predictions to raw flow."""
    pred_norm: list[np.ndarray] = []
    true_norm: list[np.ndarray] = []
    pred_raw: list[np.ndarray] = []
    true_raw: list[np.ndarray] = []
    days: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sample_index: list[np.ndarray] = []
    for batch in loader:
        batch = _move_batch(batch, device)
        pred = predict_fn(batch)
        if not torch.is_tensor(pred):
            raise ReimplementationError("predict_fn must return a tensor")
        if pred.shape != batch["y"].shape:
            raise ReimplementationError(
                f"prediction {tuple(pred.shape)} != target {tuple(batch['y'].shape)}"
            )
        pred_cpu = pred.detach().cpu()
        inverted = invert_target(pred_cpu, float(mean_y), float(std_y))
        pred_norm.append(_to_numpy(pred_cpu).astype(np.float32, copy=False))
        true_norm.append(_to_numpy(batch["y"]).astype(np.float32, copy=False))
        pred_raw.append(_to_numpy(inverted).astype(np.float32, copy=False))
        true_raw.append(_to_numpy(batch["y_raw"]).astype(np.float32, copy=False))
        days.append(_to_numpy(batch["day_index"]).astype(np.int64, copy=False).reshape(-1))
        starts.append(_to_numpy(batch["window_start_slot"]).astype(np.int64, copy=False).reshape(-1))
        targets.append(_to_numpy(batch["target_slot"]).astype(np.int64, copy=False).reshape(-1))
        sample_index.append(_to_numpy(batch["sample_index"]).astype(np.int64, copy=False).reshape(-1))
    if not pred_raw:
        raise ReimplementationError("evaluation loader produced zero batches")
    packed = {
        "y_pred_normalized": np.concatenate(pred_norm, axis=0),
        "y_true_normalized": np.concatenate(true_norm, axis=0),
        "pred_raw": np.concatenate(pred_raw, axis=0),
        "y_raw": np.concatenate(true_raw, axis=0),
        "day_index": np.concatenate(days, axis=0),
        "window_start_slot": np.concatenate(starts, axis=0),
        "target_slot": np.concatenate(targets, axis=0),
        "sample_index": np.concatenate(sample_index, axis=0),
    }
    packed["y_pred_raw"] = packed["pred_raw"]
    packed["y_true_raw"] = packed["y_raw"]
    order = np.argsort(packed["sample_index"], kind="stable")
    packed = {key: value[order] for key, value in packed.items()}
    if packed["pred_raw"].shape != packed["y_raw"].shape:
        raise ReimplementationError("pred_raw shape does not match y_raw")
    if not np.isfinite(packed["pred_raw"]).all() or not np.isfinite(packed["y_raw"]).all():
        raise ReimplementationError("evaluation predictions contain NaN or inf")
    return packed


def compute_raw_metrics(pred_raw: np.ndarray, y_raw: np.ndarray) -> dict[str, float]:
    return traffic_metrics(pred_raw, y_raw, clip_negative=False)


def save_predictions_npz(path: Path, packed: Mapping[str, np.ndarray]) -> None:
    missing = [key for key in ("pred_raw", "y_raw") if key not in packed]
    if missing:
        raise ReimplementationError(f"prediction archive missing {missing}")
    payload = {key: packed[key] for key in PREDICTION_KEYS if key in packed}
    atomic_write_npz(path, **payload)
    verify_prediction_npz(path, expected_count=int(packed["pred_raw"].shape[0]))


def verify_prediction_npz(
    path: Path,
    *,
    expected_count: int,
    n_his: int = 12,
) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ReimplementationError(f"prediction NPZ missing or empty: {path}")
    with np.load(path, allow_pickle=False) as handle:
        files = set(handle.files)
        if "pred_raw" not in files or "y_raw" not in files:
            raise ReimplementationError(f"{path} must contain pred_raw and y_raw, got {sorted(files)}")
        pred = np.array(handle["pred_raw"])
        true = np.array(handle["y_raw"])
        extras = {}
        for key in ("day_index", "window_start_slot", "target_slot", "sample_index"):
            if key in handle.files:
                extras[key] = np.array(handle[key])
    if pred.shape != true.shape:
        raise ReimplementationError(f"{path} pred_raw {pred.shape} != y_raw {true.shape}")
    if int(pred.shape[0]) != int(expected_count):
        raise ReimplementationError(f"{path} sample count {pred.shape[0]} != {expected_count}")
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ReimplementationError(f"{path} contains NaN or inf")
    if "window_start_slot" in extras and "target_slot" in extras:
        start = extras["window_start_slot"].reshape(-1)
        target = extras["target_slot"].reshape(-1)
        if not np.array_equal(target, start + (int(n_his) - 1)):
            raise ReimplementationError(
                f"{path} target_slot != window_start_slot + {int(n_his) - 1}"
            )
    return {
        "path": path.as_posix(),
        "size_bytes": int(path.stat().st_size),
        "sample_count": int(pred.shape[0]),
        "pred_shape": list(pred.shape),
        "keys": sorted(files),
    }


def save_split_outputs(
    output_dir: Path,
    split: str,
    packed: Mapping[str, np.ndarray],
    *,
    expected_count: int,
    n_his: int = 12,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    metrics = compute_raw_metrics(packed["pred_raw"], packed["y_raw"])
    if int(packed["pred_raw"].shape[0]) != int(expected_count):
        raise ReimplementationError(
            f"{split} sample count {packed['pred_raw'].shape[0]} != {expected_count}"
        )
    atomic_write_json(output_dir / f"{split}_metrics.json", metrics)
    save_predictions_npz(output_dir / f"{split}_predictions.npz", packed)
    verify_prediction_npz(
        output_dir / f"{split}_predictions.npz",
        expected_count=expected_count,
        n_his=n_his,
    )
    for key in ("mae", "rmse", "mape_nonzero", "wape"):
        if key not in metrics or not np.isfinite(float(metrics[key])):
            raise ReimplementationError(f"{split} metric {key} missing or non-finite")
    return {
        "metrics": metrics,
        "sample_count": int(packed["pred_raw"].shape[0]),
        "pred_shape": list(packed["pred_raw"].shape),
    }


def evaluate_loader(
    loader,
    predict_fn: Callable[[Mapping[str, Any]], torch.Tensor],
    *,
    mean_y: float,
    std_y: float,
    device: torch.device,
    output_dir: Path,
    split: str,
    expected_count: int,
    n_his: int = 12,
) -> dict[str, Any]:
    packed = collect_split_predictions(
        loader,
        predict_fn,
        mean_y=mean_y,
        std_y=std_y,
        device=device,
    )
    saved = save_split_outputs(
        output_dir,
        split,
        packed,
        expected_count=expected_count,
        n_his=n_his,
    )
    return saved


def reload_best_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ReimplementationError(f"best checkpoint missing or empty: {path}")
    payload = load_checkpoint(path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return payload


def assert_official_rate_outputs(
    rate_dir: Path,
    *,
    expected_test_count: int,
    expected_val_count: int | None = None,
    n_his: int = 12,
) -> dict[str, Any]:
    rate_dir = Path(rate_dir)
    best = rate_dir / "best_checkpoint.pt"
    history = rate_dir / "training_history.json"
    test_metrics = rate_dir / "test_metrics.json"
    test_pred = rate_dir / "test_predictions.npz"
    for path in (best, history, test_metrics, test_pred):
        if not path.is_file() or path.stat().st_size <= 0:
            raise ReimplementationError(f"required output missing or empty: {path}")
    pred_info = verify_prediction_npz(test_pred, expected_count=expected_test_count, n_his=n_his)
    val_pred = rate_dir / "validation_predictions.npz"
    if expected_val_count is not None and val_pred.is_file():
        verify_prediction_npz(val_pred, expected_count=expected_val_count, n_his=n_his)
    return {
        "checkpoint_exists": True,
        "checkpoint_size": int(best.stat().st_size),
        "test_metrics_exists": True,
        "test_predictions_exists": True,
        "test_predictions_size": int(test_pred.stat().st_size),
        "prediction_info": pred_info,
    }


def upsert_overall_summary(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge rate summaries by ``rate`` so a later p10–p70 job does not drop p05."""
    path = Path(path)
    existing: list[Any] = []
    if path.is_file() and path.stat().st_size > 0:
        text = path.read_text(encoding="utf-8")
        if text.strip():
            loaded = json.loads(text)
            if isinstance(loaded, list):
                existing = loaded
    by_rate: dict[str, dict[str, Any]] = {}
    for row in existing:
        if isinstance(row, dict) and row.get("rate") is not None:
            by_rate[str(row["rate"])] = row
    for row in rows:
        by_rate[str(row["rate"])] = row
    order = ("p05", "p10", "p20", "p30", "p40", "p50", "p70")
    merged = [by_rate[tag] for tag in order if tag in by_rate]
    for tag, row in by_rate.items():
        if tag not in order:
            merged.append(row)
    atomic_write_json(path, merged)
    return merged
