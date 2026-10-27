"""Official R-only baseline protocol: rates, timing, GPU metadata, cosine LR.

Training loops stay model-specific. This module does not change Dataset
semantics, scalers, or model math.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reimplementation.common.data.r_only_npz_dataset import RATE_TAGS, rate_tag
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.evaluation import assert_official_rate_outputs, verify_prediction_npz
from reimplementation.common.utils.atomic_io import atomic_write_json, atomic_write_npz

OFFICIAL_RATES = RATE_TAGS
FORBIDDEN_RATES = ("p60",)
OFFICIAL_TEST_SAMPLES = 831
OFFICIAL_VAL_SAMPLES = 831
OFFICIAL_TRAIN_SAMPLES = 3878
OFFICIAL_SEED = 42

PREDICTION_ARRAY_KEYS = (
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


def assert_official_rates(rates: Sequence[str | int]) -> list[str]:
    tags = [rate_tag(item) for item in rates]
    illegal = [tag for tag in tags if tag in FORBIDDEN_RATES or tag not in OFFICIAL_RATES]
    if illegal:
        raise ReimplementationError(f"official baseline forbids rates {illegal}; allowed={list(OFFICIAL_RATES)}")
    return tags


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def physical_gpu_id() -> int | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return None
    first = visible.split(",")[0].strip()
    if first == "":
        return None
    try:
        return int(first)
    except ValueError:
        return None


def gpu_runtime_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "physical_gpu_id": physical_gpu_id(),
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "pytorch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "logical_device": str(device),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(index)
        info["gpu_name"] = torch.cuda.get_device_name(index)
        info["gpu_total_memory"] = int(props.total_memory)
        info["gpu_total_memory_mb"] = round(float(props.total_memory) / (1024.0 * 1024.0), 3)
    else:
        info["gpu_name"] = None
        info["gpu_total_memory"] = None
        info["gpu_total_memory_mb"] = None
    return info


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "peak_gpu_memory_allocated_mb": None,
            "peak_gpu_memory_reserved_mb": None,
        }
    return {
        "peak_gpu_memory_allocated_mb": round(
            float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0), 3
        ),
        "peak_gpu_memory_reserved_mb": round(
            float(torch.cuda.max_memory_reserved(device)) / (1024.0 * 1024.0), 3
        ),
    }


class ExperimentClock:
    """Wall-clock protocol: train time excludes formal test; experiment time includes it."""

    def __init__(self) -> None:
        self.experiment_start: float | None = None
        self.train_start: float | None = None
        self.train_end: float | None = None
        self.experiment_end: float | None = None

    def mark_experiment_start(self) -> None:
        self.experiment_start = time.perf_counter()

    def mark_train_start(self) -> None:
        if self.train_start is None:
            self.train_start = time.perf_counter()

    def mark_train_end(self) -> None:
        self.train_end = time.perf_counter()

    def mark_experiment_end(self) -> None:
        self.experiment_end = time.perf_counter()

    def summary(self, epochs_executed: int) -> dict[str, float]:
        if self.experiment_start is None or self.experiment_end is None:
            raise ReimplementationError("experiment clock was not closed")
        if self.train_start is None or self.train_end is None:
            raise ReimplementationError("training clock was not closed")
        train_wall = float(self.train_end - self.train_start)
        executed = int(epochs_executed)
        if train_wall <= 0.0:
            raise ReimplementationError(f"train_wall_time_sec must be > 0, got {train_wall}")
        if executed <= 0:
            raise ReimplementationError("total_epochs_executed must be > 0")
        return {
            "train_wall_time_sec": train_wall,
            "total_experiment_wall_time_sec": float(self.experiment_end - self.experiment_start),
            "average_epoch_time_sec": train_wall / float(executed),
            "total_epochs_executed": float(executed),
        }


def libcity_cosine_lr(
    t: int,
    *,
    base_lr: float,
    t_initial: int,
    lr_min: float,
    warmup_t: int,
    warmup_lr_init: float,
    warmup_prefix: bool = False,
) -> float:
    """timm/LibCity CosineLRScheduler with ``t_mul=1``, ``decay_rate=1``, ``cycle_limit=0``."""
    step = int(t)
    warmup = int(warmup_t)
    horizon = max(int(t_initial), 1)
    if warmup > 0 and step < warmup:
        return float(warmup_lr_init) + float(step) * (float(base_lr) - float(warmup_lr_init)) / float(warmup)
    cosine_t = step - warmup if warmup_prefix else step
    cycle = cosine_t // horizon
    t_curr = cosine_t - horizon * cycle
    return float(lr_min) + 0.5 * (float(base_lr) - float(lr_min)) * (
        1.0 + math.cos(math.pi * float(t_curr) / float(horizon))
    )


class LibCityCosineLRScheduler:
    """LibCity ``CosineLRScheduler`` (timm) with ``warmup_prefix=False`` by default."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        t_initial: int,
        lr_min: float,
        warmup_t: int,
        warmup_lr_init: float,
        warmup_prefix: bool = False,
    ) -> None:
        self.optimizer = optimizer
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.t_initial = max(int(t_initial), 1)
        self.lr_min = float(lr_min)
        self.warmup_t = int(warmup_t)
        self.warmup_lr_init = float(warmup_lr_init)
        self.warmup_prefix = bool(warmup_prefix)
        self.last_epoch = -1
        if self.warmup_t > 0:
            self._set_lr(self.warmup_lr_init)

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    def get_lr(self, epoch: int) -> float:
        return libcity_cosine_lr(
            int(epoch),
            base_lr=self.base_lrs[0],
            t_initial=self.t_initial,
            lr_min=self.lr_min,
            warmup_t=self.warmup_t,
            warmup_lr_init=self.warmup_lr_init,
            warmup_prefix=self.warmup_prefix,
        )

    def step(self, epoch: int | None = None) -> None:
        if epoch is None:
            self.last_epoch += 1
            epoch = self.last_epoch
        else:
            self.last_epoch = int(epoch)
        self._set_lr(self.get_lr(int(epoch)))

    def state_dict(self) -> dict[str, Any]:
        return {
            "last_epoch": self.last_epoch,
            "warmup_prefix": self.warmup_prefix,
            "t_initial": self.t_initial,
            "lr_min": self.lr_min,
            "warmup_t": self.warmup_t,
            "warmup_lr_init": self.warmup_lr_init,
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        self.last_epoch = int(payload["last_epoch"])
        self._set_lr(self.get_lr(max(self.last_epoch, 0)))


def attach_prediction_aliases(packed: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(packed)
    if "pred_raw" not in out and "y_pred_raw" in out:
        out["pred_raw"] = out["y_pred_raw"]
    if "y_raw" not in out and "y_true_raw" in out:
        out["y_raw"] = out["y_true_raw"]
    if "y_pred_raw" not in out and "pred_raw" in out:
        out["y_pred_raw"] = out["pred_raw"]
    if "y_true_raw" not in out and "y_raw" in out:
        out["y_true_raw"] = out["y_raw"]
    return out


def write_prediction_npz(
    path: Path,
    packed: Mapping[str, Any],
    *,
    expected_count: int,
    n_his: int = 12,
) -> dict[str, Any]:
    payload = attach_prediction_aliases(packed)
    arrays: dict[str, Any] = {}
    for key in PREDICTION_ARRAY_KEYS:
        value = payload.get(key)
        if value is None or key == "metrics":
            continue
        arrays[key] = value
    if "pred_raw" not in arrays or "y_raw" not in arrays:
        raise ReimplementationError(f"{path} prediction archive missing pred_raw/y_raw")
    atomic_write_npz(path, **arrays)
    return verify_prediction_npz(path, expected_count=int(expected_count), n_his=int(n_his))


def best_epoch_from_history(
    history: Sequence[Mapping[str, Any]],
    metric_key: str,
    *,
    rule: str,
) -> int:
    if not history:
        raise ReimplementationError("training history is empty")
    pairs = [(int(row["epoch"]), float(row[metric_key])) for row in history]
    best_value = min(value for _, value in pairs)
    matching = [epoch for epoch, value in pairs if value == best_value]
    if str(rule) == "le":
        return int(max(matching))
    if str(rule) == "lt":
        return int(min(matching))
    raise ReimplementationError(f"unknown best-epoch rule {rule}")


def build_run_metadata(
    *,
    model: str,
    penetration: str,
    seed: int,
    device: torch.device,
    parameter_count: int,
    training_loss: str,
    optimizer: str,
    initial_lr: float,
    scheduler: str | None,
    max_epochs: int,
    epochs_executed: int,
    best_epoch: int,
    best_metric: float,
    early_stopping: bool,
    early_stopping_patience: int | None,
    selection_metric: str,
    clock: ExperimentClock,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    tag = rate_tag(penetration)
    if tag not in OFFICIAL_RATES:
        raise ReimplementationError(f"penetration {tag} is not an official baseline rate")
    timing = clock.summary(int(epochs_executed))
    payload: dict[str, Any] = {
        "model": str(model),
        "penetration": tag,
        "penetration_rate": tag,
        "seed": int(seed),
        "parameter_count": int(parameter_count),
        "trainable_parameter_count": int(parameter_count),
        "training_loss": str(training_loss),
        "optimizer": str(optimizer),
        "initial_lr": float(initial_lr),
        "scheduler": scheduler,
        "max_epochs": int(max_epochs),
        "epochs_executed": int(epochs_executed),
        "total_epochs_executed": int(epochs_executed),
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "early_stopping": bool(early_stopping),
        "early_stopping_patience": early_stopping_patience,
        "selection_metric": str(selection_metric),
        "did_not_use_test_for_selection": True,
        "target_mode": "last-observed-step",
        **gpu_runtime_info(device),
        **peak_memory_mb(device),
        **timing,
    }
    if extra:
        payload.update(dict(extra))
    return payload


def finalize_completed_rate(
    rate_dir: Path,
    *,
    metadata: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    expected_test_count: int = OFFICIAL_TEST_SAMPLES,
    expected_val_count: int | None = OFFICIAL_VAL_SAMPLES,
    n_his: int = 12,
    history_metric_key: str | None = None,
    history_rule: str = "lt",
) -> dict[str, Any]:
    rate_dir = Path(rate_dir)
    payload = dict(metadata)
    if str(payload.get("status", "completed")) != "completed":
        payload["status"] = "completed"
    atomic_write_json(rate_dir / "run_metadata.json", payload)
    checks = assert_official_rate_outputs(
        rate_dir,
        expected_test_count=int(expected_test_count),
        expected_val_count=expected_val_count,
        n_his=int(n_his),
    )
    if float(payload.get("train_wall_time_sec", 0.0)) <= 0.0:
        raise ReimplementationError("train_wall_time_sec missing or not positive")
    if int(payload.get("epochs_executed", 0)) <= 0:
        raise ReimplementationError("epochs_executed missing or not positive")
    if payload.get("physical_gpu_id") is None and str(payload.get("logical_device", "")).startswith("cuda"):
        raise ReimplementationError("physical_gpu_id missing while training on CUDA")
    tag = str(payload.get("penetration") or payload.get("penetration_rate"))
    if tag not in OFFICIAL_RATES:
        raise ReimplementationError(f"completed rate {tag} is not official")
    if history_metric_key:
        expected_best = best_epoch_from_history(history, history_metric_key, rule=history_rule)
        if int(payload["best_epoch"]) != int(expected_best):
            raise ReimplementationError(
                f"best_epoch {payload['best_epoch']} != history {history_rule} "
                f"{history_metric_key} epoch {expected_best}"
            )
    test_metrics_path = rate_dir / "test_metrics.json"
    metrics = json.loads(test_metrics_path.read_text(encoding="utf-8"))
    for key in ("mae", "rmse", "mape_nonzero", "wape"):
        if key not in metrics or not np.isfinite(float(metrics[key])):
            raise ReimplementationError(f"test metric {key} missing or non-finite")
    payload["status"] = "completed"
    atomic_write_json(rate_dir / "run_metadata.json", payload)
    checks["status"] = "completed"
    checks["run_metadata"] = True
    return checks


def mark_rate_failed(rate_dir: Path, *, reason: str, extra: Mapping[str, Any] | None = None) -> None:
    payload = {"status": "failed", "reason": str(reason)}
    if extra:
        payload.update(dict(extra))
    rate_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(rate_dir / "run_metadata.json", payload)
