#!/usr/bin/env python3
"""CPU-only offline extraction and error analysis of the 56 official baselines.

Reads training_results artifacts and the shared r-only node mapping. Never
loads checkpoints, never trains, never imports torch, never writes under
training_results.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

CODE_ROOT = Path("/home/tzhang174/TrafficFlow")
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from data_root import (  # noqa: E402
    assert_not_code_root_output,
    data_root,
    node_table,
    rate_training_results_dir,
    training_results_root,
)

RATES = ("p05", "p10", "p20", "p30", "p40", "p50", "p70")
MODELS: tuple[tuple[str, str], ...] = (
    ("stgcn", "STGCN"),
    ("dcrnn", "DCRNN"),
    ("graph_wavenet", "Graph WaveNet"),
    ("stsgcn", "STSGCN"),
    ("stfgnn", "STFGNN"),
    ("stid", "STID"),
    ("staeformer", "STAEformer"),
    ("pdformer", "PDFormer"),
)
MODEL_DIRS = [slug for slug, _ in MODELS]
DISPLAY = {slug: name for slug, name in MODELS}
EXPECTED_T = 831
EXPECTED_N = 56
N_HIS = 12
SLOTS_PER_HOUR = 12
TARGETS_PER_DAY = 277
MAE_ATOL = 1e-5
MAE_RTOL = 1e-6
WAPE_ATOL = 1e-6
WAPE_RTOL = 1e-6
FLOW_BINS = (
    ("bin0_y_eq_0", lambda y: y == 0, False),
    ("bin1_0_lt_y_le_2", lambda y: (y > 0) & (y <= 2), True),
    ("bin2_2_lt_y_le_5", lambda y: (y > 2) & (y <= 5), True),
    ("bin3_5_lt_y_le_10", lambda y: (y > 5) & (y <= 10), True),
    ("bin4_y_gt_10", lambda y: y > 10, True),
)
MODEL_COLORS = {
    "STGCN": "#1f77b4",
    "DCRNN": "#ff7f0e",
    "Graph WaveNet": "#2ca02c",
    "STSGCN": "#d62728",
    "STFGNN": "#9467bd",
    "STID": "#8c564b",
    "STAEformer": "#e377c2",
    "PDFormer": "#7f7f7f",
}


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _squeeze_tn(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    squeezed = np.squeeze(arr)
    if squeezed.ndim == 2 and squeezed.shape == (EXPECTED_T, EXPECTED_N):
        return squeezed.astype(np.float64, copy=False)
    if squeezed.ndim == 2 and squeezed.shape == (EXPECTED_N, EXPECTED_T):
        return squeezed.T.astype(np.float64, copy=False)
    raise ValueError(f"unexpected prediction shape {arr.shape} -> {squeezed.shape}")


def _corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    xx = np.asarray(x[mask], dtype=np.float64)
    yy = np.asarray(y[mask], dtype=np.float64)
    if xx.size < 3 or np.unique(xx).size < 2 or np.unique(yy).size < 2:
        return float("nan"), float("nan")
    pearson = float(pearsonr(xx, yy).statistic)
    spearman = float(spearmanr(xx, yy).statistic)
    return pearson, spearman


def _quantile_label(values: np.ndarray, value: float) -> str:
    q1, q2, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    if value <= q1:
        return "Q1"
    if value <= q2:
        return "Q2"
    if value <= q3:
        return "Q3"
    return "Q4"


def load_edge_ids(node_mapping: Path) -> list[str]:
    rows: list[tuple[int, str]] = []
    with node_mapping.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append((int(row["node_index"]), str(row["edge_id"])))
    rows.sort(key=lambda item: item[0])
    if [idx for idx, _ in rows] != list(range(len(rows))):
        raise RuntimeError(f"node_mapping node_index is not 0..N-1 in {node_mapping}")
    if len(rows) != EXPECTED_N:
        raise RuntimeError(f"expected {EXPECTED_N} edges, got {len(rows)}")
    return [edge_id for _, edge_id in rows]


def run_dir(model: str, rate: str) -> Path:
    return rate_training_results_dir(model, rate, seed=42)


def audit_run(path: Path) -> dict[str, Any]:
    problems: list[str] = []
    status = "missing"
    test_samples = None
    pred_valid = False
    metrics_valid = False
    timing_valid = False
    meta: dict[str, Any] = {}
    test_metrics: dict[str, Any] = {}
    val_metrics: dict[str, Any] | None = None

    required = {
        "best_checkpoint.pt": path / "best_checkpoint.pt",
        "training_history.json": path / "training_history.json",
        "validation_metrics.json": path / "validation_metrics.json",
        "test_metrics.json": path / "test_metrics.json",
        "test_predictions.npz": path / "test_predictions.npz",
        "run_metadata.json": path / "run_metadata.json",
    }
    for name, file_path in required.items():
        if not file_path.is_file():
            problems.append(f"missing {name}")
        elif file_path.stat().st_size <= 0:
            problems.append(f"empty {name}")

    if (path / "run_metadata.json").is_file() and (path / "run_metadata.json").stat().st_size > 0:
        try:
            meta = _load_json(path / "run_metadata.json")
            status = str(meta.get("status", "unknown"))
            if status != "completed":
                problems.append(f"status={status}")
            train_wall = float(meta.get("train_wall_time_sec", 0.0) or 0.0)
            epochs = int(meta.get("epochs_executed", 0) or 0)
            if train_wall > 0 and epochs > 0:
                timing_valid = True
            else:
                problems.append("timing invalid")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"run_metadata unreadable: {exc}")

    if (path / "training_history.json").is_file() and (path / "training_history.json").stat().st_size > 0:
        try:
            history = _load_json(path / "training_history.json")
            if not history:
                problems.append("training_history empty")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"training_history unreadable: {exc}")

    if (path / "validation_metrics.json").is_file() and (path / "validation_metrics.json").stat().st_size > 0:
        try:
            val_metrics = _load_json(path / "validation_metrics.json")
            if not _finite(val_metrics.get("mae")):
                problems.append("validation mae missing")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"validation_metrics unreadable: {exc}")
            val_metrics = None

    if (path / "test_metrics.json").is_file() and (path / "test_metrics.json").stat().st_size > 0:
        try:
            test_metrics = _load_json(path / "test_metrics.json")
            needed = ("mae", "rmse", "mape_nonzero", "wape")
            if all(_finite(test_metrics.get(key)) for key in needed):
                metrics_valid = True
            else:
                problems.append("test metrics non-finite")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"test_metrics unreadable: {exc}")

    pred = y = None
    target_slot = None
    if (path / "test_predictions.npz").is_file() and (path / "test_predictions.npz").stat().st_size > 0:
        try:
            with np.load(path / "test_predictions.npz", allow_pickle=False) as npz:
                keys = set(npz.files)
                if "pred_raw" not in keys:
                    problems.append("pred_raw missing")
                if "y_raw" not in keys:
                    problems.append("y_raw missing")
                if "pred_raw" in keys and "y_raw" in keys:
                    pred = np.asarray(npz["pred_raw"])
                    y = np.asarray(npz["y_raw"])
                    if pred.shape != y.shape:
                        problems.append(f"shape mismatch pred={pred.shape} y={y.shape}")
                    try:
                        pred_tn = _squeeze_tn(pred)
                        y_tn = _squeeze_tn(y)
                        test_samples = int(pred_tn.shape[0])
                        if test_samples != EXPECTED_T:
                            problems.append(f"test samples={test_samples}")
                        if not np.isfinite(pred_tn).all():
                            problems.append("pred NaN/Inf")
                        if not np.isfinite(y_tn).all():
                            problems.append("y NaN/Inf")
                        if (
                            test_samples == EXPECTED_T
                            and pred_tn.shape[1] == EXPECTED_N
                            and np.isfinite(pred_tn).all()
                            and np.isfinite(y_tn).all()
                        ):
                            pred_valid = True
                    except Exception as exc:  # noqa: BLE001
                        problems.append(f"squeeze failed: {exc}")
                if "target_slot" in keys:
                    target_slot = np.asarray(npz["target_slot"])
        except Exception as exc:  # noqa: BLE001
            problems.append(f"npz unreadable: {exc}")

    return {
        "status": status,
        "test_samples": test_samples,
        "prediction_valid": pred_valid,
        "metrics_valid": metrics_valid,
        "timing_valid": timing_valid,
        "problems": "; ".join(problems) if problems else "",
        "ok": not problems and pred_valid and metrics_valid and timing_valid and status == "completed",
        "meta": meta,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "pred": None,
        "y": None,
        "target_slot": target_slot,
        "path": str(path),
    }


def load_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path / "test_predictions.npz", allow_pickle=False) as npz:
        pred = _squeeze_tn(npz["pred_raw"])
        y = _squeeze_tn(npz["y_raw"])
        if "target_slot" not in npz.files:
            raise RuntimeError("target_slot missing; hourly analysis unavailable")
        target_slot = np.asarray(npz["target_slot"]).reshape(-1)
        if target_slot.shape[0] != pred.shape[0]:
            raise RuntimeError("target_slot length != T")
    return pred, y, target_slot.astype(np.int64, copy=False)


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def analyze_all(output_dir: Path) -> dict[str, Any]:
    results_root = training_results_root()
    node_mapping = node_table()
    edge_ids = load_edge_ids(node_mapping)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    audits: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    edge_wape_rows: list[dict[str, Any]] = []
    edge_bias_rows: list[dict[str, Any]] = []
    edge_summary_rows: list[dict[str, Any]] = []
    edge_flow_corr_rows: list[dict[str, Any]] = []
    edge_quartile_rows: list[dict[str, Any]] = []
    hourly_rows: list[dict[str, Any]] = []
    hourly_bias_rows: list[dict[str, Any]] = []
    hourly_corr_rows: list[dict[str, Any]] = []
    flow_bin_rows: list[dict[str, Any]] = []
    zero_rows: list[dict[str, Any]] = []
    nonzero_rows: list[dict[str, Any]] = []
    bias_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    conv_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    sanity_failures: list[dict[str, Any]] = []
    hourly_available = True
    arrays: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for model, display in MODELS:
        for rate in RATES:
            path = run_dir(model, rate)
            audit = audit_run(path)
            audit_row = {
                "model": display,
                "model_dir": model,
                "penetration": rate,
                "status": audit["status"],
                "test_samples": audit["test_samples"],
                "prediction_valid": audit["prediction_valid"],
                "metrics_valid": audit["metrics_valid"],
                "timing_valid": audit["timing_valid"],
                "problem": audit["problems"],
            }
            audits.append(audit_row)
            if not audit["ok"]:
                continue
            pred, y, target_slot = load_arrays(path)
            arrays[(model, rate)] = (pred, y, target_slot)
            abs_err = np.abs(pred - y)
            signed = pred - y
            y_sum = float(y.sum())
            recomputed_mae = float(abs_err.mean())
            recomputed_wape = float(abs_err.sum() / y_sum * 100.0) if y_sum != 0 else float("nan")
            reported_mae = float(audit["test_metrics"]["mae"])
            reported_wape = float(audit["test_metrics"]["wape"])
            mae_ok = math.isclose(recomputed_mae, reported_mae, rel_tol=MAE_RTOL, abs_tol=MAE_ATOL)
            wape_ok = math.isclose(recomputed_wape, reported_wape, rel_tol=WAPE_RTOL, abs_tol=WAPE_ATOL)
            if not mae_ok:
                sanity_failures.append(
                    {
                        "model": display,
                        "penetration": rate,
                        "metric": "mae",
                        "reported": reported_mae,
                        "recomputed": recomputed_mae,
                        "abs_diff": abs(recomputed_mae - reported_mae),
                    }
                )
                continue
            if not wape_ok:
                sanity_failures.append(
                    {
                        "model": display,
                        "penetration": rate,
                        "metric": "wape",
                        "reported": reported_wape,
                        "recomputed": recomputed_wape,
                        "abs_diff": abs(recomputed_wape - reported_wape),
                    }
                )

            meta = audit["meta"]
            epochs = int(meta["epochs_executed"])
            train_wall = float(meta["train_wall_time_sec"])
            sec_epoch = train_wall / epochs if epochs else float("nan")
            best_epoch = int(meta["best_epoch"])
            early_stopped = bool(meta.get("early_stopped", False))
            params = int(meta["parameter_count"])
            peak_gpu = float(meta.get("peak_gpu_memory_allocated_mb"))
            val_mae = None
            if audit["val_metrics"] is not None and _finite(audit["val_metrics"].get("mae")):
                val_mae = float(audit["val_metrics"]["mae"])

            final_rows.append(
                {
                    "model": display,
                    "model_dir": model,
                    "penetration": rate,
                    "test_mae": reported_mae,
                    "rmse": float(audit["test_metrics"]["rmse"]),
                    "mape_nonzero": float(audit["test_metrics"]["mape_nonzero"]),
                    "wape": reported_wape,
                    "best_epoch": best_epoch,
                    "epochs_run": epochs,
                    "early_stopped": early_stopped,
                    "train_time_sec": train_wall,
                    "sec_per_epoch": sec_epoch,
                    "params": params,
                    "peak_gpu_mb": peak_gpu,
                    "recomputed_mae": recomputed_mae,
                    "recomputed_wape": recomputed_wape,
                    "mae_sanity_ok": mae_ok,
                    "wape_sanity_ok": wape_ok,
                    "mae_abs_diff": abs(recomputed_mae - reported_mae),
                    "wape_abs_diff": abs(recomputed_wape - reported_wape) if _finite(recomputed_wape) else float("nan"),
                    "selection_metric": meta.get("selection_metric"),
                    "average_epoch_time_sec": meta.get("average_epoch_time_sec"),
                    "max_epochs": meta.get("max_epochs"),
                    "early_stopping": meta.get("early_stopping"),
                    "early_stopping_patience": meta.get("early_stopping_patience"),
                }
            )

            if val_mae is None:
                gap_rows.append(
                    {
                        "model": display,
                        "penetration": rate,
                        "val_mae": None,
                        "test_mae": reported_mae,
                        "gap": None,
                        "gap_percent": None,
                        "note": "unavailable",
                    }
                )
            else:
                gap = reported_mae - val_mae
                gap_rows.append(
                    {
                        "model": display,
                        "penetration": rate,
                        "val_mae": val_mae,
                        "test_mae": reported_mae,
                        "gap": gap,
                        "gap_percent": gap / val_mae * 100.0 if val_mae != 0 else float("nan"),
                        "note": "",
                    }
                )

            conv_rows.append(
                {
                    "model": display,
                    "penetration": rate,
                    "best_epoch": best_epoch,
                    "epochs_run": epochs,
                    "best_epoch_ratio": (best_epoch + 1) / epochs if epochs else float("nan"),
                    "early_stopped": early_stopped,
                    "max_epochs": meta.get("max_epochs"),
                }
            )

            flow_sum = y.sum(axis=0)
            abs_sum = abs_err.sum(axis=0)
            mean_signed_edge = signed.mean(axis=0)
            over_edge = (signed > 0).mean(axis=0)
            under_edge = (signed < 0).mean(axis=0)
            valid_wapes: list[float] = []
            valid_flows: list[float] = []
            for n, edge_id in enumerate(edge_ids):
                true_sum = float(flow_sum[n])
                abs_s = float(abs_sum[n])
                if true_sum == 0:
                    wape_edge = None
                    valid = False
                    note = "undefined_zero_volume"
                else:
                    wape_edge = abs_s / true_sum * 100.0
                    valid = True
                    note = ""
                    valid_wapes.append(wape_edge)
                    valid_flows.append(true_sum)
                edge_wape_rows.append(
                    {
                        "model": display,
                        "model_dir": model,
                        "penetration": rate,
                        "edge_index": n,
                        "edge_id": edge_id,
                        "true_flow_sum": true_sum,
                        "absolute_error_sum": abs_s,
                        "wape": wape_edge,
                        "valid": valid,
                        "note": note,
                    }
                )
                edge_bias_rows.append(
                    {
                        "model": display,
                        "penetration": rate,
                        "edge_id": edge_id,
                        "edge_index": n,
                        "mean_signed_error": float(mean_signed_edge[n]),
                        "overprediction_fraction": float(over_edge[n]),
                        "underprediction_fraction": float(under_edge[n]),
                    }
                )

            vw = np.asarray(valid_wapes, dtype=np.float64)
            vf = np.asarray(valid_flows, dtype=np.float64)
            edge_summary_rows.append(
                {
                    "model": display,
                    "penetration": rate,
                    "n_valid": int(vw.size),
                    "n_undefined": int(EXPECTED_N - vw.size),
                    "mean": float(vw.mean()) if vw.size else float("nan"),
                    "median": float(np.median(vw)) if vw.size else float("nan"),
                    "std": float(vw.std(ddof=1)) if vw.size > 1 else float("nan"),
                    "min": float(vw.min()) if vw.size else float("nan"),
                    "max": float(vw.max()) if vw.size else float("nan"),
                    "p25": percentile(vw, 25),
                    "p75": percentile(vw, 75),
                    "p90": percentile(vw, 90),
                }
            )
            pearson_e, spearman_e = _corr(vf, vw)
            edge_flow_corr_rows.append(
                {
                    "model": display,
                    "penetration": rate,
                    "pearson_flow_wape": pearson_e,
                    "spearman_flow_wape": spearman_e,
                    "n_valid": int(vw.size),
                }
            )
            if vw.size:
                quartiles = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
                for true_sum, wape_edge in zip(vf, vw):
                    quartiles[_quantile_label(vf, float(true_sum))].append(wape_edge)
                edge_quartile_rows.append(
                    {
                        "model": display,
                        "penetration": rate,
                        "q1_median_wape": float(np.median(quartiles["Q1"])) if quartiles["Q1"] else float("nan"),
                        "q2_median_wape": float(np.median(quartiles["Q2"])) if quartiles["Q2"] else float("nan"),
                        "q3_median_wape": float(np.median(quartiles["Q3"])) if quartiles["Q3"] else float("nan"),
                        "q4_median_wape": float(np.median(quartiles["Q4"])) if quartiles["Q4"] else float("nan"),
                        "q1_n": len(quartiles["Q1"]),
                        "q2_n": len(quartiles["Q2"]),
                        "q3_n": len(quartiles["Q3"]),
                        "q4_n": len(quartiles["Q4"]),
                    }
                )

            hours = target_slot // SLOTS_PER_HOUR
            hour_flow = []
            hour_wape = []
            for hour in range(24):
                mask = hours == hour
                n_samples = int(mask.sum())
                true_sum = float(y[mask].sum()) if n_samples else 0.0
                abs_s = float(abs_err[mask].sum()) if n_samples else 0.0
                if n_samples == 0 or true_sum == 0:
                    wape_h = None
                else:
                    wape_h = abs_s / true_sum * 100.0
                    hour_flow.append(true_sum)
                    hour_wape.append(wape_h)
                hourly_rows.append(
                    {
                        "model": display,
                        "penetration": rate,
                        "hour": hour,
                        "num_samples": n_samples,
                        "true_flow_sum": true_sum,
                        "absolute_error_sum": abs_s,
                        "wape": wape_h,
                        "partial_hour": hour == 0,
                    }
                )
                hourly_bias_rows.append(
                    {
                        "model": display,
                        "penetration": rate,
                        "hour": hour,
                        "num_samples": n_samples,
                        "mean_signed_error": float(signed[mask].mean()) if n_samples else None,
                        "partial_hour": hour == 0,
                    }
                )
            pearson_h, spearman_h = _corr(np.asarray(hour_flow), np.asarray(hour_wape))
            hourly_corr_rows.append(
                {
                    "model": display,
                    "penetration": rate,
                    "pearson_hour_flow_wape": pearson_h,
                    "spearman_hour_flow_wape": spearman_h,
                }
            )

            y_flat = y.reshape(-1)
            pred_flat = pred.reshape(-1)
            abs_flat = np.abs(pred_flat - y_flat)
            for bin_name, predicate, mape_ok in FLOW_BINS:
                mask = predicate(y_flat)
                count = int(mask.sum())
                if count == 0:
                    flow_bin_rows.append(
                        {
                            "model": display,
                            "penetration": rate,
                            "bin": bin_name,
                            "count": 0,
                            "mae": None,
                            "rmse": None,
                            "mean_prediction": None,
                            "mean_true_flow": None,
                            "mape": None,
                        }
                    )
                    continue
                yt = y_flat[mask]
                pt = pred_flat[mask]
                ae = abs_flat[mask]
                mape = None
                if mape_ok:
                    mape = float(np.mean(ae / yt) * 100.0)
                flow_bin_rows.append(
                    {
                        "model": display,
                        "penetration": rate,
                        "bin": bin_name,
                        "count": count,
                        "mae": float(ae.mean()),
                        "rmse": float(np.sqrt(np.mean((pt - yt) ** 2))),
                        "mean_prediction": float(pt.mean()),
                        "mean_true_flow": float(yt.mean()),
                        "mape": mape,
                    }
                )

            zero_mask = y_flat == 0
            pos_mask = y_flat > 0
            zero_count = int(zero_mask.sum())
            zero_rows.append(
                {
                    "model": display,
                    "penetration": rate,
                    "zero_cell_count": zero_count,
                    "zero_fraction": zero_count / y_flat.size,
                    "zero_mae": float(abs_flat[zero_mask].mean()) if zero_count else None,
                    "mean_pred_at_zero": float(pred_flat[zero_mask].mean()) if zero_count else None,
                    "median_pred_at_zero": float(np.median(pred_flat[zero_mask])) if zero_count else None,
                    "zero_rmse": float(np.sqrt(np.mean(pred_flat[zero_mask] ** 2))) if zero_count else None,
                }
            )
            pos_y = y_flat[pos_mask]
            pos_pred = pred_flat[pos_mask]
            pos_ae = abs_flat[pos_mask]
            pos_sum = float(pos_y.sum())
            nonzero_rows.append(
                {
                    "model": display,
                    "penetration": rate,
                    "nonzero_count": int(pos_mask.sum()),
                    "nonzero_mae": float(pos_ae.mean()),
                    "nonzero_rmse": float(np.sqrt(np.mean((pos_pred - pos_y) ** 2))),
                    "nonzero_wape": float(pos_ae.sum() / pos_sum * 100.0) if pos_sum else None,
                    "mape": float(np.mean(pos_ae / pos_y) * 100.0),
                }
            )
            bias_rows.append(
                {
                    "model": display,
                    "penetration": rate,
                    "mean_signed_error": float(signed.mean()),
                    "median_signed_error": float(np.median(signed)),
                    "total_signed_error": float(signed.sum()),
                    "overprediction_fraction": float((signed > 0).mean()),
                    "underprediction_fraction": float((signed < 0).mean()),
                    "exact_zero_error_fraction": float((signed == 0).mean()),
                }
            )

    audit_df = pd.DataFrame(audits)
    final_df = pd.DataFrame(final_rows)
    if final_df.empty:
        raise RuntimeError("no valid runs to analyze")

    rank_metrics = ("test_mae", "wape", "rmse", "mape_nonzero")
    for rate in RATES:
        subset = final_df[final_df["penetration"] == rate].copy()
        for metric in rank_metrics:
            ordered = subset.sort_values(metric, kind="mergesort")
            for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
                ranking_rows.append(
                    {
                        "penetration": rate,
                        "metric": metric,
                        "rank": rank,
                        "model": row["model"],
                        "value": row[metric],
                        "test_mae": row["test_mae"],
                        "wape": row["wape"],
                        "rmse": row["rmse"],
                        "mape_nonzero": row["mape_nonzero"],
                        "train_time_sec": row["train_time_sec"],
                    }
                )
    rank_df = pd.DataFrame(ranking_rows)
    mae_rank = rank_df[rank_df["metric"] == "test_mae"].copy()
    consistency_rows = []
    for display in DISPLAY.values():
        ranks = mae_rank[mae_rank["model"] == display]["rank"].tolist()
        consistency_rows.append(
            {
                "model": display,
                "rank1_count": ranks.count(1),
                "rank2_count": ranks.count(2),
                "rank3_count": ranks.count(3),
                "mean_rank": float(np.mean(ranks)) if ranks else float("nan"),
                "n_rates": len(ranks),
            }
        )
    consistency_df = pd.DataFrame(consistency_rows).sort_values("mean_rank", kind="mergesort")

    trend_rows = []
    nonmonotonic = []
    for display in DISPLAY.values():
        sub = final_df[final_df["model"] == display].set_index("penetration")
        for metric in ("test_mae", "rmse", "wape"):
            values = [float(sub.loc[rate, metric]) for rate in RATES]
            diffs = np.diff(values)
            if np.any(diffs > 1e-12):
                for i, delta in enumerate(diffs):
                    if delta > 1e-12:
                        nonmonotonic.append(
                            {
                                "model": display,
                                "metric": metric,
                                "from": RATES[i],
                                "to": RATES[i + 1],
                                "before": values[i],
                                "after": values[i + 1],
                                "delta": float(delta),
                            }
                        )
        for i in range(len(RATES) - 1):
            before = float(sub.loc[RATES[i], "test_mae"])
            after = float(sub.loc[RATES[i + 1], "test_mae"])
            delta = before - after
            trend_rows.append(
                {
                    "model": display,
                    "from_to": f"{RATES[i]}→{RATES[i + 1]}",
                    "mae_before": before,
                    "mae_after": after,
                    "delta_mae": delta,
                    "improvement_pct": delta / before * 100.0 if before else float("nan"),
                }
            )
        p05 = float(sub.loc["p05", "test_mae"])
        p70 = float(sub.loc["p70", "test_mae"])
        trend_rows.append(
            {
                "model": display,
                "from_to": "p05→p70",
                "mae_before": p05,
                "mae_after": p70,
                "delta_mae": p05 - p70,
                "improvement_pct": (p05 - p70) / p05 * 100.0,
            }
        )
    trend_df = pd.DataFrame(trend_rows)

    efficiency_rows = []
    timing_anomalies = []
    for display in DISPLAY.values():
        sub = final_df[final_df["model"] == display]
        times = sub["train_time_sec"].astype(float).tolist()
        secs = sub["sec_per_epoch"].astype(float).tolist()
        epochs = sub["epochs_run"].astype(float).tolist()
        mean_sec = float(np.mean(secs))
        std_sec = float(np.std(secs, ddof=1)) if len(secs) > 1 else 0.0
        cv = std_sec / mean_sec if mean_sec else float("nan")
        stopped = sub[sub["early_stopped"] == True]  # noqa: E712
        mean_stop = float(stopped["epochs_run"].mean()) if not stopped.empty else None
        params = int(sub["params"].iloc[0])
        peak = float(sub["peak_gpu_mb"].max())
        for _, row in sub.iterrows():
            if mean_sec and abs(float(row["sec_per_epoch"]) - mean_sec) / mean_sec > 0.20:
                timing_anomalies.append(
                    {
                        "model": display,
                        "penetration": row["penetration"],
                        "sec_per_epoch": float(row["sec_per_epoch"]),
                        "model_mean": mean_sec,
                        "rel_dev": abs(float(row["sec_per_epoch"]) - mean_sec) / mean_sec,
                    }
                )
        efficiency_rows.append(
            {
                "model": display,
                "mean_train_time_sec": float(np.mean(times)),
                "min_train_time_sec": float(np.min(times)),
                "max_train_time_sec": float(np.max(times)),
                "mean_sec_per_epoch": mean_sec,
                "std_sec_per_epoch": std_sec,
                "cv_sec_per_epoch": cv,
                "params": params,
                "peak_gpu_mb": peak,
                "mean_epochs": float(np.mean(epochs)),
                "mean_early_stop_epoch": mean_stop,
                "n_early_stopped": int(stopped.shape[0]),
            }
        )
    efficiency_df = pd.DataFrame(efficiency_rows)

    model_summary_rows = []
    for display in DISPLAY.values():
        sub = final_df[final_df["model"] == display]
        p05_mae = float(sub.loc[sub["penetration"] == "p05", "test_mae"].iloc[0])
        p70_mae = float(sub.loc[sub["penetration"] == "p70", "test_mae"].iloc[0])
        model_summary_rows.append(
            {
                "model": display,
                "mean_mae": float(sub["test_mae"].mean()),
                "p05_mae": p05_mae,
                "p70_mae": p70_mae,
                "p05_to_p70_improvement_pct": (p05_mae - p70_mae) / p05_mae * 100.0,
                "mean_wape": float(sub["wape"].mean()),
                "mean_train_time_sec": float(sub["train_time_sec"].mean()),
                "mean_sec_per_epoch": float(sub["sec_per_epoch"].mean()),
                "params": int(sub["params"].iloc[0]),
            }
        )
    model_summary_df = pd.DataFrame(model_summary_rows)

    edge_wape_df = pd.DataFrame(edge_wape_rows)
    edge_summary_df = pd.DataFrame(edge_summary_rows)
    hourly_df = pd.DataFrame(hourly_rows)
    flow_bin_df = pd.DataFrame(flow_bin_rows)
    zero_df = pd.DataFrame(zero_rows)
    nonzero_df = pd.DataFrame(nonzero_rows)
    bias_df = pd.DataFrame(bias_rows)
    gap_df = pd.DataFrame(gap_rows)
    conv_df = pd.DataFrame(conv_rows)
    edge_bias_df = pd.DataFrame(edge_bias_rows)
    hourly_bias_df = pd.DataFrame(hourly_bias_rows)
    edge_corr_df = pd.DataFrame(edge_flow_corr_rows)
    edge_q_df = pd.DataFrame(edge_quartile_rows)
    hourly_corr_df = pd.DataFrame(hourly_corr_rows)

    extreme_rows = []
    for display in DISPLAY.values():
        for rate in RATES:
            sub = edge_wape_df[(edge_wape_df["model"] == display) & (edge_wape_df["penetration"] == rate)]
            valid = sub[sub["valid"] == True].copy()  # noqa: E712
            if valid.empty:
                continue
            easiest = valid.nsmallest(5, "wape")
            hardest = valid.nlargest(5, "wape")
            for kind, chunk in (("lowest", easiest), ("highest", hardest)):
                for rank, (_, row) in enumerate(chunk.iterrows(), start=1):
                    extreme_rows.append(
                        {
                            "model": display,
                            "penetration": rate,
                            "kind": kind,
                            "rank": rank,
                            "edge_id": row["edge_id"],
                            "true_flow_sum": row["true_flow_sum"],
                            "wape": row["wape"],
                        }
                    )
    extreme_df = pd.DataFrame(extreme_rows)

    def _save(df: pd.DataFrame, name: str) -> Path:
        dest = output_dir / name
        df.to_csv(dest, index=False)
        return dest

    _save(audit_df, "completion_audit.csv")
    _save(final_df, "final_results_56.csv")
    _save(efficiency_df, "model_efficiency_summary.csv")
    _save(rank_df, "model_rankings.csv")
    _save(edge_wape_df, "edge_wape.csv")
    _save(edge_summary_df, "edge_wape_summary.csv")
    _save(extreme_df, "edge_wape_extremes.csv")
    _save(hourly_df, "hourly_wape.csv")
    _save(flow_bin_df, "flow_bin_metrics.csv")
    _save(zero_df, "zero_flow_metrics.csv")
    _save(nonzero_df, "nonzero_flow_metrics.csv")
    _save(edge_bias_df, "edge_bias.csv")
    _save(hourly_bias_df, "hourly_bias.csv")
    _save(conv_df, "convergence_summary.csv")
    _save(gap_df, "validation_test_gap.csv")
    _save(trend_df, "mae_penetration_trends.csv")
    _save(consistency_df, "top_model_consistency.csv")
    _save(model_summary_df, "model_level_summary.csv")
    _save(bias_df, "signed_bias_summary.csv")
    _save(edge_corr_df, "edge_flow_wape_correlation.csv")
    _save(edge_q_df, "edge_flow_quartile_wape.csv")
    _save(hourly_corr_df, "hourly_flow_wape_correlation.csv")
    if nonmonotonic:
        _save(pd.DataFrame(nonmonotonic), "nonmonotonic_metrics.csv")
    else:
        _save(pd.DataFrame(columns=["model", "metric", "from", "to", "before", "after", "delta"]), "nonmonotonic_metrics.csv")
    if timing_anomalies:
        _save(pd.DataFrame(timing_anomalies), "timing_anomalies.csv")
    else:
        _save(pd.DataFrame(columns=["model", "penetration", "sec_per_epoch", "model_mean", "rel_dev"]), "timing_anomalies.csv")

    plot_mae_wape(final_df, figures_dir)
    plot_efficiency(efficiency_df, figures_dir)
    plot_p05_edge_wape(edge_wape_df, figures_dir)
    plot_hourly(hourly_df, "p05", figures_dir / "p05_hourly_wape.png")
    plot_hourly(hourly_df, "p70", figures_dir / "p70_hourly_wape.png")
    plot_scatter(model_summary_df, figures_dir / "accuracy_efficiency_scatter_p05.png")

    n_ok = int(audit_df["problem"].eq("").sum()) if "problem" in audit_df else 0
    # empty problem string means ok; also require prediction/metrics/timing
    n_ok = int(
        (
            (audit_df["status"] == "completed")
            & (audit_df["prediction_valid"] == True)  # noqa: E712
            & (audit_df["metrics_valid"] == True)  # noqa: E712
            & (audit_df["timing_valid"] == True)  # noqa: E712
            & (audit_df["problem"].fillna("") == "")
        ).sum()
    )

    summary = {
        "n_expected": 56,
        "n_audited": int(len(audit_df)),
        "n_completed_ok": n_ok,
        "all_56_ok": n_ok == 56,
        "n_analyzed": int(len(final_df)),
        "sanity_failures": sanity_failures,
        "hourly_source": "test_predictions.npz:target_slot",
        "node_mapping": str(node_mapping),
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "training_results_modified": False,
        "nonmonotonic": nonmonotonic,
        "timing_anomalies": timing_anomalies,
        "wrote_training_results": False,
        "cuda_used": False,
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return summary


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def plot_mae_wape(final_df: pd.DataFrame, figures_dir: Path) -> None:
    x = list(range(len(RATES)))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for display in DISPLAY.values():
        ys = [float(final_df[(final_df["model"] == display) & (final_df["penetration"] == rate)]["test_mae"].iloc[0]) for rate in RATES]
        ax.plot(x, ys, marker="o", label=display, color=MODEL_COLORS[display])
    ax.set_xticks(x, RATES)
    ax.set_xlabel("Penetration")
    ax.set_ylabel("Test MAE")
    ax.set_title("Test MAE vs penetration")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "mae_vs_penetration.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for display in DISPLAY.values():
        ys = [float(final_df[(final_df["model"] == display) & (final_df["penetration"] == rate)]["wape"].iloc[0]) for rate in RATES]
        ax.plot(x, ys, marker="o", label=display, color=MODEL_COLORS[display])
    ax.set_xticks(x, RATES)
    ax.set_xlabel("Penetration")
    ax.set_ylabel("Test WAPE (%)")
    ax.set_title("Test WAPE vs penetration")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "wape_vs_penetration.png", dpi=150)
    plt.close(fig)


def plot_efficiency(efficiency_df: pd.DataFrame, figures_dir: Path) -> None:
    names = efficiency_df["model"].tolist()
    colors = [MODEL_COLORS[name] for name in names]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.bar(names, efficiency_df["mean_train_time_sec"], color=colors)
    ax.set_ylabel("Mean train wall time (s)")
    ax.set_title("Mean training time by model")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(figures_dir / "mean_train_time_by_model.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.bar(names, efficiency_df["mean_sec_per_epoch"], color=colors)
    ax.set_ylabel("Mean seconds / epoch")
    ax.set_title("Mean sec/epoch by model")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(figures_dir / "mean_sec_per_epoch_by_model.png", dpi=150)
    plt.close(fig)


def plot_p05_edge_wape(edge_wape_df: pd.DataFrame, figures_dir: Path) -> None:
    sub = edge_wape_df[(edge_wape_df["penetration"] == "p05") & (edge_wape_df["valid"] == True)]  # noqa: E712
    data = [sub[sub["model"] == name]["wape"].astype(float).tolist() for name in DISPLAY.values()]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bp = ax.boxplot(data, tick_labels=list(DISPLAY.values()), patch_artist=True, showfliers=True)
    for patch, name in zip(bp["boxes"], DISPLAY.values()):
        patch.set_facecolor(MODEL_COLORS[name])
        patch.set_alpha(0.6)
    ax.set_ylabel("Edge WAPE (%)")
    ax.set_title("p05 edge-wise WAPE")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(figures_dir / "p05_edge_wape_boxplot.png", dpi=150)
    plt.close(fig)


def plot_hourly(hourly_df: pd.DataFrame, rate: str, dest: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    hours = list(range(24))
    for display in DISPLAY.values():
        sub = hourly_df[(hourly_df["model"] == display) & (hourly_df["penetration"] == rate)].sort_values("hour")
        ys = sub["wape"].astype(float).tolist()
        ax.plot(hours, ys, marker="o", markersize=3.5, label=display, color=MODEL_COLORS[display])
    ax.set_xticks(hours)
    ax.set_xlabel("Hour of day (hour 0 is partial)")
    ax.set_ylabel("Hourly WAPE (%)")
    ax.set_title(f"{rate} hourly WAPE")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)


def plot_scatter(model_summary_df: pd.DataFrame, dest: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for _, row in model_summary_df.iterrows():
        ax.scatter(row["mean_train_time_sec"], row["p05_mae"], s=70, color=MODEL_COLORS[row["model"]])
        ax.annotate(row["model"], (row["mean_train_time_sec"], row["p05_mae"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Mean train time (s)")
    ax.set_ylabel("p05 Test MAE")
    ax.set_title("Accuracy–efficiency scatter (p05 MAE vs mean train time)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)


def main() -> None:
    output_dir = data_root() / "analysis_results"
    assert_not_code_root_output(output_dir, what="analysis_results")
    summary = analyze_all(output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
