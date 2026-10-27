#!/usr/bin/env python3
"""CPU-only Historical Mean baseline: edge × time-of-day train-day prior.

Does not train a network, does not use GPU, does not read observed flow,
does not write under training_results or prepared NPZ directories.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

CODE_ROOT = Path("/home/tzhang174/TrafficFlow")
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from data_root import (  # noqa: E402
    assert_not_code_root_output,
    data_root,
    node_table,
    rate_training_results_dir,
    shared_training_data_dir,
    subgraph_trajectories_dir,
)
from reimplementation.common.metrics.traffic_metrics import traffic_metrics  # noqa: E402


@dataclass
class NodeMapping:
    edge_ids: list[str]
    index_by_edge: dict[str, int]


def load_node_mapping(path: Path, expected_nodes: int) -> NodeMapping:
    if not path.is_file():
        raise HistoricalMeanError(f"r_nodes.csv not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "node_index" not in reader.fieldnames or "edge_id" not in reader.fieldnames:
            raise HistoricalMeanError(f"{path} must contain node_index and edge_id")
        rows = list(reader)
    parsed = sorted((int(row["node_index"]), str(row["edge_id"])) for row in rows)
    if [index for index, _edge in parsed] != list(range(len(parsed))):
        raise HistoricalMeanError(f"{path} node_index is not contiguous from 0")
    edge_ids = [edge for _index, edge in parsed]
    if len(edge_ids) != expected_nodes or len(set(edge_ids)) != len(edge_ids):
        raise HistoricalMeanError(f"{path} has {len(edge_ids)} nodes, expected {expected_nodes}")
    return NodeMapping(edge_ids=edge_ids, index_by_edge={edge: index for index, edge in enumerate(edge_ids)})


def build_daily_flow_matrix(
    path: Path,
    mapping: NodeMapping,
    *,
    n_slots: int,
    window_seconds: int,
    expected_day: str | None = None,
    expected_percent: int | None = None,
) -> np.ndarray:
    grid = np.zeros((n_slots, len(mapping.edge_ids)), dtype=np.int64)
    filled = np.zeros(grid.shape, dtype=bool)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise HistoricalMeanError(f"{path} has no header")
        missing = [name for name in ("window_index", "edge_id", "vehicle_count") if name not in reader.fieldnames]
        if missing:
            raise HistoricalMeanError(f"{path} missing fields {missing}")
        for row in reader:
            edge_id = row["edge_id"]
            if edge_id not in mapping.index_by_edge:
                raise HistoricalMeanError(f"{path} unknown edge_id {edge_id}")
            node_index = mapping.index_by_edge[edge_id]
            if expected_day is not None and row.get("day") not in {None, "", expected_day}:
                raise HistoricalMeanError(f"{path} contains day {row.get('day')!r}, expected {expected_day}")
            if expected_percent is not None and row.get("penetration_rate") not in {None, ""}:
                if abs(float(row["penetration_rate"]) - expected_percent / 100.0) > 1e-12:
                    raise HistoricalMeanError(f"{path} penetration_rate {row['penetration_rate']}")
            slot = int(row["window_index"])
            if slot < 0 or slot >= n_slots or filled[slot, node_index]:
                raise HistoricalMeanError(f"{path} bad or duplicate window {slot} edge {edge_id}")
            if row.get("window_start") not in {None, ""} and int(row["window_start"]) != slot * window_seconds:
                raise HistoricalMeanError(f"{path} window_start mismatch at slot {slot}")
            grid[slot, node_index] = int(row["vehicle_count"])
            filled[slot, node_index] = True
    if not filled.all():
        raise HistoricalMeanError(f"{path} is missing {int((~filled).sum())} cells")
    return grid

RATES = ("p05", "p10", "p20", "p30", "p40", "p50", "p70")
SPLITS = ("train", "validation", "test")
N_SLOTS = 288
N_EDGES = 56
N_HIS = 12
SLOT_SECONDS = 300
TRAIN_DAYS = 14
TEST_SAMPLES = 831
VAL_SAMPLES = 831
TRAIN_SAMPLES = 3878
TARGETS_PER_DAY = 277
MIN_TARGET_SLOT = N_HIS - 1
MAX_TARGET_SLOT = N_SLOTS - 1
SLOTS_PER_HOUR = 12
MAE_ATOL = 1e-5
MAE_RTOL = 1e-6
WAPE_ATOL = 1e-6
WAPE_RTOL = 1e-6
FLOW_BINS: tuple[tuple[str, str, Any, bool], ...] = (
    ("bin0_y_eq_0", "y == 0", lambda y: y == 0, False),
    ("bin1_0_lt_y_le_2", "0 < y <= 2", lambda y: (y > 0) & (y <= 2), True),
    ("bin2_2_lt_y_le_5", "2 < y <= 5", lambda y: (y > 2) & (y <= 5), True),
    ("bin3_5_lt_y_le_10", "5 < y <= 10", lambda y: (y > 5) & (y <= 10), True),
    ("bin4_y_gt_10", "y > 10", lambda y: y > 10, True),
)


class HistoricalMeanError(RuntimeError):
    pass


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _squeeze_tn(array: np.ndarray, expected_t: int) -> np.ndarray:
    arr = np.asarray(array)
    squeezed = np.squeeze(arr)
    if squeezed.ndim == 2 and squeezed.shape == (expected_t, N_EDGES):
        return squeezed.astype(np.float64, copy=False)
    if squeezed.ndim == 2 and squeezed.shape == (N_EDGES, expected_t):
        return squeezed.T.astype(np.float64, copy=False)
    raise HistoricalMeanError(f"unexpected array shape {arr.shape} -> {squeezed.shape}")


def _finite_array(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise HistoricalMeanError(f"{name} contains NaN or Inf")


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def load_split_npz(data_root: Path, rate: str, split: str) -> dict[str, np.ndarray]:
    path = data_root / rate / f"{split}.npz"
    if not path.is_file() or path.stat().st_size <= 0:
        raise HistoricalMeanError(f"missing NPZ: {path}")
    with np.load(path, allow_pickle=False) as handle:
        required = ("y_raw", "target_slot", "day_index")
        missing = [key for key in required if key not in handle.files]
        if missing:
            raise HistoricalMeanError(f"{path} missing {missing}")
        return {key: np.array(handle[key]) for key in handle.files}


def assert_y_raw_identical(prepared_root: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"identical": True, "splits": {}}
    for split, expected in (("train", TRAIN_SAMPLES), ("validation", VAL_SAMPLES), ("test", TEST_SAMPLES)):
        reference = None
        reference_rate = None
        for rate in RATES:
            loaded = load_split_npz(prepared_root, rate, split)
            y_raw = _squeeze_tn(loaded["y_raw"], expected)
            target_slot = np.asarray(loaded["target_slot"]).reshape(-1)
            day_index = np.asarray(loaded["day_index"]).reshape(-1)
            if y_raw.shape[0] != expected:
                raise HistoricalMeanError(f"{rate}/{split} y_raw rows {y_raw.shape[0]} != {expected}")
            if target_slot.shape[0] != expected or day_index.shape[0] != expected:
                raise HistoricalMeanError(f"{rate}/{split} metadata length mismatch")
            if reference is None:
                reference = {
                    "y_raw": y_raw,
                    "target_slot": target_slot.astype(np.int64, copy=False),
                    "day_index": day_index.astype(np.int64, copy=False),
                }
                reference_rate = rate
                continue
            if not np.array_equal(y_raw, reference["y_raw"]):
                report["identical"] = False
                raise HistoricalMeanError(
                    f"{split} y_raw differs between {reference_rate} and {rate}; stopping"
                )
            if not np.array_equal(target_slot.astype(np.int64), reference["target_slot"]):
                raise HistoricalMeanError(f"{split} target_slot differs between {reference_rate} and {rate}")
            if not np.array_equal(day_index.astype(np.int64), reference["day_index"]):
                raise HistoricalMeanError(f"{split} day_index differs between {reference_rate} and {rate}")
        assert reference is not None
        report["splits"][split] = {
            "rows": expected,
            "reference_rate": reference_rate,
            "compared_rates": list(RATES),
        }
    return report


def load_train_full_flow(
    *,
    trajectories_root: Path,
    mapping,
    train_day_names: list[str],
) -> np.ndarray:
    """Return [14, 288, 56] raw full flow. Validation/test days are never opened."""
    forbidden = {f"day_{index:02d}" for index in range(15, 21)}
    overlap = set(train_day_names) & forbidden
    if overlap:
        raise HistoricalMeanError(f"refusing to load non-train days into HM: {sorted(overlap)}")
    if len(train_day_names) != TRAIN_DAYS:
        raise HistoricalMeanError(f"expected {TRAIN_DAYS} train days, got {train_day_names}")
    stack = np.zeros((TRAIN_DAYS, N_SLOTS, N_EDGES), dtype=np.float64)
    for i, day_name in enumerate(train_day_names):
        path = trajectories_root / day_name / "edge_flow_5min.csv"
        if not path.is_file():
            raise HistoricalMeanError(f"missing train full-flow CSV: {path}")
        matrix = build_daily_flow_matrix(
            path,
            mapping,
            n_slots=N_SLOTS,
            window_seconds=SLOT_SECONDS,
            expected_day=day_name,
            expected_percent=None,
        )
        stack[i] = matrix.astype(np.float64, copy=False)
    return stack


def verify_npz_against_train_csv(
    train_npz: dict[str, np.ndarray],
    train_stack: np.ndarray,
) -> None:
    y_raw = _squeeze_tn(train_npz["y_raw"], TRAIN_SAMPLES)
    slots = np.asarray(train_npz["target_slot"]).reshape(-1).astype(np.int64)
    days = np.asarray(train_npz["day_index"]).reshape(-1).astype(np.int64)
    if days.min() < 0 or days.max() >= TRAIN_DAYS:
        raise HistoricalMeanError(f"train day_index outside 0..{TRAIN_DAYS - 1}: {days.min()}..{days.max()}")
    reconstructed = train_stack[days, slots, :]
    if not np.allclose(y_raw, reconstructed, rtol=0.0, atol=1e-6):
        max_err = float(np.max(np.abs(y_raw - reconstructed)))
        raise HistoricalMeanError(f"train NPZ y_raw does not match train CSV full flow; max abs err={max_err}")


def predict_from_template(target_slot: np.ndarray, template: np.ndarray) -> np.ndarray:
    slots = np.asarray(target_slot).reshape(-1).astype(np.int64)
    if slots.min() < MIN_TARGET_SLOT or slots.max() > MAX_TARGET_SLOT:
        raise HistoricalMeanError(f"target_slot range {int(slots.min())}..{int(slots.max())} != 11..287")
    return template[slots]


def edge_wape_rows(
    pred: np.ndarray,
    y: np.ndarray,
    edge_ids: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for index, edge_id in enumerate(edge_ids):
        true_sum = float(y[:, index].sum())
        abs_sum = float(np.abs(pred[:, index] - y[:, index]).sum())
        wape = (abs_sum / true_sum * 100.0) if true_sum > 0.0 else float("nan")
        rows.append(
            {
                "edge_index": index,
                "edge_id": edge_id,
                "true_flow_sum": true_sum,
                "absolute_error_sum": abs_sum,
                "WAPE": wape,
            }
        )
    return rows


def wape_summary(values: np.ndarray) -> dict[str, float]:
    valid = values[np.isfinite(values)]
    return {
        "n_valid": int(valid.size),
        "n_undefined": int(values.size - valid.size),
        "mean": float(valid.mean()) if valid.size else float("nan"),
        "median": float(np.median(valid)) if valid.size else float("nan"),
        "p90": percentile(valid, 90),
        "min": float(valid.min()) if valid.size else float("nan"),
        "max": float(valid.max()) if valid.size else float("nan"),
    }


def hourly_wape_rows(pred: np.ndarray, y: np.ndarray, target_slot: np.ndarray) -> list[dict[str, Any]]:
    hours = target_slot // SLOTS_PER_HOUR
    rows = []
    abs_err = np.abs(pred - y)
    for hour in range(24):
        mask = hours == hour
        n_samples = int(mask.sum())
        true_sum = float(y[mask].sum()) if n_samples else 0.0
        abs_sum = float(abs_err[mask].sum()) if n_samples else 0.0
        wape = (abs_sum / true_sum * 100.0) if n_samples and true_sum > 0.0 else float("nan")
        rows.append(
            {
                "hour": hour,
                "num_samples": n_samples,
                "true_flow_sum": true_sum,
                "absolute_error_sum": abs_sum,
                "WAPE": wape,
                "partial_hour": hour == 0,
            }
        )
    return rows


def flow_bin_rows(pred: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    y_flat = y.reshape(-1)
    pred_flat = pred.reshape(-1)
    rows = []
    for bin_id, label, predicate, mape_ok in FLOW_BINS:
        mask = predicate(y_flat)
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin": bin_id,
                    "label": label,
                    "count": 0,
                    "MAE": None,
                    "RMSE": None,
                    "mean_true_flow": None,
                    "mean_prediction": None,
                    "MAPE": None,
                }
            )
            continue
        yt = y_flat[mask]
        pt = pred_flat[mask]
        ae = np.abs(pt - yt)
        rows.append(
            {
                "bin": bin_id,
                "label": label,
                "count": count,
                "MAE": float(ae.mean()),
                "RMSE": float(np.sqrt(np.mean((pt - yt) ** 2))),
                "mean_true_flow": float(yt.mean()),
                "mean_prediction": float(pt.mean()),
                "MAPE": float(np.mean(ae / yt) * 100.0) if mape_ok else None,
            }
        )
    return rows


def residual_stats(residual: np.ndarray) -> dict[str, float]:
    flat = residual.reshape(-1)
    return {
        "count": int(flat.size),
        "residual_mae": float(np.mean(np.abs(flat))),
        "residual_rmse": float(np.sqrt(np.mean(flat**2))),
        "residual_mean": float(flat.mean()),
        "residual_std": float(flat.std(ddof=0)),
    }


def load_official_test_metrics(model: str, rate: str) -> dict[str, float]:
    path = rate_training_results_dir(model, rate, seed=42) / "test_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "mae": float(payload["mae"]),
        "rmse": float(payload["rmse"]),
        "mape_nonzero": float(payload["mape_nonzero"]),
        "wape": float(payload["wape"]),
    }


def load_official_predictions(model: str, rate: str) -> tuple[np.ndarray, np.ndarray]:
    path = rate_training_results_dir(model, rate, seed=42) / "test_predictions.npz"
    with np.load(path, allow_pickle=False) as handle:
        pred = _squeeze_tn(handle["pred_raw"], TEST_SAMPLES)
        y = _squeeze_tn(handle["y_raw"], TEST_SAMPLES)
    return pred, y


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    hm = payload["test_metrics"]
    stid = payload["stid"]
    zero = payload["zero_nonzero"]["zero"]
    nonzero = payload["zero_nonzero"]["nonzero"]
    train_var = payload["train_variability"]
    residual = payload["residual"]
    diagnosis = payload["diagnosis"]
    lines = [
        "# Historical Mean baseline",
        "",
        "## A. Definition",
        "",
        "For each edge `n` and slot-of-day `s`:",
        "",
        "```text",
        "HM[s, n] = mean_{d in train days 1..14} full_flow[d, s, n]",
        "         = (1/14) * sum_{d=1..14} full_flow[d, s, n]",
        "```",
        "",
        "Test/validation prediction for any day `d` is `pred[d, s, n] = HM[s, n]`.",
        "",
        "- Historical mean template was constructed using the 14 training days only.",
        "- Validation and test full-flow CSVs were not opened.",
        "- Observed trajectories, penetration, 12-step history, adjacency, and DOW were not used.",
        "- No parameters were trained. CPU only; no GPU; no checkpoints.",
        "- Official evaluation uses target_slot 11..287 (831 test samples), matching the 8 neural baselines.",
        "",
        "HM does not depend on penetration. The same prediction is compared against every STID rate.",
        "",
        "## B. Main Result (official test, raw scale)",
        "",
        "| Baseline | MAE | RMSE | MAPE_nonzero | WAPE |",
        "|---|---:|---:|---:|---:|",
        f"| Historical Mean | {fmt(hm['mae'])} | {fmt(hm['rmse'])} | {fmt(hm['mape_nonzero'], 2)} | {fmt(hm['wape'], 2)} |",
        "",
        f"Validation MAE (same protocol, not used to fit HM): {fmt(payload['validation_metrics']['mae'])}.",
        "",
        "## C. Historical Mean vs STID",
        "",
        "STID metrics were read from `training_results/stid/pXX/seed_42/test_metrics.json`.",
        "",
        "| p | Historical Mean MAE | STID MAE | STID improvement over HM (%) |",
        "|---|---:|---:|---:|",
    ]
    for rate in RATES:
        row = stid[rate]
        lines.append(
            f"| {rate} | {fmt(hm['mae'])} | {fmt(row['mae'])} | {fmt(row['mae_improvement_pct'], 2)} |"
        )
    lines += [
        "",
        "| p | Historical Mean WAPE | STID WAPE | STID WAPE improvement over HM (%) |",
        "|---|---:|---:|---:|",
    ]
    for rate in RATES:
        row = stid[rate]
        lines.append(
            f"| {rate} | {fmt(hm['wape'], 2)} | {fmt(row['wape'], 2)} | {fmt(row['wape_improvement_pct'], 2)} |"
        )
    lines += [
        "",
        "Historical Mean does not depend on penetration; the HM column is repeated for comparison only.",
        "",
        "## D. Zero / Nonzero",
        "",
        "| subset | count | fraction | MAE | RMSE | extra |",
        "|---|---:|---:|---:|---:|---|",
        (
            f"| y == 0 | {zero['count']} | {fmt(zero['fraction'], 4)} | {fmt(zero['mae'])} | "
            f"{fmt(zero['rmse'])} | mean pred={fmt(zero['mean_prediction'])} |"
        ),
        (
            f"| y > 0 | {nonzero['count']} | {fmt(nonzero['fraction'], 4)} | {fmt(nonzero['mae'])} | "
            f"{fmt(nonzero['rmse'])} | WAPE={fmt(nonzero['wape'], 2)}, MAPE_nonzero={fmt(nonzero['mape_nonzero'], 2)} |"
        ),
        "",
        "## E. Flow bins",
        "",
        "| Bin | count | MAE | RMSE | mean true | mean pred | MAPE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["flow_bins"]:
        lines.append(
            f"| {row['label']} | {row['count']} | {fmt(row['MAE'])} | {fmt(row['RMSE'])} | "
            f"{fmt(row['mean_true_flow'])} | {fmt(row['mean_prediction'])} | {fmt(row['MAPE'], 2)} |"
        )
    edge_s = payload["edge_wape_summary"]
    lines += [
        "",
        "## F. Edge-wise WAPE",
        "",
        f"- mean={fmt(edge_s['mean'], 2)}, median={fmt(edge_s['median'], 2)}, "
        f"P90={fmt(edge_s['p90'], 2)}, min={fmt(edge_s['min'], 2)}, max={fmt(edge_s['max'], 2)}",
        f"- STID p05 edge-wise WAPE: mean={fmt(payload['stid_p05_edge_wape']['mean'], 2)}, "
        f"median={fmt(payload['stid_p05_edge_wape']['median'], 2)}, "
        f"P90={fmt(payload['stid_p05_edge_wape']['p90'], 2)}",
        f"- STAEformer p05 edge-wise WAPE: mean={fmt(payload['staeformer_p05_edge_wape']['mean'], 2)}, "
        f"median={fmt(payload['staeformer_p05_edge_wape']['median'], 2)}, "
        f"P90={fmt(payload['staeformer_p05_edge_wape']['p90'], 2)}",
        "",
        "Highest HM edge WAPE:",
        "",
        "| edge_index | edge_id | WAPE | true_flow_sum |",
        "|---:|---|---:|---:|",
    ]
    for row in payload["edge_wape_highest"]:
        lines.append(
            f"| {row['edge_index']} | {row['edge_id']} | {fmt(row['WAPE'], 2)} | {fmt(row['true_flow_sum'], 1)} |"
        )
    lines += [
        "",
        "## G. Hourly WAPE",
        "",
        "| hour | samples | WAPE | partial_hour |",
        "|---:|---:|---:|---|",
    ]
    for row in payload["hourly_wape"]:
        lines.append(
            f"| {row['hour']} | {row['num_samples']} | {fmt(row['WAPE'], 2)} | {str(row['partial_hour']).lower()} |"
        )
    lines += [
        "",
        "## H. Train-day variability (14 training days, not a test metric)",
        "",
        "For each (slot, edge), `std` is the population std (`ddof=0`) of the 14 train-day full-flow values.",
        "CV = std / mean; undefined when mean = 0.",
        "",
        f"- mean std = {fmt(train_var['mean_std'])}",
        f"- median std = {fmt(train_var['median_std'])}",
        f"- mean valid CV = {fmt(train_var['mean_valid_cv'])}",
        f"- median valid CV = {fmt(train_var['median_valid_cv'])}",
        f"- P90 CV = {fmt(train_var['p90_valid_cv'])}",
        f"- valid CV cells = {train_var['n_valid_cv']} / {train_var['n_cells']}",
        "",
        "Highest mean-std edges (average over 288 slots):",
        "",
        "| edge_index | edge_id | mean_std | median_cv |",
        "|---:|---|---:|---:|",
    ]
    for row in payload["train_variability_highest_edges"]:
        lines.append(
            f"| {row['edge_index']} | {row['edge_id']} | {fmt(row['mean_std'])} | {fmt(row['median_valid_cv'])} |"
        )
    lines += [
        "",
        "Highest mean-std hours:",
        "",
        "| hour | mean_std |",
        "|---:|---:|",
    ]
    for row in payload["train_variability_highest_hours"]:
        lines.append(f"| {row['hour']} | {fmt(row['mean_std'])} |")
    lines += [
        "",
        "## I. Historical residual (test y − HM[target_slot])",
        "",
        f"- residual MAE = {fmt(residual['overall']['residual_mae'])}",
        f"- residual RMSE = {fmt(residual['overall']['residual_rmse'])}",
        f"- residual mean = {fmt(residual['overall']['residual_mean'])}",
        f"- residual std = {fmt(residual['overall']['residual_std'])}",
        "",
        "Residual MAE equals test MAE by construction.",
        "",
        "## J. Diagnosis",
        "",
    ]
    for index, item in enumerate(diagnosis, start=1):
        lines.append(f"{index}. {item}")
        lines.append("")
    lines += [
        "## Files",
        "",
    ]
    for name in payload["output_files"]:
        lines.append(f"- `{name}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    prepared_root = shared_training_data_dir()
    output_dir = data_root() / "analysis_results" / "historical_mean"
    assert_not_code_root_output(output_dir, what="historical_mean")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise HistoricalMeanError(
            f"{output_dir} already has files; refusing to overwrite. "
            "Move or empty that subdirectory first."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    split_manifest = json.loads((prepared_root / "split_manifest.json").read_text(encoding="utf-8"))
    train_day_names = list(split_manifest["training_days"])
    if train_day_names != [f"day_{i:02d}" for i in range(1, 15)]:
        raise HistoricalMeanError(f"unexpected training_days: {train_day_names}")
    if split_manifest["test_days"] != ["day_18", "day_19", "day_20"]:
        raise HistoricalMeanError(f"unexpected test_days: {split_manifest['test_days']}")

    mapping = load_node_mapping(node_table(), N_EDGES)
    identity = assert_y_raw_identical(prepared_root)

    train_npz = load_split_npz(prepared_root, "p05", "train")
    val_npz = load_split_npz(prepared_root, "p05", "validation")
    test_npz = load_split_npz(prepared_root, "p05", "test")

    train_stack = load_train_full_flow(
        trajectories_root=subgraph_trajectories_dir(),
        mapping=mapping,
        train_day_names=train_day_names,
    )
    verify_npz_against_train_csv(train_npz, train_stack)

    template = train_stack.mean(axis=0)
    if template.shape != (N_SLOTS, N_EDGES):
        raise HistoricalMeanError(f"HM shape {template.shape} != {(N_SLOTS, N_EDGES)}")
    _finite_array("HM template", template)

    test_y = _squeeze_tn(test_npz["y_raw"], TEST_SAMPLES)
    test_slots = np.asarray(test_npz["target_slot"]).reshape(-1).astype(np.int64)
    test_days = np.asarray(test_npz["day_index"]).reshape(-1).astype(np.int64)
    val_y = _squeeze_tn(val_npz["y_raw"], VAL_SAMPLES)
    val_slots = np.asarray(val_npz["target_slot"]).reshape(-1).astype(np.int64)

    unique_days, day_counts = np.unique(test_days, return_counts=True)
    if unique_days.tolist() != [17, 18, 19] or not np.all(day_counts == TARGETS_PER_DAY):
        raise HistoricalMeanError(f"unexpected test days/counts: days={unique_days.tolist()} counts={day_counts.tolist()}")
    if test_slots.min() != MIN_TARGET_SLOT or test_slots.max() != MAX_TARGET_SLOT:
        raise HistoricalMeanError("test target_slot is not 11..287")

    test_pred = predict_from_template(test_slots, template)
    val_pred = predict_from_template(val_slots, template)
    _finite_array("test pred", test_pred)
    _finite_array("test y", test_y)
    if test_pred.shape != (TEST_SAMPLES, N_EDGES) or test_y.shape != (TEST_SAMPLES, N_EDGES):
        raise HistoricalMeanError(f"pred/y shape {test_pred.shape} {test_y.shape}")

    test_metrics = traffic_metrics(test_pred, test_y, clip_negative=False)
    val_metrics = traffic_metrics(val_pred, val_y, clip_negative=False)
    recomputed_mae = float(np.mean(np.abs(test_pred - test_y)))
    true_sum = float(np.sum(np.abs(test_y)))
    recomputed_wape = float(np.sum(np.abs(test_pred - test_y)) / true_sum * 100.0)
    if not math.isclose(recomputed_mae, float(test_metrics["mae"]), rel_tol=MAE_RTOL, abs_tol=MAE_ATOL):
        raise HistoricalMeanError("offline MAE does not match traffic_metrics")
    if not math.isclose(recomputed_wape, float(test_metrics["wape"]), rel_tol=WAPE_RTOL, abs_tol=WAPE_ATOL):
        raise HistoricalMeanError("offline WAPE does not match traffic_metrics")

    y_flat = test_y.reshape(-1)
    pred_flat = test_pred.reshape(-1)
    zero_mask = y_flat == 0
    pos_mask = y_flat > 0
    zero_count = int(zero_mask.sum())
    pos_count = int(pos_mask.sum())
    zero_block = {
        "count": zero_count,
        "fraction": float(zero_count / y_flat.size),
        "mae": float(np.mean(np.abs(pred_flat[zero_mask] - y_flat[zero_mask]))) if zero_count else None,
        "rmse": float(np.sqrt(np.mean(pred_flat[zero_mask] ** 2))) if zero_count else None,
        "mean_prediction": float(pred_flat[zero_mask].mean()) if zero_count else None,
    }
    pos_ae = np.abs(pred_flat[pos_mask] - y_flat[pos_mask])
    pos_sum = float(y_flat[pos_mask].sum())
    nonzero_block = {
        "count": pos_count,
        "fraction": float(pos_count / y_flat.size),
        "mae": float(pos_ae.mean()) if pos_count else None,
        "rmse": float(np.sqrt(np.mean((pred_flat[pos_mask] - y_flat[pos_mask]) ** 2))) if pos_count else None,
        "wape": float(pos_ae.sum() / pos_sum * 100.0) if pos_sum else None,
        "mape_nonzero": float(np.mean(pos_ae / y_flat[pos_mask]) * 100.0) if pos_count else None,
    }

    hm_edge_rows = edge_wape_rows(test_pred, test_y, mapping.edge_ids)
    hm_edge_wapes = np.asarray([row["WAPE"] for row in hm_edge_rows], dtype=np.float64)
    stid_p05_pred, stid_p05_y = load_official_predictions("stid", "p05")
    if not np.allclose(stid_p05_y, test_y, rtol=0.0, atol=1e-6):
        raise HistoricalMeanError("STID p05 y_raw does not match prepared test y_raw")
    stae_p05_pred, stae_p05_y = load_official_predictions("staeformer", "p05")
    if not np.allclose(stae_p05_y, test_y, rtol=0.0, atol=1e-6):
        raise HistoricalMeanError("STAEformer p05 y_raw does not match prepared test y_raw")
    stid_edge = wape_summary(np.asarray([row["WAPE"] for row in edge_wape_rows(stid_p05_pred, stid_p05_y, mapping.edge_ids)]))
    stae_edge = wape_summary(np.asarray([row["WAPE"] for row in edge_wape_rows(stae_p05_pred, stae_p05_y, mapping.edge_ids)]))

    hourly_rows = hourly_wape_rows(test_pred, test_y, test_slots)
    bins = flow_bin_rows(test_pred, test_y)

    residual = test_y - test_pred
    residual_overall = residual_stats(residual)
    if not math.isclose(residual_overall["residual_mae"], float(test_metrics["mae"]), rel_tol=MAE_RTOL, abs_tol=MAE_ATOL):
        raise HistoricalMeanError("residual MAE != test MAE")

    residual_rows: list[dict[str, Any]] = [
        {"scope": "overall", "key": "all", **residual_overall}
    ]
    for index, edge_id in enumerate(mapping.edge_ids):
        stats = residual_stats(residual[:, index])
        residual_rows.append({"scope": "edge", "key": edge_id, "edge_index": index, **stats})
    hours = test_slots // SLOTS_PER_HOUR
    for hour in range(24):
        mask = hours == hour
        stats = residual_stats(residual[mask]) if int(mask.sum()) else {
            "count": 0,
            "residual_mae": float("nan"),
            "residual_rmse": float("nan"),
            "residual_mean": float("nan"),
            "residual_std": float("nan"),
        }
        residual_rows.append({"scope": "hour", "key": str(hour), **stats})
    y_flat_res = test_y.reshape(-1)
    res_flat = residual.reshape(-1)
    for bin_id, label, predicate, _mape in FLOW_BINS:
        mask = predicate(y_flat_res)
        stats = residual_stats(res_flat[mask]) if int(mask.sum()) else {
            "count": 0,
            "residual_mae": float("nan"),
            "residual_rmse": float("nan"),
            "residual_mean": float("nan"),
            "residual_std": float("nan"),
        }
        residual_rows.append({"scope": "flow_bin", "key": bin_id, "label": label, **stats})

    train_mean = template
    train_std = train_stack.std(axis=0, ddof=0)
    cv = np.full_like(train_mean, np.nan, dtype=np.float64)
    valid_mean = train_mean > 0.0
    cv[valid_mean] = train_std[valid_mean] / train_mean[valid_mean]
    valid_cv = cv[np.isfinite(cv)]
    train_var = {
        "mean_std": float(train_std.mean()),
        "median_std": float(np.median(train_std)),
        "mean_valid_cv": float(valid_cv.mean()) if valid_cv.size else float("nan"),
        "median_valid_cv": float(np.median(valid_cv)) if valid_cv.size else float("nan"),
        "p90_valid_cv": percentile(valid_cv, 90),
        "n_valid_cv": int(valid_cv.size),
        "n_undefined_cv": int(cv.size - valid_cv.size),
        "n_cells": int(cv.size),
        "std_ddof": 0,
        "note": "std/CV are over the 14 training days at each (slot, edge); not a test metric",
    }

    variability_rows: list[dict[str, Any]] = [
        {
            "scope": "overall",
            "key": "all",
            "mean_std": train_var["mean_std"],
            "median_std": train_var["median_std"],
            "mean_valid_cv": train_var["mean_valid_cv"],
            "median_valid_cv": train_var["median_valid_cv"],
            "p90_valid_cv": train_var["p90_valid_cv"],
            "n_valid_cv": train_var["n_valid_cv"],
            "n_cells": train_var["n_cells"],
        }
    ]
    mean_flat = train_mean.reshape(-1)
    std_flat = train_std.reshape(-1)
    cv_flat = cv.reshape(-1)
    for bin_id, label, predicate, _mape in FLOW_BINS:
        mask = predicate(mean_flat)
        std_m = std_flat[mask]
        cv_m = cv_flat[mask]
        cv_valid = cv_m[np.isfinite(cv_m)]
        variability_rows.append(
            {
                "scope": "flow_bin_of_train_mean",
                "key": bin_id,
                "label": label,
                "count": int(mask.sum()),
                "mean_std": float(std_m.mean()) if mask.any() else float("nan"),
                "median_std": float(np.median(std_m)) if mask.any() else float("nan"),
                "mean_valid_cv": float(cv_valid.mean()) if cv_valid.size else float("nan"),
                "median_valid_cv": float(np.median(cv_valid)) if cv_valid.size else float("nan"),
                "p90_valid_cv": percentile(cv_valid, 90),
                "n_valid_cv": int(cv_valid.size),
            }
        )
    edge_var_rows = []
    for index, edge_id in enumerate(mapping.edge_ids):
        std_e = train_std[:, index]
        cv_e = cv[:, index]
        cv_valid = cv_e[np.isfinite(cv_e)]
        row = {
            "scope": "edge",
            "key": edge_id,
            "edge_index": index,
            "edge_id": edge_id,
            "mean_std": float(std_e.mean()),
            "median_std": float(np.median(std_e)),
            "mean_valid_cv": float(cv_valid.mean()) if cv_valid.size else float("nan"),
            "median_valid_cv": float(np.median(cv_valid)) if cv_valid.size else float("nan"),
            "p90_valid_cv": percentile(cv_valid, 90),
            "n_valid_cv": int(cv_valid.size),
        }
        variability_rows.append(row)
        edge_var_rows.append(row)
    hour_var_rows = []
    for hour in range(24):
        sl = slice(hour * SLOTS_PER_HOUR, (hour + 1) * SLOTS_PER_HOUR)
        std_h = train_std[sl]
        row = {
            "scope": "hour",
            "key": str(hour),
            "hour": hour,
            "mean_std": float(std_h.mean()),
            "median_std": float(np.median(std_h)),
            "partial_hour": hour == 0,
        }
        variability_rows.append(row)
        hour_var_rows.append(row)

    stid_compare: dict[str, Any] = {}
    vs_rows = []
    hm_mae = float(test_metrics["mae"])
    hm_wape = float(test_metrics["wape"])
    for rate in RATES:
        official = load_official_test_metrics("stid", rate)
        mae_imp = (hm_mae - official["mae"]) / hm_mae * 100.0
        wape_imp = (hm_wape - official["wape"]) / hm_wape * 100.0
        stid_compare[rate] = {
            **official,
            "mae_improvement_pct": mae_imp,
            "wape_improvement_pct": wape_imp,
        }
        vs_rows.append(
            {
                "penetration": rate,
                "historical_mean_mae": hm_mae,
                "stid_mae": official["mae"],
                "stid_mae_improvement_pct": mae_imp,
                "historical_mean_wape": hm_wape,
                "stid_wape": official["wape"],
                "stid_wape_improvement_pct": wape_imp,
                "historical_mean_independent_of_penetration": True,
            }
        )

    highest_edges = sorted(hm_edge_rows, key=lambda row: -float(row["WAPE"] if math.isfinite(row["WAPE"]) else -1.0))[:5]
    highest_var_edges = sorted(edge_var_rows, key=lambda row: -row["mean_std"])[:5]
    highest_var_hours = sorted(hour_var_rows, key=lambda row: -row["mean_std"])[:5]

    p05_imp = stid_compare["p05"]["mae_improvement_pct"]
    p70_imp = stid_compare["p70"]["mae_improvement_pct"]
    p05_gap = hm_mae - stid_compare["p05"]["mae"]
    p70_gap = hm_mae - stid_compare["p70"]["mae"]
    residual_std = residual_overall["residual_std"]
    mean_cv = train_var["mean_valid_cv"]

    diagnosis = [
        (
            f"Historical Mean test MAE is {hm_mae:.4f}. STID p05 MAE is {stid_compare['p05']['mae']:.4f} "
            f"(STID minus HM = {p05_gap:.4f}; STID improvement vs HM = {p05_imp:.2f}%). "
            + (
                "Historical Mean is already better than STID p05."
                if p05_imp < 0
                else "STID p05 is better than Historical Mean."
            )
        ),
        (
            f"STID p05 vs Historical Mean: MAE {stid_compare['p05']['mae']:.4f} vs {hm_mae:.4f}, "
            f"relative improvement {p05_imp:.2f}% "
            f"({'negative means STID is worse' if p05_imp < 0 else 'positive means STID is better'})."
        ),
        (
            f"STID p70 MAE is {stid_compare['p70']['mae']:.4f} versus HM {hm_mae:.4f}; "
            f"absolute gap {p70_gap:.4f}, relative improvement {p70_imp:.2f}%."
        ),
        (
            f"Train-day mean std at a fixed (slot, edge) is {train_var['mean_std']:.4f} vehicles / 5 min; "
            f"median std {train_var['median_std']:.4f}; mean valid CV {mean_cv:.4f}. "
            + (
                "This is a strong edge × time-of-day repeat pattern: the same cell is relatively stable across the 14 train days."
                if mean_cv < 0.35
                else "Cross-day CV is moderate to large; the TOD template does not fully determine a day."
            )
        ),
        (
            f"Test residual vs the train TOD template has MAE {residual_overall['residual_mae']:.4f}, "
            f"mean {residual_overall['residual_mean']:.4f}, std {residual_std:.4f}."
        ),
        (
            "Largest train-day std edges: "
            + ", ".join(
                f"{row['edge_id']} (mean std {row['mean_std']:.3f})"
                for row in highest_var_edges[:3]
            )
            + ". Largest train-day std hours: "
            + ", ".join(f"{row['hour']:02d}:00 ({row['mean_std']:.3f})" for row in highest_var_hours[:3])
            + "."
        ),
        (
            f"The 14-day TOD template already reaches test MAE {hm_mae:.4f} and WAPE {hm_wape:.2f} "
            f"without seeing the test day. Combined with mean valid CV {mean_cv:.4f} and residual std {residual_std:.4f}, "
            + (
                "the current 20-day synthetic series shows relatively weak day-to-day deviation from a repeating diurnal template."
                if mean_cv < 0.35 and residual_std < hm_mae * 1.5
                else "day-to-day deviation is visible; repeating TOD structure is present but not enough to call randomness 'very weak' from CV/residual alone."
            )
        ),
    ]

    sanity = {
        "pred_shape": list(test_pred.shape),
        "y_shape": list(test_y.shape),
        "shapes_ok": test_pred.shape == (TEST_SAMPLES, N_EDGES) and test_y.shape == (TEST_SAMPLES, N_EDGES),
        "no_nan_inf": bool(np.isfinite(test_pred).all() and np.isfinite(test_y).all() and np.isfinite(template).all()),
        "target_slot_min": int(test_slots.min()),
        "target_slot_max": int(test_slots.max()),
        "samples_per_test_day": day_counts.tolist(),
        "n_test_days": int(unique_days.size),
        "full_flow_targets_identical_across_penetration": True,
        "hm_train_days_only": True,
        "train_days": train_day_names,
        "opened_full_flow_csv_days": train_day_names,
        "did_not_open_val_test_full_flow": True,
        "did_not_use_observed_or_penetration": True,
        "test_prediction_independent_of_penetration": True,
        "offline_mae_matches_traffic_metrics": True,
        "offline_wape_matches_traffic_metrics": True,
        "clip_negative": False,
        "statement": "Historical mean template was constructed using the 14 training days only.",
    }

    metrics_payload = {
        "baseline": "historical_mean_edge_time_of_day",
        "historical_mean_independent_of_penetration": True,
        "leakage": sanity,
        "y_raw_identity": identity,
        "template_shape": [N_SLOTS, N_EDGES],
        "train_day_count": TRAIN_DAYS,
        "test_metrics": test_metrics,
        "validation_metrics": val_metrics,
        "recomputed_mae": recomputed_mae,
        "recomputed_wape": recomputed_wape,
        "zero_nonzero": {"zero": zero_block, "nonzero": nonzero_block},
        "flow_bins": bins,
        "edge_wape_summary": wape_summary(hm_edge_wapes),
        "stid_p05_edge_wape": stid_edge,
        "staeformer_p05_edge_wape": stae_edge,
        "hourly_wape": hourly_rows,
        "residual": {"overall": residual_overall},
        "train_variability": train_var,
        "stid": stid_compare,
        "edge_wape_highest": highest_edges,
        "train_variability_highest_edges": highest_var_edges,
        "train_variability_highest_hours": highest_var_hours,
        "diagnosis": diagnosis,
    }

    np.save(output_dir / "historical_mean_template.npy", template.astype(np.float64, copy=False))
    np.savez(
        output_dir / "historical_mean_predictions.npz",
        pred_raw=test_pred.astype(np.float32),
        y_raw=test_y.astype(np.float32),
        target_slot=test_slots.astype(np.int32),
        day_index=test_days.astype(np.int32),
        edge_ids=np.asarray(mapping.edge_ids),
        sample_index=np.arange(TEST_SAMPLES, dtype=np.int32),
    )
    _write_csv(
        output_dir / "historical_mean_edge_wape.csv",
        ["edge_index", "edge_id", "true_flow_sum", "absolute_error_sum", "WAPE"],
        hm_edge_rows,
    )
    _write_csv(
        output_dir / "historical_mean_hourly_wape.csv",
        ["hour", "num_samples", "true_flow_sum", "absolute_error_sum", "WAPE", "partial_hour"],
        hourly_rows,
    )
    _write_csv(
        output_dir / "historical_mean_flow_bins.csv",
        ["bin", "label", "count", "MAE", "RMSE", "mean_true_flow", "mean_prediction", "MAPE"],
        bins,
    )
    _write_csv(
        output_dir / "historical_mean_zero_nonzero.csv",
        ["subset", "count", "fraction", "MAE", "RMSE", "mean_prediction", "WAPE", "MAPE_nonzero"],
        [
            {
                "subset": "y_eq_0",
                "count": zero_block["count"],
                "fraction": zero_block["fraction"],
                "MAE": zero_block["mae"],
                "RMSE": zero_block["rmse"],
                "mean_prediction": zero_block["mean_prediction"],
                "WAPE": "",
                "MAPE_nonzero": "",
            },
            {
                "subset": "y_gt_0",
                "count": nonzero_block["count"],
                "fraction": nonzero_block["fraction"],
                "MAE": nonzero_block["mae"],
                "RMSE": nonzero_block["rmse"],
                "mean_prediction": "",
                "WAPE": nonzero_block["wape"],
                "MAPE_nonzero": nonzero_block["mape_nonzero"],
            },
        ],
    )
    residual_fields = [
        "scope",
        "key",
        "edge_index",
        "label",
        "count",
        "residual_mae",
        "residual_rmse",
        "residual_mean",
        "residual_std",
    ]
    _write_csv(output_dir / "historical_mean_residual_summary.csv", residual_fields, residual_rows)
    variability_fields = [
        "scope",
        "key",
        "edge_index",
        "edge_id",
        "hour",
        "label",
        "count",
        "mean_std",
        "median_std",
        "mean_valid_cv",
        "median_valid_cv",
        "p90_valid_cv",
        "n_valid_cv",
        "n_cells",
        "partial_hour",
    ]
    _write_csv(output_dir / "historical_mean_train_variability.csv", variability_fields, variability_rows)
    _write_csv(
        output_dir / "historical_mean_vs_stid.csv",
        [
            "penetration",
            "historical_mean_mae",
            "stid_mae",
            "stid_mae_improvement_pct",
            "historical_mean_wape",
            "stid_wape",
            "stid_wape_improvement_pct",
            "historical_mean_independent_of_penetration",
        ],
        vs_rows,
    )

    output_files = sorted(p.name for p in output_dir.iterdir() if p.is_file())
    metrics_payload["output_files"] = [str(output_dir / name) for name in output_files]
    # report.md is written after this list; include it explicitly
    report_path = output_dir / "report.md"
    metrics_payload["output_files"] = sorted(
        {*(metrics_payload["output_files"]), str(report_path), str(output_dir / "historical_mean_metrics.json")}
    )
    write_report(report_path, metrics_payload)
    _write_json(output_dir / "historical_mean_metrics.json", {**metrics_payload, "sanity": sanity})

    print(json.dumps(
        {
            "status": "ok",
            "output_dir": str(output_dir),
            "test_mae": hm_mae,
            "test_rmse": float(test_metrics["rmse"]),
            "test_mape_nonzero": float(test_metrics["mape_nonzero"]),
            "test_wape": hm_wape,
            "statement": sanity["statement"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HistoricalMeanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
