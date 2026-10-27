#!/usr/bin/env python3
"""Cut Xuancheng26 inflow into same-day reconstruction windows.

X is 12 partial-inflow steps. Y is the full inflow at the last of those
steps. Windows never cross midnight. Normalization is fit on training
windows only, after the windows have been cut.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

import numpy as np

PARTIAL_ROOT = Path(
    "/data5/tzhang174/TrafficFlow/xuancheng26/training_data/partial_flow"
)
FULL_ROOT = Path(
    "/data5/tzhang174/TrafficFlow/xuancheng26/training_data/full_flow/r-level"
)
GRAPH_PATH = Path(
    "/data5/tzhang174/TrafficFlow/xuancheng26/training_data/graph/r_graph.npz"
)
OUT_ROOT = Path(
    "/data5/tzhang174/TrafficFlow/xuancheng26/training_data/window_data"
)
RATES = ("p05", "p10", "p20", "p30", "p40", "p50", "p70")
TRAIN_DAYS = (
    "2023_04_01", "2023_04_02", "2023_04_03", "2023_04_04", "2023_04_05",
    "2023_04_06", "2023_04_07", "2023_04_08", "2023_04_10", "2023_04_11",
    "2023_04_12", "2023_04_13", "2023_04_14", "2023_04_15", "2023_04_16",
    "2023_04_17", "2023_04_18", "2023_04_19", "2023_04_20", "2023_04_21",
)
VAL_DAYS = ("2023_04_22", "2023_04_23", "2023_04_24", "2023_04_25")
TEST_DAYS = ("2023_04_26", "2023_04_27", "2023_04_28", "2023_04_29", "2023_04_30")
SPLIT_DAYS = {"train": TRAIN_DAYS, "validation": VAL_DAYS, "test": TEST_DAYS}
N_BINS = 288
N_HIS = 12
N_NODES = 1406
N_WINDOWS = N_BINS - N_HIS + 1
DDOF = 0
NPZ_KEYS = (
    "x",
    "y",
    "x_raw",
    "y_raw",
    "day_index",
    "window_start_slot",
    "target_slot",
)


def log(message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"{stamp} {message}", flush=True)


def weekday_index(day_name: str) -> int:
    """Monday=0 ... Sunday=6. ``value % 7`` is this same weekday."""
    year, month, day = (int(part) for part in day_name.split("_"))
    return int(date(year, month, day).weekday())


def split_of(day_name: str) -> str:
    for name, days in SPLIT_DAYS.items():
        if day_name in days:
            return name
    raise KeyError(day_name)


def load_edge_ids() -> list[str]:
    with np.load(GRAPH_PATH, allow_pickle=False) as payload:
        values = payload["edge_ids"]
        count = int(payload["num_nodes"])
    edge_ids = [str(item) for item in values.tolist()]
    if len(edge_ids) != N_NODES or count != N_NODES or len(set(edge_ids)) != N_NODES:
        raise RuntimeError(f"expected {N_NODES} unique edge ids, found {len(edge_ids)}")
    return edge_ids


def load_aligned_matrix(path: Path, edge_ids: list[str]) -> np.ndarray:
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split(",")
    if header[:2] != ["time_bin", "start_time"]:
        raise RuntimeError(f"{path} header does not start with time_bin,start_time")
    columns = header[2:]
    if len(columns) != N_NODES:
        raise RuntimeError(f"{path} has {len(columns)} edges, expected {N_NODES}")
    if set(columns) != set(edge_ids):
        missing = sorted(set(edge_ids) - set(columns))[:8]
        extra = sorted(set(columns) - set(edge_ids))[:8]
        raise RuntimeError(f"{path} edge set mismatch missing={missing} extra={extra}")
    table = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if table.shape != (N_BINS, N_NODES + 2):
        raise RuntimeError(f"{path} shape {table.shape}")
    if not np.array_equal(table[:, 0], np.arange(N_BINS)) or not np.array_equal(table[:, 1], np.arange(N_BINS) * 300):
        raise RuntimeError(f"{path} time axis is not 288 five-minute bins")
    values = table[:, 2:]
    if not np.isfinite(values).all() or np.any(values < 0):
        raise RuntimeError(f"{path} contains NaN, inf, or a negative count")
    if columns != edge_ids:
        order = [columns.index(edge_id) for edge_id in edge_ids]
        values = values[:, order]
    if not np.all(values == np.rint(values)):
        raise RuntimeError(f"{path} contains a non-integer count")
    return values.astype(np.int64, copy=False)


def make_windows(partial: np.ndarray, full: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if partial.shape != (N_BINS, N_NODES) or full.shape != partial.shape:
        raise RuntimeError(f"daily matrix shape {partial.shape} / {full.shape}")
    starts = np.arange(N_WINDOWS, dtype=np.int32)
    targets = starts + (N_HIS - 1)
    history = starts[:, None] + np.arange(N_HIS, dtype=np.int32)[None, :]
    x_raw = partial[history].astype(np.float32)[:, :, :, None]
    y_raw = full[targets].astype(np.float32)[:, None, :, None]
    if x_raw.shape != (N_WINDOWS, N_HIS, N_NODES, 1) or y_raw.shape != (N_WINDOWS, 1, N_NODES, 1):
        raise RuntimeError("window shape drifted from the reconstruction definition")
    if not np.array_equal(targets, starts + 11):
        raise RuntimeError("target_slot is not window_start_slot + 11")
    for start in (0, 1, 100, N_WINDOWS - 1):
        target = int(starts[start] + 11)
        if not np.array_equal(y_raw[start, 0, :, 0], full[target].astype(np.float32)):
            raise RuntimeError(f"Y at start {start} is not full[{target}]")
        if target + 1 < N_BINS and np.array_equal(y_raw[start, 0, :, 0], full[target + 1].astype(np.float32)):
            raise RuntimeError(f"Y at start {start} accidentally used full[{target + 1}]")
    return x_raw, y_raw, starts, targets


def process_day(task: tuple) -> dict[str, object]:
    day_name, edge_ids = task
    try:
        full = load_aligned_matrix(FULL_ROOT / day_name / "entry_flow.csv", edge_ids)
        partials = []
        x_raw: dict[str, np.ndarray] = {}
        y_raw = None
        starts = None
        targets = None
        for rate in RATES:
            partial = load_aligned_matrix(
                PARTIAL_ROOT / rate / day_name / "inflow.csv",
                edge_ids,
            )
            partials.append(partial)
            x_day, y_day, starts, targets = make_windows(partial, full)
            x_raw[rate] = x_day
            if y_raw is None:
                y_raw = y_day
            elif not np.array_equal(y_raw, y_day):
                raise RuntimeError(f"{day_name} {rate} y_raw differs from the shared full target")
        for left, right in zip(partials, partials[1:]):
            if np.any(left > right):
                raise RuntimeError(f"{day_name} partial inflow nesting failed")
        if np.any(partials[-1] > full):
            raise RuntimeError(f"{day_name} p70 inflow exceeds full inflow")
        log(f"[{day_name}] raw windows ok")
        return {
            "date": day_name,
            "ok": True,
            "weekday": weekday_index(day_name),
            "x_raw": x_raw,
            "y_raw": y_raw,
            "window_start_slot": starts,
            "target_slot": targets,
        }
    except Exception:
        detail = traceback.format_exc(limit=6).strip().replace("\n", " | ")
        log(f"[{day_name}] FAIL {detail}")
        return {"date": day_name, "ok": False, "error": detail}


def scalar_stats(array: np.ndarray) -> tuple[float, float]:
    values = np.asarray(array, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=DDOF))
    if not np.isfinite(mean) or not np.isfinite(std):
        raise RuntimeError("non-finite normalization statistic")
    if std == 0.0:
        raise RuntimeError("standard deviation is 0")
    return mean, std


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_written(path: Path, weekday: int) -> None:
    with np.load(path, allow_pickle=False) as payload:
        missing = [key for key in NPZ_KEYS if key not in payload.files]
        if missing:
            raise RuntimeError(f"{path} missing {missing}")
        x = payload["x"]
        y = payload["y"]
        x_raw = payload["x_raw"]
        y_raw = payload["y_raw"]
        day_index = payload["day_index"]
        starts = payload["window_start_slot"]
        targets = payload["target_slot"]
    if x.shape != (N_WINDOWS, N_HIS, N_NODES, 1) or x_raw.shape != x.shape:
        raise RuntimeError(f"{path} x shape {x.shape}")
    if y.shape != (N_WINDOWS, 1, N_NODES, 1) or y_raw.shape != y.shape:
        raise RuntimeError(f"{path} y shape {y.shape}")
    for array in (x, y, x_raw, y_raw):
        if not np.isfinite(array).all():
            raise RuntimeError(f"{path} contains NaN or inf")
    if np.any(x_raw < 0) or np.any(y_raw < 0):
        raise RuntimeError(f"{path} raw window has a negative count")
    expected_starts = np.arange(N_WINDOWS, dtype=starts.dtype)
    if not np.array_equal(starts, expected_starts) or not np.array_equal(targets, expected_starts + 11):
        raise RuntimeError(f"{path} slot metadata is wrong")
    if not np.array_equal(day_index, np.full(N_WINDOWS, weekday, dtype=day_index.dtype)):
        raise RuntimeError(f"{path} day_index is not the calendar weekday")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Xuancheng26 same-day reconstruction windows")
    parser.add_argument("--workers", type=int, default=29)
    args = parser.parse_args(argv)
    edge_ids = load_edge_ids()
    days = TRAIN_DAYS + VAL_DAYS + TEST_DAYS
    if len(days) != 29 or len(set(days)) != 29:
        raise RuntimeError("split does not contain 29 unique dates")
    workers = max(1, min(args.workers, len(days)))
    log(f"dates={len(days)} workers={workers} nodes={len(edge_ids)}")
    results: dict[str, dict[str, object]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_day, (day_name, edge_ids)): day_name for day_name in days}
        for future in as_completed(futures):
            day_name = futures[future]
            try:
                results[day_name] = future.result()
            except Exception:
                detail = traceback.format_exc(limit=6).strip().replace("\n", " | ")
                log(f"[{day_name}] FAIL {detail}")
                results[day_name] = {"date": day_name, "ok": False, "error": detail}
    failed = [day for day in days if not results[day].get("ok")]
    if failed:
        for day in failed:
            log(f"FAILED {day} {results[day].get('error', '')}")
        return 2

    input_stats: dict[str, tuple[float, float]] = {}
    target_stats: tuple[float, float] | None = None
    for rate in RATES:
        train_x = np.concatenate([results[day]["x_raw"][rate] for day in TRAIN_DAYS], axis=0)
        if train_x.shape[0] != 20 * N_WINDOWS:
            raise RuntimeError(f"{rate} training sample count {train_x.shape[0]}")
        input_stats[rate] = scalar_stats(train_x)
        train_y = np.concatenate([results[day]["y_raw"] for day in TRAIN_DAYS], axis=0)
        stats = scalar_stats(train_y)
        if target_stats is None:
            target_stats = stats
        elif stats != target_stats:
            raise RuntimeError(f"{rate} target scaler differs from the shared full-flow scaler")
    assert target_stats is not None
    mean_y, std_y = target_stats

    written = 0
    for day in days:
        item = results[day]
        weekday = int(item["weekday"])
        starts = item["window_start_slot"]
        targets = item["target_slot"]
        day_index = np.full(N_WINDOWS, weekday, dtype=np.int32)
        for rate in RATES:
            mean_x, std_x = input_stats[rate]
            x_raw = item["x_raw"][rate]
            y_raw = item["y_raw"]
            arrays = {
                "x": ((x_raw - mean_x) / std_x).astype(np.float32, copy=False),
                "y": ((y_raw - mean_y) / std_y).astype(np.float32, copy=False),
                "x_raw": x_raw,
                "y_raw": y_raw,
                "day_index": day_index,
                "window_start_slot": starts,
                "target_slot": targets,
            }
            path = OUT_ROOT / rate / split_of(day) / f"{day}.npz"
            write_npz(path, arrays)
            verify_written(path, weekday)
            written += 1
        log(f"[{day}] wrote 7 npz weekday={weekday}")

    for rate, (mean_x, std_x) in input_stats.items():
        write_json(
            OUT_ROOT / rate / "normalization.json",
            {
                "mean_x": mean_x,
                "std_x": std_x,
                "mean_y_full": mean_y,
                "std_y_full": std_y,
                "ddof": DDOF,
                "fit_split": "train",
                "training_days": list(TRAIN_DAYS),
                "formula": "(value - mean) / std",
                "scope": "global_scalar",
                "feature": "inflow",
            },
        )
    write_json(
        OUT_ROOT / "split_manifest.json",
        {
            "train": list(TRAIN_DAYS),
            "validation": list(VAL_DAYS),
            "test": list(TEST_DAYS),
            "samples_per_day": N_WINDOWS,
            "train_samples_per_rate": 20 * N_WINDOWS,
            "validation_samples_per_rate": 4 * N_WINDOWS,
            "test_samples_per_rate": 5 * N_WINDOWS,
            "num_nodes": N_NODES,
            "n_his": N_HIS,
            "output_steps": 1,
            "target_mode": "last-observed-step",
            "label_definition": "Y is full inflow at window_start_slot + 11, not the next slot",
            "cross_day_windows_allowed": False,
            "day_index_encoding": "python_weekday_monday_0",
            "day_index_note": (
                "STID and STAEformer default loaders use day_index % 7 as the weekday. "
                "The value stored here is already that weekday, so a missing 2023-04-09 "
                "does not shift later days. It is not a 0-based position in the 29-day list. "
                "PDFormer's optional calendar mode indexes all_days by day_index and is not "
                "enabled. Metric tables that group by day_index will group equal weekdays together."
            ),
            "rates": list(RATES),
        },
    )
    log(
        f"files={written} train_days={len(TRAIN_DAYS)} val_days={len(VAL_DAYS)} "
        f"test_days={len(TEST_DAYS)} samples_per_rate="
        f"{20 * N_WINDOWS}/{4 * N_WINDOWS}/{5 * N_WINDOWS}"
    )
    log(f"mean_y_full={mean_y:.8f} std_y_full={std_y:.8f}")
    for rate, (mean_x, std_x) in input_stats.items():
        log(f"{rate} mean_x={mean_x:.8f} std_x={std_x:.8f}")
    if written == 203:
        log("Xuancheng26 R-level window dataset generated successfully.")
    return 0 if written == 203 else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
