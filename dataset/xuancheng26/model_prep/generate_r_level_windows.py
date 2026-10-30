#!/usr/bin/env python3
"""Cut Xuancheng26 states into same-day reconstruction windows.

Each window is 12 same-day 5-minute steps. Y is the full state at the last
of those steps. Legacy flat ``{split}/{date}.npz`` files and
``normalization.json`` are left untouched. New files go to
``{split}/{date}/{state}.npz``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

import numpy as np

import prep_paths

PARTIAL_ROOT = prep_paths.DEFAULT_TRAINING_ROOT / "partial_flow"
FULL_ROOT = prep_paths.DEFAULT_TRAINING_ROOT / "full_flow" / "r-level"
GRAPH_PATH = prep_paths.DEFAULT_TRAINING_ROOT / "graph" / "r_graph.npz"
OUT_ROOT = prep_paths.DEFAULT_TRAINING_ROOT / "window_data"
RATES = ("p05", "p10", "p20", "p30", "p40", "p50", "p70")
CHANNELS = ("inflow", "outflow", "accumulation_start", "accumulation_end")
STATE_FILES = CHANNELS + ("all",)
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


def _moment(array: np.ndarray) -> tuple[float, float, int]:
    values = np.asarray(array, dtype=np.float64)
    return float(values.sum()), float(np.square(values).sum()), int(values.size)


def _mean_std(total: float, total_sq: float, count: int) -> tuple[float, float]:
    if count <= 0:
        raise RuntimeError("empty normalization sample")
    mean = total / count
    variance = total_sq / count - mean * mean
    if -1e-8 < variance < 0.0:
        variance = 0.0
    std = float(np.sqrt(variance))
    if not np.isfinite(mean) or not np.isfinite(std) or std == 0.0:
        raise RuntimeError(f"invalid scaler mean={mean} std={std}")
    return float(mean), std


def _check_conservation(states: dict[str, np.ndarray], label: str) -> None:
    end = states["accumulation_start"] + states["inflow"] - states["outflow"]
    if not np.array_equal(end, states["accumulation_end"]):
        raise RuntimeError(f"{label} failed N_end = N_start + inflow - outflow")


def _load_states(directory: Path, edge_ids: list[str]) -> dict[str, np.ndarray]:
    return {name: load_aligned_matrix(directory / f"{name}.csv", edge_ids) for name in CHANNELS}


def _normalize(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((raw - mean.astype(np.float32)) / std.astype(np.float32)).astype(np.float32, copy=False)


def summarize_day(day_name: str) -> dict[str, object]:
    try:
        edge_ids = load_edge_ids()
        full = _load_states(FULL_ROOT / day_name, edge_ids)
        _check_conservation(full, f"full {day_name}")
        partials: dict[str, dict[str, np.ndarray]] = {}
        for rate in RATES:
            partial = _load_states(PARTIAL_ROOT / rate / day_name, edge_ids)
            _check_conservation(partial, f"{rate} {day_name}")
            partials[rate] = partial
        for left_rate, right_rate in zip(RATES, RATES[1:]):
            for name in CHANNELS:
                if np.any(partials[left_rate][name] > partials[right_rate][name]):
                    raise RuntimeError(f"{day_name} {name} nesting failed: {left_rate} > {right_rate}")
        for rate in RATES:
            for name in CHANNELS:
                if np.any(partials[rate][name] > full[name]):
                    raise RuntimeError(f"{day_name} {rate} {name} exceeds full")
        moments: dict[str, object] = {}
        reference_starts = None
        reference_targets = None
        for name in CHANNELS:
            y_raw = None
            rate_moments: dict[str, dict[str, float | int]] = {}
            starts = None
            targets = None
            for rate in RATES:
                x_raw, y_day, starts, targets = make_windows(partials[rate][name], full[name])
                if y_raw is None:
                    y_raw = y_day
                elif not np.array_equal(y_raw, y_day):
                    raise RuntimeError(f"{day_name} {name} {rate} full target differs across rates")
                x_sum, x_sq, x_count = _moment(x_raw)
                rate_moments[rate] = {"x_sum": x_sum, "x_sq": x_sq, "x_count": x_count}
                del x_raw
            if y_raw is None or starts is None or targets is None:
                raise RuntimeError(f"{day_name} {name} produced no windows")
            y_sum, y_sq, y_count = _moment(y_raw)
            if reference_starts is None:
                reference_starts = starts
                reference_targets = targets
            elif not np.array_equal(reference_starts, starts) or not np.array_equal(reference_targets, targets):
                raise RuntimeError(f"{day_name} {name} window index differs from inflow")
            moments[name] = {
                "rates": rate_moments,
                "y_sum": y_sum,
                "y_sq": y_sq,
                "y_count": y_count,
            }
        log(f"[{day_name}] moments ok")
        return {"date": day_name, "ok": True, "split": split_of(day_name), "moments": moments}
    except Exception:
        detail = traceback.format_exc(limit=6).strip().replace("\n", " | ")
        log(f"[{day_name}] FAIL {detail}")
        return {"date": day_name, "ok": False, "error": detail}


def _channel_scales(stats: dict[str, object], rate: str, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    input_stats = stats["input"]
    target_stats = stats["target"]
    mean_x = np.asarray([input_stats[rate][name][0] for name in names], dtype=np.float32)
    std_x = np.asarray([input_stats[rate][name][1] for name in names], dtype=np.float32)
    mean_y = np.asarray([target_stats[name][0] for name in names], dtype=np.float32)
    std_y = np.asarray([target_stats[name][1] for name in names], dtype=np.float32)
    return mean_x, std_x, mean_y, std_y


def _file_matches(
    path: Path,
    *,
    channels: int,
    weekday: int,
    mean_x: np.ndarray,
    std_x: np.ndarray,
    mean_y: np.ndarray,
    std_y: np.ndarray,
) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            missing = [key for key in NPZ_KEYS if key not in payload.files]
            if missing:
                return False
            x = payload["x"]
            y = payload["y"]
            x_raw = payload["x_raw"]
            y_raw = payload["y_raw"]
            day_index = payload["day_index"]
            starts = payload["window_start_slot"]
            targets = payload["target_slot"]
            if tuple(x.shape) != (N_WINDOWS, N_HIS, N_NODES, channels):
                return False
            if tuple(y.shape) != (N_WINDOWS, 1, N_NODES, channels):
                return False
            if tuple(x_raw.shape) != tuple(x.shape) or tuple(y_raw.shape) != tuple(y.shape):
                return False
            if not np.array_equal(starts, np.arange(N_WINDOWS, dtype=starts.dtype)):
                return False
            if not np.array_equal(targets, starts + 11):
                return False
            if not np.array_equal(day_index, np.full(N_WINDOWS, weekday, dtype=day_index.dtype)):
                return False
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                return False
            if np.any(x_raw < 0) or np.any(y_raw < 0):
                return False
            expected_x = _normalize(x_raw, mean_x.reshape(1, 1, 1, -1), std_x.reshape(1, 1, 1, -1))
            expected_y = _normalize(y_raw, mean_y.reshape(1, 1, 1, -1), std_y.reshape(1, 1, 1, -1))
            if not np.allclose(x, expected_x, rtol=0.0, atol=1e-5):
                return False
            if not np.allclose(y, expected_y, rtol=0.0, atol=1e-5):
                return False
    except Exception:
        return False
    return True


def write_day(task: tuple) -> dict[str, object]:
    day_name, stats = task
    try:
        edge_ids = load_edge_ids()
        weekday = weekday_index(day_name)
        split = split_of(day_name)
        full = _load_states(FULL_ROOT / day_name, edge_ids)
        written = 0
        skipped = 0
        reference_starts = None
        reference_targets = None
        for rate in RATES:
            partial = _load_states(PARTIAL_ROOT / rate / day_name, edge_ids)
            single: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            starts = None
            targets = None
            for name in CHANNELS:
                x_raw, y_raw, starts, targets = make_windows(partial[name], full[name])
                single[name] = (x_raw, y_raw)
            if starts is None or targets is None:
                raise RuntimeError(f"{day_name} {rate} produced no windows")
            if reference_starts is None:
                reference_starts = starts
                reference_targets = targets
            elif not np.array_equal(reference_starts, starts) or not np.array_equal(reference_targets, targets):
                raise RuntimeError(f"{day_name} {rate} window index drifted")
            x_all = np.concatenate([single[name][0] for name in CHANNELS], axis=-1)
            y_all = np.concatenate([single[name][1] for name in CHANNELS], axis=-1)
            bundles = {name: single[name] for name in CHANNELS}
            bundles["all"] = (x_all, y_all)
            day_dir = OUT_ROOT / rate / split / day_name
            if day_dir.exists() and not day_dir.is_dir():
                raise RuntimeError(f"{day_dir} exists and is not a directory")
            day_index = np.full(N_WINDOWS, weekday, dtype=np.int32)
            for state_name, (x_raw, y_raw) in bundles.items():
                names = CHANNELS if state_name == "all" else (state_name,)
                mean_x, std_x, mean_y, std_y = _channel_scales(stats, rate, names)
                path = day_dir / f"{state_name}.npz"
                channels = 4 if state_name == "all" else 1
                if _file_matches(
                    path,
                    channels=channels,
                    weekday=weekday,
                    mean_x=mean_x,
                    std_x=std_x,
                    mean_y=mean_y,
                    std_y=std_y,
                ):
                    skipped += 1
                    continue
                arrays = {
                    "x": _normalize(x_raw, mean_x.reshape(1, 1, 1, -1), std_x.reshape(1, 1, 1, -1)),
                    "y": _normalize(y_raw, mean_y.reshape(1, 1, 1, -1), std_y.reshape(1, 1, 1, -1)),
                    "x_raw": np.ascontiguousarray(x_raw),
                    "y_raw": np.ascontiguousarray(y_raw),
                    "day_index": day_index,
                    "window_start_slot": starts,
                    "target_slot": targets,
                }
                write_npz(path, arrays)
                if not _file_matches(
                    path,
                    channels=channels,
                    weekday=weekday,
                    mean_x=mean_x,
                    std_x=std_x,
                    mean_y=mean_y,
                    std_y=std_y,
                ):
                    raise RuntimeError(f"{path} failed verification after write")
                written += 1
        log(f"[{day_name}] wrote={written} skipped={skipped}")
        return {"date": day_name, "ok": True, "written": written, "skipped": skipped}
    except Exception:
        detail = traceback.format_exc(limit=6).strip().replace("\n", " | ")
        log(f"[{day_name}] FAIL {detail}")
        return {"date": day_name, "ok": False, "error": detail}


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


def _legacy_files(root: Path) -> list[Path]:
    protected = [
        root / "normalization.json",
        root / "split_manifest.json",
        root / "dataset_metadata.json",
        root / "validation_summary.json",
    ]
    for rate in RATES:
        protected.append(root / rate / "normalization.json")
        for split in SPLIT_DAYS:
            directory = root / rate / split
            if not directory.is_dir():
                continue
            protected.extend(path for path in directory.glob("*.npz") if path.is_file())
    return [path for path in protected if path.is_file()]


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_pool(fn, tasks: list, workers: int) -> list[dict[str, object]]:
    if workers <= 1:
        return [fn(task) for task in tasks]
    results: dict[str, dict[str, object]] = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
        futures = {pool.submit(fn, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            day_name = task if isinstance(task, str) else task[0]
            try:
                results[day_name] = future.result()
            except Exception:
                detail = traceback.format_exc(limit=6).strip().replace("\n", " | ")
                results[day_name] = {"date": day_name, "ok": False, "error": detail}
    ordered = []
    for task in tasks:
        day_name = task if isinstance(task, str) else task[0]
        ordered.append(results[day_name])
    return ordered


def _reduce_stats(rows: list[dict[str, object]]) -> dict[str, object]:
    input_totals: dict[str, dict[str, list[float]]] = {
        rate: {name: [0.0, 0.0, 0.0] for name in CHANNELS} for rate in RATES
    }
    target_totals = {name: [0.0, 0.0, 0.0] for name in CHANNELS}
    for row in rows:
        if row.get("split") != "train":
            continue
        moments = row["moments"]
        for name in CHANNELS:
            item = moments[name]
            target_totals[name][0] += float(item["y_sum"])
            target_totals[name][1] += float(item["y_sq"])
            target_totals[name][2] += float(item["y_count"])
            for rate in RATES:
                rate_item = item["rates"][rate]
                bucket = input_totals[rate][name]
                bucket[0] += float(rate_item["x_sum"])
                bucket[1] += float(rate_item["x_sq"])
                bucket[2] += float(rate_item["x_count"])
    return {
        "input": {
            rate: {name: _mean_std(*input_totals[rate][name]) for name in CHANNELS}
            for rate in RATES
        },
        "target": {name: _mean_std(*target_totals[name]) for name in CHANNELS},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Xuancheng26 five-state reconstruction windows")
    prep_paths.add_path_arguments(parser)
    parser.add_argument("--workers", type=int, default=29)
    args = parser.parse_args(argv)
    root = prep_paths.resolved(args.training_root)
    prep_paths.bind(
        sys.modules[__name__],
        PARTIAL_ROOT=root / "partial_flow",
        FULL_ROOT=root / "full_flow" / "r-level",
        GRAPH_PATH=root / "graph" / "r_graph.npz",
        OUT_ROOT=root / "window_data",
    )
    prep_paths.publish_runtime(training_root=root)
    log(f"training_root={root}")
    days = TRAIN_DAYS + VAL_DAYS + TEST_DAYS
    if len(days) != 29 or len(set(days)) != 29:
        raise RuntimeError("split does not contain 29 unique dates")
    legacy = _legacy_files(OUT_ROOT)
    legacy_hashes = {path.as_posix(): _sha256(path) for path in legacy}
    log(f"protected_files={len(legacy_hashes)}")
    workers = max(1, min(int(args.workers), len(days)))
    summary_rows = _run_pool(summarize_day, list(days), workers)
    failed = [row for row in summary_rows if not row.get("ok")]
    if failed:
        for row in failed:
            log(f"FAILED {row.get('date')} {row.get('error', '')}")
        return 2
    stats = _reduce_stats(summary_rows)
    inflow_mean, inflow_std = stats["input"]["p05"]["inflow"]
    log(f"p05 inflow mean_x={inflow_mean:.12f} std_x={inflow_std:.12f}")
    target_mean, target_std = stats["target"]["inflow"]
    log(f"full inflow mean_y={target_mean:.12f} std_y={target_std:.12f}")
    write_rows = _run_pool(write_day, [(day, stats) for day in days], workers)
    failed = [row for row in write_rows if not row.get("ok")]
    if failed:
        for row in failed:
            log(f"FAILED {row.get('date')} {row.get('error', '')}")
        return 2
    written = sum(int(row["written"]) for row in write_rows)
    skipped = sum(int(row["skipped"]) for row in write_rows)
    normalization = {
        "fit_split": "train",
        "ddof": DDOF,
        "formula": "(value - mean) / std",
        "scope": "global_scalar_per_variable",
        "channels": list(CHANNELS),
        "training_days": list(TRAIN_DAYS),
        "target": {
            name: {"mean": stats["target"][name][0], "std": stats["target"][name][1]}
            for name in CHANNELS
        },
        "input": {
            rate: {
                name: {"mean": stats["input"][rate][name][0], "std": stats["input"][rate][name][1]}
                for name in CHANNELS
            }
            for rate in RATES
        },
    }
    write_json(OUT_ROOT / "state_normalization.json", normalization)
    write_json(
        OUT_ROOT / "state_window_manifest.json",
        {
            "train": list(TRAIN_DAYS),
            "validation": list(VAL_DAYS),
            "test": list(TEST_DAYS),
            "samples_per_day": N_WINDOWS,
            "num_nodes": N_NODES,
            "n_his": N_HIS,
            "output_steps": 1,
            "target_mode": "last-observed-step",
            "label_definition": "Y is the full state at window_start_slot + 11, not the next slot",
            "channels": list(CHANNELS),
            "state_files": list(STATE_FILES),
            "layout": "{rate}/{split}/{date}/{state}.npz",
            "split_directory_name": "validation",
            "rates": list(RATES),
            "normalization_file": (OUT_ROOT / "state_normalization.json").as_posix(),
            "legacy_flat_npz_preserved": True,
        },
    )
    changed = [
        path
        for path, digest in legacy_hashes.items()
        if not Path(path).is_file() or _sha256(Path(path)) != digest
    ]
    if changed:
        log(f"PROTECTED FILES CHANGED {changed[:8]}")
        return 2
    expected = len(days) * len(RATES) * len(STATE_FILES)
    if written + skipped != expected:
        log(f"file count {written}+{skipped} != {expected}")
        return 2
    log(f"files={expected} written={written} skipped={skipped} protected={len(legacy_hashes)}")
    log("Xuancheng26 five-state window dataset generated successfully.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)


