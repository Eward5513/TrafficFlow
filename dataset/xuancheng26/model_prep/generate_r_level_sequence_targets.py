#!/usr/bin/env python3
"""Per-day full-state tables for Xuancheng26 12->12 sequence reconstruction.

The 12->1 windows define ``X = partial[s:s+12]`` and ``Y = full[s+11]``. The
12->12 task keeps the same ``X`` and reconstructs every observed step,
``Y = full[s:s+12]``. It is not a forecast of the next 12 slots.

``X`` is not rewritten: loaders keep reading the ``x`` member of the existing
``{rate}/{split}/{date}/{state}.npz`` files. The full state does not depend on
the penetration rate, so each date is stored once:

    window_data/sequence_targets_12step/{split}/{date}.npz
        full_raw    float32 [288, 1406, 4]  raw counts, channel order CHANNELS,
                                            node order graph/r_graph.npz edge_ids
    window_data/sequence_targets_12step/sequence_manifest.json

Windows are cut at read time: ``full_raw[s:s+12]`` for ``s = 0..276``. Files are
written uncompressed so ``full_raw`` can be memory-mapped. Existing window NPZ,
normalization, and manifest files are only read; their sizes, mtimes, and a
sample of hashes are compared before and after every run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

import prep_paths
from generate_r_level_windows import (
    CHANNELS,
    N_BINS,
    N_HIS,
    N_NODES,
    N_WINDOWS,
    RATES,
    SPLIT_DAYS,
    STATE_FILES,
    TRAIN_DAYS,
    load_aligned_matrix,
    log,
    split_of,
    weekday_index,
)

SEQUENCE_DIRNAME = "sequence_targets_12step"
MANIFEST_NAME = "sequence_manifest.json"
FORMAT_VERSION = 1
OUTPUT_STEPS = N_HIS
SPLITS = ("train", "validation", "test")
ALL_DAYS = tuple(day for split in SPLITS for day in SPLIT_DAYS[split])
EXPECTED_COUNTS = {split: len(SPLIT_DAYS[split]) * N_WINDOWS for split in SPLITS}
DAY_KEYS = ("full_raw", "edge_ids", "channels", "time_bin", "start_time", "day_index", "date", "split")
NORMALIZED_ATOL = 1e-6
# Old window files whose full SHA-256 is compared before and after a run. Every
# other file under window_data is compared by size and mtime.
HASHED_SAMPLE_NPZ = (
    "p05/train/2023_04_01/inflow.npz",
    "p05/train/2023_04_01/all.npz",
    "p30/validation/2023_04_22/accumulation_end.npz",
    "p70/test/2023_04_30/all.npz",
)
SCALER_NOTE = (
    "Targets reuse the 12->1 full-state scaler from state_normalization.json. It was fit on "
    "full[s+11] for s = 0..276 over the 20 training days, i.e. slots 11..287 weighted once each. "
    "The 12->12 target supervises full[s:s+12], which weights slot t by min(t+1, 12, 288-t), so "
    "early and late slots of each day carry less weight than in the scaler fit. The scaler is "
    "therefore not the exact mean/std of the 12->12 label distribution; it is kept so 12->1 and "
    "12->12 errors stay on the same normalized scale. Validation and test were not used."
)

TRAINING_ROOT = prep_paths.TRAINING_ROOT_3S
WINDOW_ROOT = TRAINING_ROOT / "window_data"
FULL_ROOT = TRAINING_ROOT / "full_flow" / "r-level"
GRAPH_PATH = TRAINING_ROOT / "graph" / "r_graph.npz"
OUT_ROOT = WINDOW_ROOT / SEQUENCE_DIRNAME


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_edge_ids() -> np.ndarray:
    with np.load(GRAPH_PATH, allow_pickle=False) as payload:
        edge_ids = np.array(payload["edge_ids"])
        count = int(payload["num_nodes"])
    if edge_ids.shape != (N_NODES,) or count != N_NODES or len(set(edge_ids.tolist())) != N_NODES:
        raise RuntimeError(f"{GRAPH_PATH} does not hold {N_NODES} unique edge ids")
    return edge_ids


def edge_ids_digest(edge_ids: np.ndarray) -> str:
    return hashlib.sha256("\n".join(str(item) for item in edge_ids.tolist()).encode("utf-8")).hexdigest()


def target_scaler() -> tuple[np.ndarray, np.ndarray]:
    payload = load_json(WINDOW_ROOT / "state_normalization.json")
    target = payload["target"]
    mean = np.asarray([float(target[name]["mean"]) for name in CHANNELS], dtype=np.float32)
    std = np.asarray([float(target[name]["std"]) for name in CHANNELS], dtype=np.float32)
    return mean, std


def normalize(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Same float32 arithmetic as generate_r_level_windows._normalize."""
    return ((raw - mean) / std).astype(np.float32, copy=False)


def check_package() -> dict[str, object]:
    """Fail unless the 12->1 package still has the split, channels, and scaler we rely on."""
    split_manifest = load_json(WINDOW_ROOT / "split_manifest.json")
    state_manifest = load_json(WINDOW_ROOT / "state_window_manifest.json")
    normalization = load_json(WINDOW_ROOT / "state_normalization.json")
    for split in SPLITS:
        for name, payload in (("split_manifest", split_manifest), ("state_window_manifest", state_manifest)):
            if tuple(str(day) for day in payload[split]) != SPLIT_DAYS[split]:
                raise RuntimeError(f"{name}.json {split} dates differ from the frozen split")
    if len(ALL_DAYS) != 29 or len(set(ALL_DAYS)) != 29:
        raise RuntimeError("split does not contain 29 unique dates")
    if tuple(state_manifest["channels"]) != CHANNELS or tuple(normalization["channels"]) != CHANNELS:
        raise RuntimeError("channel order differs from inflow, outflow, accumulation_start, accumulation_end")
    if normalization.get("fit_split") != "train" or tuple(normalization["training_days"]) != TRAIN_DAYS:
        raise RuntimeError("state_normalization.json is not the training-only scaler")
    if int(state_manifest["samples_per_day"]) != N_WINDOWS or int(state_manifest["n_his"]) != N_HIS:
        raise RuntimeError("state_window_manifest window size changed")
    if int(state_manifest["num_nodes"]) != N_NODES or state_manifest["target_mode"] != "last-observed-step":
        raise RuntimeError("state_window_manifest node count or target mode changed")
    counts = split_manifest.get("actual_sample_counts") or {}
    if {split: int(counts.get(split, -1)) for split in SPLITS} != EXPECTED_COUNTS:
        raise RuntimeError(f"split_manifest sample counts {counts} != {EXPECTED_COUNTS}")
    mean, std = target_scaler()
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise RuntimeError("invalid target scaler")
    return {
        "split_manifest_sha256": sha256_file(WINDOW_ROOT / "split_manifest.json"),
        "state_window_manifest_sha256": sha256_file(WINDOW_ROOT / "state_window_manifest.json"),
        "state_normalization_sha256": sha256_file(WINDOW_ROOT / "state_normalization.json"),
    }


def build_day(day_name: str, edge_ids: np.ndarray) -> tuple[np.ndarray, dict[str, str], dict[str, object]]:
    """Read the four full-state CSVs into ``[288, N, 4]`` float32 raw counts."""
    ids = [str(item) for item in edge_ids.tolist()]
    states: dict[str, np.ndarray] = {}
    digests: dict[str, str] = {}
    for name in CHANNELS:
        path = FULL_ROOT / day_name / f"{name}.csv"
        states[name] = load_aligned_matrix(path, ids)
        digests[name] = sha256_file(path)
    within = states["accumulation_start"] + states["inflow"] - states["outflow"]
    if not np.array_equal(within, states["accumulation_end"]):
        raise RuntimeError(f"{day_name} failed N_end = N_start + inflow - outflow")
    continuity = bool(np.array_equal(states["accumulation_start"][1:], states["accumulation_end"][:-1]))
    if not continuity:
        raise RuntimeError(f"{day_name} failed N_start[t+1] = N_end[t]")
    full = np.stack([states[name] for name in CHANNELS], axis=-1)
    full_raw = full.astype(np.float32)
    if not np.array_equal(full_raw.astype(np.int64), full):
        raise RuntimeError(f"{day_name} counts are not exactly representable in float32")
    stats = {
        "conservation_within_slot": True,
        "continuity_across_slots": continuity,
        "channel_totals": {name: int(states[name].sum()) for name in CHANNELS},
        "channel_max": {name: int(states[name].max()) for name in CHANNELS},
    }
    return full_raw, digests, stats


def day_arrays(day_name: str, full_raw: np.ndarray, edge_ids: np.ndarray) -> dict[str, np.ndarray]:
    slots = np.arange(N_BINS, dtype=np.int32)
    return {
        "full_raw": np.ascontiguousarray(full_raw, dtype=np.float32),
        "edge_ids": edge_ids,
        "channels": np.asarray(CHANNELS),
        "time_bin": slots,
        "start_time": slots * 300,
        "day_index": np.asarray(weekday_index(day_name), dtype=np.int32),
        "date": np.asarray(day_name),
        "split": np.asarray(split_of(day_name)),
    }


def check_day_file(path: Path, day_name: str, edge_ids: np.ndarray) -> np.ndarray:
    """Structural check of a written day file; returns ``full_raw``."""
    with np.load(path, allow_pickle=False) as payload:
        missing = [key for key in DAY_KEYS if key not in payload.files]
        if missing:
            raise RuntimeError(f"{path} missing {missing}")
        full_raw = np.array(payload["full_raw"])
        if full_raw.shape != (N_BINS, N_NODES, len(CHANNELS)) or full_raw.dtype != np.float32:
            raise RuntimeError(f"{path} full_raw {full_raw.shape} {full_raw.dtype}")
        if not np.array_equal(payload["edge_ids"], edge_ids):
            raise RuntimeError(f"{path} edge_ids differ from r_graph.npz")
        if tuple(str(item) for item in payload["channels"].tolist()) != CHANNELS:
            raise RuntimeError(f"{path} channel order differs")
        slots = np.arange(N_BINS)
        if not np.array_equal(payload["time_bin"], slots) or not np.array_equal(payload["start_time"], slots * 300):
            raise RuntimeError(f"{path} time axis is not 288 five-minute slots")
        if int(payload["day_index"]) != weekday_index(day_name) or str(payload["date"]) != day_name:
            raise RuntimeError(f"{path} date/day_index metadata differ from {day_name}")
        if str(payload["split"]) != split_of(day_name):
            raise RuntimeError(f"{path} split metadata differs")
    if not np.isfinite(full_raw).all() or np.any(full_raw < 0):
        raise RuntimeError(f"{path} has NaN, inf, or a negative count")
    return full_raw


def sequence_windows(full_raw: np.ndarray) -> np.ndarray:
    """``[277, 12, N, C]`` view with ``out[s] = full_raw[s:s+12]``."""
    view = np.lib.stride_tricks.sliding_window_view(full_raw, N_HIS, axis=0)
    return np.moveaxis(view, -1, 1)


def audit_day(day_name: str, full_raw: np.ndarray) -> dict[str, object]:
    """Compare the 12-step targets with every 12->1 window file of this date."""
    split = split_of(day_name)
    weekday = weekday_index(day_name)
    starts = np.arange(N_WINDOWS)
    indices = starts[:, None] + np.arange(N_HIS)[None, :]
    if not (np.all(np.diff(indices, axis=1) == 1) and indices.max() == N_BINS - 1 and indices.min() == 0):
        raise RuntimeError("window slot indices are not strictly increasing within one day")
    windows = sequence_windows(full_raw)
    if windows.shape != (N_WINDOWS, OUTPUT_STEPS, N_NODES, len(CHANNELS)):
        raise RuntimeError(f"{day_name} window view shape {windows.shape}")
    for probe in (0, 1, 137, N_WINDOWS - 1):
        if not np.array_equal(windows[probe], full_raw[indices[probe]]):
            raise RuntimeError(f"{day_name} window {probe} != full_raw[{probe}:{probe + 12}]")
    if not np.array_equal(windows[:-1, 1:], windows[1:, :-1]):
        raise RuntimeError(f"{day_name} adjacent windows do not overlap by 11 steps")
    last_raw = np.ascontiguousarray(windows[:, -1])
    if not np.array_equal(last_raw, full_raw[N_HIS - 1 :]):
        raise RuntimeError(f"{day_name} last step is not full_raw[s+11]")
    mean, std = target_scaler()
    last_norm = normalize(last_raw, mean, std)
    compared = 0
    exact_y = True
    max_y_diff = 0.0
    for rate in RATES:
        for state in STATE_FILES:
            channel_index = list(range(len(CHANNELS))) if state == "all" else [CHANNELS.index(state)]
            path = WINDOW_ROOT / rate / split / day_name / f"{state}.npz"
            with np.load(path, allow_pickle=False) as payload:
                if not np.array_equal(payload["window_start_slot"], starts):
                    raise RuntimeError(f"{path} window_start_slot is not 0..276")
                if not np.array_equal(payload["target_slot"], starts + N_HIS - 1):
                    raise RuntimeError(f"{path} target_slot is not window_start_slot + 11")
                if not np.array_equal(payload["day_index"], np.full(N_WINDOWS, weekday)):
                    raise RuntimeError(f"{path} day_index is not weekday {weekday}")
                old_raw = payload["y_raw"]
                old_y = payload["y"]
            expected_shape = (N_WINDOWS, 1, N_NODES, len(channel_index))
            if old_raw.shape != expected_shape or old_y.shape != expected_shape:
                raise RuntimeError(f"{path} y shape {old_raw.shape} != {expected_shape}")
            if not np.array_equal(old_raw[:, 0], last_raw[..., channel_index]):
                raise RuntimeError(f"{path} y_raw[:, 0] != new y_raw_seq[:, -1]")
            diff = float(np.max(np.abs(old_y[:, 0] - last_norm[..., channel_index])))
            if diff > NORMALIZED_ATOL:
                raise RuntimeError(f"{path} y[:, 0] differs from normalized new y_seq[:, -1] by {diff}")
            exact_y = exact_y and bool(np.array_equal(old_y[:, 0], last_norm[..., channel_index]))
            max_y_diff = max(max_y_diff, diff)
            compared += 1
    return {
        "window_files_compared": compared,
        "last_step_y_raw_exact": True,
        "last_step_y_bitwise_exact": exact_y,
        "last_step_y_max_abs_diff": max_y_diff,
        "adjacent_windows_overlap": True,
        "window_indices_strictly_increasing": True,
    }


def process_day(task: tuple[str, str, bool, bool]) -> dict[str, object]:
    day_name, mode, skip_complete, overwrite = task
    try:
        edge_ids = load_edge_ids()
        split = split_of(day_name)
        path = OUT_ROOT / split / f"{day_name}.npz"
        full_raw, csv_digests, stats = build_day(day_name, edge_ids)
        status = "written"
        if path.exists():
            try:
                existing = check_day_file(path, day_name, edge_ids)
                problem = None if np.array_equal(existing, full_raw) else "full_raw differs from the CSVs"
            except Exception as error:  # noqa: BLE001 - reported to the user below
                problem = f"unreadable or invalid: {error}"
            if problem is None:
                status = "verified" if mode == "verify" else "unchanged"
                if skip_complete and mode == "generate":
                    log(f"[{day_name}] complete, skipped")
                    return {
                        "date": day_name,
                        "ok": True,
                        "status": "skipped",
                        "sha256": sha256_file(path),
                    }
            elif mode == "verify" or not overwrite:
                raise RuntimeError(f"{path} exists but {problem}; rerun with --dates {day_name} --overwrite")
            else:
                log(f"[{day_name}] replacing existing file: {problem}")
        elif mode == "verify":
            raise RuntimeError(f"{path} is missing")
        if status == "written":
            write_npz(path, day_arrays(day_name, full_raw, edge_ids))
            if not np.array_equal(check_day_file(path, day_name, edge_ids), full_raw):
                raise RuntimeError(f"{path} failed verification after write")
        checks = dict(stats)
        checks["csv_equal"] = True
        checks.update(audit_day(day_name, full_raw))
        record = {
            "date": day_name,
            "ok": True,
            "status": status,
            "split": split,
            "file": f"{split}/{day_name}.npz",
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
            "day_index": weekday_index(day_name),
            "source_csv_sha256": csv_digests,
            "checks": checks,
        }
        log(f"[{day_name}] {status}; 12->1 files compared={checks['window_files_compared']}")
        return record
    except Exception:
        detail = traceback.format_exc(limit=6).strip().replace("\n", " | ")
        log(f"[{day_name}] FAIL {detail}")
        return {"date": day_name, "ok": False, "error": detail}


def run_pool(tasks: list[tuple], workers: int) -> list[dict[str, object]]:
    if workers <= 1:
        return [process_day(task) for task in tasks]
    results: dict[str, dict[str, object]] = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
        futures = {pool.submit(process_day, task): task[0] for task in tasks}
        for future in as_completed(futures):
            day_name = futures[future]
            try:
                results[day_name] = future.result()
            except Exception:
                detail = traceback.format_exc(limit=6).strip().replace("\n", " | ")
                results[day_name] = {"date": day_name, "ok": False, "error": detail}
    return [results[task[0]] for task in tasks]


def protected_snapshot() -> dict[str, object]:
    """Size and mtime of every pre-existing window_data file, plus SHA-256 of key files."""
    stats: dict[str, list[int]] = {}
    for path in sorted(WINDOW_ROOT.rglob("*")):
        if path == OUT_ROOT or OUT_ROOT in path.parents or not path.is_file():
            continue
        status = path.stat()
        stats[path.relative_to(WINDOW_ROOT).as_posix()] = [int(status.st_size), int(status.st_mtime_ns)]
    hashed = [path for path in sorted(WINDOW_ROOT.glob("*")) if path.is_file()]
    hashed += [WINDOW_ROOT / rate / "normalization.json" for rate in RATES]
    hashed += [WINDOW_ROOT / item for item in HASHED_SAMPLE_NPZ]
    digests = {path.relative_to(WINDOW_ROOT).as_posix(): sha256_file(path) for path in hashed if path.is_file()}
    digests["../graph/r_graph.npz"] = sha256_file(GRAPH_PATH)
    return {"stat": stats, "sha256": digests}


def compare_snapshots(before: dict[str, object], after: dict[str, object]) -> list[str]:
    changed = []
    for kind in ("stat", "sha256"):
        left, right = before[kind], after[kind]
        changed += [f"{kind}:{key}" for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)]
    return changed


def build_manifest(
    records: dict[str, dict[str, object]],
    package: dict[str, object],
    edge_ids: np.ndarray,
    protected: dict[str, object],
) -> dict[str, object]:
    previous_path = OUT_ROOT / MANIFEST_NAME
    previous = load_json(previous_path).get("dates", {}) if previous_path.is_file() else {}
    dates: dict[str, object] = {}
    for day_name in ALL_DAYS:
        path = OUT_ROOT / split_of(day_name) / f"{day_name}.npz"
        if not path.is_file():
            continue
        digest = sha256_file(path)
        record = records.get(day_name)
        if record is None or "checks" not in record:
            record = previous.get(day_name)
        if record is None or record.get("sha256") != digest:
            raise RuntimeError(f"{path} has no audit record matching its hash; rerun without --skip-complete")
        dates[day_name] = {key: value for key, value in record.items() if key not in ("ok", "status", "date")}
    complete = len(dates) == len(ALL_DAYS)
    mean, std = target_scaler()
    scaler_table = load_json(WINDOW_ROOT / "state_normalization.json")["target"]
    max_diff = max((float(item["checks"]["last_step_y_max_abs_diff"]) for item in dates.values()), default=None)
    return {
        "format_version": FORMAT_VERSION,
        "complete": complete,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generator": "dataset/xuancheng26/model_prep/generate_r_level_sequence_targets.py",
        "task": "12-step sequence reconstruction (not forecasting)",
        "definition": {
            "x": "partial[s:s+12], read from the existing {rate}/{split}/{date}/{input_type}.npz member x (already z-scored with the rate's input scaler)",
            "y": "full[s:s+12], i.e. full_raw[window_start_slot : window_start_slot + 12] of the same date",
            "output_slots": "window_start_slot + [0..11]; y[:, t] is aligned with x[:, t]",
            "last_step_equals_12to1_target": "y_raw_seq[-1] == old y_raw[0] (full[window_start_slot + 11])",
        },
        "n_his": N_HIS,
        "output_steps": OUTPUT_STEPS,
        "steps_per_day": N_BINS,
        "samples_per_day": N_WINDOWS,
        "window_start_slot_range": [0, N_WINDOWS - 1],
        "cross_day_windows": False,
        "channels": list(CHANNELS),
        "num_nodes": N_NODES,
        "node_order": {"source": "graph/r_graph.npz edge_ids", "edge_ids_sha256": edge_ids_digest(edge_ids)},
        "storage": {
            "layout": "{split}/{date}.npz",
            "array": "full_raw",
            "shape": [N_BINS, N_NODES, len(CHANNELS)],
            "dtype": "float32",
            "values": "raw vehicle counts copied from full_flow/r-level/{date}/{channel}.csv",
            "compression": "none (np.savez) so full_raw can be memory-mapped",
            "shared_across_rates": True,
            "other_keys": [key for key in DAY_KEYS if key != "full_raw"],
        },
        "splits": {split: list(SPLIT_DAYS[split]) for split in SPLITS},
        "sample_counts": dict(EXPECTED_COUNTS),
        "rates": list(RATES),
        "source_package": package,
        "normalization": {
            "file": "../state_normalization.json",
            "sha256": package["state_normalization_sha256"],
            "input": "reused; x is read already z-scored from the 12->1 files",
            "target": {
                name: {"mean": float(scaler_table[name]["mean"]), "std": float(scaler_table[name]["std"])}
                for name in CHANNELS
            },
            "formula": "y = (y_raw - mean[c]) / std[c] in float32, per output channel c",
            "inverse_formula": "y_raw = y * std[c] + mean[c]",
            "float32_mean": [float(value) for value in mean],
            "float32_std": [float(value) for value in std],
            "refit": False,
            "used_validation_to_fit": False,
            "used_test_to_fit": False,
            "note": SCALER_NOTE,
        },
        "dates": dates,
        "verification": {
            "dates_audited": len(dates),
            "window_files_compared": sum(int(item["checks"]["window_files_compared"]) for item in dates.values()),
            "last_step_y_raw_exact": all(item["checks"]["last_step_y_raw_exact"] for item in dates.values()),
            "last_step_y_bitwise_exact": all(item["checks"]["last_step_y_bitwise_exact"] for item in dates.values()),
            "last_step_y_max_abs_diff": max_diff,
            "normalized_atol": NORMALIZED_ATOL,
            "conservation_within_slot": all(item["checks"]["conservation_within_slot"] for item in dates.values()),
            "continuity_across_slots": all(item["checks"]["continuity_across_slots"] for item in dates.values()),
            "adjacent_windows_overlap": all(item["checks"]["adjacent_windows_overlap"] for item in dates.values()),
        },
        "protected_files": protected,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write per-day full-state tables for 12->12 sequence reconstruction")
    prep_paths.add_3s_path_arguments(parser)
    parser.add_argument("--dates", nargs="+", default=None, help="only these YYYY_MM_DD dates")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--skip-complete", action="store_true", help="skip dates whose file already matches the CSVs")
    parser.add_argument("--verify-only", action="store_true", help="check existing files; write nothing")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing file that differs from the CSVs")
    args = parser.parse_args(argv)
    if args.verify_only and (args.skip_complete or args.overwrite):
        parser.error("--verify-only cannot be combined with --skip-complete or --overwrite")
    root = prep_paths.resolved(args.training_root)
    window_root = root / "window_data"
    prep_paths.bind(
        sys.modules[__name__],
        TRAINING_ROOT=root,
        WINDOW_ROOT=window_root,
        FULL_ROOT=root / "full_flow" / "r-level",
        GRAPH_PATH=root / "graph" / "r_graph.npz",
        OUT_ROOT=prep_paths.refuse_code_root(window_root / SEQUENCE_DIRNAME),
    )
    days = tuple(args.dates) if args.dates else ALL_DAYS
    unknown = sorted(set(days) - set(ALL_DAYS))
    if unknown:
        parser.error(f"unknown dates {unknown}")
    mode = "verify" if args.verify_only else "generate"
    log(f"training_root={root} out={OUT_ROOT} mode={mode} dates={len(days)}")
    package = check_package()
    edge_ids = load_edge_ids()
    before = protected_snapshot()
    log(f"protected files: stat={len(before['stat'])} sha256={len(before['sha256'])}")
    if mode == "generate" and OUT_ROOT.is_dir():
        for stale in OUT_ROOT.rglob(".*.tmp*"):
            log(f"removing stale temporary file {stale}")
            stale.unlink()

    workers = max(1, min(int(args.workers), len(days)))
    rows = run_pool([(day, mode, bool(args.skip_complete), bool(args.overwrite)) for day in days], workers)
    failed = [row for row in rows if not row.get("ok")]
    after = protected_snapshot()
    changed = compare_snapshots(before, after)
    if changed:
        log(f"PROTECTED FILES CHANGED {changed[:8]}")
        return 2
    if failed:
        for row in failed:
            log(f"FAILED {row['date']} {row.get('error', '')}")
        return 2
    statuses = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    log(f"days ok={len(rows)} {statuses}; protected files unchanged")

    records = {str(row["date"]): row for row in rows}
    protected = {
        "checked_files": len(after["stat"]),
        "method": "size + mtime_ns of every file under window_data outside sequence_targets_12step; SHA-256 of the files below",
        "unchanged": True,
        "sha256": after["sha256"],
    }
    if mode == "verify":
        manifest_path = OUT_ROOT / MANIFEST_NAME
        if not manifest_path.is_file():
            log(f"FAILED {manifest_path} is missing")
            return 2
        manifest = load_json(manifest_path)
        for day_name, row in records.items():
            entry = manifest.get("dates", {}).get(day_name)
            if entry is None or entry.get("sha256") != row["sha256"]:
                log(f"FAILED manifest entry for {day_name} does not match the file hash")
                return 2
        if manifest.get("normalization", {}).get("sha256") != package["state_normalization_sha256"]:
            log("FAILED state_normalization.json changed since the targets were generated")
            return 2
        log(f"verify-only passed for {len(records)} dates; manifest complete={manifest.get('complete')}")
        return 0
    manifest = build_manifest(records, package, edge_ids, protected)
    write_json(OUT_ROOT / MANIFEST_NAME, manifest)
    log(f"manifest complete={manifest['complete']} dates={len(manifest['dates'])}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
