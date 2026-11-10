#!/usr/bin/env python3
"""Build the two Xuancheng26 variable-penetration datasets for one probability range.

One FCD scan per date updates every requested dataset and sampling seed. Fixed
seven-rate nesting is not reused. Normalization is fit only after every
sampling-seed-42 training day has a verified state CSV.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

CODE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_r_level_full_flow as full  # noqa: E402
import generate_r_level_windows as windows  # noqa: E402
import prep_paths  # noqa: E402
from data_root import data_root  # noqa: E402
from independent_states import PHYSICAL_VERSION, definition_sha256  # noqa: E402
from variable_penetration import (  # noqa: E402
    ALGORITHM_VERSION,
    SCHEMA_VERSION,
    PopulationConflict,
    build_schedule,
    population_from_tripinfo,
    sample_day,
    schedule_sha256,
    segments_for_date,
    validate_generator_config,
    verify_depart_prefix,
)
from vehicle_sampling import read_ids, write_ids  # noqa: E402

CHANNELS = ("inflow", "outflow", "accumulation_start", "accumulation_end")
STATE_FILES = tuple(f"{name}.csv" for name in CHANNELS)
RUNTIME: dict = {}


def log(message: str) -> None:
    from datetime import datetime

    print(f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        return validate_generator_config(payload)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def split_dates(training_root: Path) -> tuple[dict[str, list[str]], list[str]]:
    payload = json.loads((training_root / "shared" / "split_manifest.json").read_text(encoding="utf-8"))
    splits = {name: [str(day) for day in payload[name]] for name in ("train", "validation", "test")}
    dates = [day for name in ("train", "validation", "test") for day in splits[name]]
    if [len(splits[name]) for name in ("train", "validation", "test")] != [20, 4, 5] or len(set(dates)) != 29:
        raise RuntimeError("authoritative split is not 20/4/5 unique days")
    if payload.get("n_his") != 12 or payload.get("target_mode") != "last-observed-step":
        raise RuntimeError("reconstruction contract changed")
    from variable_penetration import validate_dates

    try:
        validate_dates(dates)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    return splits, dates


def plan_workers(root: Path, requested: int, tasks: int, estimate: int) -> dict:
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        fields = line.split()
        mem[fields[0].rstrip(":")] = int(fields[1]) * 1024
    cpus = os.cpu_count() or 1
    memory_workers = max(1, int((mem["MemAvailable"] - (32 << 30)) // (3 << 30)))
    workers = max(1, min(int(requested), tasks, max(1, cpus - 4), memory_workers))
    free = shutil.disk_usage(root).free
    if free < estimate + (20 << 30):
        raise RuntimeError(f"insufficient disk: free={free} estimate={estimate}")
    return {
        "workers": workers,
        "requested_workers": int(requested),
        "cpu_count": cpus,
        "reserved_cpus": 4,
        "available_memory_bytes": mem["MemAvailable"],
        "reserved_memory_bytes": 32 << 30,
        "disk_free_bytes": free,
        "estimated_bytes": estimate,
        "gpu_used": False,
        "fcd_read_buffer_bytes": 64 << 20,
    }


def estimate_bytes(datasets: int, days: int, seeds: int) -> int:
    one = 4 * (277 * 12 * 1406 * 4 + 277 * 1406 * 4) * 2
    all_states = one * 4
    per_day_seed = one * 4 + all_states
    return int(datasets * days * seeds * per_day_seed * 1.4) + (8 << 30)


def dataset_root(training_root: Path, dataset_id: str) -> Path:
    return training_root / "datasets" / dataset_id


def link_relative(path: Path, target: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = os.path.relpath(target, path.parent)
    if path.is_symlink() and os.readlink(path) == text and path.resolve() == target.resolve():
        return
    if path.is_symlink() or path.exists():
        if path.resolve() == target.resolve():
            return
        raise RuntimeError(f"refusing to retarget {path}")
    path.symlink_to(text)


def prepare_dataset(training_root: Path, spec: dict, schedule: dict, splits: dict[str, list[str]], seeds: list[int]) -> None:
    root = prep_paths.refuse_code_root(dataset_root(training_root, spec["dataset_id"]))
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "dataset_manifest.json"
    manifest = {
        "dataset_id": spec["dataset_id"],
        "kind": f"variable_{spec['mode']}",
        "mode": spec["mode"],
        "observation_keys": [spec["observation_key"]],
        "p_min": schedule["p_min"],
        "p_max": schedule["p_max"],
        "segment_seconds": schedule["segment_seconds"],
        "schedule_seed": schedule["schedule_seed"],
        "sampling_seeds": [int(seed) for seed in seeds],
        "fit_sampling_seed": 42,
        "algorithm_version": ALGORITHM_VERSION,
        "schema_version": SCHEMA_VERSION,
        "schedule_sha256": schedule["schedule_sha256"],
        "distribution": "uniform_continuous",
        "physical_definition_version": PHYSICAL_VERSION,
        "definition_sha256": definition_sha256(),
        "sampling_algorithm_version": ALGORITHM_VERSION,
    }
    schedule_path = root / "penetration_schedule.json"
    if schedule_path.is_file():
        existing = json.loads(schedule_path.read_text(encoding="utf-8"))
        if existing.get("schedule_sha256") != schedule["schedule_sha256"] or schedule_sha256(existing) != schedule["schedule_sha256"]:
            raise RuntimeError(f"schedule conflict for {spec['dataset_id']}")
    else:
        write_json(schedule_path, schedule)
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = {key: existing.get(key) for key in manifest}
        if comparable != manifest:
            raise RuntimeError(f"dataset manifest conflict for {spec['dataset_id']}")
    else:
        write_json(manifest_path, manifest)
    shared = root / "shared"
    shared.mkdir(exist_ok=True)
    for name in (
        "split_manifest.json",
        "node_mapping.csv",
        "physical_definition.json",
        "target_normalization.json",
        "sequence_targets_12step",
    ):
        target = training_root / "shared" / name
        if not target.exists():
            raise RuntimeError(f"missing shared input {target}")
        before = target.stat().st_ino
        link_relative(shared / name, target)
        if (shared / name).resolve() != target.resolve():
            raise RuntimeError(f"{name} link does not reach the shared file")
        if target.stat().st_ino != before:
            raise RuntimeError(f"linking {name} replaced the shared file")
    for seed in seeds:
        (root / "sampling_seeds" / f"seed_{int(seed)}" / "flow" / "window_data").mkdir(parents=True, exist_ok=True)
        (root / "sampling_seeds" / f"seed_{int(seed)}" / "sampled_vehicle" / spec["observation_key"]).mkdir(parents=True, exist_ok=True)
    (root / "audit" / "days").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    del splits


def publish_ids(path: Path, vehicle_ids: list[str]) -> str:
    expected = sorted(vehicle_ids)
    if path.is_file():
        if read_ids(path) == expected:
            return "skipped"
        raise RuntimeError(f"vehicle list conflict: {path}")
    write_ids(path, expected)
    if read_ids(path) != expected:
        raise RuntimeError(f"vehicle list reread failed: {path}")
    return "written"


def process_vehicle_day(task: dict) -> dict:
    try:
        population = population_from_tripinfo(Path(task["tripinfo"]))
        if population["conflicts"]:
            raise PopulationConflict(f"{task['date']} tripinfo口径冲突 {population['conflicts']} {population['counts']}")
        results = []
        for spec in task["datasets"]:
            segments = segments_for_date(spec["schedule"], task["date"])
            for seed in task["sampling_seeds"]:
                chosen, stats, derived = sample_day(
                    population["ids"], population["depart"], date=task["date"], sampling_seed=seed,
                    segments=segments, segment_seconds=int(spec["schedule"]["segment_seconds"]),
                )
                prefix = verify_depart_prefix(
                    population["ids"], population["depart"], date=task["date"], sampling_seed=seed,
                    segments=segments, segment_seconds=int(spec["schedule"]["segment_seconds"]), chosen=chosen,
                    p_min=float(spec["schedule"]["p_min"]) if spec["mode"] == "day" else None,
                    p_max=float(spec["schedule"]["p_max"]) if spec["mode"] == "day" else None,
                )
                if prefix["derived_seed"] != derived:
                    raise RuntimeError(f"{task['date']} seed {seed} permutation seed drifted")
                path = Path(spec["vehicle_paths"][str(seed)])
                action = publish_ids(path, chosen)
                record = {
                    "dataset_id": spec["dataset_id"],
                    "sampling_seed": seed,
                    "derived_seed": derived,
                    "algorithm_version": ALGORITHM_VERSION,
                    "selected_count": len(chosen),
                    "sha256": sha256_file(path),
                    "action": action,
                    "segments": stats,
                    "prefix_check": prefix["check"],
                }
                results.append(record)
        sets = {}
        for record in results:
            sets.setdefault(record["dataset_id"], []).append(record["sha256"])
        return {
            "date": task["date"],
            "ok": True,
            "eligible_count": len(population["ids"]),
            "trip_counts": population["counts"],
            "cross_boundary_examples": population["cross_boundary_examples"],
            "all_ids_count": len(population["all_ids"]),
            "groups": results,
            "distinct_seed_hashes": {key: len(set(values)) for key, values in sets.items()},
        }
    except Exception:
        return {"date": task["date"], "ok": False, "error": traceback.format_exc(limit=8)}


def _group_dir(dataset: Path, seed: int, key: str, date: str) -> Path:
    return dataset / "sampling_seeds" / f"seed_{seed}" / "flow" / "partial_flow" / key / date


def _flow_digest(directory: Path) -> dict[str, str]:
    return {name: sha256_file(directory / f"{name}.csv") for name in CHANNELS}


def _flow_matches(directory: Path, edges: list[str], digest: dict[str, str] | None) -> bool:
    if not directory.is_dir() or not all((directory / name).is_file() for name in STATE_FILES):
        return False
    if digest and _flow_digest(directory) != digest:
        return False
    inflow = full.read_flow_csv(directory / "inflow.csv", edges)
    outflow = full.read_flow_csv(directory / "outflow.csv", edges)
    return full.states_match(directory, inflow, outflow, edges, directory.name)


def publish_flow(directory: Path, inflow, outflow, edges: list[str], label: str) -> str:
    if directory.is_dir() and full.states_match(directory, inflow, outflow, edges, label):
        return "skipped"
    if directory.exists():
        raise RuntimeError(f"existing flow does not match the new count: {directory}")
    staging = directory.with_name(f".{directory.name}.{os.getpid()}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    try:
        full.write_state_matrices(staging, inflow, outflow, edges, label)
        directory.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging, directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    if not full.states_match(directory, inflow, outflow, edges, label):
        raise RuntimeError(f"published flow failed reread: {directory}")
    return "written"


def _cell_summary(partial, full_matrix) -> dict:
    import numpy as np

    zero = full_matrix == 0
    valid = ~zero
    ratios = (partial[valid] / full_matrix[valid]) if int(valid.sum()) else np.array([])
    bins = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    histogram = [int(((ratios >= low) & (ratios < high)).sum()) for low, high in zip(bins, bins[1:])] if ratios.size else [0] * (len(bins) - 1)
    if ratios.size:
        histogram.append(int((ratios >= bins[-1]).sum()))
    else:
        histogram.append(0)
    percentiles = None if not ratios.size else [float(np.percentile(ratios, q)) for q in (0, 5, 25, 50, 75, 95, 100)]
    return {
        "cells": int(partial.size),
        "zero_denominator": int(zero.sum()),
        "positive_numerator_when_denominator_zero": int(((partial > 0) & zero).sum()),
        "histogram_edges": bins + ["inf"],
        "histogram": histogram,
        "percentiles": percentiles,
    }


def _network_ratio(partial, full_matrix):
    import numpy as np

    partial_sum = partial.sum(axis=1).astype(np.int64)
    full_sum = full_matrix.sum(axis=1).astype(np.int64)
    ratio = np.divide(partial_sum, full_sum, out=np.full(full_sum.shape, np.nan, dtype=np.float64), where=full_sum > 0)
    return partial_sum, full_sum, ratio


def process_flow_day(task: dict) -> dict:
    raise RuntimeError("variable flow is written by independent_states.py scan; this cumsum path is retired")



def _run(function, tasks: list[dict], workers: int) -> list[dict]:
    if workers <= 1:
        return [function(task) for task in tasks]
    results = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
        futures = {pool.submit(function, task): task["date"] for task in tasks}
        for future in as_completed(futures):
            date = futures[future]
            try:
                results[date] = future.result()
            except Exception:
                results[date] = {"date": date, "ok": False, "error": traceback.format_exc(limit=6)}
    return [results[task["date"]] for task in tasks]


def fit_scaler(training_root: Path, spec: dict, schedule: dict, train_days: list[str], edges: list[str]) -> dict:
    import numpy as np

    root = dataset_root(training_root, spec["dataset_id"])
    key = spec["observation_key"]
    totals = {name: [0.0, 0.0, 0] for name in CHANNELS}
    full_root = training_root / "full_flow" / "r-level"
    partial_root = root / "sampling_seeds" / "seed_42" / "flow" / "partial_flow" / key
    for day in train_days:
        full_states = {name: windows.load_aligned_matrix(full_root / day / f"{name}.csv", edges) for name in CHANNELS}
        partial = {name: windows.load_aligned_matrix(partial_root / day / f"{name}.csv", edges) for name in CHANNELS}
        windows._check_conservation(full_states, f"full {day}")
        windows._check_conservation(partial, f"{spec['dataset_id']} {day}")
        for name in CHANNELS:
            if np.any(partial[name] > full_states[name]):
                raise RuntimeError(f"{day} {name} partial exceeds full")
            raw, _y, _starts, _targets = windows.make_windows(partial[name], full_states[name])
            total, square, count = windows._moment(raw)
            bucket = totals[name]
            bucket[0] += total
            bucket[1] += square
            bucket[2] += count
            del raw
    fitted = {name: windows._mean_std(*totals[name]) for name in CHANNELS}
    source = training_root / "shared" / "target_normalization.json"
    before = sha256_file(source)
    authoritative = json.loads(source.read_text(encoding="utf-8"))
    if authoritative.get("physical_definition_version") != PHYSICAL_VERSION or authoritative.get("definition_sha256") != definition_sha256():
        raise RuntimeError(f"shared target scaler is not {PHYSICAL_VERSION}")
    if list(authoritative.get("training_days") or []) != list(train_days):
        raise RuntimeError("shared target scaler was not fit on the authoritative training days")
    import subprocess

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, text=True).strip()
    payload = {
        "physical_definition_version": PHYSICAL_VERSION,
        "definition_sha256": definition_sha256(),
        "dataset_status": "building",
        "dataset_id": spec["dataset_id"],
        "fit_split": "train",
        "ddof": 0,
        "formula": "(value - mean) / std",
        "scope": "global_scalar_per_variable",
        "channels": list(CHANNELS),
        "training_days": list(train_days),
        "fit_sampling_seed": 42,
        "observation_key": key,
        "input_source": "sampling seed 42 training-window x_raw, overlap retained, ddof 0",
        "target_source": "shared/target_normalization.json",
        "target_source_sha256": before,
        "schedule_sha256": schedule["schedule_sha256"],
        "target": authoritative["target"],
        "input": {key: {name: {"mean": fitted[name][0], "std": fitted[name][1]} for name in CHANNELS}},
        "git_commit": commit,
    }
    if authoritative["target"] != payload["target"]:
        raise RuntimeError("target scaler copy diverged")
    if sha256_file(source) != before:
        raise RuntimeError("shared target scaler changed while it was being copied")
    destination = root / "shared" / "state_normalization.json"
    if destination.is_symlink():
        raise RuntimeError(f"refusing scaler symlink: {destination}")
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("input") != payload["input"] or existing.get("target") != payload["target"] or existing.get("fit_sampling_seed") != 42:
            raise RuntimeError(f"scaler conflict: {destination}")
        if existing.get("physical_definition_version") not in (None, PHYSICAL_VERSION):
            raise RuntimeError(f"scaler physical version conflict: {destination}")
        if existing.get("definition_sha256") not in (None, definition_sha256()):
            raise RuntimeError(f"scaler definition hash conflict: {destination}")
        if existing.get("schedule_sha256") not in (None, schedule["schedule_sha256"]):
            raise RuntimeError(f"scaler was fit for a different schedule: {destination}")
        payload = existing
    else:
        write_json(destination, payload)
        if sha256_file(source) != before:
            raise RuntimeError("authoritative target scaler changed while fitting")
    manifest = {
        "train": train_days,
        "validation": json.loads((training_root / "shared" / "split_manifest.json").read_text())["validation"],
        "test": json.loads((training_root / "shared" / "split_manifest.json").read_text())["test"],
        "samples_per_day": windows.N_WINDOWS,
        "num_nodes": windows.N_NODES,
        "n_his": windows.N_HIS,
        "output_steps": 1,
        "target_mode": "last-observed-step",
        "label_definition": "Y is the full state at window_start_slot + 11, not the next slot",
        "channels": list(CHANNELS),
        "state_files": list(windows.STATE_FILES),
        "layout": "{rate}/{split}/{date}/{state}.npz",
        "split_directory_name": "validation",
        "rates": [key],
        "dataset_id": spec["dataset_id"],
        "sampling_seeds": list(json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))["sampling_seeds"]),
        "schedule_sha256": schedule["schedule_sha256"],
        "normalization_file": str(destination),
        "fit_sampling_seed": 42,
        "physical_definition_version": PHYSICAL_VERSION,
        "definition_sha256": definition_sha256(),
    }
    manifest_path = root / "shared" / "state_window_manifest.json"
    if manifest_path.is_symlink():
        raise RuntimeError(f"refusing manifest symlink: {manifest_path}")
    write_json(manifest_path, manifest)
    summary = root / "shared" / "validation_summary.json"
    if not summary.exists():
        write_json(summary, {
            "overall_validation_passed": False,
            "dataset_status": "building",
            "failures": ["windows and release checks are not finished"],
            "dataset_id": spec["dataset_id"],
            "physical_definition_version": PHYSICAL_VERSION,
            "definition_sha256": definition_sha256(),
        })
    for seed in json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))["sampling_seeds"]:
        window = root / "sampling_seeds" / f"seed_{seed}" / "flow" / "window_data"
        for name in ("split_manifest.json", "node_mapping.csv", "state_normalization.json", "state_window_manifest.json", "validation_summary.json"):
            link_relative(window / name, root / "shared" / name)
        normalization = (window / "state_normalization.json").resolve()
        if normalization == (training_root / "shared" / "state_normalization.json").resolve():
            raise RuntimeError("variable normalization points at the fixed scaler")
        if (window / "split_manifest.json").resolve() != (training_root / "shared" / "split_manifest.json").resolve():
            raise RuntimeError("split link does not reach the authoritative manifest")
    return payload


def _window_identity(day_dir: Path) -> dict | None:
    path = day_dir / "day_identity.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _source_stat(path: Path) -> dict:
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def process_window_day(task: dict) -> dict:
    import numpy as np

    try:
        date = task["date"]
        edges = task.get("edges") or RUNTIME["edges"]
        full_root = Path(task.get("full_root") or RUNTIME["full_root"])
        weekday = windows.weekday_index(date)
        full_states = {name: windows.load_aligned_matrix(full_root / date / f"{name}.csv", edges) for name in CHANNELS}
        written = skipped = 0
        for spec in task["datasets"]:
            stats = spec["scaler"]
            key = spec["observation_key"]
            for seed in task["sampling_seeds"]:
                partial_root = Path(spec["dataset_root"]) / "sampling_seeds" / f"seed_{seed}" / "flow" / "partial_flow" / key / date
                members = Path(spec["dataset_root"]) / "sampling_seeds" / f"seed_{seed}" / "sampled_vehicle" / key / f"{date}.csv"
                partial = {name: windows.load_aligned_matrix(partial_root / f"{name}.csv", edges) for name in CHANNELS}
                identity = {
                    "dataset_id": spec["dataset_id"],
                    "sampling_seed": int(seed),
                    "observation_key": key,
                    "date": date,
                    "split": task["split"],
                    "schedule_sha256": spec["schedule_sha256"],
                    "vehicle_list_sha256": sha256_file(members),
                    "physical_definition_version": PHYSICAL_VERSION,
                    "definition_sha256": definition_sha256(),
                    "node_order_sha256": task["node_order_sha256"],
                    "scaler_sha256": spec["scaler_sha256"],
                    "full_source": {
                        name: _source_stat(full_root / date / f"{name}.csv") for name in CHANNELS
                    },
                }
                single = {}
                starts = targets = None
                for name in CHANNELS:
                    x_raw, y_raw, starts, targets = windows.make_windows(partial[name], full_states[name])
                    single[name] = (x_raw, y_raw)
                bundles = dict(single)
                bundles["all"] = (
                    np.concatenate([single[name][0] for name in CHANNELS], axis=-1),
                    np.concatenate([single[name][1] for name in CHANNELS], axis=-1),
                )
                day_dir = Path(spec["dataset_root"]) / "sampling_seeds" / f"seed_{seed}" / "flow" / "window_data" / key / task["split"] / date
                stored_identity = _window_identity(day_dir)
                identity_ok = stored_identity == identity
                day_index = np.full(windows.N_WINDOWS, weekday, dtype=np.int32)
                for state_name, (x_raw, y_raw) in bundles.items():
                    names = CHANNELS if state_name == "all" else (state_name,)
                    mean_x = np.asarray([stats["input"][key][name]["mean"] for name in names], dtype=np.float32)
                    std_x = np.asarray([stats["input"][key][name]["std"] for name in names], dtype=np.float32)
                    mean_y = np.asarray([stats["target"][name]["mean"] for name in names], dtype=np.float32)
                    std_y = np.asarray([stats["target"][name]["std"] for name in names], dtype=np.float32)
                    path = day_dir / f"{state_name}.npz"
                    channels = 4 if state_name == "all" else 1
                    if identity_ok and windows._file_matches(path, channels=channels, weekday=weekday, mean_x=mean_x, std_x=std_x, mean_y=mean_y, std_y=std_y):
                        skipped += 1
                        continue
                    windows.write_npz(path, {
                        "x": windows._normalize(x_raw, mean_x.reshape(1, 1, 1, -1), std_x.reshape(1, 1, 1, -1)),
                        "y": windows._normalize(y_raw, mean_y.reshape(1, 1, 1, -1), std_y.reshape(1, 1, 1, -1)),
                        "x_raw": np.ascontiguousarray(x_raw),
                        "y_raw": np.ascontiguousarray(y_raw),
                        "day_index": day_index,
                        "window_start_slot": starts,
                        "target_slot": targets,
                    })
                    if not windows._file_matches(path, channels=channels, weekday=weekday, mean_x=mean_x, std_x=std_x, mean_y=mean_y, std_y=std_y):
                        raise RuntimeError(f"window verification failed: {path}")
                    written += 1
                write_json(day_dir / "day_identity.json", identity)
        log(f"[{date}] windows written={written} skipped={skipped}")
        return {"date": date, "ok": True, "written": written, "skipped": skipped}
    except Exception:
        return {"date": task["date"], "ok": False, "error": traceback.format_exc(limit=8)}


def _save_stage(training_root: Path, datasets: list[dict], name: str, payload: dict, *, complete: bool) -> None:
    for spec in datasets:
        root = dataset_root(training_root, spec["dataset_id"]) / "audit"
        write_json(root / f"stage_{name}.json" if complete else root / f"stage_{name}_partial.json", payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate variable-penetration Xuancheng26 datasets")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, default=data_root() / "xuancheng26/training_data_3s")
    parser.add_argument("--fcd-root", type=Path, default=data_root() / "xuancheng26/simulation_data/fcd_data_3s")
    parser.add_argument("--net-file", type=Path, default=CODE_ROOT / "dataset/xuancheng26/xuancheng_tls_3s.net.xml")
    parser.add_argument("--dates", nargs="+")
    parser.add_argument("--workers", type=int, default=29)
    parser.add_argument("--stage", choices=("all", "prepare", "vehicles", "flow", "scaler", "windows", "ready", "models"), default="all")
    parser.add_argument("--prepare-models", choices=("none", "all-baselines"), default="none")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--cross-check-date")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    training_root = prep_paths.refuse_code_root(args.training_root)
    fcd_root = args.fcd_root.expanduser().resolve()
    splits, all_dates = split_dates(training_root)
    dates = list(args.dates) if args.dates else list(all_dates)
    unknown = sorted(set(dates) - set(all_dates))
    if unknown:
        raise RuntimeError(f"dates outside the authoritative manifest: {unknown}")
    complete = dates == all_dates
    resources = plan_workers(training_root, args.workers, max(1, len(dates)), estimate_bytes(len(config["datasets"]), len(all_dates), 5))
    log(f"resources {resources}")
    if not args.execute:
        print(json.dumps({"dry_run": True, "dates": dates, "datasets": [item["dataset_id"] for item in config["datasets"]], "resources": resources}, indent=2))
        return 0
    lock_path = training_root / "datasets" / ".variable_penetration.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        schedules = []
        for spec in config["datasets"]:
            schedule = build_schedule(
                dataset_id=spec["dataset_id"], mode=spec["mode"], p_min=float(config["p_min"]), p_max=float(config["p_max"]),
                segment_seconds=int(spec["segment_seconds"]), schedule_seed=int(config["schedule_seed"]),
                dates=all_dates, observation_key=spec["observation_key"],
            )
            spec["schedule"] = schedule
            schedules.append(schedule)
            prepare_dataset(training_root, spec, schedule, splits, list(config["sampling_seeds"]))
        if args.stage == "prepare":
            log("prepare complete")
            return 0
        vehicle_tasks = []
        for date in dates:
            trip = full.prefer_xml(fcd_root / date, f"tripinfo_{date}")
            vehicle_tasks.append({
                "date": date,
                "tripinfo": str(trip),
                "training_root": str(training_root),
                "sampling_seeds": list(config["sampling_seeds"]),
                "datasets": [{
                    "dataset_id": spec["dataset_id"],
                    "mode": spec["mode"],
                    "observation_key": spec["observation_key"],
                    "schedule": spec["schedule"],
                    "vehicle_paths": {
                        str(seed): str(dataset_root(training_root, spec["dataset_id"]) / "sampling_seeds" / f"seed_{seed}" / "sampled_vehicle" / spec["observation_key"] / f"{date}.csv")
                        for seed in config["sampling_seeds"]
                    },
                } for spec in config["datasets"]],
            })
        if args.stage in {"all", "vehicles", "flow", "windows"}:
            vehicle_rows = _run(process_vehicle_day, vehicle_tasks, resources["workers"])
            failed = [row for row in vehicle_rows if not row.get("ok")]
            _save_stage(training_root, config["datasets"], "vehicles", {"complete_dataset": complete and not failed, "dates": dates, "rows": vehicle_rows}, complete=complete and not failed)
            if failed:
                for row in failed:
                    log(f"VEHICLE FAIL {row['date']} {row.get('error','')[:500]}")
                return 1
            log(f"vehicles ok dates={len(vehicle_rows)}")
        if args.stage == "vehicles":
            return 0
        edges, lane_to_edge, internal = full.load_network(args.net_file)
        sample_header = full.read_flow_csv(training_root / "full_flow" / "r-level" / dates[0] / "inflow.csv", edges)
        del sample_header
        RUNTIME.update(edges=edges, lane_to_edge=lane_to_edge, internal=internal, edge_index={edge: index for index, edge in enumerate(edges)},
                       full_root=str(training_root / "full_flow" / "r-level"), fcd_root=str(fcd_root))
        if args.stage in {"all", "flow"}:
            from independent_states import scan_selected

            scan_specs = [{
                "dataset_id": spec["dataset_id"],
                "observation_key": spec["observation_key"],
                "sampling_seeds": list(config["sampling_seeds"]),
                "schedule_sha256": spec["schedule"]["schedule_sha256"],
            } for spec in config["datasets"]]
            if scan_selected(
                training_root=training_root,
                fcd_root=fcd_root,
                net_file=args.net_file,
                specs=scan_specs,
                dates=dates,
                workers=resources["workers"],
            ):
                return 1
            _save_stage(training_root, config["datasets"], "flow", {"complete_dataset": complete, "writer": "independent_states.scan_selected", "resources": resources}, complete=complete)
            log("flow published for the selected datasets")
        if args.stage == "flow":
            return 0
        if args.stage in {"all", "scaler", "windows"}:
            if not set(splits["train"]).issubset(set(dates) if args.stage != "windows" else set(all_dates)):
                if args.stage == "scaler" or not set(splits["train"]).issubset(set(all_dates)):
                    raise RuntimeError("refusing to fit a scaler before every training day exists")
            missing_train = []
            for spec in config["datasets"]:
                for day in splits["train"]:
                    path = _group_dir(dataset_root(training_root, spec["dataset_id"]), 42, spec["observation_key"], day) / "inflow.csv"
                    if not path.is_file():
                        missing_train.append(str(path))
            if missing_train:
                raise RuntimeError(f"training CSVs missing, scaler not fit: {missing_train[:5]}")
            scalers = []
            for spec in config["datasets"]:
                scalers.append(fit_scaler(training_root, spec, spec["schedule"], splits["train"], RUNTIME["edges"]))
            log("scalers frozen")
            if args.stage == "scaler":
                return 0
            split_of = {day: name for name, days in splits.items() for day in days}
            if "edges" not in RUNTIME:
                edges, lane_to_edge, internal = full.load_network(args.net_file)
                RUNTIME.update(edges=edges, lane_to_edge=lane_to_edge, internal=internal, full_root=str(training_root / "full_flow" / "r-level"))
            node_order_sha256 = sha256_file(training_root / "shared" / "node_mapping.csv")
            window_tasks = []
            for date in (dates if args.stage == "all" else all_dates):
                window_tasks.append({
                    "date": date,
                    "split": split_of[date],
                    "sampling_seeds": list(config["sampling_seeds"]),
                    "node_order_sha256": node_order_sha256,
                    "edges": list(RUNTIME["edges"]),
                    "full_root": RUNTIME["full_root"],
                    "datasets": [{
                        "dataset_id": spec["dataset_id"],
                        "dataset_root": str(dataset_root(training_root, spec["dataset_id"])),
                        "observation_key": spec["observation_key"],
                        "schedule_sha256": spec["schedule"]["schedule_sha256"],
                        "scaler": scalers[index],
                        "scaler_sha256": hashlib.sha256(json.dumps({
                            "input": scalers[index]["input"],
                            "target": scalers[index]["target"],
                            "definition_sha256": scalers[index].get("definition_sha256"),
                        }, sort_keys=True).encode("utf-8")).hexdigest(),
                    } for index, spec in enumerate(config["datasets"])],
                })
            window_rows = _run(process_window_day, window_tasks, resources["workers"])
            failed = [row for row in window_rows if not row.get("ok")]
            _save_stage(training_root, config["datasets"], "windows", {"complete_dataset": complete and not failed, "rows": window_rows}, complete=complete and not failed)
            if failed:
                for row in failed:
                    log(f"WINDOW FAIL {row['date']} {row.get('error','')[:800]}")
                return 1
            log("windows written")
        if args.stage in {"all", "windows", "ready", "models"} and complete:
            from variable_release import audit_and_release

            report = audit_and_release(
                training_root, config["datasets"], splits, RUNTIME["edges"], resources["workers"],
            )
            if report["failures"]:
                for item in report["failures"][:20]:
                    log(f"RELEASE FAIL {item}")
                return 1
            log(f"datasets ready counts={report['counts']}")
        elif args.stage in {"all", "windows", "ready", "models"}:
            log("partial date list; dataset status stays unfinished")
        prepare_models = args.prepare_models == "all-baselines" or args.stage == "models"
        if prepare_models and not complete:
            log("skipping model preparation until every date is present")
            return 0
        if prepare_models:
            from prepare_baseline_resources import prepare_models as prepare_baseline_models

            model_report = prepare_baseline_models(
                training_root=training_root,
                fcd_root=fcd_root,
                net_file=args.net_file,
                dataset_ids=[spec["dataset_id"] for spec in config["datasets"]],
                workers=resources["workers"],
            )
            if model_report["failures"]:
                for item in model_report["failures"][:20]:
                    log(f"MODEL FAIL {item}")
                return 1
            log("baseline resources ready")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BlockingIOError:
        print("variable penetration lock is held", file=sys.stderr)
        sys.exit(2)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
