#!/usr/bin/env python3
"""Independent checks for variable-penetration datasets.

This does not trust the generator's success flag. A passed
``validation_summary.json`` is written only when every manifest date, seed,
state file, and window passes and ``--publish-ready`` is set.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_r_level_full_flow as full  # noqa: E402
import generate_r_level_windows as windows  # noqa: E402
import numpy as np  # noqa: E402
from data_root import data_root  # noqa: E402
from variable_penetration import (  # noqa: E402
    population_from_tripinfo,
    sample_day,
    schedule_sha256,
    segments_for_date,
)
from vehicle_sampling import read_ids  # noqa: E402

CHANNELS = ("inflow", "outflow", "accumulation_start", "accumulation_end")
SEEDS = [42, 43, 44, 45, 46]
RUNTIME: dict = {}


def log(message: str) -> None:
    print(message, flush=True)


def check_schedule(dataset: Path) -> tuple[dict, list[str]]:
    schedule = json.loads((dataset / "penetration_schedule.json").read_text(encoding="utf-8"))
    failures = []
    if schedule_sha256(schedule) != schedule.get("schedule_sha256"):
        failures.append("schedule hash does not match its contents")
    if not schedule["p_min"] <= min(item["p"] for item in schedule["segments"]) <= max(item["p"] for item in schedule["segments"]) <= schedule["p_max"]:
        failures.append("a target probability is outside the declared range")
    manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schedule_sha256") != schedule["schedule_sha256"]:
        failures.append("dataset manifest schedule hash differs")
    if manifest.get("observation_keys") != [schedule["observation_key"]]:
        failures.append("observation key is not the manifest key")
    normalization = dataset / "shared" / "state_normalization.json"
    if normalization.is_symlink():
        failures.append("state_normalization.json must be a real file for this dataset")
    return schedule, failures


def check_vehicles_date(dataset: Path, schedule: dict, fcd_root: Path, date: str) -> list[str]:
    failures = []
    key = schedule["observation_key"]
    trip = full.prefer_xml(fcd_root / date, f"tripinfo_{date}")
    population = population_from_tripinfo(trip)
    if population["conflicts"]:
        return [f"{dataset.name} {date} tripinfo conflict {population['conflicts']}"]
    segments = segments_for_date(schedule, date)
    chosen_sets = []
    for seed in SEEDS:
        chosen, stats, _derived = sample_day(
            population["ids"], population["depart"], date=date, sampling_seed=seed,
            segments=segments, segment_seconds=int(schedule["segment_seconds"]),
        )
        path = dataset / "sampling_seeds" / f"seed_{seed}" / "sampled_vehicle" / key / f"{date}.csv"
        if not path.is_file():
            failures.append(f"missing vehicle list {path}")
            continue
        stored = read_ids(path)
        if stored != sorted(chosen):
            failures.append(f"{dataset.name} {date} seed {seed} vehicle list is not the schedule prefix")
        if len(stored) != len(set(stored)):
            failures.append(f"{dataset.name} {date} seed {seed} duplicate vehicle ids")
        if sum(item["selected_count"] for item in stats) != len(chosen):
            failures.append(f"{dataset.name} {date} seed {seed} segment counts do not add up")
        for item in stats:
            if item["eligible_count"] == 0 and item["realized_ratio"] is not None:
                failures.append(f"{dataset.name} {date} seed {seed} empty segment has a ratio")
            if item["eligible_count"] and item["selected_count"] != math.floor(item["eligible_count"] * item["target_p"]):
                failures.append(f"{dataset.name} {date} seed {seed} segment {item['segment_index']} is not floor(N*p)")
        chosen_sets.append(frozenset(stored))
    if len(chosen_sets) == len(SEEDS) and len(set(chosen_sets)) == 1 and len(population["ids"]) > 1000:
        failures.append(f"{dataset.name} {date} all sampling seeds produced the same vehicle set")
    return failures


def check_states_date(dataset: Path, schedule: dict, training_root: Path, date: str, edges: list[str]) -> list[str]:
    failures = []
    key = schedule["observation_key"]
    full_root = training_root / "full_flow" / "r-level"
    full_states = {name: full.read_flow_csv(full_root / date / f"{name}.csv", edges) for name in CHANNELS}
    for seed in SEEDS:
        directory = dataset / "sampling_seeds" / f"seed_{seed}" / "flow" / "partial_flow" / key / date
        if not all((directory / f"{name}.csv").is_file() for name in CHANNELS):
            failures.append(f"missing states {directory}")
            continue
        stored = {name: full.read_flow_csv(directory / f"{name}.csv", edges) for name in CHANNELS}
        try:
            full.accumulate_states(stored["inflow"], stored["outflow"], edges, f"{dataset.name}:{seed}:{date}")
        except RuntimeError as exc:
            failures.append(str(exc))
        for name in CHANNELS:
            if np.any(stored[name] > full_states[name]):
                failures.append(f"{dataset.name} {date} seed {seed} {name} exceeds full")
            if stored[name].shape != (288, 1406) or int(stored[name].min()) < 0:
                failures.append(f"{dataset.name} {date} seed {seed} {name} shape or sign")
    return failures


def recount_one(dataset: Path, schedule: dict, training_root: Path, fcd_root: Path, date: str, edges, lane_to_edge, internal) -> list[str]:
    key = schedule["observation_key"]
    seed = 42
    ids = set(read_ids(dataset / "sampling_seeds" / f"seed_{seed}" / "sampled_vehicle" / key / f"{date}.csv"))
    fcd_path = full.prefer_xml(fcd_root / date, f"fcd_{date}")
    arrivals = full.load_arrivals(full.prefer_xml(fcd_root / date, f"tripinfo_{date}"))
    full.READ_BYTES = 64 << 20
    entry, exit_flow, _counts, _seen = full.stream_fcd(
        fcd_path, edges, lane_to_edge, internal, arrivals, date, include=ids,
    )
    directory = dataset / "sampling_seeds" / f"seed_{seed}" / "flow" / "partial_flow" / key / date
    stored_in = full.read_flow_csv(directory / "inflow.csv", edges)
    stored_out = full.read_flow_csv(directory / "outflow.csv", edges)
    if not np.array_equal(entry, stored_in) or not np.array_equal(exit_flow, stored_out):
        return [f"{dataset.name} {date} seed 42 recount disagrees with the stored CSV"]
    return []


def check_windows_metadata(dataset: Path, training_root: Path, schedule: dict) -> list[str]:
    failures = []
    manifest = json.loads((training_root / "shared" / "split_manifest.json").read_text(encoding="utf-8"))
    normalization_path = dataset / "shared" / "state_normalization.json"
    if not normalization_path.is_file() or normalization_path.is_symlink():
        return ["normalization file is missing or is a symlink"]
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    fixed = json.loads((training_root / "shared" / "state_normalization.json").read_text(encoding="utf-8"))
    if normalization.get("target") != fixed.get("target"):
        failures.append("target scaler differs from the authoritative fixed scaler")
    if normalization.get("fit_sampling_seed") != 42 or normalization.get("training_days") != manifest["train"]:
        failures.append("input scaler was not fit on sampling seed 42 training days")
    if normalization_path.resolve() == (training_root / "shared" / "state_normalization.json").resolve():
        failures.append("variable normalization inode is the fixed file")
    if schedule["observation_key"] not in normalization.get("input", {}):
        failures.append("input scaler is not indexed by this dataset's observation key")
    for seed in SEEDS:
        root = dataset / "sampling_seeds" / f"seed_{seed}" / "flow" / "window_data"
        for name in ("split_manifest.json", "node_mapping.csv", "state_normalization.json", "state_window_manifest.json", "validation_summary.json"):
            path = root / name
            if not path.is_symlink() or not path.exists():
                failures.append(f"seed {seed} metadata link broken: {name}")
        if (root / "state_normalization.json").resolve() != normalization_path.resolve():
            failures.append(f"seed {seed} normalization does not resolve to this dataset")
    return failures


def check_windows_date(dataset: Path, schedule: dict, training_root: Path, date: str, split: str) -> tuple[list[str], bool]:
    failures = []
    key = schedule["observation_key"]
    label = None
    observed = []
    for seed in SEEDS:
        directory = dataset / "sampling_seeds" / f"seed_{seed}" / "flow" / "window_data" / key / split / date
        for state in (*CHANNELS, "all"):
            path = directory / f"{state}.npz"
            if not path.is_file():
                failures.append(f"missing window {path}")
                continue
            with np.load(path) as payload:
                if list(payload.files) != list(windows.NPZ_KEYS):
                    failures.append(f"{path} keys {payload.files}")
                    continue
                channels = 4 if state == "all" else 1
                if payload["x"].shape != (277, 12, 1406, channels) or payload["y"].shape != (277, 1, 1406, channels):
                    failures.append(f"{path} shape {payload['x'].shape} {payload['y'].shape}")
                if not np.array_equal(payload["target_slot"], payload["window_start_slot"] + 11):
                    failures.append(f"{path} target is not the last observed step")
                if state == "inflow":
                    y_raw = np.array(payload["y_raw"])
                    if label is None:
                        label = y_raw
                    elif not np.array_equal(label, y_raw):
                        failures.append(f"{dataset.name} {date} seed {seed} label differs")
                    observed.append(np.array(payload["x_raw"][:1]).tobytes())
                if state == "all":
                    inflow = np.load(directory / "inflow.npz")
                    try:
                        if not np.array_equal(payload["x"][..., 0], inflow["x"][..., 0]):
                            failures.append(f"{directory} all channel 0 differs from inflow")
                    finally:
                        inflow.close()
    return failures, len(set(observed)) > 1


def publish(dataset: Path, failures: list[str], counts: dict) -> None:
    payload = {
        "overall_validation_passed": not failures,
        "failures": failures,
        "dataset_id": dataset.name,
        "verified_vehicle_lists": counts.get("vehicles"),
        "verified_state_csv": counts.get("states"),
        "verified_windows": counts.get("windows"),
        "sampling_seeds": [42, 43, 44, 45, 46],
    }
    path = dataset / "shared" / "validation_summary.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def recount_date(dataset_specs: list[dict], date: str, fcd_root: Path) -> list[str]:
    """One FCD scan checks seed 42 of every dataset. Twenty-nine dates run together."""
    from xml.parsers import expat

    edges = RUNTIME["edges"]
    failures = []
    groups = []
    for spec in dataset_specs:
        ids = set(read_ids(Path(spec["dataset"]) / "sampling_seeds" / "seed_42" / "sampled_vehicle" / spec["key"] / f"{date}.csv"))
        groups.append({
            "name": spec["dataset_id"],
            "include": ids,
            "entry": np.zeros((full.N_BINS, len(edges)), dtype=np.int64),
            "exit": np.zeros((full.N_BINS, len(edges)), dtype=np.int64),
            "states": {},
            "seen": set(),
            "counts": {"entry": 0, "exit": 0, "unknown": 0, "unresolved": 0, "final_exit": 0},
            "directory": Path(spec["dataset"]) / "sampling_seeds" / "seed_42" / "flow" / "partial_flow" / spec["key"] / date,
        })
    members: dict[str, list[dict]] = {}
    for group in groups:
        for vehicle_id in group["include"]:
            members.setdefault(vehicle_id, []).append(group)
    arrivals = full.load_arrivals(full.prefer_xml(fcd_root / date, f"tripinfo_{date}"))
    clock = {"time": 0.0}

    def start(name: str, attrs) -> None:
        if name == "timestep":
            clock["time"] = float(attrs.get("time", "0"))
            return
        if name != "vehicle":
            return
        vehicle_id = attrs.get("id", "")
        targets = members.get(vehicle_id)
        if not vehicle_id or not targets:
            return
        lane = attrs.get("lane", "")
        if lane in RUNTIME["lane_to_edge"]:
            kind, column = "real", RUNTIME["edge_index"][RUNTIME["lane_to_edge"][lane]]
        elif lane in RUNTIME["internal"]:
            kind, column = "internal", None
        else:
            kind, column = "unknown", None
        for group in targets:
            full.observe(group["states"], group["seen"], vehicle_id, clock["time"], kind, column, group["entry"], group["exit"], group["counts"])

    full.READ_BYTES = 64 << 20
    parser = expat.ParserCreate()
    parser.StartElementHandler = start
    for block in full.read_chunks(full.prefer_xml(fcd_root / date, f"fcd_{date}")):
        parser.Parse(block, False)
    parser.Parse(b"", True)
    for group in groups:
        full.impute_final_exits(group["states"], arrivals, group["exit"], group["counts"])
        stored_in = full.read_flow_csv(group["directory"] / "inflow.csv", edges)
        stored_out = full.read_flow_csv(group["directory"] / "outflow.csv", edges)
        if not np.array_equal(group["entry"], stored_in) or not np.array_equal(group["exit"], stored_out):
            failures.append(f"{group['name']} {date} seed 42 recount disagrees with the stored CSV")
    log(f"[{date}] recount datasets={len(groups)}")
    return failures


def audit_date(task: dict) -> dict:
    try:
        failures = []
        changed = {spec["dataset_id"]: False for spec in task["datasets"]}
        fcd_root = Path(task["fcd_root"])
        training_root = Path(task["training_root"])
        for spec in task["datasets"]:
            dataset = Path(spec["dataset"])
            schedule = spec["schedule"]
            if task["vehicles"]:
                failures.extend(check_vehicles_date(dataset, schedule, fcd_root, task["date"]))
            if task["states"]:
                failures.extend(check_states_date(dataset, schedule, training_root, task["date"], RUNTIME["edges"]))
            if task["windows"]:
                day_failures, day_changed = check_windows_date(dataset, schedule, training_root, task["date"], task["split"])
                failures.extend(day_failures)
                changed[spec["dataset_id"]] = changed[spec["dataset_id"]] or day_changed
        if task["recount"]:
            failures.extend(recount_date(task["datasets"], task["date"], fcd_root))
        log(f"[{task['date']}] audit failures={len(failures)}")
        return {"date": task["date"], "failures": failures, "input_changed": changed}
    except Exception:
        return {"date": task["date"], "failures": [traceback.format_exc(limit=6)], "input_changed": False}


def run_dates(tasks: list[dict], workers: int) -> list[dict]:
    if workers <= 1:
        return [audit_date(task) for task in tasks]
    results = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
        futures = {pool.submit(audit_date, task): task["date"] for task in tasks}
        for future in as_completed(futures):
            date = futures[future]
            try:
                results[date] = future.result()
            except Exception:
                results[date] = {"date": date, "failures": [traceback.format_exc(limit=6)], "input_changed": False}
    return [results[task["date"]] for task in tasks]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independently audit variable-penetration datasets")
    parser.add_argument("--training-root", type=Path, default=data_root() / "xuancheng26/training_data_3s")
    parser.add_argument("--fcd-root", type=Path, default=data_root() / "xuancheng26/simulation_data/fcd_data_3s")
    parser.add_argument("--net-file", type=Path, default=CODE_ROOT / "dataset/xuancheng26/xuancheng_tls_3s.net.xml")
    parser.add_argument("--dataset-id", action="append", required=True)
    parser.add_argument("--workers", type=int, default=29)
    parser.add_argument("--skip-vehicles", action="store_true")
    parser.add_argument("--skip-states", action="store_true")
    parser.add_argument("--skip-windows", action="store_true")
    parser.add_argument("--recount", action="store_true")
    parser.add_argument("--publish-ready", action="store_true")
    args = parser.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    training_root = args.training_root.expanduser().resolve()
    fcd_root = args.fcd_root.expanduser().resolve()
    manifest = json.loads((training_root / "shared" / "split_manifest.json").read_text(encoding="utf-8"))
    split_of = {day: name for name, days in (("train", manifest["train"]), ("validation", manifest["validation"]), ("test", manifest["test"])) for day in days}
    dates = [day for name in ("train", "validation", "test") for day in manifest[name]]
    workers = max(29, int(args.workers))
    workers = min(workers, len(dates), max(29, (os.cpu_count() or 29) - 4))
    log(f"audit workers={workers} dates={len(dates)} recount={args.recount}")
    specs = []
    header_failures: dict[str, list[str]] = {}
    for dataset_id in args.dataset_id:
        dataset = training_root / "datasets" / dataset_id
        schedule, schedule_failures = check_schedule(dataset)
        header_failures[dataset_id] = list(schedule_failures)
        if not args.skip_windows:
            header_failures[dataset_id].extend(check_windows_metadata(dataset, training_root, schedule))
        specs.append({
            "dataset_id": dataset_id,
            "dataset": str(dataset),
            "key": schedule["observation_key"],
            "schedule": schedule,
        })
    if not args.skip_states or args.recount:
        edges, lane_to_edge, internal = full.load_network(args.net_file)
        RUNTIME.update(
            edges=edges,
            lane_to_edge=lane_to_edge,
            internal=internal,
            edge_index={edge: index for index, edge in enumerate(edges)},
        )
    tasks = [{
        "date": date,
        "split": split_of[date],
        "datasets": specs,
        "training_root": str(training_root),
        "fcd_root": str(fcd_root),
        "vehicles": not args.skip_vehicles,
        "states": not args.skip_states,
        "windows": not args.skip_windows,
        "recount": bool(args.recount),
    } for date in dates]
    rows = run_dates(tasks, workers)
    by_dataset = {item: list(header_failures[item]) for item in header_failures}
    changed = {dataset_id: False for dataset_id in by_dataset}
    for row in rows:
        for dataset_id, flag in (row.get("input_changed") or {}).items():
            changed[dataset_id] = changed[dataset_id] or bool(flag)
        for failure in row["failures"]:
            placed = False
            for dataset_id in by_dataset:
                if failure.startswith(dataset_id):
                    by_dataset[dataset_id].append(failure)
                    placed = True
            if not placed:
                for dataset_id in by_dataset:
                    by_dataset[dataset_id].append(failure)
    if not args.skip_windows:
        for dataset_id, flag in changed.items():
            if not flag:
                by_dataset[dataset_id].append("input windows did not change across sampling seeds")
    counts = {"vehicles": 29 * 5, "states": 29 * 5 * 4, "windows": 29 * 5 * 5}
    for dataset_id, failures in by_dataset.items():
        if args.publish_ready:
            publish(training_root / "datasets" / dataset_id, failures, counts)
        log(f"{dataset_id} failures={len(failures)}")
        for item in failures[:12]:
            log(f"  {item}")
    return 1 if any(by_dataset.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
