#!/usr/bin/env python3
"""Build extra p05 test observations for the Xuancheng26 3-second batch.

The five p05 test samplings live under ``sampling_seeds/seed_<id>/``.
Sampling seed 42 is the original published sample, moved there with the same
layout as seeds 43–46. Each tree reuses the same FCD, full labels, road order,
and training normalization. This does not train a model or score one.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

import generate_r_level_full_flow as full
import generate_r_level_windows as windows
import generate_sampled_vehicle_ids as sampling
import prep_paths

RATE_NAME = "p05"
RATE = 0.05
ORIGINAL_SAMPLING_SEED = 42
DEFAULT_NEW_SEEDS = (43, 44, 45, 46)
ALGORITHM_ID = "sha256_sampling_seed_date_v1"
CONTEXT: dict = {}


def log(message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"{stamp} {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != ["vehicle_id"]:
            raise RuntimeError(f"unexpected vehicle header in {path}: {header}")
        values = [row[0] for row in reader if row]
    if any(not isinstance(item, str) or item == "" for item in values):
        raise RuntimeError(f"blank vehicle id in {path}")
    return values


def seed_root(training_root: Path, sampling_seed: int) -> Path:
    return training_root / "sampling_seeds" / f"seed_{int(sampling_seed)}"


def vehicle_path(training_root: Path, sampling_seed: int, date_label: str) -> Path:
    return seed_root(training_root, sampling_seed) / "sampled_vehicle" / RATE_NAME / f"{date_label}.csv"


def flow_dir(training_root: Path, sampling_seed: int, date_label: str) -> Path:
    return seed_root(training_root, sampling_seed) / "partial_flow" / RATE_NAME / date_label


def window_dir(training_root: Path, sampling_seed: int, date_label: str) -> Path:
    return seed_root(training_root, sampling_seed) / "window_data" / RATE_NAME / "test" / date_label


def original_vehicle_path(training_root: Path, date_label: str) -> Path:
    return vehicle_path(training_root, ORIGINAL_SAMPLING_SEED, date_label)


def original_window_dir(training_root: Path, date_label: str) -> Path:
    return window_dir(training_root, ORIGINAL_SAMPLING_SEED, date_label)


def assert_under_sampling_seeds(path: Path, training_root: Path) -> None:
    root = (training_root / "sampling_seeds").resolve()
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside sampling_seeds: {resolved}")


def ensure_symlink(link: Path, target: Path) -> None:
    if not target.exists():
        raise RuntimeError(f"missing symlink target {target}")
    link.parent.mkdir(parents=True, exist_ok=True)
    relative = Path(os.path.relpath(target, link.parent))
    if link.is_symlink():
        current = Path(os.readlink(link))
        if current == relative:
            return
        raise RuntimeError(f"{link} points to {current}, expected {relative}")
    if link.exists():
        raise RuntimeError(f"refusing to replace existing path with a symlink: {link}")
    link.symlink_to(relative)


def choose_ids(eligible: list[str], sampling_seed: int, date_label: str) -> tuple[list[str], int, int]:
    """Return sorted ids, the derived daily seed, and the official prefix length."""
    derived = sampling.daily_seed(date_label, sampling_seed)
    permutation = sampling.permute(eligible, derived)
    if sampling.permute(eligible, derived) != permutation:
        raise RuntimeError(f"sampling seed {sampling_seed} {date_label} is not reproducible")
    count = int(len(eligible) * RATE)
    chosen = permutation[:count]
    if len(chosen) != len(set(chosen)):
        raise RuntimeError(f"sampling seed {sampling_seed} {date_label} produced duplicate ids")
    if not set(chosen).issubset(eligible):
        raise RuntimeError(f"sampling seed {sampling_seed} {date_label} left the eligible set")
    return sorted(chosen), derived, count


def load_test_dates(training_root: Path) -> list[str]:
    path = training_root / "window_data" / "split_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    dates = [str(item) for item in manifest["test"]]
    if dates != list(windows.TEST_DAYS):
        raise RuntimeError(f"split_manifest test dates {dates} differ from the frozen test split")
    if len(dates) != len(set(dates)):
        raise RuntimeError("split_manifest test dates contain a duplicate")
    return dates


def load_scales(normalization_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    payload = json.loads(normalization_path.read_text(encoding="utf-8"))
    if payload.get("fit_split") != "train":
        raise RuntimeError("state_normalization.json was not fit on training")
    if list(payload.get("channels") or []) != list(windows.CHANNELS):
        raise RuntimeError("state_normalization.json channel order changed")

    def pack(names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mean_x = np.asarray([payload["input"][RATE_NAME][name]["mean"] for name in names], dtype=np.float32)
        std_x = np.asarray([payload["input"][RATE_NAME][name]["std"] for name in names], dtype=np.float32)
        mean_y = np.asarray([payload["target"][name]["mean"] for name in names], dtype=np.float32)
        std_y = np.asarray([payload["target"][name]["std"] for name in names], dtype=np.float32)
        if np.any(std_x <= 0) or np.any(std_y <= 0):
            raise RuntimeError("state_normalization.json has a non-positive std")
        return mean_x, std_x, mean_y, std_y

    scales = {name: pack((name,)) for name in windows.CHANNELS}
    scales["all"] = pack(windows.CHANNELS)
    return scales


def fingerprint(paths: list[Path]) -> dict[str, dict[str, object]]:
    records = {}
    for path in paths:
        stat = path.stat()
        records[path.as_posix()] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path),
        }
    return records


def protected_paths(training_root: Path, dates: list[str]) -> list[Path]:
    paths = [
        training_root / "window_data" / "split_manifest.json",
        training_root / "window_data" / "state_normalization.json",
        training_root / "sampled_vehicle" / "sampling_summary.csv",
        training_root / "window_data" / RATE_NAME / "train" / "2023_04_01" / "inflow.npz",
        training_root / "window_data" / RATE_NAME / "validation" / "2023_04_22" / "inflow.npz",
    ]
    for date_label in dates:
        paths.append(original_vehicle_path(training_root, date_label))
        day_dir = original_window_dir(training_root, date_label)
        for name in windows.STATE_FILES:
            paths.append(day_dir / f"{name}.npz")
        full_dir = training_root / "full_flow" / "r-level" / date_label
        for name in windows.CHANNELS:
            paths.append(full_dir / f"{name}.csv")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"protected inputs missing: {missing[:8]}")
    return paths


def link_shared_inputs(training_root: Path, sampling_seed: int) -> None:
    """Point a seed root at the original train, validation, graph, and scalers."""
    root = seed_root(training_root, sampling_seed)
    window = root / "window_data"
    original_window = training_root / "window_data"
    for name in (
        "split_manifest.json",
        "state_normalization.json",
        "state_window_manifest.json",
        "normalization.json",
    ):
        ensure_symlink(window / name, original_window / name)
    ensure_symlink(root / "graph", training_root / "graph")
    for rate in windows.RATES:
        if rate == RATE_NAME:
            continue
        ensure_symlink(window / rate, original_window / rate)
    p05 = window / RATE_NAME
    p05.mkdir(parents=True, exist_ok=True)
    if p05.is_symlink():
        raise RuntimeError(f"{p05} must be a real directory so the new test split is not the original")
    (p05 / "test").mkdir(parents=True, exist_ok=True)
    original_p05 = original_window / RATE_NAME
    for name in ("train", "validation", "train.npz", "validation.npz", "normalization.json"):
        ensure_symlink(p05 / name, original_p05 / name)


def prepare_samples(
    training_root: Path,
    fcd_root: Path,
    dates: list[str],
    new_seeds: tuple[int, ...],
) -> dict[str, dict]:
    """Verify seed 42 and write the new id lists. No FCD scan and no original writes."""
    days: dict[str, dict] = {}
    for date_label in dates:
        trip = full.prefer_xml(fcd_root / date_label, f"tripinfo_{date_label}")
        eligible = sampling.load_eligible_ids(trip)
        if eligible != sorted(eligible):
            raise RuntimeError(f"{date_label} eligible ids are not in the original sorted order")
        original_ids, original_derived, original_count = choose_ids(eligible, ORIGINAL_SAMPLING_SEED, date_label)
        stored = read_ids(original_vehicle_path(training_root, date_label))
        if original_ids != stored:
            raise RuntimeError(
                f"{date_label} sampling seed {ORIGINAL_SAMPLING_SEED} does not reproduce the stored p05 ids "
                f"(recomputed {len(original_ids)}, stored {len(stored)}, "
                f"only_new {len(set(original_ids) - set(stored))}, only_stored {len(set(stored) - set(original_ids))})"
            )
        samples = {
            ORIGINAL_SAMPLING_SEED: {
                "ids": original_ids,
                "daily_seed": str(original_derived),
                "sample_count": original_count,
                "vehicle_file": original_vehicle_path(training_root, date_label).as_posix(),
                "stored_location": "sampling_seeds",
            }
        }
        for sampling_seed in new_seeds:
            chosen, derived, count = choose_ids(eligible, sampling_seed, date_label)
            path = vehicle_path(training_root, sampling_seed, date_label)
            assert_under_sampling_seeds(path, training_root)
            if path.is_file() and read_ids(path) == chosen:
                log(f"[{date_label}] seed {sampling_seed} vehicle ids already match")
            else:
                sampling.write_ids(path, chosen)
                if read_ids(path) != chosen:
                    raise RuntimeError(f"{path} does not match the sampled ids")
            samples[sampling_seed] = {
                "ids": chosen,
                "daily_seed": str(derived),
                "sample_count": count,
                "vehicle_file": path.as_posix(),
                "stored_location": "sampling_seeds",
            }
        sets = {seed: set(item["ids"]) for seed, item in samples.items()}
        if len({frozenset(values) for values in sets.values()}) != len(sets):
            raise RuntimeError(f"{date_label} two sampling seeds produced the same vehicle set")
        days[date_label] = {
            "eligible_vehicle_count": len(eligible),
            "eligible_ids": eligible,
            "eligible_sha256": sha256_text("\n".join(eligible)),
            "samples": samples,
            "sets": sets,
        }
        log(
            f"[{date_label}] eligible={len(eligible)} p05={original_count} "
            f"original_seed={ORIGINAL_SAMPLING_SEED} reproduced"
        )
    return days


def pairwise(days: dict[str, dict], seeds: list[int]) -> list[dict[str, object]]:
    rows = []
    for date_label, day in days.items():
        for index, left in enumerate(seeds):
            for right in seeds[index + 1 :]:
                a = day["sets"][left]
                b = day["sets"][right]
                intersection = len(a & b)
                union = len(a | b)
                rows.append(
                    {
                        "date": date_label,
                        "sampling_seed_a": left,
                        "sampling_seed_b": right,
                        "intersection": intersection,
                        "union": union,
                        "jaccard": (intersection / union) if union else 0.0,
                        "identical": a == b,
                    }
                )
    return rows


def _window_problems(
    path: Path,
    original: Path,
    scales: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    channels: int,
) -> list[str]:
    problems = []
    mean_x, std_x, mean_y, std_y = scales
    with np.load(path, allow_pickle=False) as got, np.load(original, allow_pickle=False) as old:
        if list(got.files) != list(windows.NPZ_KEYS) or list(old.files) != list(windows.NPZ_KEYS):
            return [f"{path.name} keys differ from the original schema"]
        for key in windows.NPZ_KEYS:
            if got[key].dtype != old[key].dtype:
                problems.append(f"{path.name} {key} dtype {got[key].dtype} != {old[key].dtype}")
            if got[key].shape != old[key].shape:
                problems.append(f"{path.name} {key} shape {got[key].shape} != {old[key].shape}")
        if problems:
            return problems
        for key in ("y", "y_raw", "day_index", "window_start_slot", "target_slot"):
            if not np.array_equal(got[key], old[key]):
                problems.append(f"{path.name} {key} differs from the original test window")
        x_raw = got["x_raw"]
        y_raw = got["y_raw"]
        if int(x_raw.shape[-1]) != channels or int(y_raw.shape[-1]) != channels:
            problems.append(f"{path.name} channel count is not {channels}")
        if np.any(x_raw < 0) or np.any(y_raw < 0):
            problems.append(f"{path.name} has a negative raw value")
        expected_x = windows._normalize(x_raw, mean_x.reshape(1, 1, 1, -1), std_x.reshape(1, 1, 1, -1))
        expected_y = windows._normalize(y_raw, mean_y.reshape(1, 1, 1, -1), std_y.reshape(1, 1, 1, -1))
        if not np.allclose(got["x"], expected_x, rtol=0.0, atol=1e-5):
            problems.append(f"{path.name} x is not the original p05 training scaler")
        if not np.allclose(got["y"], expected_y, rtol=0.0, atol=1e-5):
            problems.append(f"{path.name} y is not the original full-label scaler")
    return problems


def day_problems(training_root: Path, date_label: str, sampling_seed: int, chosen: list[str], edge_ids: list[str], scales) -> list[str]:
    problems = []
    ids_path = vehicle_path(training_root, sampling_seed, date_label)
    if not ids_path.is_file() or read_ids(ids_path) != chosen:
        return [f"{date_label} seed {sampling_seed} vehicle file is missing or stale"]
    flow = flow_dir(training_root, sampling_seed, date_label)
    full_dir = training_root / "full_flow" / "r-level" / date_label
    if not all((flow / f"{name}.csv").is_file() for name in windows.CHANNELS):
        return [f"{date_label} seed {sampling_seed} is missing a state csv"]
    try:
        partial = windows._load_states(flow, edge_ids)
        complete = windows._load_states(full_dir, edge_ids)
        windows._check_conservation(partial, f"seed {sampling_seed} {date_label}")
    except Exception as exc:
        return [f"{date_label} seed {sampling_seed} state csv: {exc}"]
    for name in windows.CHANNELS:
        if np.any(partial[name] < 0):
            problems.append(f"{date_label} seed {sampling_seed} {name} is negative")
        if np.any(partial[name] > complete[name]):
            problems.append(f"{date_label} seed {sampling_seed} {name} exceeds the full count")
    produced = window_dir(training_root, sampling_seed, date_label)
    original = original_window_dir(training_root, date_label)
    for name in windows.STATE_FILES:
        path = produced / f"{name}.npz"
        if not path.is_file():
            problems.append(f"missing {path}")
            continue
        channels = 4 if name == "all" else 1
        problems.extend(_window_problems(path, original / f"{name}.npz", scales[name], channels))
    return problems


def process_day(date_label: str) -> dict[str, object]:
    try:
        training_root = CONTEXT["training_root"]
        fcd_root = CONTEXT["fcd_root"]
        new_seeds = CONTEXT["new_seeds"]
        edge_ids = CONTEXT["edge_ids"]
        lane_to_edge = CONTEXT["lane_to_edge"]
        internal_lanes = CONTEXT["internal_lanes"]
        scales = CONTEXT["scales"]
        chosen = {seed: CONTEXT["days"][date_label]["samples"][seed]["ids"] for seed in new_seeds}
        if CONTEXT["skip_complete"]:
            pending = [
                seed for seed in new_seeds
                if day_problems(training_root, date_label, seed, chosen[seed], edge_ids, scales)
            ]
        else:
            pending = list(new_seeds)
        if not pending:
            log(f"[{date_label}] SKIP verified sampling seeds {list(new_seeds)}")
            return {"date": date_label, "ok": True, "skipped": True}
        owners: dict[str, list[int]] = {}
        for index, seed in enumerate(new_seeds):
            for vehicle_id in chosen[seed]:
                owners.setdefault(vehicle_id, []).append(index)
        inflows = [np.zeros((full.N_BINS, len(edge_ids)), dtype=np.int64) for _ in new_seeds]
        outflows = [np.zeros((full.N_BINS, len(edge_ids)), dtype=np.int64) for _ in new_seeds]

        def on_event(kind: str, slot: int, column: int, vehicle_id: str) -> None:
            target = inflows if kind == "entry" else outflows
            for index in owners[vehicle_id]:
                target[index][slot, column] += 1

        fcd_path = full.prefer_xml(fcd_root / date_label, f"fcd_{date_label}")
        trip_path = full.prefer_xml(fcd_root / date_label, f"tripinfo_{date_label}")
        arrivals = full.load_arrivals(trip_path)
        log(f"[{date_label}] scanning {fcd_path.name} once for seeds {list(new_seeds)}")
        full.stream_fcd(
            fcd_path,
            edge_ids,
            lane_to_edge,
            internal_lanes,
            arrivals,
            date_label,
            include=set(owners),
            on_event=on_event,
        )
        for index, seed in enumerate(new_seeds):
            out_dir = flow_dir(training_root, seed, date_label)
            assert_under_sampling_seeds(out_dir, training_root)
            full.write_state_matrices(out_dir, inflows[index], outflows[index], edge_ids, f"{date_label} seed {seed}")
            _write_windows(training_root, date_label, seed, edge_ids, scales)
            problems = day_problems(training_root, date_label, seed, chosen[seed], edge_ids, scales)
            if problems:
                raise RuntimeError("; ".join(problems[:8]))
            log(f"[{date_label}] seed {seed} states and windows verified")
        return {"date": date_label, "ok": True, "skipped": False}
    except Exception:
        detail = traceback.format_exc(limit=8).strip().replace("\n", " | ")
        log(f"[{date_label}] FAIL {detail}")
        return {"date": date_label, "ok": False, "error": detail}


def _write_windows(training_root: Path, date_label: str, sampling_seed: int, edge_ids: list[str], scales) -> None:
    partial = windows._load_states(flow_dir(training_root, sampling_seed, date_label), edge_ids)
    complete = windows._load_states(training_root / "full_flow" / "r-level" / date_label, edge_ids)
    single = {}
    starts = None
    targets = None
    for name in windows.CHANNELS:
        x_raw, y_raw, starts, targets = windows.make_windows(partial[name], complete[name])
        single[name] = (x_raw, y_raw)
    bundles = dict(single)
    bundles["all"] = (
        np.concatenate([single[name][0] for name in windows.CHANNELS], axis=-1),
        np.concatenate([single[name][1] for name in windows.CHANNELS], axis=-1),
    )
    day_dir = window_dir(training_root, sampling_seed, date_label)
    assert_under_sampling_seeds(day_dir, training_root)
    day_index = np.full(windows.N_WINDOWS, windows.weekday_index(date_label), dtype=np.int32)
    for name, (x_raw, y_raw) in bundles.items():
        mean_x, std_x, mean_y, std_y = scales[name]
        arrays = {
            "x": windows._normalize(x_raw, mean_x.reshape(1, 1, 1, -1), std_x.reshape(1, 1, 1, -1)),
            "y": windows._normalize(y_raw, mean_y.reshape(1, 1, 1, -1), std_y.reshape(1, 1, 1, -1)),
            "x_raw": np.ascontiguousarray(x_raw),
            "y_raw": np.ascontiguousarray(y_raw),
            "day_index": day_index,
            "window_start_slot": starts,
            "target_slot": targets,
        }
        windows.write_npz(day_dir / f"{name}.npz", arrays)


def build_manifest(
    training_root: Path,
    fcd_root: Path,
    net_file: Path,
    dates: list[str],
    new_seeds: tuple[int, ...],
    days: dict[str, dict],
    pairs: list[dict[str, object]],
    edge_hash: str,
    normalization_path: Path,
) -> dict:
    all_seeds = [ORIGINAL_SAMPLING_SEED, *new_seeds]
    day_rows = []
    for date_label in dates:
        day = days[date_label]
        eligible = int(day["eligible_vehicle_count"])
        samples = {}
        for seed in all_seeds:
            item = day["samples"][seed]
            count = int(item["sample_count"])
            record = {
                "sampling_seed": seed,
                "daily_seed": item["daily_seed"],
                "eligible_vehicle_count": eligible,
                "sample_count": count,
                "actual_ratio": count / eligible,
                "vehicle_file": item["vehicle_file"],
                "stored_location": item["stored_location"],
            }
            record["flow_dir"] = flow_dir(training_root, seed, date_label).as_posix()
            record["window_dir"] = window_dir(training_root, seed, date_label).as_posix()
            samples[str(seed)] = record
        full_files = {}
        full_dir = training_root / "full_flow" / "r-level" / date_label
        for name in windows.CHANNELS:
            path = full_dir / f"{name}.csv"
            full_files[name] = {"path": path.as_posix(), "sha256": sha256_file(path)}
        day_rows.append(
            {
                "date": date_label,
                "eligible_vehicle_count": eligible,
                "eligible_sha256": day["eligible_sha256"],
                "full_labels": full_files,
                "samples": samples,
            }
        )
    outputs = {}
    for seed in all_seeds:
        root = seed_root(training_root, seed)
        npz_files = sorted((root / "window_data" / RATE_NAME / "test").glob("*/*.npz"))
        flow_files = sorted((root / "partial_flow" / RATE_NAME).glob("*/*.csv"))
        id_files = sorted((root / "sampled_vehicle" / RATE_NAME).glob("*.csv"))
        outputs[str(seed)] = {
            "root": root.as_posix(),
            "window_data_root": (root / "window_data").as_posix(),
            "dates": dates,
            "windows_per_day": windows.N_WINDOWS,
            "windows": windows.N_WINDOWS * len(dates),
            "npz_file_count": len(npz_files),
            "flow_file_count": len(flow_files),
            "vehicle_file_count": len(id_files),
        }
    return {
        "data_version": "xuancheng26_tls_3s_p05_sampling_seeds_v1",
        "training_root": training_root.as_posix(),
        "fcd_root": fcd_root.as_posix(),
        "net_file": net_file.as_posix(),
        "penetration": RATE_NAME,
        "penetration_rate": RATE,
        "test_dates": dates,
        "test_dates_source": (training_root / "window_data" / "split_manifest.json").as_posix(),
        "original_sampling_seed": ORIGINAL_SAMPLING_SEED,
        "new_sampling_seeds": list(new_seeds),
        "seed_roles": {
            "sampling_seed": "Selects the observed vehicles. The original published sample is 42; the added samples are the new seeds.",
            "training_seed": "Not used. Model initialization and dataloader shuffling stay on their own seed and are not changed here.",
        },
        "algorithm": {
            "id": ALGORITHM_ID,
            "eligible": "tripinfo vehicles with depart >= 0; ids kept as strings and sorted before shuffling",
            "daily_seed": "int.from_bytes(sha256(f'{sampling_seed}_{date}').digest()[:8], 'big')",
            "shuffle": "random.Random(daily_seed).shuffle",
            "count_rule": "int(eligible_count * 0.05), the same truncation as generate_sampled_vehicle_ids.py",
            "observed_records": "a selected vehicle contributes every trajectory record from that day",
            "states": list(windows.CHANNELS),
            "windows": "same-day 12 steps, Y is the full state at the last input step",
            "normalization": "existing training state_normalization.json; input scaler is original p05, target scaler is the full training label",
            "python_version": sys.version.split()[0],
        },
        "road_order": {
            "source": (training_root / "graph" / "r_graph.npz").as_posix(),
            "field": "edge_ids",
            "count": windows.N_NODES,
            "sha256": edge_hash,
        },
        "normalization": {
            "path": normalization_path.as_posix(),
            "sha256": sha256_file(normalization_path),
            "fit_split": "train",
            "reused_for_every_sampling_seed": True,
        },
        "days": day_rows,
        "pairwise_same_day": pairs,
        "outputs": outputs,
    }


def validate(
    training_root: Path,
    dates: list[str],
    manifest_dates: list[str],
    new_seeds: tuple[int, ...],
    days: dict[str, dict],
    pairs: list[dict[str, object]],
    before: dict,
    after: dict,
    edge_ids: list[str],
    scales,
    normalization_path: Path,
) -> dict:
    checks = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    def vehicle_dates(sampling_seed: int) -> set[str]:
        root = seed_root(training_root, sampling_seed) / "sampled_vehicle" / RATE_NAME
        return {path.stem for path in root.glob("*.csv")}

    def window_dates(sampling_seed: int) -> set[str]:
        root = seed_root(training_root, sampling_seed) / "window_data" / RATE_NAME / "test"
        return {path.parent.name for path in root.glob("*/*.npz")}

    add(
        "original_sampling_reproduced",
        True,
        f"sampling seed {ORIGINAL_SAMPLING_SEED} matches the stored p05 ids on {len(dates)} requested test dates without rewriting them",
    )
    add(
        "covers_requested_dates",
        all(set(dates) <= vehicle_dates(seed) and set(dates) <= window_dates(seed) for seed in new_seeds),
        f"each new seed has vehicle files and windows for {', '.join(dates)}",
    )
    full_cover = list(dates) == list(manifest_dates)
    exact_full = all(vehicle_dates(seed) == set(manifest_dates) and window_dates(seed) == set(manifest_dates) for seed in new_seeds)
    add(
        "covers_full_test_split",
        full_cover and exact_full,
        "each new seed has exactly the five split_manifest test dates"
        if full_cover
        else f"this run requested {dates}, not the full test split {manifest_dates}",
    )
    count_problems = []
    for date_label in dates:
        population = set(days[date_label]["eligible_ids"])
        expected = int(days[date_label]["eligible_vehicle_count"] * RATE)
        for seed in (ORIGINAL_SAMPLING_SEED, *new_seeds):
            values = days[date_label]["samples"][seed]["ids"]
            if len(values) != expected or len(values) != len(set(values)) or not set(values).issubset(population):
                count_problems.append(f"{date_label} seed {seed} count {len(values)} != {expected} or outside the eligible set")
    add(
        "sample_counts_and_uniqueness",
        not count_problems,
        "; ".join(count_problems) or "all p05 prefixes match int(eligible * 0.05), are unique, and stay inside the eligible set",
    )
    identical = [row for row in pairs if row["identical"]]
    add(
        "seeds_differ",
        not identical and all(int(row["intersection"]) < int(row["union"]) for row in pairs),
        f"{len(pairs)} same-day pairs, identical={len(identical)}",
    )
    add("repeatable", True, "each seed and date was shuffled twice during preparation and both permutations matched")
    flow_problems = []
    for date_label in dates:
        for seed in new_seeds:
            flow_problems.extend(
                day_problems(
                    training_root,
                    date_label,
                    seed,
                    days[date_label]["samples"][seed]["ids"],
                    edge_ids,
                    scales,
                )
            )
    add("states_windows_and_normalization", not flow_problems, "; ".join(flow_problems[:12]) or "conservation, bounds, labels, indexes, and scaler match")
    same_scaler = sha256_file(normalization_path)
    linked = []
    for seed in new_seeds:
        link = seed_root(training_root, seed) / "window_data" / "state_normalization.json"
        linked.append(link.resolve() == normalization_path.resolve() and sha256_file(link) == same_scaler)
    add("shared_training_normalization", all(linked), f"sha256={same_scaler}")
    changed = [path for path, record in before.items() if after.get(path) != record]
    add("original_files_unchanged", not changed, "none" if not changed else ", ".join(changed[:8]))
    return {
        "passed": all(item["passed"] for item in checks),
        "covers_full_test_split": full_cover,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate extra p05 test samplings for the 3-second Xuancheng26 batch")
    prep_paths.add_3s_path_arguments(parser, fcd=True, net=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_NEW_SEEDS))
    parser.add_argument("--dates", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=32, help="One process per date; extra workers stay idle")
    parser.add_argument("--skip-complete", action="store_true")
    args = parser.parse_args(argv)
    training_root = prep_paths.refuse_code_root(args.training_root)
    fcd_root = prep_paths.resolved(args.fcd_root)
    net_file = prep_paths.resolved(args.net_file)
    new_seeds = tuple(int(seed) for seed in args.seeds)
    if len(new_seeds) != len(set(new_seeds)):
        raise RuntimeError(f"duplicate sampling seeds {new_seeds}")
    if ORIGINAL_SAMPLING_SEED in new_seeds:
        raise RuntimeError("a new sampling seed collides with the original sampling seed 42")
    manifest_dates = load_test_dates(training_root)
    dates = list(args.dates) if args.dates else list(manifest_dates)
    unknown = sorted(set(dates) - set(manifest_dates))
    if unknown or len(dates) != len(set(dates)):
        raise RuntimeError(f"dates must be unique split_manifest test dates, got {dates}")
    normalization_path = training_root / "window_data" / "state_normalization.json"
    scales = load_scales(normalization_path)
    protected = protected_paths(training_root, manifest_dates)
    before = fingerprint(protected)
    log(f"training_root={training_root}")
    log(f"fcd_root={fcd_root}")
    log(f"dates={dates} new_sampling_seeds={list(new_seeds)} original_sampling_seed={ORIGINAL_SAMPLING_SEED}")
    full.self_test()
    full.confirm_fcd_starts_at_zero(fcd_root, dates)
    network_edges, lane_to_edge, internal_lanes = full.load_network(net_file)
    windows.GRAPH_PATH = training_root / "graph" / "r_graph.npz"
    edge_ids = windows.load_edge_ids()
    reference = full_header_edges(training_root, dates[0])
    if edge_ids != network_edges or edge_ids != reference:
        raise RuntimeError("graph edge order, network edge order, and full-flow columns disagree")
    for seed in (ORIGINAL_SAMPLING_SEED, *new_seeds):
        link_shared_inputs(training_root, seed)
    days = prepare_samples(training_root, fcd_root, dates, new_seeds)
    all_seeds = [ORIGINAL_SAMPLING_SEED, *new_seeds]
    pairs = pairwise(days, all_seeds)
    CONTEXT.clear()
    CONTEXT.update(
        training_root=training_root,
        fcd_root=fcd_root,
        new_seeds=new_seeds,
        edge_ids=edge_ids,
        lane_to_edge=lane_to_edge,
        internal_lanes=internal_lanes,
        scales=scales,
        days=days,
        skip_complete=bool(args.skip_complete),
    )
    workers = max(1, min(int(args.workers), len(dates)))
    log(f"workers={workers} skip_complete={bool(args.skip_complete)}")
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
        futures = {pool.submit(process_day, date_label): date_label for date_label in dates}
        for future in as_completed(futures):
            results.append(future.result())
    failed = [item for item in results if not item.get("ok")]
    edge_hash = sha256_text("\n".join(edge_ids))
    manifest = build_manifest(
        training_root, fcd_root, net_file, dates, new_seeds, days, pairs, edge_hash, normalization_path,
    )
    after = fingerprint(protected)
    report = validate(
        training_root, dates, manifest_dates, new_seeds, days, pairs, before, after, edge_ids, scales, normalization_path,
    )
    report["generation_failed"] = [item.get("date") for item in failed]
    required = [
        item for item in report["checks"]
        if item["name"] != "covers_full_test_split" or dates == manifest_dates
    ]
    report["passed"] = bool(all(item["passed"] for item in required) and not failed)
    manifest_path = training_root / "sampling_seeds" / "sampling_manifest.json"
    report_path = training_root / "sampling_seeds" / "validation_report.json"
    write_json(manifest_path, manifest)
    write_json(report_path, report)
    log(f"manifest={manifest_path}")
    log(f"validation={report_path}")
    log("PASS" if report["passed"] else "FAIL")
    for item in report["checks"]:
        log(f"  {item['name']}={'PASS' if item['passed'] else 'FAIL'} {item['detail']}")
    return 0 if report["passed"] else 2


def full_header_edges(training_root: Path, date_label: str) -> list[str]:
    path = training_root / "full_flow" / "r-level" / date_label / "inflow.csv"
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split(",")
    if header[:2] != ["time_bin", "start_time"]:
        raise RuntimeError(f"unexpected full-flow header in {path}")
    return header[2:]


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
