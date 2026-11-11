#!/usr/bin/env python3
"""Build uncapped speed inputs for the MSTT ``all_sp_c`` channel pair.

One worker reads one day's 3-second FCD once and updates every selected
dataset and sampling seed. ``V`` is the mean of the valid 1-second snapshots
in km/h, with no speed-limit cap and no limit fill. ``C`` is the snapshot
count, not a unique-vehicle count and not divided by 300. Empty bins store
``V=0`` and ``C=0``.

The four published states, their windows, and their scalers are only read.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import csv
import gc
import hashlib
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

CODE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import generate_r_level_windows as windows
import prepare_rf_features as rf
from data_root import (
    all_sp_c_normalization_path,
    all_sp_c_ready_path,
    data_root,
    rf_observation_feature_path,
    sampling_window_root,
    traffic_observation_feature_path,
)
from independent_states import PHYSICAL_VERSION
from MSTT.config import ALL_SP_C_CHANNELS, STATE_CHANNELS
from prep_paths import refuse_code_root

DATASET_IDS = (
    "variable_30min_p10_p20",
    "variable_1h_p10_p20",
    "variable_2h_p10_p20",
    "variable_day_p10_p20",
)
OBSERVATION_KEY = "p10_p20"
SEEDS = (42, 43, 44, 45, 46)
DEFINITION_VERSION = "all_sp_c_raw_mean_v1"
EXTRA_CHANNELS = ALL_SP_C_CHANNELS[len(STATE_CHANNELS) :]
if EXTRA_CHANNELS != ("average_speed_raw", "speed_observation_count"):
    raise RuntimeError(f"unexpected extra channels {EXTRA_CHANNELS}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def raw_speed_and_count(speed_sum_ms: np.ndarray, count: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean km/h over snapshots. A zero count stays zero and is not filled."""
    if int(count.min()) < 0:
        raise rf.FeatureError("negative observation count")
    if int(count.max()) > np.iinfo(np.int32).max:
        raise rf.FeatureError("observation count does not fit int32")
    mean_kmh = np.zeros(speed_sum_ms.shape, dtype=np.float64)
    np.divide(speed_sum_ms * rf.MS_TO_KMH, count, out=mean_kmh, where=count > 0)
    if not np.isfinite(mean_kmh).all():
        raise rf.FeatureError("raw speed mean is not finite")
    empty = count == 0
    if not np.all(mean_kmh[empty] == 0.0):
        raise rf.FeatureError("an empty bin did not stay at V=0")
    return mean_kmh.astype(np.float32), count.astype(np.int32, copy=False)


def load_edges(training_root: Path) -> tuple[list[str], str]:
    mapped = []
    node_file = training_root / "shared" / "node_mapping.csv"
    with node_file.open(encoding="utf-8", newline="") as handle:
        mapped = [row["edge_id"] for row in csv.DictReader(handle)]
    if mapped != list(rf.EDGES):
        raise rf.FeatureError("r_nodes.csv order differs from shared/node_mapping.csv")
    return mapped, sha256_file(node_file)


def split_of(manifest: dict, date: str) -> str:
    for name in ("train", "validation", "test"):
        if date in manifest[name]:
            return name
    raise rf.FeatureError(f"{date} is not in the shared split manifest")


def groups_for(training: Path, date: str, *, split: str, edges: list[str]) -> list[dict]:
    groups = []
    for dataset_id in DATASET_IDS:
        manifest_path = training / "datasets" / dataset_id / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_id") != dataset_id:
            raise rf.FeatureError(f"{dataset_id} manifest id is {manifest.get('dataset_id')!r}")
        if tuple(manifest.get("observation_keys") or ()) != (OBSERVATION_KEY,):
            raise rf.FeatureError(f"{dataset_id} observation keys are not {(OBSERVATION_KEY,)}")
        if tuple(int(item) for item in manifest.get("sampling_seeds") or ()) != SEEDS:
            raise rf.FeatureError(f"{dataset_id} sampling seeds are not {SEEDS}")
        for seed in SEEDS:
            members = (
                training / "datasets" / dataset_id / "sampling_seeds" / f"seed_{seed}"
                / "sampled_vehicle" / OBSERVATION_KEY / f"{date}.csv"
            )
            indices, _owners = rf_load_indices(members)
            window = (
                sampling_window_root(training / "datasets" / dataset_id / "sampling_seeds", seed)
                / OBSERVATION_KEY / split / date / "all.npz"
            )
            if not window.is_file():
                raise rf.FeatureError(f"missing published window {window}")
            groups.append({
                "dataset_id": dataset_id,
                "sampling_seed": int(seed),
                "key": OBSERVATION_KEY,
                "date": date,
                "split": split,
                "members": str(members),
                "members_sha256": sha256_file(members),
                "out": str(traffic_observation_feature_path(training, dataset_id, seed, OBSERVATION_KEY, date)),
                "window_all": str(window),
                "rf": str(rf_observation_feature_path(training, dataset_id, seed, OBSERVATION_KEY, date)),
                "indices": indices,
                "n_edges": len(edges),
            })
    if len(groups) != len(DATASET_IDS) * len(SEEDS):
        raise rf.FeatureError(f"{date} produced {len(groups)} groups")
    return groups


def rf_load_indices(path: Path) -> tuple[list[int], dict[int, str]]:
    from prepare_published_rf_features import load_indices

    return load_indices(path)


def _text(payload, key: str) -> str:
    if key not in payload.files:
        return ""
    value = payload[key]
    if getattr(value, "shape", None) == ():
        return str(value)
    return str(value.reshape(-1)[0]) if value.size else ""


def daily_current(path: Path, group: dict, *, edges: list[str], source_fcd: Path, node_sha: str) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with np.load(path, mmap_mode="r", allow_pickle=False) as payload:
            required = (
                "average_speed_raw",
                "speed_observation_count",
                "edge_ids",
                "time_bins",
                "date",
                "dataset_id",
                "sampling_seed",
                "observation_key",
                "vehicle_list_sha256",
                "vehicle_list",
                "source_fcd",
                "fcd_size",
                "fcd_mtime_ns",
                "node_order_sha256",
                "definition_version",
                "speed_unit",
                "count_unit",
                "channels",
            )
            if any(key not in payload.files for key in required):
                return False
            speed = payload["average_speed_raw"]
            count = payload["speed_observation_count"]
            stat = source_fcd.stat()
            channels = tuple(str(item) for item in payload["channels"].tolist())
            ok = (
                channels == EXTRA_CHANNELS
                and speed.shape == (rf.N_BINS, len(edges))
                and speed.dtype == np.float32
                and count.shape == speed.shape
                and count.dtype == np.int32
                and [str(item) for item in payload["edge_ids"].tolist()] == edges
                and np.array_equal(payload["time_bins"], np.arange(rf.N_BINS))
                and _text(payload, "date") == group["date"]
                and _text(payload, "dataset_id") == group["dataset_id"]
                and int(payload["sampling_seed"]) == int(group["sampling_seed"])
                and _text(payload, "observation_key") == group["key"]
                and _text(payload, "vehicle_list_sha256") == group["members_sha256"]
                and _text(payload, "vehicle_list") == group["members"]
                and _text(payload, "source_fcd") == str(source_fcd)
                and int(payload["fcd_size"]) == int(stat.st_size)
                and int(payload["fcd_mtime_ns"]) == int(stat.st_mtime_ns)
                and _text(payload, "node_order_sha256") == node_sha
                and _text(payload, "definition_version") == DEFINITION_VERSION
                and _text(payload, "speed_unit") == "km/h"
                and _text(payload, "count_unit") == "fcd_snapshot"
                and int(payload["complete_day"]) == 1
                and np.isfinite(speed).all()
                and int(count.min()) >= 0
                and bool(np.all(speed[count == 0] == 0))
            )
            return bool(ok)
    except Exception:
        return False


def _init_worker(payload: dict) -> None:
    rf.FCD_ROOT = Path(payload["fcd_root"])
    rf.LANE_TO_INDEX = payload["lane_to_index"]
    rf.INTERNAL_LANES = payload["internal_lanes"]
    rf.BASE_MS = payload["base_ms"]
    rf.BASE_KMH = payload["base_kmh"]
    rf.EDGES = payload["edges"]


def process_feature_day(task: dict) -> dict:
    date = task["date"]
    started = time.perf_counter()
    try:
        source = Path(task["source_fcd"])
        edges = list(task["edges"])
        pending = []
        for group in task["groups"]:
            path = Path(group["out"])
            if task["force"] or not daily_current(
                path, group, edges=edges, source_fcd=source, node_sha=task["node_order_sha256"],
            ):
                pending.append(group)
        if not pending:
            return {"date": date, "ok": True, "action": "skipped", "groups": len(task["groups"]), "written": 0}
        mask = rf.build_mask([group["indices"] for group in pending])
        rf.log(f"[{date}] all_sp_c groups={len(pending)} fcd={source.name}")
        gc.disable()
        try:
            sums, counts, stats = rf.parse_fcd_stream(
                rf.iter_file_blocks(source),
                mask=mask,
                lane_to_index=rf.LANE_TO_INDEX,
                internal_lanes=rf.INTERNAL_LANES,
                base_ms=rf.BASE_MS,
                n_rates=len(pending),
                n_edges=len(edges),
                until_time=None,
                progress_label=date,
            )
        finally:
            gc.enable()
        if bool(stats["stopped_early"]):
            raise rf.FeatureError("parser stopped early on a full-day request")
        if int(stats["unknown_lane"]) > 0:
            raise rf.FeatureError(f"unknown lanes {stats['unknown_examples']}")
        weekday = rf.weekday_index(date)
        written = 0
        for index, group in enumerate(pending):
            speed, count = raw_speed_and_count(sums[index], counts[index])
            _check_rf_count(group, count)
            out = Path(group["out"])
            refuse_code_root(out)
            atomic_npz(
                out,
                average_speed_raw=speed,
                speed_observation_count=count,
                edge_ids=np.asarray(edges, dtype="U32"),
                time_bins=np.arange(rf.N_BINS, dtype=np.int16),
                hour=(np.arange(rf.N_BINS, dtype=np.int16) // 12),
                weekday=np.int16(weekday),
                complete_day=np.int8(1),
                date=np.asarray(date),
                dataset_id=np.asarray(group["dataset_id"]),
                sampling_seed=np.int16(group["sampling_seed"]),
                observation_key=np.asarray(group["key"]),
                vehicle_list=np.asarray(group["members"]),
                vehicle_list_sha256=np.asarray(group["members_sha256"]),
                source_fcd=np.asarray(str(source)),
                fcd_size=np.int64(source.stat().st_size),
                fcd_mtime_ns=np.int64(source.stat().st_mtime_ns),
                node_order_sha256=np.asarray(task["node_order_sha256"]),
                definition_version=np.asarray(DEFINITION_VERSION),
                physical_definition_version=np.asarray(PHYSICAL_VERSION),
                speed_unit=np.asarray("km/h"),
                count_unit=np.asarray("fcd_snapshot"),
                channels=np.asarray(EXTRA_CHANNELS, dtype="U32"),
                used_record_count=np.int64(stats["used_records"][index]),
                above_base_observation_count=np.int64(stats["above_base_observations"][index]),
                empty_bin_count=np.int64(np.count_nonzero(count == 0)),
                zero_mean_positive_count=np.int64(np.count_nonzero((count > 0) & (speed == 0))),
            )
            if not daily_current(out, group, edges=edges, source_fcd=source, node_sha=task["node_order_sha256"]):
                raise rf.FeatureError(f"daily feature failed reread: {out}")
            written += 1
            del speed, count
        elapsed = time.perf_counter() - started
        rf.log(f"[{date}] all_sp_c written={written} seconds={elapsed:.1f}")
        return {"date": date, "ok": True, "action": "written", "groups": len(pending), "written": written}
    except Exception:
        return {"date": date, "ok": False, "error": traceback.format_exc(limit=12)}


def _check_rf_count(group: dict, count: np.ndarray) -> None:
    """Published RF counts are a sampling check, not a source for V."""
    path = Path(group["rf"])
    if not path.is_file():
        return
    with np.load(path, mmap_mode="r", allow_pickle=False) as payload:
        if "speed_observation_count" not in payload.files:
            return
        if "vehicle_list_sha256" in payload.files and str(payload["vehicle_list_sha256"]) != group["members_sha256"]:
            return
        published = payload["speed_observation_count"]
        if published.shape != count.shape or not np.array_equal(published, count):
            raise rf.FeatureError(
                f"RF snapshot count differs for {group['dataset_id']} seed {group['sampling_seed']} {group['date']}"
            )


def _history(starts: np.ndarray) -> np.ndarray:
    if not np.array_equal(starts, np.arange(windows.N_WINDOWS, dtype=starts.dtype)):
        raise rf.FeatureError("window_start_slot is not 0..276")
    return starts.astype(np.int64)[:, None] + np.arange(windows.N_HIS, dtype=np.int64)[None, :]


def slice_extra(speed: np.ndarray, count: np.ndarray, starts: np.ndarray) -> np.ndarray:
    history = _history(starts)
    if int(history.max()) >= speed.shape[0]:
        raise rf.FeatureError("window index leaves the day")
    stacked = np.stack([speed[history], count[history].astype(np.float32)], axis=-1)
    if stacked.shape != (windows.N_WINDOWS, windows.N_HIS, windows.N_NODES, 2):
        raise rf.FeatureError(f"sliced speed shape {stacked.shape}")
    return np.ascontiguousarray(stacked)


def window_current(
    path: Path,
    *,
    all_path: Path,
    daily_path: Path,
    scaler: dict,
    group: dict,
) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with np.load(all_path, mmap_mode="r", allow_pickle=False) as published, np.load(
            daily_path, mmap_mode="r", allow_pickle=False
        ) as daily, np.load(path, mmap_mode="r", allow_pickle=False) as payload:
            if any(key not in payload.files for key in windows.NPZ_KEYS):
                return False
            x = payload["x"]
            x_raw = payload["x_raw"]
            expected_raw = slice_extra(daily["average_speed_raw"], daily["speed_observation_count"], published["window_start_slot"])
            mean = np.asarray([scaler["input"][name]["mean"] for name in EXTRA_CHANNELS], dtype=np.float32)
            std = np.asarray([scaler["input"][name]["std"] for name in EXTRA_CHANNELS], dtype=np.float32)
            expected_x = windows._normalize(expected_raw, mean.reshape(1, 1, 1, -1), std.reshape(1, 1, 1, -1))
            channels = tuple(str(item) for item in payload["channels"].tolist()) if "channels" in payload.files else ()
            return bool(
                channels == ALL_SP_C_CHANNELS
                and x.shape == (windows.N_WINDOWS, windows.N_HIS, windows.N_NODES, 6)
                and x.dtype == np.float32
                and x_raw.shape == x.shape
                and x_raw.dtype == np.float32
                and payload["y"].shape == published["y"].shape
                and np.array_equal(x[..., :4], published["x"])
                and np.array_equal(x_raw[..., :4], published["x_raw"])
                and np.array_equal(payload["y"], published["y"])
                and np.array_equal(payload["y_raw"], published["y_raw"])
                and np.array_equal(payload["day_index"], published["day_index"])
                and np.array_equal(payload["window_start_slot"], published["window_start_slot"])
                and np.array_equal(payload["target_slot"], published["target_slot"])
                and np.array_equal(x_raw[..., 4:], expected_raw)
                and np.allclose(x[..., 4:], expected_x, rtol=0.0, atol=1e-5)
                and _text(payload, "dataset_id") == group["dataset_id"]
                and int(payload["sampling_seed"]) == int(group["sampling_seed"])
                and _text(payload, "definition_version") == DEFINITION_VERSION
                and _text(payload, "daily_feature_sha256") == sha256_file(daily_path)
                and _text(payload, "source_all_sha256") == sha256_file(all_path)
                and _text(payload, "scaler_sha256") == scaler["sha256"]
                and np.isfinite(x).all()
                and np.isfinite(payload["y"]).all()
            )
    except Exception:
        return False


def process_window_day(task: dict) -> dict:
    date = task["date"]
    try:
        written = skipped = 0
        for group in task["groups"]:
            dataset_id = group["dataset_id"]
            scaler = task["scalers"][dataset_id]
            out = (
                sampling_window_root(
                    Path(task["training_root"]) / "datasets" / dataset_id / "sampling_seeds",
                    int(group["sampling_seed"]),
                )
                / OBSERVATION_KEY / group["split"] / date / "all_sp_c.npz"
            )
            daily = Path(group["out"])
            all_path = Path(group["window_all"])
            if not task["force"] and window_current(out, all_path=all_path, daily_path=daily, scaler=scaler, group=group):
                skipped += 1
                continue
            with np.load(all_path, mmap_mode="r", allow_pickle=False) as published, np.load(
                daily, mmap_mode="r", allow_pickle=False
            ) as daily_payload:
                extra_raw = slice_extra(
                    np.array(daily_payload["average_speed_raw"]),
                    np.array(daily_payload["speed_observation_count"]),
                    np.array(published["window_start_slot"]),
                )
                mean = np.asarray([scaler["input"][name]["mean"] for name in EXTRA_CHANNELS], dtype=np.float32)
                std = np.asarray([scaler["input"][name]["std"] for name in EXTRA_CHANNELS], dtype=np.float32)
                extra_x = windows._normalize(extra_raw, mean.reshape(1, 1, 1, -1), std.reshape(1, 1, 1, -1))
                arrays = {
                    "x": np.concatenate([np.array(published["x"]), extra_x], axis=-1),
                    "x_raw": np.concatenate([np.array(published["x_raw"]), extra_raw], axis=-1),
                    "y": np.array(published["y"]),
                    "y_raw": np.array(published["y_raw"]),
                    "day_index": np.array(published["day_index"]),
                    "window_start_slot": np.array(published["window_start_slot"]),
                    "target_slot": np.array(published["target_slot"]),
                }
            arrays.update({
                "channels": np.asarray(ALL_SP_C_CHANNELS, dtype="U32"),
                "dataset_id": np.asarray(dataset_id),
                "sampling_seed": np.int16(group["sampling_seed"]),
                "observation_key": np.asarray(OBSERVATION_KEY),
                "date": np.asarray(date),
                "definition_version": np.asarray(DEFINITION_VERSION),
                "daily_feature_sha256": np.asarray(sha256_file(daily)),
                "source_all_sha256": np.asarray(sha256_file(all_path)),
                "scaler_sha256": np.asarray(scaler["sha256"]),
                "target_source": np.asarray("all.npz"),
            })
            refuse_code_root(out)
            atomic_npz(out, **arrays)
            if not window_current(out, all_path=all_path, daily_path=daily, scaler=scaler, group=group):
                raise rf.FeatureError(f"window failed reread: {out}")
            written += 1
            del arrays
        rf.log(f"[{date}] all_sp_c windows written={written} skipped={skipped}")
        return {"date": date, "ok": True, "written": written, "skipped": skipped}
    except Exception:
        return {"date": date, "ok": False, "error": traceback.format_exc(limit=12)}


def _moment_pair(speed: np.ndarray, count: np.ndarray) -> tuple[tuple[float, float, int], tuple[float, float, int]]:
    return windows._moment(speed), windows._moment(count.astype(np.float64))


def fit_scaler(training: Path, dataset_id: str, train_days: list[str], *, force: bool) -> dict:
    totals = {name: [0.0, 0.0, 0] for name in EXTRA_CHANNELS}
    for day in train_days:
        path = traffic_observation_feature_path(training, dataset_id, 42, OBSERVATION_KEY, day)
        with np.load(path, mmap_mode="r", allow_pickle=False) as payload:
            if str(payload["dataset_id"]) != dataset_id or int(payload["sampling_seed"]) != 42:
                raise rf.FeatureError(f"scaler source identity mismatch: {path}")
            speed_moment, count_moment = _moment_pair(payload["average_speed_raw"], payload["speed_observation_count"])
        for name, moment in zip(EXTRA_CHANNELS, (speed_moment, count_moment)):
            bucket = totals[name]
            bucket[0] += moment[0]
            bucket[1] += moment[1]
            bucket[2] += moment[2]
    fitted = {name: windows._mean_std(*totals[name]) for name in EXTRA_CHANNELS}
    try:
        commit = __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=CODE, text=True).strip()
    except Exception:
        commit = ""
    payload = {
        "definition_version": DEFINITION_VERSION,
        "dataset_id": dataset_id,
        "observation_key": OBSERVATION_KEY,
        "fit_split": "train",
        "fit_sampling_seed": 42,
        "ddof": 0,
        "formula": "(value - mean) / std",
        "scope": "global_scalar_per_channel",
        "aggregation": "sampling seed 42 training days, raw 288 bins by 1406 nodes, zeros included, no sliding-window reweight",
        "empty_bin": "V=0 and C=0 participate",
        "std_zero_policy": "generate_r_level_windows._mean_std rejects a zero standard deviation",
        "channels": list(EXTRA_CHANNELS),
        "training_days": list(train_days),
        "input": {
            name: {"mean": fitted[name][0], "std": fitted[name][1], "unit": "km/h" if name == "average_speed_raw" else "fcd_snapshot"}
            for name in EXTRA_CHANNELS
        },
        "cell_count": {name: totals[name][2] for name in EXTRA_CHANNELS},
        "physical_definition_version_reference": PHYSICAL_VERSION,
        "git_commit": commit,
    }
    path = all_sp_c_normalization_path(training, dataset_id)
    if path.is_file() and not force:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not _scaler_same(existing, payload):
            raise rf.FeatureError(f"all_sp_c scaler conflict: {path}")
        payload = existing
    else:
        atomic_json(path, payload)
        stored = json.loads(path.read_text(encoding="utf-8"))
        if not _scaler_same(stored, payload):
            raise rf.FeatureError(f"scaler failed reread: {path}")
        payload = stored
    payload = dict(payload)
    payload["sha256"] = sha256_file(path)
    return payload


def _scaler_same(left: dict, right: dict) -> bool:
    keys = (
        "definition_version",
        "dataset_id",
        "observation_key",
        "fit_split",
        "fit_sampling_seed",
        "ddof",
        "channels",
        "training_days",
        "aggregation",
    )
    if any(left.get(key) != right.get(key) for key in keys):
        return False
    for name in EXTRA_CHANNELS:
        for field in ("mean", "std"):
            if abs(float(left["input"][name][field]) - float(right["input"][name][field])) > 1e-12:
                return False
    return True


def protected_stats(training: Path, manifest: dict) -> dict[str, tuple[int, int]]:
    paths = [
        training / "shared" / "target_normalization.json",
        training / "shared" / "physical_definition.json",
        training / "shared" / "split_manifest.json",
        training / "shared" / "state_normalization.json",
    ]
    for dataset_id in DATASET_IDS:
        root = training / "datasets" / dataset_id
        paths.extend([
            root / "dataset_manifest.json",
            root / "shared" / "state_normalization.json",
            root / "shared" / "validation_summary.json",
            root / "shared" / "state_window_manifest.json",
        ])
        for seed in SEEDS:
            window_root = sampling_window_root(root / "sampling_seeds", seed)
            for split, days in (("train", manifest["train"]), ("validation", manifest["validation"]), ("test", manifest["test"])):
                for day in days:
                    directory = window_root / OBSERVATION_KEY / split / day
                    paths.append(directory / "all.npz")
                    paths.append(directory / "inflow.npz")
    stats = {}
    for path in paths:
        if not path.is_file():
            raise rf.FeatureError(f"protected input missing before all_sp_c generation: {path}")
        item = path.stat()
        stats[str(path)] = (int(item.st_size), int(item.st_mtime_ns))
    return stats


def assert_unchanged(before: dict[str, tuple[int, int]]) -> None:
    changed = []
    for name, (size, mtime) in before.items():
        path = Path(name)
        item = path.stat()
        if int(item.st_size) != size or int(item.st_mtime_ns) != mtime:
            changed.append(name)
    if changed:
        raise rf.FeatureError(f"published files changed during all_sp_c generation: {changed[:8]}")


def _run_pool(function, tasks: list[dict], workers: int, init: dict) -> list[dict]:
    import multiprocessing as mp

    rows = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("fork"),
        initializer=_init_worker,
        initargs=(init,),
    ) as pool:
        futures = {pool.submit(function, task): task["date"] for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            if not row.get("ok"):
                rf.log(f"[{row.get('date')}] FAIL {(row.get('error') or '')[-800:]}")
    return rows


def expected_paths(training: Path, manifest: dict) -> tuple[list[Path], list[Path]]:
    daily = []
    windows_out = []
    for dataset_id in DATASET_IDS:
        for seed in SEEDS:
            for split, days in (("train", manifest["train"]), ("validation", manifest["validation"]), ("test", manifest["test"])):
                for day in days:
                    daily.append(traffic_observation_feature_path(training, dataset_id, seed, OBSERVATION_KEY, day))
                    windows_out.append(
                        sampling_window_root(training / "datasets" / dataset_id / "sampling_seeds", seed)
                        / OBSERVATION_KEY / split / day / "all_sp_c.npz"
                    )
    return daily, windows_out


def write_ready(training: Path, manifest: dict, scalers: dict[str, dict]) -> None:
    for dataset_id in DATASET_IDS:
        daily = windows_out = 0
        for seed in SEEDS:
            for _split, days in (("train", manifest["train"]), ("validation", manifest["validation"]), ("test", manifest["test"])):
                daily += len(days)
                windows_out += len(days)
        payload = {
            "feature": "all_sp_c",
            "status": "ready",
            "dataset_id": dataset_id,
            "observation_key": OBSERVATION_KEY,
            "definition_version": DEFINITION_VERSION,
            "sampling_seeds": list(SEEDS),
            "daily_files": daily,
            "window_files": windows_out,
            "normalization": all_sp_c_normalization_path(training, dataset_id).name,
            "normalization_sha256": scalers[dataset_id]["sha256"],
            "fit_sampling_seed": 42,
            "fit_split": "train",
            "note": "This file does not change dataset_status of the four-state package.",
        }
        if daily != 5 * 29 or windows_out != 5 * 29:
            raise rf.FeatureError(f"{dataset_id} ready count drifted: {daily} {windows_out}")
        atomic_json(all_sp_c_ready_path(training, dataset_id), payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare all_sp_c speed channels for four variable datasets")
    parser.add_argument("--training-root", type=Path, default=data_root() / "xuancheng26/training_data_3s")
    parser.add_argument("--fcd-root", type=Path, default=data_root() / "xuancheng26/simulation_data/fcd_data_3s")
    parser.add_argument("--net-file", type=Path, default=CODE / "dataset/xuancheng26/xuancheng_tls_3s.net.xml")
    parser.add_argument("--workers", type=int, default=29)
    parser.add_argument("--dates", nargs="*")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    training = refuse_code_root(args.training_root)
    rf.FCD_ROOT = args.fcd_root.expanduser().resolve()
    rf.TRAIN_ROOT = training
    rf.NET_FILE = args.net_file.expanduser().resolve()
    rf.NODES_PATH = training / "graph" / "r_nodes.csv"
    rf.GRAPH_PATH = training / "graph" / "r_graph.npz"
    rf.FULL_ROOT = training / "full_flow" / "r-level"
    edges, _node_lanes, speed_text = rf.load_canonical_edges()
    lane_to_index, internal_lanes, speed_ms, _lane_count = rf.load_network_attributes(edges, _node_lanes, speed_text)
    rf.LANE_TO_INDEX = lane_to_index
    rf.INTERNAL_LANES = internal_lanes
    rf.BASE_MS = speed_ms
    rf.BASE_KMH = speed_ms * rf.MS_TO_KMH
    rf.EDGES = edges
    edges, node_sha = load_edges(training)
    manifest = json.loads((training / "shared" / "split_manifest.json").read_text(encoding="utf-8"))
    dates = [str(day) for name in ("train", "validation", "test") for day in manifest[name]]
    if args.dates:
        unknown = sorted(set(args.dates) - set(dates))
        if unknown:
            raise rf.FeatureError(f"dates outside the shared split: {unknown}")
        selected = [day for day in dates if day in set(args.dates)]
    else:
        selected = dates
    if len(dates) != 29:
        raise rf.FeatureError(f"expected 29 dates, found {len(dates)}")
    before = protected_stats(training, manifest)
    fcd_format = rf.benchmark_fcd(selected[0])
    tasks = []
    for date in selected:
        source = rf.resolve_fcd(date, fcd_format)
        tasks.append({
            "date": date,
            "source_fcd": str(source),
            "force": bool(args.force),
            "edges": edges,
            "node_order_sha256": node_sha,
            "groups": groups_for(training, date, split=split_of(manifest, date), edges=edges),
        })
    init = {
        "fcd_root": str(rf.FCD_ROOT),
        "lane_to_index": lane_to_index,
        "internal_lanes": internal_lanes,
        "base_ms": speed_ms,
        "base_kmh": rf.BASE_KMH,
        "edges": edges,
    }
    workers = max(1, min(int(args.workers), len(tasks)))
    rf.log(f"all_sp_c days={len(tasks)} workers={workers} fcd_format={fcd_format} definition={DEFINITION_VERSION}")
    feature_rows = _run_pool(process_feature_day, tasks, workers, init)
    failed = [row for row in feature_rows if not row.get("ok")]
    if failed:
        raise rf.FeatureError(f"{len(failed)} feature days failed")
    if args.dates and set(args.dates) != set(dates):
        rf.log("partial date list: scaler and windows wait for a full 29-day feature pass")
        assert_unchanged(before)
        return 0
    scalers = {dataset_id: fit_scaler(training, dataset_id, list(manifest["train"]), force=bool(args.force)) for dataset_id in DATASET_IDS}
    for task in tasks:
        task["scalers"] = scalers
        task["training_root"] = str(training)
    window_rows = _run_pool(process_window_day, tasks, workers, init)
    failed = [row for row in window_rows if not row.get("ok")]
    if failed:
        raise rf.FeatureError(f"{len(failed)} window days failed")
    daily_paths, window_paths = expected_paths(training, manifest)
    missing = [str(path) for path in daily_paths + window_paths if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise rf.FeatureError(f"missing all_sp_c products: {missing[:8]} count={len(missing)}")
    write_ready(training, manifest, scalers)
    assert_unchanged(before)
    summary = {
        "daily_files": len(daily_paths),
        "window_files": len(window_paths),
        "definition_version": DEFINITION_VERSION,
        "scalers": {
            dataset_id: {
                "sha256": item["sha256"],
                "fit_sampling_seed": item["fit_sampling_seed"],
                "training_days": len(item["training_days"]),
                "input": item["input"],
            }
            for dataset_id, item in scalers.items()
        },
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
