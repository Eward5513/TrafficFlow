#!/usr/bin/env python3
"""Build uncapped speed inputs for the MSTT ``all_sp_c`` channel pair.

One worker reads one day's 3-second FCD once and updates every selected
dataset and sampling seed. ``V`` is the mean of the valid 1-second snapshots
in km/h, with no speed-limit cap and no limit fill. ``C`` is the snapshot
count, not a unique-vehicle count and not divided by 300. Empty bins store
``V=0`` and ``C=0``.

``--dataset-ids`` selects which published datasets to extend. The default is
still the four ``p10_p20`` datasets. Each dataset contributes exactly one
observation key, read from its manifest. A mixed list keeps each dataset on
its own vehicle lists, windows, and scaler.

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
    PathLayoutError,
    all_sp_c_normalization_path,
    all_sp_c_ready_path,
    data_root,
    observation_key,
    resolve_xuancheng_dataset,
    rf_observation_feature_path,
    sampled_vehicle_root,
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
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
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


def dedupe_dataset_ids(dataset_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in dataset_ids:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    if not ordered:
        raise rf.FeatureError("dataset id list is empty")
    return ordered


def load_range_configs() -> dict[str, dict]:
    """Read the range JSON files. They check a published schedule; they do not rebuild it."""
    found: dict[str, dict] = {}
    paths = sorted(CONFIG_DIR.glob("variable_*.json"))
    if not paths:
        raise rf.FeatureError(f"no variable range configs in {CONFIG_DIR}")
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in payload.get("datasets") or []:
            dataset_id = str(entry.get("dataset_id") or "")
            if not dataset_id:
                raise rf.FeatureError(f"{path} has a dataset entry without dataset_id")
            if dataset_id in found:
                raise rf.FeatureError(f"{dataset_id} is declared by both {found[dataset_id]['config_path']} and {path}")
            found[dataset_id] = {
                "config_path": path,
                "entry": entry,
                "p_min": float(payload["p_min"]),
                "p_max": float(payload["p_max"]),
                "sampling_seeds": tuple(int(item) for item in payload["sampling_seeds"]),
                "schedule_seed": int(payload["schedule_seed"]),
            }
    return found


def resolve_dataset_specs(training: Path, dataset_ids: list[str]) -> list[dict]:
    """One spec per dataset. The observation key comes from that dataset's manifest."""
    configs = load_range_configs()
    specs = []
    for dataset_id in dedupe_dataset_ids(dataset_ids):
        try:
            resolved = resolve_xuancheng_dataset(dataset_id=dataset_id, training_root=training)
        except PathLayoutError as exc:
            raise rf.FeatureError(str(exc)) from exc
        keys = tuple(str(item) for item in resolved["observation_keys"])
        if len(keys) != 1:
            raise rf.FeatureError(f"{dataset_id} must declare exactly one observation key, got {keys}")
        key = keys[0]
        manifest_path = Path(resolved["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_id") != dataset_id:
            raise rf.FeatureError(f"{dataset_id} manifest id is {manifest.get('dataset_id')!r}")
        seeds = tuple(int(item) for item in manifest.get("sampling_seeds") or ())
        if seeds != SEEDS:
            raise rf.FeatureError(f"{dataset_id} sampling seeds are not {SEEDS}")
        if manifest.get("physical_definition_version") != PHYSICAL_VERSION:
            raise rf.FeatureError(
                f"{dataset_id} physical version is {manifest.get('physical_definition_version')!r}, "
                f"expected {PHYSICAL_VERSION}"
            )
        if dataset_id not in configs:
            raise rf.FeatureError(f"{dataset_id} has no range config under {CONFIG_DIR}")
        config = configs[dataset_id]
        expected_key = observation_key(config["p_min"], config["p_max"])
        if expected_key != key:
            raise rf.FeatureError(
                f"{dataset_id} manifest key {key} does not match config range "
                f"{config['p_min']}..{config['p_max']} ({expected_key})"
            )
        entry = config["entry"]
        if str(entry.get("mode")) != str(manifest.get("mode")):
            raise rf.FeatureError(
                f"{dataset_id} mode {manifest.get('mode')!r} does not match config {entry.get('mode')!r}"
            )
        if int(entry.get("segment_seconds")) != int(manifest.get("segment_seconds")):
            raise rf.FeatureError(
                f"{dataset_id} segment_seconds {manifest.get('segment_seconds')!r} "
                f"does not match config {entry.get('segment_seconds')!r}"
            )
        if config["sampling_seeds"] != seeds:
            raise rf.FeatureError(f"{dataset_id} config sampling seeds {config['sampling_seeds']} != manifest {seeds}")
        if int(manifest.get("schedule_seed")) != int(config["schedule_seed"]):
            raise rf.FeatureError(
                f"{dataset_id} schedule_seed {manifest.get('schedule_seed')!r} != config {config['schedule_seed']}"
            )
        schedule_path = Path(resolved["dataset_root"]) / "penetration_schedule.json"
        if not schedule_path.is_file():
            raise rf.FeatureError(f"missing penetration schedule {schedule_path}")
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        if schedule.get("schedule_sha256") != manifest.get("schedule_sha256"):
            raise rf.FeatureError(f"{dataset_id} schedule sha does not match the manifest")
        specs.append({
            "dataset_id": dataset_id,
            "dataset_root": Path(resolved["dataset_root"]),
            "sampling_root": Path(resolved["sampling_root"]),
            "training_root": Path(resolved["training_root"]),
            "observation_key": key,
            "sampling_seeds": seeds,
            "manifest_path": manifest_path,
            "schedule_path": schedule_path,
            "manifest": manifest,
        })
    return specs


def planned_paths(spec: dict, seed: int, date: str, split: str) -> dict[str, Path]:
    """Output and source paths for one dataset, seed, and day. The key is the spec's key."""
    key = str(spec["observation_key"])
    dataset_id = str(spec["dataset_id"])
    sampling_root = Path(spec["sampling_root"])
    training = Path(spec["training_root"])
    return {
        "members": sampled_vehicle_root(sampling_root, seed) / key / f"{date}.csv",
        "window_all": sampling_window_root(sampling_root, seed) / key / split / date / "all.npz",
        "daily": traffic_observation_feature_path(training, dataset_id, seed, key, date),
        "rf": rf_observation_feature_path(training, dataset_id, seed, key, date),
        "window_all_sp_c": sampling_window_root(sampling_root, seed) / key / split / date / "all_sp_c.npz",
        "scaler": all_sp_c_normalization_path(training, dataset_id),
        "ready": all_sp_c_ready_path(training, dataset_id),
    }


def require_published_window(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise rf.FeatureError(f"missing published window {path}")
    with np.load(path, mmap_mode="r", allow_pickle=False) as payload:
        starts = payload["window_start_slot"]
        if tuple(payload["x"].shape) != (windows.N_WINDOWS, windows.N_HIS, windows.N_NODES, 4):
            raise rf.FeatureError(f"{path} x shape {payload['x'].shape}")
        if payload["x"].dtype != np.float32 or payload["y_raw"].dtype != np.float32:
            raise rf.FeatureError(f"{path} dtype is not float32")
        if not np.array_equal(starts, np.arange(windows.N_WINDOWS, dtype=starts.dtype)):
            raise rf.FeatureError(f"{path} window_start_slot is not 0..276")
        if not np.array_equal(payload["target_slot"], starts + (windows.N_HIS - 1)):
            raise rf.FeatureError(f"{path} target_slot is not the last observed step")


def groups_for(
    training: Path,
    date: str,
    *,
    split: str,
    edges: list[str],
    specs: list[dict],
    node_order_sha256: str,
) -> list[dict]:
    groups = []
    seen_members: dict[str, str] = {}
    for spec in specs:
        dataset_id = spec["dataset_id"]
        key = spec["observation_key"]
        manifest = spec["manifest"]
        if manifest.get("dataset_id") != dataset_id:
            raise rf.FeatureError(f"{dataset_id} manifest id is {manifest.get('dataset_id')!r}")
        if tuple(str(item) for item in manifest.get("observation_keys") or ()) != (key,):
            raise rf.FeatureError(f"{dataset_id} observation keys are not {(key,)}")
        if tuple(int(item) for item in manifest.get("sampling_seeds") or ()) != tuple(spec["sampling_seeds"]):
            raise rf.FeatureError(f"{dataset_id} sampling seeds are not {spec['sampling_seeds']}")
        if Path(spec["training_root"]) != Path(training):
            raise rf.FeatureError(f"{dataset_id} training root does not match this run")
        for seed in spec["sampling_seeds"]:
            paths = planned_paths(spec, int(seed), date, split)
            members = paths["members"]
            if dataset_id not in members.parts or key not in members.parts:
                raise rf.FeatureError(f"vehicle list path lost its dataset or key: {members}")
            previous = seen_members.get(str(members))
            if previous is not None:
                raise rf.FeatureError(f"vehicle list {members} is shared by {previous} and {dataset_id}")
            seen_members[str(members)] = dataset_id
            if not members.is_file():
                raise rf.FeatureError(f"missing sampled vehicles {members}")
            indices, _owners = rf_load_indices(members)
            require_published_window(paths["window_all"])
            groups.append({
                "dataset_id": dataset_id,
                "sampling_seed": int(seed),
                "key": key,
                "date": date,
                "split": split,
                "members": str(members),
                "members_sha256": sha256_file(members),
                "out": str(paths["daily"]),
                "window_all": str(paths["window_all"]),
                "window_all_sp_c": str(paths["window_all_sp_c"]),
                "rf": str(paths["rf"]),
                "indices": indices,
                "n_edges": len(edges),
                "node_order_sha256": node_order_sha256,
            })
    expected = sum(len(spec["sampling_seeds"]) for spec in specs)
    if len(groups) != expected:
        raise rf.FeatureError(f"{date} produced {len(groups)} groups, expected {expected}")
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


def expire_ready(training: Path, dataset_id: str) -> None:
    """Drop only the auxiliary ready marker. The four-state dataset status stays put."""
    path = all_sp_c_ready_path(training, dataset_id)
    try:
        path.unlink()
    except FileNotFoundError:
        return


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
            return {
                "date": date,
                "ok": True,
                "action": "skipped",
                "groups": len(task["groups"]),
                "written": 0,
                "skipped": len(task["groups"]),
            }
        if task.get("publish"):
            for dataset_id in sorted({group["dataset_id"] for group in pending}):
                expire_ready(Path(task["training_root"]), dataset_id)
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
        return {
            "date": date,
            "ok": True,
            "action": "written",
            "groups": len(pending),
            "written": written,
            "skipped": len(task["groups"]) - len(pending),
        }
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
    record: dict | None = None,
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
            ok = bool(
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
                and _text(payload, "observation_key") == group["key"]
                and _text(payload, "date") == group["date"]
                and _text(payload, "vehicle_list_sha256") == group["members_sha256"]
                and _text(payload, "node_order_sha256") == group["node_order_sha256"]
                and _text(payload, "definition_version") == DEFINITION_VERSION
                and _text(payload, "daily_feature_sha256") == sha256_file(daily_path)
                and _text(payload, "source_all_sha256") == sha256_file(all_path)
                and _text(payload, "scaler_sha256") == scaler["sha256"]
                and np.isfinite(x).all()
                and np.isfinite(payload["y"]).all()
                and np.isfinite(x_raw[..., 4:]).all()
                and int(np.asarray(x_raw[..., 5]).min()) >= 0
                and bool(np.all(x_raw[..., 4][x_raw[..., 5] == 0] == 0))
            )
            if ok and record is not None:
                record["y_raw_sha256"] = hashlib.sha256(np.ascontiguousarray(payload["y_raw"]).tobytes()).hexdigest()
            return ok
    except Exception:
        return False


def process_window_day(task: dict) -> dict:
    date = task["date"]
    try:
        written = skipped = 0
        labels = []
        for group in task["groups"]:
            dataset_id = group["dataset_id"]
            scaler = task["scalers"][dataset_id]
            out = Path(group["window_all_sp_c"])
            daily = Path(group["out"])
            all_path = Path(group["window_all"])
            record: dict = {}
            if not task["force"] and window_current(
                out, all_path=all_path, daily_path=daily, scaler=scaler, group=group, record=record,
            ):
                skipped += 1
                labels.append(_label_row(group, record))
                continue
            if task.get("publish"):
                expire_ready(Path(task["training_root"]), dataset_id)
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
                "observation_key": np.asarray(group["key"]),
                "date": np.asarray(date),
                "definition_version": np.asarray(DEFINITION_VERSION),
                "vehicle_list_sha256": np.asarray(group["members_sha256"]),
                "node_order_sha256": np.asarray(group["node_order_sha256"]),
                "daily_feature_sha256": np.asarray(sha256_file(daily)),
                "source_all_sha256": np.asarray(sha256_file(all_path)),
                "scaler_sha256": np.asarray(scaler["sha256"]),
                "target_source": np.asarray("all.npz"),
            })
            refuse_code_root(out)
            atomic_npz(out, **arrays)
            record = {}
            if not window_current(out, all_path=all_path, daily_path=daily, scaler=scaler, group=group, record=record):
                raise rf.FeatureError(f"window failed reread: {out}")
            labels.append(_label_row(group, record))
            written += 1
            del arrays
        rf.log(f"[{date}] all_sp_c windows written={written} skipped={skipped}")
        return {"date": date, "ok": True, "written": written, "skipped": skipped, "labels": labels}
    except Exception:
        return {"date": date, "ok": False, "error": traceback.format_exc(limit=12)}


def _label_row(group: dict, record: dict) -> dict:
    return {
        "dataset_id": group["dataset_id"],
        "sampling_seed": int(group["sampling_seed"]),
        "date": group["date"],
        "observation_key": group["key"],
        "y_raw_sha256": record["y_raw_sha256"],
    }


def _moment_pair(speed: np.ndarray, count: np.ndarray) -> tuple[tuple[float, float, int], tuple[float, float, int]]:
    return windows._moment(speed), windows._moment(count.astype(np.float64))


def _accumulate_extra(training: Path, spec: dict, train_days: list[str]) -> tuple[dict[str, tuple[float, float]], dict[str, list]]:
    dataset_id = spec["dataset_id"]
    key = spec["observation_key"]
    totals = {name: [0.0, 0.0, 0] for name in EXTRA_CHANNELS}
    for day in train_days:
        path = traffic_observation_feature_path(training, dataset_id, 42, key, day)
        with np.load(path, mmap_mode="r", allow_pickle=False) as payload:
            if str(payload["dataset_id"]) != dataset_id or int(payload["sampling_seed"]) != 42:
                raise rf.FeatureError(f"scaler source identity mismatch: {path}")
            if _text(payload, "observation_key") != key or _text(payload, "date") != str(day):
                raise rf.FeatureError(f"scaler source key or date mismatch: {path}")
            if _text(payload, "definition_version") != DEFINITION_VERSION:
                raise rf.FeatureError(f"scaler source definition mismatch: {path}")
            speed_moment, count_moment = _moment_pair(payload["average_speed_raw"], payload["speed_observation_count"])
        for name, moment in zip(EXTRA_CHANNELS, (speed_moment, count_moment)):
            bucket = totals[name]
            bucket[0] += moment[0]
            bucket[1] += moment[1]
            bucket[2] += moment[2]
    expected_cells = len(train_days) * rf.N_BINS * windows.N_NODES
    if any(totals[name][2] != expected_cells for name in EXTRA_CHANNELS):
        raise rf.FeatureError(f"{dataset_id} scaler cell count is not {expected_cells}")
    fitted = {name: windows._mean_std(*totals[name]) for name in EXTRA_CHANNELS}
    return fitted, totals


def fit_scaler(training: Path, spec: dict, train_days: list[str], *, force: bool) -> dict:
    dataset_id = spec["dataset_id"]
    key = spec["observation_key"]
    fitted, totals = _accumulate_extra(training, spec, train_days)
    try:
        commit = __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=CODE, text=True).strip()
    except Exception:
        commit = ""
    payload = {
        "definition_version": DEFINITION_VERSION,
        "dataset_id": dataset_id,
        "observation_key": key,
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
    payload["path"] = str(path)
    return payload


def assert_scaler_matches_days(training: Path, spec: dict, train_days: list[str], scaler: dict) -> None:
    fitted, totals = _accumulate_extra(training, spec, train_days)
    if scaler.get("observation_key") != spec["observation_key"]:
        raise rf.FeatureError(f"{spec['dataset_id']} scaler key {scaler.get('observation_key')}")
    if int(scaler.get("fit_sampling_seed", -1)) != 42 or scaler.get("fit_split") != "train":
        raise rf.FeatureError(f"{spec['dataset_id']} scaler was not fit on seed 42 train")
    if list(scaler.get("training_days") or []) != list(train_days):
        raise rf.FeatureError(f"{spec['dataset_id']} scaler training days do not match the shared train split")
    for name in EXTRA_CHANNELS:
        for field, index in (("mean", 0), ("std", 1)):
            if abs(float(scaler["input"][name][field]) - float(fitted[name][index])) > 1e-12:
                raise rf.FeatureError(f"{spec['dataset_id']} {name} {field} does not match the raw training days")
        if int(scaler["cell_count"][name]) != int(totals[name][2]):
            raise rf.FeatureError(f"{spec['dataset_id']} {name} cell count drifted")


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


def _split_days(manifest: dict) -> tuple[tuple[str, list[str]], ...]:
    return (
        ("train", list(manifest["train"])),
        ("validation", list(manifest["validation"])),
        ("test", list(manifest["test"])),
    )


def protected_stats(training: Path, manifest: dict, specs: list[dict]) -> dict[str, tuple[int, int]]:
    paths = [
        training / "shared" / "target_normalization.json",
        training / "shared" / "physical_definition.json",
        training / "shared" / "split_manifest.json",
        training / "shared" / "state_normalization.json",
        training / "shared" / "node_mapping.csv",
    ]
    for spec in specs:
        root = Path(spec["dataset_root"])
        paths.extend([
            Path(spec["manifest_path"]),
            Path(spec["schedule_path"]),
            root / "shared" / "state_normalization.json",
            root / "shared" / "validation_summary.json",
            root / "shared" / "state_window_manifest.json",
        ])
        for seed in spec["sampling_seeds"]:
            window_root = sampling_window_root(spec["sampling_root"], seed)
            for split, days in _split_days(manifest):
                for day in days:
                    paths.append(sampled_vehicle_root(spec["sampling_root"], seed) / spec["observation_key"] / f"{day}.csv")
                    directory = window_root / spec["observation_key"] / split / day
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


def expected_paths(training: Path, manifest: dict, specs: list[dict]) -> tuple[list[Path], list[Path]]:
    daily = []
    windows_out = []
    for spec in specs:
        for seed in spec["sampling_seeds"]:
            for split, days in _split_days(manifest):
                for day in days:
                    paths = planned_paths(spec, int(seed), day, split)
                    daily.append(paths["daily"])
                    windows_out.append(paths["window_all_sp_c"])
    return daily, windows_out


def verify_cross_seed_labels(rows: list[dict], specs: list[dict], manifest: dict) -> None:
    found: dict[tuple[str, str], dict[int, str]] = {}
    for row in rows:
        for item in row.get("labels") or []:
            bucket = found.setdefault((item["dataset_id"], item["date"]), {})
            seed = int(item["sampling_seed"])
            digest = str(item["y_raw_sha256"])
            if seed in bucket and bucket[seed] != digest:
                raise rf.FeatureError(f"label hash changed for {item['dataset_id']} seed {seed} {item['date']}")
            bucket[seed] = digest
    dates = [day for _name, days in _split_days(manifest) for day in days]
    seed_order = tuple(specs[0]["sampling_seeds"])
    missing = []
    mismatched = []
    for spec in specs:
        if tuple(spec["sampling_seeds"]) != seed_order:
            raise rf.FeatureError(f"{spec['dataset_id']} sampling seed order differs")
        for day in dates:
            bucket = found.get((spec["dataset_id"], day))
            if bucket is None or tuple(sorted(bucket)) != tuple(sorted(seed_order)):
                missing.append(f"{spec['dataset_id']}:{day}")
                continue
            if len(set(bucket.values())) != 1:
                mismatched.append(f"{spec['dataset_id']}:{day}")
    if missing or mismatched:
        raise rf.FeatureError(f"cross-seed labels missing={missing[:4]} mismatched={mismatched[:4]}")


def write_ready(training: Path, manifest: dict, scalers: dict[str, dict], specs: list[dict]) -> None:
    for spec in specs:
        dataset_id = spec["dataset_id"]
        daily = windows_out = 0
        for _seed in spec["sampling_seeds"]:
            for _split, days in _split_days(manifest):
                daily += len(days)
                windows_out += len(days)
        payload = {
            "feature": "all_sp_c",
            "status": "ready",
            "dataset_id": dataset_id,
            "observation_key": spec["observation_key"],
            "definition_version": DEFINITION_VERSION,
            "sampling_seeds": list(spec["sampling_seeds"]),
            "daily_files": daily,
            "window_files": windows_out,
            "normalization": all_sp_c_normalization_path(training, dataset_id).name,
            "normalization_sha256": scalers[dataset_id]["sha256"],
            "fit_sampling_seed": 42,
            "fit_split": "train",
            "note": "This file does not change dataset_status of the four-state package.",
        }
        if daily != len(spec["sampling_seeds"]) * 29 or windows_out != len(spec["sampling_seeds"]) * 29:
            raise rf.FeatureError(f"{dataset_id} ready count drifted: {daily} {windows_out}")
        atomic_json(all_sp_c_ready_path(training, dataset_id), payload)


def _count_actions(rows: list[dict]) -> dict[str, int]:
    written = skipped = 0
    for row in rows:
        written += int(row.get("written") or 0)
        skipped += int(row.get("skipped") or 0)
    return {"written": written, "skipped": skipped, "failed": sum(not row.get("ok") for row in rows)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare all_sp_c speed channels for selected variable datasets")
    parser.add_argument("--training-root", type=Path, default=data_root() / "xuancheng26/training_data_3s")
    parser.add_argument("--fcd-root", type=Path, default=data_root() / "xuancheng26/simulation_data/fcd_data_3s")
    parser.add_argument("--net-file", type=Path, default=CODE / "dataset/xuancheng26/xuancheng_tls_3s.net.xml")
    parser.add_argument("--workers", type=int, default=29)
    parser.add_argument("--dates", nargs="*")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dataset-ids",
        nargs="+",
        default=None,
        help="Dataset ids to prepare. Omit to keep the four p10_p20 datasets. Repeated ids are dropped, order is kept.",
    )
    return parser


def selected_dataset_ids(args: argparse.Namespace) -> list[str]:
    if args.dataset_ids:
        return dedupe_dataset_ids(list(args.dataset_ids))
    return list(DATASET_IDS)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_ids = selected_dataset_ids(args)
    training = refuse_code_root(args.training_root)
    specs = resolve_dataset_specs(training, dataset_ids)
    if args.dataset_ids is None:
        for spec in specs:
            if spec["observation_key"] != OBSERVATION_KEY:
                raise rf.FeatureError(
                    f"default dataset {spec['dataset_id']} observation key is {spec['observation_key']}, "
                    f"expected {OBSERVATION_KEY}"
                )
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
    if len(dates) != 29 or len(manifest["train"]) != 20 or len(manifest["validation"]) != 4 or len(manifest["test"]) != 5:
        raise rf.FeatureError(
            f"expected 20/4/5 days, found {len(manifest['train'])}/"
            f"{len(manifest['validation'])}/{len(manifest['test'])}"
        )
    publish = not (args.dates and set(args.dates) != set(dates))
    before = protected_stats(training, manifest, specs)
    fcd_format = rf.benchmark_fcd(selected[0])
    tasks = []
    for date in selected:
        source = rf.resolve_fcd(date, fcd_format)
        tasks.append({
            "date": date,
            "source_fcd": str(source),
            "force": bool(args.force),
            "publish": publish,
            "training_root": str(training),
            "edges": edges,
            "node_order_sha256": node_sha,
            "groups": groups_for(
                training,
                date,
                split=split_of(manifest, date),
                edges=edges,
                specs=specs,
                node_order_sha256=node_sha,
            ),
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
    described = ", ".join(f"{spec['dataset_id']}:{spec['observation_key']}" for spec in specs)
    rf.log(
        f"all_sp_c datasets={len(specs)} [{described}] days={len(tasks)} "
        f"workers={workers} fcd_format={fcd_format} definition={DEFINITION_VERSION}"
    )
    feature_rows = _run_pool(process_feature_day, tasks, workers, init)
    failed = [row for row in feature_rows if not row.get("ok")]
    if failed:
        raise rf.FeatureError(f"{len(failed)} feature days failed")
    feature_counts = _count_actions(feature_rows)
    if not publish:
        rf.log("partial date list: scaler and windows wait for a full 29-day feature pass")
        assert_unchanged(before)
        print(json.dumps({
            "partial": True,
            "datasets": [spec["dataset_id"] for spec in specs],
            "observation_keys": [spec["observation_key"] for spec in specs],
            "features": feature_counts,
            "definition_version": DEFINITION_VERSION,
        }, indent=2), flush=True)
        return 0
    scalers = {
        spec["dataset_id"]: fit_scaler(training, spec, list(manifest["train"]), force=bool(args.force))
        for spec in specs
    }
    for spec in specs:
        assert_scaler_matches_days(training, spec, list(manifest["train"]), scalers[spec["dataset_id"]])
    for task in tasks:
        task["scalers"] = scalers
        task["training_root"] = str(training)
    window_rows = _run_pool(process_window_day, tasks, workers, init)
    failed = [row for row in window_rows if not row.get("ok")]
    if failed:
        raise rf.FeatureError(f"{len(failed)} window days failed")
    verify_cross_seed_labels(window_rows, specs, manifest)
    daily_paths, window_paths = expected_paths(training, manifest, specs)
    missing = [str(path) for path in daily_paths + window_paths if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise rf.FeatureError(f"missing all_sp_c products: {missing[:8]} count={len(missing)}")
    assert_unchanged(before)
    write_ready(training, manifest, scalers, specs)
    summary = {
        "partial": False,
        "datasets": [
            {"dataset_id": spec["dataset_id"], "observation_key": spec["observation_key"]}
            for spec in specs
        ],
        "daily_files": len(daily_paths),
        "window_files": len(window_paths),
        "features": feature_counts,
        "windows": _count_actions(window_rows),
        "definition_version": DEFINITION_VERSION,
        "node_order_sha256": node_sha,
        "cross_seed_labels": "identical_y_raw",
        "scalers": {
            dataset_id: {
                "sha256": item["sha256"],
                "path": item["path"],
                "observation_key": item["observation_key"],
                "fit_sampling_seed": item["fit_sampling_seed"],
                "fit_split": item["fit_split"],
                "training_days": len(item["training_days"]),
                "source": "sampling seed 42 raw training days, zeros included, no sliding-window reweight",
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
