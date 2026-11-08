"""Load Xuancheng26 road-time samples for the RF baseline.

Each sample is one directed edge and one 5-minute bin. Bins 0-10 are kept
out of the supervised matrices so the evaluated slots match the existing
sequence models: target bins 11 through 287.

A single state keeps the original seven columns, with that state's partial
count in column 0. ``all`` input stacks the four partial counts first, then
the same six context columns. The label is the matching full-count CSV, or
four full counts when the target is ``all``.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from data_root import (xuancheng26_training_3s_root, sampling_partial_flow_root,
                       xuancheng26_sampling_root, SAMPLING_SEEDS)
TRAIN_ROOT = Path(os.environ.get("XUANCHENG_RF_TRAINING_ROOT", str(xuancheng26_training_3s_root()))).expanduser().resolve()
RF_ROOT = TRAIN_ROOT / "config" / "RF"
FULL_ROOT = TRAIN_ROOT / "full_flow" / "r-level"
NODES_PATH = TRAIN_ROOT / "graph" / "r_nodes.csv"
GRAPH_PATH = TRAIN_ROOT / "graph" / "r_graph.npz"
MANIFEST_PATH = TRAIN_ROOT / "shared" / "split_manifest.json"
RATES = ("p05", "p10", "p20", "p30", "p40", "p50", "p70")
N_EDGES = 1406
N_BINS = 288
TARGET_START = 11
N_TARGETS = N_BINS - TARGET_START
STATES = ("inflow", "outflow", "accumulation_start", "accumulation_end")
CONTEXT_FEATURES = (
    "fcd_average_speed",
    "normalized_speed",
    "num_lanes",
    "speed_limit",
    "weekday",
    "hour",
)
FEATURE_NAMES = ("fcd_flow",) + CONTEXT_FEATURES
SINGLE_INPUT_WIDTH = len(FEATURE_NAMES)
ALL_INPUT_WIDTH = len(STATES) + len(CONTEXT_FEATURES)


class DataError(RuntimeError):
    """Xuancheng26 RF inputs are missing or inconsistent."""


def observed_feature_name(state: str) -> str:
    if state == "inflow":
        return "fcd_flow"
    if state not in STATES:
        raise DataError(f"unknown state {state}")
    return f"fcd_{state}"


def validate_task(input_type: str, target_type: str) -> None:
    """Accept one state predicting itself, or ``all`` predicting one state or all four."""
    if input_type not in STATES and input_type != "all":
        raise DataError(f"unknown input_type {input_type}")
    if target_type not in STATES and target_type != "all":
        raise DataError(f"unknown target_type {target_type}")
    if input_type != "all" and input_type != target_type:
        raise DataError(f"single-state input {input_type} must predict itself, not {target_type}")
    if target_type == "all" and input_type != "all":
        raise DataError("target all requires input all")


def feature_names_for(input_type: str) -> tuple[str, ...]:
    validate_task(input_type, input_type if input_type != "all" else "inflow")
    if input_type == "all":
        return tuple(observed_feature_name(state) for state in STATES) + CONTEXT_FEATURES
    return (observed_feature_name(input_type),) + CONTEXT_FEATURES


def flow_column_for(input_type: str, target_type: str) -> int:
    """Column of the partial count used by the univariate line."""
    validate_task(input_type, target_type)
    if target_type == "all":
        raise DataError("all output has one flow column per state")
    if input_type != "all":
        return 0
    return STATES.index(target_type)


def task_dir_name(input_type: str, target_type: str) -> str:
    validate_task(input_type, target_type)
    return f"in_{input_type}__out_{target_type}"


def _header(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return handle.readline().rstrip("\n").split(",")


def load_edges() -> list[str]:
    with NODES_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    edges = [str(row["edge_id"]) for row in rows]
    if len(edges) != N_EDGES or len(set(edges)) != N_EDGES:
        raise DataError(f"{NODES_PATH} does not contain {N_EDGES} unique edges")
    with np.load(GRAPH_PATH, allow_pickle=False) as payload:
        graph_ids = [str(item) for item in payload["edge_ids"].tolist()]
    if graph_ids != edges:
        raise DataError("r_graph.npz edge order differs from r_nodes.csv")
    return edges


def load_static(edges: list[str]) -> tuple[np.ndarray, np.ndarray]:
    with (RF_ROOT / "road_attributes.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [row["edge_id"] for row in rows] != edges:
        raise DataError("road_attributes.csv edge order differs from r_nodes.csv")
    lanes = np.asarray([int(row["num_lanes"]) for row in rows], dtype=np.float32)
    limit = np.asarray([float(row["speed_limit"]) for row in rows], dtype=np.float32)
    base = np.asarray([float(row["normalized_speed_base"]) for row in rows], dtype=np.float32)
    if np.any(lanes <= 0) or np.any(limit <= 0) or not np.allclose(limit, base):
        raise DataError("road attributes have an invalid lane count or speed limit")
    if float(limit.max()) > 120.0:
        raise DataError(
            f"speed_limit max {float(limit.max()):.3f} km/h is outside the SUMO 20-80 km/h range"
        )
    return lanes, limit


def load_manifest() -> dict[str, list[str]]:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    days = {}
    seen: set[str] = set()
    for split in ("train", "validation", "test"):
        values = [str(item) for item in payload[split]]
        if len(values) != len(set(values)):
            raise DataError(f"duplicate dates in {split}")
        overlap = seen.intersection(values)
        if overlap:
            raise DataError(f"dates overlap splits: {sorted(overlap)[:5]}")
        seen.update(values)
        days[split] = values
    if [len(days[split]) for split in ("train", "validation", "test")] != [20, 4, 5]:
        raise DataError(f"unexpected split sizes {[len(days[s]) for s in days]}")
    return days


def _load_flow(path: Path, edges: list[str]) -> np.ndarray:
    header = _header(path)
    if header[:2] != ["time_bin", "start_time"] or header[2:] != edges:
        raise DataError(f"flow edge order differs from r_nodes.csv: {path}")
    values = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=range(2, 2 + len(edges)),
        dtype=np.float32,
    )
    if values.shape != (N_BINS, N_EDGES):
        raise DataError(f"{path} shape {values.shape}")
    if not np.isfinite(values).all() or float(values.min()) < 0:
        raise DataError(f"{path} has a non-finite or negative count")
    return values


def _load_speed(path: Path, edges: list[str], penetration: str, day_name: str, limit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        speed = np.asarray(payload["average_speed"], dtype=np.float32)
        norm = np.asarray(payload["normalized_speed"], dtype=np.float32)
        count = payload["speed_observation_count"]
        stored_edges = [str(item) for item in payload["edge_ids"].tolist()]
        if stored_edges != edges:
            raise DataError(f"{path} edge order mismatch")
        if speed.shape != (N_BINS, N_EDGES) or norm.shape != speed.shape or count.shape != speed.shape:
            raise DataError(f"{path} shape {speed.shape}")
        if str(payload["date"]) != day_name or str(payload["penetration"]) != penetration:
            raise DataError(f"{path} date/penetration metadata mismatch")
        if int(payload["complete_day"]) != 1:
            raise DataError(f"{path} is not a complete day")
        if not np.array_equal(payload["time_bins"], np.arange(N_BINS)):
            raise DataError(f"{path} time bins are not 0..287")
        if not np.array_equal(payload["hour"], np.arange(N_BINS) // 12):
            raise DataError(f"{path} hour is not time_bin // 12")
        if not np.allclose(payload["base_speed_kmh"], limit, atol=1e-4):
            raise DataError(f"{path} base speed differs from road_attributes.csv")
    if not np.isfinite(speed).all() or not np.isfinite(norm).all():
        raise DataError(f"{path} speed contains NaN or Inf")
    if int(np.min(count)) < 0 or float(norm.min()) < -1e-5 or float(norm.max()) > 1 + 1e-5:
        raise DataError(f"{path} has a negative count or normalized speed outside [0, 1]")
    if float(np.max(speed - limit)) > 1e-3:
        raise DataError(f"{path} average speed exceeds the speed limit")
    return speed, norm


class XuanchengRFData:
    def __init__(self, sampling_seed: int = 42) -> None:
        if sampling_seed not in SAMPLING_SEEDS:
            raise DataError(f"unknown sampling seed {sampling_seed}")
        self.sampling_seed = sampling_seed
        self.partial_root = sampling_partial_flow_root(TRAIN_ROOT / "sampling_seeds", sampling_seed)
        self.edges = load_edges()
        self.lanes, self.speed_limit = load_static(self.edges)
        self.days = load_manifest()
        self.weekday = {day: _weekday(day) for day in self.all_days()}

    def all_days(self) -> list[str]:
        return [day for split in ("train", "validation", "test") for day in self.days[split]]

    def audit(self, input_type: str = "inflow", target_type: str = "inflow") -> dict[str, int]:
        validate_task(input_type, target_type)
        checked = 0
        for rate in RATES:
            for day_name in self.all_days():
                self.day_cube(rate, day_name, input_type, target_type)
                checked += 1
        return {
            "rate_days_checked": checked,
            "edges": len(self.edges),
            "input_type": input_type,
            "target_type": target_type,
            "feature_width": len(feature_names_for(input_type)),
        }

    def day_cube(
        self,
        penetration: str,
        day_name: str,
        input_type: str = "inflow",
        target_type: str = "inflow",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return supervised bins 11..287.

        Single input is ``X[277, 1406, 7]`` and one ``y[277, 1406]``. ``all``
        input is ``X[277, 1406, 10]``. ``all`` target adds a last axis of four
        full counts in ``STATES`` order.
        """
        validate_task(input_type, target_type)
        names = feature_names_for(input_type)
        speed_path = RF_ROOT / penetration / f"{day_name}.npz"
        if not speed_path.is_file():
            raise DataError(f"missing {speed_path}")
        observed_states = STATES if input_type == "all" else (input_type,)
        label_states = STATES if target_type == "all" else (target_type,)
        observed = []
        for state in observed_states:
            path = self.partial_root / penetration / day_name / f"{state}.csv"
            if not path.is_file():
                raise DataError(f"missing {path}")
            observed.append(_load_flow(path, self.edges))
        labels = []
        for state in label_states:
            path = FULL_ROOT / day_name / f"{state}.csv"
            if not path.is_file():
                raise DataError(f"missing {path}")
            labels.append(_load_flow(path, self.edges))
        speed, norm = _load_speed(speed_path, self.edges, penetration, day_name, self.speed_limit)
        sl = slice(TARGET_START, N_BINS)
        features = np.empty((N_TARGETS, N_EDGES, len(names)), dtype=np.float32)
        for index, flow in enumerate(observed):
            features[:, :, index] = flow[sl]
        context = len(observed)
        features[:, :, context] = speed[sl]
        features[:, :, context + 1] = norm[sl]
        features[:, :, context + 2] = self.lanes
        features[:, :, context + 3] = self.speed_limit
        features[:, :, context + 4] = np.float32(self.weekday[day_name])
        hours = (np.arange(TARGET_START, N_BINS, dtype=np.int32) // 12).astype(np.float32)
        features[:, :, context + 5] = hours[:, None]
        if target_type == "all":
            target = np.stack([label[sl] for label in labels], axis=-1).astype(np.float32, copy=False)
        else:
            target = labels[0][sl]
        if not np.isfinite(features).all() or not np.isfinite(target).all():
            raise DataError(f"{penetration} {day_name} produced a non-finite sample")
        return features, target

    def stack(
        self,
        penetration: str,
        days: list[str],
        input_type: str = "inflow",
        target_type: str = "inflow",
    ) -> tuple[np.ndarray, np.ndarray]:
        if target_type == "all":
            raise DataError("stack one target state; all-to-all fits each state separately")
        validate_task(input_type, target_type)
        width = len(feature_names_for(input_type))
        features = []
        targets = []
        for day_name in days:
            cube, target = self.day_cube(penetration, day_name, input_type, target_type)
            features.append(cube.reshape(-1, width))
            targets.append(np.asarray(target).reshape(-1))
        return np.concatenate(features, axis=0), np.concatenate(targets, axis=0)


def _weekday(day_name: str) -> int:
    year, month, day = (int(part) for part in day_name.split("_"))
    import datetime

    return int(datetime.date(year, month, day).weekday())
