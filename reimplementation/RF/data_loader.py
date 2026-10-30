"""Load Xuancheng26 road-time samples for the RF baseline.

Each sample is one directed edge and one 5-minute bin. Bins 0-10 are kept
out of the supervised matrices so the evaluated slots match the existing
sequence models: target bins 11 through 287.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

DATA_ROOT = Path("/data5/tzhang174/TrafficFlow/xuancheng26")
TRAIN_ROOT = Path(
    os.environ.get("XUANCHENG_RF_TRAINING_ROOT", str(DATA_ROOT / "training_data"))
).expanduser().resolve()
RF_ROOT = TRAIN_ROOT / "config" / "RF"
PARTIAL_ROOT = TRAIN_ROOT / "partial_flow"
FULL_ROOT = TRAIN_ROOT / "full_flow" / "r-level"
NODES_PATH = TRAIN_ROOT / "graph" / "r_nodes.csv"
GRAPH_PATH = TRAIN_ROOT / "graph" / "r_graph.npz"
MANIFEST_PATH = TRAIN_ROOT / "window_data" / "split_manifest.json"
RATES = ("p05", "p10", "p20", "p30", "p40", "p50", "p70")
N_EDGES = 1406
N_BINS = 288
TARGET_START = 11
N_TARGETS = N_BINS - TARGET_START
FEATURE_NAMES = (
    "fcd_flow",
    "fcd_average_speed",
    "normalized_speed",
    "num_lanes",
    "speed_limit",
    "weekday",
    "hour",
)


class DataError(RuntimeError):
    """Xuancheng26 RF inputs are missing or inconsistent."""


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
    def __init__(self) -> None:
        self.edges = load_edges()
        self.lanes, self.speed_limit = load_static(self.edges)
        self.days = load_manifest()
        self.weekday = {day: _weekday(day) for day in self.all_days()}

    def all_days(self) -> list[str]:
        return [day for split in ("train", "validation", "test") for day in self.days[split]]

    def audit(self) -> dict[str, int]:
        checked = 0
        for rate in RATES:
            for day_name in self.all_days():
                self.day_cube(rate, day_name)
                checked += 1
        return {"rate_days_checked": checked, "edges": len(self.edges)}

    def day_cube(self, penetration: str, day_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return X[277, 1406, 7] and y[277, 1406] for supervised bins 11..287."""
        inflow_path = PARTIAL_ROOT / penetration / day_name / "inflow.csv"
        label_path = FULL_ROOT / day_name / "inflow.csv"
        speed_path = RF_ROOT / penetration / f"{day_name}.npz"
        for path in (inflow_path, label_path, speed_path):
            if not path.is_file():
                raise DataError(f"missing {path}")
        flow = _load_flow(inflow_path, self.edges)
        label = _load_flow(label_path, self.edges)
        speed, norm = _load_speed(speed_path, self.edges, penetration, day_name, self.speed_limit)
        sl = slice(TARGET_START, N_BINS)
        features = np.empty((N_TARGETS, N_EDGES, len(FEATURE_NAMES)), dtype=np.float32)
        features[:, :, 0] = flow[sl]
        features[:, :, 1] = speed[sl]
        features[:, :, 2] = norm[sl]
        features[:, :, 3] = self.lanes
        features[:, :, 4] = self.speed_limit
        features[:, :, 5] = np.float32(self.weekday[day_name])
        hours = (np.arange(TARGET_START, N_BINS, dtype=np.int32) // 12).astype(np.float32)
        features[:, :, 6] = hours[:, None]
        target = label[sl]
        if not np.isfinite(features).all() or not np.isfinite(target).all():
            raise DataError(f"{penetration} {day_name} produced a non-finite sample")
        return features, target

    def stack(self, penetration: str, days: list[str]) -> tuple[np.ndarray, np.ndarray]:
        features = []
        targets = []
        for day_name in days:
            cube, target = self.day_cube(penetration, day_name)
            features.append(cube.reshape(-1, len(FEATURE_NAMES)))
            targets.append(target.reshape(-1))
        return np.concatenate(features, axis=0), np.concatenate(targets, axis=0)


def _weekday(day_name: str) -> int:
    year, month, day = (int(part) for part in day_name.split("_"))
    import datetime

    return int(datetime.date(year, month, day).weekday())
