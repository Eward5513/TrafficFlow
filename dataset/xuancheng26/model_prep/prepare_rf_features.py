#!/usr/bin/env python3
"""Prepare auxiliary RF features for the Xuancheng26 Othman et al. (2022) reproduction.

The script does not train a model and does not rewrite flow, FCD, graph, or
window files. One worker handles one simulation day and reads that day's FCD
once, updating every requested penetration in the same pass.

Speed in ``r_nodes.csv`` and in SUMO FCD is metres per second. Outputs use
kilometres per hour.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from xml.parsers import expat

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

import generate_r_level_full_flow as full

CODE_ROOT = Path("/home/tzhang174/TrafficFlow")
DATA_ROOT = Path("/data5/tzhang174/TrafficFlow/xuancheng26")
TRAIN_ROOT = DATA_ROOT / "training_data"
FCD_ROOT = DATA_ROOT / "simulation_data" / "fcd_data"
PARTIAL_ROOT = TRAIN_ROOT / "partial_flow"
FULL_ROOT = TRAIN_ROOT / "full_flow" / "r-level"
SAMPLED_ROOT = TRAIN_ROOT / "sampled_vehicle"
GRAPH_PATH = TRAIN_ROOT / "graph" / "r_graph.npz"
NODES_PATH = TRAIN_ROOT / "graph" / "r_nodes.csv"
MANIFEST_PATH = TRAIN_ROOT / "window_data" / "split_manifest.json"
NET_FILE = CODE_ROOT / "dataset" / "xuancheng26" / "xuancheng_tls_rebuilt.net.xml"
DEFAULT_OUT = TRAIN_ROOT / "config" / "RF"
N_BINS = full.N_BINS
BIN_SECONDS = full.BIN_SECONDS
DAY_SECONDS = full.DAY_SECONDS
N_EDGES = 1406
READ_BYTES = 8 << 20
PROGRESS_RECORDS = 20_000_000
DEFAULT_RATES = ("p05", "p10", "p20", "p30", "p40", "p50", "p70")
MS_TO_KMH = 3.6
# Above the passenger/freight vType maxSpeed of 27.78 m/s, with margin.
# These samples stay in the mean and are counted, then the mean is capped.
EXTREME_SPEED_MS = 40.0


class StopParse(Exception):
    """Raised to end a streaming parse before the end of the file."""


class FeatureError(RuntimeError):
    """RF feature preparation failed a required check."""


# Inherited by forked workers. Spawn is not used.
LANE_TO_INDEX: dict[str, int] = {}
INTERNAL_LANES: set[str] = set()
BASE_KMH: np.ndarray | None = None
BASE_MS: np.ndarray | None = None
EDGES: list[str] = []


def log(message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"{stamp} {message}", flush=True)


def weekday_index(day_name: str) -> int:
    """Monday=0 ... Sunday=6, from the calendar date encoded in the folder name."""
    year, month, day = (int(part) for part in day_name.split("_"))
    return int(date(year, month, day).weekday())


def rate_percent(name: str) -> int:
    if len(name) < 2 or name[0] != "p" or not name[1:].isdigit():
        raise FeatureError(f"penetration must look like p05, got {name}")
    return int(name[1:])


def parse_vehicle_index(vehicle_id: str) -> int | None:
    """Numeric suffix of ``veh_YYYYMMDD_000000123``. Shared by CSV and FCD ids."""
    underscore = vehicle_id.rfind("_")
    if underscore < 0 or underscore + 1 >= len(vehicle_id):
        return None
    tail = vehicle_id[underscore + 1 :]
    if not tail.isdigit():
        return None
    return int(tail)


def bin_of(event_time: float) -> int | None:
    """Same bin rule as ``generate_r_level_full_flow.bin_index``."""
    return full.bin_index(event_time)


def assert_under_rf_root(path: Path, root: Path) -> None:
    resolved = path.resolve()
    parent = root.resolve()
    if resolved != parent and parent not in resolved.parents:
        raise FeatureError(f"refusing to write outside {parent}: {resolved}")
    code = CODE_ROOT.resolve()
    if resolved == code or code in resolved.parents:
        raise FeatureError(f"refusing to write RF features into the git tree: {resolved}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        import subprocess

        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=CODE_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def load_manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def manifest_days(manifest: dict) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for split in ("train", "validation", "test"):
        values = manifest.get(split)
        if not isinstance(values, list) or not values:
            raise FeatureError(f"split_manifest.json is missing {split}")
        for day_name in values:
            label = str(day_name)
            if label in seen:
                raise FeatureError(f"date {label} appears in more than one split")
            seen.add(label)
            rows.append((label, split))
    if len(rows) != 29:
        raise FeatureError(f"expected 29 manifest dates, found {len(rows)}")
    return rows


def load_canonical_edges() -> tuple[list[str], list[int], list[str]]:
    with NODES_PATH.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if len(rows) != N_EDGES:
        raise FeatureError(f"{NODES_PATH} has {len(rows)} rows, expected {N_EDGES}")
    edges: list[str] = []
    lanes: list[int] = []
    speed_text: list[str] = []
    for position, row in enumerate(rows):
        if int(row["node_index"]) != position:
            raise FeatureError(f"node_index is not canonical at row {position}")
        edge_id = str(row["edge_id"])
        if not edge_id or edge_id.startswith(":"):
            raise FeatureError(f"invalid edge id {edge_id}")
        count = int(row["num_lanes"])
        if count <= 0:
            raise FeatureError(f"{edge_id} has num_lanes={count}")
        edges.append(edge_id)
        lanes.append(count)
        speed_text.append(str(row["speed"]))
    if len(set(edges)) != len(edges):
        raise FeatureError("r_nodes.csv has duplicate edge ids")
    with np.load(GRAPH_PATH, allow_pickle=False) as payload:
        graph_ids = [str(item) for item in payload["edge_ids"].tolist()]
    if graph_ids != edges:
        raise FeatureError("r_graph.npz edge_ids differ from r_nodes.csv order")
    sample = FULL_ROOT / "2023_04_01" / "entry_flow.csv"
    with sample.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split(",")
    if header[:2] != ["time_bin", "start_time"] or header[2:] != edges:
        raise FeatureError("full-flow edge order differs from r_nodes.csv")
    return edges, lanes, speed_text


def load_network_attributes(
    edges: list[str], node_lanes: list[int], node_speed_text: list[str]
) -> tuple[dict[str, int], set[str], np.ndarray, np.ndarray]:
    """Map lanes with the network structure. Do not split edge ids on '_'."""
    index = {edge_id: position for position, edge_id in enumerate(edges)}
    lane_to_index: dict[str, int] = {}
    internal: set[str] = set()
    speed_ms = np.zeros(len(edges), dtype=np.float64)
    lane_count = np.zeros(len(edges), dtype=np.int32)
    seen_speed: dict[str, float] = {}
    current: str | None = None
    current_real = False

    def start(name: str, attrs) -> None:
        nonlocal current, current_real
        if name == "edge":
            edge_id = attrs.get("id", "")
            function = attrs.get("function", "")
            current_real = bool(edge_id) and not edge_id.startswith(":") and function != "internal"
            current = edge_id if current_real else None
            if current_real and edge_id not in index:
                raise FeatureError(f"network has real edge {edge_id} outside the 1406-edge table")
            return
        if name != "lane" or "id" not in attrs:
            return
        if not current_real or current is None:
            internal.add(attrs["id"])
            return
        column = index[current]
        lane_to_index[attrs["id"]] = column
        lane_count[column] += 1
        speed = float(attrs["speed"])
        if speed <= 0 or not math.isfinite(speed):
            raise FeatureError(f"lane {attrs['id']} has speed {speed}")
        previous = seen_speed.get(current)
        if previous is None:
            seen_speed[current] = speed
            speed_ms[column] = speed
        elif abs(previous - speed) > 1e-6:
            raise FeatureError(f"edge {current} has mixed lane speeds {previous} and {speed}")

    parser = expat.ParserCreate()
    parser.StartElementHandler = start
    with NET_FILE.open("rb") as handle:
        parser.ParseFile(handle)
    missing = [edge_id for edge_id in edges if edge_id not in seen_speed]
    if missing:
        raise FeatureError(f"network is missing modeled edges, for example {missing[:5]}")
    for position, edge_id in enumerate(edges):
        if int(lane_count[position]) != int(node_lanes[position]):
            raise FeatureError(
                f"{edge_id} lane count is {int(lane_count[position])} in the net "
                f"and {node_lanes[position]} in r_nodes.csv"
            )
        rounded = f"{speed_ms[position]:.2f}"
        if rounded != node_speed_text[position]:
            raise FeatureError(
                f"{edge_id} speed text {node_speed_text[position]} != net {rounded}"
            )
        if float(node_speed_text[position]) <= 0:
            raise FeatureError(f"{edge_id} speed limit is not positive")
    return lane_to_index, internal, speed_ms, lane_count


def write_road_attributes(
    path: Path,
    edges: list[str],
    lane_count: np.ndarray,
    speed_ms: np.ndarray,
) -> np.ndarray:
    if float(np.nanmax(speed_ms)) > 40.0:
        raise FeatureError(
            "speed_ms exceeds 40 m/s; this looks like km/h and would be converted twice"
        )
    base_kmh = speed_ms * MS_TO_KMH
    if np.any(base_kmh <= 0) or not np.isfinite(base_kmh).all():
        raise FeatureError("speed limit conversion produced a non-positive value")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "edge_index",
                "edge_id",
                "num_lanes",
                "speed_limit",
                "normalized_speed_base",
                "speed_limit_unit",
                "source_speed_ms",
                "source_speed_unit",
            ]
        )
        for position, edge_id in enumerate(edges):
            kmh = float(base_kmh[position])
            writer.writerow(
                [
                    position,
                    edge_id,
                    int(lane_count[position]),
                    f"{kmh:.8f}",
                    f"{kmh:.8f}",
                    "km/h",
                    f"{float(speed_ms[position]):.8f}",
                    "m/s",
                ]
            )
    os.replace(temporary, path)
    return base_kmh


def write_calendar(path: Path, days: list[tuple[str, str]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["date", "weekday", "split", "weekday_encoding"])
        for day_name, split in days:
            writer.writerow([day_name, weekday_index(day_name), split, "monday_0"])
    os.replace(temporary, path)


def feature_schema(days: list[tuple[str, str]], args: argparse.Namespace, fcd_format: str) -> dict:
    return {
        "model_name": "RF",
        "paper": (
            "Othman et al. (2022), A Novel Approach To Traffic Flow Estimation based on "
            "Floating Car Data and Road Topography: Experimental Validation in Lyon, France, "
            "IEEE ITSC 2022"
        ),
        "script": str(Path(__file__).resolve()),
        "git_revision": git_revision(),
        "fcd_format_selected": fcd_format,
        "features": [
            {
                "order": 1,
                "name": "fcd_flow",
                "unit": "vehicles/5min",
                "source": "training_data/partial_flow/pXX/YYYY_MM_DD/inflow.csv",
                "paper_feature": "FCD Flow",
                "stored_in_this_directory": False,
            },
            {
                "order": 2,
                "name": "fcd_average_speed",
                "unit": "km/h",
                "source": "config/RF/pXX/YYYY_MM_DD.npz key average_speed",
                "paper_feature": "FCD Average Speed",
                "stored_in_this_directory": True,
            },
            {
                "order": 3,
                "name": "normalized_speed",
                "unit": "1",
                "source": "config/RF/pXX/YYYY_MM_DD.npz key normalized_speed",
                "paper_feature": "Normalized Speed",
                "stored_in_this_directory": True,
            },
            {
                "order": 4,
                "name": "num_lanes",
                "unit": "lanes",
                "source": "config/RF/road_attributes.csv column num_lanes",
                "paper_feature": "Functional Class",
                "stored_in_this_directory": True,
                "adaptation": "Number of lanes replaces Functional Class. No functional class is invented.",
            },
            {
                "order": 5,
                "name": "speed_limit",
                "unit": "km/h",
                "source": "config/RF/road_attributes.csv column speed_limit",
                "paper_feature": "Capacity",
                "stored_in_this_directory": True,
                "adaptation": "SUMO speed limit replaces Capacity. No capacity value is invented.",
            },
            {
                "order": 6,
                "name": "weekday",
                "unit": "monday_0_to_sunday_6",
                "source": "config/RF/calendar.csv, or datetime.date.weekday() on the simulation date",
                "paper_feature": "Week Day",
                "stored_in_this_directory": True,
                "note": "day_index in window NPZ is this weekday, not a consecutive day number.",
            },
            {
                "order": 7,
                "name": "hour",
                "unit": "hour_0_to_23",
                "source": "time_bin // 12 from the existing 5-minute bins",
                "paper_feature": "Hour",
                "stored_in_this_directory": True,
                "note": "Stored per bin in each NPZ. This is not the 0-287 slot index.",
            },
        ],
        "paper_differences": [
            "Functional Class is unavailable and is replaced by num_lanes.",
            "The paper capacity is unavailable and is replaced by the SUMO speed limit.",
            "HERE base speed is unavailable. normalized_speed_base equals the SUMO speed limit.",
            "Aggregation is 5 minutes, matching Xuancheng26, rather than 1 hour.",
            "Each sample is one directed SUMO edge.",
            "FCD flow remains the existing entry count and is not recomputed here.",
        ],
        "speed_aggregation": {
            "method": "arithmetic mean of 1 Hz FCD speed samples inside the bin",
            "weighting": "each FCD record has equal weight, including repeated samples of a vehicle that remains on the edge",
            "not_used": "one speed per vehicle entry",
            "population": "sampled_vehicle IDs for that date and penetration only",
            "source_unit": "m/s",
            "output_unit": "km/h",
            "conversion": "speed_kmh = speed_ms * 3.6",
            "accumulator": "float64 sum of m/s, converted once when the file is closed",
            "r_nodes_speed_unit": "m/s, confirmed from xuancheng_tls_rebuilt.net.xml lane speed attributes",
        },
        "missing_speed": {
            "rule": "If speed_observation_count is 0, average_speed is the edge speed limit and normalized_speed is 1.",
            "not_inferred_from": "FCD flow. A zero entry count can still have speed samples from vehicles that entered earlier.",
        },
        "cap": "average_speed = min(observed mean, speed_limit), applied to the bin mean",
        "base_speed_proxy": "SUMO edge speed limit in km/h. normalized_speed_base = speed_limit.",
        "edge_ordering": "r_nodes.csv / r_graph.npz edge_ids / full_flow and partial_flow column order",
        "edge_count": N_EDGES,
        "time_window": {
            "seconds": BIN_SECONDS,
            "bins": N_BINS,
            "definition": "generate_r_level_full_flow.bin_index: floor(time/300), with time>=86400 mapped to bin 287",
            "bins_retained": "0 through 287 inclusive",
            "later_model_target_slots": "11 through 287, matching the existing sequence models. This stage does not drop bins 0-10.",
            "hour": "time_bin // 12",
        },
        "split_manifest": str(MANIFEST_PATH),
        "split_counts": {
            split: sum(1 for _day, name in days if name == split)
            for split in ("train", "validation", "test")
        },
        "full_flow_is_label_only": True,
        "do_not_copy_existing_flow_or_splits": True,
        "command": vars(args),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fcd_candidates(day_name: str) -> tuple[Path, Path]:
    day_dir = FCD_ROOT / day_name
    return day_dir / f"fcd_{day_name}.xml", day_dir / f"fcd_{day_name}.xml.gz"


def resolve_fcd(day_name: str, choice: str) -> Path:
    plain, compressed = fcd_candidates(day_name)
    if choice == "xml":
        path = plain
    elif choice == "gz":
        path = compressed
    else:
        raise FeatureError(f"unknown fcd format {choice}")
    if not path.is_file() or path.stat().st_size <= 0:
        raise FeatureError(f"missing FCD file {path}")
    return path


def open_fcd(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rb")
    return path.open("rb")


def benchmark_fcd(day_name: str, seconds: float = 5.0) -> str:
    """Pick the faster complete file. Tie goes to plain XML, which built inflow.csv."""
    plain, compressed = fcd_candidates(day_name)
    scores: dict[str, float] = {}
    for label, path in (("xml", plain), ("gz", compressed)):
        if not path.is_file() or path.stat().st_size <= 0:
            continue
        vehicles = 0
        started = time.perf_counter()

        def start(name: str, _attrs) -> None:
            nonlocal vehicles
            if name == "vehicle":
                vehicles += 1

        parser = expat.ParserCreate()
        parser.StartElementHandler = start
        with open_fcd(path) as handle:
            while time.perf_counter() - started < seconds:
                block = handle.read(READ_BYTES)
                if not block:
                    break
                parser.Parse(block, False)
        elapsed = max(time.perf_counter() - started, 1e-6)
        scores[label] = vehicles / elapsed
        log(f"[benchmark] format={label} vehicles_per_s={scores[label]:.0f} file={path}")
    if not scores:
        raise FeatureError(f"no FCD file for benchmark day {day_name}")
    if scores.get("gz", -1) > scores.get("xml", -1):
        return "gz"
    return "xml"


def load_sampled_indices(day_name: str, rate: str) -> list[int]:
    path = SAMPLED_ROOT / rate / f"{day_name}.csv"
    indices: list[int] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != ["vehicle_id"]:
            raise FeatureError(f"unexpected sampled header in {path}: {header}")
        for row in reader:
            if not row:
                continue
            index = parse_vehicle_index(row[0])
            if index is None:
                raise FeatureError(f"unparseable sampled id {row[0]} in {path}")
            indices.append(index)
    if len(indices) != len(set(indices)):
        raise FeatureError(f"duplicate sampled ids in {path}")
    return indices


def build_mask(rate_indices: list[list[int]]) -> np.ndarray:
    if len(rate_indices) > 8:
        raise FeatureError("at most 8 penetrations can be packed into one mask byte")
    largest = 0
    for indices in rate_indices:
        if indices:
            largest = max(largest, max(indices))
    mask = np.zeros(largest + 1, dtype=np.uint8)
    for bit, indices in enumerate(rate_indices):
        if not indices:
            continue
        mask[np.asarray(indices, dtype=np.int64)] |= np.uint8(1 << bit)
    return mask


def nesting_holds(rate_names: list[str], rate_indices: list[list[int]]) -> bool:
    sets = [set(indices) for indices in rate_indices]
    order = sorted(range(len(rate_names)), key=lambda position: rate_percent(rate_names[position]))
    for left, right in zip(order, order[1:]):
        if not sets[left] <= sets[right]:
            return False
    return True


def finalize_speeds(
    speed_sum_ms: np.ndarray, count: np.ndarray, base_kmh: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return float32 capped speed, normalized speed, and the number of capped bins."""
    if base_kmh.shape != (speed_sum_ms.shape[1],):
        raise FeatureError("base speed length does not match the edge axis")
    if np.any(base_kmh <= 0) or not np.isfinite(base_kmh).all():
        raise FeatureError("base speed is invalid")
    if int(count.min()) < 0:
        raise FeatureError("negative observation count")
    mean_kmh = np.zeros(speed_sum_ms.shape, dtype=np.float64)
    np.divide(speed_sum_ms * MS_TO_KMH, count, out=mean_kmh, where=count > 0)
    if not np.isfinite(mean_kmh).all():
        raise FeatureError("speed mean is not finite")
    capped_windows = int(np.count_nonzero((count > 0) & (mean_kmh > base_kmh)))
    filled = np.where(count > 0, np.minimum(mean_kmh, base_kmh), base_kmh)
    ratio = np.where(count > 0, filled / base_kmh, 1.0)
    average = filled.astype(np.float32)
    normalized = ratio.astype(np.float32)
    base32 = base_kmh.astype(np.float32)
    np.minimum(average, base32, out=average)
    np.clip(normalized, 0.0, 1.0, out=normalized)
    missing = count == 0
    average = np.where(missing, base32, average).astype(np.float32, copy=False)
    normalized = np.where(missing, np.float32(1), normalized).astype(np.float32, copy=False)
    return average, normalized, capped_windows


def parse_fcd_stream(
    blocks,
    *,
    mask: np.ndarray,
    lane_to_index: dict[str, int],
    internal_lanes: set[str],
    base_ms: np.ndarray,
    n_rates: int,
    n_edges: int,
    until_time: float | None,
    progress_label: str,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, object]]:
    sums = [np.zeros((N_BINS, n_edges), dtype=np.float64) for _ in range(n_rates)]
    counts = [np.zeros((N_BINS, n_edges), dtype=np.int64) for _ in range(n_rates)]
    above = [0 for _ in range(n_rates)]
    used = [0 for _ in range(n_rates)]
    stats = {
        "records": 0,
        "unindexed_id": 0,
        "mask_hits": 0,
        "internal_lane": 0,
        "unknown_lane": 0,
        "negative_speed": 0,
        "nonfinite_speed": 0,
        "extreme_speed": 0,
        "bytes_read": 0,
        "unknown_examples": [],
    }
    clock = {"bin": -1}
    mask_len = int(mask.shape[0])
    next_progress = PROGRESS_RECORDS

    def start(name: str, attrs) -> None:
        nonlocal next_progress
        if name == "timestep":
            event_time = float(attrs.get("time", "nan"))
            if until_time is not None and event_time >= until_time:
                raise StopParse()
            slot = bin_of(event_time)
            clock["bin"] = -1 if slot is None else int(slot)
            return
        if name != "vehicle":
            return
        stats["records"] += 1
        if stats["records"] >= next_progress:
            log(
                f"[{progress_label}] records={stats['records']} "
                f"mask_hits={stats['mask_hits']} bytes={stats['bytes_read']}"
            )
            next_progress += PROGRESS_RECORDS
        vehicle_index = parse_vehicle_index(attrs.get("id", ""))
        if vehicle_index is None:
            stats["unindexed_id"] += 1
            return
        if vehicle_index < 0 or vehicle_index >= mask_len:
            return
        membership = int(mask[vehicle_index])
        if membership == 0:
            return
        stats["mask_hits"] += 1
        lane_id = attrs.get("lane", "")
        column = lane_to_index.get(lane_id)
        if column is None:
            if lane_id in internal_lanes:
                stats["internal_lane"] += 1
            else:
                stats["unknown_lane"] += 1
                examples = stats["unknown_examples"]
                if len(examples) < 8 and lane_id not in examples:
                    examples.append(lane_id)
            return
        slot = clock["bin"]
        if slot < 0:
            return
        try:
            speed = float(attrs.get("speed", "nan"))
        except ValueError:
            stats["nonfinite_speed"] += 1
            return
        if not math.isfinite(speed):
            stats["nonfinite_speed"] += 1
            return
        if speed < 0:
            stats["negative_speed"] += 1
            return
        if speed > EXTREME_SPEED_MS:
            stats["extreme_speed"] += 1
        limit = float(base_ms[column])
        bit = 1
        for rate_index in range(n_rates):
            if membership & bit:
                sums[rate_index][slot, column] += speed
                counts[rate_index][slot, column] += 1
                used[rate_index] += 1
                if speed > limit:
                    above[rate_index] += 1
            bit <<= 1

    parser = expat.ParserCreate()
    parser.StartElementHandler = start
    stopped = False
    try:
        for block in blocks:
            stats["bytes_read"] += len(block)
            parser.Parse(block, False)
        parser.Parse(b"", True)
    except StopParse:
        stopped = True
    stats["stopped_early"] = stopped
    stats["used_records"] = used
    stats["above_base_observations"] = above
    return sums, counts, stats


def iter_file_blocks(path: Path):
    with open_fcd(path) as handle:
        while True:
            block = handle.read(READ_BYTES)
            if not block:
                break
            yield block


def load_inflow(path: Path, edges: list[str]) -> np.ndarray:
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split(",")
    if header[2:] != edges:
        raise FeatureError(f"inflow edge order differs from r_nodes.csv: {path}")
    values = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=range(2, 2 + len(edges)),
        dtype=np.int32,
    )
    if values.shape != (N_BINS, len(edges)):
        raise FeatureError(f"{path} shape {values.shape}")
    return values


def validate_arrays(
    payload: np.lib.npyio.NpzFile,
    *,
    edges: list[str],
    base_kmh: np.ndarray,
    day_name: str,
    rate: str,
    require_complete: bool,
) -> None:
    required = (
        "average_speed",
        "normalized_speed",
        "speed_observation_count",
        "edge_ids",
        "time_bins",
        "hour",
        "base_speed_kmh",
    )
    for key in required:
        if key not in payload.files:
            raise FeatureError(f"missing {key}")
    average = payload["average_speed"]
    normalized = payload["normalized_speed"]
    count = payload["speed_observation_count"]
    if average.shape != (N_BINS, N_EDGES) or normalized.shape != average.shape or count.shape != average.shape:
        raise FeatureError(f"shape mismatch {average.shape} {normalized.shape} {count.shape}")
    if average.dtype != np.float32 or normalized.dtype != np.float32:
        raise FeatureError("speed arrays must be float32")
    if count.dtype not in (np.int32, np.int64):
        raise FeatureError(f"count dtype {count.dtype}")
    if [str(item) for item in payload["edge_ids"].tolist()] != edges:
        raise FeatureError("NPZ edge_ids differ from r_nodes.csv")
    if not np.array_equal(payload["time_bins"], np.arange(N_BINS, dtype=payload["time_bins"].dtype)):
        raise FeatureError("time_bins are not 0..287")
    if not np.array_equal(payload["hour"], payload["time_bins"] // 12):
        raise FeatureError("hour is not time_bin // 12")
    if int(payload["weekday"]) != weekday_index(day_name):
        raise FeatureError("weekday does not match the calendar date")
    if str(payload["date"]) != day_name or str(payload["penetration"]) != rate:
        raise FeatureError("NPZ date or penetration metadata mismatch")
    if require_complete and int(payload["complete_day"]) != 1:
        raise FeatureError("NPZ is marked incomplete")
    if not np.isfinite(average).all() or not np.isfinite(normalized).all():
        raise FeatureError("NaN or Inf in speed features")
    if int(count.min()) < 0:
        raise FeatureError("negative observation count")
    base32 = payload["base_speed_kmh"]
    if base32.shape != (N_EDGES,) or not np.allclose(base32, base_kmh.astype(np.float32), atol=1e-5):
        raise FeatureError("NPZ base speed differs from road_attributes.csv")
    if float(np.max(average - base32)) > 1e-5:
        raise FeatureError("average speed exceeds the base speed")
    if float(normalized.min()) < -1e-6 or float(normalized.max()) > 1 + 1e-6:
        raise FeatureError("normalized speed is outside [0, 1]")
    missing = count == 0
    if not np.allclose(average[missing], np.broadcast_to(base32, average.shape)[missing], atol=1e-5):
        raise FeatureError("a missing window was not filled with the base speed")
    if not np.allclose(normalized[missing], 1.0, atol=1e-5):
        raise FeatureError("a missing window does not have normalized speed 1")


def flow_without_speed(count: np.ndarray, inflow: np.ndarray) -> tuple[int, list[tuple[int, int, int]]]:
    bad = np.argwhere((inflow > 0) & (count == 0))
    examples = [(int(slot), int(column), int(inflow[slot, column])) for slot, column in bad[:8]]
    return int(bad.shape[0]), examples


def write_rate_npz(
    path: Path,
    *,
    average: np.ndarray,
    normalized: np.ndarray,
    count: np.ndarray,
    edges: list[str],
    base_kmh: np.ndarray,
    day_name: str,
    rate: str,
    weekday: int,
    source_fcd: Path,
    complete: bool,
    parse_stats: dict[str, object],
    rate_index: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.partial.npz")
    try:
        np.savez(
            temporary,
            average_speed=average,
            normalized_speed=normalized,
            speed_observation_count=count.astype(np.int32, copy=False),
            edge_ids=np.asarray(edges, dtype="U32"),
            time_bins=np.arange(N_BINS, dtype=np.int16),
            hour=(np.arange(N_BINS, dtype=np.int16) // 12),
            base_speed_kmh=base_kmh.astype(np.float32),
            weekday=np.int16(weekday),
            complete_day=np.int8(1 if complete else 0),
            penetration=np.asarray(rate),
            date=np.asarray(day_name),
            source_fcd=np.asarray(str(source_fcd)),
            fcd_record_count=np.int64(parse_stats["records"]),
            internal_lane_count=np.int64(parse_stats["internal_lane"]),
            unknown_lane_count=np.int64(parse_stats["unknown_lane"]),
            negative_speed_count=np.int64(parse_stats["negative_speed"]),
            nonfinite_speed_count=np.int64(parse_stats["nonfinite_speed"]),
            extreme_speed_count=np.int64(parse_stats["extreme_speed"]),
            used_record_count=np.int64(parse_stats["used_records"][rate_index]),
            above_base_observation_count=np.int64(parse_stats["above_base_observations"][rate_index]),
        )
        with np.load(temporary, allow_pickle=False) as payload:
            validate_arrays(
                payload,
                edges=edges,
                base_kmh=base_kmh,
                day_name=day_name,
                rate=rate,
                require_complete=complete,
            )
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def file_is_valid(path: Path, edges: list[str], base_kmh: np.ndarray, day_name: str, rate: str) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            validate_arrays(
                payload,
                edges=edges,
                base_kmh=base_kmh,
                day_name=day_name,
                rate=rate,
                require_complete=True,
            )
    except (OSError, ValueError, FeatureError):
        return False
    return True


def process_day(task: dict) -> dict:
    day_name = str(task["date"])
    rates = list(task["rates"])
    started = time.perf_counter()
    try:
        if BASE_KMH is None or BASE_MS is None:
            raise FeatureError("worker network tables were not initialized")
        indices = [load_sampled_indices(day_name, rate) for rate in rates]
        nested = nesting_holds(rates, indices)
        if not nested:
            raise FeatureError(f"{day_name} sampled IDs are not nested by penetration")
        mask = build_mask(indices)
        source = resolve_fcd(day_name, str(task["fcd_format"]))
        log(
            f"[{day_name}] start rates={','.join(rates)} fcd={source} "
            f"bytes={source.stat().st_size} single_pass=1"
        )
        gc.disable()
        try:
            sums, counts, stats = parse_fcd_stream(
                iter_file_blocks(source),
                mask=mask,
                lane_to_index=LANE_TO_INDEX,
                internal_lanes=INTERNAL_LANES,
                base_ms=BASE_MS,
                n_rates=len(rates),
                n_edges=len(EDGES),
                until_time=task["until_time"],
                progress_label=day_name,
            )
        finally:
            gc.enable()
        if task["until_time"] is None and bool(stats["stopped_early"]):
            raise FeatureError("parser stopped early on a full-day request")
        if int(stats["unknown_lane"]) > 0:
            raise FeatureError(
                f"unknown lanes among sampled vehicles: {stats['unknown_examples']}"
            )
        weekday = weekday_index(day_name)
        complete = task["until_time"] is None
        written: list[str] = []
        rate_stats: dict[str, dict[str, object]] = {}
        finalized: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for rate_index, rate in enumerate(rates):
            if int(counts[rate_index].max()) > np.iinfo(np.int32).max:
                raise FeatureError(f"{rate} observation count does not fit int32")
            average, normalized, capped = finalize_speeds(sums[rate_index], counts[rate_index], BASE_KMH)
            count32 = counts[rate_index].astype(np.int32, copy=False)
            covered = N_BINS if complete else int(float(task["until_time"]) // BIN_SECONDS)
            bad_count = 0
            if covered > 0:
                inflow = load_inflow(PARTIAL_ROOT / rate / day_name / "inflow.csv", EDGES)
                bad_count, examples = flow_without_speed(count32[:covered], inflow[:covered])
                if bad_count:
                    raise FeatureError(
                        f"{day_name} {rate} has {bad_count} covered bins with inflow but no speed; "
                        f"examples={examples}"
                    )
            out_path = Path(task["output_root"]) / rate / f"{day_name}.npz"
            assert_under_rf_root(out_path, Path(task["output_root"]))
            write_rate_npz(
                out_path,
                average=average,
                normalized=normalized,
                count=count32,
                edges=EDGES,
                base_kmh=BASE_KMH,
                day_name=day_name,
                rate=rate,
                weekday=weekday,
                source_fcd=source,
                complete=complete,
                parse_stats=stats,
                rate_index=rate_index,
            )
            written.append(str(out_path))
            finalized.append((average, normalized, count32))
            rate_stats[rate] = {
                "path": str(out_path),
                "capped_windows": capped,
                "used_records": int(stats["used_records"][rate_index]),
                "above_base_observations": int(stats["above_base_observations"][rate_index]),
                "observation_total": int(count32.sum()),
                "missing_windows": int(np.count_nonzero(count32 == 0)),
                "observed_windows": int(np.count_nonzero(count32 > 0)),
                "flow_without_speed": bad_count,
            }
        for left, right in zip(rates, rates[1:]):
            left_index = rates.index(left)
            right_index = rates.index(right)
            if rate_percent(left) < rate_percent(right):
                low, high = left_index, right_index
            else:
                low, high = right_index, left_index
            if np.any(finalized[low][2] > finalized[high][2]):
                for path_text in written:
                    Path(path_text).unlink(missing_ok=True)
                raise FeatureError(f"{day_name} observation counts are not nested")
        elapsed = time.perf_counter() - started
        log(
            f"[{day_name}] PASS seconds={elapsed:.1f} records={stats['records']} "
            f"mask_hits={stats['mask_hits']} internal={stats['internal_lane']} "
            f"negative={stats['negative_speed']} nonfinite={stats['nonfinite_speed']} "
            f"extreme={stats['extreme_speed']}"
        )
        return {
            "date": day_name,
            "status": "PASS",
            "seconds": elapsed,
            "fcd_path": str(source),
            "fcd_bytes": source.stat().st_size,
            "records": int(stats["records"]),
            "mask_hits": int(stats["mask_hits"]),
            "internal_lane": int(stats["internal_lane"]),
            "unknown_lane": int(stats["unknown_lane"]),
            "negative_speed": int(stats["negative_speed"]),
            "nonfinite_speed": int(stats["nonfinite_speed"]),
            "extreme_speed": int(stats["extreme_speed"]),
            "unindexed_id": int(stats["unindexed_id"]),
            "bytes_read": int(stats["bytes_read"]),
            "single_pass": True,
            "rates": rate_stats,
            "error": "",
        }
    except Exception:
        detail = traceback.format_exc(limit=8).strip().replace("\n", " | ")
        log(f"[{day_name}] FAIL {detail}")
        return {
            "date": day_name,
            "status": "FAIL",
            "seconds": time.perf_counter() - started,
            "single_pass": True,
            "rates": {},
            "error": detail,
        }


def require_inputs(days: list[tuple[str, str]], rates: list[str]) -> None:
    for day_name, _split in days:
        plain, compressed = fcd_candidates(day_name)
        if not ((plain.is_file() and plain.stat().st_size > 0) or (compressed.is_file() and compressed.stat().st_size > 0)):
            raise FeatureError(f"missing FCD for {day_name}")
        entry = FULL_ROOT / day_name / "entry_flow.csv"
        if not entry.is_file():
            raise FeatureError(f"missing full flow {entry}")
        for rate in rates:
            sampled = SAMPLED_ROOT / rate / f"{day_name}.csv"
            inflow = PARTIAL_ROOT / rate / day_name / "inflow.csv"
            if not sampled.is_file() or not inflow.is_file():
                raise FeatureError(f"missing {rate} inputs for {day_name}")


def collect_feature_stats(
    days: list[tuple[str, str]], rates: list[str], output_root: Path, edges: list[str]
) -> dict[str, object]:
    report: dict[str, object] = {}
    for rate in rates:
        speed_chunks = []
        observed_counts = []
        total = 0
        missing = 0
        observed = 0
        observation_sum = 0
        flow_zero = 0
        flow_zero_with_speed = 0
        flow_positive = 0
        flow_positive_without_speed = 0
        hour_missing = np.zeros(24, dtype=np.int64)
        hour_total = np.zeros(24, dtype=np.int64)
        for day_name, _split in days:
            path = output_root / rate / f"{day_name}.npz"
            with np.load(path, allow_pickle=False) as payload:
                average = payload["average_speed"]
                count = payload["speed_observation_count"]
            inflow = load_inflow(PARTIAL_ROOT / rate / day_name / "inflow.csv", edges)
            total += int(count.size)
            missing += int(np.count_nonzero(count == 0))
            observed += int(np.count_nonzero(count > 0))
            observation_sum += int(count.sum())
            zero_flow = inflow == 0
            flow_zero += int(np.count_nonzero(zero_flow))
            flow_zero_with_speed += int(np.count_nonzero(zero_flow & (count > 0)))
            positive = inflow > 0
            flow_positive += int(np.count_nonzero(positive))
            flow_positive_without_speed += int(np.count_nonzero(positive & (count == 0)))
            hour = (np.arange(N_BINS) // 12).repeat(N_EDGES)
            flat_missing = (count == 0).reshape(-1)
            for value in range(24):
                selected = hour == value
                hour_total[value] += int(np.count_nonzero(selected))
                hour_missing[value] += int(np.count_nonzero(flat_missing & selected))
            speed_chunks.append(average.reshape(-1))
            positive_counts = count[count > 0]
            if positive_counts.size:
                observed_counts.append(positive_counts.astype(np.int64, copy=False))
        speeds = np.concatenate(speed_chunks)
        counts = np.concatenate(observed_counts) if observed_counts else np.zeros(0, dtype=np.int64)
        percentiles = {
            name: float(np.percentile(speeds, value))
            for name, value in (("p01", 1), ("p05", 5), ("p50", 50), ("p95", 95), ("p99", 99))
        }
        report[rate] = {
            "road_time_windows": total,
            "observed_windows": observed,
            "missing_windows": missing,
            "missing_window_rate": missing / total if total else None,
            "observed_window_rate": observed / total if total else None,
            "average_speed_kmh": {
                "min": float(speeds.min()),
                "max": float(speeds.max()),
                "mean": float(speeds.mean()),
                "std": float(speeds.std()),
                **percentiles,
            },
            "mean_observations_per_observed_window": float(counts.mean()) if counts.size else None,
            "median_observations_per_observed_window": float(np.median(counts)) if counts.size else None,
            "observation_records": observation_sum,
            "flow_zero_windows": flow_zero,
            "flow_zero_and_speed_observed": flow_zero_with_speed,
            "flow_zero_and_speed_observed_rate": (flow_zero_with_speed / total) if total else None,
            "flow_zero_and_speed_observed_share_of_zero_flow": (
                flow_zero_with_speed / flow_zero if flow_zero else None
            ),
            "flow_positive_windows": flow_positive,
            "flow_positive_without_speed": flow_positive_without_speed,
            "hourly_missing_rate": {
                f"{hour:02d}": (float(hour_missing[hour] / hour_total[hour]) if hour_total[hour] else None)
                for hour in range(24)
            },
        }
    return report


def audit_parse_expat(
    path: Path,
    *,
    sampled_by_rate: dict[str, set[int]],
    lane_to_index: dict[str, int],
    until_time: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Independent spot-check accumulator. It does not call ``parse_fcd_stream``."""
    rates = list(sampled_by_rate)
    sums = {rate: np.zeros((N_BINS, N_EDGES), dtype=np.float64) for rate in rates}
    counts = {rate: np.zeros((N_BINS, N_EDGES), dtype=np.int64) for rate in rates}
    clock = {"time": None}

    def start(name: str, attrs) -> None:
        if name == "timestep":
            event_time = float(attrs["time"])
            if event_time >= until_time:
                raise StopParse()
            clock["time"] = event_time
            return
        if name != "vehicle" or clock["time"] is None:
            return
        slot = bin_of(float(clock["time"]))
        vehicle_index = parse_vehicle_index(attrs.get("id", ""))
        column = lane_to_index.get(attrs.get("lane", ""))
        if slot is None or vehicle_index is None or column is None:
            return
        try:
            speed = float(attrs.get("speed", "nan"))
        except ValueError:
            return
        if not math.isfinite(speed) or speed < 0:
            return
        for rate, chosen in sampled_by_rate.items():
            if vehicle_index in chosen:
                sums[rate][slot, column] += speed
                counts[rate][slot, column] += 1

    parser = expat.ParserCreate()
    parser.StartElementHandler = start
    try:
        for block in iter_file_blocks(path):
            parser.Parse(block, False)
        parser.Parse(b"", True)
    except StopParse:
        pass
    return {rate: (sums[rate], counts[rate]) for rate in rates}


def compare_spot(
    output_root: Path,
    day_name: str,
    rates: list[str],
    edges: list[str],
    base_kmh: np.ndarray,
    independent: dict[str, tuple[np.ndarray, np.ndarray]],
    bins: list[int],
) -> None:
    for rate in rates:
        raw_sum, raw_count = independent[rate]
        with np.load(output_root / rate / f"{day_name}.npz", allow_pickle=False) as payload:
            average = payload["average_speed"]
            normalized = payload["normalized_speed"]
            count = payload["speed_observation_count"]
        for slot in bins:
            if not np.array_equal(count[slot], raw_count[slot]):
                mismatch = np.flatnonzero(count[slot] != raw_count[slot])[:5]
                raise FeatureError(f"{day_name} {rate} bin {slot} count mismatch at {mismatch.tolist()}")
            for column in range(len(edges)):
                observed = int(raw_count[slot, column])
                base = float(base_kmh[column])
                if observed == 0:
                    expected_speed = base
                    expected_norm = 1.0
                else:
                    expected_speed = min(float(raw_sum[slot, column] * MS_TO_KMH / observed), base)
                    expected_norm = expected_speed / base
                if abs(float(average[slot, column]) - np.float32(expected_speed)) > 1e-3:
                    raise FeatureError(
                        f"{day_name} {rate} {edges[column]} bin {slot} speed "
                        f"{float(average[slot, column])} != {expected_speed}"
                    )
                if abs(float(normalized[slot, column]) - np.float32(expected_norm)) > 1e-4:
                    raise FeatureError(
                        f"{day_name} {rate} {edges[column]} bin {slot} normalized speed mismatch"
                    )


def run_spot_checks(
    days: list[str],
    rates: list[str],
    bins: list[int],
    output_root: Path,
    fcd_format: str,
    edges: list[str],
    base_kmh: np.ndarray,
) -> None:
    until_time = (max(bins) + 1) * BIN_SECONDS
    for day_name in days:
        sampled = {rate: set(load_sampled_indices(day_name, rate)) for rate in rates}
        source = resolve_fcd(day_name, fcd_format)
        with np.load(output_root / rates[0] / f"{day_name}.npz", allow_pickle=False) as payload:
            stored = Path(str(payload["source_fcd"]))
        if stored.is_file():
            source = stored
        log(f"[spot] {day_name} bins={bins} until={until_time} fcd={source}")
        independent = audit_parse_expat(
            source,
            sampled_by_rate=sampled,
            lane_to_index=LANE_TO_INDEX,
            until_time=float(until_time),
        )
        compare_spot(output_root, day_name, rates, edges, base_kmh, independent, bins)
        with np.load(output_root / rates[-1] / f"{day_name}.npz", allow_pickle=False) as payload:
            count = np.array(payload["speed_observation_count"], copy=True)
            average = np.array(payload["average_speed"], copy=True)
        observed_columns = np.flatnonzero(count[bins[0]] > 0)
        missing_columns = np.flatnonzero(count[bins[0]] == 0)
        if observed_columns.size:
            column = int(observed_columns[0])
            log(
                f"[spot] {day_name} {rates[-1]} observed example edge={edges[column]} "
                f"bin={bins[0]} count={int(count[bins[0], column])} "
                f"speed_kmh={float(average[bins[0], column]):.4f}"
            )
        if missing_columns.size:
            column = int(missing_columns[0])
            log(
                f"[spot] {day_name} {rates[-1]} missing example edge={edges[column]} "
                f"bin={bins[0]} speed_kmh={float(average[bins[0], column]):.4f}"
            )
        underscore = [
            position
            for position, edge_id in enumerate(edges)
            if "_" in edge_id and int(count[bins[0], position]) > 0
        ]
        log(f"[spot] {day_name} bin {bins[0]} underscore edges with speed samples={len(underscore)}")
        log(f"[spot] {day_name} PASS")


def self_test() -> None:
    if bin_of(0) != 0 or bin_of(299.9) != 0 or bin_of(300) != 1 or bin_of(86399.9) != 287:
        raise FeatureError("bin boundaries drifted")
    if bin_of(86400) != full.bin_index(86400) or bin_of(-1) != full.bin_index(-1):
        raise FeatureError("bin_of is not generate_r_level_full_flow.bin_index")
    if weekday_index("2023_04_01") != 5 or weekday_index("2023_04_03") != 0:
        raise FeatureError("weekday_index does not match April 2023")
    if (np.arange(288) // 12)[0] != 0 or (np.arange(288) // 12)[96] != 8 or (np.arange(288) // 12)[11] != 0:
        raise FeatureError("hour mapping drifted")
    edges = ["A_1", "B"]
    lane_to_index = {"A_1_0": 0, "B_0": 1}
    # Splitting A_1_0 on the first underscore would not produce edge A_1.
    if "A_1_0".split("_", 1)[0] == "A_1":
        raise FeatureError("the underscore test no longer demonstrates the mapping hazard")
    base_ms = np.array([16.67, 11.11], dtype=np.float64)
    base_kmh = base_ms * MS_TO_KMH
    mask = np.zeros(6, dtype=np.uint8)
    mask[1] = np.uint8(1 | 2)  # p05 and p10
    mask[2] = np.uint8(2)  # p10 only
    mask[4] = np.uint8(1)  # p05, but it stays on an internal lane
    xml = b"""<?xml version="1.0"?>
<fcd-export>
<timestep time="10.00">
  <vehicle id="veh_20230401_000000001" speed="10.00" lane="A_1_0"/>
  <vehicle id="veh_20230401_000000001" speed="12.00" lane="A_1_0"/>
  <vehicle id="veh_20230401_000000001" speed="-1" lane="A_1_0"/>
  <vehicle id="veh_20230401_000000002" speed="5.00" lane="B_0"/>
  <vehicle id="veh_20230401_000000003" speed="8.00" lane="A_1_0"/>
  <vehicle id="veh_20230401_000000004" speed="9" lane=":j_0"/>
  <vehicle id="veh_20230401_000000001" speed="3" lane="NO_SUCH"/>
</timestep>
<timestep time="300.00">
  <vehicle id="veh_20230401_000000001" speed="30.00" lane="A_1_0"/>
</timestep>
<timestep time="86400.00">
  <vehicle id="veh_20230401_000000002" speed="4.00" lane="B_0"/>
</timestep>
</fcd-export>
"""
    sums, counts, stats = parse_fcd_stream(
        [xml],
        mask=mask,
        lane_to_index=lane_to_index,
        internal_lanes={":j_0"},
        base_ms=base_ms,
        n_rates=2,
        n_edges=2,
        until_time=None,
        progress_label="self-test",
    )
    if stats["records"] != 9 or stats["negative_speed"] != 1 or stats["internal_lane"] != 1:
        raise FeatureError(f"self-test parse counts drifted: {stats}")
    if stats["unknown_lane"] != 1 or int(counts[0][0, 0]) != 2 or int(counts[1][0, 1]) != 1:
        raise FeatureError(f"self-test aggregation drifted: {counts[0][0].tolist()} {counts[1][0].tolist()}")
    if int(counts[0][0, 1]) != 0 or int(counts[0][287, 1]) != 0 or int(counts[1][287, 1]) != 1:
        raise FeatureError("self-test bin assignment drifted")
    if abs(float(sums[0][0, 0]) - 22.0) > 1e-9:
        raise FeatureError("self-test did not keep both samples of the underscored edge")
    average, normalized, capped = finalize_speeds(sums[0], counts[0], base_kmh)
    if capped < 1:
        raise FeatureError("self-test did not cap the 30 m/s sample")
    if abs(float(average[0, 0]) - np.float32(11.0 * MS_TO_KMH)) > 1e-4:
        raise FeatureError(f"self-test mean {float(average[0, 0])}")
    if int(counts[0][0, 1]) == 0 and abs(float(normalized[0, 1]) - 1) > 1e-6:
        raise FeatureError("missing normalized speed is not 1")
    if abs(float(average[1, 0]) - np.float32(base_kmh[0])) > 1e-4:
        raise FeatureError("capped speed is not the base speed")
    if parse_vehicle_index("veh_20230401_000000023") != 23:
        raise FeatureError("vehicle index parser drifted")
    log("[self-test] PASS")


def parse_rate_list(values: list[str] | None) -> list[str]:
    if not values:
        return list(DEFAULT_RATES)
    rates: list[str] = []
    for value in values:
        for part in value.split(","):
            name = part.strip()
            if name:
                rate_percent(name)
                rates.append(name)
    if len(rates) != len(set(rates)):
        raise FeatureError(f"duplicate penetrations {rates}")
    return rates


def parse_date_list(values: list[str] | None, available: list[tuple[str, str]]) -> list[tuple[str, str]]:
    if not values:
        return available
    wanted: list[str] = []
    for value in values:
        for part in value.split(","):
            name = part.strip()
            if name:
                wanted.append(name)
    lookup = {day_name: split for day_name, split in available}
    missing = [day_name for day_name in wanted if day_name not in lookup]
    if missing:
        raise FeatureError(f"dates are not in the manifest: {missing}")
    return [(day_name, lookup[day_name]) for day_name in wanted]


def parse_bin_list(text: str) -> list[int]:
    bins = [int(part) for part in text.split(",") if part.strip()]
    if not bins or any(slot < 0 or slot >= N_BINS for slot in bins):
        raise FeatureError(f"spot bins must be inside 0..287, got {text}")
    return bins


def directory_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def main(argv: list[str] | None = None) -> int:
    class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description="Prepare Xuancheng26 RF auxiliary speed and road features.",
        formatter_class=HelpFormatter,
        epilog=(
            "examples:\n"
            "  python prepare_rf_features.py\n"
            "  python prepare_rf_features.py --dates 2023_04_01\n"
            "  python prepare_rf_features.py --penetrations p05\n"
            "  python prepare_rf_features.py --validate-only\n"
            "  python prepare_rf_features.py --force\n"
        ),
    )
    parser.add_argument("--workers", type=int, default=29)
    parser.add_argument("--penetrations", nargs="*", default=None)
    parser.add_argument("--dates", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--spot-check", action="store_true")
    parser.add_argument("--spot-bins", default="0,1")
    parser.add_argument("--fcd-format", choices=("auto", "xml", "gz"), default="auto")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--until-time", type=float, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    self_test()
    if args.self_test:
        return 0

    output_root = args.output_root
    assert_under_rf_root(output_root, DEFAULT_OUT)
    if args.until_time is not None and output_root.resolve() == DEFAULT_OUT.resolve():
        raise FeatureError("--until-time cannot write incomplete days into the production RF directory")

    manifest = load_manifest()
    all_days = manifest_days(manifest)
    days = parse_date_list(args.dates, all_days)
    rates = parse_rate_list(args.penetrations)
    edges, node_lanes, speed_text = load_canonical_edges()
    lane_to_index, internal_lanes, speed_ms, lane_count = load_network_attributes(
        edges, node_lanes, speed_text
    )
    global LANE_TO_INDEX, INTERNAL_LANES, BASE_KMH, BASE_MS, EDGES
    LANE_TO_INDEX = lane_to_index
    INTERNAL_LANES = internal_lanes
    BASE_MS = speed_ms
    BASE_KMH = speed_ms * MS_TO_KMH
    EDGES = edges
    log(
        f"edges={len(edges)} lanes={len(lane_to_index)} internal_lanes={len(internal_lanes)} "
        f"speed_ms=[{float(speed_ms.min()):.2f},{float(speed_ms.max()):.2f}] "
        f"speed_kmh=[{float(BASE_KMH.min()):.3f},{float(BASE_KMH.max()):.3f}]"
    )
    require_inputs(days, rates)
    if args.validate_only and not args.spot_check:
        fcd_format = "xml" if args.fcd_format == "auto" else args.fcd_format
    elif args.fcd_format == "auto":
        fcd_format = benchmark_fcd(days[0][0])
    else:
        fcd_format = args.fcd_format
    log(f"fcd_format={fcd_format}")

    if not args.validate_only and not args.spot_check:
        output_root.mkdir(parents=True, exist_ok=True)
        write_road_attributes(output_root / "road_attributes.csv", edges, lane_count, speed_ms)
        write_calendar(output_root / "calendar.csv", all_days if output_root.resolve() == DEFAULT_OUT.resolve() else days)
        schema = feature_schema(all_days, args, fcd_format)
        schema["command"] = {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        }
        write_json(output_root / "feature_schema.json", schema)

    if args.spot_check:
        run_spot_checks(
            [day_name for day_name, _split in days],
            rates,
            parse_bin_list(args.spot_bins),
            output_root,
            fcd_format,
            edges,
            BASE_KMH,
        )
        return 0

    if args.validate_only:
        problems = []
        for day_name, _split in days:
            for rate in rates:
                path = output_root / rate / f"{day_name}.npz"
                if not file_is_valid(path, edges, BASE_KMH, day_name, rate):
                    problems.append(str(path))
                    log(f"[validate] FAIL {path}")
                else:
                    log(f"[validate] PASS {path}")
        if len(rates) >= 2:
            ordered = sorted(rates, key=rate_percent)
            for day_name, _split in days:
                arrays = []
                for rate in ordered:
                    with np.load(output_root / rate / f"{day_name}.npz", allow_pickle=False) as payload:
                        arrays.append(payload["speed_observation_count"])
                for left, right in zip(arrays, arrays[1:]):
                    if np.any(left > right):
                        problems.append(f"{day_name} nesting")
        stats = collect_feature_stats(days, rates, output_root, edges) if not problems else {}
        log(f"[validate] problems={len(problems)}")
        if stats:
            for rate, payload in stats.items():
                log(
                    f"[stats] {rate} missing_rate={payload['missing_window_rate']:.6f} "
                    f"flow0_with_speed_rate={payload['flow_zero_and_speed_observed_rate']:.6f}"
                )
        return 1 if problems else 0

    pending: list[str] = []
    skipped: list[str] = []
    for day_name, _split in days:
        needed = []
        for rate in rates:
            path = output_root / rate / f"{day_name}.npz"
            if args.force or not file_is_valid(path, edges, BASE_KMH, day_name, rate):
                needed.append(rate)
        if needed:
            pending.append(day_name)
        else:
            skipped.append(day_name)
            log(f"[{day_name}] SKIP existing valid outputs")
    # A day with any missing rate still parses FCD once and writes every requested rate
    # that is missing. Rates already valid are left untouched unless --force.
    tasks = []
    for day_name, _split in days:
        if day_name not in pending:
            continue
        needed = [
            rate
            for rate in rates
            if args.force or not file_is_valid(output_root / rate / f"{day_name}.npz", edges, BASE_KMH, day_name, rate)
        ]
        tasks.append(
            {
                "date": day_name,
                "rates": needed,
                "fcd_format": fcd_format,
                "until_time": args.until_time,
                "output_root": str(output_root),
            }
        )
    workers = min(args.workers, len(tasks)) if tasks else 0
    log(f"pending={len(tasks)} skipped={len(skipped)} workers={workers} requested_workers={args.workers}")
    results: list[dict] = []
    run_started = time.perf_counter()
    if tasks:
        import multiprocessing as mp

        context = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = {pool.submit(process_day, task): task["date"] for task in tasks}
            for future in as_completed(futures):
                day_name = futures[future]
                try:
                    results.append(future.result())
                except Exception:
                    detail = traceback.format_exc(limit=8).strip().replace("\n", " | ")
                    log(f"[{day_name}] FAIL {detail}")
                    results.append({"date": day_name, "status": "FAIL", "error": detail, "rates": {}})
    failures = [row for row in results if row.get("status") != "PASS"]
    passed = [row for row in results if row.get("status") == "PASS"]
    elapsed = time.perf_counter() - run_started
    summary = {
        "model_name": "RF",
        "started_epoch": run_started,
        "elapsed_seconds": elapsed,
        "workers_requested": args.workers,
        "workers_used": workers,
        "fcd_format": fcd_format,
        "single_fcd_pass_per_day": True,
        "dates_requested": [day_name for day_name, _split in days],
        "passed": [row["date"] for row in passed],
        "failed": [row.get("date") for row in failures],
        "skipped": skipped,
        "passed_count": len(passed),
        "failed_count": len(failures),
        "skipped_count": len(skipped),
        "day_results": results,
        "output_root": str(output_root),
        "output_bytes": directory_bytes(output_root),
    }
    if not failures and all((output_root / rate / f"{day_name}.npz").is_file() for day_name, _split in days for rate in rates):
        if args.until_time is None:
            summary["feature_stats"] = collect_feature_stats(days, rates, output_root, edges)
            for rate, payload in summary["feature_stats"].items():
                log(
                    f"[stats] {rate} windows={payload['road_time_windows']} "
                    f"missing_rate={payload['missing_window_rate']:.6f} "
                    f"flow0_with_speed={payload['flow_zero_and_speed_observed']} "
                    f"mean_obs={payload['mean_observations_per_observed_window']}"
                )
    write_json(output_root / "preprocessing_summary.json", summary)
    log(
        f"done passed={len(passed)} failed={len(failures)} skipped={len(skipped)} "
        f"elapsed_seconds={elapsed:.1f}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
