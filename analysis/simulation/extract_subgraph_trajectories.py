"""
从 SUMO FCD 中按天截取途经子图道路的车辆轨迹。

每天独立解压、筛选截断并写出。多日时先完成全部提取，再统一做验证。

FCD 是 gzip CSV。edge_id 由 lane 去掉末尾 `_数字` 得到，再与子图 ID
做精确匹配。以 `:` 开头的内部 edge 保留在轨迹里，但不算子图道路。

用法：

    python3 analysis/simulation/extract_subgraph_trajectories.py
    python3 analysis/simulation/extract_subgraph_trajectories.py --jobs 4
    python3 analysis/simulation/extract_subgraph_trajectories.py --day 1 --jobs 1
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = BASE_DIR.parent
DEFAULT_NET_FILE = ANALYSIS_DIR / "road_network" / "net_tls.net.xml"
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_SUBGRAPH_FILE = DEFAULT_DATA_DIR / "subgraph.txt"
DEFAULT_FCD_DIR = DEFAULT_DATA_DIR / "simulation"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "processed" / "subgraph_trajectories"
DEFAULT_TEMP_DIR = DEFAULT_OUTPUT_DIR / "_tmp"

DEFAULT_JOBS = 4
SAMPLE_COUNT = 3
SAMPLE_SEED = 42
PROGRESS_EVERY = 2_000_000
COPY_BUFFER = 16 * 1024 * 1024
UNPARSEABLE_EXAMPLES = 8
MISSING_EDGE_EXAMPLES = 8
ALL_DAYS_SUMMARY_NAME = "validation_summary_all_days.csv"

FCD_GZ_NAME = "fcd.csv.gz"
TEMP_CSV_NAME = "fcd.csv"
TRAJECTORIES_NAME = "trajectories.csv"
VEHICLE_IDS_NAME = "vehicle_ids.txt"
DAY_SUMMARY_NAME = "validation_summary.json"
STEM_PATTERN = re.compile(r"^(\d+)_(\d+)$")
EXTRA_FIELDS = ["timestamp", "vehicle_id", "edge_id", "is_in_subgraph", "day"]


@dataclass(slots=True)
class DaySource:
    index: int
    seed: int
    folder: Path
    gz_path: Path

    @property
    def stem(self) -> str:
        return f"{self.index}_{self.seed}"

    @property
    def day_key(self) -> str:
        return f"day_{self.index:02d}"


@dataclass(slots=True)
class VehicleWindow:
    start: float
    end: float
    currently_in: bool = True
    reentered: bool = False
    subgraph_hits: int = 1


@dataclass(slots=True)
class SampleTrack:
    vehicle_id: str
    start: float
    end: float
    first_time: str | None = None
    last_time: str | None = None
    first_edge: str | None = None
    last_edge: str | None = None
    first_in_subgraph: bool | None = None
    last_in_subgraph: bool | None = None
    first_speed: str | None = None
    first_pos: str | None = None
    first_lane: str | None = None
    last_speed: str | None = None
    last_pos: str | None = None
    last_lane: str | None = None
    n_rows: int = 0
    outside_window: int = 0


@dataclass
class ExtractStats:
    fcd_rows: int = 0
    output_rows: int = 0
    rows_in_window: int = 0
    parsed_edge_ids: set[str] = field(default_factory=set)
    seen_subgraph_edges: set[str] = field(default_factory=set)
    unique_times: set[str] = field(default_factory=set)
    time_deltas: set[float] = field(default_factory=set)
    unparseable_count: int = 0
    unparseable_examples: list[str] = field(default_factory=list)
    missing_net_edges: set[str] = field(default_factory=set)
    output_min_time: float | None = None
    output_max_time: float | None = None
    output_min_time_raw: str | None = None
    output_max_time_raw: str | None = None


class ExtractError(Exception):
    pass


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def edge_from_lane(lane_id: str | None) -> str | None:
    """SUMO lane id is `<edge_id>_<lane_index>`; keep signed / # / . suffixes."""
    if not lane_id or "_" not in lane_id:
        return None
    edge_id, lane_index = lane_id.rsplit("_", 1)
    if not lane_index.isdigit() or not edge_id:
        return None
    return edge_id


def load_subgraph_ids(path: Path) -> tuple[list[str], set[str]]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    unique: list[str] = []
    seen: set[str] = set()
    for edge_id in lines:
        if edge_id not in seen:
            unique.append(edge_id)
            seen.add(edge_id)
    return lines, seen


def load_net_edge_ids(net_file: Path) -> set[str]:
    ids: set[str] = set()
    for _, elem in ET.iterparse(net_file, events=("end",)):
        if local_name(elem.tag) == "edge":
            edge_id = elem.get("id")
            if edge_id:
                ids.add(edge_id)
        elem.clear()
    return ids


def parse_stem(name: str) -> tuple[int, int] | None:
    match = STEM_PATTERN.match(name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def discover_days(fcd_dir: Path) -> list[DaySource]:
    if not fcd_dir.is_dir():
        raise ExtractError(f"FCD directory not found: {fcd_dir}")
    by_index: dict[int, DaySource] = {}
    for folder in sorted(fcd_dir.iterdir()):
        parsed = parse_stem(folder.name) if folder.is_dir() else None
        if parsed is None:
            continue
        index, seed = parsed
        gz_path = folder / FCD_GZ_NAME
        if not gz_path.is_file():
            continue
        if index in by_index:
            raise ExtractError(
                f"day {index} maps to more than one folder in {fcd_dir}: "
                f"{by_index[index].folder.name} and {folder.name}"
            )
        by_index[index] = DaySource(index=index, seed=seed, folder=folder, gz_path=gz_path)
    if not by_index:
        raise ExtractError(f"no day folders with {FCD_GZ_NAME} under {fcd_dir}")
    return [by_index[index] for index in sorted(by_index)]


def day_from_fcd_path(gz_path: Path) -> DaySource:
    gz_path = gz_path.expanduser().resolve()
    if not gz_path.is_file():
        raise ExtractError(f"FCD gzip not found: {gz_path}")
    folder = gz_path.parent
    parsed = parse_stem(folder.name)
    if parsed is None:
        raise ExtractError(
            f"cannot parse day index from folder name {folder.name!r}; "
            "expected <index>_<seed>"
        )
    index, seed = parsed
    return DaySource(index=index, seed=seed, folder=folder, gz_path=gz_path)


def field_index(header: list[str], name: str) -> int:
    try:
        return header.index(name)
    except ValueError as exc:
        raise ExtractError(f"FCD header missing required field {name!r}: {header}") from exc


def record_unparseable(stats: ExtractStats, lane: str) -> None:
    stats.unparseable_count += 1
    if lane not in stats.unparseable_examples and len(stats.unparseable_examples) < UNPARSEABLE_EXAMPLES:
        stats.unparseable_examples.append(lane)


def decompress_fcd(gz_path: Path, csv_path: Path) -> None:
    print(f"Decompressing {gz_path} -> {csv_path}", flush=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    copied = 0
    next_mark = 1024 * 1024 * 1024
    with gzip.open(gz_path, "rb") as src, csv_path.open("wb") as dst:
        while True:
            chunk = src.read(COPY_BUFFER)
            if not chunk:
                break
            dst.write(chunk)
            copied += len(chunk)
            if copied >= next_mark:
                print(f"  decompressed {copied / (1024 ** 3):.2f} GiB", flush=True)
                next_mark += 1024 * 1024 * 1024
    print(f"  finished decompress ({copied / (1024 ** 3):.2f} GiB)", flush=True)


def cleanup_temp_csv(csv_path: Path) -> bool:
    removed = False
    if csv_path.is_file():
        csv_path.unlink()
        removed = True
        print(f"Deleted temporary CSV: {csv_path}", flush=True)
    parent = csv_path.parent
    try:
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        pass
    return removed


def scan_windows(
    csv_path: Path,
    subgraph_ids: set[str],
    net_edge_ids: set[str],
    stats: ExtractStats,
) -> dict[str, VehicleWindow]:
    print(f"Pass 1: scanning {csv_path}", flush=True)
    windows: dict[str, VehicleWindow] = {}
    prev_time: float | None = None
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        time_idx = field_index(header, "time")
        id_idx = field_index(header, "id")
        lane_idx = field_index(header, "lane")
        for row in reader:
            stats.fcd_rows += 1
            if stats.fcd_rows % PROGRESS_EVERY == 0:
                print(
                    f"  pass 1 rows={stats.fcd_rows:,} vehicles_hit={len(windows):,}",
                    flush=True,
                )
            time_raw = row[time_idx]
            vehicle_id = row[id_idx]
            lane = row[lane_idx] if lane_idx < len(row) else ""
            timestamp = float(time_raw)
            stats.unique_times.add(time_raw)
            if prev_time is not None:
                delta = timestamp - prev_time
                if delta > 0:
                    stats.time_deltas.add(round(delta, 6))
            prev_time = timestamp

            edge_id = edge_from_lane(lane)
            if edge_id is None:
                record_unparseable(stats, lane)
                in_subgraph = False
            else:
                stats.parsed_edge_ids.add(edge_id)
                if edge_id not in net_edge_ids:
                    stats.missing_net_edges.add(edge_id)
                in_subgraph = edge_id in subgraph_ids
                if in_subgraph:
                    stats.seen_subgraph_edges.add(edge_id)

            window = windows.get(vehicle_id)
            if in_subgraph:
                if window is None:
                    windows[vehicle_id] = VehicleWindow(start=timestamp, end=timestamp)
                else:
                    window.end = timestamp
                    window.subgraph_hits += 1
                    if not window.currently_in:
                        window.reentered = True
                    window.currently_in = True
            elif window is not None:
                window.currently_in = False
    print(
        f"  pass 1 done: rows={stats.fcd_rows:,} vehicles={len(windows):,}",
        flush=True,
    )
    return windows


def write_truncated(
    csv_path: Path,
    output_path: Path,
    day_index: int,
    windows: dict[str, VehicleWindow],
    subgraph_ids: set[str],
    net_edge_ids: set[str],
    samples: dict[str, SampleTrack],
    stats: ExtractStats,
) -> None:
    print(f"Pass 2: writing {output_path}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scanned = 0
    with csv_path.open("r", encoding="utf-8", newline="") as src, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst, lineterminator="\n")
        header = next(reader)
        time_idx = field_index(header, "time")
        id_idx = field_index(header, "id")
        speed_idx = field_index(header, "speed")
        pos_idx = field_index(header, "pos")
        lane_idx = field_index(header, "lane")
        writer.writerow([*header, *EXTRA_FIELDS])
        for row in reader:
            scanned += 1
            if scanned % PROGRESS_EVERY == 0:
                print(
                    f"  pass 2 rows={scanned:,} written={stats.output_rows:,}",
                    flush=True,
                )
            vehicle_id = row[id_idx]
            window = windows.get(vehicle_id)
            if window is None:
                continue
            time_raw = row[time_idx]
            timestamp = float(time_raw)
            if timestamp < window.start or timestamp > window.end:
                continue
            stats.rows_in_window += 1
            lane = row[lane_idx] if lane_idx < len(row) else ""
            edge_id = edge_from_lane(lane)
            if edge_id is None:
                edge_id = ""
                in_subgraph = False
            else:
                in_subgraph = edge_id in subgraph_ids
                if edge_id not in net_edge_ids:
                    stats.missing_net_edges.add(edge_id)
            writer.writerow(
                [
                    *row,
                    time_raw,
                    vehicle_id,
                    edge_id,
                    "true" if in_subgraph else "false",
                    str(day_index),
                ]
            )
            stats.output_rows += 1
            if stats.output_min_time is None or timestamp < stats.output_min_time:
                stats.output_min_time = timestamp
                stats.output_min_time_raw = time_raw
            if stats.output_max_time is None or timestamp > stats.output_max_time:
                stats.output_max_time = timestamp
                stats.output_max_time_raw = time_raw

            sample = samples.get(vehicle_id)
            if sample is None:
                continue
            sample.n_rows += 1
            if timestamp < sample.start or timestamp > sample.end:
                sample.outside_window += 1
            speed = row[speed_idx]
            pos = row[pos_idx]
            if sample.first_time is None:
                sample.first_time = time_raw
                sample.first_edge = edge_id
                sample.first_in_subgraph = in_subgraph
                sample.first_speed = speed
                sample.first_pos = pos
                sample.first_lane = lane
            sample.last_time = time_raw
            sample.last_edge = edge_id
            sample.last_in_subgraph = in_subgraph
            sample.last_speed = speed
            sample.last_pos = pos
            sample.last_lane = lane
    print(f"  pass 2 done: written={stats.output_rows:,}", flush=True)


def serialize_windows(windows: dict[str, VehicleWindow]) -> list[dict[str, object]]:
    return [
        {
            "vehicle_id": vid,
            "start": window.start,
            "end": window.end,
            "reentered": window.reentered,
        }
        for vid, window in windows.items()
    ]


def deserialize_windows(payload: object) -> dict[str, VehicleWindow]:
    windows: dict[str, VehicleWindow] = {}
    if not isinstance(payload, list):
        return windows
    for item in payload:
        if not isinstance(item, dict):
            continue
        vid = str(item.get("vehicle_id", ""))
        if not vid:
            continue
        windows[vid] = VehicleWindow(
            start=float(item["start"]),
            end=float(item["end"]),
            reentered=bool(item.get("reentered", False)),
        )
    return windows


def sample_tracks_payload(samples: dict[str, SampleTrack]) -> dict[str, object]:
    return {
        vid: {
            "window": [sample.start, sample.end],
            "rows": sample.n_rows,
            "outside_window": sample.outside_window,
            "first": {
                "time": sample.first_time,
                "edge_id": sample.first_edge,
                "in_subgraph": sample.first_in_subgraph,
                "speed": sample.first_speed,
                "pos": sample.first_pos,
                "lane": sample.first_lane,
            },
            "last": {
                "time": sample.last_time,
                "edge_id": sample.last_edge,
                "in_subgraph": sample.last_in_subgraph,
                "speed": sample.last_speed,
                "pos": sample.last_pos,
                "lane": sample.last_lane,
            },
        }
        for vid, sample in samples.items()
    }


def samples_from_payload(payload: object) -> dict[str, SampleTrack]:
    samples: dict[str, SampleTrack] = {}
    if not isinstance(payload, dict):
        return samples
    for vid, raw in payload.items():
        if not isinstance(raw, dict):
            continue
        window = raw.get("window") or [0.0, 0.0]
        first = raw.get("first") or {}
        last = raw.get("last") or {}
        sample = SampleTrack(vehicle_id=str(vid), start=float(window[0]), end=float(window[1]))
        sample.n_rows = int(raw.get("rows") or 0)
        sample.outside_window = int(raw.get("outside_window") or 0)
        sample.first_time = first.get("time")
        sample.first_edge = first.get("edge_id")
        sample.first_in_subgraph = first.get("in_subgraph")
        sample.first_speed = first.get("speed")
        sample.first_pos = first.get("pos")
        sample.first_lane = first.get("lane")
        sample.last_time = last.get("time")
        sample.last_edge = last.get("edge_id")
        sample.last_in_subgraph = last.get("in_subgraph")
        sample.last_speed = last.get("speed")
        sample.last_pos = last.get("pos")
        sample.last_lane = last.get("lane")
        samples[str(vid)] = sample
    return samples


def stats_from_summary(summary: dict[str, object]) -> ExtractStats:
    stats = ExtractStats()
    stats.output_rows = int(summary.get("output_row_count") or 0)
    stats.rows_in_window = int(summary.get("rows_in_window") or stats.output_rows)
    stats.fcd_rows = int(summary.get("fcd_row_count") or 0)
    stats.time_deltas = {
        float(item) for item in (summary.get("positive_time_deltas_s") or [])
    }
    return stats


def choose_sample_ids(windows: dict[str, VehicleWindow], count: int) -> list[str]:
    ids = list(windows)
    if len(ids) <= count:
        return ids
    rng = random.Random(SAMPLE_SEED)
    chosen = rng.sample(ids, count)
    reentered = [vid for vid, window in windows.items() if window.reentered]
    if reentered and not any(windows[vid].reentered for vid in chosen):
        chosen[-1] = reentered[0]
    return chosen


def write_vehicle_ids(path: Path, vehicle_ids: list[str]) -> None:
    path.write_text("".join(f"{vid}\n" for vid in vehicle_ids), encoding="utf-8")


def read_vehicle_ids(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        raise ExtractError(f"{path} is missing a trailing newline")
    return [line for line in text.splitlines() if line]


def validate_day_outputs(
    *,
    day: DaySource,
    trajectories_path: Path,
    vehicle_ids_path: Path,
    windows: dict[str, VehicleWindow],
    subgraph_ids: set[str],
    stats: ExtractStats,
    samples: dict[str, SampleTrack],
) -> tuple[list[str], dict[str, object]]:
    failures: list[str] = []
    if not trajectories_path.is_file():
        return ["trajectories.csv is missing"], {}
    if not vehicle_ids_path.is_file():
        failures.append("vehicle_ids.txt is missing")

    listed_ids = read_vehicle_ids(vehicle_ids_path) if vehicle_ids_path.is_file() else []
    if len(listed_ids) != len(set(listed_ids)):
        failures.append("vehicle_ids.txt contains duplicate ids")
    if listed_ids != list(windows):
        failures.append("vehicle_ids.txt order/set does not match first-entry vehicle order")
    if len(listed_ids) != len(windows):
        failures.append(
            f"vehicle_ids.txt count {len(listed_ids)} != candidate count {len(windows)}"
        )

    csv_ids: set[str] = set()
    first_row: dict[str, dict[str, str]] = {}
    last_row: dict[str, dict[str, str]] = {}
    counts: dict[str, int] = {}
    prev_time: float | None = None
    time_non_decreasing = True
    subgraph_flag_ok = True
    day_column_ok = True
    identity_ok = True
    row_count = 0
    expected_day = str(day.index)

    with trajectories_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return ["trajectories.csv has no header"], {}
        required = ["time", "id", "speed", "pos", "lane", "x", "y", *EXTRA_FIELDS]
        missing_cols = [name for name in required if name not in reader.fieldnames]
        if missing_cols:
            failures.append(f"trajectories.csv missing columns {missing_cols}")
        for row in reader:
            row_count += 1
            vid = row["vehicle_id"]
            csv_ids.add(vid)
            counts[vid] = counts.get(vid, 0) + 1
            timestamp = float(row["timestamp"])
            if prev_time is not None and timestamp < prev_time:
                time_non_decreasing = False
            prev_time = timestamp
            edge_id = row["edge_id"]
            flag = row["is_in_subgraph"]
            expected_flag = "true" if edge_id in subgraph_ids else "false"
            if flag != expected_flag:
                subgraph_flag_ok = False
            if row.get("day") != expected_day:
                day_column_ok = False
            if row.get("timestamp") != row.get("time") or row.get("vehicle_id") != row.get("id"):
                identity_ok = False
            if vid not in first_row:
                first_row[vid] = row
            last_row[vid] = row

    if row_count != stats.output_rows:
        failures.append(
            f"trajectories.csv rows {row_count} != written count {stats.output_rows}"
        )
    if stats.rows_in_window != stats.output_rows:
        failures.append(
            f"in-window FCD rows {stats.rows_in_window} != output rows {stats.output_rows}"
        )
    if csv_ids != set(windows):
        failures.append("unique vehicle_id in CSV does not match candidate set")
    if csv_ids != set(listed_ids):
        failures.append("unique vehicle_id in CSV does not match vehicle_ids.txt")
    if not time_non_decreasing:
        failures.append("output timestamps are not non-decreasing")
    if not subgraph_flag_ok:
        failures.append("is_in_subgraph does not match edge_id membership")
    if not day_column_ok:
        failures.append(f"output mixed with a day column other than {expected_day}")
    if not identity_ok:
        failures.append("timestamp/vehicle_id extra fields do not match original time/id")

    first_last_ok = 0
    first_last_bad: list[str] = []
    outside_window = 0
    for vid, window in windows.items():
        first = first_row.get(vid)
        last = last_row.get(vid)
        if first is None or last is None:
            first_last_bad.append(vid)
            continue
        if first["is_in_subgraph"] != "true" or last["is_in_subgraph"] != "true":
            first_last_bad.append(vid)
            continue
        if float(first["timestamp"]) != window.start or float(last["timestamp"]) != window.end:
            first_last_bad.append(vid)
            continue
        if float(first["timestamp"]) < window.start or float(last["timestamp"]) > window.end:
            outside_window += 1
            continue
        first_last_ok += 1
    if first_last_bad:
        failures.append(
            f"{len(first_last_bad)} vehicles fail first/last subgraph or window checks; "
            f"examples={first_last_bad[:8]}"
        )
    if outside_window:
        failures.append(f"{outside_window} vehicles have rows outside [start, end]")

    min_delta = min(stats.time_deltas) if stats.time_deltas else None
    timestep_ok = min_delta == 1.0 and set(stats.time_deltas) == {1.0}
    if not timestep_ok:
        failures.append(f"FCD timestep is not 1s; deltas={sorted(stats.time_deltas)[:12]}")

    for vid, sample in samples.items():
        if sample.outside_window:
            failures.append(f"sample {vid} has {sample.outside_window} rows outside window")
        if sample.first_in_subgraph is not True or sample.last_in_subgraph is not True:
            failures.append(f"sample {vid} first/last not on subgraph")

    details = {
        "csv_row_count": row_count,
        "csv_unique_vehicles": len(csv_ids),
        "vehicle_ids_file_count": len(listed_ids),
        "first_last_ok_vehicles": first_last_ok,
        "time_non_decreasing": time_non_decreasing,
        "subgraph_flag_ok": subgraph_flag_ok,
        "day_column_ok": day_column_ok,
        "identity_ok": identity_ok,
        "timestep_ok": timestep_ok,
        "min_positive_time_delta_s": min_delta,
        "positive_time_deltas_s": sorted(stats.time_deltas)[:12],
        "sample_vehicles": {
            vid: {
                "window": [sample.start, sample.end],
                "rows": sample.n_rows,
                "outside_window": sample.outside_window,
                "first": {
                    "time": sample.first_time,
                    "edge_id": sample.first_edge,
                    "in_subgraph": sample.first_in_subgraph,
                    "speed": sample.first_speed,
                    "pos": sample.first_pos,
                    "lane": sample.first_lane,
                },
                "last": {
                    "time": sample.last_time,
                    "edge_id": sample.last_edge,
                    "in_subgraph": sample.last_in_subgraph,
                    "speed": sample.last_speed,
                    "pos": sample.last_pos,
                    "lane": sample.last_lane,
                },
            }
            for vid, sample in samples.items()
        },
    }
    return failures, details


def existing_success(summary_path: Path) -> bool:
    if not summary_path.is_file():
        return False
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "ok" and payload.get("validation_passed") is True


def extraction_complete(summary_path: Path) -> bool:
    if not summary_path.is_file():
        return False
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    status = payload.get("status")
    return status in {"extracted", "ok"} and bool(payload.get("output_row_count") is not None)


def validate_extracted_day(
    day: DaySource,
    *,
    subgraph_ids: set[str],
    output_root: Path,
) -> dict[str, object]:
    out_dir = output_root / day.day_key
    summary_path = out_dir / DAY_SUMMARY_NAME
    trajectories_path = out_dir / TRAJECTORIES_NAME
    vehicle_ids_path = out_dir / VEHICLE_IDS_NAME
    print(f"\n== validating {day.day_key} ==", flush=True)
    if not summary_path.is_file():
        payload = write_failed_summary(
            day,
            output_root=output_root,
            error="validation_summary.json missing after extraction",
            gz_size_before=day.gz_path.stat().st_size if day.gz_path.is_file() else 0,
            elapsed=0.0,
        )
        return payload
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") not in {"extracted", "ok"}:
        summary["validation_passed"] = False
        summary["validation_pending"] = False
        summary["validation_failures"] = summary.get("validation_failures") or [
            str(summary.get("error") or f"extraction status={summary.get('status')}")
        ]
        write_json(summary_path, summary)
        print(f"  skip validate: extraction status={summary.get('status')}", flush=True)
        return summary

    windows = deserialize_windows(summary.get("candidate_windows"))
    extra_failures: list[str] = []
    if not windows:
        extra_failures.append("candidate_windows missing from extract summary")
    stats = stats_from_summary(summary)
    samples = samples_from_payload(summary.get("sample_vehicles"))
    try:
        failures, details = validate_day_outputs(
            day=day,
            trajectories_path=trajectories_path,
            vehicle_ids_path=vehicle_ids_path,
            windows=windows,
            subgraph_ids=subgraph_ids,
            stats=stats,
            samples=samples,
        )
    except ExtractError as exc:
        failures, details = [str(exc)], {}
    failures = [*extra_failures, *failures]
    gz_before = summary.get("gz_size_before")
    if day.gz_path.is_file() and gz_before is not None and day.gz_path.stat().st_size != gz_before:
        failures.append(
            f"original gzip size changed: {gz_before} -> {day.gz_path.stat().st_size}"
        )
    summary.update(details)
    summary["status"] = "ok" if not failures else "failed"
    summary["validation_passed"] = not failures
    summary["validation_pending"] = False
    summary["validation_failures"] = failures
    summary["trajectories_bytes"] = (
        trajectories_path.stat().st_size if trajectories_path.is_file() else 0
    )
    summary["gz_unchanged"] = (
        day.gz_path.is_file()
        and gz_before is not None
        and day.gz_path.stat().st_size == gz_before
    )
    write_json(summary_path, summary)
    print(
        f"  validation={'passed' if not failures else 'failed'} "
        f"vehicles={summary.get('candidate_vehicle_count')} "
        f"rows={summary.get('output_row_count')} failures={len(failures)}",
        flush=True,
    )
    if failures:
        for item in failures:
            print(f"  validation: {item}", flush=True)
    return summary


def process_one_day(
    day: DaySource,
    *,
    subgraph_lines: list[str],
    subgraph_ids: set[str],
    net_edge_ids: set[str],
    output_root: Path,
    temp_root: Path,
    net_file: Path,
    subgraph_file: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    out_dir = output_root / day.day_key
    trajectories_path = out_dir / TRAJECTORIES_NAME
    vehicle_ids_path = out_dir / VEHICLE_IDS_NAME
    summary_path = out_dir / DAY_SUMMARY_NAME
    temp_csv = temp_root / day.day_key / TEMP_CSV_NAME
    gz_size_before = day.gz_path.stat().st_size
    stats = ExtractStats()
    temp_removed = False
    samples: dict[str, SampleTrack] = {}
    windows: dict[str, VehicleWindow] = {}

    print(f"\n== {day.day_key} ({day.stem}) ==", flush=True)
    print(f"  fcd gz: {day.gz_path} ({gz_size_before} bytes)", flush=True)

    try:
        decompress_fcd(day.gz_path, temp_csv)
        if not day.gz_path.is_file():
            raise ExtractError(f"original gzip missing after decompress: {day.gz_path}")
        windows = scan_windows(temp_csv, subgraph_ids, net_edge_ids, stats)
        sample_ids = choose_sample_ids(windows, SAMPLE_COUNT)
        samples = {
            vid: SampleTrack(vehicle_id=vid, start=windows[vid].start, end=windows[vid].end)
            for vid in sample_ids
        }
        write_truncated(
            temp_csv,
            trajectories_path,
            day.index,
            windows,
            subgraph_ids,
            net_edge_ids,
            samples,
            stats,
        )
        write_vehicle_ids(vehicle_ids_path, list(windows))
        gz_size_after = day.gz_path.stat().st_size
        elapsed = time.perf_counter() - started
        min_delta = min(stats.time_deltas) if stats.time_deltas else None
        details = {
            "vehicle_ids_file_count": len(windows),
            "timestep_ok": min_delta == 1.0 and set(stats.time_deltas) == {1.0},
            "min_positive_time_delta_s": min_delta,
            "positive_time_deltas_s": sorted(stats.time_deltas)[:12],
            "rows_in_window": stats.rows_in_window,
            "sample_vehicles": sample_tracks_payload(samples),
        }
        summary = build_summary(
            day=day,
            subgraph_lines=subgraph_lines,
            subgraph_ids=subgraph_ids,
            net_edge_ids=net_edge_ids,
            stats=stats,
            windows=windows,
            trajectories_path=trajectories_path,
            vehicle_ids_path=vehicle_ids_path,
            summary_path=summary_path,
            net_file=net_file,
            subgraph_file=subgraph_file,
            gz_size_before=gz_size_before,
            gz_size_after=gz_size_after,
            temp_csv=temp_csv,
            temp_removed=False,
            elapsed=elapsed,
            status="extracted",
            failures=[],
            details=details,
            error=None,
        )
        summary["candidate_windows"] = serialize_windows(windows)
        summary["validation_passed"] = False
        summary["validation_pending"] = True
        write_json(summary_path, summary)
        print(
            f"  extracted vehicles={len(windows)} rows={stats.output_rows} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        return summary
    except ExtractError:
        raise
    except Exception as exc:
        raise ExtractError(f"{day.day_key} failed: {exc}") from exc
    finally:
        temp_removed = cleanup_temp_csv(temp_csv)
        if summary_path.is_file():
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                payload["temp_csv_removed"] = not temp_csv.exists()
                payload["gz_unchanged"] = (
                    day.gz_path.is_file()
                    and day.gz_path.stat().st_size == gz_size_before
                )
                write_json(summary_path, payload)
            except (json.JSONDecodeError, OSError):
                pass
        if not day.gz_path.is_file():
            print(f"WARNING: original gzip is missing: {day.gz_path}", file=sys.stderr, flush=True)


def build_summary(
    *,
    day: DaySource,
    subgraph_lines: list[str],
    subgraph_ids: set[str],
    net_edge_ids: set[str],
    stats: ExtractStats,
    windows: dict[str, VehicleWindow],
    trajectories_path: Path,
    vehicle_ids_path: Path,
    summary_path: Path,
    net_file: Path,
    subgraph_file: Path,
    gz_size_before: int,
    gz_size_after: int,
    temp_csv: Path,
    temp_removed: bool,
    elapsed: float,
    status: str,
    failures: list[str],
    details: dict[str, object],
    error: str | None,
) -> dict[str, object]:
    missing_in_fcd = sorted(subgraph_ids - stats.seen_subgraph_edges)
    missing_in_net = sorted(subgraph_ids - net_edge_ids)
    reentered_n = sum(1 for window in windows.values() if window.reentered)
    min_delta = min(stats.time_deltas) if stats.time_deltas else None
    trajectories_bytes = trajectories_path.stat().st_size if trajectories_path.is_file() else 0
    return {
        "day": day.day_key,
        "day_index": day.index,
        "simulation_folder": day.stem,
        "input_fcd_gz": str(day.gz_path),
        "net_file": str(net_file),
        "subgraph_file": str(subgraph_file),
        "output_dir": str(trajectories_path.parent),
        "output_trajectories": str(trajectories_path),
        "output_vehicle_ids": str(vehicle_ids_path),
        "output_summary": str(summary_path),
        "subgraph_text_lines": len(subgraph_lines),
        "subgraph_id_count": len(subgraph_ids),
        "subgraph_ids_in_net": len(subgraph_ids & net_edge_ids),
        "subgraph_ids_missing_from_net": missing_in_net,
        "subgraph_ids_seen_in_fcd": len(stats.seen_subgraph_edges),
        "subgraph_ids_never_in_fcd": missing_in_fcd,
        "candidate_vehicle_count": len(windows),
        "vehicle_ids_file_count": details.get("vehicle_ids_file_count", len(windows)),
        "output_row_count": stats.output_rows,
        "fcd_row_count": stats.fcd_rows,
        "output_min_timestamp": stats.output_min_time_raw,
        "output_max_timestamp": stats.output_max_time_raw,
        "unique_fcd_timestamps": len(stats.unique_times),
        "timestep_seconds": min_delta,
        "timestep_ok": details.get("timestep_ok"),
        "positive_time_deltas_s": details.get("positive_time_deltas_s", sorted(stats.time_deltas)[:12]),
        "reentered_vehicle_count": reentered_n,
        "unparseable_lane_count": stats.unparseable_count,
        "unparseable_lane_examples": stats.unparseable_examples,
        "missing_net_edge_count": len(stats.missing_net_edges),
        "missing_net_edge_examples": sorted(stats.missing_net_edges)[:MISSING_EDGE_EXAMPLES],
        "status": status,
        "validation_passed": not failures,
        "validation_failures": failures,
        "error": error,
        "elapsed_seconds": round(elapsed, 3),
        "trajectories_bytes": trajectories_bytes,
        "gz_size_before": gz_size_before,
        "gz_size_after": gz_size_after,
        "gz_unchanged": gz_size_before == gz_size_after and Path(day.gz_path).is_file(),
        "temp_csv": str(temp_csv),
        "temp_csv_removed": temp_removed,
        **{k: v for k, v in details.items() if k != "sample_vehicles"},
        "sample_vehicles": details.get("sample_vehicles", {}),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_failed_summary(
    day: DaySource,
    *,
    output_root: Path,
    error: str,
    gz_size_before: int,
    elapsed: float,
) -> dict[str, object]:
    out_dir = output_root / day.day_key
    summary_path = out_dir / DAY_SUMMARY_NAME
    if summary_path.is_file():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if "candidate_vehicle_count" in payload:
                payload["status"] = "failed"
                payload["validation_passed"] = False
                payload["error"] = error
                failures = [str(item) for item in (payload.get("validation_failures") or [])]
                if error not in failures:
                    failures.append(error)
                payload["validation_failures"] = failures
                payload["elapsed_seconds"] = round(elapsed, 3)
                write_json(summary_path, payload)
                return payload
        except (json.JSONDecodeError, OSError):
            pass
    payload = {
        "day": day.day_key,
        "day_index": day.index,
        "simulation_folder": day.stem,
        "input_fcd_gz": str(day.gz_path),
        "output_dir": str(out_dir),
        "status": "failed",
        "validation_passed": False,
        "validation_failures": [error],
        "error": error,
        "elapsed_seconds": round(elapsed, 3),
        "gz_size_before": gz_size_before,
        "gz_unchanged": day.gz_path.is_file() and day.gz_path.stat().st_size == gz_size_before,
        "temp_csv_removed": True,
    }
    write_json(summary_path, payload)
    return payload


def execute_day_job(
    day: DaySource,
    *,
    subgraph_lines: list[str],
    subgraph_ids: set[str],
    net_edge_ids: set[str],
    output_root: Path,
    temp_root: Path,
    net_file: Path,
    subgraph_file: Path,
) -> dict[str, object]:
    """One independent day. Safe to call from a worker process."""
    started = time.perf_counter()
    gz_size_before = day.gz_path.stat().st_size
    try:
        summary = process_one_day(
            day,
            subgraph_lines=subgraph_lines,
            subgraph_ids=subgraph_ids,
            net_edge_ids=net_edge_ids,
            output_root=output_root,
            temp_root=temp_root,
            net_file=net_file,
            subgraph_file=subgraph_file,
        )
        disk_summary = output_root / day.day_key / DAY_SUMMARY_NAME
        if disk_summary.is_file():
            summary = json.loads(disk_summary.read_text(encoding="utf-8"))
        return summary
    except ExtractError as exc:
        elapsed = time.perf_counter() - started
        print(f"  ERROR {day.day_key}: {exc}", file=sys.stderr, flush=True)
        return write_failed_summary(
            day,
            output_root=output_root,
            error=str(exc),
            gz_size_before=gz_size_before,
            elapsed=elapsed,
        )


def write_all_days_summary(output_root: Path, summaries: list[dict[str, object]]) -> Path:
    path = output_root / ALL_DAYS_SUMMARY_NAME
    fieldnames = [
        "day",
        "day_index",
        "simulation_folder",
        "input_fcd_gz",
        "candidate_vehicle_count",
        "output_row_count",
        "subgraph_ids_seen_in_fcd",
        "subgraph_ids_never_in_fcd_count",
        "reentered_vehicle_count",
        "output_min_timestamp",
        "output_max_timestamp",
        "trajectories_bytes",
        "elapsed_seconds",
        "validation_passed",
        "status",
        "anomalies",
    ]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for summary in summaries:
            missing = summary.get("subgraph_ids_never_in_fcd") or []
            failures = summary.get("validation_failures") or []
            error = summary.get("error")
            anomalies = "; ".join(str(item) for item in [*failures, *([error] if error else [])] if item)
            writer.writerow(
                {
                    "day": summary.get("day"),
                    "day_index": summary.get("day_index"),
                    "simulation_folder": summary.get("simulation_folder"),
                    "input_fcd_gz": summary.get("input_fcd_gz"),
                    "candidate_vehicle_count": summary.get("candidate_vehicle_count"),
                    "output_row_count": summary.get("output_row_count"),
                    "subgraph_ids_seen_in_fcd": summary.get("subgraph_ids_seen_in_fcd"),
                    "subgraph_ids_never_in_fcd_count": len(missing) if isinstance(missing, list) else missing,
                    "reentered_vehicle_count": summary.get("reentered_vehicle_count"),
                    "output_min_timestamp": summary.get("output_min_timestamp"),
                    "output_max_timestamp": summary.get("output_max_timestamp"),
                    "trajectories_bytes": summary.get("trajectories_bytes"),
                    "elapsed_seconds": summary.get("elapsed_seconds"),
                    "validation_passed": summary.get("validation_passed"),
                    "status": summary.get("status"),
                    "anomalies": anomalies,
                }
            )
    return path


def load_day_summaries(output_root: Path) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    if not output_root.is_dir():
        return summaries
    for path in sorted(output_root.glob(f"*/{DAY_SUMMARY_NAME}")):
        if not path.parent.name.startswith("day_"):
            continue
        try:
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            summaries.append(
                {
                    "day": path.parent.name,
                    "status": "failed",
                    "validation_passed": False,
                    "error": f"cannot read {path}: {exc}",
                    "validation_failures": [f"cannot read {path}: {exc}"],
                }
            )
    summaries.sort(key=lambda item: int(item["day_index"]) if item.get("day_index") is not None else 10**9)
    return summaries


def leftover_temp_csvs(temp_root: Path) -> list[str]:
    if not temp_root.exists():
        return []
    return [str(path) for path in temp_root.rglob(TEMP_CSV_NAME) if path.is_file()]


def cross_day_checks(
    *,
    selected: list[DaySource],
    output_root: Path,
    temp_root: Path,
    summaries: list[dict[str, object]],
) -> list[str]:
    issues: list[str] = []
    expected = [day.day_key for day in selected]
    if len(expected) != len(set(expected)):
        issues.append(f"selected days contain duplicates: {expected}")
    found_dirs = sorted(
        path.name for path in output_root.iterdir() if path.is_dir() and path.name.startswith("day_")
    ) if output_root.is_dir() else []
    missing_dirs = [name for name in expected if name not in found_dirs]
    if missing_dirs:
        issues.append(f"missing output directories: {missing_dirs}")
    for day in selected:
        out_dir = output_root / day.day_key
        for name in (TRAJECTORIES_NAME, VEHICLE_IDS_NAME, DAY_SUMMARY_NAME):
            if not (out_dir / name).is_file():
                issues.append(f"{day.day_key} missing {name}")
        if not day.gz_path.is_file():
            issues.append(f"{day.day_key} original gzip missing: {day.gz_path}")
        else:
            by_day = next((item for item in summaries if item.get("day") == day.day_key), None)
            expected_size = by_day.get("gz_size_before") if by_day else None
            actual_size = day.gz_path.stat().st_size
            if expected_size is not None and actual_size != expected_size:
                issues.append(
                    f"{day.day_key} gzip size changed: {expected_size} -> {actual_size}"
                )
    leftover = leftover_temp_csvs(temp_root)
    if leftover:
        issues.append(f"leftover temporary CSV files: {leftover}")
    return issues


def select_days(args: argparse.Namespace, available: list[DaySource]) -> list[DaySource]:
    by_index = {day.index: day for day in available}
    if args.fcd is not None:
        return [day_from_fcd_path(args.fcd)]

    requested: list[int]
    if args.days:
        requested = args.days
    elif args.day_count is not None:
        start = args.from_day if args.from_day is not None else 1
        requested = list(range(start, start + args.day_count))
    elif args.from_day is not None or args.to_day is not None:
        start = args.from_day if args.from_day is not None else 1
        end = args.to_day if args.to_day is not None else start
        if end < start:
            raise ExtractError(f"--to-day {end} is before --from-day {start}")
        requested = list(range(start, end + 1))
    elif args.day is not None:
        requested = [args.day]
    else:
        return list(available)

    selected: list[DaySource] = []
    missing: list[int] = []
    for index in requested:
        day = by_index.get(index)
        if day is None:
            missing.append(index)
        else:
            selected.append(day)
    if missing:
        raise ExtractError(f"no FCD gzip for day index(es): {missing}")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-day FCD rows for vehicles that touch subgraph edges, "
            "truncated to [first, last] subgraph appearance. Days are processed independently."
        )
    )
    parser.add_argument("--subgraph", type=Path, default=DEFAULT_SUBGRAPH_FILE, help="Subgraph edge-id text file.")
    parser.add_argument("--net", type=Path, default=DEFAULT_NET_FILE, help="SUMO .net.xml path.")
    parser.add_argument("--fcd", type=Path, default=None, help="Single FCD gzip path.")
    parser.add_argument("--fcd-dir", type=Path, default=DEFAULT_FCD_DIR, help="Directory of day folders containing fcd.csv.gz.")
    parser.add_argument("--day", type=int, default=None, help="Single day index, for example 1.")
    parser.add_argument("--days", type=int, nargs="+", default=None, metavar="INDEX", help="Explicit day indices.")
    parser.add_argument("--from-day", type=int, default=None, help="First day index in an inclusive range.")
    parser.add_argument("--to-day", type=int, default=None, help="Last day index in an inclusive range.")
    parser.add_argument("--day-count", type=int, default=None, help="Number of days starting at --from-day or 1.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output root for per-day folders.")
    parser.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP_DIR, help="Temporary decompress directory.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help=(
            f"Number of days to process in parallel (default {DEFAULT_JOBS}). "
            "Each job decompresses and scans one day independently."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a day whose validation_summary.json already reports status=ok.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subgraph_file = args.subgraph.expanduser().resolve()
    net_file = args.net.expanduser().resolve()
    fcd_dir = args.fcd_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    temp_root = args.temp_dir.expanduser().resolve()

    for required in (subgraph_file, net_file):
        if not required.exists():
            raise SystemExit(f"required input not found: {required}")

    subgraph_lines, subgraph_ids = load_subgraph_ids(subgraph_file)
    print("Identified inputs:", flush=True)
    print(
        f"  subgraph: {subgraph_file} ({len(subgraph_lines)} lines, {len(subgraph_ids)} unique)",
        flush=True,
    )
    print(f"  net: {net_file}", flush=True)
    print("  fcd format: gzip CSV; columns time,id,x,y,angle,type,speed,pos,lane", flush=True)

    print("Loading net edge ids...", flush=True)
    net_edge_ids = load_net_edge_ids(net_file)
    print(f"  net edges={len(net_edge_ids):,}", flush=True)

    if args.fcd is not None:
        selected = [day_from_fcd_path(args.fcd)]
    else:
        available = discover_days(fcd_dir)
        selected = select_days(args, available)

    print("Selected days:", flush=True)
    for day in selected:
        print(f"  {day.day_key}: {day.gz_path}", flush=True)

    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")

    summaries: list[dict[str, object]] = []
    failures: list[str] = []
    output_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    pending: list[DaySource] = []
    for day in selected:
        summary_path = output_root / day.day_key / DAY_SUMMARY_NAME
        if args.skip_existing and existing_success(summary_path):
            print(f"\n== {day.day_key} skip existing ok ==", flush=True)
            continue
        if args.skip_existing and extraction_complete(summary_path):
            print(f"\n== {day.day_key} skip extract (already written) ==", flush=True)
            continue
        pending.append(day)

    workers = min(args.jobs, len(pending)) if pending else 0
    print(
        f"\nExtracting {len(pending)} day(s) with {workers} parallel job(s); "
        f"{len(selected) - len(pending)} skipped",
        flush=True,
    )

    job_kwargs = {
        "subgraph_lines": subgraph_lines,
        "subgraph_ids": subgraph_ids,
        "net_edge_ids": net_edge_ids,
        "output_root": output_root,
        "temp_root": temp_root,
        "net_file": net_file,
        "subgraph_file": subgraph_file,
    }

    extract_failures: list[str] = []

    def consume_extract(summary: dict[str, object]) -> None:
        summaries.append(summary)
        if summary.get("status") not in {"extracted", "ok"}:
            extract_failures.append(
                f"{summary.get('day')}: {summary.get('validation_failures') or summary.get('error')}"
            )

    if workers <= 1:
        for day in pending:
            consume_extract(execute_day_job(day, **job_kwargs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(execute_day_job, day, **job_kwargs): day for day in pending
            }
            for future in as_completed(future_map):
                day = future_map[future]
                try:
                    consume_extract(future.result())
                except Exception as exc:
                    print(f"  ERROR {day.day_key}: {exc}", file=sys.stderr, flush=True)
                    consume_extract(
                        write_failed_summary(
                            day,
                            output_root=output_root,
                            error=str(exc),
                            gz_size_before=day.gz_path.stat().st_size,
                            elapsed=0.0,
                        )
                    )

    print("\n== Extraction finished; starting unified validation ==", flush=True)
    to_validate = [
        day
        for day in selected
        if not (args.skip_existing and existing_success(output_root / day.day_key / DAY_SUMMARY_NAME))
    ]
    validate_workers = min(args.jobs, len(to_validate)) if to_validate else 0
    print(
        f"Validating {len(to_validate)} day(s) with {validate_workers} parallel job(s)",
        flush=True,
    )

    def consume_validation(summary: dict[str, object]) -> None:
        if summary.get("status") != "ok" or summary.get("validation_passed") is not True:
            failures.append(
                f"{summary.get('day')}: {summary.get('validation_failures') or summary.get('error')}"
            )

    if validate_workers <= 1:
        for day in to_validate:
            consume_validation(
                validate_extracted_day(day, subgraph_ids=subgraph_ids, output_root=output_root)
            )
    else:
        with ProcessPoolExecutor(max_workers=validate_workers) as executor:
            future_map = {
                executor.submit(
                    validate_extracted_day,
                    day,
                    subgraph_ids=subgraph_ids,
                    output_root=output_root,
                ): day
                for day in to_validate
            }
            for future in as_completed(future_map):
                day = future_map[future]
                try:
                    consume_validation(future.result())
                except Exception as exc:
                    print(f"  ERROR validating {day.day_key}: {exc}", file=sys.stderr, flush=True)
                    consume_validation(
                        write_failed_summary(
                            day,
                            output_root=output_root,
                            error=str(exc),
                            gz_size_before=day.gz_path.stat().st_size,
                            elapsed=0.0,
                        )
                    )

    for item in extract_failures:
        if item not in failures:
            failures.append(item)

    report_summaries = load_day_summaries(output_root)
    if report_summaries:
        summary_csv = write_all_days_summary(output_root, report_summaries)
        print(f"\nWrote all-days summary: {summary_csv}", flush=True)
    leftover_tmp = leftover_temp_csvs(temp_root)
    if leftover_tmp:
        print(f"WARNING leftover temp CSV: {leftover_tmp}", file=sys.stderr, flush=True)
    if selected:
        batch_issues = cross_day_checks(
            selected=selected,
            output_root=output_root,
            temp_root=temp_root,
            summaries=report_summaries or summaries,
        )
        for issue in batch_issues:
            print(f"batch check: {issue}", file=sys.stderr, flush=True)
            if issue not in failures:
                failures.append(issue)

    if failures:
        raise SystemExit(f"{len(failures)} day(s) failed:\n" + "\n".join(str(item) for item in failures))
    print("All requested days completed.", flush=True)


if __name__ == "__main__":
    main()
