"""Count 5-minute observed subgraph edge entries from nested vehicle samples.

Reuses the road-entry definition in count_edge_flow.py: a vehicle produces one
entry when its edge_id changes (or on its first row) and the new id is a
subgraph / R-node road. Stay-on-edge seconds are not counted. Internal edges
(`:` prefix) update the previous-edge pointer but are not R nodes.

One trajectory pass per day builds the entry-event table; seven penetration
files only filter that table. Does not resample vehicles, scale by 1/p, build
STGCN windows, or construct M/MR graphs.

Usage:

    python3 analysis/simulation/count_observed_edge_flow.py --all-days
    python3 analysis/simulation/count_observed_edge_flow.py --day day_01
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from count_edge_flow import (
    DEFAULT_WINDOW_MINUTES,
    SECONDS_PER_DAY,
    FlowError,
    flag_is_true,
    flow_csv_name,
    load_subgraph,
    time_label,
    window_index_for,
    window_label,
)


SCRIPT_VERSION = "1.0.0"
DEFAULT_RATES = (5, 10, 20, 30, 40, 50, 70)
DEFAULT_EXPECTED_DAYS = 20
DEFAULT_OUTPUT_SUBDIR = "observed_edge_flow_5min"
DEFAULT_IDS_NAME = "vehicle_ids.txt"
DEFAULT_SAMPLE_SUBDIR = "sampled_vehicle_ids"
DEFAULT_TRAJECTORIES_NAME = "trajectories.csv"
PROGRESS_EVERY = 500_000
EXAMPLE_LIMIT = 8

BASE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = BASE_DIR.parent
DEFAULT_DATA_ROOT = BASE_DIR / "data" / "processed" / "subgraph_trajectories"
DEFAULT_R_NODES = ANALYSIS_DIR / "graph" / "r_graph" / "r_nodes.csv"
DEFAULT_SUBGRAPH = BASE_DIR / "data" / "subgraph.txt"


class ObservedFlowError(Exception):
    pass


@dataclass
class FileFingerprint:
    path: Path
    size_bytes: int
    mtime_ns: int

    def snapshot(self) -> tuple[int, int]:
        return (self.size_bytes, self.mtime_ns)


@dataclass
class SampleSpec:
    percent: int
    path: Path
    ids: list[str]
    id_set: set[str]
    duplicate_ids: list[str]
    empty_line_count: int


@dataclass
class EntryEvent:
    vehicle_id: str
    timestamp: float
    window_index: int
    edge_id: str
    node_index: int


@dataclass
class ScanStats:
    rows: int = 0
    entry_count: int = 0
    skipped_outside_day: int = 0
    flag_mismatches: int = 0
    vehicles_seen: set[str] = field(default_factory=set)
    timestamp_min: float | None = None
    timestamp_max: float | None = None


def sample_filename(percent: int) -> str:
    return f"vehicles_p{percent:02d}.txt"


def observed_csv_name(percent: int) -> str:
    return f"edge_flow_5min_p{percent:02d}.csv"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fingerprint(path: Path) -> FileFingerprint:
    stat = path.stat()
    return FileFingerprint(path=path, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_rates(raw: list[str]) -> list[int]:
    rates: list[int] = []
    seen: set[int] = set()
    for item in raw:
        for part in item.split(","):
            part = part.strip().rstrip("%")
            if not part:
                continue
            value = int(part)
            if value <= 0 or value > 100:
                raise ObservedFlowError(f"penetration rate out of range: {part}")
            if value not in seen:
                seen.add(value)
                rates.append(value)
    if not rates:
        raise ObservedFlowError("no penetration rates")
    return rates


def load_id_list(path: Path) -> tuple[list[str], list[str], int]:
    unique: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    empty_line_count = 0
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        vehicle_id = raw.strip()
        if not vehicle_id:
            empty_line_count += 1
            continue
        if vehicle_id in seen:
            duplicates.append(vehicle_id)
            continue
        seen.add(vehicle_id)
        unique.append(vehicle_id)
    return unique, duplicates, empty_line_count


def load_r_nodes(path: Path) -> tuple[list[str], dict[str, int]]:
    if not path.is_file():
        raise ObservedFlowError(f"r_nodes.csv not found: {path}")
    rows: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = ("node_index", "edge_id")
        if reader.fieldnames is None:
            raise ObservedFlowError(f"{path} has no header")
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise ObservedFlowError(f"{path} missing {missing}")
        for row in reader:
            rows.append((int(row["node_index"]), row["edge_id"]))
    rows.sort(key=lambda item: item[0])
    if [index for index, _edge in rows] != list(range(len(rows))):
        raise ObservedFlowError(f"{path} node_index is not contiguous from 0")
    edge_ids = [edge_id for _index, edge_id in rows]
    if len(set(edge_ids)) != len(edge_ids):
        raise ObservedFlowError(f"{path} has duplicate edge_id values")
    index_by_edge = {edge_id: index for index, edge_id in enumerate(edge_ids)}
    return edge_ids, index_by_edge


def load_full_flow(
    path: Path,
    edge_ids: list[str],
    n_windows: int,
) -> dict[str, list[int]]:
    counts = {edge_id: [0] * n_windows for edge_id in edge_ids}
    seen_keys: set[tuple[int, str]] = set()
    order: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = ("window_index", "window_start", "time_label", "edge_id", "vehicle_count")
        if reader.fieldnames is None:
            raise ObservedFlowError(f"{path} has no header")
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise ObservedFlowError(f"{path} missing {missing}")
        for row in reader:
            edge_id = row["edge_id"]
            window_index = int(row["window_index"])
            if edge_id not in counts:
                raise ObservedFlowError(f"{path} has unknown edge_id {edge_id}")
            if window_index < 0 or window_index >= n_windows:
                raise ObservedFlowError(f"{path} has window_index {window_index} outside 0..{n_windows - 1}")
            key = (window_index, edge_id)
            if key in seen_keys:
                raise ObservedFlowError(f"{path} duplicate key {key}")
            seen_keys.add(key)
            if not order or order[-1] != edge_id:
                order.append(edge_id)
            counts[edge_id][window_index] = int(row["vehicle_count"])
            expected_start = window_index * (SECONDS_PER_DAY // n_windows)
            if int(row["window_start"]) != expected_start:
                raise ObservedFlowError(
                    f"{path} window_start mismatch at {edge_id} window {window_index}"
                )
    if order != edge_ids:
        raise ObservedFlowError(
            f"{path} edge order does not match r_nodes.csv"
        )
    expected_keys = n_windows * len(edge_ids)
    if len(seen_keys) != expected_keys:
        raise ObservedFlowError(
            f"{path} has {len(seen_keys)} cells, expected {expected_keys} (zeros must be explicit)"
        )
    return counts


def discover_day_dirs(root: Path) -> list[Path]:
    days: list[Path] = []
    if not root.is_dir():
        raise ObservedFlowError(f"data root is not a directory: {root}")
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        if child.is_dir() and (child / DEFAULT_TRAJECTORIES_NAME).is_file():
            days.append(child)
    return days


def load_samples(
    day_dir: Path,
    rates: list[int],
    sample_subdir: str,
) -> dict[int, SampleSpec]:
    samples: dict[int, SampleSpec] = {}
    for percent in rates:
        path = day_dir / sample_subdir / sample_filename(percent)
        if not path.is_file():
            raise ObservedFlowError(f"missing sample file: {path}")
        ids, duplicates, empty_line_count = load_id_list(path)
        samples[percent] = SampleSpec(
            percent=percent,
            path=path,
            ids=ids,
            id_set=set(ids),
            duplicate_ids=duplicates,
            empty_line_count=empty_line_count,
        )
    return samples


def validate_sample_membership(
    samples: dict[int, SampleSpec],
    day_vehicle_ids: set[str],
    rates: list[int],
) -> list[str]:
    failures: list[str] = []
    for percent in rates:
        spec = samples[percent]
        if spec.duplicate_ids:
            failures.append(
                f"p{percent:02d} has {len(spec.duplicate_ids)} duplicate IDs, e.g. {spec.duplicate_ids[:EXAMPLE_LIMIT]}"
            )
        extra = [vehicle_id for vehicle_id in spec.ids if vehicle_id not in day_vehicle_ids]
        if extra:
            failures.append(
                f"p{percent:02d} has {len(extra)} IDs not in vehicle_ids.txt, e.g. {extra[:EXAMPLE_LIMIT]}"
            )
    for earlier, later in zip(rates, rates[1:]):
        missing = [vehicle_id for vehicle_id in samples[earlier].ids if vehicle_id not in samples[later].id_set]
        if missing:
            failures.append(
                f"nested ID set failed: p{earlier:02d} not subset of p{later:02d}; "
                f"{len(missing)} extras e.g. {missing[:EXAMPLE_LIMIT]}"
            )
    return failures


def collect_entry_events(
    trajectories: Path,
    index_by_edge: dict[str, int],
    *,
    window_seconds: int,
) -> tuple[list[EntryEvent], ScanStats]:
    """Same entry rule as count_edge_flow.count_entries, plus event records."""
    subgraph = set(index_by_edge)
    events: list[EntryEvent] = []
    last_edge: dict[str, str] = {}
    stats = ScanStats()
    with trajectories.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = ("timestamp", "vehicle_id", "edge_id", "is_in_subgraph")
        if reader.fieldnames is None:
            raise ObservedFlowError(f"{trajectories} has no header")
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise ObservedFlowError(f"{trajectories} missing {missing}")
        for row in reader:
            stats.rows += 1
            if stats.rows % PROGRESS_EVERY == 0:
                print(
                    f"  scanned {stats.rows:,} rows entries={stats.entry_count:,}",
                    flush=True,
                )
            timestamp = float(row["timestamp"])
            if stats.timestamp_min is None or timestamp < stats.timestamp_min:
                stats.timestamp_min = timestamp
            if stats.timestamp_max is None or timestamp > stats.timestamp_max:
                stats.timestamp_max = timestamp
            vehicle_id = row["vehicle_id"]
            edge_id = row["edge_id"]
            stats.vehicles_seen.add(vehicle_id)
            in_subgraph = edge_id in subgraph
            flagged = flag_is_true(row["is_in_subgraph"])
            if in_subgraph != flagged:
                stats.flag_mismatches += 1
            previous = last_edge.get(vehicle_id)
            entered = previous is None or previous != edge_id
            last_edge[vehicle_id] = edge_id
            if not entered or not in_subgraph:
                continue
            index = window_index_for(timestamp, window_seconds)
            if index is None:
                stats.skipped_outside_day += 1
                continue
            events.append(
                EntryEvent(
                    vehicle_id=vehicle_id,
                    timestamp=timestamp,
                    window_index=index,
                    edge_id=edge_id,
                    node_index=index_by_edge[edge_id],
                )
            )
            stats.entry_count += 1
    return events, stats


def empty_counts(edge_ids: list[str], n_windows: int) -> dict[str, list[int]]:
    return {edge_id: [0] * n_windows for edge_id in edge_ids}


def aggregate_events(
    events: list[EntryEvent],
    allowed_ids: set[str] | None,
    edge_ids: list[str],
    n_windows: int,
) -> dict[str, list[int]]:
    counts = empty_counts(edge_ids, n_windows)
    for event in events:
        if allowed_ids is not None and event.vehicle_id not in allowed_ids:
            continue
        counts[event.edge_id][event.window_index] += 1
    return counts


def render_observed_csv(
    *,
    day: str,
    percent: int,
    edge_ids: list[str],
    counts: dict[str, list[int]],
    n_windows: int,
    window_seconds: int,
    index_by_edge: dict[str, int],
) -> bytes:
    lines = [
        "window_index,window_start,time_label,edge_id,vehicle_count,day,node_index,penetration_rate\n"
    ]
    rate = f"{percent / 100:.2f}"
    for edge_id in edge_ids:
        node_index = index_by_edge[edge_id]
        series = counts[edge_id]
        for window_index, value in enumerate(series):
            lines.append(
                f"{window_index},{window_index * window_seconds},"
                f"{time_label(window_index, window_seconds)},{edge_id},"
                f"{value},{day},{node_index},{rate}\n"
            )
    return "".join(lines).encode("utf-8")


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def load_observed_csv(path: Path) -> tuple[list[dict[str, str]], dict[str, list[int]]]:
    rows: list[dict[str, str]] = []
    counts: dict[str, list[int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ObservedFlowError(f"{path} has no header")
        for row in reader:
            rows.append(row)
            edge_id = row["edge_id"]
            window_index = int(row["window_index"])
            counts.setdefault(edge_id, [])
            series = counts[edge_id]
            while len(series) <= window_index:
                series.append(0)
            series[window_index] = int(row["vehicle_count"])
    return rows, counts


def write_status(path: Path, payload: dict[str, object]) -> None:
    write_json(path, payload)


def process_day(
    day_dir: Path,
    *,
    rates: list[int],
    r_edge_ids: list[str],
    index_by_edge: dict[str, int],
    window_seconds: int,
    n_windows: int,
    output_subdir: str,
    sample_subdir: str,
    overwrite: bool,
    status_path: Path,
    days_done: int,
    days_total: int,
) -> dict[str, object]:
    day = day_dir.name
    trajectories = day_dir / DEFAULT_TRAJECTORIES_NAME
    vehicle_ids_path = day_dir / DEFAULT_IDS_NAME
    full_flow_path = day_dir / flow_csv_name(window_seconds)
    output_dir = day_dir / output_subdir
    print(f"PROGRESS day={day} stage=start csv_done={days_done * len(rates)}", flush=True)

    missing = [
        str(path)
        for path in (trajectories, vehicle_ids_path, full_flow_path)
        if not path.is_file()
    ]
    for percent in rates:
        sample_path = day_dir / sample_subdir / sample_filename(percent)
        if not sample_path.is_file():
            missing.append(str(sample_path))
    if missing:
        raise ObservedFlowError(f"{day} missing inputs: {missing}")

    existing = list(output_dir.glob("edge_flow_5min_p*.csv")) if output_dir.is_dir() else []
    if existing and not overwrite:
        raise ObservedFlowError(
            f"{day} already has {len(existing)} observed CSVs in {output_dir}; pass --overwrite"
        )

    input_fps = {
        "trajectories": fingerprint(trajectories),
        "vehicle_ids": fingerprint(vehicle_ids_path),
        "full_flow": fingerprint(full_flow_path),
    }
    samples = load_samples(day_dir, rates, sample_subdir)
    for percent, spec in samples.items():
        input_fps[f"sample_p{percent:02d}"] = fingerprint(spec.path)

    day_vehicle_ids, day_id_dups, _empty = load_id_list(vehicle_ids_path)
    day_vehicle_set = set(day_vehicle_ids)
    if day_id_dups:
        raise ObservedFlowError(f"{day} vehicle_ids.txt has duplicates e.g. {day_id_dups[:EXAMPLE_LIMIT]}")

    membership_failures = validate_sample_membership(samples, day_vehicle_set, rates)
    if membership_failures:
        raise ObservedFlowError(f"{day} sample membership/nesting failed: " + " | ".join(membership_failures))

    full_counts = load_full_flow(full_flow_path, r_edge_ids, n_windows)
    print(f"PROGRESS day={day} stage=scan_trajectories", flush=True)
    write_status(
        status_path,
        {
            "current_day": day,
            "current_stage": "scan_trajectories",
            "current_rate": None,
            "days_done": days_done,
            "days_total": days_total,
            "csv_written": days_done * len(rates),
            "updated_at": now_utc(),
        },
    )
    events, scan_stats = collect_entry_events(
        trajectories,
        index_by_edge,
        window_seconds=window_seconds,
    )
    full_from_events = aggregate_events(events, None, r_edge_ids, n_windows)
    if full_from_events != full_counts:
        raise ObservedFlowError(
            f"{day} entry-event aggregation does not match {full_flow_path.name}; "
            "refusing to emit observed flows with a different counting rule"
        )

    missing_in_traj = []
    for percent in rates:
        for vehicle_id in samples[percent].ids:
            if vehicle_id not in scan_stats.vehicles_seen:
                missing_in_traj.append(vehicle_id)
                if len(missing_in_traj) >= EXAMPLE_LIMIT:
                    break
        if missing_in_traj:
            break
    if missing_in_traj:
        raise ObservedFlowError(
            f"{day} sampled IDs missing from trajectories.csv e.g. {missing_in_traj[:EXAMPLE_LIMIT]}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rate_results: list[dict[str, object]] = []
    observed_counts: dict[int, dict[str, list[int]]] = {}
    nested_ok = True
    not_greater_ok = True
    alignment_ok = True
    nested_examples: list[str] = []
    greater_examples: list[str] = []

    for percent in rates:
        print(f"PROGRESS day={day} stage=aggregate rate={percent}", flush=True)
        write_status(
            status_path,
            {
                "current_day": day,
                "current_stage": "aggregate",
                "current_rate": percent,
                "days_done": days_done,
                "days_total": days_total,
                "csv_written": days_done * len(rates) + len(rate_results),
                "updated_at": now_utc(),
            },
        )
        counts = aggregate_events(events, samples[percent].id_set, r_edge_ids, n_windows)
        observed_counts[percent] = counts
        payload = render_observed_csv(
            day=day,
            percent=percent,
            edge_ids=r_edge_ids,
            counts=counts,
            n_windows=n_windows,
            window_seconds=window_seconds,
            index_by_edge=index_by_edge,
        )
        out_path = output_dir / observed_csv_name(percent)
        write_bytes(out_path, payload)
        reloaded_rows, reloaded_counts = load_observed_csv(out_path)
        if reloaded_counts != counts:
            raise ObservedFlowError(f"{out_path} round-trip counts differ")
        if payload != out_path.read_bytes():
            raise ObservedFlowError(f"{out_path} bytes changed after write")
        event_total = sum(
            1 for event in events if event.vehicle_id in samples[percent].id_set
        )
        csv_total = sum(sum(series) for series in counts.values())
        if event_total != csv_total:
            raise ObservedFlowError(
                f"{day} p{percent:02d} event total {event_total} != CSV total {csv_total}"
            )
        days_in_file = {row["day"] for row in reloaded_rows}
        rates_in_file = {row["penetration_rate"] for row in reloaded_rows}
        if days_in_file != {day}:
            raise ObservedFlowError(f"{out_path} contains unexpected day values {days_in_file}")
        if rates_in_file != {f"{percent / 100:.2f}"}:
            raise ObservedFlowError(f"{out_path} contains unexpected penetration_rate {rates_in_file}")
        if len(reloaded_rows) != n_windows * len(r_edge_ids):
            alignment_ok = False
            raise ObservedFlowError(f"{out_path} row count {len(reloaded_rows)} != {n_windows * len(r_edge_ids)}")
        keys = [(int(row["window_index"]), row["edge_id"]) for row in reloaded_rows]
        if len(keys) != len(set(keys)):
            raise ObservedFlowError(f"{out_path} has duplicate window-edge keys")
        for edge_id in r_edge_ids:
            if any(value < 0 for value in counts[edge_id]):
                raise ObservedFlowError(f"{out_path} has negative flow on {edge_id}")
            if any(not float(value).is_integer() for value in counts[edge_id]):
                raise ObservedFlowError(f"{out_path} has non-integer flow on {edge_id}")
            for window_index, value in enumerate(counts[edge_id]):
                full_value = full_counts[edge_id][window_index]
                if value > full_value:
                    not_greater_ok = False
                    if len(greater_examples) < EXAMPLE_LIMIT:
                        greater_examples.append(
                            f"p{percent:02d} {edge_id} t={window_index} obs={value} full={full_value}"
                        )
        nonzero = sum(1 for edge_id in r_edge_ids for value in counts[edge_id] if value)
        rate_results.append(
            {
                "penetration_percent": percent,
                "penetration_rate": percent / 100.0,
                "sampled_vehicle_count": len(samples[percent].ids),
                "observed_entry_event_count": csv_total,
                "nonzero_cell_count": nonzero,
                "output_file": str(out_path.as_posix()),
                "output_name": out_path.name,
                "file_size": out_path.stat().st_size,
                "sha256": sha256_file(out_path),
            }
        )

    for earlier, later in zip(rates, rates[1:]):
        for edge_id in r_edge_ids:
            for window_index in range(n_windows):
                left = observed_counts[earlier][edge_id][window_index]
                right = observed_counts[later][edge_id][window_index]
                if left > right:
                    nested_ok = False
                    if len(nested_examples) < EXAMPLE_LIMIT:
                        nested_examples.append(
                            f"{edge_id} t={window_index} p{earlier:02d}={left} p{later:02d}={right}"
                        )
    last_rate = rates[-1]
    for edge_id in r_edge_ids:
        for window_index in range(n_windows):
            obs = observed_counts[last_rate][edge_id][window_index]
            full_value = full_counts[edge_id][window_index]
            if obs > full_value:
                nested_ok = False
                if len(nested_examples) < EXAMPLE_LIMIT:
                    nested_examples.append(
                        f"{edge_id} t={window_index} p{last_rate:02d}={obs} full={full_value}"
                    )

    input_fps_after = {
        "trajectories": fingerprint(trajectories),
        "vehicle_ids": fingerprint(vehicle_ids_path),
        "full_flow": fingerprint(full_flow_path),
    }
    for percent, spec in samples.items():
        input_fps_after[f"sample_p{percent:02d}"] = fingerprint(spec.path)
    inputs_unchanged = all(
        input_fps[name].snapshot() == input_fps_after[name].snapshot() for name in input_fps
    )
    if not inputs_unchanged:
        raise ObservedFlowError(f"{day} input files changed during processing")

    reproducibility_ok = True
    for percent in rates:
        rebuilt = aggregate_events(events, samples[percent].id_set, r_edge_ids, n_windows)
        if rebuilt != observed_counts[percent]:
            reproducibility_ok = False

    overall = nested_ok and not_greater_ok and alignment_ok and reproducibility_ok and inputs_unchanged
    summary = {
        "day": day,
        "trajectories": str(trajectories.as_posix()),
        "full_edge_flow": str(full_flow_path.as_posix()),
        "vehicle_ids": str(vehicle_ids_path.as_posix()),
        "sample_files": {f"p{percent:02d}": str(samples[percent].path.as_posix()) for percent in rates},
        "output_dir": str(output_dir.as_posix()),
        "output_files": [item["output_name"] for item in rate_results],
        "r_node_count": len(r_edge_ids),
        "r_nodes_complete": len(r_edge_ids) == 56 and set(r_edge_ids) == set(index_by_edge),
        "time_slot_count": n_windows,
        "window_seconds": window_seconds,
        "trajectory_rows": scan_stats.rows,
        "full_entry_event_count": scan_stats.entry_count,
        "skipped_outside_day": scan_stats.skipped_outside_day,
        "flag_mismatches": scan_stats.flag_mismatches,
        "unique_vehicles_in_trajectories": len(scan_stats.vehicles_seen),
        "rates": rate_results,
        "membership_validation_passed": not membership_failures,
        "alignment_validation_passed": alignment_ok,
        "nested_validation_passed": nested_ok,
        "not_greater_than_full_passed": not_greater_ok,
        "reproducibility_validation_passed": reproducibility_ok,
        "inputs_unchanged": inputs_unchanged,
        "full_flow_matches_entry_events": True,
        "nested_failure_examples": nested_examples,
        "greater_than_full_examples": greater_examples,
        "overall_validation_passed": overall,
        "status": "ok" if overall else "failed",
        "script_version": SCRIPT_VERSION,
        "generated_at": now_utc(),
        "notes": "",
    }
    write_json(output_dir / "validation_summary.json", summary)
    if not overall:
        raise ObservedFlowError(
            f"{day} validation failed nested={nested_ok} "
            f"not_greater={not_greater_ok} alignment={alignment_ok} repro={reproducibility_ok}"
        )
    print(
        f"PROGRESS day={day} stage=done csv_done={(days_done + 1) * len(rates)} status=ok",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count 5-minute observed R-edge entries for nested vehicle "
            "penetration samples. One CSV per day per rate."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root containing day_* folders (default: simulation/data/processed/subgraph_trajectories).",
    )
    parser.add_argument(
        "--r-nodes",
        type=Path,
        default=DEFAULT_R_NODES,
        help="r_nodes.csv used for node_index order.",
    )
    parser.add_argument(
        "--subgraph",
        type=Path,
        default=DEFAULT_SUBGRAPH,
        help="Subgraph edge list; must match r_nodes edge_id order.",
    )
    parser.add_argument(
        "--day",
        action="append",
        default=None,
        help="Day directory name. Repeatable. Default: all discovered days.",
    )
    parser.add_argument(
        "--all-days",
        action="store_true",
        help="Process every discovered day (default if --day is omitted).",
    )
    parser.add_argument(
        "--rates",
        nargs="+",
        default=[str(rate) for rate in DEFAULT_RATES],
        help="Penetration percents (default: 5 10 20 30 40 50 70).",
    )
    parser.add_argument(
        "--output-subdir",
        default=DEFAULT_OUTPUT_SUBDIR,
        help="Per-day output subdirectory name.",
    )
    parser.add_argument(
        "--sample-subdir",
        default=DEFAULT_SAMPLE_SUBDIR,
        help="Per-day sampled ID subdirectory name.",
    )
    parser.add_argument(
        "--expected-days",
        type=int,
        default=DEFAULT_EXPECTED_DAYS,
        help="Required day count when processing all days.",
    )
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=DEFAULT_WINDOW_MINUTES,
        help="Must match full edge_flow window (default 5).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing observed CSVs.")
    parser.add_argument(
        "--status-file",
        type=Path,
        default=None,
        help="JSON status file for an external monitor.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    r_nodes_path = args.r_nodes.expanduser().resolve()
    subgraph_path = args.subgraph.expanduser().resolve()
    rates = parse_rates(list(args.rates))
    window_seconds = args.window_minutes * 60
    if SECONDS_PER_DAY % window_seconds != 0:
        raise ObservedFlowError("window must divide 86400 seconds")
    n_windows = SECONDS_PER_DAY // window_seconds
    r_edge_ids, index_by_edge = load_r_nodes(r_nodes_path)
    subgraph = load_subgraph(subgraph_path)
    if subgraph.unique_ids != r_edge_ids:
        raise ObservedFlowError("subgraph.txt unique edge order does not match r_nodes.csv")
    if len(r_edge_ids) != 56:
        raise ObservedFlowError(f"expected 56 R nodes, found {len(r_edge_ids)}")

    discovered = discover_day_dirs(data_root)
    if args.day:
        by_name = {path.name: path for path in discovered}
        wanted: list[Path] = []
        missing_names: list[str] = []
        for name in args.day:
            if name in by_name:
                wanted.append(by_name[name])
            else:
                missing_names.append(name)
        if missing_names:
            raise ObservedFlowError(
                "day directory not found: "
                + ", ".join(missing_names)
                + f"; discovered: {', '.join(path.name for path in discovered) or 'none'}"
            )
        day_dirs = wanted
    else:
        if len(discovered) != args.expected_days:
            names = ", ".join(path.name for path in discovered) or "none"
            raise ObservedFlowError(
                f"expected {args.expected_days} day directories with {DEFAULT_TRAJECTORIES_NAME}, "
                f"found {len(discovered)}: {names}"
            )
        day_dirs = discovered

    inventory_failures: list[str] = []
    for day_dir in day_dirs:
        for name in (DEFAULT_TRAJECTORIES_NAME, DEFAULT_IDS_NAME, flow_csv_name(window_seconds)):
            if not (day_dir / name).is_file():
                inventory_failures.append(f"{day_dir.name}: missing {name}")
        for percent in rates:
            if not (day_dir / args.sample_subdir / sample_filename(percent)).is_file():
                inventory_failures.append(
                    f"{day_dir.name}: missing {args.sample_subdir}/{sample_filename(percent)}"
                )
    if inventory_failures:
        raise ObservedFlowError("input inventory failed:\n" + "\n".join(inventory_failures))

    status_path = (
        args.status_file.expanduser().resolve()
        if args.status_file is not None
        else data_root / "observed_edge_flow_5min_status.json"
    )
    write_status(
        status_path,
        {
            "current_day": None,
            "current_stage": "start",
            "current_rate": None,
            "days_done": 0,
            "days_total": len(day_dirs),
            "csv_written": 0,
            "updated_at": now_utc(),
        },
    )

    summaries: list[dict[str, object]] = []
    failed_days: list[str] = []
    for index, day_dir in enumerate(day_dirs):
        try:
            summary = process_day(
                day_dir,
                rates=rates,
                r_edge_ids=r_edge_ids,
                index_by_edge=index_by_edge,
                window_seconds=window_seconds,
                n_windows=n_windows,
                output_subdir=args.output_subdir,
                sample_subdir=args.sample_subdir,
                overwrite=args.overwrite,
                status_path=status_path,
                days_done=index,
                days_total=len(day_dirs),
            )
            summaries.append(summary)
            print(
                f"{day_dir.name}: ok entries_full={summary['full_entry_event_count']} "
                + ", ".join(
                    f"p{item['penetration_percent']:02d}={item['observed_entry_event_count']}"
                    for item in summary["rates"]
                ),
                flush=True,
            )
        except (ObservedFlowError, FlowError) as exc:
            failed_days.append(day_dir.name)
            print(f"{day_dir.name}: FAIL {exc}", file=sys.stderr, flush=True)
            fail_dir = day_dir / args.output_subdir
            fail_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                fail_dir / "validation_summary.json",
                {
                    "day": day_dir.name,
                    "status": "failed",
                    "overall_validation_passed": False,
                    "notes": str(exc),
                    "generated_at": now_utc(),
                },
            )

    summary_rows: list[dict[str, object]] = []
    for summary in summaries:
        for item in summary["rates"]:
            summary_rows.append(
                {
                    "day": summary["day"],
                    "penetration_rate": item["penetration_rate"],
                    "sampled_vehicle_count": item["sampled_vehicle_count"],
                    "observed_entry_event_count": item["observed_entry_event_count"],
                    "nonzero_cell_count": item["nonzero_cell_count"],
                    "time_slot_count": summary["time_slot_count"],
                    "edge_count": summary["r_node_count"],
                    "output_file": item["output_name"],
                    "file_size": item["file_size"],
                    "membership_validation_passed": summary["membership_validation_passed"],
                    "alignment_validation_passed": summary["alignment_validation_passed"],
                    "nested_validation_passed": summary["nested_validation_passed"],
                    "not_greater_than_full_passed": summary["not_greater_than_full_passed"],
                    "reproducibility_validation_passed": summary["reproducibility_validation_passed"],
                    "overall_validation_passed": summary["overall_validation_passed"],
                    "notes": summary.get("notes", ""),
                }
            )
    summary_csv = data_root / "observed_edge_flow_5min_summary_all_days.csv"
    fieldnames = [
        "day",
        "penetration_rate",
        "sampled_vehicle_count",
        "observed_entry_event_count",
        "nonzero_cell_count",
        "time_slot_count",
        "edge_count",
        "output_file",
        "file_size",
        "membership_validation_passed",
        "alignment_validation_passed",
        "nested_validation_passed",
        "not_greater_than_full_passed",
        "reproducibility_validation_passed",
        "overall_validation_passed",
        "notes",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)

    write_status(
        status_path,
        {
            "current_day": None,
            "current_stage": "finished",
            "current_rate": None,
            "days_done": len(summaries),
            "days_total": len(day_dirs),
            "csv_written": len(summary_rows),
            "failed_days": failed_days,
            "updated_at": now_utc(),
        },
    )
    print(
        f"summary {summary_csv}: days_ok={len(summaries)}/{len(day_dirs)} "
        f"rows={len(summary_rows)} failed={failed_days or 'none'}",
        flush=True,
    )
    if failed_days or (args.day is None and len(summary_rows) != args.expected_days * len(rates)):
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except ObservedFlowError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
