#!/usr/bin/env python3
"""Build four 5-minute road-level states from Xuancheng26 FCD.

Each date directory receives ``inflow.csv``, ``outflow.csv``,
``accumulation_start.csv`` and ``accumulation_end.csv``. Inflow and outflow
keep the audited edge-event counts. Accumulation is the running stock of
those counts, starting from an empty network before t=0:

    N_end[t] = N_start[t] + inflow[t] - outflow[t]
    N_start[0] = 0
    N_start[t+1] = N_end[t]

``--migrate-legacy`` copies already audited ``entry_flow.csv`` /
``exit_flow.csv`` into the new names and does not reread FCD. ``--remove-legacy``
deletes those old names after the four new files have been validated.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from xml.parsers import expat

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

import prep_paths

FCD_ROOT = prep_paths.DEFAULT_FCD_ROOT
NET_FILE = prep_paths.DEFAULT_NET_FILE
OUT_ROOT = prep_paths.DEFAULT_TRAINING_ROOT / "full_flow" / "r-level"
SUMMARY_PATH = OUT_ROOT / "generation_summary.csv"
N_BINS = 288
BIN_SECONDS = 300
DAY_SECONDS = 86400
# One read buffer per day. Twenty-nine workers keep about 350 GiB in flight.
READ_BYTES = 12 << 30

STATE_FILES = (
    "inflow.csv",
    "outflow.csv",
    "accumulation_start.csv",
    "accumulation_end.csv",
)
LEGACY_FULL = {"inflow.csv": "entry_flow.csv", "outflow.csv": "exit_flow.csv"}

SUMMARY_FIELDS = [
    "date",
    "unique_vehicles",
    "total_entry_events",
    "total_exit_events",
    "entry_exit_difference",
    "entry_exit_relative_difference",
    "active_edges_entry",
    "active_edges_exit",
    "max_5min_entry",
    "max_5min_exit",
    "status",
]


def log(message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"{stamp} {message}", flush=True)


def bin_index(event_time: float) -> int | None:
    if event_time < 0:
        return None
    if event_time >= DAY_SECONDS:
        return N_BINS - 1
    return int(event_time // BIN_SECONDS)


@dataclass
class Stay:
    edge_index: int
    exited: bool = False


def load_network(net_path: Path) -> tuple[list[str], dict[str, str], set[str]]:
    """Return real edges in file order, lane->edge, and internal lane ids."""
    edges: list[str] = []
    seen: set[str] = set()
    lane_to_edge: dict[str, str] = {}
    internal_lanes: set[str] = set()
    current: tuple[str, str] | None = None

    def start(name: str, attrs) -> None:
        nonlocal current
        if name == "edge":
            current = (attrs.get("id", ""), attrs.get("function", ""))
            edge_id, function = current
            real = edge_id and not edge_id.startswith(":") and function != "internal"
            if real and edge_id not in seen:
                seen.add(edge_id)
                edges.append(edge_id)
            if not real:
                current = None
            return
        if name != "lane" or "id" not in attrs:
            return
        if current is None:
            internal_lanes.add(attrs["id"])
            return
        lane_to_edge[attrs["id"]] = current[0]

    def end(name: str) -> None:
        nonlocal current
        if name == "edge":
            current = None

    parser = expat.ParserCreate()
    parser.StartElementHandler = start
    parser.EndElementHandler = end
    with net_path.open("rb") as handle:
        parser.ParseFile(handle)
    if not edges:
        raise RuntimeError(f"no real edges found in {net_path}")
    return edges, lane_to_edge, internal_lanes


def read_chunks(path: Path):
    """Yield large sequential chunks. Plain XML also asks the kernel to prefetch."""
    if path.name.endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            while True:
                block = handle.read(READ_BYTES)
                if not block:
                    break
                yield block
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
        offset = 0
        while True:
            os.posix_fadvise(fd, offset, READ_BYTES * 2, os.POSIX_FADV_WILLNEED)
            block = os.read(fd, READ_BYTES)
            if not block:
                break
            yield block
            offset += len(block)
    finally:
        os.close(fd)


def prefer_xml(day_dir: Path, stem: str) -> Path:
    plain = day_dir / f"{stem}.xml"
    compressed = day_dir / f"{stem}.xml.gz"
    if plain.is_file() and plain.stat().st_size > 0:
        return plain
    if compressed.is_file() and compressed.stat().st_size > 0:
        return compressed
    raise FileNotFoundError(f"missing {plain} and {compressed}")


def load_arrivals(path: Path) -> dict[str, float]:
    """Arrival time for vehicles that reached a destination. Others are omitted."""
    arrivals: dict[str, float] = {}

    def start(name: str, attrs) -> None:
        if name != "tripinfo":
            return
        vehicle_id = attrs.get("id")
        if not vehicle_id:
            return
        try:
            arrival = float(attrs["arrival"]) if "arrival" in attrs else -1.0
            depart = float(attrs["depart"]) if "depart" in attrs else -1.0
        except ValueError:
            return
        if depart >= 0 and arrival >= 0:
            arrivals[vehicle_id] = arrival

    parser = expat.ParserCreate()
    parser.StartElementHandler = start
    for block in read_chunks(path):
        parser.Parse(block, False)
    parser.Parse(b"", True)
    return arrivals


def observe(
    states: dict[str, Stay],
    seen: set[str],
    vehicle_id: str,
    event_time: float,
    kind: str,
    edge_index: int | None,
    entry: np.ndarray,
    exit_flow: np.ndarray,
    counts: dict[str, int],
    on_event=None,
) -> None:
    def add_entry(slot: int, column: int) -> None:
        entry[slot, column] += 1
        counts["entry"] += 1
        if on_event is not None:
            on_event("entry", slot, column, vehicle_id)

    def add_exit(slot: int, column: int) -> None:
        exit_flow[slot, column] += 1
        counts["exit"] += 1
        if on_event is not None:
            on_event("exit", slot, column, vehicle_id)

    if vehicle_id not in seen:
        seen.add(vehicle_id)
    slot = bin_index(event_time)
    state = states.get(vehicle_id)
    if kind != "real":
        if kind == "internal" and state is not None and not state.exited and slot is not None:
            add_exit(slot, state.edge_index)
            state.exited = True
        elif kind == "unknown":
            counts["unknown"] += 1
        return
    if edge_index is None or slot is None:
        return
    if state is None:
        add_entry(slot, edge_index)
        states[vehicle_id] = Stay(edge_index, False)
        return
    if edge_index == state.edge_index:
        return
    if not state.exited:
        add_exit(slot, state.edge_index)
    add_entry(slot, edge_index)
    state.edge_index = edge_index
    state.exited = False


def impute_final_exits(
    states: dict[str, Stay],
    arrivals: dict[str, float],
    exit_flow: np.ndarray,
    counts: dict[str, int],
    on_event=None,
) -> None:
    for vehicle_id, state in states.items():
        if state.exited:
            continue
        arrival = arrivals.get(vehicle_id)
        if arrival is None:
            counts["unresolved"] += 1
            continue
        slot = bin_index(arrival)
        if slot is None:
            counts["unresolved"] += 1
            continue
        exit_flow[slot, state.edge_index] += 1
        counts["exit"] += 1
        counts["final_exit"] += 1
        if on_event is not None:
            on_event("exit", slot, state.edge_index, vehicle_id)
        state.exited = True


def read_flow_csv(path: Path, edges: list[str]) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    if header != ["time_bin", "start_time", *edges]:
        raise RuntimeError(f"column mismatch in {path}")
    matrix = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64)
    if matrix.shape != (N_BINS, len(edges) + 2):
        raise RuntimeError(f"{path} shape {matrix.shape}")
    if not np.array_equal(matrix[:, 0], np.arange(N_BINS)):
        raise RuntimeError(f"{path} time_bin is not 0..287")
    if not np.array_equal(matrix[:, 1], np.arange(N_BINS) * BIN_SECONDS):
        raise RuntimeError(f"{path} start_time is not 300-second bins")
    values = matrix[:, 2:]
    if int(values.min()) < 0:
        raise RuntimeError(f"negative value in {path}")
    return values


def accumulate_states(
    inflow: np.ndarray,
    outflow: np.ndarray,
    edges: list[str],
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Stock before t=0 is zero. Each window then conserves vehicles."""
    if inflow.shape != outflow.shape:
        raise RuntimeError(f"{label} inflow shape {inflow.shape} != outflow {outflow.shape}")
    if inflow.ndim != 2 or inflow.shape[1] != len(edges) or inflow.shape[0] < 1:
        raise RuntimeError(f"{label} unexpected shape {inflow.shape}")
    delta = inflow.astype(np.int64) - outflow.astype(np.int64)
    end = np.cumsum(delta, axis=0)
    start = np.zeros_like(end)
    start[1:] = end[:-1]
    for name, matrix in (("accumulation_start", start), ("accumulation_end", end)):
        negative = np.argwhere(matrix < 0)
        if len(negative):
            row, column = (int(negative[0, 0]), int(negative[0, 1]))
            raise RuntimeError(
                f"{label} {name} negative at time_bin={row} start_time={row * BIN_SECONDS} "
                f"edge={edges[column]} value={int(matrix[row, column])} "
                f"inflow={int(inflow[row, column])} outflow={int(outflow[row, column])}"
            )
    if not np.array_equal(end, start + delta):
        raise RuntimeError(f"{label} window conservation failed")
    if not np.array_equal(start[1:], end[:-1]):
        raise RuntimeError(f"{label} adjacent-window stock is not continuous")
    if int(start[0].min()) != 0 or int(start[0].max()) != 0:
        raise RuntimeError(f"{label} accumulation_start at t=0 is not zero")
    return start, end


def write_state_matrices(
    out_dir: Path,
    inflow: np.ndarray,
    outflow: np.ndarray,
    edges: list[str],
    label: str,
    *,
    rewrite_inflow: bool = True,
) -> None:
    start, end = accumulate_states(inflow, outflow, edges, label)
    written = {
        "inflow.csv": inflow,
        "outflow.csv": outflow,
        "accumulation_start.csv": start,
        "accumulation_end.csv": end,
    }
    for name, matrix in written.items():
        path = out_dir / name
        if name == "inflow.csv" and not rewrite_inflow:
            if not path.is_file():
                raise RuntimeError(f"{label} missing {path}")
        else:
            write_matrix(path, matrix, edges)
        validate_csv(path, edges)
        reread = read_flow_csv(path, edges)
        if not np.array_equal(reread, matrix):
            raise RuntimeError(f"{label} {path} does not match the counted matrix")


def states_match(
    out_dir: Path,
    inflow: np.ndarray,
    outflow: np.ndarray,
    edges: list[str],
    label: str,
) -> bool:
    paths = [out_dir / name for name in STATE_FILES]
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
        return False
    got_in = read_flow_csv(out_dir / "inflow.csv", edges)
    got_out = read_flow_csv(out_dir / "outflow.csv", edges)
    got_start = read_flow_csv(out_dir / "accumulation_start.csv", edges)
    got_end = read_flow_csv(out_dir / "accumulation_end.csv", edges)
    if not np.array_equal(got_in, inflow) or not np.array_equal(got_out, outflow):
        return False
    start, end = accumulate_states(inflow, outflow, edges, label)
    return np.array_equal(got_start, start) and np.array_equal(got_end, end)


def remove_legacy_full(out_dir: Path) -> None:
    for name in LEGACY_FULL.values():
        path = out_dir / name
        if path.is_file():
            path.unlink()


def confirm_fcd_starts_at_zero(fcd_root: Path, dates: list[str]) -> None:
    """The day begins at t=0. No FCD sample exists before that instant."""
    for date_label in dates:
        path = prefer_xml(fcd_root / date_label, f"fcd_{date_label}")
        opener = gzip.open if path.name.endswith(".gz") else open
        found = False
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            for index, line in enumerate(handle):
                if "<timestep" not in line:
                    if index > 80:
                        break
                    continue
                if 'time="0"' not in line and 'time="0.0"' not in line and 'time="0.00"' not in line:
                    raise RuntimeError(f"{date_label} first timestep is not 0: {line.strip()[:160]}")
                found = True
                break
        if not found:
            raise RuntimeError(f"{date_label} FCD has no leading timestep in {path}")
    log(f"confirmed {len(dates)} FCD files start at t=0")


def write_matrix(path: Path, matrix: np.ndarray, edges: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["time_bin", "start_time", *edges])
        for slot in range(N_BINS):
            writer.writerow([slot, slot * BIN_SECONDS, *matrix[slot].tolist()])
    os.replace(temporary, path)


def validate_csv(path: Path, edges: list[str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != ["time_bin", "start_time", *edges]:
            raise RuntimeError(f"column mismatch in {path}")
        rows = 0
        expected_bin = 0
        for row in reader:
            if len(row) != len(header):
                raise RuntimeError(f"short row in {path}")
            if int(row[0]) != expected_bin or int(row[1]) != expected_bin * BIN_SECONDS:
                raise RuntimeError(f"bin mismatch at row {expected_bin} in {path}")
            for value in row[2:]:
                parsed = int(value)
                if parsed < 0 or str(parsed) != value:
                    raise RuntimeError(f"non-integer or negative flow in {path}")
            rows += 1
            expected_bin += 1
        if rows != N_BINS:
            raise RuntimeError(f"{path} has {rows} bins, expected {N_BINS}")


def stream_fcd(
    path: Path,
    edges: list[str],
    lane_to_edge: dict[str, str],
    internal_lanes: set[str],
    arrivals: dict[str, float],
    date_label: str,
    include: set[str] | None = None,
    on_event=None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], int]:
    edge_index = {edge_id: index for index, edge_id in enumerate(edges)}
    entry = np.zeros((N_BINS, len(edges)), dtype=np.int64)
    exit_flow = np.zeros((N_BINS, len(edges)), dtype=np.int64)
    states: dict[str, Stay] = {}
    seen: set[str] = set()
    counts = {"entry": 0, "exit": 0, "unknown": 0, "unresolved": 0, "final_exit": 0, "records": 0}
    clock = {"time": 0.0}
    read_bytes = 0
    next_report = 2 << 30

    def start(name: str, attrs) -> None:
        if name == "timestep":
            clock["time"] = float(attrs.get("time", "0"))
            return
        if name != "vehicle":
            return
        counts["records"] += 1
        vehicle_id = attrs.get("id", "")
        lane = attrs.get("lane", "")
        if not vehicle_id:
            return
        if include is not None and vehicle_id not in include:
            return
        if lane in lane_to_edge:
            observe(
                states, seen, vehicle_id, clock["time"], "real",
                edge_index[lane_to_edge[lane]], entry, exit_flow, counts, on_event,
            )
        elif lane in internal_lanes:
            observe(
                states, seen, vehicle_id, clock["time"], "internal",
                None, entry, exit_flow, counts, on_event,
            )
        else:
            observe(
                states, seen, vehicle_id, clock["time"], "unknown",
                None, entry, exit_flow, counts, on_event,
            )

    parser = expat.ParserCreate()
    parser.StartElementHandler = start
    for block in read_chunks(path):
        read_bytes += len(block)
        parser.Parse(block, False)
        if read_bytes >= next_report:
            log(
                f"[{date_label}] fcd {read_bytes / 1024**3:.1f} GiB "
                f"records={counts['records']} entries={counts['entry']}"
            )
            next_report += 2 << 30
    parser.Parse(b"", True)
    impute_final_exits(states, arrivals, exit_flow, counts, on_event)
    return entry, exit_flow, counts, len(seen)


def summary_row(date_label: str, entry: np.ndarray, exit_flow: np.ndarray, vehicles: int) -> dict[str, object]:
    total_entry = int(entry.sum())
    total_exit = int(exit_flow.sum())
    difference = total_entry - total_exit
    relative = (difference / total_entry) if total_entry else 0.0
    return {
        "date": date_label,
        "unique_vehicles": vehicles,
        "total_entry_events": total_entry,
        "total_exit_events": total_exit,
        "entry_exit_difference": difference,
        "entry_exit_relative_difference": f"{relative:.8f}",
        "active_edges_entry": int((entry.sum(axis=0) > 0).sum()),
        "active_edges_exit": int((exit_flow.sum(axis=0) > 0).sum()),
        "max_5min_entry": int(entry.max()) if entry.size else 0,
        "max_5min_exit": int(exit_flow.max()) if exit_flow.size else 0,
        "status": "PASS",
    }


def load_legacy_or_state(out_dir: Path, edges: list[str]) -> tuple[np.ndarray, np.ndarray]:
    inflow_path = out_dir / "inflow.csv"
    outflow_path = out_dir / "outflow.csv"
    if not inflow_path.is_file():
        inflow_path = out_dir / "entry_flow.csv"
    if not outflow_path.is_file():
        outflow_path = out_dir / "exit_flow.csv"
    return read_flow_csv(inflow_path, edges), read_flow_csv(outflow_path, edges)


def process_day(task: tuple) -> dict[str, object]:
    date_label, edges, lane_to_edge, internal_lanes, migrate_legacy, skip_complete, remove_legacy = task
    day_dir = FCD_ROOT / date_label
    out_dir = OUT_ROOT / date_label
    try:
        log(f"[{date_label}] start fcd={day_dir} out={out_dir} migrate={migrate_legacy}")
        if migrate_legacy:
            entry, exit_flow = load_legacy_or_state(out_dir, edges)
            counts = {
                "final_exit": "",
                "unresolved": "",
                "unknown": "",
            }
            vehicles = ""
            audited_in = read_flow_csv(out_dir / "entry_flow.csv", edges)
            audited_out = read_flow_csv(out_dir / "exit_flow.csv", edges)
            if not np.array_equal(entry, audited_in) or not np.array_equal(exit_flow, audited_out):
                raise RuntimeError(f"{date_label} migrated inflow/outflow differs from audited entry/exit")
        else:
            fcd_path = prefer_xml(day_dir, f"fcd_{date_label}")
            trip_path = prefer_xml(day_dir, f"tripinfo_{date_label}")
            arrivals = load_arrivals(trip_path)
            entry, exit_flow, counts, vehicles = stream_fcd(
                fcd_path, edges, lane_to_edge, internal_lanes, arrivals, date_label
            )
        if int(entry.min()) < 0 or int(exit_flow.min()) < 0:
            raise RuntimeError("negative flow")
        if entry.shape != (N_BINS, len(edges)) or exit_flow.shape != entry.shape:
            raise RuntimeError(f"unexpected matrix shape {entry.shape}")
        if int(entry.sum()) <= 0:
            raise RuntimeError("entry flow is empty")
        if skip_complete and states_match(out_dir, entry, exit_flow, edges, date_label):
            log(f"[{date_label}] SKIP verified four-state files")
        else:
            write_state_matrices(out_dir, entry, exit_flow, edges, date_label)
        if remove_legacy:
            remove_legacy_full(out_dir)
        row = summary_row(date_label, entry, exit_flow, vehicles if vehicles != "" else 0)
        row["entry_sums"] = entry.sum(axis=0).astype(np.int64)
        row["exit_sums"] = exit_flow.sum(axis=0).astype(np.int64)
        log(
            f"[{date_label}] PASS entries={row['total_entry_events']} "
            f"exits={row['total_exit_events']} diff={row['entry_exit_difference']} "
            f"final_exits={counts['final_exit']} unresolved={counts['unresolved']} "
            f"unknown_lanes={counts['unknown']} vehicles={vehicles}"
        )
        return row
    except Exception:
        detail = traceback.format_exc(limit=6).strip().replace("\n", " | ")
        log(f"[{date_label}] FAIL {detail}")
        return {
            "date": date_label,
            "unique_vehicles": "",
            "total_entry_events": "",
            "total_exit_events": "",
            "entry_exit_difference": "",
            "entry_exit_relative_difference": "",
            "active_edges_entry": "",
            "active_edges_exit": "",
            "max_5min_entry": "",
            "max_5min_exit": "",
            "status": "FAIL",
            "error": detail,
        }


def write_summary(rows: list[dict[str, object]]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = SUMMARY_PATH.with_suffix(SUMMARY_PATH.suffix + ".tmp")
    ordered = sorted(rows, key=lambda item: str(item["date"]))
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in ordered:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})
    os.replace(temporary, SUMMARY_PATH)


def print_edge_differences(rows: list[dict[str, object]], edges: list[str]) -> None:
    usable = [row for row in rows if row.get("status") == "PASS" and "entry_sums" in row]
    if not usable:
        return
    entry = np.zeros(len(edges), dtype=np.int64)
    exit_flow = np.zeros(len(edges), dtype=np.int64)
    for row in usable:
        entry += row["entry_sums"]
        exit_flow += row["exit_sums"]
    difference = entry - exit_flow
    order = np.argsort(-np.abs(difference))
    log("top entry-exit edge differences across all days:")
    for rank in order[:20]:
        log(
            f"  {edges[rank]} entry={int(entry[rank])} exit={int(exit_flow[rank])} "
            f"diff={int(difference[rank])}"
        )


def self_test() -> None:
    edges = ["A", "B"]
    index = {"A": 0, "B": 1}
    entry = np.zeros((N_BINS, 2), dtype=np.int64)
    exit_flow = np.zeros((N_BINS, 2), dtype=np.int64)
    states: dict[str, Stay] = {}
    seen: set[str] = set()
    counts = {"entry": 0, "exit": 0, "unknown": 0, "unresolved": 0, "final_exit": 0}

    def step(vehicle: str, event_time: float, kind: str, edge: str | None = None) -> None:
        observe(
            states, seen, vehicle, event_time, kind,
            None if edge is None else index[edge], entry, exit_flow, counts,
        )

    step("v1", 100, "real", "A")
    step("v1", 101, "real", "A")
    step("v1", 102, "real", "A")
    step("v1", 103, "internal")
    step("v1", 104, "real", "B")
    step("v1", 105, "real", "B")
    step("v2", 100, "real", "A")
    step("v2", 101, "real", "B")
    step("v2", 250, "real", "A")
    step("v3", 10, "real", "A")
    impute_final_exits(
        states,
        {"v1": 110, "v2": 400, "v3": -1},
        exit_flow,
        counts,
    )
    # v1: enter A, exit A at internal t=103, enter B, exit B at arrival 110.
    # v2: enter A, exit A/enter B at 101, exit B/enter A at 250, exit A at arrival 400.
    # v3: enter A, unfinished, no final exit.
    # A entries are v1, v2's first visit, v2's return, and v3.
    if int(entry[0, 0]) != 4 or int(entry[0, 1]) != 2:
        raise RuntimeError(f"entry self-test failed: {entry[0].tolist()}")
    if int(exit_flow[0, 0]) != 2 or int(exit_flow[0, 1]) != 2 or int(exit_flow[1, 0]) != 1:
        raise RuntimeError(f"exit self-test failed: {exit_flow[0].tolist()} {exit_flow[1].tolist()}")
    if counts["unresolved"] != 1 or counts["final_exit"] != 2:
        raise RuntimeError(f"final-exit self-test failed: {counts}")
    if bin_index(86400) != 287 or bin_index(86399) != 287 or bin_index(300) != 1:
        raise RuntimeError("bin self-test failed")
    sample_in = np.array([[1, 0], [1, 2], [0, 1]], dtype=np.int64)
    sample_out = np.array([[0, 0], [1, 1], [0, 0]], dtype=np.int64)
    sample_start, sample_end = accumulate_states(sample_in, sample_out, ["A", "B"], "self-test")
    if not np.array_equal(sample_start, np.array([[0, 0], [1, 0], [1, 1]])) or not np.array_equal(
        sample_end, np.array([[1, 0], [1, 1], [1, 2]])
    ):
        raise RuntimeError(f"accumulation self-test failed: {sample_start.tolist()} {sample_end.tolist()}")


def discover_dates() -> list[str]:
    dates = []
    for path in FCD_ROOT.iterdir():
        if path.is_dir() and len(path.name) == 10 and path.name[4] == "_" and path.name[7] == "_":
            dates.append(path.name)
    return sorted(dates)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate 5-minute R-level entry and exit flow")
    prep_paths.add_path_arguments(parser, fcd=True, net=True)
    parser.add_argument("--workers", type=int, default=29)
    parser.add_argument("--dates", nargs="*", default=None)
    parser.add_argument("--migrate-legacy", action="store_true")
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--remove-legacy", action="store_true")
    args = parser.parse_args(argv)
    out_root = prep_paths.resolved(args.training_root) / "full_flow" / "r-level"
    prep_paths.bind(
        sys.modules[__name__],
        FCD_ROOT=prep_paths.resolved(args.fcd_root),
        NET_FILE=prep_paths.resolved(args.net_file),
        OUT_ROOT=out_root,
        SUMMARY_PATH=(
            out_root / "state_migration_summary.csv" if args.migrate_legacy else out_root / "generation_summary.csv"
        ),
    )
    log(f"fcd_root={FCD_ROOT}")
    log(f"net_file={NET_FILE}")
    log(f"output_root={OUT_ROOT}")
    prep_paths.publish_runtime(fcd_root=FCD_ROOT, net_file=NET_FILE, training_root=prep_paths.resolved(args.training_root))
    self_test()
    dates = args.dates or discover_dates()
    if not dates:
        log("no date directories")
        return 1
    confirm_fcd_starts_at_zero(prep_paths.resolved(args.fcd_root), dates)
    edges, lane_to_edge, internal_lanes = load_network(NET_FILE)
    log(
        f"edges={len(edges)} real_lanes={len(lane_to_edge)} "
        f"internal_lanes={len(internal_lanes)} dates={len(dates)} "
        f"workers={min(args.workers, len(dates))} read_buffer_GiB={READ_BYTES / 1024**3:.0f}"
    )
    tasks = [
        (date_label, edges, lane_to_edge, internal_lanes, args.migrate_legacy, args.skip_complete, args.remove_legacy)
        for date_label in dates
    ]
    rows: list[dict[str, object]] = []
    workers = max(1, min(args.workers, len(dates)))
    with ProcessPoolExecutor(
        max_workers=workers,
        max_tasks_per_child=1,
        initializer=prep_paths.init_full_flow_worker,
    ) as pool:
        futures = {pool.submit(process_day, task): task[0] for task in tasks}
        for future in as_completed(futures):
            date_label = futures[future]
            try:
                rows.append(future.result())
            except Exception:
                detail = traceback.format_exc(limit=6).strip().replace("\n", " | ")
                log(f"[{date_label}] FAIL {detail}")
                rows.append({"date": date_label, "status": "FAIL"})
    write_summary(rows)
    print_edge_differences(rows, edges)
    passed = sum(1 for row in rows if row.get("status") == "PASS")
    failed = [str(row["date"]) for row in rows if row.get("status") != "PASS"]
    log(f"summary={SUMMARY_PATH} pass={passed} fail={len(failed)} failed={','.join(failed) if failed else 'none'}")
    return 0 if not failed else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
