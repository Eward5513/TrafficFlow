#!/usr/bin/env python3
"""Build four 5-minute Movement-level states from Xuancheng26 FCD.

A Movement is the connector of one legal turn ``(junction, incoming edge,
outgoing edge)`` from ``graph/m_graph.npz``. Every internal (via) lane of the
turn maps to the same Movement. Per vehicle the scan keeps the last real edge
and at most one active Movement:

* first record on one of the Movement's via lanes -> inflow;
* later records on further via lanes of the same Movement -> nothing;
* first record on the outgoing real edge -> outflow;
* incoming edge -> outgoing edge with no via record in between (1 s sampling
  skipped the connector) -> inflow and outflow in the same bin, no stock;
* a vehicle still inside a Movement after the last FCD record leaves it at the
  tripinfo arrival time only when it arrived on that Movement's outgoing edge;
  otherwise it stays unresolved and remains in the stock.

Anything that does not fit (exit onto an unexpected edge, entry from an
unexpected edge, real-edge jumps without a Movement) is counted and written to
``anomalies/<date>.csv``. Accumulation is the running stock of those events:

    N_end[t] = N_start[t] + inflow[t] - outflow[t],  N_start[0] = 0

Outputs: ``<training-root>/full_flow/m-level/<date>/{inflow,outflow,
accumulation_start,accumulation_end}.csv`` with the R-level CSV layout and
Movement columns in ``m_graph.npz`` order.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
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

import generate_m_graph as mgraph
import generate_r_graph as rgraph
import generate_r_level_full_flow as rfull
import prep_paths

N_BINS = rfull.N_BINS
BIN_SECONDS = rfull.BIN_SECONDS
STATE_FILES = rfull.STATE_FILES
# 29 workers x 4 GiB blocks keep about 116 GiB of FCD in flight.
DEFAULT_READ_GIB = 4
AMBIGUOUS = -1
MAX_ANOMALY_ROWS = 200_000

COUNTER_KEYS = (
    "vehicles_seen",
    "fcd_records",
    "inflow_events",
    "outflow_events",
    "via_entries",
    "via_exits",
    "chained_via_exits",
    "direct_transitions",
    "final_exits",
    "mismatched_entries",
    "mismatched_entries_skipped_edge",
    "mismatched_exits",
    "mismatched_exits_skipped_edge",
    "unmapped_transitions",
    "unmapped_transitions_unique_two_hop",
    "first_seen_in_movement",
    "unresolved_vehicles",
    "unresolved_running_at_end",
    "unresolved_other",
    "unknown_lane_records",
    "ambiguous_lane_records",
)

SUMMARY_FIELDS = [
    "date",
    "status",
    "skipped",
    "unique_vehicles",
    "fcd_records",
    "inflow_events",
    "outflow_events",
    "inflow_outflow_difference",
    "end_of_day_stock",
    "via_entries",
    "via_exits",
    "chained_via_exits",
    "direct_transitions",
    "final_exits",
    "mismatched_entries",
    "mismatched_entries_skipped_edge",
    "mismatched_exits",
    "mismatched_exits_skipped_edge",
    "unmapped_transitions",
    "unmapped_transitions_unique_two_hop",
    "first_seen_in_movement",
    "unresolved_vehicles",
    "unresolved_running_at_end",
    "unresolved_other",
    "unknown_lane_records",
    "ambiguous_lane_records",
    "active_movements_inflow",
    "active_movements_outflow",
    "max_5min_inflow",
    "max_5min_outflow",
    "max_accumulation",
    "r_total_inflow",
    "r_total_outflow",
    "conservation_check",
    "elapsed_s",
    "graph_signature",
    "error",
]

ANOMALY_FIELDS = ["time", "vehicle_id", "kind", "movement_id", "observed_edge_id", "last_real_edge_id", "detail"]


def log(message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"{stamp} {message}", flush=True)


@dataclass
class MovementNet:
    movement_ids: list[str]
    edges: list[str]
    m_in: list[int]
    m_out: list[int]
    pair_movement: dict[int, int]
    out_edges: dict[int, list[int]]
    lane_code: dict[str, int]
    lane_to_edge_index: dict[str, int]
    signature: str


def load_movement_net(net_file: Path, graph_path: Path) -> MovementNet:
    """Movement order and via mapping from m_graph.npz, verified against the net."""
    edges, lane_to_edge, internal_lanes = rfull.load_network(net_file)
    with np.load(graph_path, allow_pickle=False) as graph:
        movement_ids = [str(v) for v in graph["movement_ids"].tolist()]
        incoming = [str(v) for v in graph["incoming_edge_ids"].tolist()]
        outgoing = [str(v) for v in graph["outgoing_edge_ids"].tolist()]
        r_edges = [str(v) for v in graph["r_edge_ids"].tolist()]
        lane_ids = [str(v) for v in graph["internal_lane_ids"].tolist()]
        lane_movement = [int(v) for v in graph["internal_lane_movement_index"].tolist()]
        ambiguous = [str(v) for v in graph["ambiguous_internal_lane_ids"].tolist()]
        signature = str(graph["graph_signature"])
    if r_edges != edges:
        raise RuntimeError("m_graph.npz edge order differs from --net-file; rerun generate_m_graph.py")
    if mgraph.graph_signature(movement_ids, incoming, outgoing, lane_ids, lane_movement) != signature:
        raise RuntimeError("m_graph.npz signature does not match its contents")
    _count, links = rgraph.collect_connections(net_file, set(edges))
    if set(zip(incoming, outgoing)) != links or len(links) != len(movement_ids):
        raise RuntimeError("m_graph.npz movements differ from the --net-file connections")
    if set(lane_ids) | set(ambiguous) != set(internal_lanes):
        raise RuntimeError("m_graph.npz via lanes do not cover the --net-file internal lanes")
    position = {edge_id: index for index, edge_id in enumerate(edges)}
    m_in = [position[e] for e in incoming]
    m_out = [position[e] for e in outgoing]
    n_edges = len(edges)
    pair_movement = {a * n_edges + b: m for m, (a, b) in enumerate(zip(m_in, m_out))}
    out_edges: dict[int, list[int]] = {}
    for a, b in zip(m_in, m_out):
        out_edges.setdefault(a, []).append(b)
    lane_code: dict[str, int] = {lane: position[edge] for lane, edge in lane_to_edge.items()}
    for lane, m in zip(lane_ids, lane_movement):
        lane_code[lane] = -2 - m
    for lane in ambiguous:
        lane_code[lane] = AMBIGUOUS
    lane_to_edge_index = {lane: position[edge] for lane, edge in lane_to_edge.items()}
    return MovementNet(movement_ids, edges, m_in, m_out, pair_movement, out_edges, lane_code, lane_to_edge_index, signature)


def load_arrival_lanes(path: Path, lane_to_edge_index: dict[str, int]) -> tuple[dict[str, tuple[float, int]], set[str]]:
    """Arrival time and arrival edge of finished trips; ids still running at the end."""
    arrivals: dict[str, tuple[float, int]] = {}
    running: set[str] = set()

    def start(name: str, attrs) -> None:
        if name != "tripinfo":
            return
        vehicle_id = attrs.get("id")
        if not vehicle_id:
            return
        try:
            arrival = float(attrs.get("arrival", "-1"))
            depart = float(attrs.get("depart", "-1"))
        except ValueError:
            return
        if depart < 0:
            return
        if arrival >= 0:
            arrivals[vehicle_id] = (arrival, lane_to_edge_index.get(attrs.get("arrivalLane", ""), -1))
        else:
            running.add(vehicle_id)

    parser = expat.ParserCreate()
    parser.StartElementHandler = start
    for block in rfull.read_chunks(path):
        parser.Parse(block, False)
    parser.Parse(b"", True)
    return arrivals, running


class Tracker:
    """Per-vehicle Movement state machine shared by the full and partial scans.

    ``level`` is the lowest nested rate a vehicle belongs to; every event is
    added to that level and all higher ones. The full scan uses one level.
    """

    def __init__(self, mnet: MovementNet, n_levels: int = 1, keep_anomalies: bool = True) -> None:
        self.mnet = mnet
        self.n_levels = n_levels
        size = len(mnet.movement_ids)
        self.inflow = np.zeros((n_levels, N_BINS, size), dtype=np.int64)
        self.outflow = np.zeros((n_levels, N_BINS, size), dtype=np.int64)
        self.counts = {key: np.zeros(n_levels, dtype=np.int64) for key in COUNTER_KEYS}
        self.states: dict[str, list] = {}
        self.anomalies: list[list[object]] = []
        self.keep_anomalies = keep_anomalies
        self.unknown_lanes: set[str] = set()
        self.observe = self._build_observe()

    def _note(self, kind: str, level: int, when: float, vehicle_id: str, movement: int, edge: int, last: int, detail: str) -> None:
        if not self.keep_anomalies or len(self.anomalies) >= MAX_ANOMALY_ROWS:
            return
        mnet = self.mnet
        self.anomalies.append([
            f"{when:.2f}",
            vehicle_id,
            kind,
            mnet.movement_ids[movement] if movement >= 0 else "",
            mnet.edges[edge] if edge >= 0 else "",
            mnet.edges[last] if last >= 0 else "",
            detail,
        ])

    def _build_observe(self):
        mnet = self.mnet
        lane_code = mnet.lane_code
        m_in, m_out = mnet.m_in, mnet.m_out
        pairs = mnet.pair_movement
        out_edges = mnet.out_edges
        n_edges = len(mnet.edges)
        inflow, outflow = self.inflow, self.outflow
        counts = self.counts
        c_in, c_out = counts["inflow_events"], counts["outflow_events"]
        states = self.states
        note = self._note

        def enter(movement: int, slot: int, level: int, key: str) -> None:
            inflow[level:, slot, movement] += 1
            c_in[level:] += 1
            counts[key][level:] += 1

        def leave(movement: int, slot: int, level: int, key: str | None) -> None:
            outflow[level:, slot, movement] += 1
            c_out[level:] += 1
            if key is not None:
                counts[key][level:] += 1

        def observe(vehicle_id: str, lane: str, when: float, slot: int, level: int) -> None:
            code = lane_code.get(lane)
            if code is None:
                counts["unknown_lane_records"][level:] += 1
                if len(self.unknown_lanes) < 50:
                    self.unknown_lanes.add(lane)
                return
            state = states.get(vehicle_id)
            if code >= 0:
                if state is None:
                    states[vehicle_id] = [code, -1, when, level]
                    counts["vehicles_seen"][level:] += 1
                    return
                active = state[1]
                if active >= 0:
                    if code == m_out[active]:
                        leave(active, slot, level, "via_exits")
                    else:
                        leave(active, slot, level, "mismatched_exits")
                        skipped = (m_out[active] * n_edges + code) in pairs
                        if skipped:
                            counts["mismatched_exits_skipped_edge"][level:] += 1
                        note("mismatched_exit", level, when, vehicle_id, active, code, state[0],
                             "outgoing_edge_skipped" if skipped else "unexpected_edge")
                    state[0] = code
                    state[1] = -1
                    return
                last = state[0]
                if code == last:
                    return
                movement = pairs.get(last * n_edges + code)
                if movement is None:
                    counts["unmapped_transitions"][level:] += 1
                    middles = [b for b in out_edges.get(last, ()) if (b * n_edges + code) in pairs]
                    if len(middles) == 1:
                        counts["unmapped_transitions_unique_two_hop"][level:] += 1
                    note("unmapped_transition", level, when, vehicle_id, -1, code, last, f"two_hop_paths={len(middles)}")
                else:
                    enter(movement, slot, level, "direct_transitions")
                    leave(movement, slot, level, None)
                state[0] = code
                return
            if code == AMBIGUOUS:
                counts["ambiguous_lane_records"][level:] += 1
                return
            movement = -2 - code
            if state is None:
                states[vehicle_id] = [m_in[movement], movement, when, level]
                counts["vehicles_seen"][level:] += 1
                enter(movement, slot, level, "first_seen_in_movement")
                note("first_seen_in_movement", level, when, vehicle_id, movement, -1, -1, "")
                return
            active = state[1]
            if active == movement:
                return
            if active >= 0:
                if m_out[active] == m_in[movement]:
                    leave(active, slot, level, "chained_via_exits")
                else:
                    leave(active, slot, level, "mismatched_exits")
                    note("mismatched_exit", level, when, vehicle_id, active, m_in[movement], state[0], "entered_unrelated_movement")
                enter(movement, slot, level, "via_entries")
            elif state[0] == m_in[movement]:
                enter(movement, slot, level, "via_entries")
            else:
                enter(movement, slot, level, "mismatched_entries")
                skipped = (state[0] * n_edges + m_in[movement]) in pairs
                if skipped:
                    counts["mismatched_entries_skipped_edge"][level:] += 1
                note("mismatched_entry", level, when, vehicle_id, movement, m_in[movement], state[0],
                     "incoming_edge_skipped" if skipped else "unexpected_predecessor")
            state[0] = m_in[movement]
            state[1] = movement
            state[2] = when

        return observe

    def finish(self, arrivals: dict[str, tuple[float, int]], running: set[str]) -> None:
        """Close Movements still active after the last FCD record."""
        m_out = self.mnet.m_out
        counts = self.counts
        for vehicle_id, state in self.states.items():
            active = state[1]
            if active < 0:
                continue
            level = state[3]
            arrival = arrivals.get(vehicle_id)
            if arrival is not None and arrival[1] == m_out[active] and arrival[0] >= state[2]:
                slot = rfull.bin_index(arrival[0])
                self.outflow[level:, slot, active] += 1
                counts["outflow_events"][level:] += 1
                counts["final_exits"][level:] += 1
                state[1] = -1
                continue
            counts["unresolved_vehicles"][level:] += 1
            if vehicle_id in running:
                counts["unresolved_running_at_end"][level:] += 1
                reason = "running_at_simulation_end"
            else:
                counts["unresolved_other"][level:] += 1
                if arrival is None:
                    reason = "no_tripinfo_arrival"
                elif arrival[1] != m_out[active]:
                    reason = "arrival_edge_differs"
                else:
                    reason = "arrival_before_entry"
            arrival_edge = arrival[1] if arrival is not None else -1
            self._note("unresolved", level, state[2], vehicle_id, active, arrival_edge, state[0], reason)

    def identities(self, level: int) -> list[str]:
        """Event-category bookkeeping that must hold exactly."""
        c = {key: int(value[level]) for key, value in self.counts.items()}
        problems = []
        if c["inflow_events"] != c["via_entries"] + c["mismatched_entries"] + c["first_seen_in_movement"] + c["direct_transitions"]:
            problems.append("inflow categories do not add up")
        if c["outflow_events"] != (
            c["via_exits"] + c["chained_via_exits"] + c["mismatched_exits"] + c["direct_transitions"] + c["final_exits"]
        ):
            problems.append("outflow categories do not add up")
        if int(self.inflow[level].sum()) != c["inflow_events"] or int(self.outflow[level].sum()) != c["outflow_events"]:
            problems.append("matrix totals differ from event counters")
        if c["inflow_events"] - c["outflow_events"] != c["unresolved_vehicles"]:
            problems.append("end-of-day stock differs from unresolved vehicles")
        return problems


def stream_movements(
    fcd_path: Path,
    mnet: MovementNet,
    arrivals: dict[str, tuple[float, int]],
    running: set[str],
    date_label: str,
    *,
    level_of: dict[str, int] | None = None,
    n_levels: int = 1,
    keep_anomalies: bool = True,
) -> Tracker:
    tracker = Tracker(mnet, n_levels, keep_anomalies)
    observe = tracker.observe
    records = tracker.counts["fcd_records"]
    clock = [0.0, 0]
    seen_records = [0]

    def start(name: str, attrs) -> None:
        if name == "vehicle":
            seen_records[0] += 1
            vehicle_id = attrs.get("id")
            if not vehicle_id:
                return
            if level_of is None:
                level = 0
            else:
                level = level_of.get(vehicle_id)
                if level is None:
                    return
            observe(vehicle_id, attrs.get("lane", ""), clock[0], clock[1], level)
        elif name == "timestep":
            when = float(attrs.get("time", "0"))
            slot = rfull.bin_index(when)
            if slot is None:
                raise RuntimeError(f"negative FCD time {when}")
            clock[0] = when
            clock[1] = slot

    parser = expat.ParserCreate()
    parser.StartElementHandler = start
    read_bytes = 0
    next_report = 4 << 30
    for block in rfull.read_chunks(fcd_path):
        read_bytes += len(block)
        parser.Parse(block, False)
        if read_bytes >= next_report:
            log(
                f"[{date_label}] fcd {read_bytes / 1024**3:.1f} GiB t={clock[0]:.0f} "
                f"records={seen_records[0]} inflow={int(tracker.counts['inflow_events'][-1])}"
            )
            next_report += 4 << 30
    parser.Parse(b"", True)
    records[:] = seen_records[0]
    tracker.finish(arrivals, running)
    return tracker


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_rows(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(temporary, path)


def read_states(out_dir: Path, ids: list[str]) -> dict[str, np.ndarray] | None:
    """Reread the four files and confirm the stock recursion. None if absent or invalid."""
    paths = {name: out_dir / name for name in STATE_FILES}
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths.values()):
        return None
    try:
        values = {name: rfull.read_flow_csv(path, ids) for name, path in paths.items()}
        start, end = rfull.accumulate_states(values["inflow.csv"], values["outflow.csv"], ids, str(out_dir))
    except Exception:
        return None
    if not np.array_equal(start, values["accumulation_start.csv"]) or not np.array_equal(end, values["accumulation_end.csv"]):
        return None
    return values


def r_totals(r_dir: Path, edges: list[str]) -> tuple[object, object]:
    try:
        return (
            int(rfull.read_flow_csv(r_dir / "inflow.csv", edges).sum()),
            int(rfull.read_flow_csv(r_dir / "outflow.csv", edges).sum()),
        )
    except Exception:
        return "", ""


def failed_row(date_label: str, detail: str) -> dict[str, object]:
    return {"date": date_label, "status": "FAIL", "skipped": "", "error": detail}


def process_day(task: dict) -> dict[str, object]:
    date_label = task["date"]
    mnet: MovementNet = task["mnet"]
    ids = mnet.movement_ids
    out_dir = Path(task["out_root"]) / date_label
    stats_path = Path(task["out_root"]) / "day_stats" / f"{date_label}.json"
    anomaly_path = Path(task["out_root"]) / "anomalies" / f"{date_label}.csv"
    rfull.READ_BYTES = int(task["read_bytes"])
    began = time.time()
    try:
        if task["skip_complete"] and stats_path.is_file():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            if stats.get("status") == "PASS" and stats.get("graph_signature") == mnet.signature:
                values = read_states(out_dir, ids)
                if (
                    values is not None
                    and int(values["inflow.csv"].sum()) == int(stats["inflow_events"])
                    and int(values["outflow.csv"].sum()) == int(stats["outflow_events"])
                ):
                    log(f"[{date_label}] SKIP verified four-state files")
                    stats["skipped"] = "yes"
                    return stats
        if stats_path.is_file():
            stats_path.unlink()
        day_dir = Path(task["fcd_root"]) / date_label
        fcd_path = rfull.prefer_xml(day_dir, f"fcd_{date_label}")
        trip_path = rfull.prefer_xml(day_dir, f"tripinfo_{date_label}")
        log(f"[{date_label}] start fcd={fcd_path} out={out_dir}")
        arrivals, running = load_arrival_lanes(trip_path, mnet.lane_to_edge_index)
        tracker = stream_movements(fcd_path, mnet, arrivals, running, date_label)
        inflow, outflow = tracker.inflow[0], tracker.outflow[0]
        if inflow.shape != (N_BINS, len(ids)) or outflow.shape != inflow.shape:
            raise RuntimeError(f"unexpected matrix shape {inflow.shape}")
        if int(inflow.min()) < 0 or int(outflow.min()) < 0:
            raise RuntimeError("negative flow")
        if int(inflow.sum()) <= 0:
            raise RuntimeError("inflow is empty")
        problems = tracker.identities(0)
        start, end = rfull.accumulate_states(inflow, outflow, ids, date_label)
        if int(end[-1].sum()) != int(tracker.counts["unresolved_vehicles"][0]):
            problems.append("last accumulation_end differs from unresolved vehicles")
        if problems:
            raise RuntimeError("; ".join(problems))
        rfull.write_state_matrices(out_dir, inflow, outflow, ids, date_label)
        write_rows(anomaly_path, ANOMALY_FIELDS, tracker.anomalies)
        r_in, r_out = r_totals(Path(task["r_full_root"]) / date_label, mnet.edges)
        c = {key: int(value[0]) for key, value in tracker.counts.items()}
        row = {
            "date": date_label,
            "status": "PASS",
            "skipped": "no",
            "unique_vehicles": c["vehicles_seen"],
            **{key: c[key] for key in COUNTER_KEYS if key != "vehicles_seen"},
            "inflow_outflow_difference": c["inflow_events"] - c["outflow_events"],
            "end_of_day_stock": int(end[-1].sum()),
            "active_movements_inflow": int((inflow.sum(axis=0) > 0).sum()),
            "active_movements_outflow": int((outflow.sum(axis=0) > 0).sum()),
            "max_5min_inflow": int(inflow.max()),
            "max_5min_outflow": int(outflow.max()),
            "max_accumulation": int(max(start.max(), end.max())),
            "r_total_inflow": r_in,
            "r_total_outflow": r_out,
            "conservation_check": "PASS",
            "elapsed_s": round(time.time() - began, 1),
            "graph_signature": mnet.signature,
            "unknown_lane_examples": sorted(tracker.unknown_lanes),
            "anomaly_rows_written": len(tracker.anomalies),
            "error": "",
        }
        write_json(stats_path, row)
        log(
            f"[{date_label}] PASS in={c['inflow_events']} out={c['outflow_events']} via_in={c['via_entries']} "
            f"direct={c['direct_transitions']} final={c['final_exits']} unresolved={c['unresolved_vehicles']} "
            f"mismatch_in={c['mismatched_entries']} mismatch_out={c['mismatched_exits']} "
            f"unmapped={c['unmapped_transitions']} unknown={c['unknown_lane_records']} "
            f"elapsed={row['elapsed_s']}s"
        )
        return row
    except Exception:
        detail = traceback.format_exc(limit=8).strip().replace("\n", " | ")
        log(f"[{date_label}] FAIL {detail}")
        return failed_row(date_label, detail)


def discover_dates(fcd_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in fcd_root.iterdir()
        if path.is_dir() and len(path.name) == 10 and path.name[4] == "_" and path.name[7] == "_"
    )


def collect_summary(rows: list[dict[str, object]], stats_dir: Path, dates: list[str]) -> list[dict[str, object]]:
    """This run's rows, plus earlier day_stats for dates not processed now."""
    by_date = {str(row["date"]): row for row in rows}
    for date_label in dates:
        if date_label in by_date:
            continue
        path = stats_dir / f"{date_label}.json"
        if path.is_file():
            previous = json.loads(path.read_text(encoding="utf-8"))
            previous["skipped"] = "not_run"
            by_date[date_label] = previous
    return [by_date[key] for key in sorted(by_date)]


def write_summary(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(temporary, path)


def self_test() -> None:
    edges = ["A", "B", "C"]
    lanes = {"A_0": 0, "B_0": 1, "C_0": 2, ":a_0": -2, ":a_1": -2, ":b_0": -3}
    mnet = MovementNet(
        movement_ids=["M::j1::A::B", "M::j2::B::C"],
        edges=edges,
        m_in=[0, 1],
        m_out=[1, 2],
        pair_movement={0 * 3 + 1: 0, 1 * 3 + 2: 1},
        out_edges={0: [1], 1: [2]},
        lane_code=lanes,
        lane_to_edge_index={"A_0": 0, "B_0": 1, "C_0": 2},
        signature="self-test",
    )
    tracker = Tracker(mnet, n_levels=2)
    level = {"v1": 0, "v2": 1, "v3": 0, "v4": 0, "v5": 0, "v6": 1, "v7": 0}
    script = [
        ("v1", ["A_0", "A_0", ":a_0", ":a_1", "B_0", "B_0", ":b_0", "C_0"], 0),
        ("v2", ["A_0", "B_0"], 10),
        ("v3", ["A_0", ":a_0"], 20),
        ("v4", ["A_0", ":a_1"], 40),
        ("v5", ["A_0", ":a_0", ":b_0", "C_0"], 50),
        ("v6", ["A_0", "C_0"], 60),
        ("v7", ["A_0", ":a_0", "A_0"], 70),
    ]
    for vehicle_id, path, begin in script:
        for offset, lane in enumerate(path):
            when = float(begin + offset)
            tracker.observe(vehicle_id, lane, when, rfull.bin_index(when), level[vehicle_id])
    tracker.finish({"v3": (400.0, 1), "v1": (8.0, 2)}, {"v4"})
    full_in, full_out = tracker.inflow[1], tracker.outflow[1]
    # m0 = A->B: v1 via, v2 direct, v3 via + final exit at 400, v4 unresolved, v5 chained, v7 mismatched.
    if int(full_in[0, 0]) != 6 or int(full_out[0, 0]) != 4 or int(full_out[1, 0]) != 1:
        raise RuntimeError(f"self-test m0 failed {full_in[:2, 0].tolist()} {full_out[:2, 0].tolist()}")
    # m1 = B->C: v1 via, v5 via after chained exit.
    if int(full_in[0, 1]) != 2 or int(full_out[0, 1]) != 2:
        raise RuntimeError(f"self-test m1 failed {full_in[0, 1]} {full_out[0, 1]}")
    c = {key: int(value[1]) for key, value in tracker.counts.items()}
    expected = {
        "direct_transitions": 1, "final_exits": 1, "unresolved_vehicles": 1, "unresolved_running_at_end": 1,
        "chained_via_exits": 1, "mismatched_exits": 1, "unmapped_transitions": 1, "via_entries": 7,
        "via_exits": 3, "vehicles_seen": 7, "unmapped_transitions_unique_two_hop": 1,
    }
    for key, value in expected.items():
        if c[key] != value:
            raise RuntimeError(f"self-test counter {key}={c[key]} expected {value}")
    if int(tracker.inflow[0].sum()) != 7 or int(tracker.counts["vehicles_seen"][0]) != 5:
        raise RuntimeError("self-test level split failed")
    for lvl in (0, 1):
        if tracker.identities(lvl):
            raise RuntimeError(f"self-test identities failed {tracker.identities(lvl)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate 5-minute M-level four-state flow from FCD")
    prep_paths.add_3s_path_arguments(parser, fcd=True, net=True)
    parser.add_argument("--workers", type=int, default=29)
    parser.add_argument("--dates", nargs="*", default=None)
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--read-buffer-gib", type=float, default=DEFAULT_READ_GIB)
    args = parser.parse_args(argv)
    root = prep_paths.refuse_code_root(args.training_root)
    fcd_root = prep_paths.resolved(args.fcd_root)
    net_file = prep_paths.resolved(args.net_file)
    graph_path = root / "graph" / "m_graph.npz"
    out_root = root / "full_flow" / "m-level"
    summary_path = out_root / "generation_summary.csv"
    log(f"fcd_root={fcd_root}")
    log(f"net_file={net_file}")
    log(f"graph={graph_path}")
    log(f"output_root={out_root}")
    self_test()
    all_dates = discover_dates(fcd_root)
    dates = args.dates or all_dates
    missing = [d for d in dates if d not in all_dates]
    if not dates or missing:
        log(f"no usable dates; missing={missing}")
        return 1
    rfull.confirm_fcd_starts_at_zero(fcd_root, dates)
    mnet = load_movement_net(net_file, graph_path)
    workers = max(1, min(args.workers, len(dates)))
    read_bytes = int(args.read_buffer_gib * (1 << 30))
    log(
        f"movements={len(mnet.movement_ids)} edges={len(mnet.edges)} lanes={len(mnet.lane_code)} "
        f"dates={len(dates)} workers={workers} read_buffer_GiB={args.read_buffer_gib:g} signature={mnet.signature[:12]}"
    )
    tasks = [
        {
            "date": date_label,
            "mnet": mnet,
            "fcd_root": str(fcd_root),
            "out_root": str(out_root),
            "r_full_root": str(root / "full_flow" / "r-level"),
            "skip_complete": args.skip_complete,
            "read_bytes": read_bytes,
        }
        for date_label in dates
    ]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
        futures = {pool.submit(process_day, task): task["date"] for task in tasks}
        for future in as_completed(futures):
            date_label = futures[future]
            try:
                rows.append(future.result())
            except Exception:
                detail = traceback.format_exc(limit=6).strip().replace("\n", " | ")
                log(f"[{date_label}] FAIL {detail}")
                rows.append(failed_row(date_label, detail))
    summary_rows = collect_summary(rows, out_root / "day_stats", all_dates)
    write_summary(summary_path, SUMMARY_FIELDS, summary_rows)
    failed = sorted(str(row["date"]) for row in rows if row.get("status") != "PASS")
    passed_all = [row for row in summary_rows if row.get("status") == "PASS"]
    complete = [d for d in all_dates if (out_root / d).is_dir() and all((out_root / d / n).is_file() for n in STATE_FILES)]
    log(
        f"summary={summary_path} run_pass={len(rows) - len(failed)} run_fail={len(failed)} "
        f"failed={','.join(failed) if failed else 'none'} summary_pass={len(passed_all)}/{len(all_dates)} "
        f"complete_dirs={len(complete)}"
    )
    if passed_all:
        totals = {key: sum(int(row[key]) for row in passed_all) for key in (
            "inflow_events", "outflow_events", "via_entries", "direct_transitions", "final_exits",
            "unresolved_vehicles", "mismatched_entries", "mismatched_exits", "unmapped_transitions",
            "unknown_lane_records", "ambiguous_lane_records",
        )}
        log("totals " + " ".join(f"{k}={v}" for k, v in totals.items()))
    return 0 if not failed else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
