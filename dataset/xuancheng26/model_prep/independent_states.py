#!/usr/bin/env python3
"""Independent FCD counts for the four Xuancheng26 road states.

``independent_fcd_v1`` counts inflow and outflow from observed edge events and
counts both accumulations from boundary snapshots. It does not build stock
with ``cumsum(inflow - outflow)``. A nonzero residual is recorded, not repaired.

Bins are ``[300k, 300(k+1))``. ``Q(b-)`` is the occupancy of real edges at the
latest complete 1-second FCD sample strictly before ``b``. ``Q(0-)`` is the
empty network before the first step. The sample at ``t=0.00`` is after
departures at time 0, so it is not ``Q(0-)``. There is no ``t=86400`` sample;
``Q(86400-)`` is the sample at ``t=86399``.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
from pathlib import Path

import numpy as np

PHYSICAL_VERSION = "independent_fcd_v1"
SAMPLING_ALGORITHM_VERSION = "xuancheng26_variable_penetration_v1"
FIXED_SAMPLING_VERSION = "xuancheng26_3s_multi_seed_v2"
N_BINS = 288
BIN_SECONDS = 300
DAY_SECONDS = 86400
N_BOUNDARIES = 288  # Q(300-), Q(600-), ..., Q(86400-)
CHANNELS = ("inflow", "outflow", "accumulation_start", "accumulation_end")

# kind codes for one observed sample
REAL = 0
INTERNAL = 1
UNKNOWN = 2


def definition_document() -> dict:
    """The version written to ``shared/physical_definition.json``."""
    return {
        "physical_definition_version": PHYSICAL_VERSION,
        "sampling_algorithm_version": SAMPLING_ALGORITHM_VERSION,
        "fixed_sampling_algorithm_version": FIXED_SAMPLING_VERSION,
        "bin": {
            "seconds": BIN_SECONDS,
            "count": N_BINS,
            "interval": "[300k, 300(k+1))",
            "day_seconds": DAY_SECONDS,
        },
        "simulation_clock": {
            "begin": 0,
            "end": 86400,
            "step_length_seconds": 1,
            "fcd_period_seconds": 1,
            "expected_samples": list_note_expected_samples(),
            "q0_minus": (
                "Zero. SUMO begins at t=0 with an empty network and no preloaded "
                "vehicles. The FCD sample at t=0.00 is the state after departures "
                "at time 0, so it is not Q(0-)."
            ),
            "q_day_end": (
                "Q(86400-) is the occupancy at t=86399. No t=86400 snapshot is "
                "invented, copied, or filled from the flow balance."
            ),
        },
        "inflow": (
            "Direct count of a vehicle entering a real edge in the 5-minute bin. "
            "Includes a departure or other first observation on that edge, a move "
            "from another real edge, and a return after the vehicle was observed "
            "off the edge. This is a source-inclusive entry count, not a pure "
            "upstream cordon count."
        ),
        "outflow": (
            "Direct count of a vehicle leaving a real edge: move to another real "
            "edge, observation on an internal lane, observation on an unknown lane, "
            "or a tripinfo arrival. Unfinished trips are not given an arrival."
        ),
        "accumulation_start": "Q(300k-): real-edge occupancy at t=300k-1 for k>0, and 0 at k=0.",
        "accumulation_end": "Q(300(k+1)-): real-edge occupancy at t=300(k+1)-1.",
        "occupancy": (
            "A vehicle counts on the real edge of its lane at that exact sample. "
            "Internal lanes and unknown lanes are not kept on the previous external edge."
        ),
        "event_classes": [
            "departure_entry",
            "first_seen_entry",
            "transition_exit",
            "transition_entry",
            "reentry_entry",
            "reentry_same_edge",
            "internal_exit",
            "unknown_exit",
            "arrival_exit",
        ],
        "arrival_rule": (
            "impute_final_exits adds an outflow only when tripinfo has a finite "
            "arrival at or after depart. A missing arrival stays unresolved."
        ),
        "residual": "accumulation_end - accumulation_start - inflow + outflow. Diagnostic only.",
        "partial_rule": "The same events and snapshots, restricted to a fixed vehicle list. Never full * p.",
        "legacy": "accumulate_states in generate_r_level_full_flow.py is the retired cumsum stock. New reads and writes do not call it.",
    }


def list_note_expected_samples() -> str:
    return "integer seconds 0,1,...,86399; a missing second is an error and is not forward-filled"


def definition_sha256() -> str:
    import hashlib

    payload = json.dumps(definition_document(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def boundary_time(index: int) -> int:
    """Sample time for ``Q(300(index+1)-)``."""
    if index < 0 or index >= N_BOUNDARIES:
        raise IndexError(index)
    return (index + 1) * BIN_SECONDS - 1


def bin_index(event_time: float) -> int | None:
    if event_time < 0:
        return None
    if event_time >= DAY_SECONDS:
        return N_BINS - 1
    return int(event_time // BIN_SECONDS)


def reduce_observation(
    edge: int | None,
    exited: bool,
    kind: int,
    column: int | None,
    slot: int | None,
) -> tuple[int | None, bool, list[tuple[str, int, int]]]:
    """One sample. ``edge is None`` means the vehicle has not yet been on a real edge.

    Returns the new edge, the exited flag, and classified events
    ``(class, slot, edge_index)``. Same-edge continuation emits nothing.
    A return to the same edge after an exit is a new entry.
    """
    if kind == UNKNOWN:
        events: list[tuple[str, int, int]] = []
        if edge is not None and not exited and slot is not None:
            events.append(("unknown_exit", slot, edge))
            exited = True
        return edge, exited, events
    if kind == INTERNAL:
        events = []
        if edge is not None and not exited and slot is not None:
            events.append(("internal_exit", slot, edge))
            exited = True
        return edge, exited, events
    if column is None or slot is None:
        return edge, exited, []
    if edge is None:
        return column, False, [("departure_entry", slot, column)]
    if column == edge and not exited:
        return edge, False, []
    events = []
    if column == edge and exited:
        events.append(("reentry_same_edge", slot, column))
        return column, False, events
    if not exited:
        events.append(("transition_exit", slot, edge))
        events.append(("transition_entry", slot, column))
    else:
        events.append(("reentry_entry", slot, column))
    return column, False, events


def assemble_accumulation(end_occupancy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``end_occupancy[k]`` is ``Q(300(k+1)-)``. Start follows the previous boundary."""
    if end_occupancy.shape[0] != N_BINS:
        raise RuntimeError(f"expected {N_BINS} boundary snapshots, got {end_occupancy.shape}")
    end = np.asarray(end_occupancy, dtype=np.int64)
    start = np.zeros_like(end)
    start[1:] = end[:-1]
    return start, end


def residual(start: np.ndarray, end: np.ndarray, inflow: np.ndarray, outflow: np.ndarray) -> np.ndarray:
    return end.astype(np.int64) - start.astype(np.int64) - inflow.astype(np.int64) + outflow.astype(np.int64)


def summarize_residual(values: np.ndarray) -> dict:
    flat = np.asarray(values, dtype=np.int64).reshape(-1)
    nonzero = flat[flat != 0]
    abs_values = np.abs(nonzero) if nonzero.size else nonzero
    percentiles = [0, 50, 90, 99, 100]
    return {
        "cells": int(flat.size),
        "nonzero_cells": int(nonzero.size),
        "nonzero_fraction": float(nonzero.size / flat.size) if flat.size else 0.0,
        "positive_cells": int((flat > 0).sum()),
        "negative_cells": int((flat < 0).sum()),
        "max_abs": int(np.abs(flat).max()) if flat.size else 0,
        "abs_percentiles": (
            [int(np.percentile(abs_values, q)) for q in percentiles] if abs_values.size else [0, 0, 0, 0, 0]
        ),
    }


def self_test() -> None:
    """Hand-counted trajectories, including the snapshot counterexample."""
    # A -> internal -> A must count the return. Continuation on A must not.
    edge, exited, events = reduce_observation(None, False, REAL, 0, 0)
    assert (edge, exited, events) == (0, False, [("departure_entry", 0, 0)])
    edge, exited, events = reduce_observation(edge, exited, REAL, 0, 0)
    assert events == []
    edge, exited, events = reduce_observation(edge, exited, INTERNAL, None, 0)
    assert events == [("internal_exit", 0, 0)] and exited
    edge, exited, events = reduce_observation(edge, exited, REAL, 0, 0)
    assert events == [("reentry_same_edge", 0, 0)] and not exited
    edge, exited, events = reduce_observation(edge, exited, REAL, 1, 1)
    assert events == [("transition_exit", 1, 0), ("transition_entry", 1, 1)]
    edge, exited, events = reduce_observation(edge, exited, UNKNOWN, None, 1)
    assert events == [("unknown_exit", 1, 1)] and exited
    # Changing only the snapshot changes accumulation while events stay put.
    inflow = np.zeros((N_BINS, 2), dtype=np.int64)
    outflow = np.zeros_like(inflow)
    inflow[0, 0] = 1
    outflow[0, 0] = 1
    present = np.zeros((N_BINS, 2), dtype=np.int64)
    present[0, 0] = 1
    start, end = assemble_accumulation(present)
    other = present.copy()
    other[0, 0] = 0
    start2, end2 = assemble_accumulation(other)
    assert not np.array_equal(end, end2)
    assert np.array_equal(inflow, inflow) and int(residual(start, end, inflow, outflow)[0, 0]) == 1
    assert int(residual(start2, end2, inflow, outflow)[0, 0]) == 0
    # Adjacent boundaries share one sample: end[k] == start[k+1].
    assert np.array_equal(end[:-1], start[1:])
    # Q(0-) is zero even if the first boundary sample is not.
    assert int(start[0].sum()) == 0 and int(end[0, 0]) == 1
    if boundary_time(0) != 299 or boundary_time(287) != 86399:
        raise RuntimeError("boundary clock drifted")
    if bin_index(299) != 0 or bin_index(300) != 1 or bin_index(86399) != 287 or bin_index(86400) != 287:
        raise RuntimeError("event bins drifted")


def _load_id_mask(paths: list[Path]) -> dict[str, int]:
    from vehicle_sampling import read_ids

    masks: dict[str, int] = {}
    if len(paths) > 63:
        raise RuntimeError(f"group mask does not fit in 64 bits: {len(paths)}")
    for index, path in enumerate(paths):
        bit = 1 << index
        for vehicle_id in read_ids(path):
            masks[vehicle_id] = masks.get(vehicle_id, 0) | bit
    return masks


def _add(full: np.ndarray, partial: np.ndarray, slot: int, column: int, mask: int) -> None:
    full[slot, column] += 1
    while mask:
        bit = mask & -mask
        partial[bit.bit_length() - 1, slot, column] += 1
        mask -= bit


def scan_groups(task: dict) -> dict:
    """One gzip FCD pass. ``task['groups']`` are partial lists; full is everyone."""
    import traceback
    from xml.parsers import expat

    import generate_r_level_full_flow as full

    self_test()
    date = task["date"]
    try:
        edges = task["edges"]
        lane_to_edge = task["lane_to_edge"]
        internal = task["internal"]
        edge_index = task["edge_index"]
        n_edges = len(edges)
        groups = task["groups"]
        masks = _load_id_mask([Path(item["members"]) for item in groups])
        from variable_penetration import population_from_tripinfo

        population = population_from_tripinfo(Path(task["tripinfo"]))
        if population["conflicts"]:
            raise RuntimeError(f"{date} tripinfo conflicts {population['conflicts']} {population['counts']}")
        departs = population["depart"]
        population_ids = population["all_ids"]
        arrivals = full.load_arrivals(Path(task["tripinfo"]))
        missing_tripinfo: set[str] = set()
        fcd_path = Path(task["fcd"])
        full_in = np.zeros((N_BINS, n_edges), dtype=np.int64)
        full_out = np.zeros_like(full_in)
        full_occ = np.zeros_like(full_in)
        partial_in = np.zeros((len(groups), N_BINS, n_edges), dtype=np.int64)
        partial_out = np.zeros_like(partial_in)
        partial_occ = np.zeros_like(partial_in)
        class_bins = {
            name: np.zeros(N_BINS, dtype=np.int64)
            for name in (
                "departure_entry", "first_seen_entry", "transition_exit", "transition_entry",
                "reentry_entry", "reentry_same_edge", "internal_exit", "unknown_exit", "arrival_exit",
            )
        }
        states: dict[str, list] = {}
        counts = {name: 0 for name in (
            "records", "departure_entry", "first_seen_entry", "transition_exit", "transition_entry",
            "reentry_entry", "reentry_same_edge", "internal_exit", "unknown_exit", "unknown_samples",
            "arrival_exit", "unresolved", "observation_gaps", "arrival_at_or_after_86400",
            "empty_timesteps", "vehicles_at_0",
        )}
        examples: dict[str, list] = {name: [] for name in (
            "reentry_same_edge", "unknown_exit", "unresolved", "observation_gaps", "unknown_lane",
        )}
        unknown_lanes: dict[str, int] = {}
        clock = {"time": -1, "snap": False, "vehicles": 0}
        seen_times = 0
        gaps: list[list[int]] = []
        snap_edge: list[int] = []
        snap_mask: list[int] = []
        boundary_of = {boundary_time(index): index for index in range(N_BOUNDARIES)}
        snapped = np.zeros(N_BOUNDARIES, dtype=np.int8)

        def note(kind: str, slot: int, column: int, vehicle_id: str, mask: int, into: str) -> None:
            if into == "in":
                _add(full_in, partial_in, slot, column, mask)
            else:
                _add(full_out, partial_out, slot, column, mask)
            counts[kind] += 1
            class_bins[kind][slot] += 1
            bucket = examples.get(kind)
            if bucket is not None and len(bucket) < 8:
                bucket.append({"vehicle_id": vehicle_id, "time": clock["time"], "slot": slot, "edge": edges[column]})

        def start(name: str, attrs) -> None:
            if name == "timestep":
                raw = attrs.get("time", "")
                value = float(raw)
                instant = int(round(value))
                if abs(value - instant) > 1e-6:
                    raise RuntimeError(f"{date} non-integer FCD time {raw}")
                previous = clock["time"]
                if previous < 0:
                    if instant != 0:
                        raise RuntimeError(f"{date} first FCD time is {instant}, expected 0")
                elif instant != previous + 1:
                    if len(gaps) < 20:
                        gaps.append([previous, instant])
                    counts["observation_gaps"] += 1
                clock["time"] = instant
                clock["snap"] = instant in boundary_of
                clock["vehicles"] = 0
                nonlocal_seen[0] += 1
                if clock["snap"]:
                    snap_edge.clear()
                    snap_mask.clear()
                return
            if name != "vehicle":
                return
            counts["records"] += 1
            clock["vehicles"] += 1
            vehicle_id = attrs.get("id", "")
            if not vehicle_id:
                return
            if vehicle_id not in population_ids:
                missing_tripinfo.add(vehicle_id)
            lane = attrs.get("lane", "")
            if lane in lane_to_edge:
                kind, column = REAL, edge_index[lane_to_edge[lane]]
            elif lane in internal:
                kind, column = INTERNAL, None
            else:
                kind, column = UNKNOWN, None
                unknown_lanes[lane] = unknown_lanes.get(lane, 0) + 1
                if len(examples["unknown_lane"]) < 8:
                    examples["unknown_lane"].append({"vehicle_id": vehicle_id, "lane": lane, "time": clock["time"]})
            mask = masks.get(vehicle_id, 0)
            slot = bin_index(clock["time"])
            current = states.get(vehicle_id)
            if current is None:
                edge, exited, last = None, False, -1
            else:
                edge, exited, last = current
            if last >= 0 and clock["time"] > last + 1:
                counts["observation_gaps"] += 1
                if len(examples["observation_gaps"]) < 8:
                    examples["observation_gaps"].append({
                        "vehicle_id": vehicle_id, "from_time": last, "to_time": clock["time"],
                    })
            if kind == REAL and clock["snap"] and column is not None:
                snap_edge.append(column)
                snap_mask.append(mask)
            if kind == UNKNOWN:
                counts["unknown_samples"] += 1
            new_edge, new_exited, events = reduce_observation(edge, exited, kind, column, slot)
            if current is None and kind == REAL and events:
                event_name = events[0][0]
                depart = departs.get(vehicle_id)
                if depart is not None and abs(float(depart) - clock["time"]) < 1.0:
                    events = [("departure_entry", events[0][1], events[0][2])]
                else:
                    events = [("first_seen_entry", events[0][1], events[0][2])]
                event_name = events[0][0]
                del event_name
            for event_name, event_slot, event_edge in events:
                note(event_name, event_slot, event_edge, vehicle_id, mask, "out" if event_name.endswith("exit") else "in")
            if new_edge is None and not events and current is None:
                return
            states[vehicle_id] = [new_edge if new_edge is not None else -1, new_exited, clock["time"]]
            if new_edge is None:
                states[vehicle_id][0] = -1

        def end(name: str) -> None:
            if name != "timestep":
                return
            if clock["vehicles"] == 0:
                counts["empty_timesteps"] += 1
            if clock["time"] == 0:
                counts["vehicles_at_0"] = clock["vehicles"]
            if not clock["snap"]:
                return
            index = boundary_of[clock["time"]]
            snapped[index] = 1
            if not snap_edge:
                return
            columns = np.asarray(snap_edge, dtype=np.int32)
            full_occ[index] = np.bincount(columns, minlength=n_edges)
            group_masks = np.asarray(snap_mask, dtype=np.uint64)
            for group_index in range(len(groups)):
                selected = (group_masks & np.uint64(1 << group_index)) != 0
                if np.any(selected):
                    partial_occ[group_index, index] = np.bincount(columns[selected], minlength=n_edges)

        nonlocal_seen = [0]
        parser = expat.ParserCreate()
        parser.StartElementHandler = start
        parser.EndElementHandler = end
        read_bytes = 0
        next_report = 4 << 30
        full.READ_BYTES = 8 << 20
        for block in full.read_chunks(fcd_path):
            read_bytes += len(block)
            parser.Parse(block, False)
            if read_bytes >= next_report:
                full.log(f"[{date}] independent fcd {read_bytes / 1024**3:.1f} GiB records={counts['records']}")
                next_report += 4 << 30
        parser.Parse(b"", True)
        if counts["vehicles_at_0"] == 0 and clock["time"] >= 0:
            # The t=0 sample may itself be an empty timestep. end() records it.
            pass
        for vehicle_id, (edge, exited, last) in states.items():
            if exited or edge is None or edge < 0:
                continue
            arrival = arrivals.get(vehicle_id)
            if arrival is None or arrival < 0:
                counts["unresolved"] += 1
                if len(examples["unresolved"]) < 8:
                    examples["unresolved"].append({"vehicle_id": vehicle_id, "last_time": last, "edge": edges[edge]})
                continue
            if arrival >= DAY_SECONDS:
                counts["arrival_at_or_after_86400"] += 1
            slot = bin_index(arrival)
            if slot is None:
                counts["unresolved"] += 1
                continue
            note("arrival_exit", slot, edge, vehicle_id, masks.get(vehicle_id, 0), "out")
        if missing_tripinfo:
            raise RuntimeError(f"{date} FCD ids missing from tripinfo: {len(missing_tripinfo)} e.g. {sorted(missing_tripinfo)[:8]}")
        missing_boundaries = [boundary_time(index) for index in range(N_BOUNDARIES) if not snapped[index]]
        if nonlocal_seen[0] != DAY_SECONDS or gaps or missing_boundaries:
            raise RuntimeError(
                f"{date} FCD clock is not 86400 consecutive seconds "
                f"seen={nonlocal_seen[0]} gaps={gaps[:5]} missing_boundaries={missing_boundaries[:5]}"
            )
        full_start, full_end = assemble_accumulation(full_occ)
        if np.any(partial_in > full_in) or np.any(partial_out > full_out):
            raise RuntimeError(f"{date} partial events exceed full")
        partial_start = np.zeros_like(partial_occ, dtype=np.int64)
        partial_end = partial_occ.astype(np.int64)
        partial_start[:, 1:] = partial_end[:, :-1]
        if np.any(partial_start > full_start) or np.any(partial_end > full_end):
            raise RuntimeError(f"{date} partial occupancy exceeds full")
        def compared(path: str | None, matrix: np.ndarray) -> dict | None:
            if not path:
                return None
            source = Path(path)
            if not source.is_file():
                return {"status": "missing"}
            old = full.read_flow_csv(source, edges)
            gap = old - matrix
            return {
                "status": "compared",
                "changed_cells": int(np.count_nonzero(gap)),
                "abs_sum": int(np.abs(gap).sum()),
                "max_abs": int(np.abs(gap).max()) if gap.size else 0,
            }

        out_root = Path(task["out_dir"])
        written = []
        matrices = {
            "inflow": full_in.astype(np.int64),
            "outflow": full_out.astype(np.int64),
            "accumulation_start": full_start,
            "accumulation_end": full_end,
        }
        full_dir = out_root / "full"
        for name, matrix in matrices.items():
            full.write_matrix(full_dir / f"{name}.csv", matrix, edges)
            written.append(str(full_dir / f"{name}.csv"))
        group_summaries = []
        for group_index, group in enumerate(groups):
            directory = out_root / group["relative"]
            group_matrices = {
                "inflow": partial_in[group_index].astype(np.int64),
                "outflow": partial_out[group_index].astype(np.int64),
                "accumulation_start": partial_start[group_index],
                "accumulation_end": partial_end[group_index],
            }
            for name, matrix in group_matrices.items():
                full.write_matrix(directory / f"{name}.csv", matrix, edges)
            group_summaries.append({
                "name": group["name"],
                "relative": group["relative"],
                "residual": summarize_residual(residual(
                    group_matrices["accumulation_start"], group_matrices["accumulation_end"],
                    group_matrices["inflow"], group_matrices["outflow"],
                )),
                "comparison": {
                    name: compared(str(Path(group["compare"]) / f"{name}.csv") if group.get("compare") else None, matrix)
                    for name, matrix in group_matrices.items()
                },
            })
        breakdown = {
            "physical_definition_version": PHYSICAL_VERSION,
            "definition_sha256": definition_sha256(),
            "date": date,
            "comparison": {
                name: compared(str(Path(task["compare_full"]) / f"{name}.csv") if task.get("compare_full") else None, matrix)
                for name, matrix in matrices.items()
            },
            "fcd": str(fcd_path),
            "samples": nonlocal_seen[0],
            "first_time": 0,
            "last_time": int(clock["time"]),
            "boundary_sample": "t=300(k+1)-1",
            "q0_minus": 0,
            "vehicles_at_time_0_sample": counts["vehicles_at_0"],
            "counts": counts,
            "class_bin_sums": {name: values.tolist() for name, values in class_bins.items()},
            "unknown_lane_ids": sorted(unknown_lanes)[:30],
            "unknown_lane_total": int(sum(unknown_lanes.values())),
            "examples": examples,
            "full_residual": summarize_residual(residual(full_start, full_end, matrices["inflow"], matrices["outflow"])),
            "groups": group_summaries,
        }
        path = full_dir / "event_breakdown.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(breakdown, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        full.log(
            f"[{date}] independent PASS records={counts['records']} "
            f"in={int(full_in.sum())} out={int(full_out.sum())} "
            f"residual_nonzero={breakdown['full_residual']['nonzero_cells']}"
        )
        marker = out_root / "complete.json"
        temporary = marker.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"date": date, "physical_definition_version": PHYSICAL_VERSION, "ok": True}) + "\n", encoding="utf-8")
        os.replace(temporary, marker)
        return {"date": date, "ok": True, "out_dir": str(out_root), "counts": counts, "full_residual": breakdown["full_residual"], "comparison": breakdown["comparison"]}
    except Exception:
        return {"date": date, "ok": False, "error": traceback.format_exc(limit=12)}


FIXED_RATES = ("p05", "p10", "p20", "p30", "p40", "p50", "p70")
SEEDS = (42, 43, 44, 45, 46)
WORKER: dict = {}


def _groups(training_root: Path, date: str, dataset_ids: list[str]) -> list[dict]:
    groups = []
    fixed = training_root / "datasets" / "fixed_uniform"
    for seed in SEEDS:
        for rate in FIXED_RATES:
            groups.append({
                "name": f"fixed:{seed}:{rate}",
                "relative": f"fixed/seed_{seed}/{rate}",
                "members": str(fixed / "sampling_seeds" / f"seed_{seed}" / "sampled_vehicle" / rate / f"{date}.csv"),
                "compare": str(fixed / "sampling_seeds" / f"seed_{seed}" / "flow" / "partial_flow" / rate / date),
            })
    for dataset_id in dataset_ids:
        root = training_root / "datasets" / dataset_id
        for seed in SEEDS:
            groups.append({
                "name": f"{dataset_id}:{seed}",
                "relative": f"variable/{dataset_id}/seed_{seed}",
                "members": str(root / "sampling_seeds" / f"seed_{seed}" / "sampled_vehicle" / "p10_p20" / f"{date}.csv"),
                "compare": str(root / "sampling_seeds" / f"seed_{seed}" / "flow" / "partial_flow" / "p10_p20" / date),
            })
    return groups


def _init_worker(payload: dict) -> None:
    global WORKER
    WORKER = payload


def _scan_task(task: dict) -> dict:
    merged = dict(task)
    merged.update(WORKER)
    return scan_groups(merged)


def scan_all(argv: list[str] | None = None) -> int:
    import argparse
    import traceback
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp

    import generate_r_level_full_flow as full
    from data_root import data_root

    parser = argparse.ArgumentParser(description="Scan FCD once per day into independent four-state staging")
    parser.add_argument("--training-root", type=Path, default=data_root() / "xuancheng26/training_data_3s")
    parser.add_argument("--fcd-root", type=Path, default=data_root() / "xuancheng26/simulation_data/fcd_data_3s")
    parser.add_argument("--net-file", type=Path, default=Path("/home/tzhang174/TrafficFlow/dataset/xuancheng26/xuancheng_tls_3s.net.xml"))
    parser.add_argument("--workers", type=int, default=29)
    parser.add_argument("--dates", nargs="*")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    self_test()
    training_root = args.training_root.expanduser().resolve()
    fcd_root = args.fcd_root.expanduser().resolve()
    split = json.loads((training_root / "shared" / "split_manifest.json").read_text(encoding="utf-8"))
    dates = list(args.dates) if args.dates else [day for name in ("train", "validation", "test") for day in split[name]]
    dataset_ids = ["variable_30min_p10_p20", "variable_1h_p10_p20", "variable_2h_p10_p20", "variable_day_p10_p20"]
    edges, lane_to_edge, internal = full.load_network(args.net_file.expanduser().resolve())
    staging = training_root / ".staging" / PHYSICAL_VERSION
    tasks = []
    for date in dates:
        out_dir = staging / date
        if (out_dir / "complete.json").is_file() and not args.force:
            full.log(f"[{date}] staging complete, skip")
            continue
        day_dir = fcd_root / date
        gzip_path = day_dir / f"fcd_{date}.xml.gz"
        plain_path = day_dir / f"fcd_{date}.xml"
        fcd_path = gzip_path if gzip_path.is_file() else plain_path
        trip = day_dir / f"tripinfo_{date}.xml.gz"
        if not trip.is_file():
            trip = day_dir / f"tripinfo_{date}.xml"
        tasks.append({
            "date": date,
            "fcd": str(fcd_path),
            "tripinfo": str(trip),
            "out_dir": str(out_dir),
            "groups": _groups(training_root, date, dataset_ids),
            "compare_full": str(training_root / "full_flow" / "r-level" / date),
        })
    full.log(f"independent scan days={len(tasks)} groups={1 + 35 + 20} workers={min(args.workers, max(1, len(tasks)))} edges={len(edges)}")
    if not tasks:
        return 0
    payload = {
        "edges": edges,
        "lane_to_edge": lane_to_edge,
        "internal": internal,
        "edge_index": {edge: index for index, edge in enumerate(edges)},
    }
    workers = max(1, min(int(args.workers), len(tasks)))
    results = []
    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker, initargs=(payload,)) as pool:
        futures = {pool.submit(_scan_task, task): task["date"] for task in tasks}
        for future in as_completed(futures):
            date = futures[future]
            try:
                row = future.result()
            except Exception:
                row = {"date": date, "ok": False, "error": traceback.format_exc(limit=8)}
            results.append(row)
            if not row.get("ok"):
                full.log(f"[{date}] FAIL {(row.get('error') or '')[-500:]}")
            else:
                comparison = row.get("comparison") or {}
                inflow = (comparison.get("inflow") or {}).get("changed_cells")
                stock = (comparison.get("accumulation_end") or {}).get("changed_cells")
                full.log(f"[{date}] done inflow_changed={inflow} accumulation_end_changed={stock}")
    failed = [row["date"] for row in results if not row.get("ok")]
    summary = staging / "scan_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps({"failed": failed, "rows": results}, ensure_ascii=False) + "\n", encoding="utf-8")
    full.log(f"scan failed={failed or 'none'} summary={summary}")
    return 1 if failed else 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "scan":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        raise SystemExit(scan_all())
    self_test()
    print("independent_states self-test passed")

