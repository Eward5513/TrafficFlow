"""Count vehicles leaving junction incoming edges during all FCD cycles.

The script uses:
    - analysis/simulation/output/<penetration>/fcd.csv
    - analysis/flow/junction_tls.json

Output:
    - analysis/flow/full_flow/<penetration>/*_edge_departures_by_cycle.json
"""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
FLOW_DIR = Path(__file__).resolve().parent

SIMULATION_OUTPUT_DIR = BASE_DIR / "simulation" / "output"
FULL_FLOW_DIR = FLOW_DIR / "full_flow"
TARGET_JUNCTION_IDS = (
    "cluster_1262396634_1746662956",
    "cluster_1247897642_2350807770",
    "cluster_1262396675_2350807772",
    "cluster_3476413627_3476413628",
    "3476413738",
    "cluster_1746667327_1746667337",
    "1746667339",
    "1746667341",
    "cluster_3476413732_3476413733",
)
TLS_FILE = FLOW_DIR / "junction_tls.json"

TIME_FIELDS = ("time", "timestep_time", "timestep")
VEHICLE_ID_FIELDS = ("id", "vehicle_id", "vehicle")
EDGE_FIELDS = ("edge", "vehicle_edge")
LANE_FIELDS = ("lane", "vehicle_lane")
FCD_CHUNK_SECONDS = 10.0


def penetration_sort_key(penetration: str) -> tuple[int, float | str]:
    try:
        return (0, float(penetration.replace("_", ".")))
    except ValueError:
        return (1, penetration)


def available_penetrations() -> list[str]:
    if not SIMULATION_OUTPUT_DIR.exists():
        raise FileNotFoundError(f"simulation output dir not found: {SIMULATION_OUTPUT_DIR}")

    penetrations = [
        path.name
        for path in SIMULATION_OUTPUT_DIR.iterdir()
        if path.is_dir() and (path / "fcd.csv").exists()
    ]
    if not penetrations:
        raise FileNotFoundError(
            f"no penetration folders with fcd.csv found under: {SIMULATION_OUTPUT_DIR}"
        )

    return sorted(penetrations, key=penetration_sort_key)


def first_present(row: dict[str, str], field_names: tuple[str, ...]) -> str | None:
    for field_name in field_names:
        value = row.get(field_name)
        if value not in (None, ""):
            return value
    return None


def edge_from_lane(lane_id: str | None) -> str | None:
    if not lane_id or "_" not in lane_id:
        return None

    edge_id, lane_index = lane_id.rsplit("_", 1)
    if not lane_index.isdigit():
        return None
    return edge_id


def lane_id(edge_id: str, lane_index: str) -> str:
    return f"{edge_id}_{lane_index}"


def row_edge(row: dict[str, str]) -> str | None:
    edge_id = first_present(row, EDGE_FIELDS)
    if edge_id:
        return edge_id
    return edge_from_lane(first_present(row, LANE_FIELDS))


def row_time(row: dict[str, str]) -> float:
    value = first_present(row, TIME_FIELDS)
    if value is None:
        raise ValueError(f"missing time field, expected one of: {TIME_FIELDS}")
    return float(value)


def row_vehicle_id(row: dict[str, str]) -> str:
    value = first_present(row, VEHICLE_ID_FIELDS)
    if value is None:
        raise ValueError(f"missing vehicle id field, expected one of: {VEHICLE_ID_FIELDS}")
    return value


def fcd_source_file(penetration: str) -> Path:
    fcd_file = BASE_DIR / "simulation" / "output" / penetration / "fcd.csv"
    if fcd_file.exists():
        return fcd_file
    raise FileNotFoundError(f"FCD CSV file not found: {fcd_file}")


def output_file_for_junction(output_dir: Path, junction_id: str) -> Path:
    return output_dir / f"{junction_id}_edge_departures_by_cycle.json"


def load_tls_file() -> dict[str, Any]:
    return json.loads(TLS_FILE.read_text(encoding="utf-8"))


def traffic_light_junctions(tls_data: dict[str, Any]) -> list[dict[str, Any]]:
    if "junctions" in tls_data:
        return [
            junction
            for junction in tls_data["junctions"]
            if junction.get("phases")
        ]

    return [tls_data]


def cycle_length(tls: dict[str, Any]) -> float:
    return sum(float(phase["duration"]) for phase in tls["phases"])


def is_yellow_phase(phase: dict[str, Any]) -> bool:
    return not phase["hasAllowedConnections"] and "y" in phase["state"].lower()


def phase_windows(tls: dict[str, Any]) -> list[dict[str, Any]]:
    start = 0.0
    windows: list[dict[str, Any]] = []

    for phase in tls["phases"]:
        duration = float(phase["duration"])
        window = {
            "index": phase["index"],
            "start": start,
            "end": start + duration,
            "duration": duration,
            "state": phase["state"],
            "hasAllowedConnections": bool(phase["allowedConnections"]),
            "allowedConnectionCount": len(phase["allowedConnections"]),
        }
        if window["hasAllowedConnections"] or is_yellow_phase(window):
            windows.append(window)
        start += duration

    return windows


def phase_at_time(
    time: float, length: float, windows: list[dict[str, Any]]
) -> dict[str, Any] | None:
    time_in_cycle = time % length
    for window in windows:
        if window["start"] <= time_in_cycle < window["end"]:
            return window
    return None


def source_edge_connections(tls: dict[str, Any]) -> dict[str, dict[str, Any]]:
    connections_by_movement: dict[str, dict[str, Any]] = {}

    for phase in tls["phases"]:
        for connection in phase["allowedConnections"]:
            movement_key = (
                f'{connection["from"]}->{connection["to"]}:{connection["dir"]}'
            )
            current = connections_by_movement.setdefault(
                movement_key,
                {
                    "from": connection["from"],
                    "to": connection["to"],
                    "dir": connection["dir"],
                    "laneCount": connection["laneCount"],
                    "lanes": connection["lanes"],
                    "allowedPhases": [],
                },
            )
            current["allowedPhases"].append(
                {
                    "phaseIndex": phase["index"],
                    "priority": connection["priority"],
                    "duration": phase["duration"],
                    "state": phase["state"],
                }
            )

    return connections_by_movement


def incoming_edges(connections_by_movement: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        {connection["from"] for connection in connections_by_movement.values()}
    )


def source_lane_connections(
    connections_by_movement: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    connections_by_source_lane: dict[str, list[dict[str, Any]]] = {}

    for connection in connections_by_movement.values():
        for lane in connection["lanes"]:
            source_lane = f'{connection["from"]}_{lane["fromLane"]}'
            lane_connection = {
                "sourceEdge": connection["from"],
                "sourceLane": source_lane,
                "targetEdge": connection["to"],
                "targetLane": f'{connection["to"]}_{lane["toLane"]}',
                "viaLane": lane["via"],
                "connection": connection,
            }
            connections_by_source_lane.setdefault(source_lane, []).append(lane_connection)

    return connections_by_source_lane


def find_departure_connection(
    connections_by_source_lane: dict[str, list[dict[str, Any]]],
    source_lane: str,
    current_edge: str | None,
    current_lane: str | None,
) -> dict[str, Any] | None:
    for lane_connection in connections_by_source_lane.get(source_lane, []):
        if current_lane in (
            lane_connection["viaLane"],
            lane_connection["targetLane"],
        ):
            return lane_connection

        if current_edge == lane_connection["targetEdge"]:
            return lane_connection

    return None


def build_cycle(cycle_index: int, length: float, windows: list[dict[str, Any]]) -> dict[str, Any]:
    cycle_start = cycle_index * length
    cycle_end = cycle_start + length
    time_slices = []
    phase_lookup = {}

    for phase in windows:
        time_slice = {
            "phaseIndex": phase["index"],
            "start": cycle_start + phase["start"],
            "end": cycle_start + phase["end"],
            "count": 0,
            "movements": [],
            "_movementIndex": {},
        }
        time_slices.append(time_slice)
        phase_lookup[str(phase["index"])] = time_slice

    return {
        "cycleIndex": cycle_index,
        "start": cycle_start,
        "end": cycle_end,
        "timeSlices": time_slices,
        "_phaseLookup": phase_lookup,
    }


def ensure_cycle(
    cycles: list[dict[str, Any]],
    cycle_index: int,
    length: float,
    windows: list[dict[str, Any]],
) -> dict[str, Any]:
    while len(cycles) <= cycle_index:
        cycles.append(build_cycle(len(cycles), length, windows))
    return cycles[cycle_index]


def add_departure_event(
    cycles: list[dict[str, Any]],
    cycle_index: int,
    length: float,
    windows: list[dict[str, Any]],
    lane_connection: dict[str, Any],
    vehicle_id: str,
    time: float,
    from_lane: str | None,
    phase: dict[str, Any],
) -> None:
    connection = lane_connection["connection"]
    entered_lane = lane_connection["targetLane"]
    phase_key = str(phase["index"])
    cycle = ensure_cycle(cycles, cycle_index, length, windows)
    time_slice = cycle["_phaseLookup"].get(phase_key)
    if time_slice is None:
        return

    movement_key = (
        connection["dir"],
        from_lane or "unknown",
        entered_lane,
        connection["to"],
    )
    movement_index = time_slice["_movementIndex"]
    movement = movement_index.get(movement_key)
    if movement is None:
        movement = {
            "direction": connection["dir"],
            "fromEdge": connection["from"],
            "fromLane": from_lane or "unknown",
            "enteredLane": entered_lane,
            "toEdge": connection["to"],
            "count": 0,
            "vehicleIds": [],
        }
        movement_index[movement_key] = movement
        time_slice["movements"].append(movement)

    movement["count"] += 1
    movement["vehicleIds"].append(vehicle_id)
    time_slice["count"] += 1


def is_internal_lane(lane_id: str | None, junction_id: str) -> bool:
    return bool(lane_id and lane_id.startswith(f":{junction_id}_"))


def process_time_step(
    cycles: list[dict[str, Any]],
    connections_by_source_lane: dict[str, list[dict[str, Any]]],
    length: float,
    windows: list[dict[str, Any]],
    waiting: dict[str, str],
    leaving: dict[str, dict[str, Any]],
    counted_events: set[tuple[int, str, str, str, str]],
    time: float,
    states: dict[str, dict[str, str | None]],
    junction_id: str,
    source_edges: set[str],
    source_state_items: list[tuple[str, dict[str, str | None]]] | None = None,
) -> None:
    for vehicle_id in list(leaving):
        state = states.get(vehicle_id)
        if state is None:
            leaving.pop(vehicle_id, None)
            continue

        lane_id = state["lane"]
        if is_internal_lane(lane_id, junction_id):
            continue

        pending = leaving[vehicle_id]
        lane_connection = find_departure_connection(
            connections_by_source_lane,
            pending["sourceLane"],
            state["edge"],
            lane_id,
        )
        if lane_connection is not None:
            connection = lane_connection["connection"]
            event_key = (
                pending["cycleIndex"],
                connection["from"],
                connection["dir"],
                connection["to"],
                vehicle_id,
            )
            if event_key not in counted_events:
                add_departure_event(
                    cycles,
                    pending["cycleIndex"],
                    length,
                    windows,
                    lane_connection,
                    vehicle_id,
                    pending["time"],
                    pending["sourceLane"],
                    pending["phase"],
                )
                counted_events.add(event_key)

        leaving.pop(vehicle_id, None)

    for vehicle_id in list(waiting):
        state = states.get(vehicle_id)
        if state is None:
            waiting.pop(vehicle_id, None)
            continue

        edge_id = state["edge"]
        lane_id = state["lane"]
        source_lane = waiting[vehicle_id]

        if edge_id in source_edges and lane_id is not None:
            waiting[vehicle_id] = lane_id
            continue

        if is_internal_lane(lane_id, junction_id):
            lane_connection = find_departure_connection(
                connections_by_source_lane, source_lane, edge_id, lane_id
            )
            if lane_connection is not None:
                phase = phase_at_time(time, length, windows)
                if phase is None:
                    waiting.pop(vehicle_id, None)
                    continue

                leaving[vehicle_id] = {
                    "time": time,
                    "cycleIndex": int(time // length),
                    "phase": phase,
                    "sourceLane": source_lane,
                }
                waiting.pop(vehicle_id, None)
            continue

        lane_connection = find_departure_connection(
            connections_by_source_lane, source_lane, edge_id, lane_id
        )
        if lane_connection is not None:
            phase = phase_at_time(time, length, windows)
            if phase is None:
                waiting.pop(vehicle_id, None)
                continue

            cycle_index = int(time // length)
            connection = lane_connection["connection"]
            event_key = (
                cycle_index,
                connection["from"],
                connection["dir"],
                connection["to"],
                vehicle_id,
            )
            if event_key not in counted_events:
                add_departure_event(
                    cycles,
                    cycle_index,
                    length,
                    windows,
                    lane_connection,
                    vehicle_id,
                    time,
                    source_lane,
                    phase,
                )
                counted_events.add(event_key)

        waiting.pop(vehicle_id, None)

    if source_state_items is None:
        source_state_items = list(states.items())

    for vehicle_id, state in source_state_items:
        if state["edge"] in source_edges and state["lane"] is not None:
            waiting[vehicle_id] = state["lane"]


def count_csv_departures(
    fcd_file: Path,
    junction_contexts: list[dict[str, Any]],
    chunk_seconds: float = FCD_CHUNK_SECONDS,
) -> None:
    if not junction_contexts:
        return

    time_steps: list[tuple[float, dict[str, dict[str, str | None]]]] = []
    chunk_start: float | None = None

    def process_chunk() -> None:
        for time, states in time_steps:
            states_by_edge: dict[str, list[tuple[str, dict[str, str | None]]]] = {}
            for vehicle_id, state in states.items():
                edge_id = state["edge"]
                if edge_id is None:
                    continue
                states_by_edge.setdefault(edge_id, []).append((vehicle_id, state))

            for context in junction_contexts:
                source_state_items: list[tuple[str, dict[str, str | None]]] = []
                for edge_id in context["sourceEdges"]:
                    source_state_items.extend(states_by_edge.get(edge_id, ()))

                process_time_step(
                    context["cycles"],
                    context["connectionsBySourceLane"],
                    context["cycleLength"],
                    context["phaseWindows"],
                    context["waiting"],
                    context["leaving"],
                    context["countedEvents"],
                    time,
                    states,
                    context["junctionId"],
                    context["sourceEdges"],
                    source_state_items,
                )

    def append_time_step(
        time: float, states: dict[str, dict[str, str | None]]
    ) -> None:
        nonlocal chunk_start, time_steps

        if chunk_start is None:
            chunk_start = time

        if time - chunk_start >= chunk_seconds and time_steps:
            process_chunk()
            time_steps = []
            chunk_start = time

        time_steps.append((time, states))

    with fcd_file.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        time_index = next(i for i, name in enumerate(header) if name in TIME_FIELDS)
        vehicle_index = next(
            i for i, name in enumerate(header) if name in VEHICLE_ID_FIELDS
        )
        lane_index = next(i for i, name in enumerate(header) if name in LANE_FIELDS)
        edge_index = next(
            (i for i, name in enumerate(header) if name in EDGE_FIELDS), None
        )

        current_time: float | None = None
        states: dict[str, dict[str, str | None]] = {}

        for row in reader:
            time = float(row[time_index])
            if current_time is not None and time != current_time:
                append_time_step(current_time, states)
                states = {}

            current_time = time

            vehicle_id = row[vehicle_index]
            lane_id = row[lane_index]
            edge_id = row[edge_index] if edge_index is not None and row[edge_index] else None
            if not edge_id:
                edge_id = edge_from_lane(lane_id)

            states[vehicle_id] = {
                "edge": edge_id,
                "lane": lane_id,
            }

        if current_time is not None:
            append_time_step(current_time, states)

        if time_steps:
            process_chunk()


def build_junction_context(tls: dict[str, Any]) -> dict[str, Any]:
    length = cycle_length(tls)
    windows = phase_windows(tls)
    connections_by_movement = source_edge_connections(tls)
    incoming_edge_list = incoming_edges(connections_by_movement)

    return {
        "tls": tls,
        "junctionId": tls["junctionId"],
        "cycleLength": length,
        "phaseWindows": windows,
        "connectionsBySourceLane": source_lane_connections(connections_by_movement),
        "incomingEdges": incoming_edge_list,
        "sourceEdges": set(incoming_edge_list),
        "cycles": [],
        "waiting": {},
        "leaving": {},
        "countedEvents": set(),
    }


def phase_time_slices(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "phaseIndex": phase["index"],
            "startInCycle": phase["start"],
            "endInCycle": phase["end"],
        }
        for phase in windows
    ]


def compact_cycles(cycles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted = []
    for cycle in cycles:
        time_slices = []
        for time_slice in cycle["timeSlices"]:
            time_slices.append(
                {
                    "phaseIndex": time_slice["phaseIndex"],
                    "start": time_slice["start"],
                    "end": time_slice["end"],
                    "count": time_slice["count"],
                    "movements": time_slice["movements"],
                }
            )

        compacted.append(
            {
                "cycleIndex": cycle["cycleIndex"],
                "start": cycle["start"],
                "end": cycle["end"],
                "timeSlices": time_slices,
            }
        )

    return compacted


def build_output(context: dict[str, Any], fcd_file: Path) -> dict[str, Any]:
    return {
        "fcdFile": str(fcd_file.resolve()),
        "tlsFile": str(TLS_FILE.resolve()),
        "junctionId": context["junctionId"],
        "incomingEdges": context["incomingEdges"],
        "cycleLength": context["cycleLength"],
        "cycleCount": len(context["cycles"]),
        "phaseTimeSlices": phase_time_slices(context["phaseWindows"]),
        "cycles": compact_cycles(context["cycles"]),
    }


def selected_junctions(
    tls_data: dict[str, Any], junction_ids: list[str] | tuple[str, ...]
) -> list[dict[str, Any]]:
    tls_by_junction_id = {
        tls["junctionId"]: tls
        for tls in traffic_light_junctions(tls_data)
    }
    selected = []
    for junction_id in junction_ids:
        tls = tls_by_junction_id.get(junction_id)
        if tls is None:
            raise ValueError(f"junction not found in TLS JSON: {junction_id}")
        selected.append(tls)
    return selected


def process_penetration(
    penetration: str,
    junction_tls_items: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    output_dir = FULL_FLOW_DIR / penetration
    fcd_file = fcd_source_file(penetration)
    output_dir.mkdir(parents=True, exist_ok=True)

    junction_contexts = [
        build_junction_context(tls)
        for tls in junction_tls_items
    ]
    count_csv_departures(fcd_file, junction_contexts)

    saved_files: list[str] = []
    for context in junction_contexts:
        output = build_output(context, fcd_file)
        output_file = output_file_for_junction(output_dir, context["junctionId"])
        output_file.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        saved_files.append(str(output_file.resolve()))

    return penetration, saved_files


def main() -> None:
    if not TLS_FILE.exists():
        raise FileNotFoundError(f"TLS JSON not found: {TLS_FILE}")

    tls_data = load_tls_file()
    junction_tls_items = selected_junctions(tls_data, TARGET_JUNCTION_IDS)
    penetrations = available_penetrations()

    print(f"processing penetrations in parallel: {', '.join(penetrations)}")
    with ProcessPoolExecutor(max_workers=len(penetrations)) as executor:
        futures = {
            executor.submit(process_penetration, penetration, junction_tls_items): penetration
            for penetration in penetrations
        }
        for future in as_completed(futures):
            penetration = futures[future]
            try:
                _, saved_files = future.result()
            except Exception as error:
                raise RuntimeError(
                    f"failed to process penetration {penetration}"
                ) from error

            for saved_file in saved_files:
                print(f"saved: {saved_file}")


if __name__ == "__main__":
    main()
