"""Count vehicles leaving one edge during the first traffic-light cycles.

The script uses:
    - analysis/output/fcd.csv
    - analysis/flow/junction_tls.json

Output:
    - analysis/flow/edge_departures_by_cycle.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
FLOW_DIR = Path(__file__).resolve().parent

FCD_CSV_FILE = BASE_DIR / "output" / "fcd.csv"
TLS_FILE = FLOW_DIR / "junction_tls.json"
OUTPUT_FILE = FLOW_DIR / "edge_departures_by_cycle.json"

SOURCE_EDGE = "850154338#1"
CYCLE_COUNT = 10

TIME_FIELDS = ("time", "timestep_time", "timestep")
VEHICLE_ID_FIELDS = ("id", "vehicle_id", "vehicle")
EDGE_FIELDS = ("edge", "vehicle_edge")
LANE_FIELDS = ("lane", "vehicle_lane")


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


def fcd_source_file() -> Path:
    if FCD_CSV_FILE.exists():
        return FCD_CSV_FILE
    raise FileNotFoundError(f"FCD CSV file not found: {FCD_CSV_FILE}")


def load_tls() -> dict[str, Any]:
    return json.loads(TLS_FILE.read_text(encoding="utf-8"))


def cycle_length(tls: dict[str, Any]) -> float:
    return sum(float(phase["duration"]) for phase in tls["phases"])


def phase_windows(tls: dict[str, Any]) -> list[dict[str, Any]]:
    start = 0.0
    windows: list[dict[str, Any]] = []

    for phase in tls["phases"]:
        duration = float(phase["duration"])
        windows.append(
            {
                "index": phase["index"],
                "start": start,
                "end": start + duration,
                "duration": duration,
                "state": phase["state"],
            }
        )
        start += duration

    return windows


def phase_at_time(time: float, length: float, windows: list[dict[str, Any]]) -> dict[str, Any]:
    time_in_cycle = time % length
    for window in windows:
        if window["start"] <= time_in_cycle < window["end"]:
            return window
    return windows[-1]


def source_edge_connections(tls: dict[str, Any]) -> dict[str, dict[str, Any]]:
    connections_by_target: dict[str, dict[str, Any]] = {}

    for phase in tls["phases"]:
        for connection in phase["allowedConnections"]:
            if connection["from"] != SOURCE_EDGE:
                continue

            target_edge = connection["to"]
            current = connections_by_target.setdefault(
                target_edge,
                {
                    "to": target_edge,
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

    return connections_by_target


def source_lane_connections(
    connections_by_target: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    connections_by_source_lane: dict[str, list[dict[str, Any]]] = {}

    for connection in connections_by_target.values():
        for lane in connection["lanes"]:
            source_lane = f'{SOURCE_EDGE}_{lane["fromLane"]}'
            lane_connection = {
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
            return lane_connection["connection"]

        if current_edge == lane_connection["targetEdge"]:
            return lane_connection["connection"]

    return None


def empty_direction_bucket(connection: dict[str, Any]) -> dict[str, Any]:
    by_source_lane = {}
    for lane in connection["lanes"]:
        source_lane = f'{SOURCE_EDGE}_{lane["fromLane"]}'
        by_source_lane[source_lane] = {
            "fromLane": source_lane,
            "fromLaneIndex": lane["fromLane"],
            "toLane": f'{connection["to"]}_{lane["toLane"]}',
            "toLaneIndex": lane["toLane"],
            "via": lane["via"],
            "linkIndex": lane["linkIndex"],
            "priority": lane["priority"],
            "connectionState": lane["connectionState"],
            "count": 0,
            "vehicleIds": [],
            "events": [],
        }

    return {
        "dir": connection["dir"],
        "to": connection["to"],
        "count": 0,
        "vehicleIds": [],
        "bySourceLane": by_source_lane,
        "connection": {
            "laneCount": connection["laneCount"],
            "lanes": connection["lanes"],
            "allowedPhases": connection["allowedPhases"],
        },
    }


def build_empty_cycles(
    connections_by_target: dict[str, dict[str, Any]], length: float
) -> list[dict[str, Any]]:
    cycles: list[dict[str, Any]] = []

    for cycle_index in range(CYCLE_COUNT):
        cycle_start = cycle_index * length
        cycle_end = cycle_start + length
        by_direction = {
            connection["dir"]: empty_direction_bucket(connection)
            for connection in connections_by_target.values()
        }
        cycles.append(
            {
                "cycleIndex": cycle_index,
                "start": cycle_start,
                "end": cycle_end,
                "byDirection": by_direction,
            }
        )

    return cycles


def add_departure_event(
    cycles: list[dict[str, Any]],
    cycle_index: int,
    connection: dict[str, Any],
    vehicle_id: str,
    time: float,
    from_lane: str | None,
    to_lane: str | None,
    phase: dict[str, Any],
) -> None:
    bucket = cycles[cycle_index]["byDirection"].setdefault(
        connection["dir"], empty_direction_bucket(connection)
    )
    if vehicle_id not in bucket["vehicleIds"]:
        bucket["vehicleIds"].append(vehicle_id)
        bucket["count"] += 1

    source_lane_key = from_lane or "unknown"
    source_lane_bucket = bucket["bySourceLane"].setdefault(
        source_lane_key,
        {
            "fromLane": source_lane_key,
            "fromLaneIndex": None,
            "toLane": to_lane,
            "toLaneIndex": None,
            "via": None,
            "linkIndex": None,
            "priority": None,
            "connectionState": None,
            "count": 0,
            "vehicleIds": [],
            "events": [],
        },
    )
    if vehicle_id not in source_lane_bucket["vehicleIds"]:
        source_lane_bucket["vehicleIds"].append(vehicle_id)
        source_lane_bucket["count"] += 1

    source_lane_bucket["events"].append(
        {
            "vehicleId": vehicle_id,
            "time": time,
            "fromLane": from_lane,
            "toLane": to_lane,
            "fromEdge": SOURCE_EDGE,
            "toEdge": connection["to"],
            "phaseIndex": phase["index"],
            "phaseState": phase["state"],
        }
    )


def record_snapshot(
    cycles: list[dict[str, Any]],
    connections_by_source_lane: dict[str, list[dict[str, Any]]],
    length: float,
    windows: list[dict[str, Any]],
    vehicles_on_source: dict[str, str],
    counted_events: set[tuple[int, str, str]],
    time: float,
    vehicle_id: str,
    edge_id: str | None,
    lane_id: str | None,
) -> None:
    if edge_id == SOURCE_EDGE and lane_id is not None:
        vehicles_on_source[vehicle_id] = lane_id
        return

    previous_source_lane = vehicles_on_source.get(vehicle_id)
    if previous_source_lane is None:
        return

    connection = find_departure_connection(
        connections_by_source_lane, previous_source_lane, edge_id, lane_id
    )
    if connection is None:
        return

    cycle_index = int(time // length)
    event_key = (cycle_index, connection["dir"], vehicle_id)
    if event_key not in counted_events:
        add_departure_event(
            cycles,
            cycle_index,
            connection,
            vehicle_id,
            time,
            previous_source_lane,
            lane_id,
            phase_at_time(time, length, windows),
        )
        counted_events.add(event_key)

    del vehicles_on_source[vehicle_id]


def count_csv_departures(
    fcd_file: Path,
    cycles: list[dict[str, Any]],
    connections_by_source_lane: dict[str, list[dict[str, Any]]],
    length: float,
    windows: list[dict[str, Any]],
) -> None:
    max_time = CYCLE_COUNT * length
    vehicles_on_source: dict[str, str] = {}
    counted_events: set[tuple[int, str, str]] = set()

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

        for row in reader:
            time = float(row[time_index])
            if time >= max_time:
                break

            vehicle_id = row[vehicle_index]
            lane_id = row[lane_index]
            edge_id = row[edge_index] if edge_index is not None and row[edge_index] else None
            if not edge_id:
                edge_id = edge_from_lane(lane_id)

            if edge_id != SOURCE_EDGE and vehicle_id not in vehicles_on_source:
                continue

            record_snapshot(
                cycles,
                connections_by_source_lane,
                length,
                windows,
                vehicles_on_source,
                counted_events,
                time,
                vehicle_id,
                edge_id,
                lane_id,
            )


def count_departures(
    fcd_file: Path,
    connections_by_target: dict[str, dict[str, Any]],
    length: float,
    windows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cycles = build_empty_cycles(connections_by_target, length)
    connections_by_source_lane = source_lane_connections(connections_by_target)

    count_csv_departures(fcd_file, cycles, connections_by_source_lane, length, windows)

    return cycles


def main() -> None:
    tls = load_tls()
    length = cycle_length(tls)
    windows = phase_windows(tls)
    connections_by_target = source_edge_connections(tls)
    fcd_file = fcd_source_file()
    cycles = count_departures(fcd_file, connections_by_target, length, windows)
    output = {
        "fcdFile": str(fcd_file.resolve()),
        "tlsFile": str(TLS_FILE.resolve()),
        "sourceEdge": SOURCE_EDGE,
        "cycleLength": length,
        "cycleCount": CYCLE_COUNT,
        "cycles": cycles,
    }

    OUTPUT_FILE.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
