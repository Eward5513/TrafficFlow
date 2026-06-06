"""Export phase-level SUMO traffic-light connection permissions as JSON.

Default target:
    analysis/road_network/net_tls.net.xml
    all traffic-light junctions
"""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET


DEFAULT_NET_FILE = (
    Path(__file__).resolve().parents[1] / "road_network" / "net_tls.net.xml"
)
DEFAULT_OUTPUT_FILE = Path(__file__).resolve().with_name("junction_tls.json")
ALLOWED_SIGNAL_STATES = {"G", "g"}


def sort_connection(connection: ET.Element) -> tuple[int, str, str, str]:
    link_index = connection.get("linkIndex")
    try:
        link_order = int(link_index) if link_index is not None else 10**9
    except ValueError:
        link_order = 10**9

    return (
        link_order,
        connection.get("from", ""),
        connection.get("fromLane", ""),
        connection.get("to", ""),
    )


def is_junction_connection(connection: ET.Element, junction_id: str) -> bool:
    if connection.get("tl") == junction_id:
        return True

    via = connection.get("via")
    return bool(via and via.startswith(f":{junction_id}_"))


def load_junction_details(
    net_file: Path, junction_id: str
) -> tuple[ET.Element | None, list[ET.Element], ET.Element | None]:
    junction: ET.Element | None = None
    connections: list[ET.Element] = []
    tls: ET.Element | None = None

    for _, elem in ET.iterparse(net_file, events=("end",)):
        if elem.tag == "junction" and elem.get("id") == junction_id:
            junction = elem
            continue

        if elem.tag == "connection" and is_junction_connection(elem, junction_id):
            connections.append(elem)
            continue

        if elem.tag == "tlLogic" and elem.get("id") == junction_id:
            tls = elem

    return junction, connections, tls


def load_all_junction_details(
    net_file: Path,
) -> dict[str, dict[str, ET.Element | list[ET.Element] | None]]:
    junctions: dict[str, dict[str, ET.Element | list[ET.Element] | None]] = {}

    for _, elem in ET.iterparse(net_file, events=("end",)):
        if elem.tag == "tlLogic":
            junction_id = elem.get("id")
            if junction_id is None:
                continue
            data = junctions.setdefault(
                junction_id, {"junction": None, "connections": [], "tls": None}
            )
            data["tls"] = elem
            continue

        if elem.tag == "junction":
            junction_id = elem.get("id")
            if junction_id is None:
                continue
            data = junctions.setdefault(
                junction_id, {"junction": None, "connections": [], "tls": None}
            )
            data["junction"] = elem
            continue

        if elem.tag == "connection":
            junction_id = elem.get("tl")
            if junction_id is None:
                continue

            data = junctions.setdefault(
                junction_id, {"junction": None, "connections": [], "tls": None}
            )
            connections = data["connections"]
            if isinstance(connections, list):
                connections.append(elem)

    return junctions


def phase_priority(phase_state: str, connection: ET.Element) -> str | None:
    link_index = connection.get("linkIndex")
    if link_index is None:
        return None

    try:
        index = int(link_index)
    except ValueError:
        return None

    if index < 0 or index >= len(phase_state):
        return None

    return phase_state[index]


def connection_group_key(connection: ET.Element) -> tuple[str, str, str]:
    return (
        connection.get("from", ""),
        connection.get("to", ""),
        connection.get("dir", ""),
    )


def build_phase_connections(tls: ET.Element, connections: list[ET.Element]) -> list[dict]:
    sorted_connections = sorted(connections, key=sort_connection)
    phases: list[dict] = []

    for phase_index, phase in enumerate(tls.findall("phase")):
        phase_state = phase.get("state", "")
        grouped_connections: dict[tuple[str, str, str], dict] = {}

        for connection in sorted_connections:
            priority = phase_priority(phase_state, connection)
            if priority not in ALLOWED_SIGNAL_STATES:
                continue

            group_key = connection_group_key(connection)
            group = grouped_connections.setdefault(
                group_key,
                {
                    "from": group_key[0],
                    "to": group_key[1],
                    "dir": group_key[2],
                    "priority": priority,
                    "laneCount": 0,
                    "lanes": [],
                },
            )
            group["lanes"].append(
                {
                    "fromLane": connection.get("fromLane"),
                    "toLane": connection.get("toLane"),
                    "linkIndex": connection.get("linkIndex"),
                    "via": connection.get("via"),
                    "priority": priority,
                    "connectionState": connection.get("state"),
                }
            )
            group["laneCount"] = len(group["lanes"])

        phases.append(
            {
                "index": phase_index,
                "duration": phase.get("duration"),
                "state": phase_state,
                "allowedConnections": list(grouped_connections.values()),
            }
        )

    return phases


def build_output(
    net_file: Path,
    junction_id: str,
    junction: ET.Element | None,
    connections: list[ET.Element],
    tls: ET.Element | None,
) -> dict:
    if tls is None:
        phases: list[dict] = []
    else:
        phases = build_phase_connections(tls, connections)

    return {
        "junctionId": junction_id,
        "phases": phases,
    }


def build_all_output(
    net_file: Path,
    junctions: dict[str, dict[str, ET.Element | list[ET.Element] | None]],
) -> dict:
    junction_outputs = []

    for junction_id in sorted(junctions):
        data = junctions[junction_id]
        if data["tls"] is None:
            continue

        junction = data["junction"]
        connections = data["connections"]
        tls = data["tls"]
        if not isinstance(connections, list):
            connections = []
        if junction is not None and not isinstance(junction, ET.Element):
            junction = None
        if tls is not None and not isinstance(tls, ET.Element):
            tls = None

        junction_outputs.append(
            build_output(net_file, junction_id, junction, connections, tls)
        )

    return {
        "netFile": str(net_file),
        "allowedSignalStates": sorted(ALLOWED_SIGNAL_STATES),
        "junctionCount": len(junction_outputs),
        "junctions": junction_outputs,
    }


def main() -> None:
    net_file = DEFAULT_NET_FILE.resolve()

    if not net_file.exists():
        raise FileNotFoundError(f"SUMO net file not found: {net_file}")

    junctions = load_all_junction_details(net_file)
    output = build_all_output(net_file, junctions)
    output_file = DEFAULT_OUTPUT_FILE.resolve()
    output_file.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved: {output_file}")


if __name__ == "__main__":
    main()
