"""Export phase-level SUMO connection permissions for a junction as JSON.

Default target:
    analysis/data/net_tls1.net.xml
    cluster_1247897642_2350807770
"""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET


DEFAULT_JUNCTION_ID = "cluster_1247897642_2350807770"
DEFAULT_NET_FILE = Path(__file__).resolve().parents[1] / "data" / "net_tls1.net.xml"
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
        "netFile": str(net_file),
        "junctionId": junction_id,
        "junctionFound": junction is not None,
        "tlsFound": tls is not None,
        "allowedSignalStates": sorted(ALLOWED_SIGNAL_STATES),
        "phases": phases,
    }


def main() -> None:
    net_file = DEFAULT_NET_FILE.resolve()

    if not net_file.exists():
        raise FileNotFoundError(f"SUMO net file not found: {net_file}")

    junction, connections, tls = load_junction_details(net_file, DEFAULT_JUNCTION_ID)
    output = build_output(net_file, DEFAULT_JUNCTION_ID, junction, connections, tls)
    output_file = DEFAULT_OUTPUT_FILE.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved: {output_file}")


if __name__ == "__main__":
    main()
