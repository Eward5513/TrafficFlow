#!/usr/bin/env python3
"""Build the directed Movement-level (M) graph for Xuancheng26.

A Movement is one legal turn ``(junction, incoming real edge, outgoing real
edge)``. Every lane-level ``<connection>`` with the same triple is merged into
one node, and its lane indices, via lanes, TLS link indices and states are kept
per lane connection. ``m_i -> m_j`` exists when ``outgoing(m_i) ==
incoming(m_j)``; the shared real edge is the transition's intermediate edge.

Nodes are ordered by the R-graph index of the incoming edge, then of the
outgoing edge. All outputs go to ``<training-root>/graph/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from xml.parsers import expat

import numpy as np

import generate_r_graph as rgraph
import generate_r_level_full_flow as full
import prep_paths

NET_FILE = prep_paths.NET_FILE_3S
TLS_FILE = prep_paths.TLS_FILE_3S
TRAINING_ROOT = prep_paths.TRAINING_ROOT_3S
MAX_VIA_CHAIN = 8
SHORT_EDGE_METERS = (5.0, 10.0, 20.0)


def log(message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"{stamp} {message}", flush=True)


def movement_id(junction_id: str, incoming: str, outgoing: str) -> str:
    return f"M::{junction_id}::{incoming}::{outgoing}"


def is_real_edge(edge_id: str, function: str) -> bool:
    return bool(edge_id) and not edge_id.startswith(":") and function != "internal"


def internal_edge_junction(edge_id: str) -> str:
    """``:J_12`` -> ``J``. Junction ids may contain underscores themselves."""
    return edge_id[1:].rsplit("_", 1)[0]


@dataclass
class Edge:
    edge_id: str
    function: str
    from_junction: str
    to_junction: str
    lanes: list[dict[str, str]] = field(default_factory=list)

    @property
    def real(self) -> bool:
        return is_real_edge(self.edge_id, self.function)

    def mean_lane(self, key: str) -> float:
        values = [float(lane[key]) for lane in self.lanes if lane.get(key, "") != ""]
        if not values:
            return float("nan")
        return float(sum(values) / len(values))


@dataclass
class Net:
    edges: dict[str, Edge]
    connections: list[dict[str, str]]
    junctions: dict[str, dict[str, str]]
    tl_logics: list[dict]


def parse_net(path: Path) -> Net:
    edges: dict[str, Edge] = {}
    connections: list[dict[str, str]] = []
    junctions: dict[str, dict[str, str]] = {}
    tl_logics: list[dict] = []
    current: dict[str, object] = {"edge": None, "tl": None}

    def start(name: str, attrs) -> None:
        if name == "edge":
            edge = Edge(attrs.get("id", ""), attrs.get("function", ""), attrs.get("from", ""), attrs.get("to", ""))
            if edge.edge_id in edges:
                raise RuntimeError(f"duplicate edge id {edge.edge_id}")
            edges[edge.edge_id] = edge
            current["edge"] = edge
        elif name == "lane" and current["edge"] is not None:
            current["edge"].lanes.append(dict(attrs))
        elif name == "junction":
            junctions[attrs.get("id", "")] = {"type": attrs.get("type", "")}
        elif name == "connection":
            connections.append(dict(attrs))
        elif name == "tlLogic":
            logic = {
                "id": attrs.get("id", ""),
                "type": attrs.get("type", ""),
                "programID": attrs.get("programID", ""),
                "offset": attrs.get("offset", ""),
                "phases": [],
            }
            tl_logics.append(logic)
            current["tl"] = logic
        elif name == "phase" and current["tl"] is not None:
            current["tl"]["phases"].append(dict(attrs))

    def end(name: str) -> None:
        if name == "edge":
            current["edge"] = None
        elif name == "tlLogic":
            current["tl"] = None

    parser = expat.ParserCreate()
    parser.StartElementHandler = start
    parser.EndElementHandler = end
    with path.open("rb") as handle:
        parser.ParseFile(handle)
    return Net(edges, connections, junctions, tl_logics)


def parse_tls_file(path: Path) -> dict:
    """WAUT, wautSwitch, wautJunction and any tlLogic in the additional file."""
    wauts: list[dict] = []
    tl_logics: list[dict] = []
    tags: Counter = Counter()
    current: dict[str, object] = {"waut": None, "tl": None}
    waut_junction_items: list[dict[str, str]] = []

    def start(name: str, attrs) -> None:
        tags[name] += 1
        if name == "WAUT":
            waut = {key: attrs[key] for key in attrs}
            waut["switches"] = []
            waut["junctions"] = []
            wauts.append(waut)
            current["waut"] = waut
        elif name == "wautSwitch":
            if current["waut"] is None:
                raise RuntimeError("wautSwitch outside WAUT")
            current["waut"]["switches"].append(dict(attrs))
        elif name == "wautJunction":
            waut_junction_items.append(dict(attrs))
        elif name == "tlLogic":
            logic = {key: attrs[key] for key in attrs}
            logic["phases"] = []
            tl_logics.append(logic)
            current["tl"] = logic
        elif name == "phase" and current["tl"] is not None:
            current["tl"]["phases"].append(dict(attrs))

    def end(name: str) -> None:
        if name == "WAUT":
            current["waut"] = None
        elif name == "tlLogic":
            current["tl"] = None

    parser = expat.ParserCreate()
    parser.StartElementHandler = start
    parser.EndElementHandler = end
    with path.open("rb") as handle:
        parser.ParseFile(handle)
    by_id = {waut.get("id", ""): waut for waut in wauts}
    orphan_junctions = []
    for item in waut_junction_items:
        waut = by_id.get(item.get("wautID", ""))
        if waut is None:
            orphan_junctions.append(item)
        else:
            waut["junctions"].append(item)
    return {
        "wauts": wauts,
        "tl_logics": tl_logics,
        "element_counts": dict(sorted(tags.items())),
        "orphan_waut_junctions": orphan_junctions,
    }


@dataclass
class LaneConnection:
    from_lane: str
    to_lane: str
    via: str
    chain: list[str]
    tl: str
    link_index: str
    state: str
    direction: str
    chain_connections: list[dict[str, str]]
    attrs: dict[str, str]


@dataclass
class Movement:
    junction_id: str
    incoming: str
    outgoing: str
    connections: list[LaneConnection] = field(default_factory=list)

    @property
    def movement_id(self) -> str:
        return movement_id(self.junction_id, self.incoming, self.outgoing)


def resolve_chain(
    connection: dict[str, str],
    internal_lane_edge: dict[str, tuple[str, str]],
    internal_connections: dict[tuple[str, str], list[dict[str, str]]],
    issues: list[str],
) -> tuple[list[str], list[dict[str, str]]]:
    """Follow via -> internal connection -> via ... until the outgoing real edge."""
    chain: list[str] = []
    steps: list[dict[str, str]] = []
    tokens = (connection.get("via") or "").split()
    current = tokens[0] if tokens else ""
    label = f"{connection.get('from')}_{connection.get('fromLane')}->{connection.get('to')}_{connection.get('toLane')}"
    if len(tokens) > 1:
        issues.append(f"{label}: via has several lanes {tokens}")
    while current:
        chain.append(current)
        if len(chain) > MAX_VIA_CHAIN:
            issues.append(f"{label}: via chain longer than {MAX_VIA_CHAIN}")
            break
        located = internal_lane_edge.get(current)
        if located is None:
            issues.append(f"{label}: via lane {current} is not an internal lane")
            break
        candidates = [
            item
            for item in internal_connections.get(located, [])
            if item.get("to") == connection.get("to") and item.get("toLane") == connection.get("toLane")
        ]
        if len(candidates) != 1:
            issues.append(f"{label}: internal lane {current} has {len(candidates)} continuations")
            break
        steps.append(candidates[0])
        current = candidates[0].get("via", "")
    return chain, steps


def build_movements(net: Net) -> dict[str, object]:
    edges = net.edges
    real_ids = [edge_id for edge_id, edge in edges.items() if edge.real]
    real_set = set(real_ids)
    r_index = {edge_id: index for index, edge_id in enumerate(real_ids)}
    internal_lane_edge: dict[str, tuple[str, str]] = {}
    internal_lane_length: dict[str, float] = {}
    index_mismatch: list[str] = []
    for edge_id, edge in edges.items():
        if edge.real:
            continue
        for position, lane in enumerate(edge.lanes):
            lane_id = lane.get("id", "")
            # A few netedit lanes repeat index="0"; connections use the lane position.
            if lane.get("index", str(position)) != str(position):
                index_mismatch.append(f"{lane_id}: index={lane.get('index')} position={position}")
            internal_lane_edge[lane_id] = (edge_id, str(position))
            internal_lane_length[lane_id] = float(lane.get("length", "nan"))
    internal_connections: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    real_connections: list[dict[str, str]] = []
    other_connections: list[dict[str, str]] = []
    for item in net.connections:
        source, target = item.get("from", ""), item.get("to", "")
        if source in real_set and target in real_set:
            real_connections.append(item)
        elif source.startswith(":"):
            internal_connections[(source, item.get("fromLane", ""))].append(item)
        else:
            other_connections.append(item)
    issues: list[str] = []
    groups: dict[tuple[str, str, str], Movement] = {}
    junction_mismatch: list[str] = []
    for item in real_connections:
        source, target = item["from"], item["to"]
        junction = edges[source].to_junction
        if edges[target].from_junction != junction:
            junction_mismatch.append(f"{source}->{target}: from.to={junction} to.from={edges[target].from_junction}")
        chain, steps = resolve_chain(item, internal_lane_edge, internal_connections, issues)
        for lane_id in chain:
            lane_junction = internal_edge_junction(internal_lane_edge.get(lane_id, (":?_0", ""))[0])
            if lane_junction != junction:
                junction_mismatch.append(f"{source}->{target}: via {lane_id} belongs to {lane_junction}")
        key = (junction, source, target)
        movement = groups.setdefault(key, Movement(junction, source, target))
        movement.connections.append(
            LaneConnection(
                from_lane=item.get("fromLane", ""),
                to_lane=item.get("toLane", ""),
                via=item.get("via", ""),
                chain=chain,
                tl=item.get("tl", ""),
                link_index=item.get("linkIndex", ""),
                state=item.get("state", ""),
                direction=item.get("dir", ""),
                chain_connections=steps,
                attrs=dict(item),
            )
        )
    ordered = sorted(groups.values(), key=lambda m: (r_index[m.incoming], r_index[m.outgoing], m.junction_id))
    for movement in ordered:
        movement.connections.sort(key=lambda c: (int(c.from_lane), int(c.to_lane), c.via))
    lane_owner: dict[str, set[int]] = defaultdict(set)
    for index, movement in enumerate(ordered):
        for connection in movement.connections:
            for lane_id in connection.chain:
                lane_owner[lane_id].add(index)
    return {
        "real_ids": real_ids,
        "r_index": r_index,
        "movements": ordered,
        "real_connections": real_connections,
        "internal_connections": internal_connections,
        "other_connections": other_connections,
        "internal_lane_edge": internal_lane_edge,
        "internal_lane_length": internal_lane_length,
        "lane_owner": lane_owner,
        "chain_issues": issues,
        "junction_mismatch": junction_mismatch,
        "internal_lane_index_mismatch": index_mismatch,
    }


def build_transitions(movements: list[Movement], edges: dict[str, Edge]) -> list[dict[str, object]]:
    by_incoming: dict[str, list[int]] = defaultdict(list)
    for index, movement in enumerate(movements):
        by_incoming[movement.incoming].append(index)
    transitions = []
    for source_index, source in enumerate(movements):
        for target_index in by_incoming.get(source.outgoing, []):
            if target_index == source_index:
                continue
            middle = edges[source.outgoing]
            length = middle.mean_lane("length")
            speed = middle.mean_lane("speed")
            if not (np.isfinite(length) and np.isfinite(speed) and speed > 0 and length >= 0):
                raise RuntimeError(f"bad length/speed on intermediate edge {middle.edge_id}")
            transitions.append(
                {
                    "source_index": source_index,
                    "target_index": target_index,
                    "source_movement_id": source.movement_id,
                    "target_movement_id": movements[target_index].movement_id,
                    "intermediate_edge_id": middle.edge_id,
                    "intermediate_edge_length": length,
                    "intermediate_edge_speed": speed,
                    "free_flow_travel_time": length / speed,
                }
            )
    transitions.sort(key=lambda t: (t["source_index"], t["target_index"]))
    return transitions


def program_states(net: Net) -> dict[str, dict[str, list[str]]]:
    """tls id -> program id -> list of phase state strings."""
    states: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for logic in net.tl_logics:
        states[logic["id"]][logic["programID"]] = [phase.get("state", "") for phase in logic["phases"]]
    return states


def link_phase_string(phases: list[str], link_index: int) -> str | None:
    if any(link_index >= len(state) for state in phases):
        return None
    return "".join(state[link_index] for state in phases)


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(temporary, path)


def write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def graph_signature(
    movement_ids: list[str],
    incoming: list[str],
    outgoing: list[str],
    lane_ids: list[str],
    lane_movement: list[int],
) -> str:
    digest = hashlib.sha256()
    for values in (movement_ids, incoming, outgoing, lane_ids, [str(v) for v in lane_movement]):
        digest.update("\x1f".join(values).encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def joined(values: list[str]) -> str:
    return "|".join(values)


def degree_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min": int(values.min()),
        "mean": round(float(values.mean()), 4),
        "max": int(values.max()),
        "zero_count": int((values == 0).sum()),
        "histogram": {str(k): int(v) for k, v in sorted(Counter(values.tolist()).items())},
    }


def r_graph_mapping(r_graph_path: Path, real_ids: list[str], movements: list[Movement]) -> dict[str, object]:
    with np.load(r_graph_path, allow_pickle=False) as payload:
        r_ids = [str(value) for value in payload["edge_ids"].tolist()]
        adjacency = payload["adjacency"]
        r_num_edges = int(payload["num_graph_edges"])
    r_pairs = {(r_ids[int(i)], r_ids[int(j)]) for i, j in np.argwhere(adjacency > 0)}
    pair_count = Counter((m.incoming, m.outgoing) for m in movements)
    id_count = Counter(m.movement_id for m in movements)
    m_pairs = set(pair_count)
    r_set = set(r_ids)
    missing_in_m = sorted(r_pairs - m_pairs)
    extra_in_m = sorted(m_pairs - r_pairs)
    duplicate_pairs = sorted(pair for pair, count in pair_count.items() if count > 1)
    duplicate_ids = sorted(mid for mid, count in id_count.items() if count > 1)
    bad_incoming = sorted({m.incoming for m in movements if m.incoming not in r_set})
    bad_outgoing = sorted({m.outgoing for m in movements if m.outgoing not in r_set})
    passed = (
        not missing_in_m
        and not extra_in_m
        and not duplicate_pairs
        and not duplicate_ids
        and not bad_incoming
        and not bad_outgoing
        and r_ids == real_ids
        and len(movements) == len(r_pairs) == r_num_edges
    )
    return {
        "r_graph_path": str(r_graph_path),
        "r_num_nodes": len(r_ids),
        "r_num_graph_edges": r_num_edges,
        "r_edge_order_equals_net_real_edge_order": r_ids == real_ids,
        "num_movements": len(movements),
        "num_unique_movement_edge_pairs": len(m_pairs),
        "r_links_missing_in_m": [list(pair) for pair in missing_in_m],
        "m_pairs_not_in_r": [list(pair) for pair in extra_in_m],
        "duplicate_movement_edge_pairs": [list(pair) for pair in duplicate_pairs],
        "duplicate_movement_ids": duplicate_ids,
        "incoming_edges_not_in_r_nodes": bad_incoming,
        "outgoing_edges_not_in_r_nodes": bad_outgoing,
        "status": "PASS" if passed else "FAIL",
    }


def build_all(net_file: Path, tls_file: Path, graph_dir: Path) -> int:
    net = parse_net(net_file)
    built = build_movements(net)
    real_ids: list[str] = built["real_ids"]
    edges_ref, _lane_to_edge, internal_lanes_ref = full.load_network(net_file)
    if edges_ref != real_ids:
        raise RuntimeError("real edge order differs from generate_r_level_full_flow.load_network")
    movements: list[Movement] = built["movements"]
    edges = net.edges
    r_index: dict[str, int] = built["r_index"]
    lane_owner: dict[str, set[int]] = built["lane_owner"]
    internal_lane_length: dict[str, float] = built["internal_lane_length"]
    num_nodes = len(movements)
    if num_nodes == 0:
        raise RuntimeError("no movements found")
    transitions = build_transitions(movements, edges)
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.uint8)
    for item in transitions:
        adjacency[item["source_index"], item["target_index"]] = 1
    if int(adjacency.sum()) != len(transitions):
        raise RuntimeError("duplicate M transitions")
    if int(np.trace(adjacency)) != 0:
        raise RuntimeError("self-loop in M graph")
    edge_index = np.array(
        [[t["source_index"] for t in transitions], [t["target_index"] for t in transitions]], dtype=np.int64
    ).reshape(2, len(transitions))

    unique_lanes = sorted((lane for lane, owners in lane_owner.items() if len(owners) == 1), key=lambda l: (min(lane_owner[l]), l))
    ambiguous_lanes = sorted(lane for lane, owners in lane_owner.items() if len(owners) > 1)
    lane_movement = [next(iter(lane_owner[lane])) for lane in unique_lanes]
    all_internal_lanes = set(built["internal_lane_edge"])
    if all_internal_lanes != set(internal_lanes_ref):
        raise RuntimeError("internal lane set differs from generate_r_level_full_flow.load_network")
    unmapped_internal = sorted(all_internal_lanes - set(lane_owner))

    movement_ids = [m.movement_id for m in movements]
    incoming_ids = [m.incoming for m in movements]
    outgoing_ids = [m.outgoing for m in movements]
    junction_ids = [m.junction_id for m in movements]
    signature = graph_signature(movement_ids, incoming_ids, outgoing_ids, unique_lanes, lane_movement)

    tl_states = program_states(net)
    tls_info = parse_tls_file(tls_file) if tls_file.is_file() else None

    # Nodes table.
    node_header = [
        "movement_index", "movement_id", "junction_id", "junction_type",
        "incoming_edge_id", "outgoing_edge_id",
        "incoming_r_index", "outgoing_r_index",
        "incoming_from_junction", "incoming_to_junction",
        "outgoing_from_junction", "outgoing_to_junction",
        "incoming_length", "outgoing_length", "incoming_speed", "outgoing_speed",
        "incoming_num_lanes", "outgoing_num_lanes",
        "turn_direction", "lane_connection_count",
        "from_lane_indices", "to_lane_indices", "via_lane_ids", "internal_lane_ids",
        "connector_length_min", "connector_length_max",
        "tls_ids", "connection_tls_ids", "link_indices", "connection_states", "connection_dirs",
        "signal_control",
    ]
    node_rows = []
    turn_counter: Counter = Counter()
    lane_dir_counter: Counter = Counter()
    mixed_turns = []
    signal_counter: Counter = Counter()
    connector_lengths = []
    chain_lengths = Counter()
    conn_rows = []
    for index, m in enumerate(movements):
        incoming, outgoing = edges[m.incoming], edges[m.outgoing]
        dirs = sorted({c.direction for c in m.connections})
        turn = joined(dirs)
        if len(dirs) > 1:
            mixed_turns.append(m.movement_id)
        turn_counter[turn] += 1
        for c in m.connections:
            lane_dir_counter[c.direction] += 1
            chain_lengths[len(c.chain)] += 1
        lengths = [sum(internal_lane_length[l] for l in c.chain) for c in m.connections]
        connector_lengths.extend(lengths)
        tls = sorted({c.tl for c in m.connections if c.tl})
        controlled = sum(1 for c in m.connections if c.tl)
        control = "all" if controlled == len(m.connections) else ("none" if controlled == 0 else "partial")
        signal_counter[control] += 1
        internal_ids = []
        for c in m.connections:
            for lane in c.chain:
                if lane not in internal_ids:
                    internal_ids.append(lane)
        node_rows.append([
            index, m.movement_id, m.junction_id, net.junctions.get(m.junction_id, {}).get("type", ""),
            m.incoming, m.outgoing, r_index[m.incoming], r_index[m.outgoing],
            incoming.from_junction, incoming.to_junction, outgoing.from_junction, outgoing.to_junction,
            f"{incoming.mean_lane('length'):.4f}", f"{outgoing.mean_lane('length'):.4f}",
            f"{incoming.mean_lane('speed'):.4f}", f"{outgoing.mean_lane('speed'):.4f}",
            len(incoming.lanes), len(outgoing.lanes),
            turn, len(m.connections),
            joined([c.from_lane for c in m.connections]),
            joined([c.to_lane for c in m.connections]),
            joined([c.via for c in m.connections]),
            joined(internal_ids),
            f"{min(lengths):.4f}", f"{max(lengths):.4f}",
            joined(tls),
            joined([c.tl or "-" for c in m.connections]),
            joined([c.link_index or "-" for c in m.connections]),
            joined([c.state for c in m.connections]),
            joined([c.direction for c in m.connections]),
            control,
        ])
        for c in m.connections:
            conn_rows.append((index, m, c))

    # Signal connections and per-link phase strings.
    signal_header = [
        "movement_index", "movement_id", "junction_id", "tls_id", "link_index",
        "incoming_edge_id", "outgoing_edge_id", "from_lane", "to_lane", "via_lane_id",
        "connection_level", "dir", "state", "programs", "phase_states_by_program",
    ]
    signal_rows = []
    link_issues = []
    movement_link_states: dict[str, dict[str, dict[str, list[str]]]] = {}
    internal_signal_connections = 0
    for index, m, c in conn_rows:
        records = [("real", c.tl, c.link_index, c.via, c.state, c.direction)]
        for step in c.chain_connections:
            if step.get("tl"):
                internal_signal_connections += 1
                records.append(("internal", step.get("tl", ""), step.get("linkIndex", ""), step.get("via", ""), step.get("state", ""), step.get("dir", "")))
        for level, tl, link, via, state, direction in records:
            if not tl:
                continue
            programs = tl_states.get(tl)
            phase_parts = []
            if programs is None:
                link_issues.append(f"{m.movement_id}: tls {tl} has no tlLogic in the net")
            elif link == "":
                link_issues.append(f"{m.movement_id}: tls {tl} without linkIndex")
            else:
                for program_id in sorted(programs):
                    text = link_phase_string(programs[program_id], int(link))
                    if text is None:
                        link_issues.append(f"{m.movement_id}: linkIndex {link} outside tls {tl} program {program_id}")
                        continue
                    phase_parts.append(f"{program_id}:{text}")
                    movement_link_states.setdefault(m.movement_id, {}).setdefault(tl, {}).setdefault(program_id, []).append(
                        f"{link}={text}"
                    )
            signal_rows.append([
                index, m.movement_id, m.junction_id, tl, link, m.incoming, m.outgoing,
                c.from_lane, c.to_lane, via, level, direction, state,
                joined(sorted(programs)) if programs else "", ";".join(phase_parts),
            ])

    # Signal programs JSON.
    logics_json = []
    for logic in net.tl_logics:
        lengths = {len(phase.get("state", "")) for phase in logic["phases"]}
        durations = []
        for phase in logic["phases"]:
            try:
                durations.append(float(phase.get("duration", "nan")))
            except ValueError:
                durations.append(float("nan"))
        logics_json.append({
            "id": logic["id"],
            "type": logic["type"],
            "programID": logic["programID"],
            "offset": logic["offset"],
            "num_phases": len(logic["phases"]),
            "cycle_seconds": float(sum(durations)),
            "state_lengths": sorted(lengths),
            "phases": logic["phases"],
        })
    programs_by_tls = {tl: sorted(programs) for tl, programs in sorted(tl_states.items())}
    waut_checks = []
    notes = []
    if tls_info is None:
        notes.append(f"tls file not found: {tls_file}; WAUT information unavailable")
    else:
        if not tls_info["wauts"]:
            notes.append("tls file has no WAUT element")
        if not tls_info["tl_logics"]:
            notes.append("tls file has no tlLogic; all signal programs come from the net file")
        for waut in tls_info["wauts"]:
            targets = {waut.get("startProg", "")} | {s.get("to", "") for s in waut["switches"]}
            for junction in waut["junctions"]:
                tl = junction.get("junctionID", "")
                have = set(programs_by_tls.get(tl, []))
                missing = sorted(targets - have)
                if missing:
                    waut_checks.append({"waut": waut.get("id", ""), "tls_id": tl, "missing_programs": missing})
        if tls_info["orphan_waut_junctions"]:
            notes.append(f"{len(tls_info['orphan_waut_junctions'])} wautJunction entries reference an unknown WAUT")
    tls_with_program_1 = sorted(tl for tl, programs in programs_by_tls.items() if len(programs) > 1)
    waut_tls = sorted({j.get("junctionID", "") for w in (tls_info or {}).get("wauts", []) for j in w["junctions"]})
    signal_payload = {
        "net_file": str(net_file),
        "tls_file": str(tls_file),
        "net_tl_logics": logics_json,
        "tls_file_tl_logics": (tls_info or {}).get("tl_logics", []),
        "tls_file_element_counts": (tls_info or {}).get("element_counts", {}),
        "wauts": (tls_info or {}).get("wauts", []),
        "programs_by_tls": programs_by_tls,
        "tls_with_several_programs": tls_with_program_1,
        "waut_junctions": waut_tls,
        "waut_junctions_without_switch_programs": waut_checks,
        "multi_program_tls_not_in_waut": sorted(set(tls_with_program_1) - set(waut_tls)),
        "link_issues": link_issues,
        "notes": notes,
        "movement_link_phase_states": movement_link_states,
    }

    # Graph statistics.
    out_degree = adjacency.sum(axis=1).astype(np.int64)
    in_degree = adjacency.sum(axis=0).astype(np.int64)
    isolated = np.flatnonzero((in_degree == 0) & (out_degree == 0))
    weak_count, weak_sizes, _members = rgraph.weakly_connected(adjacency)
    strong_count = rgraph.strongly_connected_count(adjacency)
    r_graph_path = graph_dir / "r_graph.npz"
    mapping = r_graph_mapping(r_graph_path, real_ids, movements)
    tls_junction_movements = [m for m in movements if net.junctions.get(m.junction_id, {}).get("type") == "traffic_light"]
    tls_movements_uncontrolled = [m.movement_id for m in tls_junction_movements if not any(c.tl for c in m.connections)]
    signal_mismatch_junction = sorted({
        m.movement_id for m in movements for c in m.connections if c.tl and c.tl != m.junction_id
    })
    real_lengths = np.array([edges[e].mean_lane("length") for e in real_ids])
    audit = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "net_file": str(net_file),
        "tls_file": str(tls_file),
        "graph_signature": signature,
        "num_real_edges": len(real_ids),
        "num_lane_connections_total": len(net.connections),
        "num_real_to_real_lane_connections": len(built["real_connections"]),
        "num_internal_lane_connections": sum(len(v) for v in built["internal_connections"].values()),
        "num_other_lane_connections": len(built["other_connections"]),
        "num_nodes": num_nodes,
        "num_graph_edges": len(transitions),
        "graph_density": len(transitions) / float(num_nodes * num_nodes),
        "self_loops": int(np.trace(adjacency)),
        "reciprocal_pairs": int(np.triu(np.logical_and(adjacency, adjacency.T), k=1).sum()),
        "turn_direction_distribution": dict(sorted(turn_counter.items())),
        "lane_connection_dir_distribution": dict(sorted(lane_dir_counter.items())),
        "movements_with_mixed_dirs": mixed_turns,
        "lane_connections_per_movement": {str(k): v for k, v in sorted(Counter(len(m.connections) for m in movements).items())},
        "in_degree": degree_stats(in_degree),
        "out_degree": degree_stats(out_degree),
        "isolated_nodes": [movement_ids[int(i)] for i in isolated],
        "zero_in_degree_count": int((in_degree == 0).sum()),
        "zero_out_degree_count": int((out_degree == 0).sum()),
        "weakly_connected_components": weak_count,
        "weak_component_sizes_top": sorted(weak_sizes, reverse=True)[:20],
        "strongly_connected_components": strong_count,
        "via_lanes": {
            "internal_lanes_in_net": len(all_internal_lanes),
            "internal_lanes_mapped_to_one_movement": len(unique_lanes),
            "internal_lanes_ambiguous": ambiguous_lanes,
            "internal_lanes_unmapped": unmapped_internal,
            "via_chain_length_distribution": {str(k): v for k, v in sorted(chain_lengths.items())},
            "lane_connections_without_via": sum(1 for m in movements for c in m.connections if not c.chain),
            "movements_without_via": [m.movement_id for m in movements if not any(c.chain for c in m.connections)],
            "chain_issues": built["chain_issues"],
            "junction_mismatches": built["junction_mismatch"],
            "internal_lane_index_attribute_mismatches": built["internal_lane_index_mismatch"],
            "connector_length_m": {
                "min": round(float(min(connector_lengths)), 4),
                "mean": round(float(np.mean(connector_lengths)), 4),
                "max": round(float(max(connector_lengths)), 4),
            },
        },
        "short_real_edges": {f"lt_{int(t)}m": int((real_lengths < t).sum()) for t in SHORT_EDGE_METERS},
        "transition_free_flow_time_s": {
            "min": round(float(min(t["free_flow_travel_time"] for t in transitions)), 4),
            "mean": round(float(np.mean([t["free_flow_travel_time"] for t in transitions])), 4),
            "max": round(float(max(t["free_flow_travel_time"] for t in transitions)), 4),
        },
        "signal": {
            "net_tl_logic_count": len(net.tl_logics),
            "tls_count": len(programs_by_tls),
            "traffic_light_junctions": sum(1 for j in net.junctions.values() if j["type"] == "traffic_light"),
            "signal_controlled_lane_connections": sum(1 for m in movements for c in m.connections if c.tl),
            "signal_controlled_internal_connections": internal_signal_connections,
            "movement_signal_control": dict(sorted(signal_counter.items())),
            "movement_signal_coverage": round((signal_counter["all"] + signal_counter["partial"]) / num_nodes, 6),
            "movements_at_traffic_light_junctions": len(tls_junction_movements),
            "movements_at_traffic_light_junctions_without_tl": tls_movements_uncontrolled,
            "movements_with_tls_id_different_from_junction": signal_mismatch_junction,
            "link_issues": link_issues,
            "waut_junction_count": len(waut_tls),
            "waut_junctions_without_switch_programs": waut_checks,
            "tls_with_several_programs": len(tls_with_program_1),
            "notes": notes,
        },
        "r_graph_mapping": mapping,
    }
    hard_failures = []
    if mapping["status"] != "PASS":
        hard_failures.append("R-graph mapping failed")
    if ambiguous_lanes:
        hard_failures.append(f"{len(ambiguous_lanes)} internal lanes map to several movements")
    if built["chain_issues"]:
        hard_failures.append(f"{len(built['chain_issues'])} via chain issues")
    if built["junction_mismatch"]:
        hard_failures.append(f"{len(built['junction_mismatch'])} junction mismatches")
    if link_issues:
        hard_failures.append(f"{len(link_issues)} TLS link issues")
    audit["status"] = "PASS" if not hard_failures else "FAIL"
    audit["failures"] = hard_failures

    graph_dir.mkdir(parents=True, exist_ok=True)
    npz_path = graph_dir / "m_graph.npz"
    temporary = graph_dir / "m_graph.tmp.npz"
    conn_movement = np.array([index for index, _m, _c in conn_rows], dtype=np.int64)
    np.savez_compressed(
        temporary,
        movement_ids=np.array(movement_ids, dtype=np.str_),
        incoming_edge_ids=np.array(incoming_ids, dtype=np.str_),
        outgoing_edge_ids=np.array(outgoing_ids, dtype=np.str_),
        junction_ids=np.array(junction_ids, dtype=np.str_),
        incoming_r_index=np.array([r_index[m.incoming] for m in movements], dtype=np.int64),
        outgoing_r_index=np.array([r_index[m.outgoing] for m in movements], dtype=np.int64),
        turn_directions=np.array([row[18] for row in node_rows], dtype=np.str_),
        lane_connection_count=np.array([len(m.connections) for m in movements], dtype=np.int64),
        adjacency=adjacency,
        edge_index=edge_index,
        transition_edge_ids=np.array([t["intermediate_edge_id"] for t in transitions], dtype=np.str_),
        transition_distance=np.array([t["intermediate_edge_length"] for t in transitions], dtype=np.float64),
        transition_speed=np.array([t["intermediate_edge_speed"] for t in transitions], dtype=np.float64),
        transition_free_flow_time=np.array([t["free_flow_travel_time"] for t in transitions], dtype=np.float64),
        num_nodes=np.int64(num_nodes),
        num_graph_edges=np.int64(len(transitions)),
        r_edge_ids=np.array(real_ids, dtype=np.str_),
        internal_lane_ids=np.array(unique_lanes, dtype=np.str_),
        internal_lane_movement_index=np.array(lane_movement, dtype=np.int64),
        ambiguous_internal_lane_ids=np.array(ambiguous_lanes, dtype=np.str_),
        connection_movement_index=conn_movement,
        connection_from_lane=np.array([c.from_lane for _i, _m, c in conn_rows], dtype=np.str_),
        connection_to_lane=np.array([c.to_lane for _i, _m, c in conn_rows], dtype=np.str_),
        connection_via=np.array([c.via for _i, _m, c in conn_rows], dtype=np.str_),
        connection_tls_id=np.array([c.tl for _i, _m, c in conn_rows], dtype=np.str_),
        connection_link_index=np.array([c.link_index for _i, _m, c in conn_rows], dtype=np.str_),
        connection_state=np.array([c.state for _i, _m, c in conn_rows], dtype=np.str_),
        connection_dir=np.array([c.direction for _i, _m, c in conn_rows], dtype=np.str_),
        graph_signature=np.array(signature),
    )
    os.replace(temporary, npz_path)
    write_csv(graph_dir / "m_nodes.csv", node_header, node_rows)
    write_csv(
        graph_dir / "m_transitions.csv",
        [
            "transition_index", "source_movement_index", "target_movement_index",
            "source_movement_id", "target_movement_id", "intermediate_edge_id",
            "intermediate_edge_length", "intermediate_edge_speed", "free_flow_travel_time",
        ],
        [
            [
                position, t["source_index"], t["target_index"], t["source_movement_id"], t["target_movement_id"],
                t["intermediate_edge_id"], f"{t['intermediate_edge_length']:.4f}",
                f"{t['intermediate_edge_speed']:.4f}", f"{t['free_flow_travel_time']:.6f}",
            ]
            for position, t in enumerate(transitions)
        ],
    )
    write_csv(graph_dir / "m_signal_connections.csv", signal_header, signal_rows)
    write_json(graph_dir / "m_signal_programs.json", signal_payload)
    write_json(graph_dir / "m_graph_audit.json", audit)
    write_text(graph_dir / "m_graph_audit.md", audit_markdown(audit))
    reload_and_check(npz_path, graph_dir, movement_ids, adjacency, edge_index, signature)
    log(f"nodes={num_nodes} graph_edges={len(transitions)} weak_components={weak_count} strong_components={strong_count}")
    log(f"isolated={len(isolated)} zero_in={audit['zero_in_degree_count']} zero_out={audit['zero_out_degree_count']}")
    log(f"turns={audit['turn_direction_distribution']}")
    log(f"via lanes mapped={len(unique_lanes)} ambiguous={len(ambiguous_lanes)} unmapped={len(unmapped_internal)} chains={audit['via_lanes']['via_chain_length_distribution']}")
    log(f"signal control={audit['signal']['movement_signal_control']} link_issues={len(link_issues)} waut_tls={len(waut_tls)}")
    log(
        f"r_graph mapping {mapping['status']}: r_edges={mapping['r_num_graph_edges']} movements={mapping['num_movements']} "
        f"missing_in_m={len(mapping['r_links_missing_in_m'])} extra_in_m={len(mapping['m_pairs_not_in_r'])}"
    )
    log(f"audit status={audit['status']} failures={hard_failures or 'none'} output={graph_dir}")
    return 0 if audit["status"] == "PASS" else 2


def reload_and_check(
    npz_path: Path,
    graph_dir: Path,
    movement_ids: list[str],
    adjacency: np.ndarray,
    edge_index: np.ndarray,
    signature: str,
) -> None:
    with np.load(npz_path, allow_pickle=False) as payload:
        ids = [str(v) for v in payload["movement_ids"].tolist()]
        loaded_adj = payload["adjacency"]
        loaded_index = payload["edge_index"]
        if payload["movement_ids"].dtype.kind != "U":
            raise RuntimeError("movement_ids are not strings")
        if int(payload["num_nodes"]) != len(movement_ids) or int(payload["num_graph_edges"]) != edge_index.shape[1]:
            raise RuntimeError("reloaded counts differ")
        if str(payload["graph_signature"]) != signature:
            raise RuntimeError("reloaded signature differs")
        for key in ("transition_edge_ids", "transition_distance", "transition_free_flow_time"):
            if payload[key].shape != (edge_index.shape[1],):
                raise RuntimeError(f"{key} shape {payload[key].shape}")
    if ids != movement_ids:
        raise RuntimeError("reloaded movement ids differ")
    if loaded_adj.dtype != np.uint8 or not np.array_equal(loaded_adj, adjacency):
        raise RuntimeError("reloaded adjacency differs")
    if loaded_index.shape != (2, int(adjacency.sum())) or not np.array_equal(loaded_index, edge_index):
        raise RuntimeError("reloaded edge_index differs")
    rebuilt = np.zeros_like(adjacency)
    rebuilt[loaded_index[0], loaded_index[1]] = 1
    if not np.array_equal(rebuilt, adjacency):
        raise RuntimeError("edge_index does not reproduce adjacency")
    with (graph_dir / "m_nodes.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [row["movement_id"] for row in rows] != movement_ids:
        raise RuntimeError("m_nodes.csv order differs from m_graph.npz")
    if [int(row["movement_index"]) for row in rows] != list(range(len(movement_ids))):
        raise RuntimeError("m_nodes.csv movement_index is not 0..M-1")
    with (graph_dir / "m_transitions.csv").open(encoding="utf-8", newline="") as handle:
        trans = list(csv.DictReader(handle))
    if len(trans) != edge_index.shape[1]:
        raise RuntimeError("m_transitions.csv row count differs")
    for position, row in enumerate(trans):
        if int(row["source_movement_index"]) != int(edge_index[0, position]) or int(row["target_movement_index"]) != int(edge_index[1, position]):
            raise RuntimeError("m_transitions.csv differs from edge_index")


def audit_markdown(audit: dict) -> str:
    mapping = audit["r_graph_mapping"]
    via = audit["via_lanes"]
    signal = audit["signal"]
    lines = [
        "# Xuancheng26 M-level graph audit",
        "",
        f"- Status: **{audit['status']}**" + (f" ({'; '.join(audit['failures'])})" if audit["failures"] else ""),
        f"- Net: `{audit['net_file']}`",
        f"- TLS: `{audit['tls_file']}`",
        f"- Graph signature: `{audit['graph_signature']}`",
        "",
        "## Size",
        "",
        "| item | value |",
        "| --- | --- |",
        f"| real edges (R nodes) | {audit['num_real_edges']} |",
        f"| real-to-real lane connections | {audit['num_real_to_real_lane_connections']} |",
        f"| internal lane connections | {audit['num_internal_lane_connections']} |",
        f"| Movement nodes | {audit['num_nodes']} |",
        f"| M directed edges | {audit['num_graph_edges']} |",
        f"| density | {audit['graph_density']:.8f} |",
        f"| self-loops | {audit['self_loops']} |",
        f"| reciprocal pairs | {audit['reciprocal_pairs']} |",
        f"| isolated nodes | {len(audit['isolated_nodes'])} |",
        f"| zero in-degree / zero out-degree | {audit['zero_in_degree_count']} / {audit['zero_out_degree_count']} |",
        f"| weak / strong components | {audit['weakly_connected_components']} / {audit['strongly_connected_components']} |",
        "",
        f"In-degree min/mean/max: {audit['in_degree']['min']}/{audit['in_degree']['mean']}/{audit['in_degree']['max']}. "
        f"Out-degree min/mean/max: {audit['out_degree']['min']}/{audit['out_degree']['mean']}/{audit['out_degree']['max']}.",
        "",
        "## Turn directions (per Movement)",
        "",
        "| dir | movements |",
        "| --- | --- |",
        *[f"| {key} | {value} |" for key, value in audit["turn_direction_distribution"].items()],
        "",
        f"Movements whose lane connections carry more than one `dir`: {len(audit['movements_with_mixed_dirs'])}.",
        "",
        "## Via lanes",
        "",
        f"- Internal lanes in net: {via['internal_lanes_in_net']}",
        f"- Mapped to exactly one Movement: {via['internal_lanes_mapped_to_one_movement']}",
        f"- Ambiguous: {len(via['internal_lanes_ambiguous'])}; unmapped: {len(via['internal_lanes_unmapped'])}",
        f"- Via chain length distribution (lanes per lane connection): {via['via_chain_length_distribution']}",
        f"- Lane connections without via: {via['lane_connections_without_via']}",
        f"- Chain issues: {len(via['chain_issues'])}; junction mismatches: {len(via['junction_mismatches'])}",
        f"- Internal lanes whose `index` attribute differs from their position (resolved by position): "
        f"{len(via['internal_lane_index_attribute_mismatches'])}",
        f"- Connector length (m) min/mean/max: {via['connector_length_m']['min']}/{via['connector_length_m']['mean']}/{via['connector_length_m']['max']}",
        f"- Real edges shorter than 5/10/20 m: {audit['short_real_edges']}",
        "",
        "## Signals",
        "",
        f"- tlLogic programs in net: {signal['net_tl_logic_count']} for {signal['tls_count']} TLS; traffic_light junctions: {signal['traffic_light_junctions']}",
        f"- Signal-controlled lane connections: {signal['signal_controlled_lane_connections']}; internal signal connections: {signal['signal_controlled_internal_connections']}",
        f"- Movement control (all/partial/none lane connections on a TLS): {signal['movement_signal_control']}",
        f"- Movement signal coverage: {signal['movement_signal_coverage']}",
        f"- Movements at traffic_light junctions without tl: {len(signal['movements_at_traffic_light_junctions_without_tl'])}",
        f"- Movements whose tl differs from the junction id: {len(signal['movements_with_tls_id_different_from_junction'])}",
        f"- WAUT junctions: {signal['waut_junction_count']}; TLS with several programs: {signal['tls_with_several_programs']}",
        f"- WAUT junctions missing a switch program: {len(signal['waut_junctions_without_switch_programs'])}",
        f"- linkIndex issues: {len(signal['link_issues'])}",
        *[f"- Note: {note}" for note in signal["notes"]],
        "",
        "## R-graph mapping",
        "",
        f"- Status: **{mapping['status']}**",
        f"- R nodes / R directed edges: {mapping['r_num_nodes']} / {mapping['r_num_graph_edges']}",
        f"- Movements / unique edge pairs: {mapping['num_movements']} / {mapping['num_unique_movement_edge_pairs']}",
        f"- R edge order equals net real-edge order: {mapping['r_edge_order_equals_net_real_edge_order']}",
        f"- R links missing in M: {len(mapping['r_links_missing_in_m'])}",
        f"- M pairs not in R: {len(mapping['m_pairs_not_in_r'])}",
        f"- Duplicate pairs / ids: {len(mapping['duplicate_movement_edge_pairs'])} / {len(mapping['duplicate_movement_ids'])}",
        f"- Incoming / outgoing edges outside R nodes: {len(mapping['incoming_edges_not_in_r_nodes'])} / {len(mapping['outgoing_edges_not_in_r_nodes'])}",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Xuancheng26 directed Movement-level graph")
    prep_paths.add_3s_path_arguments(parser, net=True, tls=True)
    args = parser.parse_args(argv)
    net_file = prep_paths.resolved(args.net_file)
    tls_file = prep_paths.resolved(args.tls_file)
    root = prep_paths.refuse_code_root(args.training_root)
    graph_dir = root / "graph"
    log(f"net_file={net_file}")
    log(f"tls_file={tls_file}")
    log(f"graph_dir={graph_dir}")
    if not (graph_dir / "r_graph.npz").is_file():
        raise FileNotFoundError(f"missing R graph {graph_dir / 'r_graph.npz'}")
    return build_all(net_file, tls_file, graph_dir)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
