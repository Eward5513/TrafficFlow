"""Build a directed road-level (R-only) subgraph from SUMO topology.

Nodes are the unique SUMO edge IDs in a subgraph listing. Directed R-graph
edges come only from ``<connection from="..." to="...">`` records in the
canonical ``net_tls`` network whose both endpoints are subgraph edges.

Adjacency convention
--------------------
``A[i, j] = 1`` if and only if node ``i`` is the source / from / upstream
road and node ``j`` is the target / to / downstream road. Rows are sources,
columns are targets. The saved matrix is raw, binary, directed, and has no
added self-loops, symmetrization, or normalization.

This script does not read trajectories, vehicle IDs, penetration samples,
or traffic counts, and it does not build M-only or MR graphs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


SCRIPT_VERSION = "1.0.0"
NODE_TYPE = "R"
NODE_ID_PREFIX = "R::"
RANDOM_SEED_DEFAULT = 42
EXAMPLE_LIMIT = 8
SAMPLE_EDGE_COUNT = 5

# A[i, j] = 1 iff source/from/upstream i connects to target/to/downstream j.
ADJACENCY_CONVENTION = (
    "A[i, j] = 1 iff node i is the source/from/upstream road and node j is "
    "the target/to/downstream road; rows are sources, columns are targets"
)

PREFERRED_CONNECTION_COLUMNS = (
    "from",
    "to",
    "fromLane",
    "toLane",
    "via",
    "tl",
    "linkIndex",
    "dir",
    "state",
)

OUTPUT_ARTIFACTS = (
    "r_nodes.csv",
    "r_edges.csv",
    "r_sumo_connections.csv",
    "r_adjacency.npy",
    "r_adjacency_sparse.npz",
    "r_edge_index.npy",
    "r_graph_metadata.json",
    "r_graph_validation.json",
    "r_graph_preview.png",
    "r_graph_preview_unlabeled.png",
)

GRAPH_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = GRAPH_DIR.parent
PROJECT_ROOT = ANALYSIS_DIR.parent
DEFAULT_SUBGRAPH_FILE = ANALYSIS_DIR / "simulation" / "data" / "subgraph.txt"
DEFAULT_NET_FILE = ANALYSIS_DIR / "road_network" / "net_tls.net.xml"
DEFAULT_OUTPUT_DIR = GRAPH_DIR / "r_graph"

R_NODE_FIELDNAMES = (
    "node_index",
    "node_id",
    "node_type",
    "edge_id",
    "from_junction",
    "to_junction",
    "num_lanes",
    "length",
    "speed",
    "priority",
    "edge_type",
)

R_EDGE_FIELDNAMES = (
    "source_index",
    "target_index",
    "source_node_id",
    "target_node_id",
    "source_edge_id",
    "target_edge_id",
    "raw_connection_count",
    "tls_ids",
    "link_indices",
    "directions",
)


class GraphError(Exception):
    pass


@dataclass
class SubgraphSpec:
    path: Path
    raw_line_count: int
    nonempty_line_count: int
    unique_ids: list[str]
    unique_set: set[str]
    duplicate_ids: list[str]
    duplicate_records: list[dict[str, object]]
    empty_line_count: int
    sha256: str
    size_bytes: int
    mtime_ns: int


@dataclass
class LaneRecord:
    lane_id: str
    index: str | None
    length_raw: str | None
    speed_raw: str | None
    shape: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class EdgeRecord:
    edge_id: str
    from_junction: str | None
    to_junction: str | None
    priority: str | None
    edge_type: str | None
    function: str | None
    lanes: list[LaneRecord] = field(default_factory=list)


@dataclass
class ConnectionRecord:
    attrib: dict[str, str]
    order: int

    @property
    def from_edge(self) -> str:
        return self.attrib.get("from", "")

    @property
    def to_edge(self) -> str:
        return self.attrib.get("to", "")


@dataclass
class RNode:
    node_index: int
    node_id: str
    node_type: str
    edge_id: str
    from_junction: str
    to_junction: str
    num_lanes: int
    length: str
    speed: str
    priority: str
    edge_type: str
    length_rule: str
    speed_rule: str
    found_in_net: bool
    position: tuple[float, float] | None


@dataclass
class REdge:
    source_index: int
    target_index: int
    source_node_id: str
    target_node_id: str
    source_edge_id: str
    target_edge_id: str
    raw_connection_count: int
    tls_ids: str
    link_indices: str
    directions: str
    connections: list[ConnectionRecord]


@dataclass
class RGraph:
    nodes: list[RNode]
    edges: list[REdge]
    connections: list[ConnectionRecord]
    connection_columns: list[str]
    adjacency: np.ndarray
    adjacency_sparse: sparse.csr_matrix
    edge_index: np.ndarray
    missing_from_net: list[str]
    extra_net_only_nodes: list[str]
    incomplete_connections: int
    length_speed_stats: dict[str, object]


def strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": repo_rel(path),
        "sha256": sha256_file(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def node_id_for(edge_id: str) -> str:
    return f"{NODE_ID_PREFIX}{edge_id}"


def parse_shape(raw: str | None) -> list[tuple[float, float]]:
    if not raw:
        return []
    points: list[tuple[float, float]] = []
    for token in raw.split():
        if "," not in token:
            continue
        xs, ys = token.split(",", 1)
        try:
            points.append((float(xs), float(ys)))
        except ValueError:
            continue
    return points


def polyline_midpoint(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not points:
        return None
    if len(points) == 1:
        return points[0]
    lengths = [0.0]
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
        lengths.append(total)
    if total <= 0.0:
        return points[len(points) // 2]
    half = total / 2.0
    for i in range(1, len(points)):
        if lengths[i] >= half:
            span = lengths[i] - lengths[i - 1]
            t = 0.0 if span == 0.0 else (half - lengths[i - 1]) / span
            x0, y0 = points[i - 1]
            x1, y1 = points[i]
            return (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
    return points[-1]


def unique_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value == "" or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def join_unique(values: list[str]) -> str:
    return "|".join(unique_preserve(values))


def aggregate_numeric(raw_values: list[str | None]) -> tuple[str, str]:
    present = [value for value in raw_values if value not in (None, "")]
    if not present:
        return "", "missing"
    if all(value == present[0] for value in present):
        return present[0], "identical"
    numbers = [float(value) for value in present]
    mean = sum(numbers) / len(numbers)
    return f"{mean:.6f}", "mean"


def json_ready(value: object) -> object:
    if isinstance(value, Path):
        return repo_rel(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, object]) -> None:
    text = json.dumps(json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str] | tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def load_subgraph_edge_ids(path: Path) -> SubgraphSpec:
    """Load V_R from the subgraph listing; keep first-seen order."""
    if not path.is_file():
        raise GraphError(f"subgraph edge file not found: {path}")
    stat = path.stat()
    text = path.read_text(encoding="utf-8-sig")
    raw_lines = text.splitlines()
    unique_ids: list[str] = []
    seen: set[str] = set()
    duplicate_ids: list[str] = []
    duplicate_records: list[dict[str, object]] = []
    nonempty_line_count = 0
    empty_line_count = 0
    first_index: dict[str, int] = {}
    for line_number, raw in enumerate(raw_lines, start=1):
        edge_id = raw.strip()
        if not edge_id:
            empty_line_count += 1
            continue
        nonempty_line_count += 1
        if edge_id in seen:
            duplicate_ids.append(edge_id)
            duplicate_records.append(
                {
                    "edge_id": edge_id,
                    "duplicate_line_number": line_number,
                    "first_node_index": first_index[edge_id],
                }
            )
            continue
        first_index[edge_id] = len(unique_ids)
        seen.add(edge_id)
        unique_ids.append(edge_id)
    if not unique_ids:
        raise GraphError(f"no edge IDs in {path}")
    return SubgraphSpec(
        path=path,
        raw_line_count=len(raw_lines),
        nonempty_line_count=nonempty_line_count,
        unique_ids=unique_ids,
        unique_set=seen,
        duplicate_ids=duplicate_ids,
        duplicate_records=duplicate_records,
        empty_line_count=empty_line_count,
        sha256=sha256_bytes(text.encode("utf-8")),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def parse_lane_element(elem: ET.Element) -> LaneRecord:
    return LaneRecord(
        lane_id=elem.get("id") or "",
        index=elem.get("index"),
        length_raw=elem.get("length"),
        speed_raw=elem.get("speed"),
        shape=parse_shape(elem.get("shape")),
    )


def parse_edge_element(elem: ET.Element) -> EdgeRecord:
    lanes = [
        parse_lane_element(child)
        for child in elem
        if strip_namespace(child.tag) == "lane"
    ]
    return EdgeRecord(
        edge_id=elem.get("id") or "",
        from_junction=elem.get("from"),
        to_junction=elem.get("to"),
        priority=elem.get("priority"),
        edge_type=elem.get("type"),
        function=elem.get("function"),
        lanes=lanes,
    )


def parse_sumo_net(
    net_file: Path,
    subgraph_ids: set[str],
) -> tuple[dict[str, EdgeRecord], list[ConnectionRecord], set[str], int]:
    """Stream-parse net_tls. Keep subgraph edges and every connection record."""
    if not net_file.is_file():
        raise GraphError(f"SUMO net file not found: {net_file}")

    edges: dict[str, EdgeRecord] = {}
    connections: list[ConnectionRecord] = []
    found_ids: set[str] = set()
    incomplete_connections = 0
    connection_order = 0

    try:
        context = ET.iterparse(net_file, events=("end",))
        for _event, elem in context:
            tag = strip_namespace(elem.tag)
            if tag == "edge":
                edge_id = elem.get("id")
                if edge_id in subgraph_ids and edge_id not in edges:
                    edges[edge_id] = parse_edge_element(elem)
                    found_ids.add(edge_id)
                elem.clear()
            elif tag == "connection":
                from_edge = elem.get("from")
                to_edge = elem.get("to")
                if not from_edge or not to_edge:
                    incomplete_connections += 1
                else:
                    connections.append(
                        ConnectionRecord(
                            attrib={key: value for key, value in elem.attrib.items()},
                            order=connection_order,
                        )
                    )
                    connection_order += 1
                elem.clear()
            elif tag == "lane":
                # Keep lane elements until the parent <edge> ends so length,
                # speed, and shape remain available to parse_edge_element.
                continue
            else:
                elem.clear()
    except ET.ParseError as exc:
        raise GraphError(f"failed to parse SUMO net XML: {net_file}") from exc

    return edges, connections, found_ids, incomplete_connections


def node_position(edge: EdgeRecord | None) -> tuple[float, float] | None:
    if edge is None:
        return None
    for lane in edge.lanes:
        midpoint = polyline_midpoint(lane.shape)
        if midpoint is not None:
            return midpoint
    return None


def build_r_graph(
    subgraph: SubgraphSpec,
    net_edges: dict[str, EdgeRecord],
    all_connections: list[ConnectionRecord],
    incomplete_connections: int,
) -> RGraph:
    """Assemble R nodes, unique directed R edges, and the binary adjacency."""
    missing_from_net = [edge_id for edge_id in subgraph.unique_ids if edge_id not in net_edges]
    extra_net_only_nodes: list[str] = []
    nodes: list[RNode] = []
    length_rules: dict[str, int] = defaultdict(int)
    speed_rules: dict[str, int] = defaultdict(int)
    mean_length_ids: list[str] = []
    mean_speed_ids: list[str] = []
    missing_length_ids: list[str] = []
    missing_speed_ids: list[str] = []

    index_by_edge = {edge_id: i for i, edge_id in enumerate(subgraph.unique_ids)}

    for node_index, edge_id in enumerate(subgraph.unique_ids):
        record = net_edges.get(edge_id)
        length_value, length_rule = aggregate_numeric(
            [lane.length_raw for lane in record.lanes] if record else []
        )
        speed_value, speed_rule = aggregate_numeric(
            [lane.speed_raw for lane in record.lanes] if record else []
        )
        length_rules[length_rule] += 1
        speed_rules[speed_rule] += 1
        if length_rule == "mean":
            mean_length_ids.append(edge_id)
        if speed_rule == "mean":
            mean_speed_ids.append(edge_id)
        if length_rule == "missing":
            missing_length_ids.append(edge_id)
        if speed_rule == "missing":
            missing_speed_ids.append(edge_id)
        nodes.append(
            RNode(
                node_index=node_index,
                node_id=node_id_for(edge_id),
                node_type=NODE_TYPE,
                edge_id=edge_id,
                from_junction="" if record is None or record.from_junction is None else record.from_junction,
                to_junction="" if record is None or record.to_junction is None else record.to_junction,
                num_lanes=0 if record is None else len(record.lanes),
                length=length_value,
                speed=speed_value,
                priority="" if record is None or record.priority is None else record.priority,
                edge_type="" if record is None or record.edge_type is None else record.edge_type,
                length_rule=length_rule,
                speed_rule=speed_rule,
                found_in_net=record is not None,
                position=node_position(record),
            )
        )

    filtered: list[ConnectionRecord] = []
    attr_keys: set[str] = set()
    grouped: dict[tuple[str, str], list[ConnectionRecord]] = {}
    for connection in all_connections:
        from_edge = connection.from_edge
        to_edge = connection.to_edge
        if from_edge in subgraph.unique_set and to_edge in subgraph.unique_set:
            filtered.append(connection)
            attr_keys.update(connection.attrib.keys())
            grouped.setdefault((from_edge, to_edge), []).append(connection)

    connection_columns = [name for name in PREFERRED_CONNECTION_COLUMNS if name in attr_keys]
    connection_columns.extend(sorted(name for name in attr_keys if name not in PREFERRED_CONNECTION_COLUMNS))

    r_edges: list[REdge] = []
    for (from_edge, to_edge), group in grouped.items():
        source_index = index_by_edge[from_edge]
        target_index = index_by_edge[to_edge]
        r_edges.append(
            REdge(
                source_index=source_index,
                target_index=target_index,
                source_node_id=node_id_for(from_edge),
                target_node_id=node_id_for(to_edge),
                source_edge_id=from_edge,
                target_edge_id=to_edge,
                raw_connection_count=len(group),
                tls_ids=join_unique([item.attrib.get("tl", "") for item in group]),
                link_indices=join_unique([item.attrib.get("linkIndex", "") for item in group]),
                directions=join_unique([item.attrib.get("dir", "") for item in group]),
                connections=group,
            )
        )
    r_edges.sort(key=lambda item: (item.source_index, item.target_index, item.source_edge_id, item.target_edge_id))

    n_nodes = len(nodes)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.uint8)
    for edge in r_edges:
        adjacency[edge.source_index, edge.target_index] = 1

    rows = [edge.source_index for edge in r_edges]
    cols = [edge.target_index for edge in r_edges]
    data = np.ones(len(r_edges), dtype=np.uint8)
    adjacency_sparse = sparse.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.uint8)
    edge_index = np.array([rows, cols], dtype=np.int64) if r_edges else np.zeros((2, 0), dtype=np.int64)

    length_speed_stats = {
        "rule": (
            "If every lane on an edge has the same XML value, that common value "
            "is kept. If lane values differ, the arithmetic mean of present "
            "lane values is written with 6 decimal places. Missing lane fields "
            "stay empty."
        ),
        "length_rule_counts": dict(length_rules),
        "speed_rule_counts": dict(speed_rules),
        "mean_length_edge_ids": mean_length_ids,
        "mean_speed_edge_ids": mean_speed_ids,
        "missing_length_edge_ids": missing_length_ids,
        "missing_speed_edge_ids": missing_speed_ids,
    }

    return RGraph(
        nodes=nodes,
        edges=r_edges,
        connections=filtered,
        connection_columns=connection_columns,
        adjacency=adjacency,
        adjacency_sparse=adjacency_sparse,
        edge_index=edge_index,
        missing_from_net=missing_from_net,
        extra_net_only_nodes=extra_net_only_nodes,
        incomplete_connections=incomplete_connections,
        length_speed_stats=length_speed_stats,
    )


def adjacency_from_pairs(n_nodes: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    matrix = np.zeros((n_nodes, n_nodes), dtype=np.uint8)
    for source, target in pairs:
        matrix[source, target] = 1
    return matrix


def summarize_components(labels: np.ndarray, nodes: list[RNode]) -> list[dict[str, object]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        groups[int(label)].append(index)
    ordered = sorted(groups.values(), key=lambda members: (-len(members), members[0]))
    summaries: list[dict[str, object]] = []
    for members in ordered:
        summaries.append(
            {
                "size": len(members),
                "node_index_examples": members[:EXAMPLE_LIMIT],
                "edge_id_examples": [nodes[index].edge_id for index in members[:EXAMPLE_LIMIT]],
            }
        )
    return summaries


def degree_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "mean": 0.0, "median": 0.0, "max": 0}
    return {
        "min": min(values),
        "mean": float(sum(values) / len(values)),
        "median": float(statistics.median(values)),
        "max": max(values),
    }


def generate_preview(
    graph: RGraph,
    output_dir: Path,
    labeled_name: str = "r_graph_preview.png",
    unlabeled_name: str = "r_graph_preview_unlabeled.png",
    dpi: int = 150,
) -> dict[str, object]:
    """Draw a geographic directed preview. Topology is never taken from the figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    positions = {node.node_index: node.position for node in graph.nodes if node.position is not None}
    missing = [node.edge_id for node in graph.nodes if node.position is None]
    result: dict[str, object] = {
        "generated": False,
        "labeled_path": repo_rel(output_dir / labeled_name),
        "unlabeled_path": repo_rel(output_dir / unlabeled_name),
        "nodes_with_geometry": len(positions),
        "nodes_missing_geometry": missing,
        "error": "",
    }
    if len(positions) < 2:
        result["error"] = "not enough edge/lane geometry to draw a preview"
        return result

    xs = [point[0] for point in positions.values()]
    ys = [point[1] for point in positions.values()]
    dx = max(xs) - min(xs) or 1.0
    dy = max(ys) - min(ys) or 1.0
    width = max(8.0, min(16.0, 10.0 * (dx / dy)))
    height = max(8.0, min(16.0, 10.0 * (dy / dx)))
    offset = 0.004 * max(dx, dy)
    pair_set = {(edge.source_index, edge.target_index) for edge in graph.edges}

    def draw(ax, with_labels: bool) -> None:
        ax.set_aspect("equal")
        for edge in graph.edges:
            start = positions.get(edge.source_index)
            end = positions.get(edge.target_index)
            if start is None or end is None:
                continue
            x1, y1 = start
            x2, y2 = end
            reverse = (edge.target_index, edge.source_index) in pair_set
            if reverse and edge.source_index != edge.target_index:
                vx, vy = x2 - x1, y2 - y1
                length = math.hypot(vx, vy) or 1.0
                x1, y1 = x1 - vy / length * offset, y1 + vx / length * offset
                x2, y2 = x2 - vy / length * offset, y2 + vx / length * offset
            ax.add_patch(
                FancyArrowPatch(
                    (x1, y1),
                    (x2, y2),
                    arrowstyle="-|>",
                    mutation_scale=10,
                    linewidth=0.8,
                    color="#4a4a4a",
                    shrinkA=6,
                    shrinkB=6,
                    connectionstyle="arc3,rad=0.0",
                )
            )
        px = [positions[node.node_index][0] for node in graph.nodes if node.node_index in positions]
        py = [positions[node.node_index][1] for node in graph.nodes if node.node_index in positions]
        ax.scatter(px, py, s=36, c="#1f77b4", zorder=3, edgecolors="white", linewidths=0.4)
        if with_labels:
            for node in graph.nodes:
                point = positions.get(node.node_index)
                if point is None:
                    continue
                ax.annotate(
                    str(node.node_index),
                    point,
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    color="#111111",
                )
        pad_x, pad_y = 0.05 * dx, 0.05 * dy
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
        ax.set_xlabel("SUMO x (m)")
        ax.set_ylabel("SUMO y (m)")
        ax.grid(True, linewidth=0.3, alpha=0.4)

    labeled_path = output_dir / labeled_name
    unlabeled_path = output_dir / unlabeled_name
    fig, ax = plt.subplots(figsize=(width, height))
    draw(ax, with_labels=True)
    ax.set_title(
        f"R-only directed graph  N={len(graph.nodes)}  E={len(graph.edges)}\n"
        "labels are node_index; see r_nodes.csv for edge_id"
    )
    fig.tight_layout()
    fig.savefig(labeled_path, dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(width, height))
    draw(ax, with_labels=False)
    ax.set_title(f"R-only directed graph (unlabeled)  N={len(graph.nodes)}  E={len(graph.edges)}")
    fig.tight_layout()
    fig.savefig(unlabeled_path, dpi=dpi)
    plt.close(fig)

    result["generated"] = True
    return result


def node_rows(graph: RGraph) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for node in graph.nodes:
        rows.append(
            {
                "node_index": node.node_index,
                "node_id": node.node_id,
                "node_type": node.node_type,
                "edge_id": node.edge_id,
                "from_junction": node.from_junction,
                "to_junction": node.to_junction,
                "num_lanes": node.num_lanes,
                "length": node.length,
                "speed": node.speed,
                "priority": node.priority,
                "edge_type": node.edge_type,
            }
        )
    return rows


def edge_rows(graph: RGraph) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for edge in graph.edges:
        rows.append(
            {
                "source_index": edge.source_index,
                "target_index": edge.target_index,
                "source_node_id": edge.source_node_id,
                "target_node_id": edge.target_node_id,
                "source_edge_id": edge.source_edge_id,
                "target_edge_id": edge.target_edge_id,
                "raw_connection_count": edge.raw_connection_count,
                "tls_ids": edge.tls_ids,
                "link_indices": edge.link_indices,
                "directions": edge.directions,
            }
        )
    return rows


def connection_rows(graph: RGraph) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for connection in graph.connections:
        row = {name: "" for name in graph.connection_columns}
        row.update(connection.attrib)
        rows.append(row)
    return rows


def write_graph_artifacts(
    graph: RGraph,
    output_dir: Path,
    preview: bool,
) -> dict[str, dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, dict[str, object]] = {}

    mapping = {
        "r_nodes.csv": (R_NODE_FIELDNAMES, node_rows(graph)),
        "r_edges.csv": (R_EDGE_FIELDNAMES, edge_rows(graph)),
        "r_sumo_connections.csv": (graph.connection_columns, connection_rows(graph)),
    }
    for name, (fieldnames, rows) in mapping.items():
        path = output_dir / name
        write_csv(path, fieldnames, rows)
        written[name] = file_fingerprint(path)

    adj_path = output_dir / "r_adjacency.npy"
    np.save(adj_path, graph.adjacency, allow_pickle=False)
    written["r_adjacency.npy"] = file_fingerprint(adj_path)

    sparse_path = output_dir / "r_adjacency_sparse.npz"
    sparse.save_npz(sparse_path, graph.adjacency_sparse)
    written["r_adjacency_sparse.npz"] = file_fingerprint(sparse_path)

    index_path = output_dir / "r_edge_index.npy"
    np.save(index_path, graph.edge_index, allow_pickle=False)
    written["r_edge_index.npy"] = file_fingerprint(index_path)

    preview_info: dict[str, object]
    if preview:
        preview_info = generate_preview(graph, output_dir)
        for name in ("r_graph_preview.png", "r_graph_preview_unlabeled.png"):
            path = output_dir / name
            if path.is_file():
                written[name] = file_fingerprint(path)
    else:
        preview_info = {"generated": False, "error": "preview disabled"}
    written["_preview"] = preview_info
    return written


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_r_graph(
    subgraph: SubgraphSpec,
    graph: RGraph,
    output_dir: Path,
    written: dict[str, dict[str, object]],
    random_seed: int,
    input_before: dict[str, dict[str, object]],
    input_after: dict[str, dict[str, object]],
    preview_enabled: bool,
) -> dict[str, object]:
    n_nodes = len(graph.nodes)
    n_edges = len(graph.edges)
    node_ids = [node.edge_id for node in graph.nodes]
    node_id_set = set(node_ids)
    failures: list[str] = []
    warnings: list[str] = []

    if node_ids != subgraph.unique_ids:
        failures.append("R node order/set does not match subgraph first-seen unique IDs")
    if len(node_ids) != len(subgraph.unique_set):
        failures.append("R node count does not equal subgraph unique ID count")
    extra_nodes = sorted(node_id_set - subgraph.unique_set)
    missing_nodes = [edge_id for edge_id in subgraph.unique_ids if edge_id not in node_id_set]
    if extra_nodes:
        failures.append("R graph contains extra nodes not in subgraph")
    if missing_nodes:
        failures.append("R graph is missing subgraph nodes")
    if graph.missing_from_net:
        failures.append("one or more subgraph edges were not found in net_tls")
    if any(node.node_index != i for i, node in enumerate(graph.nodes)):
        failures.append("node_index is not contiguous from 0")
    if any(node.node_type != NODE_TYPE or node.node_id != node_id_for(node.edge_id) for node in graph.nodes):
        failures.append("node_id or node_type mapping is inconsistent")

    pair_from_connections = {(item.from_edge, item.to_edge) for item in graph.connections}
    pair_from_edges = {(item.source_edge_id, item.target_edge_id) for item in graph.edges}
    if pair_from_connections != pair_from_edges:
        failures.append("unique R edges do not match filtered SUMO connection pairs")
    if len(pair_from_edges) != n_edges:
        failures.append("duplicate road-level R edges exist")
    if sum(edge.raw_connection_count for edge in graph.edges) != len(graph.connections):
        failures.append("raw_connection_count does not sum to filtered lane-level connections")

    rebuilt_from_edges = adjacency_from_pairs(
        n_nodes,
        [(edge.source_index, edge.target_index) for edge in graph.edges],
    )
    rebuilt_from_index = adjacency_from_pairs(
        n_nodes,
        list(zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist())) if graph.edge_index.size else [],
    )
    index_by_edge = {node.edge_id: node.node_index for node in graph.nodes}
    rebuilt_from_connections = adjacency_from_pairs(
        n_nodes,
        [(index_by_edge[item.from_edge], index_by_edge[item.to_edge]) for item in graph.connections],
    )

    unique_values = {int(value) for value in np.unique(graph.adjacency)}
    if unique_values - {0, 1}:
        failures.append("adjacency contains values other than 0 and 1")
    if graph.adjacency.shape != (n_nodes, n_nodes):
        failures.append("adjacency shape is not [N_R, N_R]")
    if int(graph.adjacency.sum()) != n_edges:
        failures.append("adjacency nonzero count does not equal unique R-edge count")
    if not np.array_equal(graph.adjacency, rebuilt_from_edges):
        failures.append("adjacency does not match r_edges")
    if not np.array_equal(graph.adjacency, rebuilt_from_index):
        failures.append("adjacency does not match r_edge_index")
    if not np.array_equal(graph.adjacency, rebuilt_from_connections):
        failures.append("adjacency does not match filtered SUMO connections")
    if not np.array_equal(graph.adjacency, graph.adjacency_sparse.toarray()):
        failures.append("dense and sparse adjacency differ")
    if graph.adjacency.dtype != np.uint8:
        warnings.append(f"adjacency dtype is {graph.adjacency.dtype}, expected uint8")

    identity = np.eye(n_nodes, dtype=np.uint8)
    added_identity = bool(
        n_nodes
        and np.array_equal(graph.adjacency, np.bitwise_or(rebuilt_from_connections, identity))
        and not np.array_equal(graph.adjacency, rebuilt_from_connections)
    )
    if added_identity:
        failures.append("adjacency looks like A+I rather than the raw topology")

    self_loop_pairs = [edge for edge in graph.edges if edge.source_edge_id == edge.target_edge_id]
    diag = np.diag(graph.adjacency)
    artificial_self_loops = int(diag.sum()) != len(self_loop_pairs)
    if artificial_self_loops:
        failures.append("diagonal ones do not match net_tls self-loop connections")

    reverse_without_connection = 0
    for i in range(n_nodes):
        for j in range(n_nodes):
            if graph.adjacency[i, j] and not graph.adjacency[j, i]:
                continue
            if graph.adjacency[i, j] and graph.adjacency[j, i] and i != j:
                pair = (graph.nodes[j].edge_id, graph.nodes[i].edge_id)
                if pair not in pair_from_connections:
                    reverse_without_connection += 1
    if reverse_without_connection:
        failures.append("found reverse edges that are not present in net_tls")

    # File reloads.
    nodes_csv = load_csv_rows(output_dir / "r_nodes.csv")
    edges_csv = load_csv_rows(output_dir / "r_edges.csv")
    connections_csv = load_csv_rows(output_dir / "r_sumo_connections.csv")
    adj_loaded = np.load(output_dir / "r_adjacency.npy", allow_pickle=False)
    sparse_loaded = sparse.load_npz(output_dir / "r_adjacency_sparse.npz")
    index_loaded = np.load(output_dir / "r_edge_index.npy", allow_pickle=False)

    csv_ids = [row["edge_id"] for row in nodes_csv]
    if csv_ids != subgraph.unique_ids:
        failures.append("r_nodes.csv order does not match subgraph")
    if [int(row["node_index"]) for row in nodes_csv] != list(range(n_nodes)):
        failures.append("r_nodes.csv node_index is not 0..N_R-1")
    if any(row["node_type"] != NODE_TYPE for row in nodes_csv):
        failures.append("r_nodes.csv node_type is not R")
    csv_pairs = [(row["source_edge_id"], row["target_edge_id"]) for row in edges_csv]
    if csv_pairs != [(edge.source_edge_id, edge.target_edge_id) for edge in graph.edges]:
        failures.append("r_edges.csv order or membership does not match in-memory R edges")
    if len(connections_csv) != len(graph.connections):
        failures.append("r_sumo_connections.csv row count does not match filtered connections")
    if not np.array_equal(adj_loaded, graph.adjacency):
        failures.append("reloaded r_adjacency.npy differs from in-memory matrix")
    if not np.array_equal(sparse_loaded.toarray(), graph.adjacency):
        failures.append("reloaded sparse adjacency differs from dense matrix")
    if not np.array_equal(index_loaded, graph.edge_index):
        failures.append("reloaded r_edge_index.npy differs from in-memory index")

    csv_adj = adjacency_from_pairs(
        n_nodes,
        [(int(row["source_index"]), int(row["target_index"])) for row in edges_csv],
    )
    if not np.array_equal(csv_adj, graph.adjacency):
        failures.append("r_edges.csv does not reconstruct the adjacency matrix")

    if graph.connection_columns:
        missing_cols = [name for name in graph.connection_columns if name not in (connections_csv[0].keys() if connections_csv else graph.connection_columns)]
        if connections_csv and missing_cols:
            failures.append("r_sumo_connections.csv is missing connection attributes")

    sample_n = min(SAMPLE_EDGE_COUNT, n_edges)
    rng = random.Random(random_seed)
    sampled_edges = rng.sample(graph.edges, sample_n) if sample_n else []
    sample_records: list[dict[str, object]] = []
    sample_passed = True
    for edge in sampled_edges:
        evidence = [item.attrib for item in edge.connections]
        ok = (
            len(evidence) == edge.raw_connection_count
            and len(evidence) >= 1
            and all(item.get("from") == edge.source_edge_id and item.get("to") == edge.target_edge_id for item in evidence)
            and graph.adjacency[edge.source_index, edge.target_index] == 1
        )
        if not ok:
            sample_passed = False
            failures.append(
                f"sampled R edge {edge.source_edge_id} -> {edge.target_edge_id} failed connection evidence"
            )
        sample_records.append(
            {
                "source_index": edge.source_index,
                "target_index": edge.target_index,
                "source_edge_id": edge.source_edge_id,
                "target_edge_id": edge.target_edge_id,
                "raw_connection_count": edge.raw_connection_count,
                "passed": ok,
                "connections": evidence,
            }
        )

    if n_nodes == 0:
        n_weak = 0
        n_strong = 0
        weak_labels = np.zeros((0,), dtype=int)
        strong_labels = np.zeros((0,), dtype=int)
    else:
        n_weak, weak_labels = connected_components(graph.adjacency_sparse, directed=True, connection="weak")
        n_strong, strong_labels = connected_components(graph.adjacency_sparse, directed=True, connection="strong")

    out_degree = graph.adjacency.sum(axis=1).astype(int).tolist()
    in_degree = graph.adjacency.sum(axis=0).astype(int).tolist()
    isolated = [
        {"node_index": i, "edge_id": graph.nodes[i].edge_id}
        for i, (indeg, outdeg) in enumerate(zip(in_degree, out_degree))
        if indeg == 0 and outdeg == 0
    ]
    zero_in = [
        {"node_index": i, "edge_id": graph.nodes[i].edge_id}
        for i, indeg in enumerate(in_degree)
        if indeg == 0
    ]
    zero_out = [
        {"node_index": i, "edge_id": graph.nodes[i].edge_id}
        for i, outdeg in enumerate(out_degree)
        if outdeg == 0
    ]

    weakly_connected = n_weak == 1
    if not weakly_connected:
        failures.append(
            f"weakly connected component count is {n_weak}, expected 1; no edges were added or removed"
        )

    possible = n_nodes * (n_nodes - 1) if n_nodes > 1 else 0
    density = (n_edges - len(self_loop_pairs)) / possible if possible else 0.0
    symmetric = bool(np.array_equal(graph.adjacency, graph.adjacency.T))
    normalized_like = bool(unique_values - {0, 1}) or np.issubdtype(graph.adjacency.dtype, np.floating)

    preview_info = written.get("_preview", {})
    if preview_enabled and not preview_info.get("generated"):
        failures.append(f"preview was not generated: {preview_info.get('error', 'unknown')}")

    inputs_unchanged = input_before == input_after
    if not inputs_unchanged:
        failures.append("subgraph or net_tls fingerprint changed during the run")

    rebuild_ok = (
        np.array_equal(graph.adjacency, rebuilt_from_edges)
        and np.array_equal(graph.adjacency, rebuilt_from_index)
        and np.array_equal(graph.adjacency, rebuilt_from_connections)
        and np.array_equal(graph.adjacency, adj_loaded)
    )
    if not rebuild_ok:
        failures.append("independent adjacency rebuilds are not identical")

    construction_keys = {
        "node_set_matches_subgraph",
        "all_nodes_found_in_net",
        "adjacency_binary",
        "formats_consistent",
        "no_artificial_reverse_edges",
        "no_artificial_self_loops",
        "sample_connections_passed",
        "inputs_unchanged",
    }

    node_set_ok = (
        node_ids == subgraph.unique_ids
        and not extra_nodes
        and not missing_nodes
        and not graph.extra_net_only_nodes
    )
    all_found = not graph.missing_from_net
    formats_ok = (
        np.array_equal(graph.adjacency, rebuilt_from_edges)
        and np.array_equal(graph.adjacency, rebuilt_from_index)
        and np.array_equal(graph.adjacency, rebuilt_from_connections)
        and np.array_equal(graph.adjacency, graph.adjacency_sparse.toarray())
        and np.array_equal(adj_loaded, graph.adjacency)
        and np.array_equal(sparse_loaded.toarray(), graph.adjacency)
        and np.array_equal(index_loaded, graph.edge_index)
        and csv_ids == subgraph.unique_ids
        and csv_pairs == [(edge.source_edge_id, edge.target_edge_id) for edge in graph.edges]
        and len(connections_csv) == len(graph.connections)
    )
    binary_ok = unique_values <= {0, 1} and graph.adjacency.shape == (n_nodes, n_nodes) and int(graph.adjacency.sum()) == n_edges
    no_art_reverse = reverse_without_connection == 0
    no_art_self = not artificial_self_loops

    checks = {
        "node_set_matches_subgraph": node_set_ok,
        "all_nodes_found_in_net": all_found,
        "adjacency_binary": binary_ok,
        "formats_consistent": formats_ok,
        "no_artificial_reverse_edges": no_art_reverse,
        "no_artificial_self_loops": no_art_self,
        "sample_connections_passed": sample_passed,
        "inputs_unchanged": inputs_unchanged,
        "weakly_connected": weakly_connected,
        "preview_generated": bool(preview_info.get("generated")) if preview_enabled else True,
    }
    construction_passed = all(checks[name] for name in construction_keys)
    overall_passed = construction_passed and weakly_connected and checks["preview_generated"]

    validation = {
        "status": "ok" if overall_passed else "failed",
        "overall_passed": overall_passed,
        "construction_passed": construction_passed,
        "failures": failures,
        "warnings": warnings,
        "random_seed": random_seed,
        "checks": checks,
        "nodes": {
            "subgraph_raw_line_count": subgraph.raw_line_count,
            "subgraph_nonempty_line_count": subgraph.nonempty_line_count,
            "subgraph_empty_line_count": subgraph.empty_line_count,
            "unique_edge_id_count": len(subgraph.unique_ids),
            "duplicate_edge_id_count": len(subgraph.duplicate_ids),
            "duplicate_examples": subgraph.duplicate_records[:EXAMPLE_LIMIT],
            "r_node_count": n_nodes,
            "r_node_count_equals_unique_ids": n_nodes == len(subgraph.unique_ids),
            "node_set_matches_subgraph": node_set_ok,
            "extra_nodes": extra_nodes[:EXAMPLE_LIMIT],
            "missing_nodes": missing_nodes[:EXAMPLE_LIMIT],
            "missing_from_net_count": len(graph.missing_from_net),
            "missing_from_net_examples": graph.missing_from_net[:EXAMPLE_LIMIT],
            "node_index_contiguous_from_zero": [node.node_index for node in graph.nodes] == list(range(n_nodes)),
            "internal_colon_prefix_nodes": [edge_id for edge_id in node_ids if edge_id.startswith(":")],
        },
        "edges": {
            "raw_sumo_connection_count": len(graph.connections),
            "unique_r_graph_edge_count": n_edges,
            "incomplete_connection_elements": graph.incomplete_connections,
            "self_loop_count": len(self_loop_pairs),
            "self_loop_examples": [
                {"source_edge_id": edge.source_edge_id, "target_edge_id": edge.target_edge_id}
                for edge in self_loop_pairs[:EXAMPLE_LIMIT]
            ],
            "multi_lane_aggregated_edge_count": sum(1 for edge in graph.edges if edge.raw_connection_count > 1),
            "sampled_r_edges": sample_records,
            "sample_passed": sample_passed,
            "no_artificial_reverse_edges": no_art_reverse,
            "no_artificial_self_loops": no_art_self,
        },
        "adjacency": {
            "shape": [n_nodes, n_nodes],
            "dtype": str(graph.adjacency.dtype),
            "unique_values": sorted(unique_values),
            "nonzero_count": int(graph.adjacency.sum()),
            "direction_convention": ADJACENCY_CONVENTION,
            "dense_sparse_edge_index_csv_consistent": formats_ok,
            "symmetrized": False,
            "adjacency_happens_to_be_symmetric": symmetric,
            "normalized": False,
            "normalized_like_values_detected": normalized_like,
            "identity_added": False,
            "identity_like_detected": added_identity,
        },
        "connectivity": {
            "weakly_connected_component_count": int(n_weak),
            "weakly_connected": weakly_connected,
            "weak_component_sizes": summarize_components(weak_labels, graph.nodes),
            "strongly_connected_component_count": int(n_strong),
            "largest_strongly_connected_component_size": (
                max((item["size"] for item in summarize_components(strong_labels, graph.nodes)), default=0)
            ),
            "strong_component_sizes": summarize_components(strong_labels, graph.nodes),
            "isolated_node_count": len(isolated),
            "isolated_node_examples": isolated[:EXAMPLE_LIMIT],
            "zero_in_degree_count": len(zero_in),
            "zero_in_degree_examples": zero_in[:EXAMPLE_LIMIT],
            "zero_out_degree_count": len(zero_out),
            "zero_out_degree_examples": zero_out[:EXAMPLE_LIMIT],
            "in_degree": degree_stats(in_degree),
            "out_degree": degree_stats(out_degree),
            "graph_density_without_self_loops": density,
            "possible_directed_edges_without_self_loops": possible,
        },
        "reproducibility": {
            "node_order_stable": csv_ids == subgraph.unique_ids,
            "edge_order_stable": csv_pairs == [(edge.source_edge_id, edge.target_edge_id) for edge in graph.edges],
            "adjacency_rebuilds_identical": rebuild_ok,
            "edge_index_matches_adjacency": np.array_equal(graph.adjacency, rebuilt_from_index),
            "output_sha256": {
                name: info.get("sha256")
                for name, info in written.items()
                if name != "_preview" and isinstance(info, dict) and "sha256" in info
            },
        },
        "preview": preview_info,
        "inputs": {
            "before": input_before,
            "after": input_after,
            "unchanged": inputs_unchanged,
            "trajectory_or_sampling_files_opened": False,
        },
        "length_speed": graph.length_speed_stats,
    }
    return validation


def build_metadata(
    subgraph: SubgraphSpec,
    net_file: Path,
    net_fingerprint: dict[str, object],
    graph: RGraph,
    output_dir: Path,
    written: dict[str, dict[str, object]],
    generated_at: str,
) -> dict[str, object]:
    output_files = {
        name: {key: value for key, value in info.items() if key in {"path", "sha256", "size_bytes"}}
        for name, info in written.items()
        if name != "_preview" and isinstance(info, dict)
    }
    return {
        "graph_name": "r_graph",
        "graph_type": "R-only",
        "directed": True,
        "node_type": NODE_TYPE,
        "node_definition": "Each unique nonempty SUMO edge ID in the subgraph file is one R node.",
        "edge_definition": (
            "A directed R edge R_from -> R_to exists iff net_tls contains at least one "
            "<connection from=from to=to> whose both endpoints are subgraph edge IDs. "
            "Multiple lane-level connections collapse to one binary R edge."
        ),
        "adjacency_direction_convention": ADJACENCY_CONVENTION,
        "artificial_self_loops_added": False,
        "symmetrized": False,
        "normalized": False,
        "weighted": False,
        "subgraph_file": repo_rel(subgraph.path),
        "net_tls_file": repo_rel(net_file),
        "subgraph_sha256": subgraph.sha256,
        "net_tls_sha256": net_fingerprint["sha256"],
        "input_fingerprints": {
            "subgraph": {
                "path": repo_rel(subgraph.path),
                "sha256": subgraph.sha256,
                "size_bytes": subgraph.size_bytes,
            },
            "net_tls": net_fingerprint,
        },
        "node_order_rule": (
            "First occurrence in the subgraph file; leading/trailing whitespace stripped; "
            "blank lines ignored; later duplicate IDs dropped and reported; not sorted lexicographically."
        ),
        "node_count": len(graph.nodes),
        "unique_directed_edge_count": len(graph.edges),
        "raw_lane_level_connection_count": len(graph.connections),
        "self_loop_count": sum(1 for edge in graph.edges if edge.source_index == edge.target_index),
        "missing_from_net_count": len(graph.missing_from_net),
        "connection_attribute_columns": graph.connection_columns,
        "length_speed_aggregation": graph.length_speed_stats,
        "multi_value_join_separator": "|",
        "output_directory": repo_rel(output_dir),
        "output_files": output_files,
        "script": repo_rel(Path(__file__)),
        "script_version": SCRIPT_VERSION,
        "generated_at": generated_at,
        "python_version": sys.version,
        "dependency_versions": {
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "matplotlib": __import__("matplotlib").__version__,
        },
        "random_seed_for_validation_sample": RANDOM_SEED_DEFAULT,
        "does_not_read": [
            "trajectories.csv",
            "vehicle_ids.txt",
            "sampled_vehicle_ids",
            "edge_flow_*.csv",
        ],
    }


def existing_targets(output_dir: Path) -> list[str]:
    found: list[str] = []
    for name in OUTPUT_ARTIFACTS:
        if (output_dir / name).exists():
            found.append(name)
    return found


def run_synthetic_checks() -> None:
    net_xml = """<?xml version="1.0" encoding="UTF-8"?>
<net>
    <edge id="A" from="j1" to="j2" priority="1" type="highway.primary">
        <lane id="A_0" index="0" speed="10" length="100" shape="0.00,0.00 100.00,0.00"/>
        <lane id="A_1" index="1" speed="10" length="100" shape="0.00,1.00 100.00,1.00"/>
    </edge>
    <edge id="B" from="j2" to="j3" priority="1" type="highway.primary">
        <lane id="B_0" index="0" speed="11" length="50" shape="100.00,0.00 150.00,0.00"/>
    </edge>
    <edge id="C" from="j3" to="j4" priority="2" type="highway.residential">
        <lane id="C_0" index="0" speed="8" length="80" shape="150.00,0.00 150.00,80.00"/>
        <lane id="C_1" index="1" speed="9" length="81" shape="151.00,0.00 151.00,81.00"/>
    </edge>
    <edge id="OUT" from="j9" to="j10" priority="1" type="highway.primary">
        <lane id="OUT_0" index="0" speed="1" length="1" shape="0.00,10.00 1.00,10.00"/>
    </edge>
    <connection from="A" to="B" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from="A" to="B" fromLane="1" toLane="0" dir="s" state="M" tl="tl1" linkIndex="0"/>
    <connection from="B" to="C" fromLane="0" toLane="0" dir="r" state="o"/>
    <connection from="A" to="OUT" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from="C" to="C" fromLane="0" toLane="0" dir="t" state="m"/>
</net>
"""
    subgraph_text = "A\nB\nA\nC\n"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        subgraph_path = tmp_path / "subgraph.txt"
        net_path = tmp_path / "net.net.xml"
        output_dir = tmp_path / "out"
        subgraph_path.write_text(subgraph_text, encoding="utf-8")
        net_path.write_text(net_xml, encoding="utf-8")
        subgraph = load_subgraph_edge_ids(subgraph_path)
        if subgraph.unique_ids != ["A", "B", "C"]:
            raise GraphError(f"synthetic unique IDs wrong: {subgraph.unique_ids}")
        if subgraph.duplicate_ids != ["A"]:
            raise GraphError(f"synthetic duplicates wrong: {subgraph.duplicate_ids}")
        net_edges, connections, found, incomplete = parse_sumo_net(net_path, subgraph.unique_set)
        if incomplete != 0 or found != {"A", "B", "C"}:
            raise GraphError("synthetic net parse mismatch")
        graph = build_r_graph(subgraph, net_edges, connections, incomplete)
        if [node.edge_id for node in graph.nodes] != ["A", "B", "C"]:
            raise GraphError("synthetic node order mismatch")
        if graph.nodes[2].speed_rule != "mean" or graph.nodes[2].length_rule != "mean":
            raise GraphError("synthetic mean aggregation was not applied to C")
        pairs = [(edge.source_edge_id, edge.target_edge_id) for edge in graph.edges]
        if pairs != [("A", "B"), ("B", "C"), ("C", "C")]:
            raise GraphError(f"synthetic R edges wrong: {pairs}")
        ab = next(edge for edge in graph.edges if edge.source_edge_id == "A")
        if ab.raw_connection_count != 2:
            raise GraphError("synthetic multi-lane aggregation failed")
        if graph.adjacency.shape != (3, 3):
            raise GraphError("synthetic adjacency shape wrong")
        expected = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=np.uint8)
        if not np.array_equal(graph.adjacency, expected):
            raise GraphError(f"synthetic adjacency wrong:\n{graph.adjacency}")
        if any(item.from_edge == "OUT" or item.to_edge == "OUT" for item in graph.connections):
            raise GraphError("synthetic graph leaked an outside connection")
        write_graph_artifacts(graph, output_dir, preview=True)
        if not (output_dir / "r_adjacency.npy").is_file():
            raise GraphError("synthetic artifacts were not written")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a directed R-only road graph from a subgraph edge list and "
            "net_tls <connection> records. Does not read trajectories or build "
            "M/MR graphs."
        ),
        epilog=ADJACENCY_CONVENTION,
    )
    parser.add_argument(
        "--subgraph-edges",
        type=Path,
        default=DEFAULT_SUBGRAPH_FILE,
        help="Text file with one SUMO edge ID per line (default: simulation/data/subgraph.txt).",
    )
    parser.add_argument(
        "--net-file",
        type=Path,
        default=DEFAULT_NET_FILE,
        help="Canonical SUMO net.xml (default: road_network/net_tls.net.xml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: analysis/graph/r_graph).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing R-graph artifacts in the output directory.",
    )
    parser.add_argument(
        "--generate-preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write geographic PNG previews (default: true).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=RANDOM_SEED_DEFAULT,
        help="Seed for sampled connection evidence (default: 42).",
    )
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Skip the in-process synthetic fixture check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_synthetic:
        run_synthetic_checks()
        print("synthetic fixture passed", flush=True)

    subgraph_path = args.subgraph_edges.expanduser().resolve()
    net_path = args.net_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    already = existing_targets(output_dir)
    if already and not args.overwrite:
        raise GraphError(
            "output already exists: "
            + ", ".join(already)
            + f" in {output_dir}. Pass --overwrite to replace these files only."
        )

    input_before = {
        "subgraph": file_fingerprint(subgraph_path),
        "net_tls": file_fingerprint(net_path),
    }
    subgraph = load_subgraph_edge_ids(subgraph_path)
    print(
        f"subgraph: nonempty={subgraph.nonempty_line_count} "
        f"unique={len(subgraph.unique_ids)} duplicates={len(subgraph.duplicate_ids)}",
        flush=True,
    )

    net_edges, connections, found_ids, incomplete = parse_sumo_net(net_path, subgraph.unique_set)
    print(
        f"net_tls: matched_edges={len(found_ids)} "
        f"all_connections={len(connections)} incomplete={incomplete}",
        flush=True,
    )

    graph = build_r_graph(subgraph, net_edges, connections, incomplete)
    print(
        f"R graph: N={len(graph.nodes)} raw_connections={len(graph.connections)} "
        f"unique_edges={len(graph.edges)} missing_from_net={len(graph.missing_from_net)}",
        flush=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    written = write_graph_artifacts(graph, output_dir, preview=args.generate_preview)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    net_fingerprint = {
        "path": repo_rel(net_path),
        "sha256": input_before["net_tls"]["sha256"],
        "size_bytes": input_before["net_tls"]["size_bytes"],
    }
    metadata = build_metadata(
        subgraph,
        net_path,
        net_fingerprint,
        graph,
        output_dir,
        written,
        generated_at,
    )
    metadata_path = output_dir / "r_graph_metadata.json"
    write_json(metadata_path, metadata)
    written["r_graph_metadata.json"] = file_fingerprint(metadata_path)

    input_after = {
        "subgraph": file_fingerprint(subgraph_path),
        "net_tls": file_fingerprint(net_path),
    }
    validation = validate_r_graph(
        subgraph=subgraph,
        graph=graph,
        output_dir=output_dir,
        written=written,
        random_seed=args.random_seed,
        input_before=input_before,
        input_after=input_after,
        preview_enabled=args.generate_preview,
    )
    validation["reproducibility"]["output_sha256"]["r_graph_metadata.json"] = written["r_graph_metadata.json"]["sha256"]
    validation_path = output_dir / "r_graph_validation.json"
    write_json(validation_path, validation)
    written["r_graph_validation.json"] = file_fingerprint(validation_path)

    print(f"wrote {repo_rel(output_dir)} status={validation['status']}", flush=True)
    connectivity = validation["connectivity"]
    print(
        f"WCC={connectivity['weakly_connected_component_count']} "
        f"SCC={connectivity['strongly_connected_component_count']} "
        f"max_SCC={connectivity['largest_strongly_connected_component_size']} "
        f"isolates={connectivity['isolated_node_count']} "
        f"zero_in={connectivity['zero_in_degree_count']} "
        f"zero_out={connectivity['zero_out_degree_count']}",
        flush=True,
    )
    if validation["failures"]:
        for item in validation["failures"]:
            print(f"FAIL: {item}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except GraphError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
