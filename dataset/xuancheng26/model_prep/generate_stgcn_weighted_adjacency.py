#!/usr/bin/env python3
"""Build the Xuancheng26 STGCN symmetric distance-weighted adjacency.

The kernel matches the Shanghai R-edge adaptation, not PeMS ``/10000``
scaling. Distances are computed only for undirected topological neighbours.
Laplacian and Chebyshev stay in the trainer.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import prep_paths

NET_FILE = prep_paths.DEFAULT_NET_FILE
GRAPH_DIR = prep_paths.DEFAULT_TRAINING_ROOT / "graph"
R_GRAPH = GRAPH_DIR / "r_graph.npz"
NODE_MAPPING = GRAPH_DIR / "node_mapping.csv"
R_NODES = GRAPH_DIR / "r_nodes.csv"
OUTPUT = GRAPH_DIR / "stgcn_weighted_adjacency.npy"

EXPECTED_NODES = 1406
EXPECTED_DIRECTED_EDGES = 3877
LENGTH_MISMATCH_TOL_M = 1e-3
DISTANCE_ATOL = 1e-6
SKIP_EDGE_FUNCTIONS = frozenset({"walkingarea", "crossing"})
EPSILON = 0.0


class GraphError(RuntimeError):
    pass


@dataclass
class LaneRecord:
    lane_id: str
    index: str
    length: float
    shape: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class EdgeRecord:
    edge_id: str
    function: str
    lanes: list[LaneRecord] = field(default_factory=list)

    @property
    def is_internal(self) -> bool:
        return self.function == "internal" or self.edge_id.startswith(":")

    @property
    def skip(self) -> bool:
        return self.function in SKIP_EDGE_FUNCTIONS


@dataclass
class DirectedRoute:
    reachable: bool
    distance: float
    source_lane_id: str
    target_lane_id: str
    via_lane_ids: tuple[str, ...]


def strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


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


def polyline_length(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def parse_lane_length(elem: ET.Element) -> float:
    raw = elem.get("length")
    if raw not in (None, ""):
        try:
            value = float(raw)
            if math.isfinite(value) and value >= 0.0:
                return value
        except ValueError:
            pass
    return polyline_length(parse_shape(elem.get("shape")))


def read_edge_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "edge_id" not in reader.fieldnames or "node_index" not in reader.fieldnames:
            raise GraphError(f"{path} must contain node_index and edge_id")
        rows = list(reader)
    indices = [int(row["node_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise GraphError(f"{path} node_index is not 0..N-1 in file order")
    return [str(row["edge_id"]) for row in rows]


def load_r_graph() -> tuple[list[str], np.ndarray]:
    if not R_GRAPH.is_file():
        raise GraphError(f"missing {R_GRAPH}")
    with np.load(R_GRAPH, allow_pickle=False) as payload:
        for key in ("edge_ids", "adjacency", "num_nodes", "num_graph_edges"):
            if key not in payload.files:
                raise GraphError(f"{R_GRAPH} missing {key}")
        edge_ids = [str(item) for item in payload["edge_ids"].tolist()]
        adjacency = np.array(payload["adjacency"], copy=True)
        num_nodes = int(np.asarray(payload["num_nodes"]).reshape(-1)[0])
        num_edges = int(np.asarray(payload["num_graph_edges"]).reshape(-1)[0])
    if len(edge_ids) != num_nodes or adjacency.shape != (num_nodes, num_nodes):
        raise GraphError(f"r_graph shape drift nodes={num_nodes} ids={len(edge_ids)} A={adjacency.shape}")
    if len(set(edge_ids)) != len(edge_ids):
        raise GraphError("r_graph edge_ids are not unique")
    if set(int(item) for item in np.unique(adjacency)) - {0, 1}:
        raise GraphError("r_graph adjacency is not binary")
    directed = int(adjacency.sum())
    if directed != num_edges:
        raise GraphError(f"r_graph adjacency sum {directed} != num_graph_edges {num_edges}")
    if num_nodes != EXPECTED_NODES or directed != EXPECTED_DIRECTED_EDGES:
        raise GraphError(
            f"r_graph has {num_nodes} nodes and {directed} edges; "
            f"expected sanity check {EXPECTED_NODES} / {EXPECTED_DIRECTED_EDGES}"
        )
    return edge_ids, adjacency.astype(np.uint8, copy=False)


def validate_node_order(edge_ids: list[str]) -> None:
    mapping_ids = read_edge_ids(NODE_MAPPING)
    r_node_ids = read_edge_ids(R_NODES)
    if mapping_ids != edge_ids or r_node_ids != edge_ids:
        raise GraphError("r_graph, node_mapping, and r_nodes edge_id lists differ")


def parse_sumo_net(net_file: Path) -> tuple[dict[str, EdgeRecord], list[dict[str, str]]]:
    if not net_file.is_file():
        raise GraphError(f"SUMO net not found: {net_file}")
    edges: dict[str, EdgeRecord] = {}
    connections: list[dict[str, str]] = []
    context = ET.iterparse(net_file, events=("end",))
    for _event, elem in context:
        tag = strip_namespace(elem.tag)
        if tag == "edge":
            lanes: list[LaneRecord] = []
            for child in list(elem):
                if strip_namespace(child.tag) != "lane":
                    continue
                lane_id = child.get("id") or ""
                if not lane_id:
                    continue
                lanes.append(
                    LaneRecord(
                        lane_id=lane_id,
                        index=child.get("index") if child.get("index") is not None else str(len(lanes)),
                        length=parse_lane_length(child),
                        shape=parse_shape(child.get("shape")),
                    )
                )
            record = EdgeRecord(edge_id=elem.get("id") or "", function=elem.get("function") or "", lanes=lanes)
            if record.edge_id and record.edge_id not in edges:
                edges[record.edge_id] = record
            elem.clear()
        elif tag == "connection":
            if elem.get("from") and elem.get("to"):
                connections.append(dict(elem.attrib))
            elem.clear()
        elif tag != "lane":
            elem.clear()
    return edges, connections


def lane_by_index(edge: EdgeRecord, index_text: str) -> LaneRecord | None:
    for lane in edge.lanes:
        if lane.index == index_text:
            return lane
    try:
        position = int(index_text)
    except (TypeError, ValueError):
        return None
    if 0 <= position < len(edge.lanes):
        return edge.lanes[position]
    return None


def build_lane_lookups(edges: dict[str, EdgeRecord]) -> tuple[dict[str, LaneRecord], dict[str, str]]:
    lanes_by_id: dict[str, LaneRecord] = {}
    edge_of_lane: dict[str, str] = {}
    for edge in edges.values():
        if edge.skip:
            continue
        for lane in edge.lanes:
            if lane.lane_id:
                lanes_by_id[lane.lane_id] = lane
                edge_of_lane[lane.lane_id] = edge.edge_id
    return lanes_by_id, edge_of_lane


def index_connections(
    connections: list[dict[str, str]],
    r_edge_ids: set[str],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    index: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for attrib in connections:
        source_id = attrib.get("from", "")
        target_id = attrib.get("to", "")
        if source_id in r_edge_ids and target_id in r_edge_ids:
            index[(source_id, target_id)].append(attrib)
    return index


def lane_halves(lane: LaneRecord) -> tuple[float, float]:
    shape_len = polyline_length(lane.shape)
    official_ok = math.isfinite(lane.length) and lane.length >= 0.0
    if official_ok and shape_len > DISTANCE_ATOL and abs(float(lane.length) - shape_len) > LENGTH_MISMATCH_TOL_M:
        half = 0.5 * shape_len
        return half, half
    if official_ok:
        half = 0.5 * float(lane.length)
        return half, half
    half = 0.5 * shape_len
    return half, half


def lane_travel_length(lane: LaneRecord) -> float:
    if math.isfinite(lane.length) and lane.length >= 0.0:
        return float(lane.length)
    return polyline_length(lane.shape)


def consecutive_edge_ids(path_lane_ids: tuple[str, ...], edge_of_lane: dict[str, str]) -> tuple[str, ...]:
    edge_ids: list[str] = []
    for lane_id in path_lane_ids:
        edge_id = edge_of_lane.get(lane_id, "")
        if not edge_ids or edge_ids[-1] != edge_id:
            edge_ids.append(edge_id)
    return tuple(edge_ids)


def evaluate_connection(
    attrib: dict[str, str],
    source_edge: EdgeRecord,
    target_edge: EdgeRecord,
    edges: dict[str, EdgeRecord],
    lanes_by_id: dict[str, LaneRecord],
    edge_of_lane: dict[str, str],
) -> DirectedRoute | None:
    if attrib.get("from") != source_edge.edge_id or attrib.get("to") != target_edge.edge_id:
        return None
    source_lane = lane_by_index(source_edge, attrib.get("fromLane", "0"))
    target_lane = lane_by_index(target_edge, attrib.get("toLane", "0"))
    if source_lane is None or target_lane is None:
        return None
    via_ids = tuple(token for token in (attrib.get("via") or "").split() if token)
    via_sum = 0.0
    for via_id in via_ids:
        via_lane = lanes_by_id.get(via_id)
        via_edge = edges.get(edge_of_lane.get(via_id, ""))
        if via_lane is None or via_edge is None or not via_edge.is_internal:
            return None
        via_length = lane_travel_length(via_lane)
        if not math.isfinite(via_length) or via_length < 0.0:
            return None
        via_sum += via_length
    _start_src, mid_to_end_src = lane_halves(source_lane)
    start_to_mid_tgt, _end_tgt = lane_halves(target_lane)
    distance = mid_to_end_src + via_sum + start_to_mid_tgt
    if not math.isfinite(distance) or distance <= DISTANCE_ATOL:
        return None
    path_edge_ids = consecutive_edge_ids((source_lane.lane_id, *via_ids, target_lane.lane_id), edge_of_lane)
    if not path_edge_ids or path_edge_ids[0] != source_edge.edge_id or path_edge_ids[-1] != target_edge.edge_id:
        return None
    for edge_id in path_edge_ids[1:-1]:
        record = edges.get(edge_id)
        if record is None or not record.is_internal:
            return None
    return DirectedRoute(True, float(distance), source_lane.lane_id, target_lane.lane_id, via_ids)


def direct_center_distance(
    source_edge: EdgeRecord,
    target_edge: EdgeRecord,
    connection_index: dict[tuple[str, str], list[dict[str, str]]],
    edges: dict[str, EdgeRecord],
    lanes_by_id: dict[str, LaneRecord],
    edge_of_lane: dict[str, str],
) -> DirectedRoute:
    legal: list[DirectedRoute] = []
    for attrib in connection_index.get((source_edge.edge_id, target_edge.edge_id), []):
        candidate = evaluate_connection(attrib, source_edge, target_edge, edges, lanes_by_id, edge_of_lane)
        if candidate is not None:
            legal.append(candidate)
    if not legal:
        return DirectedRoute(False, math.inf, "", "", ())
    return min(legal, key=lambda route: (route.distance, route.source_lane_id, route.target_lane_id, route.via_lane_ids))


def symmetrize_topology(directed: np.ndarray) -> np.ndarray:
    symmetric = np.logical_or(directed, directed.T).astype(np.uint8)
    np.fill_diagonal(symmetric, 0)
    return symmetric


def gaussian_weight(distance: float, sigma: float) -> float:
    return math.exp(-(distance * distance) / (sigma * sigma))


def build_weights(
    edge_ids: list[str],
    directed: np.ndarray,
    edges: dict[str, EdgeRecord],
    connection_index: dict[tuple[str, str], list[dict[str, str]]],
    lanes_by_id: dict[str, LaneRecord],
    edge_of_lane: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    missing = [edge_id for edge_id in edge_ids if edge_id not in edges or not edges[edge_id].lanes]
    if missing:
        raise GraphError(f"R edges missing from the final net or without lanes: {missing[:8]}")
    symmetric = symmetrize_topology(directed)
    n_nodes = len(edge_ids)
    distances: list[float] = []
    pair_index: list[tuple[int, int, float]] = []
    rows, cols = np.where(np.triu(symmetric, k=1) == 1)
    for i, j in zip(rows.tolist(), cols.tolist()):
        has_forward = bool(directed[i, j])
        has_reverse = bool(directed[j, i])
        forward = (
            direct_center_distance(edges[edge_ids[i]], edges[edge_ids[j]], connection_index, edges, lanes_by_id, edge_of_lane)
            if has_forward
            else DirectedRoute(False, math.inf, "", "", ())
        )
        reverse = (
            direct_center_distance(edges[edge_ids[j]], edges[edge_ids[i]], connection_index, edges, lanes_by_id, edge_of_lane)
            if has_reverse
            else DirectedRoute(False, math.inf, "", "", ())
        )
        if has_forward and not forward.reachable:
            raise GraphError(f"directed link {edge_ids[i]}->{edge_ids[j]} has no legal lane connection")
        if has_reverse and not reverse.reachable:
            raise GraphError(f"directed link {edge_ids[j]}->{edge_ids[i]} has no legal lane connection")
        if has_forward and has_reverse and reverse.distance + DISTANCE_ATOL < forward.distance:
            selected = reverse.distance
        else:
            selected = forward.distance if has_forward else reverse.distance
        if not math.isfinite(selected) or selected <= DISTANCE_ATOL:
            raise GraphError(f"neighbour {edge_ids[i]}--{edge_ids[j]} distance {selected} is invalid")
        distances.append(float(selected))
        pair_index.append((i, j, float(selected)))
    sigma = float(np.median(np.asarray(distances, dtype=np.float64)))
    if not math.isfinite(sigma) or sigma <= DISTANCE_ATOL:
        raise GraphError(f"median sigma {sigma} is invalid")
    weights = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    removed = 0
    for i, j, distance in pair_index:
        raw = gaussian_weight(distance, sigma)
        final = 0.0 if EPSILON > 0.0 and raw < EPSILON else raw
        if final == 0.0:
            removed += 1
        value = np.float32(final)
        weights[i, j] = value
        weights[j, i] = value
    np.fill_diagonal(weights, np.float32(0.0))
    stats = {
        "directed_edges": float(int(directed.sum())),
        "raw_undirected_edges": float(len(pair_index)),
        "weighted_edges_before_threshold": float(len(pair_index)),
        "weighted_edges_removed_by_threshold": float(removed),
        "weighted_edges_after_threshold": float(len(pair_index) - removed),
        "sigma": sigma,
        "epsilon": EPSILON,
        "distance_min": float(np.min(distances)),
        "distance_max": float(np.max(distances)),
        "distance_mean": float(np.mean(distances)),
        "distance_median": float(np.median(distances)),
        "distance_std": float(np.std(distances, ddof=0)),
    }
    return weights, symmetric, stats


def validate_weights(weights: np.ndarray, symmetric: np.ndarray, num_nodes: int) -> dict[str, float]:
    if weights.shape != (num_nodes, num_nodes):
        raise GraphError(f"W shape {weights.shape}")
    if weights.dtype != np.float32:
        raise GraphError(f"W dtype {weights.dtype}")
    if not np.isfinite(weights).all():
        raise GraphError("W contains NaN or inf")
    if np.any(weights < 0.0) or np.any(weights > 1.0):
        raise GraphError("W has a weight outside [0, 1]")
    symmetry_error = float(np.max(np.abs(weights - weights.T)))
    if symmetry_error != 0.0:
        raise GraphError(f"W is not exactly symmetric, max error {symmetry_error}")
    diagonal_nonzero = int(np.count_nonzero(np.diag(weights)))
    if diagonal_nonzero:
        raise GraphError(f"W diagonal has {diagonal_nonzero} nonzeros")
    positive = weights > 0.0
    upper = np.triu(positive, k=1)
    if int(upper.sum()) * 2 != int(positive.sum()):
        raise GraphError("symmetric nonzero pattern is not paired")
    support = positive.astype(np.uint8)
    if not np.array_equal(support, np.logical_and(symmetric, positive).astype(np.uint8)):
        raise GraphError("W support is not contained in the undirected topology")
    values = weights[upper]
    return {
        "matrix_nonzero": float(int(positive.sum())),
        "unique_undirected_edges": float(int(upper.sum())),
        "min_positive": float(values.min()) if values.size else math.nan,
        "max_weight": float(weights.max()),
        "mean_positive": float(values.mean()) if values.size else math.nan,
        "median_positive": float(np.median(values)) if values.size else math.nan,
        "symmetry_max_error": symmetry_error,
        "diagonal_nonzero": float(diagonal_nonzero),
        "nan_count": float(np.isnan(weights).sum()),
        "inf_count": float(np.isinf(weights).sum()),
    }


def save_weights(weights: np.ndarray) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp.npy")
    with temporary.open("wb") as handle:
        np.save(handle, weights, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, OUTPUT)


def run_synthetic_checks() -> None:
    directed = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.uint8)
    expected = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    if not np.array_equal(symmetrize_topology(directed), expected):
        raise GraphError("synthetic symmetrize failed")
    edges = {
        "A": EdgeRecord("A", "", [LaneRecord("A_0", "0", 100.0)]),
        "B": EdgeRecord("B", "", [LaneRecord("B_0", "0", 50.0)]),
        ":J_0": EdgeRecord(":J_0", "internal", [LaneRecord(":J_0_0", "0", 10.0)]),
        ":J_1": EdgeRecord(":J_1", "internal", [LaneRecord(":J_1_0", "0", 40.0)]),
    }
    lanes_by_id, edge_of_lane = build_lane_lookups(edges)
    connections = [
        {"from": "A", "to": "B", "fromLane": "0", "toLane": "0", "via": ":J_0_0"},
        {"from": "A", "to": "B", "fromLane": "0", "toLane": "0", "via": ":J_1_0"},
    ]
    index = index_connections(connections, {"A", "B"})
    route = direct_center_distance(edges["A"], edges["B"], index, edges, lanes_by_id, edge_of_lane)
    expected_distance = 0.5 * 100.0 + 10.0 + 0.5 * 50.0
    if not route.reachable or abs(route.distance - expected_distance) > 1e-9:
        raise GraphError(f"synthetic distance {route.distance} != {expected_distance}")
    if route.via_lane_ids != (":J_0_0",):
        raise GraphError("synthetic via tie-break failed")
    sigma = 20.0
    weight = gaussian_weight(expected_distance, sigma)
    if not 0.0 < weight <= 1.0:
        raise GraphError("synthetic gaussian weight left (0, 1]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Xuancheng26 STGCN weighted adjacency")
    prep_paths.add_path_arguments(parser, net=True)
    args = parser.parse_args(argv)
    graph_dir = prep_paths.resolved(args.training_root) / "graph"
    prep_paths.bind(
        sys.modules[__name__],
        NET_FILE=prep_paths.resolved(args.net_file),
        GRAPH_DIR=graph_dir,
        R_GRAPH=graph_dir / "r_graph.npz",
        NODE_MAPPING=graph_dir / "node_mapping.csv",
        R_NODES=graph_dir / "r_nodes.csv",
        OUTPUT=graph_dir / "stgcn_weighted_adjacency.npy",
    )
    print(f"net_file={NET_FILE}")
    print(f"output={OUTPUT}")
    run_synthetic_checks()
    edge_ids, directed = load_r_graph()
    validate_node_order(edge_ids)
    edges, connections = parse_sumo_net(NET_FILE)
    lanes_by_id, edge_of_lane = build_lane_lookups(edges)
    connection_index = index_connections(connections, set(edge_ids))
    weights, symmetric, distance_stats = build_weights(
        edge_ids, directed, edges, connection_index, lanes_by_id, edge_of_lane
    )
    weight_stats = validate_weights(weights, symmetric, len(edge_ids))
    save_weights(weights)
    reloaded = np.load(OUTPUT, allow_pickle=False)
    if reloaded.shape != weights.shape or reloaded.dtype != weights.dtype or not np.array_equal(reloaded, weights):
        raise GraphError(f"reload mismatch for {OUTPUT}")
    print(f"nodes={len(edge_ids)}")
    print(f"first5={edge_ids[:5]}")
    print(f"last5={edge_ids[-5:]}")
    for key, value in {**distance_stats, **weight_stats}.items():
        print(f"{key}={value}")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GraphError as exc:
        print(f"error: {exc}")
        raise SystemExit(1)
