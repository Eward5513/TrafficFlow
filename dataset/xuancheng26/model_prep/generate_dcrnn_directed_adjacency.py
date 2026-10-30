#!/usr/bin/env python3
"""Build the Xuancheng26 directed Gaussian adjacency used by DCRNN and Graph WaveNet.

Each directed SUMO connection keeps its own center distance. The length scale
is the same median used by the symmetric STGCN kernel. Random-walk supports
are not stored.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np

import generate_stgcn_weighted_adjacency as stgcn
import prep_paths
from generate_stgcn_weighted_adjacency import (
    DISTANCE_ATOL,
    GRAPH_DIR,
    NET_FILE,
    GraphError,
    build_lane_lookups,
    direct_center_distance,
    index_connections,
    load_r_graph,
    parse_sumo_net,
    validate_node_order,
)

WEIGHTS_PATH = GRAPH_DIR / "dcrnn_weighted_adjacency.npy"
PICKLE_PATH = GRAPH_DIR / "dcrnn_adj_mx.pkl"
EPSILON = 0.0
PICKLE_PROTOCOL = 2
EXPECTED_DIRECTED_EDGES = 3877


def gaussian_weight(distance: float, sigma: float) -> float:
    return math.exp(-(distance * distance) / (sigma * sigma))


def directional_distances(
    edge_ids: list[str],
    directed: np.ndarray,
    edges,
    connection_index,
    lanes_by_id,
    edge_of_lane,
) -> tuple[list[tuple[int, int, float]], float]:
    """Return every directed edge distance and the STGCN median sigma."""
    selected: list[float] = []
    directed_edges: list[tuple[int, int, float]] = []
    rows, cols = np.where(np.triu(np.logical_or(directed, directed.T), k=1))
    for i, j in zip(rows.tolist(), cols.tolist()):
        has_forward = bool(directed[i, j])
        has_reverse = bool(directed[j, i])
        forward = (
            direct_center_distance(
                edges[edge_ids[i]], edges[edge_ids[j]], connection_index, edges, lanes_by_id, edge_of_lane
            )
            if has_forward
            else None
        )
        reverse = (
            direct_center_distance(
                edges[edge_ids[j]], edges[edge_ids[i]], connection_index, edges, lanes_by_id, edge_of_lane
            )
            if has_reverse
            else None
        )
        if has_forward and (forward is None or not forward.reachable):
            raise GraphError(f"directed link {edge_ids[i]}->{edge_ids[j]} has no legal lane connection")
        if has_reverse and (reverse is None or not reverse.reachable):
            raise GraphError(f"directed link {edge_ids[j]}->{edge_ids[i]} has no legal lane connection")
        if has_forward and has_reverse and reverse.distance + DISTANCE_ATOL < forward.distance:
            chosen = reverse.distance
        elif has_forward:
            chosen = forward.distance
        else:
            chosen = reverse.distance
        if not math.isfinite(chosen) or chosen <= DISTANCE_ATOL:
            raise GraphError(f"neighbour {edge_ids[i]}--{edge_ids[j]} has no positive distance")
        selected.append(float(chosen))
        if has_forward:
            directed_edges.append((i, j, float(forward.distance)))
        if has_reverse:
            directed_edges.append((j, i, float(reverse.distance)))
    if len(directed_edges) != int(directed.sum()):
        raise GraphError(f"directional edge count {len(directed_edges)} != adjacency sum {int(directed.sum())}")
    sigma = float(np.median(np.asarray(selected, dtype=np.float64)))
    if not math.isfinite(sigma) or sigma <= DISTANCE_ATOL:
        raise GraphError(f"sigma {sigma} is invalid")
    return directed_edges, sigma


def build_weights(
    num_nodes: int,
    directed_edges: list[tuple[int, int, float]],
    sigma: float,
) -> tuple[np.ndarray, dict[str, float]]:
    weights64 = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    distances = []
    for src, dst, distance in directed_edges:
        raw = gaussian_weight(distance, sigma)
        final = 0.0 if EPSILON > 0.0 and raw < EPSILON else raw
        weights64[src, dst] = final
        distances.append(distance)
    np.fill_diagonal(weights64, 0.0)
    weights32 = weights64.astype(np.float32)
    lost = int(np.count_nonzero((weights64 > 0.0) & ~(weights32 > 0.0)))
    positive = weights32 > 0.0
    if int(positive.sum()) + lost != len(directed_edges):
        raise GraphError(
            f"retained {int(positive.sum())} plus float32-underflow {lost} "
            f"!= topology edges {len(directed_edges)}"
        )
    if np.allclose(weights32, weights32.T, atol=1e-6):
        raise GraphError("directed weights collapsed to a symmetric matrix")
    stats = {
        "directed_edges": float(len(directed_edges)),
        "float32_underflow_edges": float(lost),
        "sigma": sigma,
        "epsilon": EPSILON,
        "distance_min": float(np.min(distances)),
        "distance_max": float(np.max(distances)),
        "distance_mean": float(np.mean(distances)),
        "distance_median": float(np.median(distances)),
        "distance_std": float(np.std(distances, ddof=0)),
        "matrix_nonzero": float(int(positive.sum())),
        "min_positive": float(weights32[positive].min()),
        "max_weight": float(weights32.max()),
        "mean_positive": float(weights32[positive].mean()),
        "median_positive": float(np.median(weights32[positive])),
        "diagonal_nonzero": float(int(np.count_nonzero(np.diag(weights32)))),
    }
    return weights32, stats


def save_outputs(weights: np.ndarray, edge_ids: list[str]) -> None:
    sensor_ids = list(edge_ids)
    sensor_id_to_ind = {edge_id: index for index, edge_id in enumerate(sensor_ids)}
    payload = [sensor_ids, sensor_id_to_ind, weights]
    for path, writer in (
        (WEIGHTS_PATH, lambda handle: np.save(handle, weights, allow_pickle=False)),
        (PICKLE_PATH, lambda handle: pickle.dump(payload, handle, protocol=PICKLE_PROTOCOL)),
    ):
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


def reload_and_check(weights: np.ndarray, edge_ids: list[str]) -> None:
    loaded = np.load(WEIGHTS_PATH, allow_pickle=False)
    if loaded.shape != weights.shape or loaded.dtype != np.float32 or not np.array_equal(loaded, weights):
        raise GraphError(f"reload mismatch for {WEIGHTS_PATH}")
    with PICKLE_PATH.open("rb") as handle:
        sensor_ids, mapping, adj_mx = pickle.load(handle)
    if list(sensor_ids) != edge_ids:
        raise GraphError("pickle sensor_ids do not follow r_graph edge order")
    expected = {edge_id: index for index, edge_id in enumerate(edge_ids)}
    if dict(mapping) != expected:
        raise GraphError("pickle sensor_id_to_ind does not match node order")
    if not np.array_equal(np.asarray(adj_mx).astype(np.float32), weights):
        raise GraphError("pickle adj_mx does not match the npy")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Xuancheng26 DCRNN directed adjacency")
    prep_paths.add_path_arguments(parser, net=True)
    args = parser.parse_args(argv)
    graph_dir = prep_paths.resolved(args.training_root) / "graph"
    net_file = prep_paths.resolved(args.net_file)
    prep_paths.bind(
        stgcn,
        NET_FILE=net_file,
        GRAPH_DIR=graph_dir,
        R_GRAPH=graph_dir / "r_graph.npz",
        NODE_MAPPING=graph_dir / "node_mapping.csv",
        R_NODES=graph_dir / "r_nodes.csv",
        OUTPUT=graph_dir / "stgcn_weighted_adjacency.npy",
    )
    prep_paths.bind(
        sys.modules[__name__],
        NET_FILE=net_file,
        GRAPH_DIR=graph_dir,
        WEIGHTS_PATH=graph_dir / "dcrnn_weighted_adjacency.npy",
        PICKLE_PATH=graph_dir / "dcrnn_adj_mx.pkl",
    )
    print(f"net_file={NET_FILE}")
    print(f"output={WEIGHTS_PATH}")
    edge_ids, directed = load_r_graph()
    validate_node_order(edge_ids)
    edges, connections = parse_sumo_net(NET_FILE)
    lanes_by_id, edge_of_lane = build_lane_lookups(edges)
    connection_index = index_connections(connections, set(edge_ids))
    directed_edges, sigma = directional_distances(
        edge_ids, directed, edges, connection_index, lanes_by_id, edge_of_lane
    )
    weights, stats = build_weights(len(edge_ids), directed_edges, sigma)
    if int(stats["directed_edges"]) != EXPECTED_DIRECTED_EDGES:
        raise GraphError(f"directed edges {stats['directed_edges']} != sanity check {EXPECTED_DIRECTED_EDGES}")
    save_outputs(weights, edge_ids)
    reload_and_check(weights, edge_ids)
    print(f"nodes={len(edge_ids)} dtype={weights.dtype} shape={weights.shape}")
    print(f"first5={edge_ids[:5]}")
    print(f"last5={edge_ids[-5:]}")
    for key, value in stats.items():
        print(f"{key}={value}")
    print(f"wrote {WEIGHTS_PATH}")
    print(f"wrote {PICKLE_PATH}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GraphError as exc:
        print(f"error: {exc}")
        raise SystemExit(1)
