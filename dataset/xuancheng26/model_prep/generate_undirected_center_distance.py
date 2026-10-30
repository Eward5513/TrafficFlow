#!/usr/bin/env python3
"""Save the Xuancheng26 undirected centre-distance adjacency.

Each undirected road neighbour gets the same metre distance already used to
build ``stgcn_weighted_adjacency.npy``. This file keeps that distance. It does
not apply the Gaussian kernel, a distance inverse, or a threshold.

A directed link uses the shortest legal SUMO connection:

    0.5 * source lane length + internal via lengths + 0.5 * target lane length

When both directions exist and the reverse distance is strictly shorter, that
reverse distance is stored on both matrix entries. Otherwise the forward
distance is stored. The diagonal stays 0. Node order is ``r_graph.npz``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

import generate_stgcn_weighted_adjacency as stgcn
import prep_paths
from generate_stgcn_weighted_adjacency import (
    DISTANCE_ATOL,
    GRAPH_DIR,
    NET_FILE,
    DirectedRoute,
    GraphError,
    build_lane_lookups,
    direct_center_distance,
    gaussian_weight,
    index_connections,
    load_r_graph,
    parse_sumo_net,
    symmetrize_topology,
    validate_node_order,
)
from generate_stsgcn_undirected_topology import EXPECTED_UNDIRECTED_EDGES

OUTPUT = GRAPH_DIR / "undirected_center_distance.npy"
METADATA = GRAPH_DIR / "undirected_center_distance_metadata.json"
STGCN_GAUSSIAN = GRAPH_DIR / "stgcn_weighted_adjacency.npy"
STSGCN_TOPOLOGY = GRAPH_DIR / "stsgcn_undirected_topology.npy"
PROTECTED = (
    STGCN_GAUSSIAN,
    GRAPH_DIR / "dcrnn_weighted_adjacency.npy",
    GRAPH_DIR / "dcrnn_adj_mx.pkl",
    STSGCN_TOPOLOGY,
    GRAPH_DIR / "r_graph.npz",
)


def undirected_center_distances(
    edge_ids: list[str],
    directed: np.ndarray,
    edges,
    connection_index,
    lanes_by_id,
    edge_of_lane,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return the symmetric metre matrix used before the STGCN Gaussian kernel."""
    missing = [edge_id for edge_id in edge_ids if edge_id not in edges or not edges[edge_id].lanes]
    if missing:
        raise GraphError(f"R edges missing from the final net or without lanes: {missing[:8]}")
    symmetric = symmetrize_topology(directed)
    distances: list[float] = []
    pair_index: list[tuple[int, int, float]] = []
    used_reverse = 0
    rows, cols = np.where(np.triu(symmetric, k=1) == 1)
    for i, j in zip(rows.tolist(), cols.tolist()):
        has_forward = bool(directed[i, j])
        has_reverse = bool(directed[j, i])
        forward = (
            direct_center_distance(
                edges[edge_ids[i]], edges[edge_ids[j]], connection_index, edges, lanes_by_id, edge_of_lane
            )
            if has_forward
            else DirectedRoute(False, math.inf, "", "", ())
        )
        reverse = (
            direct_center_distance(
                edges[edge_ids[j]], edges[edge_ids[i]], connection_index, edges, lanes_by_id, edge_of_lane
            )
            if has_reverse
            else DirectedRoute(False, math.inf, "", "", ())
        )
        if has_forward and not forward.reachable:
            raise GraphError(f"directed link {edge_ids[i]}->{edge_ids[j]} has no legal lane connection")
        if has_reverse and not reverse.reachable:
            raise GraphError(f"directed link {edge_ids[j]}->{edge_ids[i]} has no legal lane connection")
        if has_forward and has_reverse and reverse.distance + DISTANCE_ATOL < forward.distance:
            selected = reverse.distance
            used_reverse += 1
        else:
            selected = forward.distance if has_forward else reverse.distance
        if not math.isfinite(selected) or selected <= DISTANCE_ATOL:
            raise GraphError(f"neighbour {edge_ids[i]}--{edge_ids[j]} distance {selected} is invalid")
        distances.append(float(selected))
        pair_index.append((i, j, float(selected)))
    if len(pair_index) != EXPECTED_UNDIRECTED_EDGES:
        raise GraphError(f"undirected neighbours {len(pair_index)} != {EXPECTED_UNDIRECTED_EDGES}")
    matrix = np.zeros((len(edge_ids), len(edge_ids)), dtype=np.float64)
    for i, j, distance in pair_index:
        matrix[i, j] = distance
        matrix[j, i] = distance
    values = np.asarray(distances, dtype=np.float64)
    stats = {
        "directed_edges": float(int(directed.sum())),
        "undirected_edges": float(len(pair_index)),
        "pairs_using_shorter_reverse_distance": float(used_reverse),
        "distance_min_m": float(np.min(values)),
        "distance_max_m": float(np.max(values)),
        "distance_mean_m": float(np.mean(values)),
        "distance_median_m": float(np.median(values)),
        "distance_std_m": float(np.std(values, ddof=0)),
    }
    return matrix, stats


def validate_matrix(matrix: np.ndarray, directed: np.ndarray) -> None:
    topology = symmetrize_topology(directed).astype(bool)
    if matrix.shape != topology.shape or matrix.dtype != np.float64:
        raise GraphError(f"distance matrix shape/dtype {matrix.shape} {matrix.dtype}")
    if not np.isfinite(matrix).all():
        raise GraphError("distance matrix contains NaN or inf")
    if np.any(matrix < 0.0):
        raise GraphError("distance matrix has a negative entry")
    if not np.array_equal(matrix, matrix.T):
        raise GraphError("distance matrix is not exactly symmetric")
    if int(np.count_nonzero(np.diag(matrix))) != 0:
        raise GraphError("distance matrix diagonal is not 0")
    support = matrix > DISTANCE_ATOL
    if not np.array_equal(support, topology):
        raise GraphError("distance support does not match the undirected road topology")
    stored = np.load(STSGCN_TOPOLOGY, allow_pickle=False)
    if not np.array_equal(stored.astype(bool), topology):
        raise GraphError("stsgcn_undirected_topology.npy does not match the symmetrized r_graph")
    positive = matrix[np.triu(support, k=1)]
    if np.any(positive <= DISTANCE_ATOL):
        raise GraphError("an undirected neighbour has a non-positive distance")


def gaussian_from_distances(matrix: np.ndarray) -> np.ndarray:
    """Reproduce the saved STGCN kernel from these metres. Used only as a check."""
    upper = np.triu(matrix > 0.0, k=1)
    distances = matrix[upper]
    sigma = float(np.median(distances))
    if not math.isfinite(sigma) or sigma <= DISTANCE_ATOL:
        raise GraphError(f"median distance {sigma} is invalid")
    weights = np.zeros(matrix.shape, dtype=np.float32)
    rows, cols = np.nonzero(upper)
    for i, j in zip(rows.tolist(), cols.tolist()):
        value = np.float32(gaussian_weight(float(matrix[i, j]), sigma))
        weights[i, j] = value
        weights[j, i] = value
    return weights


def matches_existing_gaussian(matrix: np.ndarray) -> None:
    if not STGCN_GAUSSIAN.is_file():
        raise GraphError(f"missing {STGCN_GAUSSIAN}; refusing to write distances without the kernel check")
    rebuilt = gaussian_from_distances(matrix)
    existing = np.load(STGCN_GAUSSIAN, allow_pickle=False)
    if rebuilt.shape != existing.shape or not np.array_equal(rebuilt, existing):
        gap = float(np.max(np.abs(rebuilt.astype(np.float64) - existing.astype(np.float64))))
        raise GraphError(
            f"rebuilt Gaussian kernel does not match {STGCN_GAUSSIAN.name}; max abs gap {gap}. "
            "The distance matrix would not be the one previously used for STGCN."
        )


def save_outputs(matrix: np.ndarray, metadata: dict[str, object]) -> None:
    for path in PROTECTED:
        if OUTPUT.resolve() == path.resolve() or METADATA.resolve() == path.resolve():
            raise GraphError(f"refusing to overwrite {path}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp.npy")
    with temporary.open("wb") as handle:
        np.save(handle, matrix, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, OUTPUT)
    text_tmp = METADATA.with_name(f".{METADATA.name}.tmp")
    text_tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(text_tmp, METADATA)


def reload_and_check(matrix: np.ndarray) -> None:
    loaded = np.load(OUTPUT, allow_pickle=False)
    if loaded.shape != matrix.shape or loaded.dtype != np.float64 or not np.array_equal(loaded, matrix):
        raise GraphError(f"reload mismatch for {OUTPUT}")
    payload = json.loads(METADATA.read_text(encoding="utf-8"))
    if int(payload["num_nodes"]) != int(matrix.shape[0]):
        raise GraphError("metadata node count does not match the matrix")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Xuancheng26 undirected centre-distance matrix")
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
        OUTPUT=graph_dir / "undirected_center_distance.npy",
        METADATA=graph_dir / "undirected_center_distance_metadata.json",
        STGCN_GAUSSIAN=graph_dir / "stgcn_weighted_adjacency.npy",
        STSGCN_TOPOLOGY=graph_dir / "stsgcn_undirected_topology.npy",
        PROTECTED=(
            graph_dir / "stgcn_weighted_adjacency.npy",
            graph_dir / "dcrnn_weighted_adjacency.npy",
            graph_dir / "dcrnn_adj_mx.pkl",
            graph_dir / "stsgcn_undirected_topology.npy",
            graph_dir / "r_graph.npz",
        ),
    )
    print(f"net_file={NET_FILE}")
    print(f"output={OUTPUT}")
    edge_ids, directed = load_r_graph()
    validate_node_order(edge_ids)
    edges, connections = parse_sumo_net(NET_FILE)
    lanes_by_id, edge_of_lane = build_lane_lookups(edges)
    connection_index = index_connections(connections, set(edge_ids))
    matrix, stats = undirected_center_distances(
        edge_ids, directed, edges, connection_index, lanes_by_id, edge_of_lane
    )
    validate_matrix(matrix, directed)
    matches_existing_gaussian(matrix)
    metadata = {
        "num_nodes": len(edge_ids),
        "node_order": "r_graph.npz edge_ids, matching node_mapping.csv and r_nodes.csv",
        "distance_unit": "meters",
        "distance_definition": (
            "Undirected centre distance. For one directed SUMO connection: "
            "0.5 * source lane length + internal via lengths + 0.5 * target lane length. "
            "The shortest legal connection is kept. If both directions exist and the reverse "
            "distance is strictly shorter, that value is stored symmetrically; otherwise the "
            "forward distance is stored. Diagonal is 0. No Gaussian, inverse, or threshold."
        ),
        "gaussian_kernel_applied": False,
        "matches_stgcn_gaussian_kernel": True,
        "stgcn_gaussian_path": STGCN_GAUSSIAN.as_posix(),
        "net_file": NET_FILE.as_posix(),
        "output": OUTPUT.as_posix(),
        **stats,
    }
    save_outputs(matrix, metadata)
    reload_and_check(matrix)
    print(f"nodes={len(edge_ids)} dtype={matrix.dtype} shape={matrix.shape}")
    for key, value in stats.items():
        print(f"{key}={value}")
    print(f"wrote {OUTPUT}")
    print(f"wrote {METADATA}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GraphError as exc:
        print(f"error: {exc}")
        raise SystemExit(1)
