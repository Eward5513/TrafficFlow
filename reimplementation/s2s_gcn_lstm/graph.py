"""Undirected road graph and Kipf-Welling symmetric normalization.

Xuancheng26 graph files used here:

- ``undirected_center_distance.npy``: undirected centre distance in metres
- ``stsgcn_undirected_topology.npy``: undirected 0/1 support, diagonal 0
- ``r_graph.npz``: directed 0/1 connectivity and node order
- ``stgcn_weighted_adjacency.npy`` / ``dcrnn_weighted_adjacency.npy``: Gaussian
  kernels of those distances, not the GCN weights

``Ã = A + I`` and ``Â = D̃^{-1/2} Ã D̃^{-1/2}`` follow the paper's GCN definition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from reimplementation.common.errors import ReimplementationError
from reimplementation.common.utils.hashing import sha256_file
from reimplementation.stgcn.validation import load_edge_ids

EDGE_WEIGHT_BINARY = "binary_connectivity"
EDGE_WEIGHT_DISTANCE = "distance"
EDGE_WEIGHT_GAUSSIAN = "gaussian"


def symmetric_normalized_adjacency(
    adjacency: np.ndarray,
    *,
    expected_nodes: int | None = None,
) -> np.ndarray:
    """Return ``D̃^{-1/2} (A + I) D̃^{-1/2}`` as float64.

    ``A`` must be symmetric, finite, and nonnegative, with a zero diagonal.
    Self-loops are added here and nowhere else.
    """
    matrix = np.array(adjacency, dtype=np.float64, copy=True)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ReimplementationError(f"adjacency is not square: {matrix.shape}")
    if expected_nodes is not None and int(matrix.shape[0]) != int(expected_nodes):
        raise ReimplementationError(f"adjacency has {matrix.shape[0]} nodes, expected {expected_nodes}")
    if not np.isfinite(matrix).all():
        raise ReimplementationError("adjacency contains NaN or inf")
    if np.any(matrix < -1e-8):
        raise ReimplementationError("adjacency has negative weights; distances or binary weights must be >= 0")
    if not np.allclose(matrix, matrix.T, atol=1e-6):
        raise ReimplementationError("adjacency is not symmetric; this GCN uses an undirected graph")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-6):
        raise ReimplementationError("adjacency diagonal is not 0; self-loops are added only as A+I")
    matrix = np.maximum(matrix, 0.0)
    np.fill_diagonal(matrix, 0.0)
    identity = np.eye(matrix.shape[0], dtype=np.float64)
    augmented = matrix + identity
    degree = augmented.sum(axis=1)
    if np.any(degree <= 0.0) or not np.isfinite(degree).all():
        raise ReimplementationError("augmented degree is not strictly positive")
    inv_sqrt = np.power(degree, -0.5)
    if not np.isfinite(inv_sqrt).all():
        raise ReimplementationError("degree inverse square root is not finite")
    normalized = inv_sqrt[:, None] * augmented * inv_sqrt[None, :]
    if not np.isfinite(normalized).all():
        raise ReimplementationError("normalized adjacency contains NaN or inf")
    return normalized


def _load_square(path: Path) -> np.ndarray:
    if not path.is_file():
        raise ReimplementationError(f"graph file not found: {path}")
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if "adjacency" not in payload.files:
                raise ReimplementationError(f"{path} has no adjacency array")
            matrix = np.array(payload["adjacency"], copy=True)
    else:
        matrix = np.array(np.load(path, allow_pickle=False), copy=True)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ReimplementationError(f"{path} is not square: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ReimplementationError(f"{path} contains NaN or inf")
    return matrix


def _edge_ids_from_r_graph(path: Path) -> list[str]:
    with np.load(path, allow_pickle=False) as payload:
        if "edge_ids" not in payload.files:
            raise ReimplementationError(f"{path} has no edge_ids")
        return [str(item) for item in payload["edge_ids"].tolist()]


def load_s2s_graph(
    *,
    topology_path: Path,
    r_graph_path: Path,
    r_nodes_path: Path,
    node_mapping_path: Path,
    expected_nodes: int,
    edge_weight: str = EDGE_WEIGHT_BINARY,
    distance_path: Path | None = None,
) -> dict[str, Any]:
    """Load an undirected weighted ``A`` aligned with the NPZ node order.

    Node order is ``node_mapping.csv``, and it must match ``r_nodes.csv`` and
    ``r_graph.npz`` ``edge_ids``. The undirected topology must be the
    symmetrized directed road graph, so feature axis ``N`` and ``A`` match.
    """
    mapped = load_edge_ids(node_mapping_path)
    listed = load_edge_ids(r_nodes_path)
    graph_ids = _edge_ids_from_r_graph(r_graph_path)
    if mapped != listed:
        raise ReimplementationError("r_nodes.csv order does not match node_mapping.csv")
    if mapped != graph_ids:
        raise ReimplementationError("r_graph.npz edge_ids do not match node_mapping.csv")
    if len(mapped) != int(expected_nodes):
        raise ReimplementationError(f"node table has {len(mapped)} nodes, expected {expected_nodes}")
    topology = _load_square(topology_path)
    if tuple(topology.shape) != (int(expected_nodes), int(expected_nodes)):
        raise ReimplementationError(f"{topology_path} shape {topology.shape} != {(expected_nodes, expected_nodes)}")
    unique = {float(item) for item in np.unique(topology).tolist()}
    if not unique.issubset({0.0, 1.0}):
        raise ReimplementationError(f"{topology_path} is not a 0/1 road topology")
    if not np.array_equal(topology, topology.T):
        raise ReimplementationError(f"{topology_path} is not symmetric")
    if int(np.count_nonzero(np.diag(topology))) != 0:
        raise ReimplementationError("undirected topology diagonal is not 0")
    directed = _load_square(r_graph_path)
    if directed.shape != topology.shape:
        raise ReimplementationError("r_graph adjacency shape does not match the undirected topology")
    undirected = np.logical_or(directed, directed.T)
    np.fill_diagonal(undirected, False)
    if not np.array_equal(undirected.astype(np.uint8), topology.astype(np.uint8)):
        raise ReimplementationError(
            "undirected topology is not the symmetrized r_graph; refusing a mismatched node order"
        )
    mode = str(edge_weight)
    if mode == EDGE_WEIGHT_BINARY:
        if distance_path is not None:
            raise ReimplementationError("binary_connectivity does not take a distance matrix")
        weights = topology.astype(np.float64, copy=False)
        weight_sha = None
    elif mode in {EDGE_WEIGHT_DISTANCE, EDGE_WEIGHT_GAUSSIAN}:
        if distance_path is None:
            raise ReimplementationError(f"{mode} edge weights require a weight matrix path")
        distance = np.array(_load_square(distance_path), dtype=np.float64, copy=True)
        if distance.shape != topology.shape:
            raise ReimplementationError(f"weight matrix shape {distance.shape} != topology {topology.shape}")
        if np.any(distance < -1e-8) or not np.allclose(distance, distance.T, atol=1e-6):
            raise ReimplementationError("weight matrix must be symmetric and nonnegative")
        if not np.allclose(np.diag(distance), 0.0, atol=1e-6):
            raise ReimplementationError("weight matrix diagonal must be 0")
        support = distance > 1e-8
        topo = topology.astype(bool)
        if np.any(support & ~topo):
            raise ReimplementationError("weight matrix has an edge outside the undirected road topology")
        if mode == EDGE_WEIGHT_DISTANCE and not np.array_equal(support, topo):
            raise ReimplementationError("distance support does not match the undirected road topology")
        weights = distance
        weight_sha = sha256_file(distance_path)
    else:
        raise ReimplementationError(
            f"edge_weight must be {EDGE_WEIGHT_BINARY!r}, {EDGE_WEIGHT_DISTANCE!r}, "
            f"or {EDGE_WEIGHT_GAUSSIAN!r}, got {mode!r}"
        )
    normalized = symmetric_normalized_adjacency(weights, expected_nodes=int(expected_nodes))
    offdiag = np.array(weights, copy=True)
    np.fill_diagonal(offdiag, 0.0)
    return {
        "adjacency": np.ascontiguousarray(weights, dtype=np.float64),
        "a_hat": normalized,
        "node_ids": mapped,
        "edge_weight": mode,
        "graph_sha256": sha256_file(topology_path),
        "distance_sha256": weight_sha,
        "n_nodes": int(expected_nodes),
        "undirected_neighbor_count": int(np.count_nonzero(np.triu(offdiag > 0.0, k=1))),
        "topology_nnz": int(np.count_nonzero(topology)),
        "directed_edge_count": int(np.count_nonzero(directed)),
        "matches_symmetrized_r_graph": True,
        "did_not_use_gaussian_kernel": mode != EDGE_WEIGHT_GAUSSIAN,
        "did_not_invent_distances": mode == EDGE_WEIGHT_BINARY,
        "self_loops_in_stored_a": 0,
    }
