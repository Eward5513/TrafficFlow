"""Load the directed R graph and prove its node order matches the window tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from MSTT.model.graph import SpatialGraph, build_spatial_graph
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.utils.hashing import sha256_file
from reimplementation.stgcn.validation import load_edge_ids


def load_r_graph(
    graph_path: Path,
    *,
    r_nodes_path: Path,
    window_node_mapping: Path,
    expected_nodes: int,
    direction: str,
) -> tuple[SpatialGraph, dict[str, Any]]:
    """``graph_path`` is ``r_graph.npz`` (``adjacency[u, v] = 1``: traffic u -> v)."""
    with np.load(graph_path, allow_pickle=False) as payload:
        if "adjacency" not in payload.files or "edge_ids" not in payload.files:
            raise ReimplementationError(f"{graph_path} needs adjacency and edge_ids")
        adjacency = np.array(payload["adjacency"])
        graph_ids = [str(item) for item in payload["edge_ids"].tolist()]
    node_ids = load_edge_ids(window_node_mapping)
    if load_edge_ids(r_nodes_path) != node_ids:
        raise ReimplementationError("r_nodes.csv edge_id order differs from window_data/node_mapping.csv")
    if graph_ids != node_ids:
        raise ReimplementationError("r_graph.npz edge_ids differ from window_data/node_mapping.csv")
    if len(node_ids) != int(expected_nodes) or adjacency.shape != (expected_nodes, expected_nodes):
        raise ReimplementationError(f"graph has {adjacency.shape}, expected {expected_nodes} nodes")
    if int(np.count_nonzero(np.diag(adjacency))) != 0:
        raise ReimplementationError("r_graph.npz unexpectedly has self-loops")
    graph = build_spatial_graph(adjacency, direction)
    info = {
        "graph_path": Path(graph_path).as_posix(),
        "graph_sha256": sha256_file(Path(graph_path)),
        "direction": direction,
        "road_links": int(adjacency.sum()),
        "attention_edges": graph.num_edges,
        "self_loops": int(graph.num_nodes),
        "max_in_degree_with_self": int(graph.in_degree().max()),
        "min_in_degree_with_self": int(graph.in_degree().min()),
        "node_ids": node_ids,
    }
    return graph, info
