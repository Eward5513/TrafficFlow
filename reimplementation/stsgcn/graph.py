"""Load the R-only undirected topology used as STSGCN's spatial graph.

Original ``get_adjacency_matrix(..., type_='connectivity')`` writes 0/1
undirected edges. This port therefore reads ``stgcn_undirected_topology.npy``,
not the Gaussian weighted matrix and not DCRNN's directed graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from reimplementation.common.data.r_only_npz_dataset import load_json
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.utils.hashing import sha256_file, sha256_numpy
from reimplementation.stsgcn.model.graph import (
    LOCAL_STEPS,
    construct_localized_adjacency,
    mask_init_from_localized_adjacency,
)


def load_spatial_topology(path: Path, *, expected_nodes: int = 56) -> np.ndarray:
    if not path.is_file():
        raise ReimplementationError(f"spatial topology not found: {path}")
    loaded = np.load(path, allow_pickle=False)
    topology = np.array(loaded, dtype=np.float32, copy=True)
    if topology.ndim != 2 or topology.shape[0] != topology.shape[1]:
        raise ReimplementationError(f"{path} is not square: {topology.shape}")
    if int(topology.shape[0]) != int(expected_nodes):
        raise ReimplementationError(f"{path} has {topology.shape[0]} nodes, expected {expected_nodes}")
    if not np.isfinite(topology).all():
        raise ReimplementationError(f"{path} contains NaN or inf")
    unique = set(np.unique(topology).tolist())
    if not unique.issubset({0.0, 1.0}):
        raise ReimplementationError(
            f"{path} is not a 0/1 topology; original STSGCN uses connectivity, not distance weights"
        )
    if not np.allclose(topology, topology.T, atol=0):
        raise ReimplementationError(f"{path} is not symmetric; original connectivity graph is undirected")
    return topology


def load_stsgcn_graph(
    *,
    topology_path: Path,
    metadata_path: Path,
    validation_path: Path,
    r_nodes_path: Path,
    node_ids: list[str],
    expected_nodes: int = 56,
) -> dict[str, Any]:
    from reimplementation.stgcn.validation import load_edge_ids

    report = load_json(validation_path)
    if report.get("failures"):
        raise ReimplementationError(f"graph validation report has failures: {report['failures']}")
    status = report.get("status")
    overall = report.get("overall_validation_passed", report.get("overall_passed"))
    if status not in (None, "ok", "passed") or overall is False:
        raise ReimplementationError(f"graph validation is not ok: {validation_path}")
    metadata = load_json(metadata_path)
    r_ids = load_edge_ids(r_nodes_path)
    if r_ids != node_ids:
        raise ReimplementationError("r_nodes.csv order does not match node_mapping.csv")
    if len(node_ids) != int(expected_nodes):
        raise ReimplementationError(f"node mapping has {len(node_ids)} nodes, expected {expected_nodes}")
    topology = load_spatial_topology(topology_path, expected_nodes=expected_nodes)
    if int(np.count_nonzero(np.diag(topology))) != 0:
        raise ReimplementationError("spatial topology diagonal is not 0; self-loops are added only in construct_adj")
    localized = construct_localized_adjacency(topology, steps=LOCAL_STEPS)
    mask0 = mask_init_from_localized_adjacency(localized)
    n_nodes = int(topology.shape[0])
    offdiag = topology.copy()
    np.fill_diagonal(offdiag, 0.0)
    return {
        "spatial_topology": topology,
        "localized_adj": localized,
        "mask_init": mask0,
        "graph_sha256": sha256_file(topology_path),
        "topology_sha256": sha256_numpy(topology),
        "n_nodes": n_nodes,
        "localized_shape": list(localized.shape),
        "spatial_self_loops": 0,
        "localized_self_loops": int(np.count_nonzero(np.diag(localized))),
        "spatial_nnz": int(np.count_nonzero(topology)),
        "localized_nnz": int(np.count_nonzero(localized)),
        "undirected_neighbor_count": int(np.count_nonzero(np.triu(offdiag, k=1))),
        "did_not_use_weighted_adjacency": True,
        "did_not_use_dcrnn_directed_graph": True,
        "did_not_normalize_adjacency": True,
        "local_steps": LOCAL_STEPS,
        "prepared_graph_status": report.get("status"),
        "metadata_node_count": metadata.get("node_count"),
        "node_ids": node_ids,
    }
