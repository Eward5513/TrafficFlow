"""Load the R-only undirected topology and an STFGNN temporal adjacency.

Original ``get_adjacency_matrix(..., type_='connectivity')`` writes 0/1
undirected spatial edges. This port therefore reads
``stgcn_undirected_topology.npy``, not the Gaussian weighted matrix and not
DCRNN's directed graph.

Temporal adjacency is the DTW graph from ``Temporal_Graph_gen.py``. Official
training must pass a real file. In-memory ``test_only_temporal_adjacency``
matrices are allowed only for unit tests and ``--smoke-test-only``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from reimplementation.common.data.r_only_npz_dataset import load_json
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.utils.hashing import sha256_file, sha256_numpy
from reimplementation.stfgnn.model.dtw import adjacency_edge_stats, connected_component_stats
from reimplementation.stfgnn.model.fusion_graph import (
    FUSION_STEPS,
    construct_adj_fusion,
    mask_init_from_fusion,
)

TEST_ONLY_TEMPORAL_GRAPH = "test_only_temporal_adjacency"


def load_spatial_topology(path: Path, *, expected_nodes: int = 56) -> np.ndarray:
    if not path.is_file():
        raise ReimplementationError(f"spatial topology not found: {path}")
    topology = np.array(np.load(path, allow_pickle=False), dtype=np.float32, copy=True)
    if topology.ndim != 2 or topology.shape[0] != topology.shape[1]:
        raise ReimplementationError(f"{path} is not square: {topology.shape}")
    if int(topology.shape[0]) != int(expected_nodes):
        raise ReimplementationError(f"{path} has {topology.shape[0]} nodes, expected {expected_nodes}")
    if not np.isfinite(topology).all():
        raise ReimplementationError(f"{path} contains NaN or inf")
    unique = set(np.unique(topology).tolist())
    if not unique.issubset({0.0, 1.0}):
        raise ReimplementationError(
            f"{path} is not a 0/1 topology; original STFGNN uses connectivity, not distance weights"
        )
    if not np.allclose(topology, topology.T, atol=0):
        raise ReimplementationError(f"{path} is not symmetric; original connectivity graph is undirected")
    return topology


def load_temporal_adjacency(
    path: Path,
    *,
    expected_nodes: int,
    allow_test_only: bool = False,
) -> dict[str, Any]:
    if not path.is_file():
        raise ReimplementationError(f"temporal adjacency not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        loaded = np.array(np.load(path, allow_pickle=False), dtype=np.float32, copy=True)
    elif suffix in {".csv", ".txt"}:
        loaded = np.array(np.loadtxt(path, delimiter=","), dtype=np.float32, copy=True)
    else:
        raise ReimplementationError(f"temporal adjacency must be .npy or .csv, got {path}")
    if loaded.ndim != 2 or loaded.shape[0] != loaded.shape[1]:
        raise ReimplementationError(f"{path} is not square: {loaded.shape}")
    if int(loaded.shape[0]) != int(expected_nodes):
        raise ReimplementationError(f"{path} has {loaded.shape[0]} nodes, expected {expected_nodes}")
    if not np.isfinite(loaded).all():
        raise ReimplementationError(f"{path} contains NaN or inf")
    unique = set(np.unique(loaded).tolist())
    if not unique.issubset({0.0, 1.0}):
        raise ReimplementationError(f"{path} is not a 0/1 temporal adjacency")
    meta_path = path.with_name(path.stem + "_metadata.json")
    if not meta_path.is_file():
        meta_path = path.parent / "temporal_graph_metadata.json"
    metadata: dict[str, Any] = {}
    if meta_path.is_file():
        metadata = load_json(meta_path)
        kind = str(metadata.get("graph_kind") or metadata.get("kind") or "")
        if kind == TEST_ONLY_TEMPORAL_GRAPH and not allow_test_only:
            raise ReimplementationError(
                f"{path} is marked {TEST_ONLY_TEMPORAL_GRAPH} and cannot be used for official training"
            )
    return {
        "temporal_adjacency": loaded,
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "metadata": metadata,
        "stats": adjacency_edge_stats(loaded),
        "components": connected_component_stats(loaded),
        "is_test_only": str(metadata.get("graph_kind") or "") == TEST_ONLY_TEMPORAL_GRAPH,
    }


def load_stfgnn_graph(
    *,
    topology_path: Path,
    metadata_path: Path,
    validation_path: Path,
    r_nodes_path: Path,
    node_ids: list[str],
    expected_nodes: int = 56,
    temporal_adj: np.ndarray | None = None,
    temporal_path: Path | None = None,
    temporal_is_test_only: bool = False,
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
        raise ReimplementationError(
            "spatial topology diagonal is not 0; self-loops are added only in construct_adj_fusion"
        )
    if temporal_adj is None:
        if temporal_path is None:
            raise ReimplementationError("official STFGNN training requires --temporal-adjacency")
        temporal_pack = load_temporal_adjacency(
            temporal_path,
            expected_nodes=expected_nodes,
            allow_test_only=temporal_is_test_only,
        )
        temporal = temporal_pack["temporal_adjacency"]
        temporal_sha = temporal_pack["sha256"]
        temporal_is_test_only = bool(temporal_pack["is_test_only"] or temporal_is_test_only)
        temporal_stats = temporal_pack["stats"]
        temporal_components = temporal_pack["components"]
    else:
        temporal = np.array(temporal_adj, dtype=np.float32, copy=True)
        if temporal.shape != topology.shape:
            raise ReimplementationError(
                f"temporal adjacency {temporal.shape} != spatial {topology.shape}"
            )
        temporal_sha = sha256_numpy(temporal)
        temporal_stats = adjacency_edge_stats(temporal)
        temporal_components = connected_component_stats(temporal)
        temporal_path = None
    if temporal_is_test_only is False and temporal_path is None and temporal_adj is not None:
        # In-memory matrices used by tests must be marked explicitly.
        pass
    localized = construct_adj_fusion(topology, temporal, steps=FUSION_STEPS)
    mask0 = mask_init_from_fusion(localized)
    offdiag = topology.copy()
    np.fill_diagonal(offdiag, 0.0)
    return {
        "spatial_topology": topology,
        "temporal_adjacency": temporal,
        "localized_adj": localized,
        "mask_init": mask0,
        "graph_sha256": sha256_file(topology_path),
        "topology_sha256": sha256_numpy(topology),
        "temporal_sha256": temporal_sha,
        "n_nodes": int(topology.shape[0]),
        "localized_shape": list(localized.shape),
        "spatial_self_loops": 0,
        "localized_self_loops": int(np.count_nonzero(np.diag(localized))),
        "spatial_nnz": int(np.count_nonzero(topology)),
        "temporal_nnz": int(np.count_nonzero(temporal)),
        "localized_nnz": int(np.count_nonzero(localized)),
        "undirected_neighbor_count": int(np.count_nonzero(np.triu(offdiag, k=1))),
        "temporal_stats": temporal_stats,
        "temporal_components": temporal_components,
        "temporal_graph_is_test_only": bool(temporal_is_test_only),
        "did_not_use_weighted_adjacency": True,
        "did_not_use_dcrnn_directed_graph": True,
        "did_not_normalize_fusion_adjacency": True,
        "fusion_steps": FUSION_STEPS,
        "prepared_graph_status": report.get("status"),
        "metadata_node_count": metadata.get("node_count"),
        "node_ids": node_ids,
        "temporal_path": None if temporal_path is None else Path(temporal_path).as_posix(),
    }
