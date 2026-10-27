#!/usr/bin/env python3
"""Build the Xuancheng26 STSGCN undirected 0/1 spatial adjacency.

The matrix is ``A_sym = (A_dir OR A_dir.T)`` with a zero diagonal, in the
order of ``r_graph.npz["edge_ids"]``. The 3-step local graph and its
identity are built later by the model.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

GRAPH_DIR = Path(
    "/data5/tzhang174/TrafficFlow/xuancheng26/training_data/graph"
)
R_GRAPH = GRAPH_DIR / "r_graph.npz"
NODE_MAPPING = GRAPH_DIR / "node_mapping.csv"
R_NODES = GRAPH_DIR / "r_nodes.csv"
OUTPUT = GRAPH_DIR / "stsgcn_undirected_topology.npy"
EXPECTED_NODES = 1406
EXPECTED_DIRECTED_EDGES = 3877
EXPECTED_UNDIRECTED_EDGES = 3465


class GraphError(RuntimeError):
    pass


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
    values = set(int(item) for item in np.unique(adjacency))
    if values - {0, 1}:
        raise GraphError(f"r_graph adjacency is not binary: {sorted(values)}")
    directed = int(adjacency.sum())
    if directed != num_edges:
        raise GraphError(f"adjacency sum {directed} != num_graph_edges {num_edges}")
    if num_nodes != EXPECTED_NODES or directed != EXPECTED_DIRECTED_EDGES:
        raise GraphError(
            f"r_graph has {num_nodes} nodes and {directed} directed edges; "
            f"sanity check expected {EXPECTED_NODES} / {EXPECTED_DIRECTED_EDGES}"
        )
    return edge_ids, adjacency.astype(np.uint8, copy=False)


def validate_node_order(edge_ids: list[str]) -> None:
    if read_edge_ids(NODE_MAPPING) != edge_ids or read_edge_ids(R_NODES) != edge_ids:
        raise GraphError("r_graph, node_mapping, and r_nodes edge_id lists differ")


def build_undirected_topology(directed: np.ndarray) -> np.ndarray:
    undirected = np.logical_or(directed, directed.T).astype(np.uint8)
    np.fill_diagonal(undirected, 0)
    return undirected


def validate_topology(undirected: np.ndarray, directed: np.ndarray) -> dict[str, object]:
    num_nodes = int(directed.shape[0])
    if undirected.shape != (num_nodes, num_nodes):
        raise GraphError(f"topology shape {undirected.shape}")
    if undirected.dtype != np.uint8:
        raise GraphError(f"topology dtype {undirected.dtype}")
    if not np.array_equal(undirected, undirected.T):
        raise GraphError("topology is not exactly symmetric")
    if not np.isfinite(undirected.astype(np.float64)).all():
        raise GraphError("topology contains NaN or inf")
    unique = [int(item) for item in np.unique(undirected)]
    if unique != [0, 1] and unique != [0]:
        raise GraphError(f"topology values are not binary: {unique}")
    diagonal = int(np.count_nonzero(np.diag(undirected)))
    if diagonal:
        raise GraphError(f"diagonal has {diagonal} nonzeros")
    expected = np.logical_or(directed != 0, directed.T != 0)
    np.fill_diagonal(expected, False)
    mismatches = int(np.count_nonzero(undirected.astype(bool) != expected))
    if mismatches:
        raise GraphError(f"topology mismatches A OR A.T at {mismatches} cells")
    matrix_nonzero = int(np.count_nonzero(undirected))
    unique_edges = int(np.count_nonzero(np.triu(undirected, k=1)))
    if unique_edges * 2 != matrix_nonzero:
        raise GraphError("nonzero pattern is not a paired undirected graph")
    if unique_edges != EXPECTED_UNDIRECTED_EDGES or matrix_nonzero != EXPECTED_UNDIRECTED_EDGES * 2:
        raise GraphError(
            f"undirected edges {unique_edges} / matrix nonzeros {matrix_nonzero}; "
            f"sanity check expected {EXPECTED_UNDIRECTED_EDGES} / {EXPECTED_UNDIRECTED_EDGES * 2}"
        )
    return {
        "directed_edges": int(directed.sum()),
        "unique_undirected_edges": unique_edges,
        "matrix_nonzero": matrix_nonzero,
        "diagonal_nonzero": diagonal,
        "unique_values": unique,
        "mismatches": mismatches,
    }


def save_topology(undirected: np.ndarray) -> None:
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp.npy")
    with temporary.open("wb") as handle:
        np.save(handle, undirected, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, OUTPUT)


def main() -> int:
    edge_ids, directed = load_r_graph()
    validate_node_order(edge_ids)
    undirected = build_undirected_topology(directed)
    stats = validate_topology(undirected, directed)
    save_topology(undirected)
    reloaded = np.load(OUTPUT, allow_pickle=False)
    if reloaded.shape != undirected.shape or reloaded.dtype != undirected.dtype or not np.array_equal(reloaded, undirected):
        raise GraphError(f"reload mismatch for {OUTPUT}")
    print(f"nodes={len(edge_ids)}")
    print(f"first5={edge_ids[:5]}")
    print(f"last5={edge_ids[-5:]}")
    for key, value in stats.items():
        print(f"{key}={value}")
    print(f"dtype={undirected.dtype} shape={undirected.shape}")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GraphError as exc:
        print(f"error: {exc}")
        raise SystemExit(1)
