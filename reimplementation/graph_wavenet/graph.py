"""Static Graph WaveNet supports from the finished directed R-only adjacency.

Formulas match ``reference/Graph-WaveNet/util.py`` ``asym_adj`` / ``load_adj``.
Unlike DCRNN, Graph WaveNet has **no extra transpose**: ``nconv`` does

    einsum('ncvl,vw->ncwl')

so ``output[w] = sum_v x[v] * A[v,w]``.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from reimplementation.common.errors import ReimplementationError
from reimplementation.common.utils.hashing import sha256_file


def asym_adj(adj: np.ndarray) -> np.ndarray:
    """Original ``util.asym_adj``: row-normalized ``D_out^{-1} A``."""
    adj_sp = sp.coo_matrix(adj)
    rowsum = np.array(adj_sp.sum(1)).flatten()
    with np.errstate(divide="ignore"):
        d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat = sp.diags(d_inv)
    return np.asarray(d_mat.dot(adj_sp).astype(np.float32).todense())


def build_gwn_supports(adj_mx: np.ndarray, adjtype: str) -> list[np.ndarray]:
    """Original ``util.load_adj`` support list, without re-reading the pickle."""
    name = str(adjtype)
    if name == "doubletransition":
        supports = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif name == "transition":
        supports = [asym_adj(adj_mx)]
    else:
        raise ReimplementationError(f"unsupported adjtype {adjtype!r}; official command uses doubletransition")
    dense = []
    for matrix in supports:
        array = np.asarray(matrix, dtype=np.float32)
        if array.shape != adj_mx.shape:
            raise ReimplementationError(f"support shape {array.shape} != adj {adj_mx.shape}")
        if not np.isfinite(array).all():
            raise ReimplementationError("GWN support contains NaN or inf")
        dense.append(array)
    return dense


def load_directed_adjacency(path: Path) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if loaded.ndim != 2 or loaded.shape[0] != loaded.shape[1]:
        raise ReimplementationError(f"{path} is not square: {loaded.shape}")
    if not np.isfinite(loaded).all():
        raise ReimplementationError(f"{path} contains NaN or inf")
    return loaded.astype(np.float32, copy=True)


def load_original_pickle(path: Path) -> tuple[list[str], dict[str, int], np.ndarray]:
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except UnicodeDecodeError:
        with path.open("rb") as handle:
            payload = pickle.load(handle, encoding="latin1")
    if not isinstance(payload, (list, tuple)) or len(payload) != 3:
        raise ReimplementationError(f"{path} is not [sensor_ids, sensor_id_to_ind, adj_mx]")
    sensor_ids, mapping, adj_mx = payload
    ids = [str(item) for item in list(sensor_ids)]
    index = {str(key): int(value) for key, value in dict(mapping).items()}
    matrix = np.asarray(adj_mx)
    return ids, index, matrix


def load_gwn_graph(
    adjacency_path: Path,
    *,
    pickle_path: Path | None,
    expected_nodes: int,
    node_ids: list[str],
    adjtype: str,
) -> dict[str, Any]:
    weights = load_directed_adjacency(adjacency_path)
    if weights.shape != (expected_nodes, expected_nodes):
        raise ReimplementationError(
            f"{adjacency_path} shape {weights.shape} != [{expected_nodes}, {expected_nodes}]"
        )
    if pickle_path is not None:
        sensor_ids, mapping, pickled = load_original_pickle(pickle_path)
        if sensor_ids != node_ids:
            raise ReimplementationError("dcrnn_adj_mx.pkl sensor_ids do not match r_nodes.csv")
        expected_map = {edge_id: index for index, edge_id in enumerate(node_ids)}
        if mapping != expected_map:
            raise ReimplementationError("dcrnn_adj_mx.pkl mapping does not match node_index")
        if not np.allclose(pickled.astype(np.float32), weights, atol=1e-6):
            raise ReimplementationError("pickle adj_mx does not match dcrnn_weighted_adjacency.npy")
    supports = build_gwn_supports(weights, adjtype)
    return {
        "adjacency": weights,
        "supports": supports,
        "adjtype": adjtype,
        "graph_sha256": sha256_file(adjacency_path),
        "n_nodes": expected_nodes,
    }
