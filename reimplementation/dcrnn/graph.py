"""Load the finished DCRNN directed adjacency and build diffusion supports.

Formulas are copied from ``reference/dcrnn/lib/utils.py`` and the support
construction in ``reference/dcrnn/model/dcrnn_cell.py``. The extra transpose
is the SparseTensor layout used by ``tf.sparse_tensor_dense_matmul``.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from reimplementation.common.errors import ReimplementationError
from reimplementation.common.utils.hashing import sha256_file


def calculate_random_walk_matrix(adj_mx: np.ndarray) -> sp.coo_matrix:
    """Original ``lib.utils.calculate_random_walk_matrix``."""
    adj_mx = sp.coo_matrix(adj_mx)
    degree = np.array(adj_mx.sum(1))
    with np.errstate(divide="ignore"):
        d_inv = np.power(degree, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)
    return d_mat_inv.dot(adj_mx).tocoo()


def calculate_scaled_laplacian(
    adj_mx: np.ndarray,
    lambda_max: float | None = 2.0,
    undirected: bool = True,
) -> sp.csr_matrix:
    """Original ``lib.utils.calculate_scaled_laplacian``."""
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    adj = sp.coo_matrix(adj_mx)
    degree = np.array(adj.sum(1))
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.power(degree, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    laplacian = sp.csr_matrix(normalized)
    if lambda_max is None:
        from scipy.sparse.linalg import eigsh

        lambda_max = float(eigsh(laplacian, 1, which="LM")[0][0])
    identity = sp.identity(laplacian.shape[0], format="csr", dtype=laplacian.dtype)
    scaled = (2.0 / float(lambda_max) * laplacian) - identity
    return scaled.astype(np.float32)


def build_dcrnn_supports(adj_mx: np.ndarray, filter_type: str) -> list[np.ndarray]:
    """Mirror ``DCGRUCell.__init__`` without TensorFlow SparseTensors."""
    name = str(filter_type)
    supports: list[sp.spmatrix]
    if name == "laplacian":
        supports = [calculate_scaled_laplacian(adj_mx, lambda_max=None)]
    elif name == "random_walk":
        supports = [calculate_random_walk_matrix(adj_mx).T]
    elif name == "dual_random_walk":
        supports = [
            calculate_random_walk_matrix(adj_mx).T,
            calculate_random_walk_matrix(adj_mx.T).T,
        ]
    else:
        raise ReimplementationError(f"unsupported filter_type {filter_type!r}")
    dense = []
    for matrix in supports:
        array = np.asarray(matrix.todense() if sp.issparse(matrix) else matrix, dtype=np.float32)
        if array.shape != adj_mx.shape:
            raise ReimplementationError(f"support shape {array.shape} != adj {adj_mx.shape}")
        if not np.isfinite(array).all():
            raise ReimplementationError("diffusion support contains NaN or inf")
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
    """Same triple as ``lib.utils.load_graph_data`` / ``load_pickle``."""
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


def load_dcrnn_graph(
    adjacency_path: Path,
    *,
    pickle_path: Path | None,
    expected_nodes: int,
    node_ids: list[str],
    filter_type: str,
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
        if not np.allclose(pickled.astype(np.float32), weights, atol=0.0):
            if not np.allclose(pickled, weights, atol=1e-6):
                raise ReimplementationError("pickle adj_mx does not match dcrnn_weighted_adjacency.npy")
    supports = build_dcrnn_supports(weights, filter_type)
    return {
        "adjacency": weights,
        "supports": supports,
        "filter_type": filter_type,
        "graph_sha256": sha256_file(adjacency_path),
        "n_nodes": expected_nodes,
    }
