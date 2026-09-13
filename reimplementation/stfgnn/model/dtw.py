"""DTW temporal-graph math from official ``data/Temporal_Graph_gen.py``.

Original repo: https://github.com/MengzhangLI/STFGNN
The local ``reference/STFGNN`` copy does not include ``Temporal_Graph_gen.py``;
this module follows that upstream file, not a third-party rewrite.
"""

from __future__ import annotations

import numpy as np


def normalize_per_day(series: np.ndarray) -> np.ndarray:
    """Z-score each day independently. ``series`` shape ``(n_days, period)``."""
    mu = np.mean(series, axis=1, keepdims=True)
    std = np.std(series, axis=1, keepdims=True)
    return (series - mu) / std


def pairwise_slot_cost(left: np.ndarray, right: np.ndarray, order: int = 1) -> np.ndarray:
    """``||left[d, t_i] - right[d, t_j]||`` over days, as in original ``compute_dtw``.

    ``left``/``right``: ``(n_days, period)``. Returns ``(period, period)``.
    """
    period = int(left.shape[1])
    delta = np.reshape(left, (-1, 1, period)) - np.reshape(right, (-1, period, 1))
    return np.linalg.norm(delta, axis=0, ord=order)


def constrained_dtw_distance(
    local_cost: np.ndarray,
    *,
    order: int = 1,
    window: int = 12,
) -> float:
    """Sakoe-Chiba band DTW on a ``(period, period)`` cost matrix.

    Matches the nested loops in original ``compute_dtw`` (``Ts=window``).
    Out-of-band cells stay 0 and are never used as path predecessors.
    """
    period = int(local_cost.shape[0])
    if local_cost.shape != (period, period):
        raise ValueError(f"local cost must be square, got {local_cost.shape}")
    cost = np.asarray(local_cost, dtype=np.float64)
    accum = np.zeros((period, period), dtype=np.float64)
    ts = int(window)
    for i in range(period):
        j_start = max(0, i - ts)
        j_end = min(period, i + ts + 1)
        for j in range(j_start, j_end):
            step = cost[i, j] ** order
            if i == 0 and j == 0:
                accum[i, j] = step
                continue
            if i == 0:
                accum[i, j] = step + accum[i, j - 1]
                continue
            if j == 0:
                accum[i, j] = step + accum[i - 1, j]
                continue
            if j == i - ts:
                accum[i, j] = step + min(accum[i - 1, j - 1], accum[i - 1, j])
                continue
            if j == i + ts:
                accum[i, j] = step + min(accum[i - 1, j - 1], accum[i, j - 1])
                continue
            accum[i, j] = step + min(accum[i - 1, j - 1], accum[i - 1, j], accum[i, j - 1])
    return float(accum[-1, -1] ** (1.0 / order))


def compute_dtw(
    left: np.ndarray,
    right: np.ndarray,
    *,
    order: int = 1,
    window: int = 12,
    normal: bool = True,
) -> float:
    """Exact port of original ``compute_dtw(a, b, order, Ts, normal)``."""
    series_a = np.array(left, dtype=np.float64, copy=True)
    series_b = np.array(right, dtype=np.float64, copy=True)
    if series_a.shape != series_b.shape or series_a.ndim != 2:
        raise ValueError(f"DTW inputs must share shape (n_days, period), got {series_a.shape} vs {series_b.shape}")
    if normal:
        series_a = normalize_per_day(series_a)
        series_b = normalize_per_day(series_b)
    local = pairwise_slot_cost(series_a, series_b, order=order)
    return constrained_dtw_distance(local, order=order, window=window)


def pairwise_dtw_distance_matrix(
    daily: np.ndarray,
    *,
    order: int = 1,
    window: int = 12,
    normal: bool = True,
) -> np.ndarray:
    """Upper-triangle DTW then ``d + d.T``, as in the original double loop.

    ``daily`` shape: ``(n_days, period, n_nodes)``. Diagonal stays 0.
    The input array is never modified.
    """
    if daily.ndim != 3:
        raise ValueError(f"daily series must be (n_days, period, n_nodes), got {daily.shape}")
    source = np.array(daily, dtype=np.float64, copy=True)
    n_nodes = int(source.shape[2])
    distances = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            distances[i, j] = compute_dtw(
                source[:, :, i],
                source[:, :, j],
                order=order,
                window=window,
                normal=normal,
            )
    return distances + distances.T


def sparsify_temporal_adjacency(
    distances: np.ndarray,
    *,
    sparsity: float = 0.01,
    top_k: int | None = None,
) -> np.ndarray:
    """Original sparsity: ``k = int(N * sparsity)`` nearest DTW neighbours, then symmetrize.

    Original then does ``adj = dtw + dtw.T`` again (ranking-invariant) and::

        w_adj[i, argsort(row)[:k]] = 1
        if w_adj[i,j] != w_adj[j,i] and w_adj[i,j] == 0: w_adj[i,j] = 1
        w_adj[i,i] = 1

    ``np.argsort(..., kind='stable')`` is used so equal distances keep a
    deterministic left-to-right index order. Original used the NumPy default
    (unstable quicksort).
    """
    matrix = np.array(distances, dtype=np.float64, copy=True)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"distance matrix must be square, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("DTW distance matrix contains NaN or inf")
    ndim = int(matrix.shape[0])
    adj = matrix + matrix.T
    k = int(top_k) if top_k is not None else int(ndim * float(sparsity))
    if k < 0:
        raise ValueError(f"top-k must be >= 0, got {k}")
    weighted = np.zeros((ndim, ndim), dtype=np.float32)
    for i in range(ndim):
        nearest = np.argsort(adj[i, :], kind="stable")[:k]
        weighted[i, nearest] = 1.0
    for i in range(ndim):
        for j in range(ndim):
            if weighted[i, j] != weighted[j, i] and weighted[i, j] == 0:
                weighted[i, j] = 1.0
            if i == j:
                weighted[i, j] = 1.0
    return weighted


def connected_component_stats(adjacency: np.ndarray) -> dict[str, int | list[int]]:
    """Undirected component counts on a 0/1 adjacency matrix."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    matrix = np.array(adjacency, dtype=np.float64, copy=True)
    graph = csr_matrix((matrix != 0).astype(np.int8))
    n_components, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels, minlength=int(n_components))
    isolated = int(np.count_nonzero(sizes == 1))
    return {
        "component_count": int(n_components),
        "largest_component_size": int(sizes.max()) if sizes.size else 0,
        "isolated_node_count": isolated,
        "component_sizes": [int(item) for item in sizes.tolist()],
    }


def adjacency_edge_stats(adjacency: np.ndarray) -> dict[str, int]:
    matrix = np.array(adjacency, dtype=np.float32, copy=True)
    n_nodes = int(matrix.shape[0])
    nnz = int(np.count_nonzero(matrix))
    self_loops = int(np.count_nonzero(np.diag(matrix)))
    offdiag = matrix.copy()
    np.fill_diagonal(offdiag, 0.0)
    undirected = int(np.count_nonzero(np.triu(offdiag, k=1)))
    return {
        "n_nodes": n_nodes,
        "nnz": nnz,
        "self_loops": self_loops,
        "offdiag_nnz": int(np.count_nonzero(offdiag)),
        "undirected_offdiag_pairs": undirected,
    }
