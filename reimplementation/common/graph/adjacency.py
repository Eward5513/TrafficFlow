"""Load and validate a finished symmetric weighted adjacency matrix.

The R-only STGCN matrix at
``reimplementation/stgcn/prepared_data/r-only/adjacency_matrix/stgcn_weighted_adjacency.npy``
is already:

    direct-connection center distance -> Gaussian kernel -> topology mask -> symmetric W

This module must not run ``W/10000``, ``exp(-W^2/sigma^2)``, epsilon thresholding,
binarization, or any other second kernel. Original STGCN ``weight_matrix()`` is
intentionally skipped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reimplementation.common.errors import ReimplementationError

WEIGHT_ATOL = 1e-6


def load_final_weighted_adjacency(path: Path) -> np.ndarray:
    if not path.is_file():
        raise ReimplementationError(f"weighted adjacency not found: {path}")
    loaded = np.load(path, allow_pickle=False)
    if loaded.ndim != 2 or loaded.shape[0] != loaded.shape[1]:
        raise ReimplementationError(f"{path} is not square: shape={loaded.shape}")
    return loaded.astype(np.float64, copy=True)


def validate_weighted_adjacency(
    weights: np.ndarray,
    *,
    expected_nodes: int | None = 56,
    expected_undirected_neighbors: int | None = 137,
    expected_nonzero_offdiag: int | None = 274,
    atol: float = WEIGHT_ATOL,
) -> dict[str, int]:
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ReimplementationError(f"W is not square: {weights.shape}")
    n_nodes = int(weights.shape[0])
    if expected_nodes is not None and n_nodes != int(expected_nodes):
        raise ReimplementationError(f"W node count {n_nodes} != {expected_nodes}")
    if not np.isfinite(weights).all():
        raise ReimplementationError("W contains NaN or inf")
    if np.isnan(weights).any() or np.isinf(weights).any():
        raise ReimplementationError("W contains NaN or inf")
    if np.any(weights < -atol) or np.any(weights > 1.0 + atol):
        raise ReimplementationError("W has values outside [0, 1]")
    if not np.allclose(weights, weights.T, atol=atol):
        raise ReimplementationError("W is not symmetric")
    if not np.allclose(np.diag(weights), 0.0, atol=atol):
        raise ReimplementationError("W diagonal is not 0; do not add self-loops here")
    nonzero = np.abs(weights) > atol
    np.fill_diagonal(nonzero, False)
    undirected = int(np.triu(nonzero, k=1).sum())
    offdiag = int(nonzero.sum())
    if expected_undirected_neighbors is not None and undirected != int(expected_undirected_neighbors):
        raise ReimplementationError(
            f"W unique undirected neighbours {undirected} != {expected_undirected_neighbors}"
        )
    if expected_nonzero_offdiag is not None and offdiag != int(expected_nonzero_offdiag):
        raise ReimplementationError(
            f"W nonzero off-diagonal entries {offdiag} != {expected_nonzero_offdiag}"
        )
    return {
        "node_count": n_nodes,
        "undirected_neighbor_count": undirected,
        "nonzero_offdiag_count": offdiag,
    }


def assert_not_distance_matrix(weights: np.ndarray) -> None:
    """Guard against accidentally loading raw metres instead of Gaussian weights."""
    finite = weights[np.isfinite(weights)]
    if finite.size == 0:
        raise ReimplementationError("W is empty")
    if float(np.nanmax(np.abs(finite))) > 1.0 + 1e-3:
        raise ReimplementationError(
            "W looks like a distance matrix, not a finished Gaussian adjacency. "
            "Do not pass stgcn_center_distance.npy or re-run weight_matrix()."
        )
