"""STGCN Chebyshev graph operators translated from original TensorFlow code."""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigs

from reimplementation.common.errors import ReimplementationError

CODE_NOTE = (
    "Faithful translation of reference/STGCN_IJCAI-18/utils/math_graph.py: "
    "scaled_laplacian and cheb_poly_approx. first_approx is not used. "
    "weight_matrix() is never called; W is the finished Gaussian adjacency."
)


def scaled_laplacian(weights: np.ndarray) -> np.ndarray:
    """Normalized Laplacian scaled to [-1, 1], matching original ``scaled_laplacian``.

    Original implementation::

        L = -W
        L[i,i] = d[i]
        L[i,j] /= sqrt(d[i]*d[j])   if d[i]>0 and d[j]>0
        return 2L/lambda_max - I

    Isolated nodes keep a zero row/column (the original loop does not force
    ``L[i,i]=1`` when ``d[i]==0``). For the R graph every node has degree > 0,
    so this matches ``I - D^{-1/2} W D^{-1/2}`` then scaled by ``2/lambda_max``.
    Self-loops are not added to ``W``.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ReimplementationError(f"W is not square: {weights.shape}")
    n_nodes = int(weights.shape[0])
    degree = np.sum(weights, axis=1)
    laplacian = -weights.copy()
    laplacian[np.diag_indices_from(laplacian)] = degree
    for i in range(n_nodes):
        for j in range(n_nodes):
            if degree[i] > 0.0 and degree[j] > 0.0:
                laplacian[i, j] = laplacian[i, j] / np.sqrt(degree[i] * degree[j])
    # Original uses scipy.sparse.linalg.eigs. That solver requires k < N-1, so
    # graphs with N<=2 (unit tests) fall back to a dense eigendecomposition.
    try:
        if n_nodes <= 2:
            raise ValueError("eigs needs k < N-1")
        lambda_max = eigs(laplacian, k=1, which="LR")[0][0].real
    except Exception:
        eigvals = np.linalg.eigvals(laplacian)
        lambda_max = float(np.max(np.real(eigvals)))
    if not np.isfinite(lambda_max) or abs(lambda_max) < 1e-12:
        raise ReimplementationError(f"invalid lambda_max={lambda_max}")
    scaled = 2.0 * laplacian / float(lambda_max) - np.identity(n_nodes)
    if not np.isfinite(scaled).all():
        raise ReimplementationError("scaled Laplacian contains non-finite values")
    return scaled.astype(np.float64, copy=False)


def cheb_poly_approx(scaled_l: np.ndarray, ks: int, n_nodes: int) -> np.ndarray:
    """Chebyshev polynomials concatenated as ``[n, Ks*n]``.

    Original recurrence::

        T0 = I
        T1 = L_tilde
        Tk = 2 L_tilde T_{k-1} - T_{k-2}

    returned with ``np.concatenate(..., axis=-1)``.
    """
    if ks < 1:
        raise ReimplementationError(f"spatial kernel Ks must be >= 1, got {ks}")
    scaled_l = np.asarray(scaled_l, dtype=np.float64)
    if scaled_l.shape != (n_nodes, n_nodes):
        raise ReimplementationError(
            f"scaled Laplacian shape {scaled_l.shape} != ({n_nodes}, {n_nodes})"
        )
    identity = np.identity(n_nodes, dtype=np.float64)
    if ks == 1:
        return identity.copy()
    terms = [identity.copy(), scaled_l.copy()]
    t0 = identity.copy()
    t1 = scaled_l.copy()
    for _ in range(ks - 2):
        tn = 2.0 * scaled_l @ t1 - t0
        terms.append(tn.copy())
        t0, t1 = t1, tn
    kernel = np.concatenate(terms, axis=-1)
    if kernel.shape != (n_nodes, ks * n_nodes):
        raise ReimplementationError(f"Chebyshev kernel shape {kernel.shape}")
    if not np.isfinite(kernel).all():
        raise ReimplementationError("Chebyshev kernel contains non-finite values")
    return kernel


def build_stgcn_chebyshev_kernel(weights: np.ndarray, ks: int) -> np.ndarray:
    n_nodes = int(weights.shape[0])
    scaled = scaled_laplacian(weights)
    return cheb_poly_approx(scaled, ks, n_nodes)


def chebyshev_t0_is_identity(kernel: np.ndarray, n_nodes: int, atol: float = 1e-8) -> bool:
    t0 = kernel[:, :n_nodes]
    return bool(np.allclose(t0, np.identity(n_nodes), atol=atol))
