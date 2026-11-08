"""PeMSD7 predefined forward and backward graphs on a directed distance matrix.

``normalized_laplacian`` in the reference is ``D^{-1/2} A D^{-1/2}`` applied
separately to ``A`` and ``A.T``. It is not ``I - D^{-1/2} A D^{-1/2}``.
Non-edges stay infinite, ``sigma`` is the population standard deviation of the
finite distances, and each lag ``i`` uses ``exp(-((i+1) distance)^2 / sigma^2)``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from reimplementation.common.errors import ReimplementationError


def node_order_sha256(edge_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(edge_ids).encode("utf-8")).hexdigest()


def symmetric_degree_normalize(adjacency: np.ndarray) -> np.ndarray:
    degree = np.sum(adjacency, axis=1)
    if np.any(~np.isfinite(degree)) or np.any(degree <= 0.0):
        raise ReimplementationError("predefined graph has a non-positive degree after the identity and threshold")
    scale = np.power(degree, -0.5)
    return (scale[:, None] * adjacency) * scale[None, :]


def build_predefined_kernels(
    distance: np.ndarray,
    *,
    horizon: int = 12,
    delta_pdf: float = 0.9,
) -> tuple[np.ndarray, np.ndarray, dict]:
    matrix = np.asarray(distance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ReimplementationError(f"distance matrix shape {matrix.shape} is not square")
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        raise ReimplementationError("distance matrix has no finite road connection; refusing a complete or random graph")
    sigma = float(np.std(finite))
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ReimplementationError(f"distance sigma {sigma} is not a positive finite standard deviation")
    nodes = int(matrix.shape[0])
    identity = np.eye(nodes, dtype=np.float64)
    forward = np.empty((int(horizon), nodes, nodes), dtype=np.float32)
    backward = np.empty((int(horizon), nodes, nodes), dtype=np.float32)
    per_lag = []
    for lag in range(int(horizon)):
        kernel = np.exp(-(((lag + 1) * matrix) ** 2) / (sigma ** 2))
        kernel = kernel + identity
        kernel[kernel < float(delta_pdf)] = 0.0
        kernel[kernel > 1.0] = 1.0
        kernel = kernel.astype(np.float64, copy=False)
        forward[lag] = symmetric_degree_normalize(kernel).astype(np.float32)
        backward[lag] = symmetric_degree_normalize(kernel.T).astype(np.float32)
        off = kernel.copy()
        np.fill_diagonal(off, 0.0)
        out_degree = off.sum(axis=1)
        in_degree = off.sum(axis=0)
        per_lag.append(
            {
                "lag": int(lag + 1),
                "graph_index": int(lag),
                "off_diagonal_edges": int(np.count_nonzero(off)),
                "nodes_without_out_neighbor": int(np.count_nonzero(out_degree == 0.0)),
                "nodes_without_in_neighbor": int(np.count_nonzero(in_degree == 0.0)),
                "isolated_nodes_self_loop_only": int(np.count_nonzero((out_degree == 0.0) & (in_degree == 0.0))),
            }
        )
    if not np.isfinite(forward).all() or not np.isfinite(backward).all():
        raise ReimplementationError("predefined STJG kernels are not finite")
    stats = {
        "sigma_population": sigma,
        "finite_distances": int(finite.size),
        "distance_min_m": float(np.min(finite)),
        "distance_max_m": float(np.max(finite)),
        "distance_mean_m": float(np.mean(finite)),
        "delta_pdf": float(delta_pdf),
        "horizon": int(horizon),
        "normalization": "D^{-1/2} A D^{-1/2} on A and on A.T separately",
        "per_lag": per_lag,
    }
    return forward, backward, stats


def load_predefined_graph(path: Path, *, edge_ids: list[str], delta_pdf: float, horizon: int) -> dict:
    path = Path(path)
    meta_path = path.with_name("metadata.json")
    if not path.is_file() or not meta_path.is_file():
        raise ReimplementationError(f"missing STJGCN graph asset {path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = node_order_sha256(edge_ids)
    if str(metadata.get("node_order_sha256")) != expected:
        raise ReimplementationError("STJGCN graph node order does not match the current node mapping")
    if int(metadata.get("num_nodes", -1)) != len(edge_ids):
        raise ReimplementationError("STJGCN graph node count does not match the node mapping")
    if abs(float(metadata.get("delta_pdf", -1.0)) - float(delta_pdf)) > 1e-12:
        raise ReimplementationError("STJGCN graph delta_pdf does not match the training config")
    if int(metadata.get("horizon", -1)) != int(horizon):
        raise ReimplementationError("STJGCN graph horizon does not match P")
    with np.load(path, allow_pickle=False) as payload:
        forward = np.array(payload["A_fw"], dtype=np.float32, copy=True)
        backward = np.array(payload["A_bw"], dtype=np.float32, copy=True)
    expected_shape = (int(horizon), len(edge_ids), len(edge_ids))
    if forward.shape != expected_shape or backward.shape != expected_shape:
        raise ReimplementationError(f"STJGCN kernel shape {forward.shape} != {expected_shape}")
    if not np.isfinite(forward).all() or not np.isfinite(backward).all():
        raise ReimplementationError("STJGCN kernels contain NaN or inf")
    return {
        "A_fw": forward,
        "A_bw": backward,
        "metadata": metadata,
        "graph_sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
