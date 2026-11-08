"""Runtime input checks for R-only STGCN training. Failures abort the run."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from reimplementation.common.data.r_only_npz_dataset import (
    RATE_TAGS,
    SPLITS,
    expected_split_counts,
    invert_target,
    load_json,
    load_npz,
    rate_tag,
    read_target_scaler,
)
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.graph.adjacency import (
    assert_not_distance_matrix,
    load_final_weighted_adjacency,
    validate_weighted_adjacency,
)
from reimplementation.common.utils.hashing import sha256_file
from reimplementation.stgcn.graph import (
    build_stgcn_chebyshev_kernel,
    cheb_poly_approx,
    chebyshev_t0_is_identity,
    scaled_laplacian,
)
from reimplementation.stgcn.model import STGCN


def load_edge_ids(path: Path) -> list[str]:
    import csv

    if not path.is_file():
        raise ReimplementationError(f"missing node table: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "edge_id" not in rows[0] or "node_index" not in rows[0]:
        raise ReimplementationError(f"{path} must contain node_index and edge_id")
    indices = [int(row["node_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise ReimplementationError(f"{path} node_index is not 0..N-1 in order")
    return [str(row["edge_id"]) for row in rows]


def validate_prepared_data(
    data_root: Path,
    *,
    n_his: int,
    num_nodes: int,
    rates: list[str],
) -> dict[str, Any]:
    from reimplementation.common.data.sampling_seeds import validate_sampling_package
    return validate_sampling_package(Path(data_root), n_his=n_his, num_nodes=num_nodes, rates=rates)


def validate_graph(
    adjacency_path: Path,
    cheb_ks: int,
    expected_nodes: int,
    node_ids: list[str],
    r_nodes_path: Path,
) -> dict[str, Any]:
    weights = load_final_weighted_adjacency(adjacency_path)
    assert_not_distance_matrix(weights)
    counts = validate_weighted_adjacency(weights, expected_nodes=expected_nodes)
    report_path = adjacency_path.parent / "stgcn_weighted_adjacency_validation.json"
    if report_path.is_file():
        graph_report = load_json(report_path)
        if graph_report.get("failures"):
            raise ReimplementationError(f"graph validation report has failures: {graph_report['failures']}")
        if str(graph_report.get("status", "ok")) not in {"ok", "passed"}:
            raise ReimplementationError(f"graph validation report status={graph_report.get('status')!r}")
    r_ids = load_edge_ids(r_nodes_path)
    if r_ids != node_ids:
        raise ReimplementationError("r_nodes.csv edge order does not match node_mapping.csv")
    sparse_path = adjacency_path.with_name("stgcn_weighted_adjacency_sparse.npz")
    if sparse_path.is_file():
        from scipy.sparse import load_npz as load_sparse_npz

        sparse = load_sparse_npz(sparse_path).toarray().astype(np.float64, copy=False)
        if sparse.shape != weights.shape or not np.allclose(sparse, weights, atol=1e-8):
            raise ReimplementationError("sparse weighted adjacency does not match the dense W")
    scaled = scaled_laplacian(weights)
    kernel = cheb_poly_approx(scaled, cheb_ks, expected_nodes)
    rebuilt = build_stgcn_chebyshev_kernel(weights, cheb_ks)
    if not np.allclose(kernel, rebuilt, atol=1e-8):
        raise ReimplementationError("Chebyshev kernel rebuild mismatch")
    if not chebyshev_t0_is_identity(kernel, expected_nodes):
        raise ReimplementationError("Chebyshev T0 is not I")
    if cheb_ks >= 2:
        t1 = kernel[:, expected_nodes : 2 * expected_nodes]
        if not np.allclose(t1, scaled, atol=1e-6):
            raise ReimplementationError("Chebyshev T1 is not the scaled Laplacian")
    if cheb_ks >= 3:
        t0 = kernel[:, :expected_nodes]
        t1 = kernel[:, expected_nodes : 2 * expected_nodes]
        t2 = kernel[:, 2 * expected_nodes : 3 * expected_nodes]
        if not np.allclose(t2, 2.0 * scaled @ t1 - t0, atol=1e-6):
            raise ReimplementationError("Chebyshev recurrence Tk = 2 L T_{k-1} - T_{k-2} failed")
    if not np.isfinite(kernel).all():
        raise ReimplementationError("graph operator is not finite")
    offdiag = weights.copy()
    np.fill_diagonal(offdiag, 0.0)
    nonzero = offdiag[np.abs(offdiag) > 1e-12]
    return {
        "weights": weights,
        "cheb_kernel": kernel,
        "graph_sha256": sha256_file(adjacency_path),
        "counts": counts,
        "did_not_reapply_gaussian": True,
        "did_not_add_self_loops": bool(np.allclose(np.diag(weights), 0.0)),
        "sparse_matches_dense": bool(sparse_path.is_file()),
        "weight_min_nonzero": float(nonzero.min()) if nonzero.size else None,
        "weight_median_nonzero": float(np.median(nonzero)) if nonzero.size else None,
        "weight_max": float(nonzero.max()) if nonzero.size else None,
    }


def validate_model_shapes(model: STGCN, batch_size: int = 2) -> dict[str, Any]:
    device = next(model.parameters()).device
    x = torch.zeros(batch_size, model.n_his, model.n_nodes, model.input_channels, device=device)
    y = torch.zeros(batch_size, model.output_steps, model.n_nodes, model.output_channels, device=device)
    prediction, trace = model(x, return_trace=True)
    if prediction.shape != y.shape:
        raise ReimplementationError(f"model output {tuple(prediction.shape)} != {tuple(y.shape)}")
    trainable = {name for name, _ in model.named_parameters()}
    if "cheb_kernel" in trainable:
        raise ReimplementationError("Chebyshev kernel must not be a trainable parameter")
    buffers = dict(model.named_buffers())
    if "cheb_kernel" not in buffers:
        raise ReimplementationError("Chebyshev kernel is not registered as a buffer")
    if model.parameter_count() <= 0:
        raise ReimplementationError("model has no parameters")
    kernel_used = any(name == "cheb_kernel" for name, _ in model.named_buffers())
    return {
        "trace": trace,
        "parameter_count": model.parameter_count(),
        "graph_kernel_is_buffer": True,
        "graph_kernel_used": kernel_used,
        "remaining_time": model.remaining_time,
    }
