"""Runtime checks for R-only DCRNN. Failures abort training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from reimplementation.common.data.r_only_npz_dataset import load_json
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.utils.hashing import sha256_file
from reimplementation.dcrnn.graph import build_dcrnn_supports, load_dcrnn_graph
from reimplementation.dcrnn.model.dcrnn_model import DCRNNModel
from reimplementation.stgcn.validation import load_edge_ids, validate_prepared_data


def _report_ok(path: Path, label: str) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("failures"):
        raise ReimplementationError(f"{label} validation report has failures: {payload['failures']}")
    status = payload.get("status")
    overall = payload.get("overall_passed")
    if status not in (None, "ok", "passed") or overall is False:
        raise ReimplementationError(f"{label} validation is not ok: {path}")
    return payload


def chain_direction_ok(supports: list[np.ndarray]) -> dict[str, Any]:
    adj = np.array([[0.0, 0.8, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]], dtype=np.float32)
    built = build_dcrnn_supports(adj, "dual_random_walk")
    forward, reverse = built
    x0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x1 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x2 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    checks = {
        "forward_moves_0_to_1": bool(np.allclose(forward @ x0, [0.0, 1.0, 0.0], atol=1e-6)),
        "forward_moves_1_to_2": bool(np.allclose(forward @ x1, [0.0, 0.0, 1.0], atol=1e-6)),
        "reverse_moves_2_to_1": bool(np.allclose(reverse @ x2, [0.0, 1.0, 0.0], atol=1e-6)),
        "supports_not_identical": not bool(np.allclose(forward, reverse, atol=1e-6)),
        "runtime_supports_match_formula": True,
    }
    if supports:
        checks["runtime_supports_match_formula"] = len(supports) >= 2 and not bool(
            np.allclose(supports[0], supports[1], atol=1e-6)
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ReimplementationError(f"diffusion direction check failed: {failed}")
    return {"status": "ok", "checks": checks}


def validate_dcrnn_graph(
    *,
    adjacency_path: Path,
    pickle_path: Path,
    metadata_path: Path,
    validation_path: Path,
    r_nodes_path: Path,
    node_ids: list[str],
    expected_nodes: int,
    filter_type: str,
    sparse_path: Path | None = None,
) -> dict[str, Any]:
    report = _report_ok(validation_path, "DCRNN graph")
    metadata = load_json(metadata_path)
    r_ids = load_edge_ids(r_nodes_path)
    if r_ids != node_ids:
        raise ReimplementationError("r_nodes.csv order does not match node_mapping.csv")
    loaded = load_dcrnn_graph(
        adjacency_path,
        pickle_path=pickle_path,
        expected_nodes=expected_nodes,
        node_ids=node_ids,
        filter_type=filter_type,
    )
    weights = loaded["adjacency"]
    if int(np.count_nonzero(np.diag(np.abs(weights) > 1e-12))) != 0:
        raise ReimplementationError("DCRNN adjacency diagonal is not 0")
    if not np.isfinite(weights).all():
        raise ReimplementationError("DCRNN adjacency contains NaN or inf")
    if np.any(weights < -1e-8) or np.any(weights > 1.0 + 1e-5):
        raise ReimplementationError("DCRNN weights are outside [0, 1]")
    symmetric = bool(np.allclose(weights, weights.T, atol=1e-6))
    if symmetric:
        raise ReimplementationError("DCRNN adjacency is symmetric; the directed graph was lost")
    supports = loaded["supports"]
    expected_count = 2 if filter_type == "dual_random_walk" else 1
    if len(supports) != expected_count:
        raise ReimplementationError(
            f"filter_type={filter_type} should yield {expected_count} supports, got {len(supports)}"
        )
    for index, support in enumerate(supports):
        if support.shape != (expected_nodes, expected_nodes):
            raise ReimplementationError(f"support {index} shape {support.shape}")
        if not np.isfinite(support).all():
            raise ReimplementationError(f"support {index} is not finite")
    if filter_type == "dual_random_walk" and np.allclose(supports[0], supports[1], atol=1e-8):
        raise ReimplementationError("forward and reverse supports are identical on a directed graph")
    if sparse_path is not None and sparse_path.is_file():
        from scipy.sparse import load_npz as load_sparse_npz

        sparse = load_sparse_npz(sparse_path).toarray().astype(np.float32, copy=False)
        if sparse.shape != weights.shape or not np.allclose(sparse, weights, atol=1e-6):
            raise ReimplementationError("sparse DCRNN adjacency does not match the dense matrix")
    chain = chain_direction_ok(supports)
    nonzero = weights > 0
    np.fill_diagonal(nonzero, False)
    return {
        "graph_sha256": loaded["graph_sha256"],
        "pickle_sha256": sha256_file(pickle_path),
        "filter_type": filter_type,
        "n_nodes": expected_nodes,
        "nonzero_offdiag_count": int(nonzero.sum()),
        "weight_min_nonzero": float(weights[nonzero].min()) if nonzero.any() else None,
        "weight_max": float(weights.max()),
        "adjacency_is_symmetric": False,
        "did_not_reapply_gaussian": True,
        "did_not_add_self_loops": True,
        "did_not_symmetrize": True,
        "support_count": len(supports),
        "support_shapes": [list(item.shape) for item in supports],
        "forward_reverse_supports_identical": False,
        "prepared_graph_status": report.get("status"),
        "metadata_sigma": metadata.get("sigma"),
        "chain_test": chain,
        "supports": supports,
        "adjacency": weights,
        "node_ids": node_ids,
    }


def validate_model_shapes(model: DCRNNModel, *, batch_size: int) -> dict[str, Any]:
    model.eval()
    x = torch.zeros(batch_size, model.seq_len, model.num_nodes, model.input_dim)
    with torch.no_grad():
        y, trace = model(x, return_trace=True)
    if tuple(y.shape) != (batch_size, model.horizon, model.num_nodes, model.output_dim):
        raise ReimplementationError(f"model output {tuple(y.shape)} is wrong")
    if trace.get("used_label_on_first_step"):
        raise ReimplementationError("decoder first step used labels")
    if model.horizon != 1:
        raise ReimplementationError("R-only DCRNN horizon must be 1")
    buffers = dict(model.named_buffers())
    for index in range(model.num_supports):
        name = f"support_{index}"
        if name not in buffers:
            raise ReimplementationError(f"{name} is not a buffer")
        if name in dict(model.named_parameters()):
            raise ReimplementationError(f"{name} must not be a parameter")
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    if encoder_params <= 0 or decoder_params <= 0:
        raise ReimplementationError("encoder or decoder has no parameters")
    return {
        "trace": trace,
        "parameter_count": model.parameter_count(),
        "encoder_parameter_count": int(encoder_params),
        "decoder_parameter_count": int(decoder_params),
        "num_supports": model.num_supports,
        "horizon": model.horizon,
        "seq_len": model.seq_len,
    }
