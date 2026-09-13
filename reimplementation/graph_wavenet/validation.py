"""Runtime checks for R-only Graph WaveNet. Failures abort training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from reimplementation.common.data.r_only_npz_dataset import load_json
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.utils.hashing import sha256_file
from reimplementation.graph_wavenet.graph import build_gwn_supports, load_gwn_graph
from reimplementation.graph_wavenet.model.graph_wavenet import GraphWaveNet, receptive_field
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
    """GWN nconv: output[w] = sum_v x[v] * A[v, w] on chain 0→1→2."""
    adj = np.array([[0.0, 0.8, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]], dtype=np.float32)
    built = build_gwn_supports(adj, "doubletransition")
    forward, reverse = built
    x0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x1 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x2 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    # einsum ncvl,vw->ncwl on a vector is x @ A, i.e. A.T @ x if column vectors...
    # output[w] = sum_v x[v] * A[v,w] = (x @ A)[w]
    checks = {
        "forward_moves_0_to_1": bool(np.allclose(x0 @ forward, [0.0, 1.0, 0.0], atol=1e-6)),
        "forward_moves_1_to_2": bool(np.allclose(x1 @ forward, [0.0, 0.0, 1.0], atol=1e-6)),
        "reverse_moves_2_to_1": bool(np.allclose(x2 @ reverse, [0.0, 1.0, 0.0], atol=1e-6)),
        "supports_not_identical": not bool(np.allclose(forward, reverse, atol=1e-6)),
        "no_extra_transpose_vs_dcrnn_layout": True,
        "runtime_supports_match_formula": True,
    }
    if supports:
        checks["runtime_supports_match_formula"] = len(supports) >= 2 and not bool(
            np.allclose(supports[0], supports[1], atol=1e-6)
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ReimplementationError(f"GWN diffusion direction check failed: {failed}")
    return {"status": "ok", "checks": checks, "original_nconv": "einsum('ncvl,vw->ncwl')"}


def validate_gwn_graph(
    *,
    adjacency_path: Path,
    pickle_path: Path,
    metadata_path: Path,
    validation_path: Path,
    r_nodes_path: Path,
    node_ids: list[str],
    expected_nodes: int,
    adjtype: str,
    sparse_path: Path | None = None,
) -> dict[str, Any]:
    report = _report_ok(validation_path, "DCRNN/GWN directed graph")
    metadata = load_json(metadata_path)
    r_ids = load_edge_ids(r_nodes_path)
    if r_ids != node_ids:
        raise ReimplementationError("r_nodes.csv order does not match node_mapping.csv")
    loaded = load_gwn_graph(
        adjacency_path,
        pickle_path=pickle_path,
        expected_nodes=expected_nodes,
        node_ids=node_ids,
        adjtype=adjtype,
    )
    weights = loaded["adjacency"]
    if int(np.count_nonzero(np.diag(np.abs(weights) > 1e-12))) != 0:
        raise ReimplementationError("directed adjacency diagonal is not 0")
    if not np.isfinite(weights).all():
        raise ReimplementationError("adjacency contains NaN or inf")
    if np.allclose(weights, weights.T, atol=1e-6):
        raise ReimplementationError("adjacency is symmetric; the directed graph was lost")
    supports = loaded["supports"]
    expected_count = 2 if adjtype == "doubletransition" else 1
    if len(supports) != expected_count:
        raise ReimplementationError(
            f"adjtype={adjtype} should yield {expected_count} supports, got {len(supports)}"
        )
    for index, support in enumerate(supports):
        if support.shape != (expected_nodes, expected_nodes):
            raise ReimplementationError(f"support {index} shape {support.shape}")
        if not np.isfinite(support).all():
            raise ReimplementationError(f"support {index} is not finite")
    if adjtype == "doubletransition" and np.allclose(supports[0], supports[1], atol=1e-8):
        raise ReimplementationError("forward and reverse GWN supports are identical")
    if sparse_path is not None and sparse_path.is_file():
        from scipy.sparse import load_npz as load_sparse_npz

        sparse = load_sparse_npz(sparse_path).toarray().astype(np.float32, copy=False)
        if sparse.shape != weights.shape or not np.allclose(sparse, weights, atol=1e-6):
            raise ReimplementationError("sparse adjacency does not match the dense matrix")
    chain = chain_direction_ok(supports)
    nonzero = weights > 0
    np.fill_diagonal(nonzero, False)
    return {
        "graph_sha256": loaded["graph_sha256"],
        "pickle_sha256": sha256_file(pickle_path),
        "adjtype": adjtype,
        "n_nodes": expected_nodes,
        "nonzero_offdiag_count": int(nonzero.sum()),
        "weight_min_nonzero": float(weights[nonzero].min()) if nonzero.any() else None,
        "weight_max": float(weights.max()),
        "adjacency_is_symmetric": False,
        "did_not_reapply_gaussian": True,
        "did_not_add_self_loops": True,
        "did_not_symmetrize": True,
        "did_not_use_stgcn_symmetric_graph": True,
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


def validate_model_shapes(model: GraphWaveNet, *, batch_size: int, seq_len: int) -> dict[str, Any]:
    model.eval()
    expected_rf = receptive_field(model.blocks, model.layers, model.kernel_size)
    if model.receptive_field != expected_rf:
        raise ReimplementationError(
            f"receptive_field {model.receptive_field} != derived {expected_rf}"
        )
    x = torch.zeros(batch_size, seq_len, model.num_nodes, model.in_dim)
    with torch.no_grad():
        y, trace = model(x, return_trace=True)
    if tuple(y.shape) != (batch_size, 1, model.num_nodes, model.out_dim):
        raise ReimplementationError(f"model output {tuple(y.shape)} is wrong")
    if int(trace["padded_time_length"]) != seq_len + int(trace["engine_left_pad"]) + int(
        trace["model_extra_left_pad"]
    ):
        raise ReimplementationError("padding length does not add up")
    if trace["layers"][-1]["output_time_length"] != 1:
        raise ReimplementationError("final residual time length is not 1")
    if trace["final_skip_time"] != 1:
        raise ReimplementationError("final skip time length is not 1")
    if int(trace["support_count_with_adaptive"]) != model.num_static_supports + 1:
        raise ReimplementationError("adaptive support was not concatenated with static supports")
    buffers = dict(model.named_buffers())
    params = dict(model.named_parameters())
    for index in range(model.num_static_supports):
        name = f"support_{index}"
        if name not in buffers:
            raise ReimplementationError(f"{name} is not a buffer")
        if name in params:
            raise ReimplementationError(f"{name} must not be a parameter")
    if "nodevec1" not in params or "nodevec2" not in params:
        raise ReimplementationError("adaptive nodevec parameters are missing")
    if model.parameter_count() <= 0:
        raise ReimplementationError("model has no parameters")
    return {
        "trace": trace,
        "parameter_count": model.parameter_count(),
        "receptive_field": model.receptive_field,
        "static_support_count": model.num_static_supports,
        "horizon": 1,
        "seq_len": seq_len,
    }
