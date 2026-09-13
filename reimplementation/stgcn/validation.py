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
    data_root = Path(data_root)
    manifest = load_json(data_root / "split_manifest.json")
    normalization = load_json(data_root / "normalization.json")
    dataset_meta = load_json(data_root / "dataset_metadata.json")
    counts = expected_split_counts(manifest)
    mean_y, std_y = read_target_scaler(normalization)
    if bool(normalization.get("used_test_to_fit")) or bool(normalization.get("used_validation_to_fit")):
        raise ReimplementationError("normalization.json was not fit on training only")
    node_ids = load_edge_ids(data_root / "node_mapping.csv")
    if len(node_ids) != num_nodes:
        raise ReimplementationError(f"node_mapping.csv has {len(node_ids)} nodes, expected {num_nodes}")
    target_mode = str(manifest.get("target_mode") or dataset_meta.get("target_mode") or "")
    if target_mode != "last-observed-step":
        raise ReimplementationError(
            f"split_manifest target_mode={target_mode!r} is not last-observed-step"
        )
    if int(manifest.get("n_his", n_his)) != n_his:
        raise ReimplementationError(f"manifest n_his {manifest.get('n_his')} != {n_his}")
    data_report = load_json(data_root / "validation_summary.json")
    if data_report.get("failures"):
        raise ReimplementationError(f"data validation report has failures: {data_report['failures']}")
    if data_report.get("overall_validation_passed") is False:
        raise ReimplementationError("data validation report overall_validation_passed is false")
    if str(data_report.get("status", "ok")) not in {"ok", "passed"}:
        raise ReimplementationError(f"data validation report status={data_report.get('status')!r}")

    reference_y: dict[str, np.ndarray] = {}
    reference_index: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    file_hashes: dict[str, str] = {}
    denorm_errors: dict[str, float] = {}
    for tag in rates:
        tag = rate_tag(tag)
        if tag not in RATE_TAGS:
            raise ReimplementationError(f"unsupported rate {tag}")
        for split in SPLITS:
            path = data_root / tag / f"{split}.npz"
            arrays = load_npz(path)
            file_hashes[f"{tag}/{split}.npz"] = sha256_file(path)
            n_samples = int(arrays["x"].shape[0])
            if n_samples != counts[split]:
                raise ReimplementationError(f"{path} has {n_samples} samples, manifest {counts[split]}")
            if tuple(arrays["x"].shape) != (n_samples, n_his, num_nodes, 1):
                raise ReimplementationError(f"{path} x shape {arrays['x'].shape}")
            if tuple(arrays["y"].shape) != (n_samples, 1, num_nodes, 1):
                raise ReimplementationError(f"{path} y shape {arrays['y'].shape}")
            for name in ("x", "y", "x_raw", "y_raw"):
                if not np.isfinite(arrays[name]).all():
                    raise ReimplementationError(f"{path} {name} has NaN or inf")
            if not np.array_equal(arrays["target_slot"], arrays["window_start_slot"] + (n_his - 1)):
                raise ReimplementationError(f"{path} target_slot != window_start_slot + {n_his - 1}")
            recovered = invert_target(arrays["y"], mean_y, std_y)
            denorm_errors[f"{tag}/{split}"] = float(np.max(np.abs(recovered - arrays["y_raw"])))
            if denorm_errors[f"{tag}/{split}"] > 1e-3:
                raise ReimplementationError(
                    f"{path} denormalization with mean_y_full/std_y_full does not recover y_raw"
                )
            key = split
            if key not in reference_y:
                reference_y[key] = arrays["y"]
                reference_index[key] = (
                    arrays["day_index"],
                    arrays["window_start_slot"],
                    arrays["target_slot"],
                )
            else:
                if not np.allclose(reference_y[key], arrays["y"], atol=1e-6):
                    raise ReimplementationError(f"{path} y does not match the first rate")
                if not np.allclose(arrays["y_raw"], invert_target(reference_y[key], mean_y, std_y), atol=1e-3):
                    raise ReimplementationError(f"{path} y_raw is not the shared full-flow target")
                day, start, target = reference_index[key]
                if not (
                    np.array_equal(day, arrays["day_index"])
                    and np.array_equal(start, arrays["window_start_slot"])
                    and np.array_equal(target, arrays["target_slot"])
                ):
                    raise ReimplementationError(f"{path} sample index does not match the first rate")
    train_days = set(map(str, manifest.get("training_days") or dataset_meta.get("training_days") or []))
    val_days = set(map(str, manifest.get("validation_days") or dataset_meta.get("validation_days") or []))
    test_days = set(map(str, manifest.get("test_days") or dataset_meta.get("test_days") or []))
    if train_days and val_days and train_days & val_days:
        raise ReimplementationError("training and validation days overlap")
    if train_days and test_days and train_days & test_days:
        raise ReimplementationError("training and test days overlap")
    if val_days and test_days and val_days & test_days:
        raise ReimplementationError("validation and test days overlap")
    split_keys = {
        split: set(zip((int(item) for item in values[0]), (int(item) for item in values[1])))
        for split, values in reference_index.items()
    }
    if split_keys.get("train") and split_keys.get("validation") and split_keys["train"] & split_keys["validation"]:
        raise ReimplementationError("train and validation sample keys overlap")
    if split_keys.get("train") and split_keys.get("test") and split_keys["train"] & split_keys["test"]:
        raise ReimplementationError("train and test sample keys overlap")
    if split_keys.get("validation") and split_keys.get("test") and split_keys["validation"] & split_keys["test"]:
        raise ReimplementationError("validation and test sample keys overlap")
    return {
        "sample_counts": counts,
        "mean_y_full": mean_y,
        "std_y_full": std_y,
        "node_ids": node_ids,
        "file_hashes": file_hashes,
        "normalization_sha256": sha256_file(data_root / "normalization.json"),
        "split_manifest_sha256": sha256_file(data_root / "split_manifest.json"),
        "denorm_max_abs_error": denorm_errors,
        "target_mode": "last-observed-step",
        "did_not_refit_scaler": True,
        "prepared_data_report_status": data_report.get("status"),
        "split_sample_keys_disjoint": True,
    }


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
