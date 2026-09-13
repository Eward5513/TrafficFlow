"""Runtime checks for R-only STSGCN. Failures abort training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from reimplementation.common.data.r_only_npz_dataset import load_json
from reimplementation.common.errors import ReimplementationError
from reimplementation.stgcn.validation import load_edge_ids, validate_prepared_data
from reimplementation.stsgcn.graph import load_stsgcn_graph
from reimplementation.stsgcn.model.stsgcn import STSGCN


def load_node_ids(node_mapping: Path) -> list[str]:
    return load_edge_ids(node_mapping)


def validate_stsgcn_graph(
    *,
    topology_path: Path,
    metadata_path: Path,
    validation_path: Path,
    r_nodes_path: Path,
    node_ids: list[str],
    expected_nodes: int,
) -> dict[str, Any]:
    return load_stsgcn_graph(
        topology_path=topology_path,
        metadata_path=metadata_path,
        validation_path=validation_path,
        r_nodes_path=r_nodes_path,
        node_ids=node_ids,
        expected_nodes=expected_nodes,
    )


def validate_model_shapes(model: STSGCN, *, batch_size: int, seq_len: int) -> dict[str, Any]:
    model.eval()
    x = torch.zeros(batch_size, seq_len, model.num_nodes, model.input_channels)
    with torch.no_grad():
        y, trace = model(x, return_trace=True)
    expected = (batch_size, model.horizon, model.num_nodes, 1)
    if tuple(y.shape) != expected:
        raise ReimplementationError(f"model output {tuple(y.shape)} != {expected}")
    if int(trace["final_time_length"]) != int(model.final_time_length):
        raise ReimplementationError(
            f"final time length {trace['final_time_length']} != {model.final_time_length}"
        )
    expected_times = [seq_len]
    current = seq_len
    for _ in range(model.n_stsgcl):
        current -= 2
        expected_times.append(current)
    if trace["time_lengths"] != expected_times:
        raise ReimplementationError(
            f"STSGCL time lengths {trace['time_lengths']} != {expected_times}"
        )
    buffers = dict(model.named_buffers())
    params = dict(model.named_parameters())
    if "localized_adj" not in buffers:
        raise ReimplementationError("localized_adj is not a buffer")
    if "localized_adj" in params:
        raise ReimplementationError("localized_adj must not be a parameter")
    if model.use_mask and "adj_mask" not in params:
        raise ReimplementationError("adj_mask parameter is missing")
    if model.input_projection is None:
        raise ReimplementationError("official config uses first_layer_embedding_size=64")
    devices = {str(param.device) for param in model.parameters()}
    devices.update(str(buf.device) for buf in model.buffers())
    if len(devices) != 1:
        raise ReimplementationError(f"model tensors span multiple devices: {devices}")
    return {
        "trace": trace,
        "parameter_count": model.parameter_count(),
        "horizon": model.horizon,
        "seq_len": seq_len,
        "final_time_length": model.final_time_length,
        "n_stsgcl": model.n_stsgcl,
        "device": next(iter(devices)),
    }


def validate_train_npz_only(
    data_root: Path,
    *,
    rate: str,
    n_his: int,
    num_nodes: int,
    expected_count: int,
) -> dict[str, Any]:
    """Read-only checks on one training NPZ. Does not open validation/test."""
    from reimplementation.common.data.r_only_npz_dataset import load_npz, invert_target, read_target_scaler
    from reimplementation.common.utils.hashing import sha256_file
    import numpy as np

    path = Path(data_root) / rate / "train.npz"
    arrays = load_npz(path)
    n_samples = int(arrays["x"].shape[0])
    if n_samples != int(expected_count):
        raise ReimplementationError(f"{path} has {n_samples} samples, manifest {expected_count}")
    if tuple(arrays["x"].shape) != (n_samples, n_his, num_nodes, 1):
        raise ReimplementationError(f"{path} x shape {arrays['x'].shape}")
    if tuple(arrays["y"].shape) != (n_samples, 1, num_nodes, 1):
        raise ReimplementationError(f"{path} y shape {arrays['y'].shape}")
    if not np.array_equal(arrays["target_slot"], arrays["window_start_slot"] + (n_his - 1)):
        raise ReimplementationError(f"{path} target_slot != window_start_slot + {n_his - 1}")
    normalization = load_json(Path(data_root) / "normalization.json")
    mean_y, std_y = read_target_scaler(normalization)
    recovered = invert_target(arrays["y"], mean_y, std_y)
    denorm_error = float(np.max(np.abs(recovered - arrays["y_raw"])))
    if denorm_error > 1e-3:
        raise ReimplementationError("y * std + mean does not recover y_raw")
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "n_samples": n_samples,
        "x_shape": list(arrays["x"].shape),
        "y_shape": list(arrays["y"].shape),
        "denorm_max_abs_error": denorm_error,
        "target_mode": "last-observed-step",
        "did_not_read_validation_or_test": True,
    }


__all__ = [
    "load_node_ids",
    "validate_prepared_data",
    "validate_stsgcn_graph",
    "validate_model_shapes",
    "validate_train_npz_only",
]
