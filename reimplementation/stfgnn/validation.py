"""Runtime checks for R-only STFGNN. Failures abort training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from reimplementation.common.data.r_only_npz_dataset import load_json
from reimplementation.common.errors import ReimplementationError
from reimplementation.stgcn.validation import load_edge_ids, validate_prepared_data
from reimplementation.stfgnn.graph import load_stfgnn_graph
from reimplementation.stfgnn.model.stfgnn import STFGNN


def load_node_ids(node_mapping: Path) -> list[str]:
    return load_edge_ids(node_mapping)


def validate_stfgnn_graph(
    *,
    topology_path: Path,
    metadata_path: Path,
    validation_path: Path,
    r_nodes_path: Path,
    node_ids: list[str],
    expected_nodes: int,
    temporal_adj: Any = None,
    temporal_path: Path | None = None,
    temporal_is_test_only: bool = False,
) -> dict[str, Any]:
    return load_stfgnn_graph(
        topology_path=topology_path,
        metadata_path=metadata_path,
        validation_path=validation_path,
        r_nodes_path=r_nodes_path,
        node_ids=node_ids,
        expected_nodes=expected_nodes,
        temporal_adj=temporal_adj,
        temporal_path=temporal_path,
        temporal_is_test_only=temporal_is_test_only,
    )


def validate_model_shapes(model: STFGNN, *, batch_size: int, seq_len: int) -> dict[str, Any]:
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
    for _ in range(model.n_stfgcl):
        current -= 3
        expected_times.append(current)
    if trace["time_lengths"] != expected_times:
        raise ReimplementationError(
            f"STFGCL time lengths {trace['time_lengths']} != {expected_times}"
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
        "n_stfgcl": model.n_stfgcl,
        "device": next(iter(devices)),
        "temporal_graph_is_test_only": model.temporal_graph_is_test_only,
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
    from reimplementation.common.data.r_only_npz_dataset import invert_target, load_npz, read_target_scaler
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


def model_structure_payload(model: STFGNN) -> dict[str, Any]:
    groups: dict[str, int] = {}
    for name, param in model.named_parameters():
        prefix = name.split(".")[0]
        groups[prefix] = groups.get(prefix, 0) + int(param.numel())
    return {
        "code_version": "0.1.0",
        "num_nodes": model.num_nodes,
        "seq_len": model.seq_len,
        "horizon": model.horizon,
        "n_stfgcl": model.n_stfgcl,
        "filters": model.filter_list,
        "first_layer_embedding_size": model.first_layer_embedding_size,
        "activation": model.activation,
        "module_type": model.module_type,
        "use_mask": model.use_mask,
        "temporal_emb": model.temporal_emb,
        "spatial_emb": model.spatial_emb,
        "final_time_length": model.final_time_length,
        "time_length_rule": "each STFGCL shortens T by 3 (4-step window, stride 1, gated CNN pad=0)",
        "expected_time_lengths": [model.seq_len - 3 * index for index in range(model.n_stfgcl + 1)],
        "localized_adj_shape": list(model.localized_adj.shape),
        "parameter_count": model.parameter_count(),
        "parameter_groups": groups,
        "temporal_graph_is_test_only": model.temporal_graph_is_test_only,
        "stfgcm_crop": "[N:2N] second timestep of each 4-step window",
        "stfgcm_aggregation": "element-wise max over GCN layers",
        "gated_cnn": {
            "kernel": [1, 2],
            "dilation": [1, 3],
            "padding": 0,
            "formula": "sigmoid(Conv_left) * tanh(Conv_right)",
            "channels": "same as layer input C",
        },
        "fusion": "graph_branch + gated_cnn_residual",
        "did_not_use_pyg_or_dgl": True,
    }


__all__ = [
    "load_node_ids",
    "validate_prepared_data",
    "validate_stfgnn_graph",
    "validate_model_shapes",
    "validate_train_npz_only",
    "model_structure_payload",
]
