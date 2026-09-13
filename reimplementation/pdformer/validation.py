"""Runtime checks for R-only PDFormer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from reimplementation.common.data.r_only_npz_dataset import load_json
from reimplementation.common.errors import ReimplementationError
from reimplementation.stgcn.validation import load_edge_ids, validate_prepared_data
from reimplementation.pdformer.graph import load_binary_topology
from reimplementation.pdformer.model.pdformer import PDFormer


def load_node_ids(node_mapping: Path) -> list[str]:
    return load_edge_ids(node_mapping)


def validate_pdformer_graph(
    *,
    topology_path: Path,
    r_nodes_path: Path,
    node_ids: list[str],
    expected_nodes: int,
) -> dict[str, Any]:
    topology = load_binary_topology(topology_path, expected_nodes=expected_nodes)
    r_ids = load_edge_ids(r_nodes_path)
    if r_ids != node_ids:
        raise ReimplementationError("r_nodes.csv edge_id order does not match node_mapping.csv")
    if int(topology.shape[0]) != len(node_ids):
        raise ReimplementationError("topology node count does not match node_mapping.csv")
    return {
        "topology": topology,
        "node_ids": node_ids,
        "num_nodes": int(topology.shape[0]),
        "directed": False,
        "source": str(topology_path),
        "used_stgcn_weighted_adjacency_as_hop": False,
    }


def validate_model_shapes(model: PDFormer, *, batch_size: int, seq_len: int) -> dict[str, Any]:
    model.eval()
    channels = model.enc_embed_layer.expected_input_channels()
    x = torch.zeros(batch_size, seq_len, model.num_nodes, channels)
    if model.add_time_in_day:
        slots = torch.arange(seq_len).float() / 288.0
        x[..., 1] = slots.view(1, seq_len, 1)
    with torch.no_grad():
        y, trace = model(x, return_trace=True)
    expected = (batch_size, model.output_window, model.num_nodes, model.output_dim)
    if tuple(y.shape) != expected:
        raise ReimplementationError(f"model output {tuple(y.shape)} != {expected}")
    buffers = dict(model.named_buffers())
    for name in ("geo_mask", "sem_mask", "laplacian_pe", "pattern_keys"):
        if name not in buffers:
            raise ReimplementationError(f"{name} is not a buffer")
        if name in dict(model.named_parameters()):
            raise ReimplementationError(f"{name} must not be a parameter")
    devices = {str(param.device) for param in model.parameters()}
    devices.update(str(buf.device) for buf in model.buffers())
    if len(devices) != 1:
        raise ReimplementationError(f"model tensors span multiple devices: {devices}")
    return {
        "input_shape": list(x.shape),
        "output_shape": list(y.shape),
        "trace": trace,
        "parameter_count": model.parameter_count(),
        "buffer_names": sorted(buffers),
        "devices": sorted(devices),
    }


def validate_train_npz_only(path: Path) -> Path:
    if path.name in {"validation.npz", "test.npz"}:
        raise ReimplementationError(f"refusing to fit or smoke on {path.name}")
    if path.name != "train.npz":
        raise ReimplementationError(f"expected train.npz, got {path}")
    return path


__all__ = [
    "load_node_ids",
    "validate_model_shapes",
    "validate_pdformer_graph",
    "validate_prepared_data",
    "validate_train_npz_only",
]
