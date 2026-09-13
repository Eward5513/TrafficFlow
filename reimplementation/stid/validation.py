"""Runtime checks for R-only STID. STID does not load a road graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from reimplementation.common.errors import ReimplementationError
from reimplementation.stgcn.validation import load_edge_ids, validate_prepared_data
from reimplementation.stid.model.stid import STID, graph_keys_in_state_dict


def load_node_ids(node_mapping: Path) -> list[str]:
    return load_edge_ids(node_mapping)


def validate_node_order(*, r_nodes_path: Path, node_mapping: Path, expected_nodes: int) -> list[str]:
    node_ids = load_node_ids(node_mapping)
    r_ids = load_edge_ids(r_nodes_path)
    if r_ids != node_ids:
        raise ReimplementationError("r_nodes.csv edge_id order does not match node_mapping.csv")
    if len(node_ids) != int(expected_nodes):
        raise ReimplementationError(
            f"node count {len(node_ids)} != expected {expected_nodes}; "
            "refusing to reorder by edge_id"
        )
    return node_ids


def validate_no_graph_config(config: Mapping[str, Any]) -> None:
    forbidden = (
        "adjacency",
        "adj",
        "supports",
        "graph",
        "laplacian",
        "cheb_ks",
        "undirected_topology",
        "weighted_adjacency",
        "dtw_path",
    )
    present = [key for key in forbidden if config.get(key) not in (None, "", False)]
    if present:
        raise ReimplementationError(
            f"STID config must not supply graph inputs {present}; "
            "r_nodes.csv is identity order only"
        )


def validate_model_shapes(
    model: STID,
    *,
    batch_size: int,
    history: torch.Tensor,
    time_of_day_index: torch.Tensor | None = None,
    day_of_week_index: torch.Tensor | None = None,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        prediction, trace = model(
            history,
            time_of_day_index=time_of_day_index,
            day_of_week_index=day_of_week_index,
            return_trace=True,
        )
    expected = (int(batch_size), model.output_len, model.num_nodes, 1)
    if tuple(prediction.shape) != expected:
        raise ReimplementationError(f"model output {tuple(prediction.shape)} != {expected}")
    graph_keys = graph_keys_in_state_dict(model.state_dict())
    if graph_keys:
        raise ReimplementationError(f"state_dict has graph-like keys {graph_keys}")
    devices = {str(param.device) for param in model.parameters()}
    if len(devices) != 1:
        raise ReimplementationError(f"model tensors span multiple devices: {devices}")
    if model.parameter_count() != model.expected_parameter_count():
        raise ReimplementationError(
            f"parameter_count {model.parameter_count()} != derived {model.expected_parameter_count()}"
        )
    return {
        "input_shape": list(history.shape),
        "output_shape": list(prediction.shape),
        "trace": trace,
        "parameter_count": model.parameter_count(),
        "devices": sorted(devices),
        "graph_keys": graph_keys,
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
    "validate_no_graph_config",
    "validate_node_order",
    "validate_prepared_data",
    "validate_train_npz_only",
]
