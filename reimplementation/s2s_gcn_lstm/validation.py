"""Runtime checks for the 12-to-1 Seq2seq GCN-LSTM port."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from reimplementation.common.errors import ReimplementationError
from reimplementation.s2s_gcn_lstm.graph import load_s2s_graph
from reimplementation.s2s_gcn_lstm.model.s2s_gcn_lstm import S2SGCNLSTM
from reimplementation.stgcn.validation import load_edge_ids, validate_prepared_data


def validate_s2s_graph(config: dict[str, Any], *, project_root: Path) -> dict[str, Any]:
    from reimplementation.stgcn.engine import resolve_path

    distance_value = config.get("distance_matrix")
    distance_path = None if not distance_value else resolve_path(project_root, distance_value)
    return load_s2s_graph(
        topology_path=resolve_path(project_root, config["spatial_adjacency"]),
        r_graph_path=resolve_path(project_root, config["r_graph"]),
        r_nodes_path=resolve_path(project_root, config["r_nodes"]),
        node_mapping_path=resolve_path(project_root, config["node_mapping"]),
        expected_nodes=int(config["num_nodes"]),
        edge_weight=str(config.get("edge_weight", "binary_connectivity")),
        distance_path=distance_path,
    )


def validate_model_shapes(model: S2SGCNLSTM, *, batch_size: int = 2) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    features = torch.zeros(
        batch_size,
        model.input_window,
        model.num_nodes,
        model.input_channels,
        device=device,
    )
    with torch.no_grad():
        prediction, trace = model(features, return_trace=True)
    expected = (batch_size, model.output_window, model.num_nodes, model.output_channels)
    if tuple(prediction.shape) != expected:
        raise ReimplementationError(f"model output {tuple(prediction.shape)} != {expected}")
    if int(trace["encoder_gcn_steps"]) != model.input_window:
        raise ReimplementationError("encoder did not cover every input step")
    if int(trace["decoder_lstm_steps"]) != model.output_window:
        raise ReimplementationError("decoder step count does not match output_window")
    if int(trace["decoder_gcn_calls"]) != 1:
        raise ReimplementationError("decoder GCN must run once on the fixed encoder context")
    if trace["decoder_initial_state"] != "zeros" or trace["prediction_feedback"] or trace["output_activation"] != "linear":
        raise ReimplementationError(f"unexpected decoder trace: {trace}")
    buffers = dict(model.named_buffers())
    params = dict(model.named_parameters())
    if "a_hat" not in buffers:
        raise ReimplementationError("Â is not a buffer")
    if "a_hat" in params:
        raise ReimplementationError("Â must not be trainable")
    if not torch.isfinite(model.a_hat).all():
        raise ReimplementationError("Â contains NaN or inf")
    if model.parameter_count() <= 0:
        raise ReimplementationError("model has no trainable parameters")
    encoder_ids = {param.data_ptr() for param in model.encoder_gcn.parameters()}
    encoder_ids.update(param.data_ptr() for param in model.encoder_lstm.parameters())
    decoder_ids = {param.data_ptr() for param in model.decoder_gcn.parameters()}
    decoder_ids.update(param.data_ptr() for param in model.decoder_lstm.parameters())
    decoder_ids.update(param.data_ptr() for param in model.fc.parameters())
    if encoder_ids & decoder_ids:
        raise ReimplementationError("encoder and decoder share parameter storage")
    return {
        "trace": trace,
        "parameter_count": model.parameter_count(),
        "output_shape": list(prediction.shape),
    }


def assert_feature_order(node_ids: list[str], graph_ids: list[str]) -> None:
    if node_ids != graph_ids:
        raise ReimplementationError("NPZ node_mapping order does not match the adjacency node order")


def assert_task_config(config: dict[str, Any]) -> None:
    if str(config.get("target_mode")) != "last-observed-step":
        raise ReimplementationError("target_mode must be last-observed-step")
    if int(config.get("n_his", config.get("input_window", 0))) != 12:
        raise ReimplementationError("R-only input window must be 12")
    if int(config.get("output_steps", config.get("output_window", 0))) != 1:
        raise ReimplementationError("R-only output window must be 1; this is reconstruction, not multi-step forecast")
    in_channels = int(config.get("input_channels", 0))
    out_channels = int(config.get("output_channels", 0))
    if in_channels not in {1, 4} or out_channels not in {1, 4}:
        raise ReimplementationError("input_channels and output_channels must be 1 or 4")
    if (in_channels == 4) != (config.get("input_type") == "all"):
        raise ReimplementationError("four input channels require --input-type all")
    if (out_channels == 4) != (config.get("target_type") == "all"):
        raise ReimplementationError("four output channels require --target-type all")
    if str(config.get("output_activation", "linear")) != "linear":
        raise ReimplementationError("targets are global z-scores, not [0, 1]; refusing a bounded output activation")
    if str(config.get("edge_weight", "binary_connectivity")) not in {"binary_connectivity", "distance", "gaussian"}:
        raise ReimplementationError("edge_weight must be binary_connectivity, distance, or gaussian")
    if int(config.get("epochs", 1)) < 1:
        raise ReimplementationError("epochs must be positive")
    if not np.isfinite(float(config.get("learning_rate", 0.0))) or float(config["learning_rate"]) <= 0.0:
        raise ReimplementationError("learning_rate must be positive")


__all__ = [
    "assert_feature_order",
    "assert_task_config",
    "load_edge_ids",
    "validate_model_shapes",
    "validate_prepared_data",
    "validate_s2s_graph",
]
