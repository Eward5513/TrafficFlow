"""Runtime checks for Xuancheng26 TLAST. The model does not read a road graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from reimplementation.common.errors import ReimplementationError
from reimplementation.stgcn.validation import load_edge_ids, validate_prepared_data


def validate_no_graph_config(config: Mapping[str, Any]) -> None:
    forbidden = (
        "adjacency",
        "adj",
        "supports",
        "laplacian",
        "cheb_ks",
        "undirected_topology",
        "weighted_adjacency",
        "dtw_path",
    )
    present = [key for key in forbidden if config.get(key) not in (None, "", False)]
    if present:
        raise ReimplementationError(f"TLAST config must not supply graph inputs {present}")


def validate_node_order(*, r_nodes_path: Path, node_mapping: Path, expected_nodes: int) -> list[str]:
    node_ids = load_edge_ids(node_mapping)
    r_ids = load_edge_ids(r_nodes_path)
    if r_ids != node_ids:
        raise ReimplementationError("r_nodes.csv edge_id order does not match node_mapping.csv")
    if len(node_ids) != int(expected_nodes):
        raise ReimplementationError(
            f"node count {len(node_ids)} != expected {expected_nodes}; refusing to reorder by edge_id"
        )
    return node_ids


def assert_task_config(config: Mapping[str, Any]) -> None:
    validate_no_graph_config(config)
    if str(config.get("target_mode")) != "last-observed-step":
        raise ReimplementationError("target_mode must be last-observed-step")
    if int(config.get("output_len", config.get("output_window", 0))) != 1:
        raise ReimplementationError("TLAST output_len must be 1")
    if int(config.get("input_len", config.get("n_his", 0))) != 12:
        raise ReimplementationError("TLAST input_len must be 12")
    if int(config.get("num_nodes", 0)) != 1406:
        raise ReimplementationError("Xuancheng26 TLAST num_nodes must be 1406")
    if not config.get("input_type") or not config.get("target_type"):
        raise ReimplementationError("pass --input-type and --target-type; TLAST does not train the legacy flat inflow task")
    if str(config.get("att_type", "proxy")) != "proxy":
        raise ReimplementationError("TLAST att_type must stay proxy")
    if not bool(int(config.get("query_cross_time", 1))):
        raise ReimplementationError("query_cross_time must stay enabled")
    if int(config.get("slice_size_per_day", config.get("time_of_day_size", 288))) != 288:
        raise ReimplementationError("slice_size_per_day must stay 288")


__all__ = [
    "assert_task_config",
    "validate_no_graph_config",
    "validate_node_order",
    "validate_prepared_data",
]
