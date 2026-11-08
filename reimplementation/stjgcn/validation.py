"""Input checks for the Xuancheng26 STJGCN run."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from reimplementation.common.errors import ReimplementationError
from reimplementation.stgcn.validation import load_edge_ids, validate_prepared_data

_SMOKE_KEYS = ("max_batches", "max_train_batches", "max_eval_batches", "protocol_limit")


def assert_task_config(config: Mapping[str, Any]) -> None:
    leaked = [key for key in _SMOKE_KEYS if config.get(key) not in (None, "", False)]
    if leaked:
        raise ReimplementationError(f"official STJGCN config must not set smoke limits {leaked}")
    if str(config.get("target_mode")) != "last-observed-step":
        raise ReimplementationError("target_mode must be last-observed-step")
    if int(config.get("P", config.get("n_his", 0))) != 12:
        raise ReimplementationError("STJGCN P must stay 12")
    if int(config.get("Q", config.get("output_steps", 0))) != 1:
        raise ReimplementationError("STJGCN Q is 1 for this reconstruction task")
    if int(config.get("num_nodes", 0)) != 1406:
        raise ReimplementationError("Xuancheng26 STJGCN num_nodes must be 1406")
    if int(config.get("K", 2)) != 2 or int(config.get("d", 64)) != 64:
        raise ReimplementationError("STJGCN keeps the PeMSD7 width K=2, d=64")
    if abs(float(config.get("delta_pdf", 0.9)) - 0.9) > 1e-12:
        raise ReimplementationError("STJGCN delta_pdf stays at the PeMSD7 default 0.9")
    if abs(float(config.get("delta_adt", 0.7)) - 0.7) > 1e-12:
        raise ReimplementationError("STJGCN delta_adt stays at the PeMSD7 default 0.7")
    if abs(float(config.get("beta", 0.5)) - 0.5) > 1e-12:
        raise ReimplementationError("STJGCN beta stays at the PeMSD7 default 0.5")
    if str(config.get("lr_scheduler", "none")) not in {"none", "off"}:
        raise ReimplementationError("PeMSD7 STJGCN has no learning-rate schedule")


def validate_node_order(*, r_nodes_path: Path, node_mapping: Path, expected_nodes: int) -> list[str]:
    node_ids = load_edge_ids(node_mapping)
    r_ids = load_edge_ids(r_nodes_path)
    if r_ids != node_ids:
        raise ReimplementationError("r_nodes.csv edge_id order does not match node_mapping.csv")
    if len(node_ids) != int(expected_nodes):
        raise ReimplementationError(f"node count {len(node_ids)} != {expected_nodes}")
    return node_ids


def validate_inputs(data_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    return validate_prepared_data(
        data_root,
        n_his=int(config.get("n_his", 12)),
        num_nodes=int(config["num_nodes"]),
        rates=[str(item) for item in config["rates"]],
    )
