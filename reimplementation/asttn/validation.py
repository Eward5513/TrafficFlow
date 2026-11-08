"""Input checks for the Xuancheng26 ASTTN run."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from reimplementation.common.errors import ReimplementationError
from reimplementation.stgcn.validation import load_edge_ids, validate_prepared_data

_SMOKE_KEYS = ("max_batches", "max_train_batches", "max_eval_batches", "protocol_limit")


def assert_task_config(config: Mapping[str, Any]) -> None:
    leaked = [key for key in _SMOKE_KEYS if config.get(key) not in (None, "", False)]
    if leaked:
        raise ReimplementationError(f"official ASTTN config must not set smoke limits {leaked}")
    if str(config.get("target_mode")) != "last-observed-step":
        raise ReimplementationError("target_mode must be last-observed-step")
    if int(config.get("n_his", config.get("num_his", 0))) != 12:
        raise ReimplementationError("ASTTN history length must stay 12")
    if int(config.get("output_len", config.get("output_steps", 0))) != 1:
        raise ReimplementationError("ASTTN predicts the last observed step only")
    if int(config.get("num_nodes", 0)) != 1406:
        raise ReimplementationError("Xuancheng26 ASTTN num_nodes must be 1406")
    if int(config.get("L", 1)) != 1 or int(config.get("K", 8)) != 8 or int(config.get("d", 8)) != 8:
        raise ReimplementationError("ASTTN keeps the reference Model_Adp width L=1, K=8, d=8")
    if int(config.get("window", 6)) != 6:
        raise ReimplementationError("ASTTN encoder window must stay 6")
    if str(config.get("model_branch", "adp")) != "adp":
        raise ReimplementationError("ASTTN uses Model_Adp, not Model_Both")


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
