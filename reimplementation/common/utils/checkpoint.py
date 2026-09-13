"""Checkpoint save/load with experiment-identity checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from reimplementation.common.errors import ReimplementationError
from reimplementation.common.utils.atomic_io import atomic_write_bytes

CHECKPOINT_KEYS = (
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "epoch",
    "best_metric",
    "config",
    "penetration_rate",
    "seed",
    "model_parameter_count",
    "graph_sha256",
    "data_file_sha256",
    "normalization_sha256",
    "code_version",
)


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    missing = [key for key in CHECKPOINT_KEYS if key not in payload]
    if missing:
        raise ReimplementationError(f"checkpoint missing fields: {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = _torch_save_bytes(dict(payload))
    atomic_write_bytes(path, buffer)


def _torch_save_bytes(payload: dict[str, Any]) -> bytes:
    import io

    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    if not path.is_file():
        raise ReimplementationError(f"checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ReimplementationError("checkpoint is not a dict")
    missing = [key for key in CHECKPOINT_KEYS if key not in payload]
    if missing:
        raise ReimplementationError(f"checkpoint missing fields: {missing}")
    return payload


def assert_checkpoint_compatible(
    payload: Mapping[str, Any],
    *,
    penetration_rate: str,
    graph_sha256: str,
    data_file_sha256: str,
    num_nodes: int,
    n_his: int,
    output_steps: int,
    input_channels: int,
    output_channels: int,
) -> None:
    if str(payload["penetration_rate"]) != str(penetration_rate):
        raise ReimplementationError(
            "refusing to restore a checkpoint from a different penetration rate: "
            f"{payload['penetration_rate']} vs {penetration_rate}"
        )
    if payload["graph_sha256"] != graph_sha256:
        raise ReimplementationError("checkpoint graph SHA256 does not match the loaded adjacency")
    if payload["data_file_sha256"] != data_file_sha256:
        raise ReimplementationError("checkpoint data SHA256 does not match the loaded NPZ")
    config = payload.get("config") or {}
    checks = {
        "num_nodes": num_nodes,
        "n_his": n_his,
        "output_steps": output_steps,
        "input_channels": input_channels,
        "output_channels": output_channels,
    }
    for key, expected in checks.items():
        actual = config.get(key)
        if actual is not None and int(actual) != int(expected):
            raise ReimplementationError(
                f"checkpoint config {key}={actual} does not match runtime {expected}"
            )
