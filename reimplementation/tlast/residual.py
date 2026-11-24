"""Latest partial-observation residual in the target normalized space.

The released TLAST head adds the newest z-scored input, then inverts with
that same scaler. That arithmetic is

``inverse(head + (x_raw - mean) / std) = head * std + x_raw``.

Partial counts and full labels do not share a scaler, so the residual is
built from ``x_raw`` with the **target** mean and std. The input z-score is
not reused. The full label is never added. All-to-single uses the target
channel, not channel 0.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from reimplementation.common.data.r_only_npz_dataset import loss_scale
from reimplementation.common.data.state_windows import ALL_SP_C_CHANNELS, ALL_SP_C_INPUT, CHANNELS
from reimplementation.common.errors import ReimplementationError


def target_channel_index(target_type: str) -> int | None:
    if target_type == "all":
        return None
    if target_type not in CHANNELS:
        raise ReimplementationError(f"unknown target state {target_type}")
    return CHANNELS.index(target_type)


def select_latest_observation(
    latest_raw: torch.Tensor,
    *,
    input_type: str,
    target_type: str,
) -> torch.Tensor:
    """Slice ``[B, 1, N, C_in]`` down to the channels the head predicts."""
    if latest_raw.ndim != 4 or int(latest_raw.shape[1]) != 1:
        raise ReimplementationError(f"latest observation must be [B,1,N,C], got {tuple(latest_raw.shape)}")
    channels = int(latest_raw.shape[-1])
    if target_type == "all":
        if input_type == "all" and channels == 4:
            return latest_raw
        if input_type == ALL_SP_C_INPUT and channels == len(ALL_SP_C_CHANNELS):
            # Speed and the snapshot count are inputs only. The residual is the four states.
            return latest_raw[..., : len(CHANNELS)]
        raise ReimplementationError(
            "all-target residual requires four state channels from input all, "
            f"or the first four of all_sp_c; got input {input_type} with {channels} channels"
        )
    if input_type in {"all", ALL_SP_C_INPUT}:
        expected = 4 if input_type == "all" else len(ALL_SP_C_CHANNELS)
        if channels != expected:
            raise ReimplementationError(
                f"{input_type}-to-single residual requires {expected} input channels, got {channels}"
            )
        index = target_channel_index(target_type)
        assert index is not None
        return latest_raw[..., index : index + 1]
    if input_type != target_type:
        raise ReimplementationError(
            f"single-to-single residual input {input_type} does not match target {target_type}"
        )
    if channels != 1:
        raise ReimplementationError("single-to-single residual expects one input channel")
    return latest_raw


def _as_broadcast(value: Any, reference: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(value) and value.ndim == 0:
        return value.to(dtype=reference.dtype, device=reference.device)
    array = np.asarray(loss_scale(value), dtype=np.float64).reshape(-1)
    if array.size == 1:
        return torch.as_tensor(float(array[0]), dtype=reference.dtype, device=reference.device)
    if int(array.size) != int(reference.shape[-1]):
        raise ReimplementationError(
            f"target scaler width {array.size} != residual channels {reference.shape[-1]}"
        )
    view = (1,) * (reference.ndim - 1) + (int(array.size),)
    return torch.as_tensor(array, dtype=reference.dtype, device=reference.device).reshape(view)


def normalize_with_target_scaler(raw: torch.Tensor, mean: Any, std: Any) -> torch.Tensor:
    mean_t = _as_broadcast(mean, raw)
    std_t = _as_broadcast(std, raw)
    if bool((std_t <= 0).any().item()) if std_t.ndim else float(std_t) <= 0.0:
        raise ReimplementationError("target std must be positive")
    return (raw - mean_t) / std_t


def latest_residual(
    batch: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    mean_y: Any,
    std_y: Any,
) -> torch.Tensor | None:
    if not bool(config.get("use_input_residual", False)):
        return None
    latest = batch["x_raw"][:, -1:, :, :]
    selected = select_latest_observation(
        latest,
        input_type=str(config.get("input_type") or "inflow"),
        target_type=str(config.get("target_type") or "inflow"),
    )
    return normalize_with_target_scaler(selected, mean_y, std_y)
