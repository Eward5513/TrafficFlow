"""Assemble STID history channels from prepared R-only NPZ metadata.

Traffic stays the already-normalized ``x``. Time-of-day and day-of-week
fractions are attached as channels 1 and 2 so the original PeMS
``input_dim=3`` Conv2d and last-step identity lookup both work. NPZ files
are not rewritten.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch

from reimplementation.common.errors import ReimplementationError
from reimplementation.stid.temporal_identity import (
    ORIGINAL_DAY_SOURCE,
    SLOTS_PER_DAY,
    assert_same_day_window,
    assert_target_is_last_history,
    day_of_week_fraction,
    time_of_day_fraction,
)


def assemble_history_features(
    x: np.ndarray | torch.Tensor,
    *,
    window_start_slot: np.ndarray | torch.Tensor,
    day_index: np.ndarray | torch.Tensor,
    n_his: int,
    if_time_in_day: bool,
    if_day_in_week: bool,
    input_dim: int,
    day_of_week_source: str = ORIGINAL_DAY_SOURCE,
    weekday_mapping: Mapping[str, int] | None = None,
    all_days: Sequence[str] | None = None,
    target_slot: np.ndarray | torch.Tensor | None = None,
) -> torch.Tensor:
    """Return history ``[B, T, N, C]`` with C matching original PeMS layout.

    Channel 0 is traffic. Channel 1 is ``slot / 288``. Channel 2 is
    ``day_of_week_index / 7``. Identity embeddings later read the **last**
    time step of channels 1 and 2, which is the reconstruction target slot.
    """
    if torch.is_tensor(x):
        traffic = x.detach().cpu().numpy()
    else:
        traffic = np.asarray(x)
    if traffic.ndim != 4:
        raise ReimplementationError(f"x must be [B,T,N,C], got {traffic.shape}")
    batch, time, nodes, channels = traffic.shape
    if int(time) != int(n_his):
        raise ReimplementationError(f"x time {time} != n_his {n_his}")
    if int(channels) != 1:
        raise ReimplementationError("R-only traffic channel count must be 1")
    slots = assert_same_day_window(window_start_slot, n_his)
    if slots.shape[0] != batch:
        raise ReimplementationError("window_start_slot batch does not match x")
    if target_slot is not None:
        assert_target_is_last_history(window_start_slot, target_slot, n_his)
    need_tod = bool(if_time_in_day) or int(input_dim) >= 2
    need_dow = bool(if_day_in_week) or int(input_dim) >= 3
    if need_dow:
        channel_count = 3
    elif need_tod:
        channel_count = 2
    else:
        channel_count = 1
    if int(input_dim) not in {1, 2, 3}:
        raise ReimplementationError(f"unsupported input_dim {input_dim}")
    if int(input_dim) > channel_count:
        raise ReimplementationError(
            f"input_dim {input_dim} requires {input_dim} assembled channels, got plan {channel_count}"
        )
    if if_time_in_day and channel_count < 2:
        raise ReimplementationError("if_T_i_D requires a time-of-day channel at index 1")
    if if_day_in_week and channel_count < 3:
        raise ReimplementationError("if_D_i_W requires a day-of-week channel at index 2")
    history = np.zeros((batch, time, nodes, channel_count), dtype=np.float32)
    history[..., 0:1] = traffic.astype(np.float32, copy=False)
    if channel_count >= 2:
        fractions = time_of_day_fraction(slots)
        history[..., 1] = np.repeat(fractions[:, :, None], nodes, axis=2).astype(np.float32)
    if channel_count >= 3:
        dow = day_of_week_fraction(
            day_index,
            source=day_of_week_source,
            weekday_mapping=weekday_mapping,
            all_days=all_days,
        )
        if dow.shape[0] != batch:
            raise ReimplementationError("day_index batch does not match x")
        history[..., 2] = dow[:, None, None].astype(np.float32)
    last_slots = slots[:, -1]
    if np.any(last_slots < 0) or np.any(last_slots >= SLOTS_PER_DAY):
        raise ReimplementationError("last history slot is outside [0, 288)")
    return torch.from_numpy(np.ascontiguousarray(history))
