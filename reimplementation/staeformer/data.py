"""Assemble official 3-channel STAEformer history from R-only NPZ metadata.

Channel 0 is already-normalized flow. Channel 1 is ``slot / 288``. Channel 2
is the integer weekday 0..6 (official PeMS npz encoding). NPZ files are not
rewritten. Integer TOD/DOW indices are returned separately for embedding
lookup so float32 ``slot/288*288`` truncation cannot skip an embedding row.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch

from reimplementation.common.errors import ReimplementationError
from reimplementation.staeformer.temporal_features import (
    ORIGINAL_PEMS_NPZ_DOW,
    SLOTS_PER_DAY,
    assert_same_day_window,
    assert_target_is_last_history,
    day_of_week_index,
    time_of_day_fraction,
)


def assemble_history_and_indices(
    x: np.ndarray | torch.Tensor,
    *,
    window_start_slot: np.ndarray | torch.Tensor,
    day_index: np.ndarray | torch.Tensor,
    n_his: int,
    input_dim: int,
    tod_embedding_dim: int,
    dow_embedding_dim: int,
    day_of_week_source: str = ORIGINAL_PEMS_NPZ_DOW,
    weekday_mapping: Mapping[str, int] | None = None,
    all_days: Sequence[str] | None = None,
    target_slot: np.ndarray | torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
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
    if int(input_dim) not in {1, 2, 3}:
        raise ReimplementationError(f"unsupported input_dim {input_dim}")
    slots = assert_same_day_window(window_start_slot, n_his)
    if slots.shape[0] != batch:
        raise ReimplementationError("window_start_slot batch does not match x")
    if target_slot is not None:
        assert_target_is_last_history(window_start_slot, target_slot, n_his)
    history = np.zeros((batch, time, nodes, int(input_dim)), dtype=np.float32)
    history[..., 0:1] = traffic.astype(np.float32, copy=False)
    tod_idx = torch.from_numpy(np.ascontiguousarray(slots)).to(dtype=torch.long)
    if int(input_dim) >= 2 or int(tod_embedding_dim) > 0:
        if np.any(slots[:, -1] < 0) or np.any(slots[:, -1] >= SLOTS_PER_DAY):
            raise ReimplementationError("last history slot is outside [0, 288)")
        fractions = time_of_day_fraction(slots)
        if int(input_dim) >= 2:
            history[..., 1] = np.repeat(fractions[:, :, None], nodes, axis=2)
    dow_idx = None
    if int(input_dim) >= 3 or int(dow_embedding_dim) > 0:
        dow = day_of_week_index(
            day_index,
            source=day_of_week_source,
            weekday_mapping=weekday_mapping,
            all_days=all_days,
        )
        if dow.shape[0] != batch:
            raise ReimplementationError("day_index batch does not match x")
        if int(input_dim) >= 3:
            history[..., 2] = dow[:, None, None].astype(np.float32)
        dow_idx = torch.from_numpy(np.repeat(dow[:, None], n_his, axis=1).astype(np.int64))
    payload = {
        "history": torch.from_numpy(np.ascontiguousarray(history)),
        "time_of_day_index": tod_idx,
    }
    if dow_idx is not None:
        payload["day_of_week_index"] = dow_idx
    return payload
