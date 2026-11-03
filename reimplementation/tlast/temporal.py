"""Per-step TOD/DOW indexes for TLAST.

Xuancheng26 stores ``day_index`` as Monday=0 weekday. ``day_index % 7``
keeps that weekday, matching the STID and STAEformer loaders. Windows stay
inside one day, so every history step shares the weekday and the slot
index is ``window_start_slot + offset``.
"""

from __future__ import annotations

from typing import Mapping

import torch

from reimplementation.common.errors import ReimplementationError

SLOTS_PER_DAY = 288


def time_index_from_batch(
    batch: Mapping[str, torch.Tensor],
    n_his: int,
    *,
    slots_per_day: int = SLOTS_PER_DAY,
) -> torch.Tensor:
    """Return ``[B, T, 2]`` long indexes ``(time_of_day, day_of_week)``."""
    start = batch["window_start_slot"].to(dtype=torch.long).reshape(-1)
    steps = int(n_his)
    offsets = torch.arange(steps, device=start.device, dtype=torch.long)
    slots = start[:, None] + offsets[None, :]
    if bool((slots < 0).any().item()) or bool((slots >= int(slots_per_day)).any().item()):
        raise ReimplementationError(
            "history slots leave [0, 288). Prepared windows are not allowed to cross days."
        )
    target = batch["target_slot"].to(dtype=torch.long).reshape(-1)
    if target.shape != slots[:, -1].shape or not torch.equal(slots[:, -1], target):
        raise ReimplementationError("target_slot must be the last history slot")
    weekday = batch["day_index"].to(dtype=torch.long).reshape(-1)
    if bool((weekday < 0).any().item()):
        raise ReimplementationError("day_index must be non-negative")
    weekday = torch.remainder(weekday, 7)
    weekday = weekday[:, None].expand(-1, steps)
    return torch.stack((slots, weekday), dim=-1)
