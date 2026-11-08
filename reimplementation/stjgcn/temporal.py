"""Calendar indexes for STJGCN.

The PeMSD7 loader builds day-of-week by assuming the first interval is Monday.
Xuancheng26 instead stores the real weekday in ``day_index`` (Monday=0). Only
the 12 observed slots are embedded. The target is the last of those slots, so
there is no future time step and no full-state label in the time input.
"""

from __future__ import annotations

from typing import Mapping

import torch

from reimplementation.common.errors import ReimplementationError

SLOTS_PER_DAY = 288


def time_index(batch: Mapping[str, torch.Tensor], n_his: int, *, slots_per_day: int = SLOTS_PER_DAY) -> torch.Tensor:
    """Return ``[B, n_his, 2]`` as ``(day_of_week, time_of_day)``."""
    start = batch["window_start_slot"].to(dtype=torch.long).reshape(-1)
    steps = int(n_his)
    slots = start[:, None] + torch.arange(steps, device=start.device)
    if bool((slots < 0).any().item()) or bool((slots >= int(slots_per_day)).any().item()):
        raise ReimplementationError("history slots must stay inside one day of 288 slots")
    target = batch["target_slot"].to(dtype=torch.long).reshape(-1)
    if target.shape != slots[:, -1].shape or not torch.equal(slots[:, -1], target):
        raise ReimplementationError("target_slot must be the last history slot")
    weekday = batch["day_index"].to(dtype=torch.long).reshape(-1)
    if bool((weekday < 0).any().item()):
        raise ReimplementationError("day_index must be non-negative")
    weekday = torch.remainder(weekday, 7)[:, None].expand(-1, steps)
    return torch.stack((weekday, slots), dim=-1)
