"""Calendar indexes for ASTTN.

``day_index`` in the Xuancheng26 windows is already the Python weekday with
Monday as 0. ``day_index % 7`` is that public encoding. Time of day is
``window_start_slot + offset`` inside one day of 288 slots. The extra embedding
step repeats the last input slot, which is the reconstruction target. It is
not the next slot, and it is not produced by assuming day 0 is Monday.
"""

from __future__ import annotations

from typing import Mapping

import torch

from reimplementation.common.errors import ReimplementationError

SLOTS_PER_DAY = 288


def history_calendar(
    batch: Mapping[str, torch.Tensor],
    n_his: int,
    *,
    slots_per_day: int = SLOTS_PER_DAY,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    return weekday, slots


def time_embedding(batch: Mapping[str, torch.Tensor], n_his: int, *, slots_per_day: int = SLOTS_PER_DAY) -> torch.Tensor:
    """Return ``[B, n_his + 1, 2]`` as ``(day_of_week, time_of_day)``."""
    weekday, slots = history_calendar(batch, n_his, slots_per_day=slots_per_day)
    history = torch.stack((weekday, slots), dim=-1)
    embedding = torch.cat((history, history[:, -1:]), dim=1)
    if not torch.equal(embedding[:, -1], embedding[:, -2]):
        raise ReimplementationError("ASTTN target calendar must copy the last history slot")
    return embedding
