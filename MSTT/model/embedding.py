"""Input embedding: value projection plus road, time-of-day, and day-of-week embeddings.

Every part maps to ``d_model`` and the parts are summed. TOD/DOW are known
calendar features passed as integer indexes; they are not traffic channels
and do not count toward ``in_channels``.
"""

from __future__ import annotations

import torch
from torch import nn


class InputEmbedding(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        d_model: int,
        num_nodes: int,
        steps_per_day: int = 288,
        use_node_embedding: bool = True,
        use_tod: bool = True,
        use_dow: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_nodes = int(num_nodes)
        self.steps_per_day = int(steps_per_day)
        self.value_projection = nn.Linear(self.in_channels, d_model)
        self.node_embedding = nn.Embedding(self.num_nodes, d_model) if use_node_embedding else None
        self.tod_embedding = nn.Embedding(self.steps_per_day, d_model) if use_tod else None
        self.dow_embedding = nn.Embedding(7, d_model) if use_dow else None
        self.dropout = nn.Dropout(dropout)
        for table in (self.node_embedding, self.tod_embedding, self.dow_embedding):
            if table is not None:
                nn.init.trunc_normal_(table.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        time_of_day: torch.Tensor | None = None,
        day_of_week: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``x`` [B, T, N, C_in]; ``time_of_day``/``day_of_week`` [B, T] long -> [B, T, N, D]."""
        if x.ndim != 4 or int(x.shape[-1]) != self.in_channels:
            raise ValueError(f"expected [B, T, N, {self.in_channels}], got {tuple(x.shape)}")
        hidden = self.value_projection(x)
        if self.node_embedding is not None:
            if int(x.shape[2]) != self.num_nodes:
                raise ValueError(f"node axis {x.shape[2]} != embedding table {self.num_nodes}")
            hidden = hidden + self.node_embedding.weight[None, None, :, :]
        batch, steps = int(x.shape[0]), int(x.shape[1])
        if self.tod_embedding is not None:
            tod = _check_index(time_of_day, "time_of_day", batch, steps, self.steps_per_day)
            hidden = hidden + self.tod_embedding(tod)[:, :, None, :]
        if self.dow_embedding is not None:
            dow = _check_index(day_of_week, "day_of_week", batch, steps, 7)
            hidden = hidden + self.dow_embedding(dow)[:, :, None, :]
        return self.dropout(hidden)


def _check_index(index: torch.Tensor | None, name: str, batch: int, steps: int, size: int) -> torch.Tensor:
    if index is None:
        raise ValueError(f"{name} is required when its embedding is enabled")
    if tuple(index.shape) != (batch, steps):
        raise ValueError(f"{name} must be [B, T]=({batch}, {steps}), got {tuple(index.shape)}")
    if index.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be an integer tensor")
    if bool((index < 0).any()) or bool((index >= size).any()):
        raise ValueError(f"{name} outside [0, {size})")
    return index.long()
