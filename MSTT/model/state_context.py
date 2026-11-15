"""One TOD, DOW, and adaptive table shared by every state branch.

The module owns the only copies of those three parameters. It returns tensors;
it does not register itself on the branches and it does not read files.
Adaptive values stay the relative-history table from ``AdaptiveEmbedding``.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from MSTT.model.adaptive_embedding import AdaptiveEmbedding
from MSTT.model.embedding import _check_index


class StateContext(NamedTuple):
    """Calendar features are ``[B, T, 1, D]``. Adaptive is ``[1, T, N, D]``."""

    tod: torch.Tensor
    dow: torch.Tensor
    adaptive: torch.Tensor


class SharedStateContextEmbedding(nn.Module):
    def __init__(
        self,
        *,
        steps_per_day: int,
        tod_dim: int,
        dow_dim: int,
        input_window: int,
        num_nodes: int,
        adaptive_dim: int,
    ) -> None:
        super().__init__()
        self.steps_per_day = int(steps_per_day)
        self.tod_dim = int(tod_dim)
        self.dow_dim = int(dow_dim)
        self.input_window = int(input_window)
        self.num_nodes = int(num_nodes)
        self.adaptive_dim = int(adaptive_dim)
        if min(self.steps_per_day, self.tod_dim, self.dow_dim, self.input_window, self.num_nodes, self.adaptive_dim) < 1:
            raise ValueError("shared state context widths must be positive")
        self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_dim)
        self.dow_embedding = nn.Embedding(7, self.dow_dim)
        nn.init.trunc_normal_(self.tod_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.dow_embedding.weight, std=0.02)
        # Xavier lives on this parameter. Do not allocate a second table.
        self.adaptive = AdaptiveEmbedding(
            input_window=self.input_window,
            num_nodes=self.num_nodes,
            d_model=self.adaptive_dim,
            fusion="concat",
        )

    def forward(
        self,
        x: torch.Tensor,
        time_of_day: torch.Tensor | None,
        day_of_week: torch.Tensor | None,
    ) -> StateContext:
        if x.ndim != 4:
            raise ValueError(f"x must be [B, T, N, C], got {tuple(x.shape)}")
        batch, steps, nodes = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        if steps != self.input_window or nodes != self.num_nodes:
            raise ValueError(
                f"x has T={steps}, N={nodes}; shared context is T={self.input_window}, N={self.num_nodes}"
            )
        tod = self.tod_embedding(
            _check_index(time_of_day, "time_of_day", batch, steps, self.steps_per_day)
        )[:, :, None, :]
        dow = self.dow_embedding(
            _check_index(day_of_week, "day_of_week", batch, steps, 7)
        )[:, :, None, :]
        adaptive = self.adaptive.adaptive_embedding.unsqueeze(0)
        return StateContext(tod, dow, adaptive)
