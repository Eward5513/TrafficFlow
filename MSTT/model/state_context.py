"""TOD, DOW, and adaptive tables shared by every state branch.

The module owns the only copies of those parameters. It may also own the only
road table. It returns tensors; it does not register itself on the branches
and it does not read files. Adaptive values stay the relative-history table
from ``AdaptiveEmbedding``.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from MSTT.model.adaptive_embedding import AdaptiveEmbedding
from MSTT.model.embedding import _check_index


class StateContext(NamedTuple):
    """Calendar features are ``[B, T, 1, D]``. Adaptive is ``[1, T, N, D]``.

    ``node`` is ``[1, 1, N, D]`` when this module owns the road table, and
    ``None`` when each branch keeps its own table. A length-3 context still
    means the historical TOD/DOW/adaptive path.
    """

    tod: torch.Tensor
    dow: torch.Tensor
    adaptive: torch.Tensor
    node: torch.Tensor | None = None


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
        node_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.steps_per_day = int(steps_per_day)
        self.tod_dim = int(tod_dim)
        self.dow_dim = int(dow_dim)
        self.input_window = int(input_window)
        self.num_nodes = int(num_nodes)
        self.adaptive_dim = int(adaptive_dim)
        self.node_dim = None if node_dim is None else int(node_dim)
        if min(self.steps_per_day, self.tod_dim, self.dow_dim, self.input_window, self.num_nodes, self.adaptive_dim) < 1:
            raise ValueError("shared state context widths must be positive")
        if self.node_dim is not None and self.node_dim < 1:
            raise ValueError("shared node width must be positive")
        self.node_embedding = None if self.node_dim is None else nn.Embedding(self.num_nodes, self.node_dim)
        self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_dim)
        self.dow_embedding = nn.Embedding(7, self.dow_dim)
        for table in (self.node_embedding, self.tod_embedding, self.dow_embedding):
            if table is not None:
                nn.init.trunc_normal_(table.weight, std=0.02)
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
        node = None
        if self.node_embedding is not None:
            # A view of the only registered table. Do not detach or clone it.
            node = self.node_embedding.weight.view(1, 1, self.num_nodes, self.node_dim)
        return StateContext(tod, dow, adaptive, node)
