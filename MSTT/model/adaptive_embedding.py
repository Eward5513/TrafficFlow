"""Spatio-temporal adaptive embedding (STAEformer ``adaptive_embedding``).

One learnable ``[T, N, D]`` table shared by every sample: ``T`` is the
relative history position 0 .. input_window-1, ``N`` the road in R-graph
order, ``D`` the hidden width. It does not depend on the input values or on
the calendar, so it is neither the TOD/DOW nor the node embedding.
MSTT supports addition (legacy default) or independent concatenation after
the base embedding dropout. Initialization follows STAEformer
(``xavier_uniform_``).
"""

from __future__ import annotations

import torch
from torch import nn


class AdaptiveEmbedding(nn.Module):
    def __init__(self, *, input_window: int, num_nodes: int, d_model: int, fusion: str = "add") -> None:
        super().__init__()
        if fusion not in ("add", "concat"):
            raise ValueError(f"unknown adaptive fusion {fusion!r}")
        self.fusion = fusion
        self.input_window = int(input_window)
        self.num_nodes = int(num_nodes)
        self.d_model = int(d_model)
        self.adaptive_embedding = nn.Parameter(torch.empty(self.input_window, self.num_nodes, self.d_model))
        nn.init.xavier_uniform_(self.adaptive_embedding)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """``hidden`` [B, T, N, D] + table broadcast over the batch -> [B, T, N, D]."""
        expected = (self.input_window, self.num_nodes, self.d_model)
        if hidden.ndim != 4 or tuple(hidden.shape[1:3]) != expected[:2] or (self.fusion == "add" and hidden.shape[-1] != self.d_model):
            raise ValueError(f"expected [B, {expected[0]}, {expected[1]}, {expected[2]}], got {tuple(hidden.shape)}")
        table = self.adaptive_embedding.unsqueeze(0)
        if self.fusion == "concat":
            return torch.cat((hidden, table.expand(hidden.shape[0], -1, -1, -1)), dim=-1)
        return hidden + table
