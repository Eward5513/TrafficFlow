"""Spatio-temporal adaptive embedding (STAEformer ``adaptive_embedding``).

One learnable ``[T, N, D]`` table shared by every sample: ``T`` is the
relative history position 0 .. input_window-1, ``N`` the road in R-graph
order, ``D`` the hidden width. It does not depend on the input values or on
the calendar, so it is neither the TOD/DOW nor the node embedding.
STAEformer concatenates it; MSTT adds it to the fused input embedding so the
backbone width stays ``d_model``. Initialization follows STAEformer
(``xavier_uniform_``).
"""

from __future__ import annotations

import torch
from torch import nn


class AdaptiveEmbedding(nn.Module):
    def __init__(self, *, input_window: int, num_nodes: int, d_model: int) -> None:
        super().__init__()
        self.input_window = int(input_window)
        self.num_nodes = int(num_nodes)
        self.d_model = int(d_model)
        self.adaptive_embedding = nn.Parameter(torch.empty(self.input_window, self.num_nodes, self.d_model))
        nn.init.xavier_uniform_(self.adaptive_embedding)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """``hidden`` [B, T, N, D] + table broadcast over the batch -> [B, T, N, D]."""
        expected = (self.input_window, self.num_nodes, self.d_model)
        if hidden.ndim != 4 or tuple(hidden.shape[1:]) != expected:
            raise ValueError(f"expected [B, {expected[0]}, {expected[1]}, {expected[2]}], got {tuple(hidden.shape)}")
        return hidden + self.adaptive_embedding.unsqueeze(0)
