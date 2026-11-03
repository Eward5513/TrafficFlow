"""Reconstruction head: ``[B, T, N, D] -> [B, N, C_out]`` for the last observed slot.

``flatten`` reads every history step (``Linear(T * D, C_out)``, like
STAEformer's mixed projection). ``last`` reads only step ``T-1``.
"""

from __future__ import annotations

import torch
from torch import nn


class PredictionHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        input_window: int,
        out_channels: int,
        *,
        head_type: str = "flatten",
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if head_type not in ("flatten", "last"):
            raise ValueError(f"head_type {head_type!r} not in ('flatten', 'last')")
        self.head_type = head_type
        self.input_window = int(input_window)
        self.norm = nn.RMSNorm(d_model, eps=norm_eps)
        width = d_model * self.input_window if head_type == "flatten" else d_model
        self.proj = nn.Linear(width, int(out_channels))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, width = hidden.shape
        hidden = self.norm(hidden)
        if self.head_type == "last":
            return self.proj(hidden[:, -1])
        if steps != self.input_window:
            raise ValueError(f"flatten head expects T={self.input_window}, got {steps}")
        return self.proj(hidden.permute(0, 2, 1, 3).reshape(batch, nodes, steps * width))
