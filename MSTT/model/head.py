"""Reconstruction head: ``[B, T, N, D] -> [B, N, C_out]`` for the last observed slot.

``flatten`` reads every history step (``Linear(T * D, C_out)``, like
STAEformer's mixed projection) and keeps a single ``Linear`` at ``proj``.
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
        if head_type != "flatten":
            raise ValueError(
                "head_type must be flatten; flatten_mlp, last, and state_token_transformer were removed, "
                f"got {head_type!r}"
            )
        self.head_type = head_type
        self.input_window = int(input_window)
        self.norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.proj = nn.Linear(int(d_model) * self.input_window, int(out_channels))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, width = hidden.shape
        hidden = self.norm(hidden)
        if steps != self.input_window:
            raise ValueError(f"{self.head_type} head expects T={self.input_window}, got {steps}")
        flat = hidden.permute(0, 2, 1, 3).reshape(batch, nodes, steps * width)
        return self.proj(flat)

    def describe(self) -> dict[str, object]:
        return {
            "type": self.head_type,
            "norm": "RMSNorm",
            "in_features": int(self.proj.in_features),
            "out_features": int(self.proj.out_features),
        }
