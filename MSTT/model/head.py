"""Reconstruction head: ``[B, T, N, D] -> [B, N, C_out]`` for the last observed slot.

``flatten`` reads every history step (``Linear(T * D, C_out)``, like
STAEformer's mixed projection). ``last`` reads only step ``T-1``.
``flatten_mlp`` uses the same flattened history and a shared two-layer map
``Linear(T * D, H) -> SiLU -> Linear(H, C_out)``. ``flatten`` and ``last``
keep a single ``Linear`` at ``proj`` so older checkpoints still load strictly.
"""

from __future__ import annotations

import torch
from torch import nn

_HEAD_TYPES = ("flatten", "last", "flatten_mlp")


class PredictionHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        input_window: int,
        out_channels: int,
        *,
        head_type: str = "flatten",
        norm_eps: float = 1e-6,
        fusion_mlp_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if head_type not in _HEAD_TYPES:
            raise ValueError(f"head_type {head_type!r} not in {_HEAD_TYPES}")
        self.head_type = head_type
        self.input_window = int(input_window)
        self.fusion_mlp_hidden_dim = int(fusion_mlp_hidden_dim)
        self.norm = nn.RMSNorm(d_model, eps=norm_eps)
        if head_type == "flatten_mlp":
            if self.fusion_mlp_hidden_dim < 1:
                raise ValueError("fusion_mlp_hidden_dim must be positive")
            flat_dim = int(d_model) * self.input_window
            # Indexes 0 and 2 are the checkpoint contract: proj.0.* and proj.2.*.
            self.proj = nn.Sequential(
                nn.Linear(flat_dim, self.fusion_mlp_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.fusion_mlp_hidden_dim, int(out_channels)),
            )
        else:
            width = d_model * self.input_window if head_type == "flatten" else d_model
            self.proj = nn.Linear(width, int(out_channels))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, width = hidden.shape
        hidden = self.norm(hidden)
        if self.head_type == "last":
            return self.proj(hidden[:, -1])
        if steps != self.input_window:
            raise ValueError(f"{self.head_type} head expects T={self.input_window}, got {steps}")
        flat = hidden.permute(0, 2, 1, 3).reshape(batch, nodes, steps * width)
        return self.proj(flat)

    def describe(self) -> dict[str, object]:
        if self.head_type != "flatten_mlp":
            return {
                "type": self.head_type,
                "norm": "RMSNorm",
                "in_features": int(self.proj.in_features),
                "out_features": int(self.proj.out_features),
            }
        first, second = self.proj[0], self.proj[2]
        return {
            "type": "flatten_mlp",
            "norm": "RMSNorm",
            "input_dim": int(first.in_features),
            "hidden_dim": int(first.out_features),
            "output_dim": int(second.out_features),
            "activation": "SiLU",
            "dropout": False,
            "shared_across_roads": True,
        }
