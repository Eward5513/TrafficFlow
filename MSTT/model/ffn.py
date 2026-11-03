"""SwiGLU feed-forward: ``W_out(SiLU(W_gate x) * W_value x)``."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_value = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_out = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(F.silu(self.w_gate(x)) * self.w_value(x))
