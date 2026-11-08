"""Rotary position embedding over the history-step axis.

Rotate-half convention: dimension ``i`` pairs with ``i + head_dim/2``.
Positions are the step offsets ``0..T-1`` inside the window, so the score
between two steps depends only on their distance.
"""

from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if int(head_dim) % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
        self.head_dim = int(head_dim)
        inv_freq = 1.0 / (float(base) ** (torch.arange(0, self.head_dim, 2, dtype=torch.float64) / self.head_dim))
        self.register_buffer("inv_freq", inv_freq.float(), persistent=False)

    def cos_sin(self, steps: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(int(steps), device=device, dtype=torch.float32)
        angles = positions[:, None] * self.inv_freq.to(device)[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos().to(dtype), angles.sin().to(dtype)

    def rotate_time(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the second-to-last axis as positions ``0 .. T-1``.

        Leading axes share those positions. A road index must not be placed on
        this axis: it is a time offset inside the observed window, not a
        flattened time-by-node coordinate.
        """
        cos, sin = self.cos_sin(int(x.shape[-2]), x.device, x.dtype)
        return apply_rotary(x, cos, sin)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``q``/``k`` [..., T, head_dim] with the step axis second to last."""
        return self.rotate_time(q), self.rotate_time(k)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin
