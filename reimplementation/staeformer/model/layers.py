"""Self-attention block and mixed-projection flatten from original STAEformer.

Post-norm order (official)::

    x = LN(x + Dropout(Attn(x)))
    x = LN(x + Dropout(FFN(x)))

FFN is Linear-ReLU-Linear. Attention axis is moved to -2, then restored.
"""

from __future__ import annotations

import torch
from torch import nn

from reimplementation.common.errors import ReimplementationError
from reimplementation.staeformer.model.attention import AttentionLayer


def flatten_time_hidden(hidden: torch.Tensor) -> torch.Tensor:
    """``[B, T, V, D] -> [B, V, T*D]`` with D fastest, then T old-to-new.

    Matches official mixed projection::

        transpose(1, 2); reshape(B, V, T * D)
    """
    if hidden.ndim != 4:
        raise ReimplementationError(f"mixed-proj input must be [B,T,V,D], got {tuple(hidden.shape)}")
    batch, time, nodes, dim = hidden.shape
    node_major = hidden.transpose(1, 2).contiguous()
    flat = node_major.reshape(batch, nodes, time * dim)
    if tuple(flat.shape) != (batch, nodes, time * dim):
        raise ReimplementationError(f"flatten {tuple(flat.shape)} != {(batch, nodes, time * dim)}")
    return flat


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        feed_forward_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        mask: bool = False,
    ) -> None:
        super().__init__()
        if dropout < 0.0 or dropout > 1.0:
            raise ReimplementationError(f"dropout must be in [0, 1], got {dropout}")
        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, dim: int = -2) -> torch.Tensor:
        x = x.transpose(dim, -2)
        residual = x
        out = self.attn(x, x, x)
        out = self.dropout1(out)
        out = self.ln1(residual + out)
        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)
        return out.transpose(dim, -2)
