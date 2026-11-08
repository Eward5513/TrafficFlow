"""Pre-RMSNorm blocks.

``MSTTBlock`` is the default T, then S, then SwiGLU stack.
``JointSTBlock`` replaces those two attention sublayers with one joint window.
"""

from __future__ import annotations

import torch
from torch import nn


class JointSTBlock(nn.Module):
    """One joint space-time attention sublayer, then the original SwiGLU.

    ```text
    H = H + Dropout(JointSTAttention(RMSNorm(H)))
    H = H + Dropout(SwiGLU(RMSNorm(H)))
    ```
    """

    def __init__(
        self,
        d_model: int,
        *,
        joint: nn.Module,
        ffn: nn.Module,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.joint_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.joint = joint
        self.ffn_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.ffn = ffn
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.dropout(self.joint(self.joint_norm(hidden)))
        hidden = hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))
        return hidden


class MSTTBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        temporal: nn.Module,
        spatial: nn.Module,
        ffn: nn.Module,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.temporal_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.temporal = temporal
        self.spatial_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.spatial = spatial
        self.ffn_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.ffn = ffn
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.dropout(self.temporal(self.temporal_norm(hidden)))
        hidden = hidden + self.dropout(self.spatial(self.spatial_norm(hidden)))
        hidden = hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))
        return hidden
