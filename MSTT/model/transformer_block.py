"""Pre-RMSNorm spatio-temporal block.

```text
H = H + Dropout(TemporalAttention(RMSNorm_t(H)))
H = H + Dropout(SpatialAttention(RMSNorm_s(H)))
H = H + Dropout(SwiGLU(RMSNorm_f(H)))
```
"""

from __future__ import annotations

import torch
from torch import nn


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
