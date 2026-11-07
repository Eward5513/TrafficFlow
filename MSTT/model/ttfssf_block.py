"""Two temporal layers, one FFN, two spatial layers, one FFN.

The legacy TSF and SFTF blocks are unchanged. This block is one stack:

```text
H = H + Dropout(Temporal_1(RMSNorm(H)))
H = H + Dropout(Temporal_2(RMSNorm(H)))
H = H + Dropout(FFN_T(RMSNorm(H)))
H = H + Dropout(Spatial_1(RMSNorm(H)))
H = H + Dropout(Spatial_2(RMSNorm(H)))
H = H + Dropout(FFN_S(RMSNorm(H)))
```
"""

from __future__ import annotations

import torch
from torch import nn


class TTFSSFBlock(nn.Module):
    execution_order = ("T", "T", "FFN_T", "S", "S", "FFN_S")

    def __init__(
        self,
        d_model: int,
        *,
        temporal: nn.Module,
        temporal_2: nn.Module,
        spatial: nn.Module,
        spatial_2: nn.Module,
        ffn_t: nn.Module,
        ffn_s: nn.Module,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.temporal_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.temporal = temporal
        self.temporal_2_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.temporal_2 = temporal_2
        self.ffn_t_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.ffn_t = ffn_t
        self.spatial_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.spatial = spatial
        self.spatial_2_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.spatial_2 = spatial_2
        self.ffn_s_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.ffn_s = ffn_s
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.dropout(self.temporal(self.temporal_norm(hidden)))
        hidden = hidden + self.dropout(self.temporal_2(self.temporal_2_norm(hidden)))
        hidden = hidden + self.dropout(self.ffn_t(self.ffn_t_norm(hidden)))
        hidden = hidden + self.dropout(self.spatial(self.spatial_norm(hidden)))
        hidden = hidden + self.dropout(self.spatial_2(self.spatial_2_norm(hidden)))
        hidden = hidden + self.dropout(self.ffn_s(self.ffn_s_norm(hidden)))
        return hidden
