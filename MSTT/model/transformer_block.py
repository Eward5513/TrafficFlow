"""Pre-RMSNorm block.

``MSTTBlock`` is the T, then S, then SwiGLU stack.
``build_factorized_block`` allocates a new T, S, and FFN on every call.
"""

from __future__ import annotations

import torch
from torch import nn

from MSTT.model.attention import TemporalSelfAttention, build_spatial_attention
from MSTT.model.ffn import SwiGLU


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


def build_factorized_block(config, graph) -> MSTTBlock:
    """One new T→S→FFN block. Each call creates new parameters; nothing is cached."""
    temporal_heads = config.num_heads if config.temporal_num_heads is None else int(config.temporal_num_heads)
    spatial_heads = config.num_heads if config.spatial_num_heads is None else int(config.spatial_num_heads)
    return MSTTBlock(
        config.d_model,
        temporal=TemporalSelfAttention(
            config.d_model,
            temporal_heads,
            attention_dropout=config.attention_dropout,
            use_qk_norm=config.use_qk_norm,
            use_rope=config.use_rope,
            rope_base=config.rope_base,
            use_gated_attention=config.use_gated_attention,
            norm_eps=config.norm_eps,
            head_dim=config.attention_head_dim,
        ),
        spatial=build_spatial_attention(
            config.spatial_attention,
            d_model=config.d_model,
            num_heads=spatial_heads,
            graph=graph,
            attention_dropout=config.attention_dropout,
            use_qk_norm=config.use_qk_norm,
            use_gated_attention=config.use_gated_attention,
            use_edge_type_bias=config.use_edge_type_bias,
            chunk_size=config.spatial_chunk_size,
            use_checkpoint=config.spatial_checkpoint,
            norm_eps=config.norm_eps,
            head_dim=config.attention_head_dim,
        ),
        ffn=SwiGLU(config.d_model, config.ffn_hidden_dim),
        dropout=config.dropout,
        norm_eps=config.norm_eps,
    )
