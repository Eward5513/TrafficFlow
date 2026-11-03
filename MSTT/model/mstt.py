"""MSTT v0.1: encoder-only Modern Spatio-Temporal Transformer.

``forward(x [B, T, N, C_in], time_of_day [B, T], day_of_week [B, T]) -> [B, N, C_out]``.
The model knows nothing about files, tasks, or scalers; the spatial graph
is handed in as a :class:`SpatialGraph`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

import torch
from torch import nn

from MSTT.model.attention import TemporalSelfAttention, build_spatial_attention
from MSTT.model.embedding import InputEmbedding
from MSTT.model.ffn import SwiGLU
from MSTT.model.graph import SpatialGraph
from MSTT.model.head import PredictionHead
from MSTT.model.transformer_block import MSTTBlock


@dataclass
class MSTTConfig:
    num_nodes: int
    in_channels: int
    out_channels: int
    input_window: int = 12
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ffn_hidden_dim: int = 176
    dropout: float = 0.1
    attention_dropout: float = 0.1
    embedding_dropout: float = 0.1
    use_qk_norm: bool = True
    use_rope: bool = True
    rope_base: float = 10000.0
    use_gated_attention: bool = False
    spatial_attention: str = "sparse"
    use_edge_type_bias: bool = True
    spatial_chunk_size: int | None = 64
    spatial_checkpoint: bool = True
    use_tod: bool = True
    use_dow: bool = True
    use_node_embedding: bool = True
    steps_per_day: int = 288
    norm_eps: float = 1e-6
    head_type: str = "flatten"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], **overrides: Any) -> "MSTTConfig":
        names = {item.name for item in fields(cls)}
        values = {key: payload[key] for key in names if key in payload}
        values.update(overrides)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MSTT(nn.Module):
    def __init__(self, config: MSTTConfig, graph: SpatialGraph | None) -> None:
        super().__init__()
        self.config = config
        if graph is None:
            raise ValueError(f"spatial_attention={config.spatial_attention!r} needs a SpatialGraph")
        if int(graph.num_nodes) != int(config.num_nodes):
            raise ValueError(f"graph has {graph.num_nodes} nodes, config {config.num_nodes}")
        self.embedding = InputEmbedding(
            in_channels=config.in_channels,
            d_model=config.d_model,
            num_nodes=config.num_nodes,
            steps_per_day=config.steps_per_day,
            use_node_embedding=config.use_node_embedding,
            use_tod=config.use_tod,
            use_dow=config.use_dow,
            dropout=config.embedding_dropout,
        )
        self.blocks = nn.ModuleList(
            MSTTBlock(
                config.d_model,
                temporal=TemporalSelfAttention(
                    config.d_model,
                    config.num_heads,
                    attention_dropout=config.attention_dropout,
                    use_qk_norm=config.use_qk_norm,
                    use_rope=config.use_rope,
                    rope_base=config.rope_base,
                    use_gated_attention=config.use_gated_attention,
                    norm_eps=config.norm_eps,
                ),
                spatial=build_spatial_attention(
                    config.spatial_attention,
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    graph=graph,
                    attention_dropout=config.attention_dropout,
                    use_qk_norm=config.use_qk_norm,
                    use_gated_attention=config.use_gated_attention,
                    use_edge_type_bias=config.use_edge_type_bias,
                    chunk_size=config.spatial_chunk_size,
                    use_checkpoint=config.spatial_checkpoint,
                    norm_eps=config.norm_eps,
                ),
                ffn=SwiGLU(config.d_model, config.ffn_hidden_dim),
                dropout=config.dropout,
                norm_eps=config.norm_eps,
            )
            for _ in range(config.num_layers)
        )
        self.head = PredictionHead(
            config.d_model,
            config.input_window,
            config.out_channels,
            head_type=config.head_type,
            norm_eps=config.norm_eps,
        )

    def forward(
        self,
        x: torch.Tensor,
        time_of_day: torch.Tensor | None = None,
        day_of_week: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must be [B, T, N, C_in], got {tuple(x.shape)}")
        if int(x.shape[1]) != self.config.input_window:
            raise ValueError(f"x has T={x.shape[1]}, model input_window={self.config.input_window}")
        hidden = self.embedding(x, time_of_day, day_of_week)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden)

    def parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.parameters() if param.requires_grad))
