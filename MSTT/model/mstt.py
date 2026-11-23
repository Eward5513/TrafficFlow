"""MSTT v0.4: encoder-only Modern Spatio-Temporal Transformer.

``forward(x [B, T, N, C_in], time_of_day [B, T], day_of_week [B, T]) -> [B, N, C_out]``.
The model reads no files or tasks; graph and optional target scaler are injected.
One shared TSF follows the value embedding. Four independent value MLPs, when
selected, still feed that single TSF.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

import torch
from torch import nn

from MSTT.model.adaptive_embedding import AdaptiveEmbedding
from MSTT.model.embedding import InputEmbedding, resolve_embedding_dims
from MSTT.model.graph import SpatialGraph
from MSTT.model.ffn import SwiGLU
from MSTT.model.head import PredictionHead
from MSTT.model.conservation import ConservationProjection
from MSTT.model.transformer_block import build_factorized_block


@dataclass
class MSTTConfig:
    num_nodes: int
    in_channels: int
    out_channels: int
    input_window: int = 12
    d_model: int = 128
    num_heads: int = 4
    temporal_num_heads: int | None = None
    spatial_num_heads: int | None = None
    attention_head_dim: int | None = None
    num_layers: int = 1
    ffn_hidden_dim: int = 176
    dropout: float = 0.1
    attention_dropout: float = 0.1
    embedding_dropout: float = 0.1
    use_qk_norm: bool = True
    use_rope: bool = True
    rope_base: float = 10000.0
    use_gated_attention: bool = False
    spatial_attention: str = "global"
    use_edge_type_bias: bool = False
    spatial_chunk_size: int | None = 64
    spatial_checkpoint: bool = True
    use_tod: bool = True
    use_dow: bool = True
    use_node_embedding: bool = True
    embedding_fusion: str = "concat"
    value_embedding_mode: str = "joint_linear"
    value_embedding_dim: int = 32
    node_embedding_dim: int = 16
    tod_embedding_dim: int = 8
    dow_embedding_dim: int = 8
    use_adaptive_embedding: bool = True
    base_embedding_dim: int | None = 64
    adaptive_embedding_dim: int | None = 64
    adaptive_embedding_fusion: str = "concat"
    ffn_placement: str = "after_spatial"
    attention_order: str = "TS"
    attention_layout: str = "factorized_ts"
    steps_per_day: int = 288
    norm_eps: float = 1e-6
    head_type: str = "flatten"
    conservation_mode: str = "none"
    projection_mlp_hidden_dim: int = 32

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], **overrides: Any) -> "MSTTConfig":
        names = {item.name for item in fields(cls)}
        values = {key: payload[key] for key in names if key in payload}
        values.update(overrides)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MSTT(nn.Module):
    def __init__(self, config: MSTTConfig, graph: SpatialGraph | None, *, target_mean=None, target_std=None) -> None:
        super().__init__()
        self.config = config
        if config.conservation_mode not in ("none", "soft", "projected"):
            raise ValueError(f"unknown conservation_mode {config.conservation_mode!r}")
        if config.conservation_mode != "none" and (
            config.out_channels != 4 or config.in_channels not in (4, 6)
        ):
            raise ValueError(
                "soft/projected conservation requires four output states and four or six input channels, "
                f"got in={config.in_channels} out={config.out_channels}"
            )
        if config.projection_mlp_hidden_dim < 1:
            raise ValueError("projection_mlp_hidden_dim must be positive")
        self.graph_direction = None if graph is None else str(graph.direction)
        self.graph_edges = None if graph is None else int(graph.num_edges)
        if graph is None:
            raise ValueError("MSTT needs a SpatialGraph to check the node count")
        if int(graph.num_nodes) != int(config.num_nodes):
            raise ValueError(f"graph has {graph.num_nodes} nodes, config {config.num_nodes}")
        base_dim, adaptive_dim = resolve_embedding_dims(
            config.d_model, config.base_embedding_dim, config.adaptive_embedding_dim,
            adaptive=config.use_adaptive_embedding, fusion=config.adaptive_embedding_fusion,
        )
        if config.num_layers != 1:
            raise ValueError("MSTT has exactly one block")
        layout = str(config.attention_layout)
        if layout != "factorized_ts":
            raise ValueError(
                "attention_layout must be factorized_ts; joint_st was removed, "
                f"got {layout!r}"
            )
        if config.attention_order != "TS" or config.ffn_placement != "after_spatial":
            raise ValueError("factorized_ts requires one T→S→FFN block")
        if config.head_type != "flatten":
            raise ValueError(
                "head_type must be flatten; flatten_mlp, last, and state_token_transformer were removed, "
                f"got {config.head_type!r}"
            )
        self.embedding = InputEmbedding(
            in_channels=config.in_channels,
            d_model=base_dim,
            num_nodes=config.num_nodes,
            steps_per_day=config.steps_per_day,
            use_node_embedding=config.use_node_embedding,
            use_tod=config.use_tod,
            use_dow=config.use_dow,
            dropout=config.embedding_dropout,
            fusion=config.embedding_fusion,
            value_embedding_mode=config.value_embedding_mode,
            value_dim=config.value_embedding_dim,
            node_dim=config.node_embedding_dim,
            tod_dim=config.tod_embedding_dim,
            dow_dim=config.dow_embedding_dim,
        )
        self.blocks = nn.ModuleList([build_factorized_block(config, graph)])
        self.head = PredictionHead(
            config.d_model,
            config.input_window,
            config.out_channels,
            head_type=config.head_type,
            norm_eps=config.norm_eps,
        )
        # After the head so a run keeps embedding → block → head → adaptive.
        self.adaptive = (
            AdaptiveEmbedding(input_window=config.input_window, num_nodes=config.num_nodes,
                              d_model=adaptive_dim, fusion=config.adaptive_embedding_fusion)
            if config.use_adaptive_embedding
            else None
        )
        self.projection = None
        if config.conservation_mode == "projected":
            if target_mean is None or target_std is None:
                raise ValueError("projected mode requires the training-only target scaler at initialization")
            self.projection = ConservationProjection(
                config.d_model, config.projection_mlp_hidden_dim, target_mean, target_std,
            )

    def forward(
        self,
        x: torch.Tensor,
        time_of_day: torch.Tensor | None = None,
        day_of_week: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.projection is not None:
            return self.forward_with_aux(x, time_of_day, day_of_week)["final_pred"]
        return self.head(self._encode(x, time_of_day, day_of_week))

    def forward_with_aux(self, x, time_of_day=None, day_of_week=None) -> dict[str, torch.Tensor]:
        """Last-step head, then the conservation projection when it is enabled.

        ``final_pred`` is the tensor ``forward`` returns. ``raw_pred`` is that
        head output before projection. Both are ``[B, N, C_out]``.
        """
        hidden = self._encode(x, time_of_day, day_of_week)
        raw = self.head(hidden)
        if self.projection is None:
            return {"raw_pred": raw, "final_pred": raw}
        return self.projection(hidden, raw)

    def _embed_shared(self, values, time_of_day=None, day_of_week=None) -> torch.Tensor:
        """One embedding call: value MLP, road, TOD, DOW, then adaptive."""
        hidden = self.embedding(values, time_of_day, day_of_week)
        if self.adaptive is not None:
            hidden = self.adaptive(hidden)
        return hidden

    def _encode(self, x, time_of_day=None, day_of_week=None) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must be [B, T, N, C_in], got {tuple(x.shape)}")
        if int(x.shape[1]) != self.config.input_window:
            raise ValueError(f"x has T={x.shape[1]}, model input_window={self.config.input_window}")
        hidden = self._embed_shared(x, time_of_day, day_of_week)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden

    def parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.parameters() if param.requires_grad))

    def architecture_summary(self) -> dict[str, Any]:
        backbone_ffns = [module for module in self.modules() if isinstance(module, SwiGLU)]
        summary = {
            "backbone_mode": "shared",
            "embedding": self.embedding_summary(),
            "head": self.head.describe(),
            "attention_layout": str(self.config.attention_layout),
            "d_model": int(self.config.d_model),
            "temporal_causal_mask": False,
            "independent_ffn_count": len(backbone_ffns),
            "backbone_ffn_count": len(backbone_ffns),
            "ffn_hidden_dims": [int(ffn.w_gate.out_features) for ffn in backbone_ffns],
            "total_parameters": self.parameter_count(),
        }
        if self.projection is not None:
            first, last = self.projection.mlp[0], self.projection.mlp[-1]
            summary["projection_mlp"] = {
                "input_dim": int(first.in_features),
                "hidden_dim": int(first.out_features),
                "output_dim": int(last.out_features),
            }
        summary.update(
            branch_count=1,
            joint_attention_count=0,
            temporal_attention_count=1,
            spatial_attention_count=1,
            temporal_num_heads=int(self.blocks[0].temporal.num_heads),
            spatial_num_heads=int(self.blocks[0].spatial.num_heads),
            temporal_head_dim=int(self.blocks[0].temporal.head_dim),
            spatial_head_dim=int(self.blocks[0].spatial.head_dim),
            attention_inner_dim=int(self.blocks[0].temporal.inner_dim),
            block_order=["T", "S", "FFN"],
        )
        return summary

    def spatial_summary(self) -> dict[str, Any]:
        """Which attention mixes roads, and what restricts its keys."""
        name = self.config.spatial_attention
        if name == "global":
            return {
                "type": "global",
                "branch_count": 1,
                "keys_per_query": int(self.config.num_nodes),
                "graph_mask": False,
                "edge_type_bias": "not_applicable",
                "spatial_direction": "not_applicable",
                "chunk_size": self.config.spatial_chunk_size,
                "checkpoint": bool(self.config.spatial_checkpoint),
            }
        return {
            "type": name,
            "branch_count": 1,
            "graph_mask": True,
            "graph_edges": self.graph_edges,
            "edge_type_bias": bool(self.config.use_edge_type_bias),
            "spatial_direction": self.graph_direction,
            "chunk_size": self.config.spatial_chunk_size,
            "checkpoint": bool(self.config.spatial_checkpoint),
        }

    def embedding_summary(self) -> dict[str, Any]:
        """Fusion mode and the widths actually built (``add`` uses ``d_model`` for every part)."""
        embedding = self.embedding
        return {
            "backbone_mode": "shared",
            "fusion": embedding.fusion,
            "widths": dict(embedding.widths),
            "value_embedding": embedding.value_embedding_summary(),
            "enabled": {
                "value": True,
                "node": embedding.node_embedding is not None,
                "tod": embedding.tod_embedding is not None,
                "dow": embedding.dow_embedding is not None,
            },
            "output_dim": int(self.config.d_model),
            "base_output_dim": embedding.d_model,
            "adaptive": {
                "enabled": self.adaptive is not None,
                "fusion": f"{self.config.adaptive_embedding_fusion}_after_embedding_dropout",
                "shape": None if self.adaptive is None else list(self.adaptive.adaptive_embedding.shape),
                "parameters": 0 if self.adaptive is None else int(self.adaptive.adaptive_embedding.numel()),
            },
        }
