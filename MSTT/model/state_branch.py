"""One traffic-state encoder: value MLP, calendar embeddings, adaptive table, one TSF.

Four ``StateTSFBranch`` modules are constructed separately. They do not share
parameters. The branch has no prediction head and no conservation layer.
"""

from __future__ import annotations

import torch
from torch import nn

from MSTT.model.adaptive_embedding import AdaptiveEmbedding
from MSTT.model.embedding import InputEmbedding
from MSTT.model.transformer_block import build_factorized_block


class StateTSFBranch(nn.Module):
    """Encode one input channel. ``branch_cfg`` already stores per-branch widths."""

    def __init__(self, branch_cfg, graph) -> None:
        super().__init__()
        config = branch_cfg
        self.embedding = InputEmbedding(
            in_channels=1,
            d_model=config.base_embedding_dim,
            num_nodes=config.num_nodes,
            steps_per_day=config.steps_per_day,
            use_node_embedding=config.use_node_embedding,
            use_tod=config.use_tod,
            use_dow=config.use_dow,
            dropout=config.embedding_dropout,
            fusion="concat",
            value_embedding_mode="separate_mlp",
            value_dim=config.value_embedding_dim,
            node_dim=config.node_embedding_dim,
            tod_dim=config.tod_embedding_dim,
            dow_dim=config.dow_embedding_dim,
        )
        self.adaptive = AdaptiveEmbedding(
            input_window=config.input_window,
            num_nodes=config.num_nodes,
            d_model=config.adaptive_embedding_dim,
            fusion="concat",
        )
        self.block = build_factorized_block(config, graph)

    def forward(self, x_one, time_of_day=None, day_of_week=None) -> torch.Tensor:
        hidden = self.embedding(x_one, time_of_day, day_of_week)
        hidden = self.adaptive(hidden)
        return self.block(hidden)

    def summary(self) -> dict[str, object]:
        embedding = self.embedding
        block = self.block
        return {
            "fusion": embedding.fusion,
            "widths": dict(embedding.widths),
            "value_embedding": embedding.value_embedding_summary(),
            "enabled": {
                "value": True,
                "node": embedding.node_embedding is not None,
                "tod": embedding.tod_embedding is not None,
                "dow": embedding.dow_embedding is not None,
            },
            "base_output_dim": int(embedding.d_model),
            "adaptive": {
                "enabled": True,
                "fusion": "concat_after_embedding_dropout",
                "shape": list(self.adaptive.adaptive_embedding.shape),
                "parameters": int(self.adaptive.adaptive_embedding.numel()),
            },
            "d_model": int(block.temporal.d_model),
            "temporal_num_heads": int(block.temporal.num_heads),
            "spatial_num_heads": int(block.spatial.num_heads),
            "temporal_head_dim": int(block.temporal.head_dim),
            "spatial_head_dim": int(block.spatial.head_dim),
            "attention_inner_dim": int(block.temporal.inner_dim),
            "ffn_hidden_dim": int(block.ffn.w_gate.out_features),
        }
