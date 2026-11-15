"""One traffic-state encoder: value MLP, calendar features, adaptive features, one TSF.

``external_context=False`` owns the TOD, DOW, and adaptive tables. Four of
those branches do not share parameters. ``external_context=True`` keeps the
value MLP, node table, dropout, and TSF, and reads TOD, DOW, and adaptive
tensors supplied by the caller. The branch does not store those parameters.
"""

from __future__ import annotations

import torch
from torch import nn

from MSTT.model.adaptive_embedding import AdaptiveEmbedding
from MSTT.model.embedding import InputEmbedding
from MSTT.model.transformer_block import build_factorized_block


def _context_tensors(context) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(context, (tuple, list)):
        if len(context) != 3:
            raise ValueError(f"context must be (tod, dow, adaptive), got {len(context)} items")
        tod, dow, adaptive = context
    elif all(hasattr(context, name) for name in ("tod", "dow", "adaptive")):
        tod, dow, adaptive = context.tod, context.dow, context.adaptive
    else:
        raise TypeError("context must be a (tod, dow, adaptive) tuple or a StateContext")
    if not all(torch.is_tensor(item) for item in (tod, dow, adaptive)):
        raise TypeError("context tod, dow, and adaptive must be tensors")
    return tod, dow, adaptive


class StateTSFBranch(nn.Module):
    """Encode one input channel. ``branch_cfg`` already stores per-branch widths."""

    def __init__(self, branch_cfg, graph, *, external_context: bool = False) -> None:
        super().__init__()
        config = branch_cfg
        self.external_context = bool(external_context)
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
            external_calendar_embeddings=self.external_context,
        )
        self.context_adaptive_shape = (
            int(config.input_window),
            int(config.num_nodes),
            int(config.adaptive_embedding_dim),
        )
        if self.external_context:
            self.adaptive = None
        else:
            self.adaptive = AdaptiveEmbedding(
                input_window=config.input_window,
                num_nodes=config.num_nodes,
                d_model=config.adaptive_embedding_dim,
                fusion="concat",
            )
        self.block = build_factorized_block(config, graph)

    def forward(self, x_one, time_of_day=None, day_of_week=None, *, context=None) -> torch.Tensor:
        if self.external_context:
            if context is None:
                raise ValueError("external_context branch requires context=(tod, dow, adaptive)")
            tod_features, dow_features, adaptive = _context_tensors(context)
            hidden = self.embedding(
                x_one,
                time_of_day,
                day_of_week,
                tod_features=tod_features,
                dow_features=dow_features,
            )
            hidden = self._append_adaptive(hidden, adaptive)
        else:
            if context is not None:
                raise ValueError("this state branch owns its TOD/DOW/adaptive tables and rejects external context")
            hidden = self.embedding(x_one, time_of_day, day_of_week)
            hidden = self.adaptive(hidden)
        return self.block(hidden)

    def _append_adaptive(self, hidden: torch.Tensor, adaptive: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, _width = hidden.shape
        expected = (1, steps, nodes, self.context_adaptive_shape[-1])
        if tuple(adaptive.shape) != expected:
            raise ValueError(f"adaptive context must be [1, T, N, D]={expected}, got {tuple(adaptive.shape)}")
        if adaptive.device != hidden.device:
            raise ValueError(f"adaptive context is on {adaptive.device}, hidden is on {hidden.device}")
        if not adaptive.is_floating_point() or adaptive.dtype != hidden.dtype:
            raise ValueError(f"adaptive context must be {hidden.dtype}, got {adaptive.dtype}")
        expanded = adaptive.expand(batch, steps, nodes, expected[-1])
        return torch.cat((hidden, expanded), dim=-1)

    def summary(self) -> dict[str, object]:
        embedding = self.embedding
        block = self.block
        calendar = embedding.calendar_summary()
        if self.external_context:
            adaptive = {
                "enabled": True,
                "source": "external",
                "fusion": "concat_after_embedding_dropout",
                "shape": list(self.context_adaptive_shape),
                "parameters": 0,
            }
        else:
            adaptive = {
                "enabled": True,
                "source": "parameter",
                "fusion": "concat_after_embedding_dropout",
                "shape": [int(size) for size in self.adaptive.adaptive_embedding.shape],
                "parameters": int(self.adaptive.adaptive_embedding.numel()),
            }
        return {
            "fusion": embedding.fusion,
            "widths": dict(embedding.widths),
            "value_embedding": embedding.value_embedding_summary(),
            "enabled": {
                "value": True,
                "node": embedding.node_embedding is not None,
                "tod": bool(calendar["tod"]["enabled"]),
                "dow": bool(calendar["dow"]["enabled"]),
            },
            "calendar": calendar,
            "base_output_dim": int(embedding.d_model),
            "adaptive": adaptive,
            "external_context": self.external_context,
            "d_model": int(block.temporal.d_model),
            "temporal_num_heads": int(block.temporal.num_heads),
            "spatial_num_heads": int(block.spatial.num_heads),
            "temporal_head_dim": int(block.temporal.head_dim),
            "spatial_head_dim": int(block.spatial.head_dim),
            "attention_inner_dim": int(block.temporal.inner_dim),
            "ffn_hidden_dim": int(block.ffn.w_gate.out_features),
        }
