"""PyTorch STFGNN matching ``stsgcn_4n_res.py`` + ``construct_model``."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from reimplementation.stfgnn.model.fusion_graph import (
    FUSION_STEPS,
    construct_adj_fusion,
    mask_init_from_fusion,
    time_length_after_stfgcl,
)
from reimplementation.stfgnn.model.graph_layers import OutputLayer, STFGCL, apply_mxnet_xavier

CODE_VERSION = "0.1.0"


class STFGNN(nn.Module):
    """Public tensors ``[B, T, V, C]``. Output ``[B, 1, V, 1]``."""

    def __init__(
        self,
        spatial_adj: np.ndarray,
        temporal_adj: np.ndarray,
        *,
        seq_len: int = 12,
        num_nodes: int = 56,
        input_channels: int = 1,
        first_layer_embedding_size: int = 64,
        filters: Sequence[Sequence[int]] | None = None,
        module_type: str = "individual",
        activation: str = "GLU",
        use_mask: bool = True,
        temporal_emb: bool = True,
        spatial_emb: bool = True,
        horizon: int = 1,
        output_hidden: int = 128,
        xavier_magnitude: float = 0.0003,
        temporal_graph_is_test_only: bool = False,
    ) -> None:
        super().__init__()
        if filters is None:
            filters = [[64, 64, 64], [64, 64, 64], [64, 64, 64]]
        if module_type != "individual":
            raise ValueError("official STFGNN entry only implements module_type=individual")
        self.seq_len = int(seq_len)
        self.num_nodes = int(num_nodes)
        self.input_channels = int(input_channels)
        self.first_layer_embedding_size = int(first_layer_embedding_size)
        self.filter_list = [[int(channel) for channel in group] for group in filters]
        self.module_type = module_type
        self.activation = str(activation)
        self.use_mask = bool(use_mask)
        self.temporal_emb = bool(temporal_emb)
        self.spatial_emb = bool(spatial_emb)
        self.horizon = int(horizon)
        self.output_hidden = int(output_hidden)
        self.temporal_graph_is_test_only = bool(temporal_graph_is_test_only)
        self.n_stfgcl = len(self.filter_list)
        self.final_time_length = time_length_after_stfgcl(self.seq_len, self.n_stfgcl)
        if self.final_time_length < 1:
            raise ValueError(
                f"seq_len {self.seq_len} is too short for {self.n_stfgcl} STFGCL layers"
            )
        if spatial_adj.shape != (self.num_nodes, self.num_nodes):
            raise ValueError(f"spatial adjacency {spatial_adj.shape} != ({self.num_nodes}, {self.num_nodes})")
        if temporal_adj.shape != spatial_adj.shape:
            raise ValueError(f"temporal adjacency {temporal_adj.shape} != spatial {spatial_adj.shape}")

        localized = construct_adj_fusion(spatial_adj, temporal_adj, steps=FUSION_STEPS)
        self.register_buffer("localized_adj", torch.tensor(localized, dtype=torch.float32))
        if self.use_mask:
            mask = mask_init_from_fusion(localized)
            self.adj_mask = nn.Parameter(torch.tensor(mask, dtype=torch.float32))
        else:
            self.register_parameter("adj_mask", None)

        if self.first_layer_embedding_size > 0:
            self.input_projection = nn.Linear(self.input_channels, self.first_layer_embedding_size)
            layer_channels = self.first_layer_embedding_size
        else:
            self.input_projection = None
            layer_channels = self.input_channels

        layers = []
        time_length = self.seq_len
        channels = layer_channels
        for group in self.filter_list:
            layers.append(
                STFGCL(
                    time_length,
                    self.num_nodes,
                    channels,
                    group,
                    activation=self.activation,
                    temporal_emb=temporal_emb,
                    spatial_emb=spatial_emb,
                )
            )
            time_length -= 3
            channels = group[-1]
        self.stfgcl_layers = nn.ModuleList(layers)
        self.output_heads = nn.ModuleList(
            [
                OutputLayer(
                    self.final_time_length,
                    self.num_nodes,
                    channels,
                    hidden_channels=output_hidden,
                    predict_length=1,
                )
                for _ in range(self.horizon)
            ]
        )
        apply_mxnet_xavier(self, magnitude=xavier_magnitude, skip=("adj_mask",))
        if self.adj_mask is not None:
            with torch.no_grad():
                self.adj_mask.copy_(torch.tensor(mask, dtype=torch.float32))

    def effective_adjacency(self) -> torch.Tensor:
        if self.adj_mask is None:
            return self.localized_adj
        return self.adj_mask * self.localized_adj

    def parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.parameters()))

    def forward(self, data: torch.Tensor, return_trace: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        if data.ndim != 4:
            raise ValueError(f"input must be [B, T, V, C], got {tuple(data.shape)}")
        if data.shape[1] != self.seq_len or data.shape[2] != self.num_nodes:
            raise ValueError(
                f"input {tuple(data.shape)} != [B, {self.seq_len}, {self.num_nodes}, C]"
            )
        current = data
        if self.input_projection is not None:
            current = F.relu(self.input_projection(current))
        adj = self.effective_adjacency()
        time_lengths = [int(current.size(1))]
        for layer in self.stfgcl_layers:
            current = layer(adj, current)
            time_lengths.append(int(current.size(1)))
        heads = [head(current) for head in self.output_heads]
        pred = torch.cat(heads, dim=1).unsqueeze(-1)
        if not return_trace:
            return pred
        trace = {
            "time_lengths": time_lengths,
            "final_time_length": int(current.size(1)),
            "expected_final_time_length": self.final_time_length,
            "n_stfgcl": self.n_stfgcl,
            "horizon": self.horizon,
            "temporal_graph_is_test_only": self.temporal_graph_is_test_only,
            "localized_adj_shape": list(self.localized_adj.shape),
            "input_projection_applied": self.input_projection is not None,
        }
        return pred, trace
