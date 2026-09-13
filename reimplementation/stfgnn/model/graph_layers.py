"""Graph layers matching ``reference/STFGNN/models/stsgcn_4n_res.py``.

GCN/GLU (4N nodes)::

    Y = A_STFG @ X
    P, Q = split(Linear(Y, 2C'), last dim)
    out = P * sigmoid(Q)
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from reimplementation.stfgnn.model.fusion_graph import flatten_time_major_window
from reimplementation.stfgnn.model.temporal_conv import GatedTemporalConv


def mxnet_xavier_uniform_(tensor: torch.Tensor, magnitude: float = 0.0003) -> torch.Tensor:
    if tensor.ndim < 2:
        raise ValueError(f"Xavier requires at least 2D, got {tuple(tensor.shape)}")
    shape = tensor.shape
    hw_scale = 1.0
    if tensor.ndim > 2:
        hw_scale = float(torch.prod(torch.tensor(shape[2:], dtype=torch.float64)))
    fan_in = float(shape[1]) * hw_scale
    fan_out = float(shape[0]) * hw_scale
    factor = (fan_in + fan_out) / 2.0
    scale = (magnitude / factor) ** 0.5
    with torch.no_grad():
        tensor.uniform_(-scale, scale)
    return tensor


def apply_mxnet_xavier(module: nn.Module, magnitude: float = 0.0003, skip: Sequence[str] = ()) -> None:
    skip_names = set(skip)
    for name, param in module.named_parameters():
        if name in skip_names or any(name.endswith(item) for item in skip_names):
            continue
        if param.ndim >= 2:
            mxnet_xavier_uniform_(param, magnitude=magnitude)
        else:
            nn.init.zeros_(param)


class PositionEmbedding(nn.Module):
    def __init__(
        self,
        time_length: int,
        num_nodes: int,
        channels: int,
        *,
        temporal: bool = True,
        spatial: bool = True,
    ) -> None:
        super().__init__()
        self.time_length = int(time_length)
        self.num_nodes = int(num_nodes)
        self.channels = int(channels)
        if temporal:
            self.temporal_emb = nn.Parameter(torch.empty(1, self.time_length, 1, self.channels))
        else:
            self.register_parameter("temporal_emb", None)
        if spatial:
            self.spatial_emb = nn.Parameter(torch.empty(1, 1, self.num_nodes, self.channels))
        else:
            self.register_parameter("spatial_emb", None)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if data.shape[1] != self.time_length:
            raise ValueError(f"temporal embedding T {self.time_length} != input T {data.shape[1]}")
        if data.shape[2] != self.num_nodes:
            raise ValueError(f"spatial embedding N {self.num_nodes} != input N {data.shape[2]}")
        out = data
        if self.temporal_emb is not None:
            out = out + self.temporal_emb
        if self.spatial_emb is not None:
            out = out + self.spatial_emb
        return out


class STFGNNGraphConv(nn.Module):
    """Original ``gcn_operation`` on ``(4N, B, C)``."""

    def __init__(self, in_channels: int, out_channels: int, activation: str) -> None:
        super().__init__()
        if activation not in {"GLU", "relu"}:
            raise ValueError(f"activation must be GLU or relu, got {activation!r}")
        self.activation = activation
        self.out_channels = int(out_channels)
        hidden = 2 * self.out_channels if activation == "GLU" else self.out_channels
        self.linear = nn.Linear(int(in_channels), hidden)

    def forward(self, adj: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        mixed = torch.einsum("ij,jbc->ibc", adj, data)
        projected = self.linear(mixed)
        if self.activation == "GLU":
            lhs, rhs = torch.chunk(projected, 2, dim=-1)
            return lhs * torch.sigmoid(rhs)
        return F.relu(projected)


class STFGCM(nn.Module):
    """Stacked GCN, crop nodes ``[N:2N]`` of the 4N window, element-wise max."""

    def __init__(
        self,
        in_channels: int,
        filters: Sequence[int],
        num_nodes: int,
        activation: str,
    ) -> None:
        super().__init__()
        if not filters:
            raise ValueError("STFGCM filters must be non-empty")
        self.num_nodes = int(num_nodes)
        self.filters = [int(item) for item in filters]
        layers = []
        channels = int(in_channels)
        for out_channels in self.filters:
            layers.append(STFGNNGraphConv(channels, out_channels, activation))
            channels = out_channels
        self.gcn_layers = nn.ModuleList(layers)

    def forward(self, adj: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        center_slices = []
        current = data
        for gcn in self.gcn_layers:
            current = gcn(adj, current)
            center_slices.append(current[self.num_nodes : 2 * self.num_nodes])
        stacked = torch.stack(center_slices, dim=0)
        return stacked.max(dim=0).values


class STFGCL(nn.Module):
    """Original ``sthgcn_layer_individual``: gated CNN residual + 4-step STFGCM windows."""

    def __init__(
        self,
        time_length: int,
        num_nodes: int,
        in_channels: int,
        filters: Sequence[int],
        *,
        activation: str,
        temporal_emb: bool = True,
        spatial_emb: bool = True,
    ) -> None:
        super().__init__()
        self.time_length = int(time_length)
        self.num_nodes = int(num_nodes)
        self.in_channels = int(in_channels)
        self.filters = [int(item) for item in filters]
        if self.filters[-1] != self.in_channels:
            raise ValueError(
                "original residual adds gated-CNN features of size C to STFGCM output C'; "
                f"C={self.in_channels} vs C'={self.filters[-1]}"
            )
        self.n_windows = self.time_length - 3
        if self.n_windows < 1:
            raise ValueError(f"STFGCL time_length {time_length} must be >= 4")
        self.embedding = PositionEmbedding(
            self.time_length,
            self.num_nodes,
            self.in_channels,
            temporal=temporal_emb,
            spatial=spatial_emb,
        )
        self.gated_cnn = GatedTemporalConv(self.in_channels)
        self.stfgcm_windows = nn.ModuleList(
            [
                STFGCM(self.in_channels, self.filters, self.num_nodes, activation)
                for _ in range(self.n_windows)
            ]
        )

    def _window_nodes(self, embedded: torch.Tensor, start: int) -> torch.Tensor:
        window = embedded[:, start : start + 4]
        return flatten_time_major_window(window.contiguous())

    def forward(self, adj: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(data)
        residual = self.gated_cnn(embedded)
        outputs = []
        for index, stfgcm in enumerate(self.stfgcm_windows):
            localized = self._window_nodes(embedded, index)
            center = stfgcm(adj, localized)
            outputs.append(center.transpose(0, 1).unsqueeze(1))
        graph_out = torch.cat(outputs, dim=1)
        if graph_out.shape != residual.shape:
            raise ValueError(
                f"fusion residual shape {tuple(residual.shape)} != graph {tuple(graph_out.shape)}"
            )
        return graph_out + residual


class OutputLayer(nn.Module):
    def __init__(
        self,
        time_length: int,
        num_nodes: int,
        in_channels: int,
        *,
        hidden_channels: int = 128,
        predict_length: int = 1,
    ) -> None:
        super().__init__()
        self.time_length = int(time_length)
        self.num_nodes = int(num_nodes)
        self.fc1 = nn.Linear(self.time_length * int(in_channels), int(hidden_channels))
        self.fc2 = nn.Linear(int(hidden_channels), int(predict_length))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        swapped = data.transpose(1, 2)
        batch, n_nodes, time_length, channels = swapped.shape
        flat = swapped.reshape(batch, n_nodes, time_length * channels)
        hidden = F.relu(self.fc1(flat))
        pred = self.fc2(hidden)
        return pred.transpose(1, 2)
