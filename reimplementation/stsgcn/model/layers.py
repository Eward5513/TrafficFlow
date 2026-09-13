"""PyTorch layers matching ``reference/STSGCN/models/stsgcn.py``.

GCN/GLU follows the original operator, not the two-branch paper shorthand::

    Y = A_ST @ X                         # (3N, B, C)
    P, Q = split(Linear(Y, 2C'), dim=-1)
    out = P * sigmoid(Q)
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from reimplementation.stsgcn.model.graph import flatten_time_major_window


def mxnet_xavier_uniform_(tensor: torch.Tensor, magnitude: float = 0.0003) -> torch.Tensor:
    """MXNet 1.4 ``Xavier(rnd_type='uniform', factor_type='avg', magnitude)``.

    ``scale = sqrt(magnitude / ((fan_in + fan_out) / 2))`` with conv-style
    ``hw_scale = prod(shape[2:])`` when ``ndim > 2``. Weight layout is
    ``(fan_out, fan_in, ...)``, matching both MXNet FullyConnected and
    ``nn.Linear``.
    """
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
    """Original ``position_embedding``: broadcast-add per-layer T/N embeddings.

    Added to the current STSGCL input *before* the length-3 sliding window.
    Shapes match original Variables::

        temporal: (1, T, 1, C)
        spatial:  (1, 1, N, C)
    """

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
        self.use_temporal = bool(temporal)
        self.use_spatial = bool(spatial)
        if self.use_temporal:
            self.temporal_emb = nn.Parameter(torch.empty(1, self.time_length, 1, self.channels))
        else:
            self.register_parameter("temporal_emb", None)
        if self.use_spatial:
            self.spatial_emb = nn.Parameter(torch.empty(1, 1, self.num_nodes, self.channels))
        else:
            self.register_parameter("spatial_emb", None)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data: (B, T, N, C)
        if data.shape[1] != self.time_length:
            raise ValueError(
                f"temporal embedding length {self.time_length} != input T {data.shape[1]}"
            )
        if data.shape[2] != self.num_nodes:
            raise ValueError(f"spatial embedding N {self.num_nodes} != input N {data.shape[2]}")
        out = data
        if self.temporal_emb is not None:
            out = out + self.temporal_emb
        if self.spatial_emb is not None:
            out = out + self.spatial_emb
        return out


class STSGCNGraphConv(nn.Module):
    """Original ``gcn_operation``. Input/output layout is ``(3N, B, C)``."""

    def __init__(self, in_channels: int, out_channels: int, activation: str) -> None:
        super().__init__()
        if activation not in {"GLU", "relu"}:
            raise ValueError(f"activation must be GLU or relu, got {activation!r}")
        self.activation = activation
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        hidden = 2 * self.out_channels if activation == "GLU" else self.out_channels
        self.linear = nn.Linear(self.in_channels, hidden)

    def forward(self, adj: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        # adj: (3N, 3N), data: (3N, B, C)
        mixed = torch.einsum("ij,jbc->ibc", adj, data)
        projected = self.linear(mixed)
        if self.activation == "GLU":
            lhs, rhs = torch.chunk(projected, 2, dim=-1)
            return lhs * torch.sigmoid(rhs)
        return F.relu(projected)


class STSGCM(nn.Module):
    """Original ``stsgcm``: stacked GCN, crop center time, element-wise max."""

    def __init__(
        self,
        in_channels: int,
        filters: Sequence[int],
        num_nodes: int,
        activation: str,
    ) -> None:
        super().__init__()
        if not filters:
            raise ValueError("STSGCM filters must be non-empty")
        self.num_nodes = int(num_nodes)
        self.filters = [int(item) for item in filters]
        self.activation = activation
        layers = []
        channels = int(in_channels)
        for out_channels in self.filters:
            layers.append(STSGCNGraphConv(channels, out_channels, activation))
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


class STSGCL(nn.Module):
    """Original ``stsgcl`` / ``sthgcn_layer_{individual,sharing}``.

    Sliding windows of length 3 and stride 1. Output time length is ``T - 2``.
    """

    def __init__(
        self,
        time_length: int,
        num_nodes: int,
        in_channels: int,
        filters: Sequence[int],
        *,
        module_type: str,
        activation: str,
        temporal_emb: bool = True,
        spatial_emb: bool = True,
    ) -> None:
        super().__init__()
        if module_type not in {"individual", "sharing"}:
            raise ValueError(f"module_type must be individual or sharing, got {module_type!r}")
        self.module_type = module_type
        self.time_length = int(time_length)
        self.num_nodes = int(num_nodes)
        self.in_channels = int(in_channels)
        self.filters = [int(item) for item in filters]
        self.n_windows = self.time_length - 2
        if self.n_windows < 1:
            raise ValueError(f"STSGCL time_length {time_length} must be >= 3")
        self.embedding = PositionEmbedding(
            self.time_length,
            self.num_nodes,
            self.in_channels,
            temporal=temporal_emb,
            spatial=spatial_emb,
        )
        if module_type == "individual":
            self.stsgcm_windows = nn.ModuleList(
                [
                    STSGCM(self.in_channels, self.filters, self.num_nodes, activation)
                    for _ in range(self.n_windows)
                ]
            )
            self.shared_stsgcm = None
        else:
            self.stsgcm_windows = None
            self.shared_stsgcm = STSGCM(self.in_channels, self.filters, self.num_nodes, activation)

    def _window_nodes(self, embedded: torch.Tensor, start: int) -> torch.Tensor:
        window = embedded[:, start : start + 3]
        return flatten_time_major_window(window.contiguous())

    def forward(self, adj: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(data)
        batch = int(embedded.size(0))
        if self.module_type == "individual":
            outputs = []
            for index, stsgcm in enumerate(self.stsgcm_windows):
                localized = self._window_nodes(embedded, index)
                center = stsgcm(adj, localized)
                outputs.append(center.transpose(0, 1).unsqueeze(1))
            return torch.cat(outputs, dim=1)
        windows = [self._window_nodes(embedded, index) for index in range(self.n_windows)]
        concatenated = torch.cat(windows, dim=1)
        center = self.shared_stsgcm(adj, concatenated)
        channels = int(self.filters[-1])
        reshaped = center.reshape(self.num_nodes, self.n_windows, batch, channels)
        return reshaped.permute(2, 1, 0, 3).contiguous()


class OutputLayer(nn.Module):
    """Original ``output_layer`` with ``predict_length`` (horizon) as the last FC size.

    For the R-only task we use ``predict_length=1`` and keep the two-FC body.
    """

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
        self.in_channels = int(in_channels)
        self.predict_length = int(predict_length)
        self.fc1 = nn.Linear(self.time_length * self.in_channels, int(hidden_channels))
        self.fc2 = nn.Linear(int(hidden_channels), self.predict_length)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data: (B, T, N, C) -> (B, predict_length, N)
        swapped = data.transpose(1, 2)
        batch, n_nodes, time_length, channels = swapped.shape
        flat = swapped.reshape(batch, n_nodes, time_length * channels)
        hidden = F.relu(self.fc1(flat))
        pred = self.fc2(hidden)
        return pred.transpose(1, 2)
