"""Original STID layers: history flatten and residual 1x1 MLP.

Source: ``reference/STID/stid/arch/stid_arch.py`` and ``mlp.py``.
Tensors after flatten use Conv2d layout ``[B, C, N, 1]`` (N is the node/width axis).
Public history remains ``[B, T, N, C]``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from reimplementation.common.errors import ReimplementationError

ORIGINAL_MLP_DROPOUT = 0.15
GRAPH_FORWARD_KEYS = frozenset(
    {
        "adjacency",
        "adj",
        "adj_mx",
        "supports",
        "graph",
        "laplacian",
        "cheb",
        "gso",
        "dtw",
        "hop",
        "mask",
    }
)


def refuse_graph_forward_kwargs(kwargs: dict[str, Any]) -> None:
    hit = sorted(GRAPH_FORWARD_KEYS.intersection(kwargs))
    if hit:
        raise ReimplementationError(
            f"STID forward does not accept graph inputs {hit}; the original model has no GCN"
        )


def flatten_history_for_conv2d(
    history: torch.Tensor,
    *,
    input_len: int,
    input_dim: int,
) -> torch.Tensor:
    """Flatten each node's history for the original 1x1 Conv2d.

    Steps, matching ``stid_arch.py``::

        [B, T, N, C]
        -> transpose(1, 2) -> [B, N, T, C]
        -> contiguous()
        -> view(B, N, T * C)   # C is fastest, then T from old to new
        -> transpose(1, 2) -> [B, T * C, N]
        -> unsqueeze(-1)   -> [B, T * C, N, 1]

    Each node is flattened independently. Batches and node order are preserved.
    Time is not reversed. Nodes are not concatenated into the channel axis.
    """
    if history.ndim != 4:
        raise ReimplementationError(f"history must be [B,T,N,C], got {tuple(history.shape)}")
    batch, time, nodes, channels = history.shape
    if int(time) != int(input_len):
        raise ReimplementationError(
            f"history time {time} != input_len {input_len}; refusing to reshape"
        )
    if int(channels) != int(input_dim):
        raise ReimplementationError(
            f"history channels {channels} != input_dim {input_dim}; refusing to reshape"
        )
    if batch < 1 or nodes < 1 or time < 1:
        raise ReimplementationError(f"illegal history shape {tuple(history.shape)}")
    # Original: input_data.transpose(1, 2).contiguous().view(B, N, -1).transpose(1, 2).unsqueeze(-1)
    node_major = history.transpose(1, 2).contiguous()
    if tuple(node_major.shape) != (batch, nodes, time, channels):
        raise ReimplementationError(
            f"after T/N transpose expected {(batch, nodes, time, channels)}, got {tuple(node_major.shape)}"
        )
    flat_nodes = node_major.view(batch, nodes, time * channels)
    conv_in = flat_nodes.transpose(1, 2).unsqueeze(-1)
    expected = (batch, time * channels, nodes, 1)
    if tuple(conv_in.shape) != expected:
        raise ReimplementationError(f"Conv2d input {tuple(conv_in.shape)} != {expected}")
    return conv_in


def compute_hidden_dim(
    *,
    embed_dim: int,
    node_dim: int,
    temp_dim_tid: int,
    temp_dim_diw: int,
    if_spatial: bool,
    if_time_in_day: bool,
    if_day_in_week: bool,
) -> int:
    """Sum of enabled embedding widths. Do not hard-code 128."""
    return (
        int(embed_dim)
        + int(node_dim) * int(bool(if_spatial))
        + int(temp_dim_tid) * int(bool(if_time_in_day))
        + int(temp_dim_diw) * int(bool(if_day_in_week))
    )


def identity_channel_slices(
    *,
    embed_dim: int,
    node_dim: int,
    temp_dim_tid: int,
    temp_dim_diw: int,
    if_spatial: bool,
    if_time_in_day: bool,
    if_day_in_week: bool,
) -> dict[str, slice]:
    """Channel slices on dim=1 after original concat order.

    Order: time-series, spatial, time-of-day, day-of-week.
    """
    offset = 0
    slices: dict[str, slice] = {"time_series": slice(offset, offset + int(embed_dim))}
    offset += int(embed_dim)
    if if_spatial:
        slices["spatial"] = slice(offset, offset + int(node_dim))
        offset += int(node_dim)
    if if_time_in_day:
        slices["time_of_day"] = slice(offset, offset + int(temp_dim_tid))
        offset += int(temp_dim_tid)
    if if_day_in_week:
        slices["day_of_week"] = slice(offset, offset + int(temp_dim_diw))
        offset += int(temp_dim_diw)
    return slices


class MultiLayerPerceptron(nn.Module):
    """Residual 1x1 MLP from ``reference/STID/stid/arch/mlp.py``.

    Original order::

        hidden = fc2(dropout(ReLU(fc1(x)))) + x

    Both layers are ``Conv2d(k=1)`` with bias. Dropout defaults to the original
    hardcoded ``p=0.15``. There is no BatchNorm, LayerNorm, attention, or
    residual scaling. ``input_dim`` must equal ``hidden_dim`` so the residual
    add is the original unprojected sum.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        dropout: float = ORIGINAL_MLP_DROPOUT,
    ) -> None:
        super().__init__()
        if int(input_dim) != int(hidden_dim):
            raise ReimplementationError(
                "original STID MLP residual requires input_dim == hidden_dim; "
                "refusing a projection residual"
            )
        if dropout < 0.0 or dropout > 1.0:
            raise ReimplementationError(f"dropout must be in [0, 1], got {dropout}")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.fc1 = nn.Conv2d(
            in_channels=int(input_dim),
            out_channels=int(hidden_dim),
            kernel_size=(1, 1),
            bias=True,
        )
        self.fc2 = nn.Conv2d(
            in_channels=int(hidden_dim),
            out_channels=int(hidden_dim),
            kernel_size=(1, 1),
            bias=True,
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=float(dropout))

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        if input_data.ndim != 4:
            raise ReimplementationError(
                f"MLP expects Conv2d layout [B, D, N, 1], got {tuple(input_data.shape)}"
            )
        if int(input_data.size(1)) != self.input_dim:
            raise ReimplementationError(
                f"MLP channel {int(input_data.size(1))} != input_dim {self.input_dim}"
            )
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))
        return hidden + input_data
