"""STID architecture ported from ``reference/STID/stid/arch/stid_arch.py``.

Public history: ``[B, T, N, C]``. Prediction: ``[B, output_len, N, 1]``.
This is not a graph model. Adjacency is neither an argument nor a weight.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from reimplementation.common.errors import ReimplementationError
from reimplementation.stid.model.layers import (
    GRAPH_FORWARD_KEYS,
    MultiLayerPerceptron,
    compute_hidden_dim,
    flatten_history_for_conv2d,
    identity_channel_slices,
    refuse_graph_forward_kwargs,
)
from reimplementation.stid.temporal_identity import long_index_from_fraction

CODE_VERSION = "0.1.0"
GRAPH_STATE_FRAGMENTS = (
    "adj",
    "adjacency",
    "laplacian",
    "cheb",
    "gcn",
    "gat",
    "diffusion",
    "supports",
    "dtw",
    "hop_matrix",
    "geo_mask",
    "sem_mask",
)


class STID(nn.Module):
    """Spatial-Temporal Identity (Shao et al., CIKM 2022).

    Original constructor keys: ``if_node``, ``if_T_i_D``, ``if_D_i_W``,
    ``embed_dim``, ``num_layer``. Aliases ``if_spatial``, ``if_time_in_day``,
    ``if_day_in_week``, ``time_series_emb_dim``, ``num_block`` are accepted
    with the same meaning.
    """

    def __init__(
        self,
        *,
        num_nodes: int,
        input_len: int,
        input_dim: int,
        embed_dim: int,
        output_len: int,
        num_layer: int,
        if_node: bool,
        node_dim: int,
        if_T_i_D: bool,
        if_D_i_W: bool,
        temp_dim_tid: int,
        temp_dim_diw: int,
        time_of_day_size: int,
        day_of_week_size: int,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if int(num_nodes) < 1:
            raise ReimplementationError("num_nodes must be positive")
        if int(input_len) < 1 or int(output_len) < 1:
            raise ReimplementationError("input_len and output_len must be positive")
        if int(input_dim) < 1:
            raise ReimplementationError("input_dim must be positive")
        if int(num_layer) < 1:
            raise ReimplementationError("num_layer must be positive")
        if int(time_of_day_size) < 1 or int(day_of_week_size) < 1:
            raise ReimplementationError("identity table sizes must be positive")
        self.num_nodes = int(num_nodes)
        self.node_dim = int(node_dim)
        self.input_len = int(input_len)
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.output_len = int(output_len)
        self.num_layer = int(num_layer)
        self.temp_dim_tid = int(temp_dim_tid)
        self.temp_dim_diw = int(temp_dim_diw)
        self.time_of_day_size = int(time_of_day_size)
        self.day_of_week_size = int(day_of_week_size)
        self.if_spatial = bool(if_node)
        self.if_time_in_day = bool(if_T_i_D)
        self.if_day_in_week = bool(if_D_i_W)
        self.dropout = float(dropout)

        if self.if_spatial:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid)
            )
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw)
            )
            nn.init.xavier_uniform_(self.day_in_week_emb)

        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len,
            out_channels=self.embed_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.hidden_dim = compute_hidden_dim(
            embed_dim=self.embed_dim,
            node_dim=self.node_dim,
            temp_dim_tid=self.temp_dim_tid,
            temp_dim_diw=self.temp_dim_diw,
            if_spatial=self.if_spatial,
            if_time_in_day=self.if_time_in_day,
            if_day_in_week=self.if_day_in_week,
        )
        self.encoder = nn.Sequential(
            *[
                MultiLayerPerceptron(self.hidden_dim, self.hidden_dim, dropout=self.dropout)
                for _ in range(self.num_layer)
            ]
        )
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.output_len,
            kernel_size=(1, 1),
            bias=True,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "STID":
        if_node = bool(config.get("if_node", config.get("if_spatial", True)))
        if_tid = bool(config.get("if_T_i_D", config.get("if_time_in_day", True)))
        if_diw = bool(config.get("if_D_i_W", config.get("if_day_in_week", True)))
        return cls(
            num_nodes=int(config["num_nodes"]),
            input_len=int(config.get("input_len", config.get("n_his", config.get("input_window", 12)))),
            input_dim=int(config.get("input_dim", 3)),
            embed_dim=int(config.get("embed_dim", config.get("time_series_emb_dim", 32))),
            output_len=int(config.get("output_len", config.get("output_window", 1))),
            num_layer=int(config.get("num_layer", config.get("num_block", 3))),
            if_node=if_node,
            node_dim=int(config.get("node_dim", 32)),
            if_T_i_D=if_tid,
            if_D_i_W=if_diw,
            temp_dim_tid=int(config.get("temp_dim_tid", 32)),
            temp_dim_diw=int(config.get("temp_dim_diw", 32)),
            time_of_day_size=int(config.get("time_of_day_size", 288)),
            day_of_week_size=int(config.get("day_of_week_size", 7)),
            dropout=float(config.get("dropout", 0.15)),
        )

    def identity_slices(self) -> dict[str, slice]:
        return identity_channel_slices(
            embed_dim=self.embed_dim,
            node_dim=self.node_dim,
            temp_dim_tid=self.temp_dim_tid,
            temp_dim_diw=self.temp_dim_diw,
            if_spatial=self.if_spatial,
            if_time_in_day=self.if_time_in_day,
            if_day_in_week=self.if_day_in_week,
        )

    def parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.parameters()))

    def expected_parameter_count(self) -> int:
        """Closed-form count of the original Parameter + Conv2d weights/biases."""
        count = 0
        if self.if_spatial:
            count += self.num_nodes * self.node_dim
        if self.if_time_in_day:
            count += self.time_of_day_size * self.temp_dim_tid
        if self.if_day_in_week:
            count += self.day_of_week_size * self.temp_dim_diw
        ts_in = self.input_dim * self.input_len
        count += ts_in * self.embed_dim + self.embed_dim
        mlp = 2 * (self.hidden_dim * self.hidden_dim + self.hidden_dim)
        count += self.num_layer * mlp
        count += self.hidden_dim * self.output_len + self.output_len
        return int(count)

    def _expand_table_index(
        self,
        index: torch.Tensor,
        batch: int,
        size: int,
        name: str,
    ) -> torch.Tensor:
        idx = index.to(dtype=torch.long)
        if idx.ndim == 1:
            if int(idx.size(0)) != batch:
                raise ReimplementationError(f"{name} batch {tuple(idx.shape)} != {batch}")
            idx = idx.view(batch, 1).expand(batch, self.num_nodes)
        if idx.ndim != 2 or int(idx.size(0)) != batch or int(idx.size(1)) != self.num_nodes:
            raise ReimplementationError(f"{name} must be [B] or [B, N], got {tuple(idx.shape)}")
        if bool((idx < 0).any() or (idx >= size).any()):
            raise ReimplementationError(f"{name} index is outside [0, {size})")
        return idx.contiguous()

    def _lookup_identity(
        self,
        table: torch.Tensor,
        fraction: torch.Tensor | None,
        size: int,
        *,
        name: str,
        integer_index: torch.Tensor | None,
        batch: int,
    ) -> torch.Tensor:
        if integer_index is not None:
            index = self._expand_table_index(integer_index, batch, size, name)
        else:
            if fraction is None:
                raise ReimplementationError(f"{name} needs a history channel or an integer index")
            if fraction.ndim != 2:
                raise ReimplementationError(f"{name} last-step feature must be [B, N], got {tuple(fraction.shape)}")
            if int(fraction.size(1)) != self.num_nodes:
                raise ReimplementationError(
                    f"{name} node axis {int(fraction.size(1))} != num_nodes {self.num_nodes}"
                )
            # Original: (frac * size).type(torch.LongTensor) truncation.
            index = long_index_from_fraction(fraction, size)
            if bool((index < 0).any() or (index >= size).any()):
                raise ReimplementationError(f"{name} index is outside [0, {size})")
        return table[index.to(device=table.device)]

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor | None = None,
        batch_seen: int | None = None,
        epoch: int | None = None,
        train: bool | None = None,
        time_of_day_index: torch.Tensor | None = None,
        day_of_week_index: torch.Tensor | None = None,
        *,
        return_trace: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        del future_data, batch_seen, epoch, train
        refuse_graph_forward_kwargs(kwargs)
        extra = set(kwargs) - GRAPH_FORWARD_KEYS
        if extra:
            raise ReimplementationError(f"unexpected STID forward kwargs {sorted(extra)}")
        if history_data.ndim != 4:
            raise ReimplementationError(f"history_data must be [B,T,N,C], got {tuple(history_data.shape)}")
        batch, time, nodes, channels = history_data.shape
        if int(time) != self.input_len:
            raise ReimplementationError(f"T={time} != input_len={self.input_len}")
        if int(nodes) != self.num_nodes:
            raise ReimplementationError(f"N={nodes} != num_nodes={self.num_nodes}")
        needed = self.input_dim
        if self.if_time_in_day and time_of_day_index is None:
            needed = max(needed, 2)
        if self.if_day_in_week and day_of_week_index is None:
            needed = max(needed, 3)
        if int(channels) < needed:
            raise ReimplementationError(
                f"history channels {channels} < required {needed}"
            )
        if not torch.isfinite(history_data).all():
            raise ReimplementationError("history_data contains NaN or inf")

        input_data = history_data[..., : self.input_dim]
        time_in_day_emb = None
        day_in_week_emb = None
        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1] if history_data.size(-1) > 1 else None
            time_in_day_emb = self._lookup_identity(
                self.time_in_day_emb,
                None if t_i_d_data is None else t_i_d_data[:, -1, :],
                self.time_of_day_size,
                name="time_of_day",
                integer_index=time_of_day_index,
                batch=batch,
            )
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2] if history_data.size(-1) > 2 else None
            day_in_week_emb = self._lookup_identity(
                self.day_in_week_emb,
                None if d_i_w_data is None else d_i_w_data[:, -1, :],
                self.day_of_week_size,
                name="day_of_week",
                integer_index=day_of_week_index,
                batch=batch,
            )

        conv_in = flatten_history_for_conv2d(
            input_data,
            input_len=self.input_len,
            input_dim=self.input_dim,
        )
        time_series_emb = self.time_series_emb_layer(conv_in)

        node_emb_list: list[torch.Tensor] = []
        if self.if_spatial:
            node_emb_list.append(
                self.node_emb.unsqueeze(0).expand(batch, -1, -1).transpose(1, 2).unsqueeze(-1)
            )
        tem_emb: list[torch.Tensor] = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        hidden = torch.cat([time_series_emb] + node_emb_list + tem_emb, dim=1)
        if int(hidden.size(1)) != self.hidden_dim:
            raise ReimplementationError(
                f"concat channels {int(hidden.size(1))} != hidden_dim {self.hidden_dim}"
            )
        if tuple(hidden.shape[2:]) != (self.num_nodes, 1):
            raise ReimplementationError(f"hidden spatial layout {tuple(hidden.shape)} is not [B,D,N,1]")
        encoded = self.encoder(hidden)
        prediction = self.regression_layer(encoded)
        expected = (batch, self.output_len, self.num_nodes, 1)
        if tuple(prediction.shape) != expected:
            raise ReimplementationError(f"prediction {tuple(prediction.shape)} != {expected}")
        if not torch.isfinite(prediction).all():
            raise ReimplementationError("STID prediction contains NaN or inf")
        if not return_trace:
            return prediction
        slices = self.identity_slices()
        trace = {
            "history_shape": list(history_data.shape),
            "conv_input_shape": list(conv_in.shape),
            "time_series_emb_shape": list(time_series_emb.shape),
            "hidden_shape": list(hidden.shape),
            "encoded_shape": list(encoded.shape),
            "prediction_shape": list(prediction.shape),
            "hidden_dim": self.hidden_dim,
            "identity_slices": {key: [item.start, item.stop] for key, item in slices.items()},
            "used_last_history_step_for_time_identity": True,
            "used_integer_time_of_day_index": time_of_day_index is not None,
            "used_integer_day_of_week_index": day_of_week_index is not None,
            "used_future_data": False,
            "used_adjacency": False,
        }
        if self.if_spatial:
            trace["spatial_emb_shape"] = list(node_emb_list[0].shape)
        if time_in_day_emb is not None:
            trace["time_of_day_emb_shape"] = list(tem_emb[0].shape)
        if day_in_week_emb is not None:
            trace["day_of_week_emb_shape"] = list(tem_emb[-1].shape)
        return prediction, trace


def graph_keys_in_state_dict(state: Mapping[str, Any]) -> list[str]:
    hits: list[str] = []
    for name in state:
        lower = str(name).lower()
        if any(fragment in lower for fragment in GRAPH_STATE_FRAGMENTS):
            hits.append(str(name))
    return hits
