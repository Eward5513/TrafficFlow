"""STAEformer ported from ``reference/STAEformer/model/STAEformer.py``.

Public flow is ``[B, T, V, 1]``. Official PeMS configs use ``input_dim=3`` so
time-of-day fraction and day-of-week integers are concatenated for
``input_proj``, then also used as embedding indices. Spatio-temporal adaptive
embedding is ``[T, V, D_a]``, shared across samples, not a graph encoding.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from reimplementation.common.errors import ReimplementationError
from reimplementation.staeformer.model.attention import GRAPH_FORWARD_KEYS, refuse_graph_forward_kwargs
from reimplementation.staeformer.model.layers import SelfAttentionLayer, flatten_time_hidden

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


def compute_model_dim(
    *,
    input_embedding_dim: int,
    tod_embedding_dim: int,
    dow_embedding_dim: int,
    spatial_embedding_dim: int,
    adaptive_embedding_dim: int,
) -> int:
    return (
        int(input_embedding_dim)
        + int(tod_embedding_dim)
        + int(dow_embedding_dim)
        + int(spatial_embedding_dim)
        + int(adaptive_embedding_dim)
    )


def embedding_slices(
    *,
    input_embedding_dim: int,
    tod_embedding_dim: int,
    dow_embedding_dim: int,
    spatial_embedding_dim: int,
    adaptive_embedding_dim: int,
) -> dict[str, slice]:
    offset = 0
    slices: dict[str, slice] = {}
    if input_embedding_dim > 0:
        slices["feature"] = slice(offset, offset + int(input_embedding_dim))
        offset += int(input_embedding_dim)
    if tod_embedding_dim > 0:
        slices["time_of_day"] = slice(offset, offset + int(tod_embedding_dim))
        offset += int(tod_embedding_dim)
    if dow_embedding_dim > 0:
        slices["day_of_week"] = slice(offset, offset + int(dow_embedding_dim))
        offset += int(dow_embedding_dim)
    if spatial_embedding_dim > 0:
        slices["spatial"] = slice(offset, offset + int(spatial_embedding_dim))
        offset += int(spatial_embedding_dim)
    if adaptive_embedding_dim > 0:
        slices["adaptive"] = slice(offset, offset + int(adaptive_embedding_dim))
        offset += int(adaptive_embedding_dim)
    return slices


class STAEformer(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_steps: int = 12,
        out_steps: int = 12,
        steps_per_day: int = 288,
        input_dim: int = 3,
        output_dim: int = 1,
        input_embedding_dim: int = 24,
        tod_embedding_dim: int = 24,
        dow_embedding_dim: int = 24,
        spatial_embedding_dim: int = 0,
        adaptive_embedding_dim: int = 80,
        feed_forward_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_mixed_proj: bool = True,
        attention_mask: bool = False,
    ) -> None:
        super().__init__()
        if int(num_nodes) < 1 or int(in_steps) < 1 or int(out_steps) < 1:
            raise ReimplementationError("num_nodes, in_steps, out_steps must be positive")
        if int(input_dim) < 1 or int(output_dim) < 1:
            raise ReimplementationError("input_dim and output_dim must be positive")
        if int(num_layers) < 1:
            raise ReimplementationError("num_layers must be positive")
        self.num_nodes = int(num_nodes)
        self.in_steps = int(in_steps)
        self.out_steps = int(out_steps)
        self.steps_per_day = int(steps_per_day)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.input_embedding_dim = int(input_embedding_dim)
        self.tod_embedding_dim = int(tod_embedding_dim)
        self.dow_embedding_dim = int(dow_embedding_dim)
        self.spatial_embedding_dim = int(spatial_embedding_dim)
        self.adaptive_embedding_dim = int(adaptive_embedding_dim)
        self.feed_forward_dim = int(feed_forward_dim)
        self.model_dim = compute_model_dim(
            input_embedding_dim=self.input_embedding_dim,
            tod_embedding_dim=self.tod_embedding_dim,
            dow_embedding_dim=self.dow_embedding_dim,
            spatial_embedding_dim=self.spatial_embedding_dim,
            adaptive_embedding_dim=self.adaptive_embedding_dim,
        )
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.use_mixed_proj = bool(use_mixed_proj)
        self.attention_mask = bool(attention_mask)
        self.dropout = float(dropout)
        if self.model_dim % self.num_heads != 0:
            raise ReimplementationError(
                f"model_dim {self.model_dim} is not divisible by num_heads {self.num_heads}"
            )

        self.input_proj = nn.Linear(self.input_dim, self.input_embedding_dim)
        if self.tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_embedding_dim)
        if self.dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)
        if self.spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.spatial_embedding_dim))
            nn.init.xavier_uniform_(self.node_emb)
        if self.adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(self.in_steps, self.num_nodes, self.adaptive_embedding_dim))
            )

        if self.use_mixed_proj:
            self.output_proj = nn.Linear(self.in_steps * self.model_dim, self.out_steps * self.output_dim)
        else:
            self.temporal_proj = nn.Linear(self.in_steps, self.out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(
                    self.model_dim,
                    self.feed_forward_dim,
                    self.num_heads,
                    self.dropout,
                    self.attention_mask,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(
                    self.model_dim,
                    self.feed_forward_dim,
                    self.num_heads,
                    self.dropout,
                    self.attention_mask,
                )
                for _ in range(self.num_layers)
            ]
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "STAEformer":
        return cls(
            num_nodes=int(config["num_nodes"]),
            in_steps=int(config.get("in_steps", config.get("n_his", 12))),
            out_steps=int(config.get("out_steps", config.get("output_window", 1))),
            steps_per_day=int(config.get("steps_per_day", 288)),
            input_dim=int(config.get("input_dim", 3)),
            output_dim=int(config.get("output_dim", 1)),
            input_embedding_dim=int(config.get("input_embedding_dim", 24)),
            tod_embedding_dim=int(config.get("tod_embedding_dim", 24)),
            dow_embedding_dim=int(config.get("dow_embedding_dim", 24)),
            spatial_embedding_dim=int(config.get("spatial_embedding_dim", 0)),
            adaptive_embedding_dim=int(config.get("adaptive_embedding_dim", 80)),
            feed_forward_dim=int(config.get("feed_forward_dim", 256)),
            num_heads=int(config.get("num_heads", 4)),
            num_layers=int(config.get("num_layers", 3)),
            dropout=float(config.get("dropout", 0.1)),
            use_mixed_proj=bool(config.get("use_mixed_proj", True)),
            attention_mask=bool(config.get("attention_mask", False)),
        )

    def parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.parameters()))

    def expected_parameter_count(self) -> int:
        count = self.input_dim * self.input_embedding_dim + self.input_embedding_dim
        if self.tod_embedding_dim > 0:
            count += self.steps_per_day * self.tod_embedding_dim
        if self.dow_embedding_dim > 0:
            count += 7 * self.dow_embedding_dim
        if self.spatial_embedding_dim > 0:
            count += self.num_nodes * self.spatial_embedding_dim
        if self.adaptive_embedding_dim > 0:
            count += self.in_steps * self.num_nodes * self.adaptive_embedding_dim
        if self.use_mixed_proj:
            count += (self.in_steps * self.model_dim) * (self.out_steps * self.output_dim)
            count += self.out_steps * self.output_dim
        else:
            count += self.in_steps * self.out_steps + self.out_steps
            count += self.model_dim * self.output_dim + self.output_dim
        linear = self.model_dim * self.model_dim + self.model_dim
        attn = 4 * linear
        ffn = self.model_dim * self.feed_forward_dim + self.feed_forward_dim
        ffn += self.feed_forward_dim * self.model_dim + self.model_dim
        ln = 4 * self.model_dim
        count += self.num_layers * 2 * (attn + ffn + ln)
        return int(count)

    def embedding_slices(self) -> dict[str, slice]:
        return embedding_slices(
            input_embedding_dim=self.input_embedding_dim,
            tod_embedding_dim=self.tod_embedding_dim,
            dow_embedding_dim=self.dow_embedding_dim,
            spatial_embedding_dim=self.spatial_embedding_dim,
            adaptive_embedding_dim=self.adaptive_embedding_dim,
        )

    def _expand_index(self, index: torch.Tensor, batch: int, time: int, name: str, size: int) -> torch.Tensor:
        idx = index.to(dtype=torch.long)
        if idx.ndim == 1:
            if int(idx.shape[0]) != batch:
                raise ReimplementationError(f"{name} batch {int(idx.shape[0])} != {batch}")
            idx = idx.view(batch, 1).expand(batch, time)
        if idx.ndim == 2:
            if tuple(idx.shape) != (batch, time):
                raise ReimplementationError(f"{name} shape {tuple(idx.shape)} != {(batch, time)}")
            idx = idx.unsqueeze(-1).expand(batch, time, self.num_nodes)
        if idx.ndim != 3 or tuple(idx.shape) != (batch, time, self.num_nodes):
            raise ReimplementationError(f"{name} must broadcast to [B,T,V], got {tuple(idx.shape)}")
        if bool((idx < 0).any() or (idx >= size).any()):
            raise ReimplementationError(f"{name} index is outside [0, {size})")
        return idx.contiguous()

    def _channel_from_index(
        self,
        index: torch.Tensor,
        *,
        batch: int,
        time: int,
        dtype: torch.dtype,
        device: torch.device,
        scale: float | None,
        name: str,
        size: int,
    ) -> torch.Tensor:
        idx = self._expand_index(index, batch, time, name, size)
        values = idx.to(dtype=dtype, device=device)
        if scale is not None:
            values = values / float(scale)
        return values.unsqueeze(-1)

    def _assemble_input_channels(
        self,
        x: torch.Tensor,
        time_of_day_index: torch.Tensor | None,
        day_of_week_index: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build the official ``input_dim`` last axis from flow plus optional indices.

        Official PeMS YAML uses ``input_dim=3`` so ``input_proj`` sees
        ``[flow, tod_fraction, dow_integer]``. R-only NPZ only stores flow;
        the adapter or this method concatenates the extra channels. Integer
        embedding indices are still passed separately so float32
        ``slot/288*288`` truncation cannot skip an embedding row.
        """
        batch, time, _nodes, channels = x.shape
        if channels >= self.input_dim:
            return x[..., : self.input_dim]
        if channels != 1:
            raise ReimplementationError(
                f"x last dim is {channels}; expected 1 (flow) or {self.input_dim} (official)"
            )
        parts = [x]
        if self.input_dim >= 2:
            if time_of_day_index is None:
                raise ReimplementationError("input_dim>=2 needs time_of_day_index when x is flow-only")
            parts.append(
                self._channel_from_index(
                    time_of_day_index,
                    batch=batch,
                    time=time,
                    dtype=x.dtype,
                    device=x.device,
                    scale=float(self.steps_per_day),
                    name="time_of_day",
                    size=self.steps_per_day,
                )
            )
        if self.input_dim >= 3:
            if day_of_week_index is None:
                raise ReimplementationError("input_dim>=3 needs day_of_week_index when x is flow-only")
            parts.append(
                self._channel_from_index(
                    day_of_week_index,
                    batch=batch,
                    time=time,
                    dtype=x.dtype,
                    device=x.device,
                    scale=None,
                    name="day_of_week",
                    size=7,
                )
            )
        assembled = torch.cat(parts, dim=-1)
        if int(assembled.size(-1)) != self.input_dim:
            raise ReimplementationError(
                f"assembled channels {int(assembled.size(-1))} != input_dim {self.input_dim}"
            )
        return assembled

    def forward(
        self,
        x: torch.Tensor,
        time_of_day_index: torch.Tensor | None = None,
        day_of_week_index: torch.Tensor | None = None,
        *,
        return_trace: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        refuse_graph_forward_kwargs(kwargs)
        extra = set(kwargs) - GRAPH_FORWARD_KEYS
        if extra:
            raise ReimplementationError(f"unexpected STAEformer forward kwargs {sorted(extra)}")
        if x.ndim != 4:
            raise ReimplementationError(f"x must be [B,T,V,C], got {tuple(x.shape)}")
        batch, time, nodes, _channels = x.shape
        if int(time) != self.in_steps:
            raise ReimplementationError(f"T={time} != in_steps={self.in_steps}")
        if int(nodes) != self.num_nodes:
            raise ReimplementationError(f"V={nodes} != num_nodes={self.num_nodes}")
        if not torch.isfinite(x).all():
            raise ReimplementationError("x contains NaN or inf")

        features_in = self._assemble_input_channels(x, time_of_day_index, day_of_week_index)
        tod_values = features_in[..., 1] if self.input_dim >= 2 else None
        dow_values = features_in[..., 2] if self.input_dim >= 3 else None

        feature_emb = self.input_proj(features_in)
        pieces = [feature_emb]
        tod_emb = None
        dow_emb = None
        if self.tod_embedding_dim > 0:
            if time_of_day_index is not None:
                tod_idx = self._expand_index(
                    time_of_day_index, batch, time, "time_of_day", self.steps_per_day
                )
            else:
                if tod_values is None:
                    raise ReimplementationError("time-of-day embedding needs channel 1 or integer indices")
                tod_idx = (tod_values * self.steps_per_day).long()
                if bool((tod_idx < 0).any() or (tod_idx >= self.steps_per_day).any()):
                    raise ReimplementationError("time-of-day index is outside [0, 288)")
            tod_emb = self.tod_embedding(tod_idx)
            pieces.append(tod_emb)
        if self.dow_embedding_dim > 0:
            if day_of_week_index is not None:
                dow_idx = self._expand_index(day_of_week_index, batch, time, "day_of_week", 7)
            else:
                if dow_values is None:
                    raise ReimplementationError("day-of-week embedding needs channel 2 or integer indices")
                dow_idx = dow_values.long()
                if bool((dow_idx < 0).any() or (dow_idx >= 7).any()):
                    raise ReimplementationError("day-of-week index is outside [0, 7)")
            dow_emb = self.dow_embedding(dow_idx)
            pieces.append(dow_emb)
        spatial_emb = None
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(batch, self.in_steps, *self.node_emb.shape)
            pieces.append(spatial_emb)
        adaptive_emb = None
        if self.adaptive_embedding_dim > 0:
            adaptive_emb = self.adaptive_embedding.expand(batch, *self.adaptive_embedding.shape)
            pieces.append(adaptive_emb)
        hidden = torch.cat(pieces, dim=-1)
        if int(hidden.size(-1)) != self.model_dim:
            raise ReimplementationError(f"concat dim {int(hidden.size(-1))} != model_dim {self.model_dim}")

        for attn in self.attn_layers_t:
            hidden = attn(hidden, dim=1)
        temporal_out = hidden
        for attn in self.attn_layers_s:
            hidden = attn(hidden, dim=2)
        spatial_out = hidden

        if self.use_mixed_proj:
            flat = flatten_time_hidden(hidden)
            projected = self.output_proj(flat).view(batch, self.num_nodes, self.out_steps, self.output_dim)
            prediction = projected.transpose(1, 2)
        else:
            tmp = hidden.transpose(1, 3)
            tmp = self.temporal_proj(tmp)
            prediction = self.output_proj(tmp.transpose(1, 3))

        expected = (batch, self.out_steps, self.num_nodes, self.output_dim)
        if tuple(prediction.shape) != expected:
            raise ReimplementationError(f"prediction {tuple(prediction.shape)} != {expected}")
        if not torch.isfinite(prediction).all():
            raise ReimplementationError("STAEformer prediction contains NaN or inf")
        if not return_trace:
            return prediction
        slices = self.embedding_slices()
        trace = {
            "history_shape": list(x.shape),
            "feature_emb_shape": list(feature_emb.shape),
            "hidden_shape": list(hidden.shape),
            "temporal_out_shape": list(temporal_out.shape),
            "spatial_out_shape": list(spatial_out.shape),
            "prediction_shape": list(prediction.shape),
            "model_dim": self.model_dim,
            "embedding_slices": {key: [item.start, item.stop] for key, item in slices.items()},
            "used_mixed_proj": self.use_mixed_proj,
            "attention_mask": self.attention_mask,
            "spatial_embedding_enabled": self.spatial_embedding_dim > 0,
            "used_integer_time_of_day_index": time_of_day_index is not None,
            "used_integer_day_of_week_index": day_of_week_index is not None,
            "used_adjacency": False,
            "used_future_data": False,
            "num_temporal_layers": len(self.attn_layers_t),
            "num_spatial_layers": len(self.attn_layers_s),
        }
        if tod_emb is not None:
            trace["tod_emb_shape"] = list(tod_emb.shape)
        if dow_emb is not None:
            trace["dow_emb_shape"] = list(dow_emb.shape)
        if adaptive_emb is not None:
            trace["adaptive_emb_shape"] = list(adaptive_emb.shape)
        return prediction, trace


def graph_keys_in_state_dict(state: Mapping[str, Any]) -> list[str]:
    hits: list[str] = []
    for name in state:
        lower = str(name).lower()
        if any(fragment in lower for fragment in GRAPH_STATE_FRAGMENTS):
            hits.append(str(name))
    return hits
