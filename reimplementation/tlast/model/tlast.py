"""TLAST ported from ``reference/TLAST/models/TLAST.py``.

The public reconstruction tensor stays ``[B, T, N, C]`` in and
``[B, 1, N, C_out]`` out. That singleton horizon is the project layout of
the requested ``[B, N, C_out]`` current-slot state.

Original class map:

- ``DataEncoding`` -> :class:`CrossTimeDataEncoding`
- ``getTimeEmbedding`` -> :class:`TemporalEmbedding`
- ``getTimeLagEmbedding`` -> :class:`TimeLagEmbedding`
- ``getSpatialEmbedding`` -> :class:`SpatialEmbedding`
- ``MultiHeadsAttention`` / ``AttentionLayer`` -> proxy MHA
- ``EncoderLayer`` (``att_type='proxy'``, ``dynamic_proxy=True``) -> :class:`ProxyEncoderLayer`
- ``time_wise_predictor`` -> :class:`TimeWisePredictor`
- ``Model`` -> :class:`TLAST`

Only the published proxy attention path is implemented. Full, ProbSparse,
and DIFFormer branches in the reference file are not the TLAST model used
by the released configurations.
"""

from __future__ import annotations

from math import sqrt
from typing import Any, Mapping

import torch
from torch import nn

from reimplementation.common.errors import ReimplementationError

CODE_VERSION = "0.1.0"
MODEL_NAME = "tlast"


def apply_author_initialization(model: nn.Module) -> None:
    """Match ``run.py``: Xavier uniform on every parameter with dim > 1."""
    for parameter in model.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention. Scores are ``[B, heads, query, key]``."""

    def __init__(self, attention_dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # queries [B, L, H, E], keys/values [B, S, H, E]
        _, _, _, head_dim = queries.shape
        scale = 1.0 / sqrt(head_dim)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        weights = self.dropout(torch.softmax(scale * scores, dim=-1))
        mixed = torch.einsum("bhls,bshd->blhd", weights, values)
        if return_attention:
            return mixed.contiguous(), weights
        return mixed.contiguous(), None


class AttentionLayer(nn.Module):
    """Reference ``AttentionLayer`` with ``att_type='proxy'`` and ``mix=True``."""

    def __init__(self, hid_dim: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if int(hid_dim) % int(n_heads) != 0:
            raise ReimplementationError(f"hid_dim {hid_dim} is not divisible by n_heads {n_heads}")
        self.n_heads = int(n_heads)
        self.query_projection = nn.Linear(hid_dim, hid_dim)
        self.key_projection = nn.Linear(hid_dim, hid_dim)
        self.value_projection = nn.Linear(hid_dim, hid_dim)
        self.out_projection = nn.Linear(hid_dim, hid_dim)
        self.inner_attention = MultiHeadAttention(attention_dropout=dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, query_len, _ = queries.shape
        key_len = int(keys.shape[1])
        heads = self.n_heads
        queries = self.query_projection(queries).view(batch, query_len, heads, -1)
        keys = self.key_projection(keys).view(batch, key_len, heads, -1)
        values = self.value_projection(values).view(batch, key_len, heads, -1)
        mixed, weights = self.inner_attention(
            queries, keys, values, return_attention=return_attention
        )
        # Informer ``mix=True``: swap head and query length, then flatten heads.
        mixed = mixed.transpose(2, 1).contiguous().view(batch, query_len, -1)
        mixed = self.out_projection(mixed)
        if return_attention:
            return mixed, weights
        return mixed, None


class ProxyEncoderLayer(nn.Module):
    """One ST encoder block. Proxy tokens come only from the last time step.

    ``node2proxy`` and ``proxy2node`` are separate modules. Every history
    step is packed into the batch dimension and reuses those parameters.
    The same last-step proxy is repeated across time; it is not updated
    recursively.
    """

    def __init__(
        self,
        *,
        proxy_num: int,
        hid_dim: int,
        n_heads: int,
        num_nodes: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
        norm_flag: str = "none",
    ) -> None:
        super().__init__()
        if norm_flag not in {"none", "first", "after"}:
            raise ReimplementationError(f"norm_flag must be none/first/after, got {norm_flag}")
        if activation not in {"gelu", "relu"}:
            raise ReimplementationError(f"encoder activation must be gelu or relu, got {activation}")
        hidden_ff = int(d_ff) if d_ff is not None else 4 * int(hid_dim)
        self.norm_flag = norm_flag
        self.proxy_num = int(proxy_num)
        self.hid_dim = int(hid_dim)
        self.readout = nn.Linear(int(num_nodes), int(proxy_num))
        self.node2proxy = AttentionLayer(hid_dim, n_heads, dropout=attention_dropout)
        self.proxy2node = AttentionLayer(hid_dim, n_heads, dropout=attention_dropout)
        self.dropout = nn.Dropout(dropout)
        nonlinearity: nn.Module = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(hid_dim, hidden_ff),
            nonlinearity,
            nn.Linear(hidden_ff, hid_dim),
        )
        if norm_flag == "first":
            self.norm0 = nn.LayerNorm(hid_dim)
        if norm_flag != "none":
            self.norm1 = nn.LayerNorm(hid_dim)
            self.norm2 = nn.LayerNorm(hid_dim)

    def forward(
        self,
        data: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        if data.ndim != 4:
            raise ReimplementationError(f"encoder input must be [B,T,N,C], got {tuple(data.shape)}")
        batch, steps, _, _ = data.shape
        last = data[:, -1, :, :]
        projected = self.readout(last.transpose(1, 2))
        proxy = projected.transpose(1, 2).contiguous()
        proxy = (
            proxy.unsqueeze(1)
            .expand(batch, steps, self.proxy_num, self.hid_dim)
            .contiguous()
            .reshape(batch * steps, self.proxy_num, self.hid_dim)
        )
        packed = data.contiguous().reshape(batch * steps, data.shape[2], data.shape[3])
        if self.norm_flag == "first":
            proxy = self.norm0(proxy)
            packed = self.norm1(packed)
            proxy_feature, weights_to_proxy = self.node2proxy(
                proxy, packed, packed, return_attention=return_attention
            )
            node_feature, weights_to_node = self.proxy2node(
                packed, proxy_feature, proxy_feature, return_attention=return_attention
            )
            hidden = packed + self.dropout(node_feature)
            hidden = self.norm2(hidden)
            hidden = hidden + self.dropout(self.ffn(hidden))
        elif self.norm_flag == "after":
            proxy_feature, weights_to_proxy = self.node2proxy(
                proxy, packed, packed, return_attention=return_attention
            )
            node_feature, weights_to_node = self.proxy2node(
                packed, proxy_feature, proxy_feature, return_attention=return_attention
            )
            hidden = packed + self.dropout(node_feature)
            hidden = self.norm1(hidden)
            hidden = hidden + self.dropout(self.ffn(hidden))
            hidden = self.norm2(hidden)
        else:
            proxy_feature, weights_to_proxy = self.node2proxy(
                proxy, packed, packed, return_attention=return_attention
            )
            node_feature, weights_to_node = self.proxy2node(
                packed, proxy_feature, proxy_feature, return_attention=return_attention
            )
            hidden = packed + self.dropout(node_feature)
            hidden = hidden + self.dropout(self.ffn(hidden))
        restored = hidden.contiguous().reshape(batch, steps, data.shape[2], self.hid_dim)
        if not return_attention:
            return restored, None
        if weights_to_proxy is None or weights_to_node is None:
            raise ReimplementationError("proxy attention did not return scores")
        attention = {
            "node_to_proxy": weights_to_proxy.reshape(batch, steps, *weights_to_proxy.shape[1:]),
            "proxy_to_node": weights_to_node.reshape(batch, steps, *weights_to_node.shape[1:]),
        }
        return restored, attention


class TimeWisePredictor(nn.Module):
    """Concatenated history ``[B, N, T*d]`` -> GELU hidden -> per-step head.

    This task uses one head because the label is the last observed slot,
    not a 12-step forecast. The head width is ``output_dim``.
    """

    def __init__(
        self,
        *,
        input_length: int,
        predict_length: int,
        hid_dim: int,
        output_dim: int,
        d_out: int,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if activation not in {"gelu", "relu"}:
            raise ReimplementationError(f"decoder activation must be gelu or relu, got {activation}")
        self.predict_length = int(predict_length)
        self.hidden = nn.Sequential(
            nn.Linear(int(input_length) * int(hid_dim), int(d_out)),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(int(d_out), int(output_dim)) for _ in range(self.predict_length)]
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # [B, T, N, C] -> [B, N, T*C]
        batch, steps, nodes, channels = data.shape
        flat = data.permute(0, 2, 1, 3).reshape(batch, nodes, steps * channels)
        hidden = self.hidden(flat)
        steps_out = [head(hidden).unsqueeze(1) for head in self.heads]
        return torch.cat(steps_out, dim=1)


class TimeLagEmbedding(nn.Module):
    """``MLP(E_time[t] - E_time[t-i])``. The two arguments are embeddings."""

    def __init__(self, hid_dim: int, emb_dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
        )
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, latest: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.mlp(latest - history))


class TemporalEmbedding(nn.Module):
    """Learned time-of-day and day-of-week vectors, summed."""

    def __init__(self, hid_dim: int, slice_size_per_day: int, emb_dropout: float = 0.1) -> None:
        super().__init__()
        self.time_in_day = nn.Embedding(int(slice_size_per_day), hid_dim)
        self.day_in_week = nn.Embedding(7, hid_dim)
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, time_index: torch.Tensor) -> torch.Tensor:
        tod = time_index[..., 0:1]
        dow = time_index[..., 1:2]
        return self.dropout(self.time_in_day(tod) + self.day_in_week(dow))


class SpatialEmbedding(nn.Module):
    """One learned vector per node, broadcast over batch and time."""

    def __init__(self, hid_dim: int, num_nodes: int, emb_dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(num_nodes), hid_dim)
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, reference: torch.Tensor) -> torch.Tensor:
        index = torch.arange(reference.shape[2], device=reference.device, dtype=torch.long)
        return self.dropout(self.embedding(index).view(1, 1, -1, reference.shape[-1]))


class CrossTimeDataEncoding(nn.Module):
    """``MLP(concat(X[t-i], X[t]))`` with one shared MLP."""

    def __init__(self, input_dim: int, hid_dim: int, activation: str = "relu") -> None:
        super().__init__()
        if activation not in {"gelu", "relu"}:
            raise ReimplementationError(f"data activation must be gelu or relu, got {activation}")
        self.linear1 = nn.Linear(int(input_dim) * 2, hid_dim)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.linear2 = nn.Linear(hid_dim, hid_dim)

    def forward(self, history: torch.Tensor, latest: torch.Tensor) -> torch.Tensor:
        hidden = self.linear1(torch.cat([history, latest], dim=-1))
        return self.linear2(self.activation(hidden))


class TLAST(nn.Module):
    """Time-lag aware spatial-temporal transformer for one reconstruction step."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        num_nodes: int,
        seq_len: int = 12,
        predict_length: int = 1,
        hid_dim: int = 64,
        d_out: int = 1024,
        n_heads: int = 1,
        proxy_num: int = 8,
        num_layers: int = 1,
        tau: int = 3,
        d_ff: int | None = None,
        enc_dropout: float = 0.1,
        att_dropout: float = 0.1,
        emb_dropout: float = 0.1,
        te_emb_dropout: float = 0.1,
        se_emb_dropout: float = 0.1,
        tle_emb_dropout: float = 0.1,
        norm_flag: str = "none",
        activation_data: str = "relu",
        activation_enc: str = "gelu",
        activation_dec: str = "gelu",
        slice_size_per_day: int = 288,
        query_cross_time: bool = True,
        use_input_residual: bool = False,
    ) -> None:
        super().__init__()
        if not bool(query_cross_time):
            raise ReimplementationError(
                "TLAST proxy tokens are built from the last history step only; "
                "query_cross_time must stay true"
            )
        if int(num_layers) < 1:
            raise ReimplementationError("num_layers must be >= 1")
        if int(proxy_num) < 1:
            raise ReimplementationError("proxy_num must be >= 1")
        if int(predict_length) != 1:
            raise ReimplementationError("Xuancheng26 TLAST predict_length must be 1")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_nodes = int(num_nodes)
        self.seq_len = int(seq_len)
        self.predict_length = int(predict_length)
        self.hid_dim = int(hid_dim)
        self.n_heads = int(n_heads)
        self.proxy_num = int(proxy_num)
        self.num_layers = int(num_layers)
        self.tau = int(tau)
        self.norm_flag = str(norm_flag)
        self.slice_size_per_day = int(slice_size_per_day)
        self.use_input_residual = bool(use_input_residual)
        self.data_encoding = CrossTimeDataEncoding(self.input_dim, self.hid_dim, activation_data)
        self.input_dropout = nn.Dropout(emb_dropout)
        self.temporal_embedding = TemporalEmbedding(self.hid_dim, self.slice_size_per_day, te_emb_dropout)
        self.time_lag_embedding = TimeLagEmbedding(self.hid_dim, tle_emb_dropout)
        self.spatial_embedding = SpatialEmbedding(self.hid_dim, self.num_nodes, se_emb_dropout)
        self.use_tcn = self.tau > 0
        if self.use_tcn:
            if self.tau not in (2, 3):
                raise ReimplementationError(f"temporal conv kernel tau must be 2 or 3, got {self.tau}")
            pad_left, pad_right = self.tau - 2, 1
            self.time_pad = nn.ReplicationPad2d((pad_left, pad_right, 0, 0))
            self.time_conv = nn.Conv2d(self.hid_dim, self.hid_dim, kernel_size=(1, self.tau))
        self.encoder_layers = nn.ModuleList(
            [
                ProxyEncoderLayer(
                    proxy_num=self.proxy_num,
                    hid_dim=self.hid_dim,
                    n_heads=self.n_heads,
                    num_nodes=self.num_nodes,
                    d_ff=d_ff,
                    dropout=enc_dropout,
                    attention_dropout=att_dropout,
                    activation=activation_enc,
                    norm_flag=self.norm_flag,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.predictor = TimeWisePredictor(
            input_length=self.seq_len,
            predict_length=self.predict_length,
            hid_dim=self.hid_dim,
            output_dim=self.output_dim,
            d_out=int(d_out),
            activation=activation_dec,
        )
        apply_author_initialization(self)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TLAST":
        d_ff = config.get("d_ff")
        return cls(
            input_dim=int(config["input_dim"]),
            output_dim=int(config["output_dim"]),
            num_nodes=int(config["num_nodes"]),
            seq_len=int(config.get("input_len", config.get("n_his", config.get("seq_len", 12)))),
            predict_length=int(config.get("output_len", config.get("output_window", 1))),
            hid_dim=int(config.get("hid_dim", 64)),
            d_out=int(config.get("d_out", 1024)),
            n_heads=int(config.get("n_heads", 1)),
            proxy_num=int(config.get("proxy_num", config.get("M", 8))),
            num_layers=int(config.get("num_layers", 1)),
            tau=int(config.get("tau", 3)),
            d_ff=None if d_ff in (None, "") else int(d_ff),
            enc_dropout=float(config.get("enc_dropout", 0.1)),
            att_dropout=float(config.get("att_dropout", 0.1)),
            emb_dropout=float(config.get("emb_dropout", 0.1)),
            te_emb_dropout=float(config.get("te_emb_dropout", 0.1)),
            se_emb_dropout=float(config.get("se_emb_dropout", 0.1)),
            tle_emb_dropout=float(config.get("tle_emb_dropout", 0.1)),
            norm_flag=str(config.get("norm_flag", "none")),
            activation_data=str(config.get("activation_data", "relu")),
            activation_enc=str(config.get("activation_enc", "gelu")),
            activation_dec=str(config.get("activation_dec", "gelu")),
            slice_size_per_day=int(config.get("slice_size_per_day", config.get("time_of_day_size", 288))),
            query_cross_time=bool(int(config.get("query_cross_time", 1))),
            use_input_residual=bool(config.get("use_input_residual", False)),
        )

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad))

    def forward(
        self,
        x: torch.Tensor,
        time_index: torch.Tensor,
        residual: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        if x.ndim != 4:
            raise ReimplementationError(f"x must be [B,T,N,C], got {tuple(x.shape)}")
        batch, steps, nodes, channels = x.shape
        if steps != self.seq_len or nodes != self.num_nodes or channels != self.input_dim:
            raise ReimplementationError(
                f"x shape {tuple(x.shape)} != "
                f"[{batch}, {self.seq_len}, {self.num_nodes}, {self.input_dim}]"
            )
        if time_index.shape[:2] != (batch, steps) or int(time_index.shape[-1]) != 2:
            raise ReimplementationError(f"time_index must be [B,T,2], got {tuple(time_index.shape)}")
        if time_index.dtype != torch.long:
            time_index = time_index.to(dtype=torch.long)
        if int(time_index[..., 0].min()) < 0 or int(time_index[..., 0].max()) >= self.slice_size_per_day:
            raise ReimplementationError("time-of-day index is outside the embedding table")
        if int(time_index[..., 1].min()) < 0 or int(time_index[..., 1].max()) > 6:
            raise ReimplementationError("day-of-week index is outside 0..6")
        latest = x[:, -1:, :, :].expand(-1, steps, -1, -1)
        hidden = self.input_dropout(self.data_encoding(x, latest))
        time_embedding = self.temporal_embedding(time_index)
        hidden = hidden + time_embedding
        hidden = hidden + self.time_lag_embedding(time_embedding[:, -1:], time_embedding)
        hidden = hidden + self.spatial_embedding(hidden)
        if self.use_tcn:
            # [B, T, N, C] -> [B, C, N, T]. Pad time only.
            conv_in = hidden.permute(0, 3, 2, 1)
            conv_in = self.time_pad(conv_in)
            hidden = self.time_conv(conv_in).permute(0, 3, 2, 1)
            if int(hidden.shape[1]) != self.seq_len:
                raise ReimplementationError("temporal convolution changed the history length")
        skip = hidden
        attentions: list[dict[str, torch.Tensor]] = []
        for layer in self.encoder_layers:
            hidden, attention = layer(hidden, return_attention=return_attention)
            if attention is not None:
                attentions.append(attention)
        hidden = hidden + skip
        prediction = self.predictor(hidden)
        if self.use_input_residual:
            if residual is None:
                raise ReimplementationError("use_input_residual=True requires the latest partial observation")
            if residual.shape != prediction.shape:
                raise ReimplementationError(
                    f"input residual {tuple(residual.shape)} != prediction {tuple(prediction.shape)}"
                )
            prediction = prediction + residual
        elif residual is not None:
            raise ReimplementationError("input residual was passed while use_input_residual is false")
        if return_attention:
            return prediction, attentions
        return prediction
