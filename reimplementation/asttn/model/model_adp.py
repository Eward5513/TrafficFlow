"""ASTTN ``Model_Adp`` for last-observed-step reconstruction.

The adaptive graph, per-offset ``edge_softmax``, residual, and head split follow
``reference/ASTTN_pytorch/model/model_adp.py``. DGL is not installed here, so the
message passing is the same incoming-edge softmax:

* an edge is ``(src, dst) = nonzero(masked adjacency)``;
* the score is the dot product of the source key and the destination query;
* softmax is over edges that share a destination (DGL ``edge_softmax`` default);
* messages are source values times that score, summed at the destination;
* a destination with no incoming edge stays zero before the residual.

Edge weights of the adaptive matrix are still discarded. The reference assigns
them only in a commented line, so ``nodevec1`` / ``nodevec2`` affect the edge
set and do not receive gradient through ``nonzero``. That is left unchanged.

The decoder is the single-step task adaptation: ``T=1`` and ``window=1``.
The encoder keeps ``T=12`` and ``window=6``, including the original shift wrap.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from reimplementation.asttn.model.model_utils import CONVs, PostProcess, STEmbedding, TransformAttention, TransformerSelfOutput
from reimplementation.common.errors import ReimplementationError


def reference_shift_list(length: int, window: int) -> np.ndarray:
    """Copy of ``STAttention_Adp.get_shift_list``."""
    index = np.arange(int(length))
    window_size = int(window)
    offsets = np.arange(-(window_size - 1) // 2, (window_size - 1) // 2 + 1, 1)
    shifts = []
    for offset in offsets:
        shifted = index + int(offset)
        shifted[shifted < 0] = shifted[shifted < 0] + window_size
        shifted[shifted > (int(length) - 1)] = shifted[shifted > (int(length) - 1)] - window_size
        shifts.append(shifted)
    return np.asarray(shifts, dtype=np.int64)


def incoming_edge_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    dropout: float,
    training: bool,
) -> torch.Tensor:
    """DGL ``u_dot_v('k','q')`` + destination ``edge_softmax`` + ``u_mul_e`` sum.

    ``query``, ``key``, and ``value`` are ``[N, B, T, H, D]``. ``src`` and
    ``dst`` index nodes. The result is ``[N, B, T, H, D]``.
    """
    num_nodes = int(query.shape[0])
    if src.numel() == 0:
        return query.new_zeros(query.shape)
    head_dim = int(query.shape[-1])
    destination_query = query.index_select(0, dst) / (head_dim ** 0.5)
    source_key = key.index_select(0, src)
    source_value = value.index_select(0, src)
    score = (source_key * destination_query).sum(dim=-1, keepdim=True)
    score_max = torch.full(
        (num_nodes, *score.shape[1:]),
        torch.finfo(score.dtype).min,
        device=score.device,
        dtype=score.dtype,
    )
    index = dst.view(-1, *([1] * (score.ndim - 1))).expand_as(score)
    score_max = score_max.scatter_reduce(0, index, score, reduce="amax", include_self=True)
    exp_score = torch.exp(score - score_max.index_select(0, dst))
    denom = score.new_zeros((num_nodes, *score.shape[1:]))
    denom.index_add_(0, dst, exp_score)
    attention = exp_score / denom.index_select(0, dst)
    attention = F.dropout(attention, p=float(dropout), training=training)
    messages = source_value * attention
    out = query.new_zeros(query.shape)
    out.index_add_(0, dst, messages)
    return out


class STAttentionAdp(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        length: int,
        window: int,
        num_nodes: int,
        max_neighbors: int = 40,
        node_dim: int = 20,
        dropout: float = 0.1,
        batch_chunk: int = 4,
    ):
        super().__init__()
        hidden = int(num_heads) * int(head_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.window = int(window)
        self.length = int(length)
        self.num_nodes = int(num_nodes)
        self.max_neighbors = int(max_neighbors)
        self.dropout_p = float(dropout)
        self.batch_chunk = max(int(batch_chunk), 1)
        self.fc_q = nn.Linear(2 * hidden, hidden)
        self.fc_k = nn.Linear(2 * hidden, hidden)
        self.fc_v = nn.Linear(2 * hidden, hidden)
        self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, int(node_dim)))
        self.nodevec2 = nn.Parameter(torch.randn(int(node_dim), self.num_nodes))
        self.attn_output = TransformerSelfOutput(hidden, hidden, dropout=self.dropout_p)
        self.register_buffer("shift_list", torch.as_tensor(reference_shift_list(self.length, self.window), dtype=torch.long), persistent=False)

    def adaptive_edges(self) -> tuple[torch.Tensor, torch.Tensor]:
        adjacency = F.softmax(F.relu(self.nodevec1.matmul(self.nodevec2)), dim=1)
        threshold = 1.0 / float(self.num_nodes)
        order = int(self.max_neighbors) + 1
        if order > int(adjacency.shape[1]):
            raise ReimplementationError(
                f"adaptive top-k {self.max_neighbors} does not fit {adjacency.shape[1]} nodes"
            )
        kth, _ = torch.kthvalue(-adjacency, order, dim=1, keepdim=True)
        hard = torch.logical_and(adjacency > threshold, adjacency > -kth).type_as(adjacency)
        mask = (hard - adjacency).detach() + adjacency
        adjacency = adjacency * mask
        index = torch.nonzero(adjacency, as_tuple=False)
        if index.numel() == 0:
            empty = adjacency.new_empty((0,), dtype=torch.long)
            return empty, empty
        return index[:, 0].detach(), index[:, 1].detach()

    def forward(self, hidden: torch.Tensor, spatial_temporal: torch.Tensor) -> torch.Tensor:
        fused = torch.cat((hidden, spatial_temporal), dim=-1)
        query = self.fc_q(fused)
        key = self.fc_k(fused)
        value = self.fc_v(fused)
        batch, time, nodes, channels = query.shape
        head_dim = channels // self.num_heads
        query = query.view(batch, time, nodes, self.num_heads, head_dim).permute(2, 0, 1, 3, 4)
        key = key.view(batch, time, nodes, self.num_heads, head_dim).permute(2, 0, 1, 3, 4)
        value = value.view(batch, time, nodes, self.num_heads, head_dim).permute(2, 0, 1, 3, 4)
        src, dst = self.adaptive_edges()
        accumulated = hidden.new_zeros(batch, time, nodes, channels)
        shifts = int(self.shift_list.shape[0])
        for row in range(shifts):
            shift = self.shift_list[row].to(device=key.device)
            attended = self._shifted_attention(query, key.index_select(2, shift), value.index_select(2, shift), src, dst)
            attended = attended.permute(1, 2, 0, 3, 4).reshape(batch, time, nodes, channels)
            accumulated = accumulated + attended
        accumulated = accumulated / float(shifts)
        return self.attn_output(accumulated, hidden)

    def _shifted_attention(self, query, key, value, src, dst):
        """Batch-chunk the edge messages. Softmax and dropout stay per sample.

        Checkpointing recomputes a chunk on the backward pass with the saved
        RNG state, so the dropout mask matches the forward pass. Batch norm
        is outside this method and still sees the whole batch.
        """
        parts = []
        width = int(query.shape[1])
        step = self.batch_chunk
        for start in range(0, width, step):
            end = min(width, start + step)
            q = query[:, start:end]
            k = key[:, start:end]
            v = value[:, start:end]
            if torch.is_grad_enabled():
                attended = checkpoint(self._attend_chunk, q, k, v, src, dst, use_reentrant=False)
            else:
                attended = self._attend_chunk(q, k, v, src, dst)
            parts.append(attended)
        return torch.cat(parts, dim=1)

    def _attend_chunk(self, query, key, value, src, dst):
        return incoming_edge_attention(
            query,
            key,
            value,
            src,
            dst,
            dropout=self.dropout_p,
            training=self.training,
        )


class STLayer(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, *, length: int, window: int, num_nodes: int, max_neighbors: int, node_dim: int, dropout: float, batch_chunk: int = 4):
        super().__init__()
        self.attention = STAttentionAdp(
            num_heads,
            head_dim,
            length=length,
            window=window,
            num_nodes=num_nodes,
            max_neighbors=max_neighbors,
            node_dim=node_dim,
            dropout=dropout,
            batch_chunk=batch_chunk,
        )
        self.mlp = PostProcess(num_heads * head_dim)

    def forward(self, hidden: torch.Tensor, spatial_temporal: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.attention(hidden, spatial_temporal))


class ASTTN(nn.Module):
    def __init__(
        self,
        spatial_embedding: torch.Tensor,
        *,
        num_nodes: int,
        input_channels: int,
        output_channels: int,
        num_his: int = 12,
        num_layers: int = 1,
        num_heads: int = 8,
        head_dim: int = 8,
        encoder_window: int = 6,
        max_neighbors: int = 40,
        node_dim: int = 20,
        dropout: float = 0.1,
        slots_per_day: int = 288,
        attention_batch_chunk: int = 4,
    ):
        super().__init__()
        if int(spatial_embedding.shape[0]) != int(num_nodes):
            raise ReimplementationError(
                f"spatial embedding rows {spatial_embedding.shape[0]} != num_nodes {num_nodes}"
            )
        hidden = int(num_heads) * int(head_dim)
        self.num_his = int(num_his)
        self.num_nodes = int(num_nodes)
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.encoder_window = int(encoder_window)
        self.decoder_length = 1
        self.decoder_window = 1
        self.register_buffer("spatial_embedding", spatial_embedding.detach().to(dtype=torch.float32).contiguous())
        self.embedding = STEmbedding(hidden, int(spatial_embedding.shape[1]), slots_per_day=int(slots_per_day))
        layer_kwargs = {
            "num_nodes": self.num_nodes,
            "max_neighbors": int(max_neighbors),
            "node_dim": int(node_dim),
            "dropout": float(dropout),
            "batch_chunk": int(attention_batch_chunk),
        }
        self.encoder = nn.ModuleList(
            [
                STLayer(num_heads, head_dim, length=self.num_his, window=self.encoder_window, **layer_kwargs)
                for _ in range(int(num_layers))
            ]
        )
        self.decoder = nn.ModuleList(
            [
                STLayer(num_heads, head_dim, length=self.decoder_length, window=self.decoder_window, **layer_kwargs)
                for _ in range(int(num_layers))
            ]
        )
        self.transform = TransformAttention(num_heads, head_dim)
        self.input_mlp = CONVs(
            input_dims=[self.input_channels, hidden],
            units=[hidden, hidden],
            activations=[F.relu, None],
        )
        self.output_mlp = CONVs(
            input_dims=[hidden, hidden],
            units=[hidden, self.output_channels],
            activations=[F.relu, None],
        )

    @classmethod
    def from_config(cls, config: dict, spatial_embedding: torch.Tensor) -> "ASTTN":
        return cls(
            spatial_embedding,
            num_nodes=int(config["num_nodes"]),
            input_channels=int(config["input_channels"]),
            output_channels=int(config["output_channels"]),
            num_his=int(config.get("n_his", config.get("num_his", 12))),
            num_layers=int(config.get("L", 1)),
            num_heads=int(config.get("K", 8)),
            head_dim=int(config.get("d", 8)),
            encoder_window=int(config.get("window", 6)),
            max_neighbors=int(config.get("max_neighbors", 40)),
            node_dim=int(config.get("adaptive_node_dim", 20)),
            dropout=float(config.get("dropout", 0.1)),
            slots_per_day=int(config.get("slots_per_day", 288)),
            attention_batch_chunk=int(config.get("attention_batch_chunk", 4)),
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ReimplementationError(f"ASTTN x must be [B, T, N, C], got {tuple(x.shape)}")
        if time_embedding.ndim != 3 or int(time_embedding.shape[1]) != self.num_his + 1:
            raise ReimplementationError(
                f"ASTTN time embedding must be [B, {self.num_his + 1}, 2], got {tuple(time_embedding.shape)}"
            )
        if int(x.shape[1]) != self.num_his or int(x.shape[-1]) != self.input_channels:
            raise ReimplementationError(
                f"ASTTN x {tuple(x.shape)} does not match T={self.num_his} C={self.input_channels}"
            )
        hidden = self.input_mlp(x)
        spatial_temporal = self.embedding(self.spatial_embedding, time_embedding)
        history = spatial_temporal[:, : self.num_his]
        target = spatial_temporal[:, self.num_his :]
        for layer in self.encoder:
            hidden = layer(hidden, history)
        hidden = self.transform(hidden, history, target)
        for layer in self.decoder:
            hidden = layer(hidden, target)
        output = self.output_mlp(hidden)
        if output.ndim != 4 or int(output.shape[1]) != 1 or int(output.shape[-1]) != self.output_channels:
            raise ReimplementationError(f"ASTTN output {tuple(output.shape)} is not [B, 1, N, C_out]")
        return output
