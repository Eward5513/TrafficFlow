"""Temporal self-attention and sparse spatial graph attention.

Both modules take an already normalized ``[B, T, N, D]`` tensor and return
the sublayer update with the same shape; the residual lives in the block.

QK-Norm is ``nn.RMSNorm`` over each head's channels (learnable scale),
applied to q and k before RoPE.

Optional gate (``use_gated_attention``): ``sigmoid(W_g x + b_g)`` with
``W_g`` [D, D] is computed from the normalized sublayer input at the query
position and multiplies the concatenated head outputs elementwise (one gate
per head and channel) before the output projection.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from MSTT.model.graph import SpatialGraph
from MSTT.model.rope import RotaryEmbedding


class AttentionOutputGate(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(x)) * attention


class TemporalSelfAttention(nn.Module):
    """Bidirectional multi-head attention over the T history steps of each node."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        attention_dropout: float = 0.0,
        use_qk_norm: bool = True,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        use_gated_attention: bool = False,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = d_model // num_heads
        self.attention_dropout = float(attention_dropout)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=norm_eps) if use_qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim, eps=norm_eps) if use_qk_norm else None
        self.rope = RotaryEmbedding(self.head_dim, rope_base) if use_rope else None
        self.gate = AttentionOutputGate(d_model) if use_gated_attention else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, width = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(batch * nodes, steps, width)
        q, k, v = self.qkv(seq).view(batch * nodes, steps, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q, k)
        attention = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attention = attention.transpose(1, 2).reshape(batch * nodes, steps, width)
        if self.gate is not None:
            attention = self.gate(seq, attention)
        out = self.out_proj(attention)
        return out.view(batch, nodes, steps, width).permute(0, 2, 1, 3)


def _edge_attention_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    bias: torch.Tensor | None,
    dropout_p: float,
) -> torch.Tensor:
    """``q/k/v`` [m, N, H, d] -> [m, N, H, d]; scores and sums in at least float32."""
    count, nodes, heads, width = q.shape
    edges = int(src.numel())
    scale = 1.0 / math.sqrt(width)
    compute = torch.promote_types(q.dtype, torch.float32)
    scores = (q.index_select(1, dst).to(compute) * k.index_select(1, src).to(compute)).sum(-1) * scale
    if bias is not None:
        scores = scores + bias.to(compute)
    index = dst.view(1, edges, 1).expand(count, edges, heads)
    peak = scores.new_full((count, nodes, heads), float("-inf"))
    peak = peak.scatter_reduce(1, index, scores.detach(), reduce="amax", include_self=True)
    weights = torch.exp(scores - peak.index_select(1, dst))
    total = scores.new_zeros((count, nodes, heads)).index_add(1, dst, weights)
    alpha = weights / total.index_select(1, dst)
    if dropout_p > 0.0:
        alpha = F.dropout(alpha, p=dropout_p, training=True)
    messages = alpha.unsqueeze(-1) * v.index_select(1, src).to(compute)
    out = messages.new_zeros((count, nodes, heads, width)).index_add(1, dst, messages)
    return out.to(v.dtype)


def edge_softmax_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    chunk_size: int | None = None,
    use_checkpoint: bool = False,
) -> torch.Tensor:
    """Scaled dot-product attention restricted to graph edges.

    ``score[e, h] = <q[dst_e, h], k[src_e, h]> / sqrt(d) + bias[e, h]`` and the
    softmax runs over the edges of each ``dst``. Only ``E`` scores per head
    exist; no ``N x N`` tensor is formed. ``chunk_size`` splits the leading
    ``M = B * T`` axis; with ``use_checkpoint`` each chunk's edge tensors are
    recomputed in backward instead of stored.
    """
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError(f"q/k/v must share [M, N, H, d], got {q.shape}, {k.shape}, {v.shape}")
    total = int(q.shape[0])
    step = total if not chunk_size else max(1, int(chunk_size))
    checkpointing = use_checkpoint and torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (q, k, v) + ((bias,) if bias is not None else ())
    )
    pieces = []
    for start in range(0, total, step):
        part = slice(start, start + step)
        args = (q[part], k[part], v[part], src, dst, bias, float(dropout_p))
        if checkpointing:
            pieces.append(activation_checkpoint(_edge_attention_chunk, *args, use_reentrant=False))
        else:
            pieces.append(_edge_attention_chunk(*args))
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)


class SparseSpatialAttention(nn.Module):
    """Multi-head Q/K/V attention over R-graph neighbours (plus self) at each step.

    ``edge_type_bias`` is one learnable logit per (relation type, head) so the
    bidirectional graph still distinguishes upstream from downstream roads.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        graph: SpatialGraph,
        attention_dropout: float = 0.0,
        use_qk_norm: bool = True,
        use_gated_attention: bool = False,
        use_edge_type_bias: bool = True,
        chunk_size: int | None = 64,
        use_checkpoint: bool = True,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = d_model // num_heads
        self.num_nodes = int(graph.num_nodes)
        self.attention_dropout = float(attention_dropout)
        self.chunk_size = chunk_size
        self.use_checkpoint = bool(use_checkpoint)
        self.register_buffer("src", graph.src.clone(), persistent=False)
        self.register_buffer("dst", graph.dst.clone(), persistent=False)
        self.register_buffer("edge_type", graph.edge_type.clone(), persistent=False)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=norm_eps) if use_qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim, eps=norm_eps) if use_qk_norm else None
        self.edge_type_bias = (
            nn.Parameter(torch.zeros(graph.num_edge_types, self.num_heads)) if use_edge_type_bias else None
        )
        self.gate = AttentionOutputGate(d_model) if use_gated_attention else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, width = x.shape
        if nodes != self.num_nodes:
            raise ValueError(f"node axis {nodes} != graph nodes {self.num_nodes}")
        q, k, v = self.qkv(x).view(batch * steps, nodes, 3, self.num_heads, self.head_dim).unbind(2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        bias = None if self.edge_type_bias is None else self.edge_type_bias[self.edge_type]
        attention = edge_softmax_attention(
            q,
            k,
            v,
            self.src,
            self.dst,
            bias=bias,
            dropout_p=self.attention_dropout if self.training else 0.0,
            chunk_size=self.chunk_size,
            use_checkpoint=self.use_checkpoint,
        )
        attention = attention.reshape(batch, steps, nodes, width)
        if self.gate is not None:
            attention = self.gate(x, attention)
        return self.out_proj(attention)


SPATIAL_ATTENTION: dict[str, Callable[..., nn.Module]] = {
    "sparse": SparseSpatialAttention,
}


def build_spatial_attention(name: str, **kwargs) -> nn.Module:
    if name not in SPATIAL_ATTENTION:
        raise ValueError(f"spatial_attention {name!r} not in {sorted(SPATIAL_ATTENTION)}")
    return SPATIAL_ATTENTION[name](**kwargs)
