"""Centered joint space-time attention over a short observed-time window.

For each time ``t`` the queries are the ``N`` roads at that step. Keys and
values are every road at the valid times around ``t``, laid out time-major
then node-minor. One softmax covers that whole set. Boundary steps use the
two available times; the missing outside step is omitted, not copied.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from MSTT.model.attention import AttentionOutputGate
from MSTT.model.rope import RotaryEmbedding


def valid_time_indices(t: int, steps: int, window: int) -> list[int]:
    """Centered window clipped to ``0 .. steps-1``, without repeating an edge step."""
    if int(window) < 1 or int(window) % 2 == 0:
        raise ValueError(f"joint temporal window must be a positive odd integer, got {window}")
    if int(steps) < 1 or not 0 <= int(t) < int(steps):
        raise ValueError(f"time index {t} is outside 0..{int(steps) - 1}")
    radius = int(window) // 2
    start = max(0, int(t) - radius)
    end = min(int(steps), int(t) + radius + 1)
    return list(range(start, end))


def _attend_queries(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    query_chunk: int,
) -> torch.Tensor:
    """SDPA with ``Q`` length N and ``K`` length 2N or 3N.

    Query chunks share one softmax over the full key set. The key axis is
    never split into separate softmaxes, and no ``(3N) x (3N)`` score is built.
    """
    if q.shape[-1] != k.shape[-1] or k.shape != v.shape or q.ndim != 4 or k.ndim != 4:
        raise ValueError(f"expected q [B,H,N,d] and k/v [B,H,K,d], got {q.shape}, {k.shape}, {v.shape}")
    if int(q.shape[0]) != int(k.shape[0]) or int(q.shape[1]) != int(k.shape[1]):
        raise ValueError("q and k/v batch and head axes must match")
    width = int(q.shape[2])
    step = width if int(query_chunk) <= 0 else max(1, int(query_chunk))
    if step >= width:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=float(dropout_p), is_causal=False)
    parts = []
    for start in range(0, width, step):
        parts.append(
            F.scaled_dot_product_attention(
                q[:, :, start : start + step],
                k,
                v,
                dropout_p=float(dropout_p),
                is_causal=False,
            )
        )
    return torch.cat(parts, dim=2)


class JointSTAttention(nn.Module):
    """Shared-QKV joint attention. Input and output are ``[B, T, N, D]``."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        window: int = 3,
        alignment: str = "centered",
        query_chunk: int = 256,
        use_checkpoint: bool = True,
        attention_dropout: float = 0.0,
        use_qk_norm: bool = True,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        use_gated_attention: bool = False,
        norm_eps: float = 1e-6,
        head_dim: int | None = None,
    ) -> None:
        super().__init__()
        if alignment != "centered":
            raise ValueError(f"joint temporal alignment {alignment!r} is not centered")
        if int(window) < 1 or int(window) % 2 == 0:
            raise ValueError(f"joint temporal window must be a positive odd integer, got {window}")
        if int(query_chunk) < 1:
            raise ValueError("joint query chunk must be positive")
        self.window = int(window)
        self.alignment = str(alignment)
        self.query_chunk = int(query_chunk)
        self.use_checkpoint = bool(use_checkpoint)
        self.num_heads = int(num_heads)
        self.d_model = int(d_model)
        if head_dim is None:
            if d_model % num_heads != 0:
                raise ValueError("d_model must be divisible by num_heads")
            self.head_dim = d_model // num_heads
        else:
            if int(head_dim) < 1:
                raise ValueError("head_dim must be positive")
            self.head_dim = int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.attention_dropout = float(attention_dropout)
        self.qkv = nn.Linear(d_model, 3 * self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=norm_eps) if use_qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim, eps=norm_eps) if use_qk_norm else None
        self.rope = RotaryEmbedding(self.head_dim, rope_base) if use_rope else None
        self.gate = AttentionOutputGate(d_model, self.inner_dim) if use_gated_attention else None

    def key_time_indices(self, steps: int) -> list[list[int]]:
        return [valid_time_indices(t, steps, self.window) for t in range(int(steps))]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, _width = x.shape
        q, k, v = self.qkv(x).view(batch, steps, nodes, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.rope is not None:
            # Positions are the window offsets 0..T-1, shared by every road at that step.
            q = self.rope.rotate_time(q.permute(0, 1, 3, 2, 4)).permute(0, 1, 3, 2, 4)
            k = self.rope.rotate_time(k.permute(0, 1, 3, 2, 4)).permute(0, 1, 3, 2, 4)
        dropout_p = self.attention_dropout if self.training else 0.0
        checkpointing = self.use_checkpoint and torch.is_grad_enabled() and q.requires_grad
        outputs = []
        for t, times in enumerate(self.key_time_indices(steps)):
            count = len(times)
            index = torch.as_tensor(times, device=k.device)
            selected_k = k.index_select(2, index)
            selected_v = v.index_select(2, index)
            kk = selected_k.reshape(batch, self.num_heads, count * nodes, self.head_dim)
            vv = selected_v.reshape(batch, self.num_heads, count * nodes, self.head_dim)
            if int(kk.shape[2]) != count * nodes:
                raise RuntimeError("joint key axis is not time-major valid nodes")
            qq = q[:, :, t]
            if checkpointing:
                attended = activation_checkpoint(
                    _attend_queries,
                    qq,
                    kk,
                    vv,
                    dropout_p,
                    self.query_chunk,
                    use_reentrant=False,
                )
            else:
                attended = _attend_queries(qq, kk, vv, dropout_p, self.query_chunk)
            outputs.append(attended)
        attention = torch.stack(outputs, dim=2).permute(0, 2, 3, 1, 4).reshape(batch, steps, nodes, self.inner_dim)
        if self.gate is not None:
            attention = self.gate(x, attention)
        return self.out_proj(attention)
