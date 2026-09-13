"""Spatiotemporal self-attention with geographic, semantic, and temporal heads.

Original class: ``STSelfAttention``. Concatenation order is
``[temporal, geographic, semantic]``. Boolean masks use ``True`` = mask out
(``masked_fill_(-inf)``). There is no causal temporal mask.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from reimplementation.common.errors import ReimplementationError


def assert_head_partition(
    embed_dim: int,
    geo_num_heads: int,
    sem_num_heads: int,
    t_num_heads: int,
) -> tuple[int, int]:
    total = int(geo_num_heads) + int(sem_num_heads) + int(t_num_heads)
    if total <= 0:
        raise ReimplementationError("PDFormer needs at least one attention head")
    if int(embed_dim) % total != 0:
        raise ReimplementationError(
            f"embed_dim={embed_dim} is not divisible by geo+sem+t heads={total}; "
            "refusing to change embed_dim or head counts"
        )
    head_dim = int(embed_dim) // total
    return total, head_dim


class STSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        s_attn_size: int,
        t_attn_size: int,
        geo_num_heads: int = 4,
        sem_num_heads: int = 2,
        t_num_heads: int = 2,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        total_heads, head_dim = assert_head_partition(dim, geo_num_heads, sem_num_heads, t_num_heads)
        self.geo_num_heads = int(geo_num_heads)
        self.sem_num_heads = int(sem_num_heads)
        self.t_num_heads = int(t_num_heads)
        self.num_heads = total_heads
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.s_attn_size = int(s_attn_size)
        self.t_attn_size = int(t_attn_size)
        self.geo_ratio = self.geo_num_heads / float(self.num_heads)
        self.sem_ratio = self.sem_num_heads / float(self.num_heads)
        self.t_ratio = 1.0 - self.geo_ratio - self.sem_ratio
        self.output_dim = int(output_dim)
        self.geo_dim = int(dim * self.geo_ratio)
        self.sem_dim = int(dim * self.sem_ratio)
        self.t_dim = int(dim * self.t_ratio)
        if self.geo_dim != self.geo_num_heads * self.head_dim:
            raise ReimplementationError(
                f"geo dim int(dim*geo_ratio)={self.geo_dim} != heads*head_dim="
                f"{self.geo_num_heads * self.head_dim}"
            )
        if self.sem_dim != self.sem_num_heads * self.head_dim:
            raise ReimplementationError("semantic head dimension does not match embed_dim partition")
        if self.t_dim != self.t_num_heads * self.head_dim:
            raise ReimplementationError("temporal head dimension does not match embed_dim partition")

        self.pattern_q_linears = nn.ModuleList(
            [nn.Linear(dim, self.geo_dim) for _ in range(self.output_dim)]
        )
        self.pattern_k_linears = nn.ModuleList(
            [nn.Linear(dim, self.geo_dim) for _ in range(self.output_dim)]
        )
        self.pattern_v_linears = nn.ModuleList(
            [nn.Linear(dim, self.geo_dim) for _ in range(self.output_dim)]
        )
        self.geo_q_conv = nn.Conv2d(dim, self.geo_dim, kernel_size=1, bias=qkv_bias)
        self.geo_k_conv = nn.Conv2d(dim, self.geo_dim, kernel_size=1, bias=qkv_bias)
        self.geo_v_conv = nn.Conv2d(dim, self.geo_dim, kernel_size=1, bias=qkv_bias)
        self.geo_attn_drop = nn.Dropout(attn_drop)
        self.sem_q_conv = nn.Conv2d(dim, self.sem_dim, kernel_size=1, bias=qkv_bias)
        self.sem_k_conv = nn.Conv2d(dim, self.sem_dim, kernel_size=1, bias=qkv_bias)
        self.sem_v_conv = nn.Conv2d(dim, self.sem_dim, kernel_size=1, bias=qkv_bias)
        self.sem_attn_drop = nn.Dropout(attn_drop)
        self.t_q_conv = nn.Conv2d(dim, self.t_dim, kernel_size=1, bias=qkv_bias)
        self.t_k_conv = nn.Conv2d(dim, self.t_dim, kernel_size=1, bias=qkv_bias)
        self.t_v_conv = nn.Conv2d(dim, self.t_dim, kernel_size=1, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _apply_mask(self, attn: torch.Tensor, mask: torch.Tensor | None, *, name: str) -> torch.Tensor:
        if mask is None:
            return attn
        if mask.dtype != torch.bool:
            raise ReimplementationError(f"{name} mask must be bool, True=mask-out")
        filled = attn.masked_fill(mask, float("-inf"))
        all_masked = torch.isinf(filled).all(dim=-1)
        if bool(all_masked.any()):
            raise ReimplementationError(
                f"{name} attention has an all-masked row; refusing to softmax NaN or unmask"
            )
        return filled

    def forward(
        self,
        x: torch.Tensor,
        x_patterns: torch.Tensor,
        pattern_keys: torch.Tensor,
        geo_mask: torch.Tensor | None = None,
        sem_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        batch, time, nodes, dim = x.shape
        if dim != self.proj.in_features:
            raise ReimplementationError(f"attention input dim {dim} != embed_dim {self.proj.in_features}")

        t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_q = t_q.reshape(batch, nodes, time, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(batch, nodes, time, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(batch, nodes, time, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale
        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)
        t_x = (t_attn @ t_v).transpose(2, 3).reshape(batch, nodes, time, self.t_dim).transpose(1, 2)

        geo_q = self.geo_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_k = self.geo_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        pattern_weights = []
        for channel in range(self.output_dim):
            pattern_q = self.pattern_q_linears[channel](x_patterns[..., channel])
            pattern_k = self.pattern_k_linears[channel](pattern_keys[..., channel])
            pattern_v = self.pattern_v_linears[channel](pattern_keys[..., channel])
            pattern_attn = (pattern_q @ pattern_k.transpose(-2, -1)) * self.scale
            pattern_attn = pattern_attn.softmax(dim=-1)
            pattern_weights.append(pattern_attn)
            geo_k = geo_k + pattern_attn @ pattern_v
        geo_v = self.geo_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_q = geo_q.reshape(batch, time, nodes, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_k = geo_k.reshape(batch, time, nodes, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_v = geo_v.reshape(batch, time, nodes, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_attn = (geo_q @ geo_k.transpose(-2, -1)) * self.scale
        geo_attn = self._apply_mask(geo_attn, geo_mask, name="geographic")
        geo_attn = geo_attn.softmax(dim=-1)
        geo_attn = self.geo_attn_drop(geo_attn)
        geo_x = (geo_attn @ geo_v).transpose(2, 3).reshape(batch, time, nodes, self.geo_dim)

        sem_q = self.sem_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_k = self.sem_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_v = self.sem_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_q = sem_q.reshape(batch, time, nodes, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_k = sem_k.reshape(batch, time, nodes, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_v = sem_v.reshape(batch, time, nodes, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_attn = (sem_q @ sem_k.transpose(-2, -1)) * self.scale
        sem_attn = self._apply_mask(sem_attn, sem_mask, name="semantic")
        sem_attn = sem_attn.softmax(dim=-1)
        sem_attn = self.sem_attn_drop(sem_attn)
        sem_x = (sem_attn @ sem_v).transpose(2, 3).reshape(batch, time, nodes, self.sem_dim)

        merged = torch.cat([t_x, geo_x, sem_x], dim=-1)
        if int(merged.size(-1)) != int(dim):
            raise ReimplementationError(
                f"concatenated head dim {tuple(merged.shape)} != embed_dim {dim}"
            )
        out = self.proj_drop(self.proj(merged))
        if not return_attention:
            return out
        trace: dict[str, Any] = {
            "temporal_attention_shape": list(t_attn.shape),
            "geographic_attention_shape": list(geo_attn.shape),
            "semantic_attention_shape": list(sem_attn.shape),
            "concat_order": ["temporal", "geographic", "semantic"],
            "temporal_used": True,
            "geographic_used": True,
            "semantic_used": True,
            "pattern_used": True,
            "temporal_attention": t_attn,
            "geographic_attention": geo_attn,
            "semantic_attention": sem_attn,
            "pattern_attention": pattern_weights[0] if pattern_weights else None,
        }
        return out, trace
