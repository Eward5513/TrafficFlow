"""DropPath, TokenEmbedding, MLP, and encoder block from original PDFormer.py."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from reimplementation.pdformer.model.attention import STSelfAttention


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Original ``drop_path``: per-batch Bernoulli, then divide by keep_prob."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float | None = None) -> None:
        super().__init__()
        self.drop_prob = 0.0 if drop_prob is None else float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class TokenEmbedding(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, norm_layer: Callable[..., nn.Module] | None = None) -> None:
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(x)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class STEncoderBlock(nn.Module):
    """Pre-norm (PeMS04/08) or post-norm encoder block.

    ``type_ln='pre'``::

        x = x + DropPath(STAttn(LN(x), ...))
        x = x + DropPath(MLP(LN(x)))

    MLP hidden size is ``int(dim * mlp_ratio)``.
    """

    def __init__(
        self,
        dim: int,
        s_attn_size: int,
        t_attn_size: int,
        geo_num_heads: int = 4,
        sem_num_heads: int = 2,
        t_num_heads: int = 2,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        type_ln: str = "pre",
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        if type_ln not in {"pre", "post"}:
            raise ValueError(f"type_ln must be 'pre' or 'post', got {type_ln}")
        self.type_ln = type_ln
        self.norm1 = norm_layer(dim)
        self.st_attn = STSelfAttention(
            dim,
            s_attn_size,
            t_attn_size,
            geo_num_heads=geo_num_heads,
            sem_num_heads=sem_num_heads,
            t_num_heads=t_num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            output_dim=output_dim,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_patterns: torch.Tensor,
        pattern_keys: torch.Tensor,
        geo_mask: torch.Tensor | None = None,
        sem_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        if self.type_ln == "pre":
            attn_out = self.st_attn(
                self.norm1(x),
                x_patterns,
                pattern_keys,
                geo_mask=geo_mask,
                sem_mask=sem_mask,
                return_attention=return_attention,
            )
            if return_attention:
                attn_x, attn_trace = attn_out
            else:
                attn_x, attn_trace = attn_out, None
            x = x + self.drop_path(attn_x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            attn_out = self.st_attn(
                x,
                x_patterns,
                pattern_keys,
                geo_mask=geo_mask,
                sem_mask=sem_mask,
                return_attention=return_attention,
            )
            if return_attention:
                attn_x, attn_trace = attn_out
            else:
                attn_x, attn_trace = attn_out, None
            x = self.norm1(x + self.drop_path(attn_x))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        if return_attention:
            return x, attn_trace
        return x
