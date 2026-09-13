"""Official multi-head attention from ``reference/STAEformer/model/STAEformer.py``.

Attention is always over the second-to-last axis. Scale is ``sqrt(head_dim)``.
Heads are split on the feature axis and stacked on the batch axis, matching
the original ``torch.split`` / ``torch.cat`` layout.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from reimplementation.common.errors import ReimplementationError

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
        "mask_geo",
        "geo_mask",
        "sem_mask",
    }
)


def refuse_graph_forward_kwargs(kwargs: dict[str, Any]) -> None:
    hit = sorted(GRAPH_FORWARD_KEYS.intersection(kwargs))
    if hit:
        raise ReimplementationError(
            f"STAEformer forward does not accept graph inputs {hit}"
        )


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is ``model_dim``).

    Original: ``FC_Q/K/V`` with bias, no attention dropout, optional lower-triangular
    causal mask. Official STAEformer configs keep ``mask=False``.
    """

    def __init__(self, model_dim: int, num_heads: int = 8, mask: bool = False) -> None:
        super().__init__()
        if int(num_heads) < 1:
            raise ReimplementationError("num_heads must be positive")
        if int(model_dim) % int(num_heads) != 0:
            raise ReimplementationError(
                f"model_dim {model_dim} is not divisible by num_heads {num_heads}; "
                "refusing to retune heads or embeddings"
            )
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.mask = bool(mask)
        self.head_dim = int(model_dim) // int(num_heads)
        self.FC_Q = nn.Linear(self.model_dim, self.model_dim)
        self.FC_K = nn.Linear(self.model_dim, self.model_dim)
        self.FC_V = nn.Linear(self.model_dim, self.model_dim)
        self.out_proj = nn.Linear(self.model_dim, self.model_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if query.ndim < 3 or key.ndim < 3 or value.ndim < 3:
            raise ReimplementationError("attention tensors must be at least 3D")
        if not torch.isfinite(query).all() or not torch.isfinite(key).all() or not torch.isfinite(value).all():
            raise ReimplementationError("attention inputs contain NaN or inf")
        batch_size = int(query.shape[0])
        tgt_length = int(query.shape[-2])
        src_length = int(key.shape[-2])
        if int(key.shape[-2]) != int(value.shape[-2]):
            raise ReimplementationError("original STAEformer requires src length == K length == V length")
        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)
        key = key.transpose(-1, -2)
        attn_score = (query @ key) / (self.head_dim**0.5)
        if self.mask:
            causal = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()
            attn_score.masked_fill_(~causal, -torch.inf)
        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        return self.out_proj(out)
