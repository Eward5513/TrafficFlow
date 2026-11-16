"""Fuse four state histories with one 48-token Transformer.

Each road keeps four states and twelve observed steps. Those 48 tokens attend
only to each other. Roads do not share a token axis, and the module does not
build a graph. The TSF that produced the input stays outside this head.

Token order is state-major: ``token_index = state_index * T + time_index``.
The four readouts use the last step of each state, indexes 11, 23, 35, and 47
when ``T`` is 12.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from MSTT.model.attention import TemporalSelfAttention
from MSTT.model.ffn import SwiGLU

STATE_COUNT = 4


def token_index(state_index: int, time_index: int, input_window: int) -> int:
    """State-major index into the 4*T fusion sequence."""
    window = int(input_window)
    state = int(state_index)
    time = int(time_index)
    if window < 1 or not 0 <= state < STATE_COUNT or not 0 <= time < window:
        raise ValueError(f"token index out of range: state={state} time={time} T={window}")
    return state * window + time


def arrange_state_tokens(projected: list[torch.Tensor]) -> torch.Tensor:
    """Stack ``[B, T, N, D]`` state parts into ``[B, N, 4, T, D]``."""
    if len(projected) != STATE_COUNT:
        raise ValueError(f"fusion expects {STATE_COUNT} state parts, got {len(projected)}")
    tokens = torch.stack(projected, dim=3)
    return tokens.permute(0, 2, 3, 1, 4)


class StateTokenTransformerHead(nn.Module):
    """Project each 104-d branch, fuse 48 tokens, and read the last step of each state."""

    def __init__(
        self,
        branch_width: int,
        input_window: int,
        out_channels: int,
        *,
        token_dim: int = 128,
        num_heads: int = 4,
        ffn_hidden_dim: int = 176,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_qk_norm: bool = True,
        norm_eps: float = 1e-6,
        chunk_size: int | None = None,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.branch_width = int(branch_width)
        self.input_window = int(input_window)
        self.out_channels = int(out_channels)
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        self.ffn_hidden_dim = int(ffn_hidden_dim)
        self.chunk_size = None if chunk_size is None else int(chunk_size)
        self.use_checkpoint = bool(use_checkpoint)
        self.use_qk_norm = bool(use_qk_norm)
        if self.branch_width < 1 or self.input_window != 12 or self.out_channels != STATE_COUNT:
            raise ValueError("state token fusion is a 12-step, four-state head")
        if self.token_dim < 1 or self.num_heads < 1 or self.token_dim % self.num_heads != 0:
            raise ValueError(f"fusion_token_dim={self.token_dim} must be a positive multiple of heads={self.num_heads}")
        if self.ffn_hidden_dim < 1:
            raise ValueError("fusion FFN width must be positive")
        if self.chunk_size is not None and self.chunk_size < 1:
            raise ValueError("fusion chunk_size must be positive or omitted")
        if not self.use_qk_norm:
            raise ValueError("state token fusion keeps QK-Norm on")
        self.num_tokens = STATE_COUNT * self.input_window
        self.head_dim = self.token_dim // self.num_heads
        self.projectors = nn.ModuleList(
            nn.Linear(self.branch_width, self.token_dim) for _ in range(STATE_COUNT)
        )
        self.state_id = nn.Parameter(torch.empty(STATE_COUNT, self.token_dim))
        self.time_id = nn.Parameter(torch.empty(self.input_window, self.token_dim))
        nn.init.trunc_normal_(self.state_id, std=0.02)
        nn.init.trunc_normal_(self.time_id, std=0.02)
        self.attn_norm = nn.RMSNorm(self.token_dim, eps=norm_eps)
        self.attention = TemporalSelfAttention(
            self.token_dim,
            self.num_heads,
            attention_dropout=float(attention_dropout),
            use_qk_norm=True,
            use_rope=False,
            use_gated_attention=False,
            norm_eps=norm_eps,
            head_dim=self.head_dim,
        )
        self.ffn_norm = nn.RMSNorm(self.token_dim, eps=norm_eps)
        self.ffn = SwiGLU(self.token_dim, self.ffn_hidden_dim)
        self.output_norm = nn.RMSNorm(self.token_dim, eps=norm_eps)
        self.dropout = nn.Dropout(float(dropout))
        self.readouts = nn.ModuleList(nn.Linear(self.token_dim, 1) for _ in range(STATE_COUNT))

    def last_token_indices(self) -> list[int]:
        return [token_index(state, self.input_window - 1, self.input_window) for state in range(STATE_COUNT)]

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        tokens = self._tokens(hidden)
        batch, nodes = int(tokens.shape[0]), int(tokens.shape[1])
        flat = tokens.reshape(batch * nodes, self.num_tokens, self.token_dim)
        fused = self._fuse_flat(flat).reshape(batch, nodes, STATE_COUNT, self.input_window, self.token_dim)
        last = fused[:, :, :, -1, :]
        return torch.cat([readout(last[:, :, state, :]) for state, readout in enumerate(self.readouts)], dim=-1)

    def attention_map(self, hidden: torch.Tensor, road_index: torch.Tensor) -> torch.Tensor:
        """Eval-only head weights ``[B, N_subset, heads, 48, 48]``.

        Axis 2 is the attention head, not the traffic state. Token
        ``state_index * 12 + time_index`` is the state-major position.
        """
        if self.training:
            raise RuntimeError("fusion attention map is available in eval mode only")
        if torch.is_grad_enabled():
            raise RuntimeError("fusion attention map requires torch.no_grad()")
        if not torch.is_tensor(road_index):
            index = torch.as_tensor(list(road_index), dtype=torch.long, device=hidden.device)
        else:
            index = road_index.to(device=hidden.device, dtype=torch.long)
        index = index.reshape(-1)
        tokens = self._tokens(hidden).index_select(1, index)
        batch, chosen, states, steps, width = tokens.shape
        sequence = tokens.reshape(batch * chosen, states * steps, width)
        weights = self._attention_weights(self.attn_norm(sequence))
        return weights.reshape(batch, chosen, self.num_heads, self.num_tokens, self.num_tokens)

    def describe(self) -> dict[str, object]:
        from MSTT.config import STATE_CHANNELS

        return {
            "type": "state_token_transformer",
            "state_channels": list(STATE_CHANNELS),
            "branch_width": self.branch_width,
            "token_dim": self.token_dim,
            "tokens": self.num_tokens,
            "token_order": "state_major",
            "token_index": "state_index * input_window + time_index",
            "last_token_indices": self.last_token_indices(),
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "ffn": "SwiGLU",
            "ffn_hidden_dim": self.ffn_hidden_dim,
            "use_qk_norm": True,
            "use_rope": False,
            "position_ids": "learned_state_and_time_added_after_projection",
            "readout": "four independent Linear(token_dim, 1) on the last time token of each state",
            "chunk_size": self.chunk_size,
            "activation_checkpoint": self.use_checkpoint,
            "dropout": float(self.dropout.p),
            "attention_dropout": float(self.attention.attention_dropout),
            "shared_across_roads": True,
        }

    def _tokens(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 4:
            raise ValueError(f"fusion head expects [B, T, N, C], got {tuple(hidden.shape)}")
        batch, steps, nodes, width = hidden.shape
        expected = self.branch_width * STATE_COUNT
        if int(steps) != self.input_window or int(width) != expected:
            raise ValueError(
                f"fusion head expects T={self.input_window} and width={expected}, got {tuple(hidden.shape)}"
            )
        parts = list(hidden.split(self.branch_width, dim=-1))
        if len(parts) != STATE_COUNT or any(int(part.shape[-1]) != self.branch_width for part in parts):
            raise ValueError(f"fusion split of width {width} did not yield {STATE_COUNT} equal branches")
        projected = [projector(part) for projector, part in zip(self.projectors, parts)]
        tokens = arrange_state_tokens(projected)
        tokens = tokens + self.state_id.view(1, 1, STATE_COUNT, 1, self.token_dim)
        tokens = tokens + self.time_id.view(1, 1, 1, self.input_window, self.token_dim)
        return tokens

    def _fuse_flat(self, tokens: torch.Tensor) -> torch.Tensor:
        total = int(tokens.shape[0])
        step = total if self.chunk_size is None else self.chunk_size
        if step >= total:
            return self._run_block(tokens)
        parts = [self._run_block(tokens[start:start + step]) for start in range(0, total, step)]
        return torch.cat(parts, dim=0)

    def _run_block(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and torch.is_grad_enabled() and tokens.requires_grad:
            return activation_checkpoint(self._block, tokens, use_reentrant=False)
        return self._block(tokens)

    def _block(self, tokens: torch.Tensor) -> torch.Tensor:
        # TemporalSelfAttention is [B, T, N, D]. Here T is the 48-token axis and N is 1.
        update = self.attention(self.attn_norm(tokens).unsqueeze(2)).squeeze(2)
        hidden = tokens + self.dropout(update)
        hidden = hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))
        return self.output_norm(hidden)

    def _attention_weights(self, sequence: torch.Tensor) -> torch.Tensor:
        attention = self.attention
        rows, steps, _width = sequence.shape
        qkv = attention.qkv(sequence).view(rows, steps, 3, attention.num_heads, attention.head_dim)
        query, key, _value = qkv.permute(2, 0, 3, 1, 4)
        if attention.q_norm is not None:
            query = attention.q_norm(query)
            key = attention.k_norm(key)
        scale = 1.0 / math.sqrt(attention.head_dim)
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
        return torch.softmax(scores, dim=-1)
