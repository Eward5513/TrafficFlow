"""Per-road fusion of the last hidden state with 11 standardized history predictions.

Each sample and each road runs the same module. Attention is over the 11
history steps only, so the module never builds a road-by-road matrix and
never shares attention weights across samples.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

HISTORY_STEPS = 11
STATE_CHANNELS = 4
POSITION_DIM = 8
D_MODEL = 128
ATTENTION_DIM = 64
NUM_HEADS = 4
HEAD_DIM = 16
GATE_HIDDEN = 32
RESIDUAL_HIDDEN = 32
TOKEN_DIM = STATE_CHANNELS + POSITION_DIM


def expected_parameter_count() -> int:
    """Bias-bearing linears plus the position table. No extra trainable layers."""
    return (
        (D_MODEL * ATTENTION_DIM + ATTENTION_DIM)
        + (TOKEN_DIM * ATTENTION_DIM + ATTENTION_DIM)
        + (TOKEN_DIM * ATTENTION_DIM + ATTENTION_DIM)
        + (ATTENTION_DIM * D_MODEL + D_MODEL)
        + ((D_MODEL + D_MODEL) * GATE_HIDDEN + GATE_HIDDEN)
        + (GATE_HIDDEN * D_MODEL + D_MODEL)
        + ((D_MODEL + D_MODEL) * RESIDUAL_HIDDEN + RESIDUAL_HIDDEN)
        + (RESIDUAL_HIDDEN * STATE_CHANNELS + STATE_CHANNELS)
        + (HISTORY_STEPS * POSITION_DIM)
    )


class HistoryPredictionFusion(nn.Module):
    """Correct the last-step raw prediction from reconstructed history.

    ``h`` is ``[B, N, 128]`` and ``p_hist`` is ``[B, N, 11, 4]``, earliest
    step first, in the standardized target scale. The returned delta has
    shape ``[B, N, 4]``. The last residual linear starts at zero, so delta
    is zero until that layer moves.
    """

    def __init__(self) -> None:
        super().__init__()
        if ATTENTION_DIM != NUM_HEADS * HEAD_DIM:
            raise RuntimeError("history fusion attention width must equal heads times head dim")
        self.q_proj = nn.Linear(D_MODEL, ATTENTION_DIM)
        self.k_proj = nn.Linear(TOKEN_DIM, ATTENTION_DIM)
        self.v_proj = nn.Linear(TOKEN_DIM, ATTENTION_DIM)
        self.out_proj = nn.Linear(ATTENTION_DIM, D_MODEL)
        self.gate_in = nn.Linear(D_MODEL + D_MODEL, GATE_HIDDEN)
        self.gate_out = nn.Linear(GATE_HIDDEN, D_MODEL)
        self.res_in = nn.Linear(D_MODEL + D_MODEL, RESIDUAL_HIDDEN)
        self.res_out = nn.Linear(RESIDUAL_HIDDEN, STATE_CHANNELS)
        self.position = nn.Parameter(torch.empty(HISTORY_STEPS, POSITION_DIM))
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        nn.init.zeros_(self.res_out.weight)
        nn.init.zeros_(self.res_out.bias)

    def parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.parameters() if param.requires_grad))

    def describe(self) -> dict[str, object]:
        return {
            "enabled": True,
            "mode": "gated_cross_attention",
            "query_source": "last_step_hidden",
            "key_source": "standardized_history_prediction_plus_position",
            "value_source": "standardized_history_prediction_plus_position",
            "history_steps": HISTORY_STEPS,
            "position_dim": POSITION_DIM,
            "query_dim": ATTENTION_DIM,
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "attention_dropout": 0.0,
            "gate_hidden_dim": GATE_HIDDEN,
            "gate_output_dim": D_MODEL,
            "residual_hidden_dim": RESIDUAL_HIDDEN,
            "residual_output_dim": STATE_CHANNELS,
            "parameters": self.parameter_count(),
            "residual_output_zero_initialized": True,
        }

    def forward(self, h: torch.Tensor, p_hist: torch.Tensor) -> torch.Tensor:
        delta, _, _, _ = self._fuse(h, p_hist)
        return delta

    def forward_details(self, h: torch.Tensor, p_hist: torch.Tensor) -> dict[str, torch.Tensor]:
        """Diagnostics for tests. Training and prediction dicts do not carry these tensors."""
        delta, alpha, gate, context = self._fuse(h, p_hist)
        return {"delta": delta, "alpha": alpha, "g": gate, "context": context}

    def _fuse(
        self, h: torch.Tensor, p_hist: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if h.ndim != 3 or p_hist.ndim != 4:
            raise ValueError(
                f"h must be [B, N, {D_MODEL}] and p_hist [B, N, {HISTORY_STEPS}, {STATE_CHANNELS}], "
                f"got {tuple(h.shape)} and {tuple(p_hist.shape)}"
            )
        batch, nodes, width = (int(h.shape[0]), int(h.shape[1]), int(h.shape[2]))
        if width != D_MODEL or int(p_hist.shape[0]) != batch or int(p_hist.shape[1]) != nodes:
            raise ValueError(
                f"h and p_hist must share B,N and h must be {D_MODEL}-d, "
                f"got h={tuple(h.shape)} p_hist={tuple(p_hist.shape)}"
            )
        if int(p_hist.shape[2]) != HISTORY_STEPS or int(p_hist.shape[3]) != STATE_CHANNELS:
            raise ValueError(
                f"p_hist must be [B, N, {HISTORY_STEPS}, {STATE_CHANNELS}], got {tuple(p_hist.shape)}"
            )
        flat = batch * nodes
        hidden = h.reshape(flat, D_MODEL)
        history = p_hist.reshape(flat, HISTORY_STEPS, STATE_CHANNELS)
        position = self.position.to(dtype=history.dtype).unsqueeze(0).expand(flat, -1, -1)
        tokens = torch.cat((history, position), dim=-1)
        query = self.q_proj(hidden).view(flat, NUM_HEADS, HEAD_DIM)
        key = self.k_proj(tokens).view(flat, HISTORY_STEPS, NUM_HEADS, HEAD_DIM)
        value = self.v_proj(tokens).view(flat, HISTORY_STEPS, NUM_HEADS, HEAD_DIM)
        scores = torch.einsum("bhd,bthd->bht", query, key) / (HEAD_DIM ** 0.5)
        alpha = torch.softmax(scores, dim=-1)
        mixed = torch.einsum("bht,bthd->bhd", alpha, value).reshape(flat, ATTENTION_DIM)
        context = self.out_proj(mixed)
        gate = torch.sigmoid(self.gate_out(F.silu(self.gate_in(torch.cat((hidden, context), dim=-1)))))
        fused = hidden + gate * context
        delta = self.res_out(F.silu(self.res_in(torch.cat((fused, hidden), dim=-1))))
        return (
            delta.view(batch, nodes, STATE_CHANNELS),
            alpha.view(batch, nodes, NUM_HEADS, HISTORY_STEPS),
            gate.view(batch, nodes, D_MODEL),
            context.view(batch, nodes, D_MODEL),
        )
