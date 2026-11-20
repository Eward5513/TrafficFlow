"""Second readout that rebuilds the last step from the first 11 auxiliary predictions.

Path A is the existing flatten head on the observed window. This module does
not own that head. It embeds the 11 standardized history predictions, runs one
independent T→S→FFN block, and reads path B with its own flatten head. A
per-road cross-attention then lets the current hidden state and path A look
up those history tokens. A four-state gate mixes A and B. The caller projects
the mixture; this module does not add a residual onto A.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from MSTT.model.head import PredictionHead
from MSTT.model.transformer_block import build_factorized_block

HISTORY_STEPS = 11
STATE_CHANNELS = 4
HISTORY_WIDTH = 64
HISTORY_HEADS = 4
HISTORY_FFN = 176
CROSS_HEADS = 4
CROSS_HEAD_DIM = 16
CROSS_WIDTH = CROSS_HEADS * CROSS_HEAD_DIM
GATE_HIDDEN = 64
OBSERVED_WIDTH = 128


def mix_branch_predictions(
    branch_a: torch.Tensor, branch_b: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """``gate`` is path B's share. ``0`` returns A and ``1`` returns B."""
    if branch_a.shape != branch_b.shape or gate.shape != branch_a.shape:
        raise ValueError(
            f"A {tuple(branch_a.shape)}, B {tuple(branch_b.shape)}, and gate {tuple(gate.shape)} must match"
        )
    return (1.0 - gate) * branch_a + gate * branch_b


class DualPathHistoryTSF(nn.Module):
    """History TSF, path B, current-to-history attention, and a four-state gate.

    ``encode_history`` reads only ``[B, 11, N, 4]``. It does not accept the
    last hidden state, path A, the 12th auxiliary step, or any label.
    """

    def __init__(self, config, graph) -> None:
        super().__init__()
        num_nodes = int(config.num_nodes)
        if num_nodes < 1:
            raise ValueError(f"history TSF needs a positive node count, got {num_nodes}")
        if int(config.d_model) != OBSERVED_WIDTH or int(config.input_window) != 12:
            raise ValueError("dual path history TSF requires the 128-d 12-step observed encoder")
        hist_cfg = _history_block_config(config)
        self.num_nodes = num_nodes
        self.value_proj = nn.Linear(STATE_CHANNELS, HISTORY_WIDTH)
        self.time_table = nn.Parameter(torch.empty(HISTORY_STEPS, HISTORY_WIDTH))
        self.road_table = nn.Parameter(torch.empty(num_nodes, HISTORY_WIDTH))
        nn.init.normal_(self.time_table, mean=0.0, std=0.02)
        nn.init.normal_(self.road_table, mean=0.0, std=0.02)
        self.block = build_factorized_block(hist_cfg, graph)
        self.history_head = PredictionHead(
            HISTORY_WIDTH, HISTORY_STEPS, STATE_CHANNELS, head_type="flatten", norm_eps=float(config.norm_eps)
        )
        self.q_proj = nn.Linear(OBSERVED_WIDTH + STATE_CHANNELS, CROSS_WIDTH)
        self.k_proj = nn.Linear(HISTORY_WIDTH, CROSS_WIDTH)
        self.v_proj = nn.Linear(HISTORY_WIDTH, CROSS_WIDTH)
        self.cross_out = nn.Linear(CROSS_WIDTH, HISTORY_WIDTH)
        self.gate_in = nn.Linear(OBSERVED_WIDTH + HISTORY_WIDTH + STATE_CHANNELS + STATE_CHANNELS, GATE_HIDDEN)
        self.gate_out = nn.Linear(GATE_HIDDEN, STATE_CHANNELS)
        nn.init.zeros_(self.gate_out.weight)
        nn.init.zeros_(self.gate_out.bias)

    def parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.parameters() if param.requires_grad))

    def describe(self) -> dict[str, Any]:
        groups = {
            "value_projection": _numel(self.value_proj),
            "time_table": int(self.time_table.numel()),
            "road_table": int(self.road_table.numel()),
            "history_block": _numel(self.block),
            "branch_b_head": _numel(self.history_head),
            "cross_attention": _numel(self.q_proj) + _numel(self.k_proj) + _numel(self.v_proj) + _numel(self.cross_out),
            "gate": _numel(self.gate_in) + _numel(self.gate_out),
        }
        temporal = self.block.temporal
        spatial = self.block.spatial
        return {
            "enabled": True,
            "mode": "dual_path_history_tsf",
            "history_steps": HISTORY_STEPS,
            "num_layers": 1,
            "d_model": HISTORY_WIDTH,
            "num_heads": HISTORY_HEADS,
            "temporal_num_heads": int(temporal.num_heads),
            "spatial_num_heads": int(spatial.num_heads),
            "head_dim": int(temporal.head_dim),
            "ffn": "SwiGLU",
            "ffn_hidden_dim": int(self.block.ffn.w_gate.out_features),
            "ffn_count": 1,
            "block_order": ["T", "S", "FFN"],
            "pre_rms_norm": True,
            "qk_norm": temporal.q_norm is not None,
            "temporal_rope": temporal.rope is not None,
            "spatial_attention": "global",
            "spatial_chunk_size": spatial.chunk_size,
            "spatial_checkpoint": bool(spatial.use_checkpoint),
            "cross_attention_heads": CROSS_HEADS,
            "cross_attention_head_dim": CROSS_HEAD_DIM,
            "cross_attention_dropout": 0.0,
            "cross_attention_scale": float(CROSS_HEAD_DIM) ** 0.5,
            "query_source": "last_hidden_and_branch_a",
            "key_source": "history_tsf_output",
            "value_source": "history_tsf_output",
            "spatial_mixing": "history_tsf_global_spatial",
            "gate_hidden_dim": GATE_HIDDEN,
            "gate_output_zero_initialized": True,
            "initial_gate": 0.5,
            "branch_b_head": self.history_head.describe(),
            "parameter_groups": groups,
            "parameters": self.parameter_count(),
        }

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        """Map standardized steps 0..10 to ``[B, 11, N, 64]``. Labels and path A are not inputs."""
        if history.ndim != 4:
            raise ValueError(f"history must be [B, 11, N, 4], got {tuple(history.shape)}")
        batch, steps, nodes, channels = (int(size) for size in history.shape)
        if steps != HISTORY_STEPS or channels != STATE_CHANNELS or nodes != self.num_nodes:
            raise ValueError(
                f"history must be [B, {HISTORY_STEPS}, {self.num_nodes}, {STATE_CHANNELS}], got {tuple(history.shape)}"
            )
        if batch < 1:
            raise ValueError("history batch must be non-empty")
        tokens = self.value_proj(history)
        tokens = tokens + self.time_table.to(dtype=tokens.dtype).view(1, HISTORY_STEPS, 1, HISTORY_WIDTH)
        tokens = tokens + self.road_table.to(dtype=tokens.dtype).view(1, 1, self.num_nodes, HISTORY_WIDTH)
        return self.block(tokens)

    def forward(
        self, hidden: torch.Tensor, auxiliary: torch.Tensor, branch_a: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if hidden.ndim != 4 or int(hidden.shape[1]) != 12 or int(hidden.shape[-1]) != OBSERVED_WIDTH:
            raise ValueError(f"observed hidden must be [B, 12, N, {OBSERVED_WIDTH}], got {tuple(hidden.shape)}")
        if auxiliary.ndim != 4 or int(auxiliary.shape[1]) != 12:
            raise ValueError(f"auxiliary prediction must be [B, 12, N, 4], got {tuple(auxiliary.shape)}")
        if branch_a.shape != (hidden.shape[0], hidden.shape[2], STATE_CHANNELS):
            raise ValueError(
                f"branch A must be [B, N, 4] matching hidden, got {tuple(branch_a.shape)} for {tuple(hidden.shape)}"
            )
        # Step 11 of the auxiliary head is not a history token.
        history_state = self.encode_history(auxiliary[:, :11])
        branch_b = self.history_head(history_state)
        context = self._cross_attend(hidden[:, -1], branch_a, history_state)
        gate = self._gate(hidden[:, -1], context, branch_a, branch_b)
        return {
            "raw_pred": mix_branch_predictions(branch_a, branch_b, gate),
            "branch_a_pred": branch_a,
            "branch_b_pred": branch_b,
            "branch_gate": gate,
        }

    def _cross_attend(self, last_hidden: torch.Tensor, branch_a: torch.Tensor, history_state: torch.Tensor) -> torch.Tensor:
        batch, nodes, _width = (int(last_hidden.shape[0]), int(last_hidden.shape[1]), int(last_hidden.shape[2]))
        query = self.q_proj(torch.cat((last_hidden, branch_a), dim=-1))
        sequence = history_state.permute(0, 2, 1, 3).contiguous()
        key = self.k_proj(sequence)
        value = self.v_proj(sequence)
        flat = batch * nodes
        query = query.view(flat, CROSS_HEADS, CROSS_HEAD_DIM)
        key = key.view(flat, HISTORY_STEPS, CROSS_HEADS, CROSS_HEAD_DIM)
        value = value.view(flat, HISTORY_STEPS, CROSS_HEADS, CROSS_HEAD_DIM)
        scores = torch.einsum("bhd,bthd->bht", query, key) / (CROSS_HEAD_DIM ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        mixed = torch.einsum("bht,bthd->bhd", weights, value).reshape(flat, CROSS_WIDTH)
        return self.cross_out(mixed).view(batch, nodes, HISTORY_WIDTH)

    def _gate(
        self,
        last_hidden: torch.Tensor,
        context: torch.Tensor,
        branch_a: torch.Tensor,
        branch_b: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat((last_hidden, context, branch_a, branch_b), dim=-1)
        return torch.sigmoid(self.gate_out(F.silu(self.gate_in(features))))


def _history_block_config(config):
    from dataclasses import replace

    return replace(
        config,
        d_model=HISTORY_WIDTH,
        input_window=HISTORY_STEPS,
        num_heads=HISTORY_HEADS,
        temporal_num_heads=HISTORY_HEADS,
        spatial_num_heads=HISTORY_HEADS,
        attention_head_dim=None,
        ffn_hidden_dim=HISTORY_FFN,
        num_layers=1,
    )


def _numel(module: nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters() if param.requires_grad))
