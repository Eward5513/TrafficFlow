"""Gate the four state embeddings with speed and the observation count.

``AuxiliaryStateGate`` is the shared map
``g = 2 * sigmoid(Linear(32, 32)(SiLU(Linear(2, 32)(aux))))``.
``SplitAuxiliaryStateGate`` encodes speed and the observation count with two
separate ``1→8→8`` MLPs, concatenates those 8-d features, and applies one
joint ``Linear(16, 32) → SiLU → Linear(32, 32)``. In both classes the last
linear layer, weight and bias, is zero-initialized, so the logits start at 0
and ``g`` starts at 1 for every finite input. Widths and the ``2 * sigmoid``
range are fixed for these experiments.
"""

from __future__ import annotations

import torch
from torch import nn

AUX_CHANNELS = 2
GATE_HIDDEN_DIM = 32
GATE_OUTPUT_DIM = 32
GATE_FORMULA = "2*sigmoid"
SPLIT_ENCODER_DIM = 8


class AuxiliaryStateGate(nn.Module):
    """One shared map from the two auxiliary channels to 32 gate coefficients."""

    aux_channels = AUX_CHANNELS
    hidden_dim = GATE_HIDDEN_DIM
    output_dim = GATE_OUTPUT_DIM
    formula = GATE_FORMULA

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(AUX_CHANNELS, GATE_HIDDEN_DIM, bias=True)
        self.activation = nn.SiLU()
        self.last = nn.Linear(GATE_HIDDEN_DIM, GATE_OUTPUT_DIM, bias=True)
        if self.last.bias is None:
            raise RuntimeError("auxiliary gate output bias is required for identity initialization")
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

    def forward(self, aux: torch.Tensor) -> torch.Tensor:
        """``aux`` [..., 2] -> ``g`` [..., 32] in ``(0, 2)``."""
        if aux.shape[-1] != AUX_CHANNELS:
            raise ValueError(f"auxiliary gate expects {AUX_CHANNELS} channels, got {tuple(aux.shape)}")
        logits = self.last(self.activation(self.first(aux)))
        return 2 * torch.sigmoid(logits)


def _split_encoder() -> nn.Sequential:
    """One new ``1→8→8`` MLP. A shared Sequential would tie speed to the count."""
    return nn.Sequential(
        nn.Linear(1, SPLIT_ENCODER_DIM, bias=True),
        nn.SiLU(),
        nn.Linear(SPLIT_ENCODER_DIM, SPLIT_ENCODER_DIM, bias=True),
    )


class SplitAuxiliaryStateGate(nn.Module):
    """Independent speed and count encoders, then one joint gate of width 32."""

    aux_channels = AUX_CHANNELS
    output_dim = GATE_OUTPUT_DIM
    formula = GATE_FORMULA
    encoder_dim = SPLIT_ENCODER_DIM

    def __init__(self) -> None:
        super().__init__()
        self.speed_encoder = _split_encoder()
        self.count_encoder = _split_encoder()
        if self.speed_encoder is self.count_encoder:
            raise RuntimeError("speed and count encoders must be distinct modules")
        self.fusion = nn.Linear(SPLIT_ENCODER_DIM * 2, GATE_HIDDEN_DIM, bias=True)
        self.activation = nn.SiLU()
        self.last = nn.Linear(GATE_HIDDEN_DIM, GATE_OUTPUT_DIM, bias=True)
        if self.last.bias is None or self.fusion.bias is None:
            raise RuntimeError("split auxiliary gate linears require bias")
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

    def forward(self, aux: torch.Tensor) -> torch.Tensor:
        """``aux`` [..., 2] -> ``g`` [..., 32] in ``(0, 2)``. Column 0 is speed."""
        if aux.shape[-1] != AUX_CHANNELS:
            raise ValueError(f"split auxiliary gate expects {AUX_CHANNELS} channels, got {tuple(aux.shape)}")
        speed = self.speed_encoder(aux[..., 0:1])
        count = self.count_encoder(aux[..., 1:2])
        hidden = self.activation(self.fusion(torch.cat((speed, count), dim=-1)))
        return 2 * torch.sigmoid(self.last(hidden))
