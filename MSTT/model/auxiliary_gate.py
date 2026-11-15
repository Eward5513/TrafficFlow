"""Gate the four state embeddings with speed and the observation count.

``g = 2 * sigmoid(Linear(32, 32)(SiLU(Linear(2, 32)(aux))))``. The last linear
layer, weight and bias, is zero-initialized, so the logits start at 0 and
``g`` starts at 1. Widths and the ``2 * sigmoid`` range are fixed for this
experiment.
"""

from __future__ import annotations

import torch
from torch import nn

AUX_CHANNELS = 2
GATE_HIDDEN_DIM = 32
GATE_OUTPUT_DIM = 32
GATE_FORMULA = "2*sigmoid"


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
