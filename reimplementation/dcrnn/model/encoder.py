"""Stacked DCGRU encoder. Independent layers; state tuple feeds the decoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from reimplementation.dcrnn.model.dcgru_cell import DCGRUCell


class DCRNNEncoder(nn.Module):
    def __init__(self, cells: nn.ModuleList) -> None:
        super().__init__()
        if len(cells) == 0:
            raise ValueError("encoder needs at least one DCGRU cell")
        self.cells = cells

    def forward(
        self,
        inputs: torch.Tensor,
        supports: list[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        """``inputs`` is ``[B, T, N, C]``. Returns one state per layer."""
        if inputs.ndim != 4:
            raise ValueError(f"encoder input must be [B, T, N, C], got {tuple(inputs.shape)}")
        batch, seq_len, num_nodes, channels = inputs.shape
        first = self.cells[0]
        if not isinstance(first, DCGRUCell):
            raise TypeError("encoder cells must be DCGRUCell")
        if (num_nodes, channels) != (first.num_nodes, first.input_dim):
            raise ValueError(
                f"encoder input [N,C]=[{num_nodes},{channels}] != "
                f"cell [{first.num_nodes},{first.input_dim}]"
            )
        states = [
            inputs.new_zeros(batch, cell.num_nodes * cell.num_units) for cell in self.cells
        ]
        for time_index in range(seq_len):
            current = inputs[:, time_index].reshape(batch, num_nodes * channels)
            for layer_index, cell in enumerate(self.cells):
                current, states[layer_index] = cell(current, states[layer_index], supports)
        return tuple(states)
