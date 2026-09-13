"""DCGRU cell: faithful port of reference/dcrnn/model/dcrnn_cell.py DCGRUCell."""

from __future__ import annotations

import torch
import torch.nn as nn

from reimplementation.dcrnn.model.diffusion_conv import DiffusionLinear, build_diffusion_features


class DCGRUCell(nn.Module):
    """Graph-convolution GRU. Gates use sigmoid; candidate uses tanh.

    Original ``use_gc_for_ru=True``: reset/update and candidate all go through
    diffusion convolution. ``bias_start=1.0`` on gates, ``0.0`` on candidate.
    Projection, if present, is a bias-free ``num_units -> num_proj`` map applied
    to the output only; the recurrent state stays unprojected.
    """

    def __init__(
        self,
        num_units: int,
        num_nodes: int,
        input_dim: int,
        max_diffusion_step: int,
        num_supports: int,
        *,
        num_proj: int | None = None,
        activation: str = "tanh",
        use_gc_for_ru: bool = True,
    ) -> None:
        super().__init__()
        if not use_gc_for_ru:
            raise ValueError("original R-only port keeps use_gc_for_ru=True")
        self.num_units = int(num_units)
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.max_diffusion_step = int(max_diffusion_step)
        self.num_supports = int(num_supports)
        self.num_proj = int(num_proj) if num_proj is not None else None
        self.activation_name = str(activation)
        concat_size = self.input_dim + self.num_units
        self.gate_linear = DiffusionLinear(
            concat_size,
            2 * self.num_units,
            self.num_supports,
            self.max_diffusion_step,
            bias_start=1.0,
        )
        self.candidate_linear = DiffusionLinear(
            concat_size,
            self.num_units,
            self.num_supports,
            self.max_diffusion_step,
            bias_start=0.0,
        )
        if self.num_proj is not None:
            self.proj_w = nn.Parameter(torch.empty(self.num_units, self.num_proj))
            nn.init.xavier_uniform_(self.proj_w)
        else:
            self.proj_w = None

    @property
    def output_dim(self) -> int:
        return self.num_proj if self.num_proj is not None else self.num_units

    def _activate(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "tanh":
            return torch.tanh(tensor)
        if self.activation_name == "relu":
            return torch.relu(tensor)
        raise ValueError(f"unsupported activation {self.activation_name}")

    def _gconv(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
        linear: DiffusionLinear,
        supports: list[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = inputs.shape[0]
        inputs_3d = inputs.view(batch_size, self.num_nodes, self.input_dim)
        state_3d = state.view(batch_size, self.num_nodes, self.num_units)
        combined = torch.cat([inputs_3d, state_3d], dim=2)
        features = build_diffusion_features(combined, supports, self.max_diffusion_step)
        return linear(features).view(batch_size, self.num_nodes * linear.output_size)

    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
        supports: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Args are ``[B, N * input_dim]`` and ``[B, N * num_units]``."""
        batch_size = inputs.shape[0]
        if inputs.shape != (batch_size, self.num_nodes * self.input_dim):
            raise ValueError(
                f"DCGRU inputs {tuple(inputs.shape)} != "
                f"[{batch_size}, {self.num_nodes * self.input_dim}]"
            )
        if state.shape != (batch_size, self.num_nodes * self.num_units):
            raise ValueError(
                f"DCGRU state {tuple(state.shape)} != "
                f"[{batch_size}, {self.num_nodes * self.num_units}]"
            )
        value = torch.sigmoid(self._gconv(inputs, state, self.gate_linear, supports))
        value = value.view(batch_size, self.num_nodes, 2 * self.num_units)
        reset, update = torch.split(value, self.num_units, dim=-1)
        reset = reset.reshape(batch_size, self.num_nodes * self.num_units)
        update = update.reshape(batch_size, self.num_nodes * self.num_units)
        candidate = self._gconv(inputs, reset * state, self.candidate_linear, supports)
        candidate = self._activate(candidate)
        new_state = update * state + (1.0 - update) * candidate
        output = new_state
        if self.proj_w is not None:
            projected = torch.matmul(
                new_state.view(batch_size * self.num_nodes, self.num_units),
                self.proj_w,
            )
            output = projected.view(batch_size, self.num_nodes * self.num_proj)
        return output, new_state
