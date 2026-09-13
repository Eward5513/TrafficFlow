"""Seq2seq DCRNN matching reference/dcrnn/model/dcrnn_model.py."""

from __future__ import annotations

import torch
import torch.nn as nn

from reimplementation.dcrnn.model.dcgru_cell import DCGRUCell
from reimplementation.dcrnn.model.decoder import DCRNNDecoder
from reimplementation.dcrnn.model.encoder import DCRNNEncoder

CODE_VERSION = "0.1.0"


class DCRNNModel(nn.Module):
    """Public tensors stay ``[B, T, V, C]``.

    Encoder reads 12 observed steps. Decoder starts from a zero GO symbol and
    emits ``horizon`` frames. With ``horizon=1`` the kept output is the GO step;
    the original extra unroll step still runs and is dropped.
    """

    def __init__(
        self,
        supports: list[torch.Tensor] | list,
        *,
        num_nodes: int,
        input_dim: int = 1,
        output_dim: int = 1,
        rnn_units: int = 64,
        num_rnn_layers: int = 2,
        max_diffusion_step: int = 2,
        seq_len: int = 12,
        horizon: int = 1,
        use_curriculum_learning: bool = True,
        cl_decay_steps: int = 2000,
        filter_type: str = "dual_random_walk",
        use_gc_for_ru: bool = True,
    ) -> None:
        super().__init__()
        if num_rnn_layers < 1:
            raise ValueError("num_rnn_layers must be >= 1")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.rnn_units = int(rnn_units)
        self.num_rnn_layers = int(num_rnn_layers)
        self.max_diffusion_step = int(max_diffusion_step)
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)
        self.filter_type = str(filter_type)
        self.use_curriculum_learning = bool(use_curriculum_learning)
        self.cl_decay_steps = int(cl_decay_steps)
        tensors = []
        for index, support in enumerate(supports):
            tensor = torch.as_tensor(support, dtype=torch.float32)
            if tensor.shape != (self.num_nodes, self.num_nodes):
                raise ValueError(
                    f"support {index} shape {tuple(tensor.shape)} != "
                    f"[{self.num_nodes}, {self.num_nodes}]"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"support {index} contains NaN or inf")
            self.register_buffer(f"support_{index}", tensor)
            tensors.append(tensor)
        self.num_supports = len(tensors)
        if self.num_supports == 0:
            raise ValueError("DCRNN needs at least one diffusion support")

        encoder_cells = nn.ModuleList()
        layer_input_dim = self.input_dim
        for _ in range(self.num_rnn_layers):
            encoder_cells.append(
                DCGRUCell(
                    self.rnn_units,
                    self.num_nodes,
                    layer_input_dim,
                    self.max_diffusion_step,
                    self.num_supports,
                    num_proj=None,
                    use_gc_for_ru=use_gc_for_ru,
                )
            )
            layer_input_dim = self.rnn_units
        decoder_cells = nn.ModuleList()
        decoder_input_dim = self.output_dim
        for layer_index in range(self.num_rnn_layers):
            num_proj = self.output_dim if layer_index == self.num_rnn_layers - 1 else None
            cell_input_dim = decoder_input_dim if layer_index == 0 else self.rnn_units
            decoder_cells.append(
                DCGRUCell(
                    self.rnn_units,
                    self.num_nodes,
                    cell_input_dim,
                    self.max_diffusion_step,
                    self.num_supports,
                    num_proj=num_proj,
                    use_gc_for_ru=use_gc_for_ru,
                )
            )
        self.encoder = DCRNNEncoder(encoder_cells)
        self.decoder = DCRNNDecoder(
            decoder_cells,
            horizon=self.horizon,
            num_nodes=self.num_nodes,
            output_dim=self.output_dim,
            use_curriculum_learning=self.use_curriculum_learning,
            cl_decay_steps=self.cl_decay_steps,
        )

    def support_list(self) -> list[torch.Tensor]:
        return [getattr(self, f"support_{index}") for index in range(self.num_supports)]

    def parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.parameters()))

    def forward(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        global_step: int = 0,
        return_trace: bool = False,
    ):
        if inputs.ndim != 4:
            raise ValueError(f"DCRNN inputs must be [B, T, V, C], got {tuple(inputs.shape)}")
        batch, seq_len, nodes, channels = inputs.shape
        if (seq_len, nodes, channels) != (self.seq_len, self.num_nodes, self.input_dim):
            raise ValueError(
                f"DCRNN input {tuple(inputs.shape)} != "
                f"[B, {self.seq_len}, {self.num_nodes}, {self.input_dim}]"
            )
        supports = self.support_list()
        encoder_states = self.encoder(inputs, supports)
        outputs, decode_trace = self.decoder(
            encoder_states,
            supports=supports,
            labels=labels if self.training else None,
            global_step=int(global_step),
            is_training=bool(self.training),
        )
        if outputs.shape != (batch, self.horizon, self.num_nodes, self.output_dim):
            raise RuntimeError(
                f"DCRNN output {tuple(outputs.shape)} != "
                f"[{batch}, {self.horizon}, {self.num_nodes}, {self.output_dim}]"
            )
        trace = {
            "input": list(inputs.shape),
            "encoder_state_shapes": [list(state.shape) for state in encoder_states],
            "output": list(outputs.shape),
            "num_supports": self.num_supports,
            "filter_type": self.filter_type,
            "horizon": self.horizon,
            "seq_len": self.seq_len,
            **decode_trace,
        }
        if return_trace:
            return outputs, trace
        return outputs
