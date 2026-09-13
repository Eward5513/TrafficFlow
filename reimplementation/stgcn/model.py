"""STGCN model: two ST-Conv blocks plus the original output layer."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from reimplementation.common.errors import ReimplementationError
from reimplementation.stgcn.graph import build_stgcn_chebyshev_kernel, chebyshev_t0_is_identity
from reimplementation.stgcn.layers import OutputLayer, STConvBlock
from reimplementation.stgcn.losses import collected_weight_l2

CODE_VERSION = "0.1.0"


class STGCN(nn.Module):
    """Faithful PyTorch port of ``build_model`` in original ``base_model.py``.

    Public interface::

        x: [B, n_his, n_nodes, C] = [B, 12, 56, 1]
        y: [B, 1, n_nodes, 1]

    Internally tensors stay NHWC except inside ``Conv2d``. The Chebyshev kernel
    is a buffer, not a trainable parameter. Self-loops are not added to W.
    """

    def __init__(
        self,
        n_nodes: int,
        n_his: int,
        ks: int,
        kt: int,
        blocks: list[list[int]],
        cheb_kernel: np.ndarray | torch.Tensor,
        dropout_p: float = 0.0,
        input_channels: int = 1,
        output_channels: int = 1,
        output_steps: int = 1,
    ) -> None:
        super().__init__()
        if output_steps != 1 or output_channels != 1:
            raise ReimplementationError("this STGCN port emits a single one-channel time step")
        if not blocks:
            raise ReimplementationError("blocks must be non-empty")
        if blocks[0][0] != input_channels:
            raise ReimplementationError(
                f"first block input channels {blocks[0][0]} != input_channels {input_channels}"
            )
        self.n_nodes = int(n_nodes)
        self.n_his = int(n_his)
        self.ks = int(ks)
        self.kt = int(kt)
        self.blocks = [list(item) for item in blocks]
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.output_steps = int(output_steps)
        remaining = int(n_his)
        conv_blocks = nn.ModuleList()
        for channels in blocks:
            if len(channels) != 3:
                raise ReimplementationError(f"each ST-Conv block needs 3 channels, got {channels}")
            conv_blocks.append(
                STConvBlock(ks, kt, (int(channels[0]), int(channels[1]), int(channels[2])), n_nodes, dropout_p)
            )
            remaining -= 2 * (kt - 1)
        if remaining <= 1:
            raise ReimplementationError(
                f'ERROR: kernel size Ko must be greater than 1, but received "{remaining}".'
            )
        self.st_blocks = conv_blocks
        self.remaining_time = remaining
        last_channels = int(blocks[-1][2])
        self.output_layer = OutputLayer(remaining, last_channels, n_nodes)
        kernel = torch.as_tensor(cheb_kernel, dtype=torch.float32)
        if tuple(kernel.shape) != (n_nodes, ks * n_nodes):
            raise ReimplementationError(
                f"Chebyshev kernel shape {tuple(kernel.shape)} != ({n_nodes}, {ks * n_nodes})"
            )
        if not torch.isfinite(kernel).all():
            raise ReimplementationError("Chebyshev kernel is not finite")
        if not chebyshev_t0_is_identity(kernel.detach().cpu().numpy(), n_nodes):
            raise ReimplementationError("Chebyshev T0 is not the identity")
        self.register_buffer("cheb_kernel", kernel, persistent=True)

    def decay_weights(self) -> list[torch.Tensor]:
        weights: list[torch.Tensor] = []
        for block in self.st_blocks:
            weights.extend(block.decay_weights())
        weights.extend(self.output_layer.decay_weights())
        return weights

    def regularization_loss(self) -> torch.Tensor:
        return collected_weight_l2(self.decay_weights())

    def parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.parameters()))

    def graph_buffer_names(self) -> list[str]:
        return [name for name, _ in self.named_buffers() if name == "cheb_kernel"]

    def forward(
        self,
        x: torch.Tensor,
        return_trace: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        if x.dim() != 4:
            raise ReimplementationError(f"expected [B,T,V,C], got {tuple(x.shape)}")
        batch, time_len, n_nodes, channels = x.shape
        if time_len != self.n_his or n_nodes != self.n_nodes or channels != self.input_channels:
            raise ReimplementationError(
                f"expected [B,{self.n_his},{self.n_nodes},{self.input_channels}], got {tuple(x.shape)}"
            )
        trace: dict[str, Any] = {"input": time_len}
        hidden = x
        kernel = self.cheb_kernel
        for index, block in enumerate(self.st_blocks):
            hidden = block.temporal_in(hidden)
            trace[f"block{index}_temporal_in"] = int(hidden.size(1))
            hidden = block.spatial(hidden, kernel)
            trace[f"block{index}_spatial"] = int(hidden.size(1))
            hidden = block.temporal_out(hidden)
            hidden = block.dropout(block.norm(hidden))
            trace[f"block{index}_temporal_out"] = int(hidden.size(1))
        prediction = self.output_layer(hidden)
        trace["output"] = int(prediction.size(1))
        if tuple(prediction.shape[1:]) != (self.output_steps, self.n_nodes, self.output_channels):
            raise ReimplementationError(f"unexpected prediction shape {tuple(prediction.shape)}")
        if return_trace:
            return prediction, trace
        return prediction
