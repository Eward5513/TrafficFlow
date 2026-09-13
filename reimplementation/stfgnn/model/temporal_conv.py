"""Gated temporal CNN from ``sthgcn_layer_individual`` in ``stsgcn_4n_res.py``.

Original MXNet (NCHW, ``data`` already ``(B, C, N, T)``)::

    left  = sigmoid(Convolution(kernel=(1, 2), dilate=(1, 3), num_filter=C))
    right = tanh   (Convolution(kernel=(1, 2), dilate=(1, 3), num_filter=C))
    out   = left * right

No padding, so time length becomes ``T - 3``. This is **not** Graph WaveNet:
GWN uses tanh(filter)*sigmoid(gate) with different kernels/dilations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedTemporalConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.filter_conv = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=(1, 2),
            stride=1,
            dilation=(1, 3),
            padding=0,
            bias=True,
        )
        self.gate_conv = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=(1, 2),
            stride=1,
            dilation=(1, 3),
            padding=0,
            bias=True,
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """``data`` is ``(B, T, N, C)``. Output is ``(B, T-3, N, C)``."""
        # Original: transpose (B, T, N, C) -> (B, C, N, T)
        nchw = data.permute(0, 3, 2, 1)
        left = torch.sigmoid(self.filter_conv(nchw))
        right = torch.tanh(self.gate_conv(nchw))
        fused = left * right
        return fused.permute(0, 3, 2, 1).contiguous()
