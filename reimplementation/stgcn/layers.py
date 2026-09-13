"""STGCN layers translated from ``reference/STGCN_IJCAI-18/models/layers.py``.

Original TensorFlow tensors are NHWC ``[batch, time, n_route, channel]``.
PyTorch convolutions use NCHW internally; public tensors stay NHWC.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


ActName = Literal["GLU", "relu", "sigmoid", "linear"]


def _xavier_conv(module: nn.Conv2d) -> None:
    nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def to_nchw(x: torch.Tensor) -> torch.Tensor:
    """[B, T, V, C] -> [B, C, T, V]."""
    return x.permute(0, 3, 1, 2).contiguous()


def to_nhwc(x: torch.Tensor) -> torch.Tensor:
    """[B, C, T, V] -> [B, T, V, C]."""
    return x.permute(0, 2, 3, 1).contiguous()


class TemporalConvLayer(nn.Module):
    """Original ``temporal_conv_layer``.

    VALID temporal convolution: output time is ``T - Kt + 1``. Residual uses
    the last ``T-Kt+1`` frames of the channel-aligned input
    (``x_input[:, Kt-1:T, :, :]``), not SAME padding.
    """

    def __init__(self, kt: int, c_in: int, c_out: int, act_func: ActName = "relu") -> None:
        super().__init__()
        if kt < 1:
            raise ValueError(f"Kt must be >= 1, got {kt}")
        self.kt = kt
        self.c_in = c_in
        self.c_out = c_out
        self.act_func = act_func
        conv_out = 2 * c_out if act_func == "GLU" else c_out
        self.conv = nn.Conv2d(c_in, conv_out, kernel_size=(kt, 1), padding=0, bias=True)
        _xavier_conv(self.conv)
        self.align: nn.Conv2d | None
        if c_in > c_out:
            self.align = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
            nn.init.xavier_uniform_(self.align.weight)
        else:
            self.align = None

    def aligned_residual(self, x: torch.Tensor) -> torch.Tensor:
        time_len = x.size(1)
        if self.c_in > self.c_out:
            assert self.align is not None
            aligned = to_nhwc(self.align(to_nchw(x)))
        elif self.c_in < self.c_out:
            aligned = F.pad(x, (0, self.c_out - self.c_in))
        else:
            aligned = x
        return aligned[:, self.kt - 1 : time_len, :, :]

    def decay_weights(self) -> list[torch.Tensor]:
        weights = [self.conv.weight]
        if self.align is not None:
            weights.append(self.align.weight)
        return weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.aligned_residual(x)
        conv = to_nhwc(self.conv(to_nchw(x)))
        if self.act_func == "GLU":
            return (conv[..., : self.c_out] + residual) * torch.sigmoid(conv[..., -self.c_out :])
        if self.act_func == "linear":
            return conv
        if self.act_func == "sigmoid":
            return torch.sigmoid(conv)
        if self.act_func == "relu":
            return torch.relu(conv + residual)
        raise ValueError(f'ERROR: activation function "{self.act_func}" is not defined.')


class SpatialConvLayer(nn.Module):
    """Original ``spatio_conv_layer`` + ``gconv``.

    ``theta`` is ``[Ks * c_in, c_out]`` and the Chebyshev kernel is
    ``[n_route, Ks * n_route]``. Kernel is a buffer on the parent model.
    """

    def __init__(self, ks: int, c_in: int, c_out: int, n_nodes: int) -> None:
        super().__init__()
        self.ks = ks
        self.c_in = c_in
        self.c_out = c_out
        self.n_nodes = n_nodes
        self.theta = nn.Parameter(torch.empty(ks * c_in, c_out))
        self.bias = nn.Parameter(torch.zeros(c_out))
        limit = math.sqrt(6.0 / ((ks * c_in) + c_out))
        nn.init.uniform_(self.theta, -limit, limit)
        self.align: nn.Conv2d | None
        if c_in > c_out:
            self.align = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
            nn.init.xavier_uniform_(self.align.weight)
        else:
            self.align = None

    def decay_weights(self) -> list[torch.Tensor]:
        weights = [self.theta]
        if self.align is not None:
            weights.append(self.align.weight)
        return weights

    def _align(self, x: torch.Tensor) -> torch.Tensor:
        if self.c_in > self.c_out:
            assert self.align is not None
            return to_nhwc(self.align(to_nchw(x)))
        if self.c_in < self.c_out:
            return F.pad(x, (0, self.c_out - self.c_in))
        return x

    def gconv(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """x: [B, n, c_in], kernel: [n, Ks*n] -> [B, n, c_out]."""
        batch, n_nodes, c_in = x.shape
        if n_nodes != kernel.size(0) or kernel.size(1) != self.ks * n_nodes:
            raise ValueError(
                f"graph kernel shape {tuple(kernel.shape)} incompatible with "
                f"Ks={self.ks}, n={n_nodes}"
            )
        x_tmp = x.permute(0, 2, 1).reshape(-1, n_nodes)
        x_mul = torch.matmul(x_tmp, kernel).reshape(-1, c_in, self.ks, n_nodes)
        x_ker = x_mul.permute(0, 3, 1, 2).reshape(-1, c_in * self.ks)
        return torch.matmul(x_ker, self.theta).reshape(batch, n_nodes, self.c_out)

    def forward(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        residual = self._align(x)
        batch, time_len, n_nodes, c_in = x.shape
        x_gconv = self.gconv(x.reshape(batch * time_len, n_nodes, c_in), kernel) + self.bias
        x_gc = x_gconv.reshape(batch, time_len, n_nodes, self.c_out)
        return torch.relu(x_gc + residual)


class SpatialTemporalNorm(nn.Module):
    """Original ``layer_norm``: moments over nodes and channels, gamma/beta ``[1,1,N,C]``."""

    def __init__(self, n_nodes: int, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, n_nodes, channels))
        self.beta = nn.Parameter(torch.zeros(1, 1, n_nodes, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta


class STConvBlock(nn.Module):
    """Original ``st_conv_block``: temporal GLU, spatial, temporal ReLU, LN, dropout."""

    def __init__(
        self,
        ks: int,
        kt: int,
        channels: tuple[int, int, int],
        n_nodes: int,
        dropout_p: float,
    ) -> None:
        super().__init__()
        c_si, c_t, c_oo = channels
        self.temporal_in = TemporalConvLayer(kt, c_si, c_t, act_func="GLU")
        self.spatial = SpatialConvLayer(ks, c_t, c_t, n_nodes)
        self.temporal_out = TemporalConvLayer(kt, c_t, c_oo, act_func="relu")
        self.norm = SpatialTemporalNorm(n_nodes, c_oo)
        self.dropout = nn.Dropout(p=dropout_p)

    def decay_weights(self) -> list[torch.Tensor]:
        return (
            self.temporal_in.decay_weights()
            + self.spatial.decay_weights()
            + self.temporal_out.decay_weights()
        )

    def forward(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        x_s = self.temporal_in(x)
        x_t = self.spatial(x_s, kernel)
        x_o = self.temporal_out(x_t)
        return self.dropout(self.norm(x_o))


class FullyConLayer(nn.Module):
    """Original ``fully_con_layer``: 1x1 conv to one channel, bias ``[n, 1]``."""

    def __init__(self, n_nodes: int, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.conv.weight)
        self.bias = nn.Parameter(torch.zeros(n_nodes, 1))

    def decay_weights(self) -> list[torch.Tensor]:
        return [self.conv.weight]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return to_nhwc(self.conv(to_nchw(x))) + self.bias


class OutputLayer(nn.Module):
    """Original ``output_layer``: GLU over remaining time, LN, sigmoid k=1, FC."""

    def __init__(self, remaining_time: int, channels: int, n_nodes: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        del dropout_p
        self.temporal_in = TemporalConvLayer(remaining_time, channels, channels, act_func="GLU")
        self.norm = SpatialTemporalNorm(n_nodes, channels)
        self.temporal_out = TemporalConvLayer(1, channels, channels, act_func="sigmoid")
        self.fc = FullyConLayer(n_nodes, channels)

    def decay_weights(self) -> list[torch.Tensor]:
        return self.temporal_in.decay_weights() + self.temporal_out.decay_weights() + self.fc.decay_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_i = self.temporal_in(x)
        x_ln = self.norm(x_i)
        x_o = self.temporal_out(x_ln)
        return self.fc(x_o)
