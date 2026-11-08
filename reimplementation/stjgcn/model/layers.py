"""PeMSD7 ``tf_utils.conv2d`` and its conditional batch norm, in PyTorch.

Batch norm runs only when ``activation`` is not None. Moments use the
population variance over every axis except the channel. The moving average is
``shadow = 0.9 * shadow + 0.1 * batch`` and starts at zero, matching a
TensorFlow ``ExponentialMovingAverage`` on a non-variable. Epsilon is ``1e-3``.
Training normalizes with the current batch; eval uses the average.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def glorot_uniform_(tensor: torch.Tensor) -> torch.Tensor:
    """TensorFlow ``glorot_uniform_initializer`` fan rule, including rank > 2."""
    shape = tuple(int(dim) for dim in tensor.shape)
    if len(shape) < 1:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in, fan_out = shape
    else:
        receptive = 1
        for dim in shape[:-2]:
            receptive *= dim
        fan_in = shape[-2] * receptive
        fan_out = shape[-1] * receptive
    limit = math.sqrt(6.0 / float(fan_in + fan_out))
    with torch.no_grad():
        tensor.uniform_(-limit, limit)
    return tensor


class TFBatchNorm(nn.Module):
    def __init__(self, num_features: int, decay: float = 0.9, eps: float = 1e-3):
        super().__init__()
        self.decay = float(decay)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(num_features)))
        self.bias = nn.Parameter(torch.zeros(int(num_features)))
        self.register_buffer("running_mean", torch.zeros(int(num_features)))
        self.register_buffer("running_var", torch.zeros(int(num_features)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduce_dims = tuple(range(x.ndim - 1))
        if self.training:
            mean = x.mean(dim=reduce_dims)
            var = x.var(dim=reduce_dims, unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(self.decay).add_(mean, alpha=1.0 - self.decay)
                self.running_var.mul_(self.decay).add_(var, alpha=1.0 - self.decay)
        else:
            mean = self.running_mean
            var = self.running_var
        view = (1,) * (x.ndim - 1) + (x.shape[-1],)
        normalized = (x - mean.view(view)) / torch.sqrt(var.view(view) + self.eps)
        return normalized * self.weight.view(view) + self.bias.view(view)


class Conv2dTF(nn.Module):
    """1x1 NHWC convolution: kernel layout ``[1, 1, in, out]``."""

    def __init__(self, input_dims: int, output_dims: int, *, use_bias: bool, activation: Callable | None):
        super().__init__()
        self.activation = activation
        self.kernel = nn.Parameter(torch.empty(1, 1, int(input_dims), int(output_dims)))
        glorot_uniform_(self.kernel)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(int(output_dims)))
        else:
            self.register_parameter("bias", None)
        self.bn = TFBatchNorm(int(output_dims)) if activation is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.matmul(self.kernel[0, 0])
        if self.bias is not None:
            y = y + self.bias
        if self.activation is not None:
            y = self.activation(self.bn(y))
        return y


class FC(nn.Module):
    def __init__(self, input_dims: int, units: list[int], activations: list[Callable | None], *, use_bias: bool):
        super().__init__()
        dims = [int(input_dims), *[int(unit) for unit in units]]
        self.layers = nn.ModuleList(
            [
                Conv2dTF(dims[index], dims[index + 1], use_bias=use_bias, activation=activation)
                for index, activation in enumerate(activations)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def adaptive_attend(
    query: torch.Tensor,
    memory: torch.Tensor,
    values: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """Softmax after setting logits below ``delta`` to zero.

    Zero is not a masked logit. A suppressed entry still contributes ``exp(0)``.
    """
    logits = query.matmul(memory.transpose(-1, -2))
    logits = torch.where(logits < delta, torch.zeros_like(logits), logits)
    return torch.softmax(logits, dim=-1).matmul(values)


class AdaptiveGraph(nn.Module):
    def __init__(self, hidden: int, *, use_bias: bool, chunk_nodes: int):
        super().__init__()
        self.chunk_nodes = int(chunk_nodes)
        self.fc = Conv2dTF(hidden, hidden, use_bias=use_bias, activation=F.relu)

    def forward(self, values: torch.Tensor, query_embed: torch.Tensor, memory_embed: torch.Tensor, interaction: torch.Tensor, delta: float) -> torch.Tensor:
        query = query_embed.matmul(interaction[0, 0])
        parts = []
        nodes = int(values.shape[2])
        step = max(int(self.chunk_nodes), 1)
        threshold = values.new_tensor(float(delta))
        for start in range(0, nodes, step):
            end = min(nodes, start + step)
            block_query = query[:, :, start:end]
            if torch.is_grad_enabled():
                block = checkpoint(
                    adaptive_attend,
                    block_query,
                    memory_embed,
                    values,
                    threshold,
                    use_reentrant=False,
                )
            else:
                block = adaptive_attend(block_query, memory_embed, values, threshold)
            parts.append(block)
        return self.fc(torch.cat(parts, dim=2))
