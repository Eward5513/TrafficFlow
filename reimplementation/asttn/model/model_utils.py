"""ASTTN blocks transcribed from reference/ASTTN_pytorch/model/model_utils.py.

CONVs still apply BatchNorm on every 1x1 convolution and ReLU only when the
stored activation is not None. The original forward ignores the callable and
always uses ReLU; that behavior is kept. Device placement follows the input
tensor instead of a module-level ``.cuda()`` flag.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class conv2d_(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dims,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding="SAME",
        use_bias=True,
        activation=F.relu,
        bn_decay=None,
    ):
        super().__init__()
        self.activation = activation
        if padding == "SAME":
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(
            input_dims,
            output_dims,
            kernel_size,
            stride=stride,
            padding=0,
            bias=use_bias,
        )
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        nn.init.xavier_uniform_(self.conv.weight)
        if use_bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = F.pad(
            x,
            [self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]],
        )
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)


class CONVs(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay=0.1, use_bias=True):
        super().__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        if not isinstance(units, list):
            raise TypeError("CONVs units must be a list")
        self.convs = nn.ModuleList(
            [
                conv2d_(
                    input_dims=input_dim,
                    output_dims=num_unit,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding="VALID",
                    use_bias=use_bias,
                    activation=activation,
                    bn_decay=bn_decay,
                )
                for input_dim, num_unit, activation in zip(input_dims, units, activations)
            ]
        )

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dims, output_dims, activation=F.relu):
        super().__init__()
        self.activation = activation
        self.mlp = nn.Linear(input_dims, output_dims)
        self.layer_norm = nn.LayerNorm(output_dims)
        nn.init.xavier_uniform_(self.mlp.weight)
        nn.init.zeros_(self.mlp.bias)

    def forward(self, x):
        x = self.mlp(x)
        x = self.layer_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x


class MLPs(nn.Module):
    def __init__(self, input_dims, units, activations):
        super().__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        if not isinstance(units, list):
            raise TypeError("MLPs units must be a list")
        self.mlps = nn.ModuleList(
            [
                MLP(input_dims=input_dim, output_dims=num_unit, activation=activation)
                for input_dim, num_unit, activation in zip(input_dims, units, activations)
            ]
        )

    def forward(self, x):
        for mlp in self.mlps:
            x = mlp(x)
        return x


class STEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, emb_dim: int, slots_per_day: int = 288):
        super().__init__()
        self.slots_per_day = int(slots_per_day)
        self.mlp_se = CONVs(input_dims=[emb_dim, hidden_dim], units=[hidden_dim, hidden_dim], activations=[F.relu, None])
        self.mlp_te = CONVs(
            input_dims=[self.slots_per_day + 7, hidden_dim],
            units=[hidden_dim, hidden_dim],
            activations=[F.relu, None],
        )

    def forward(self, spatial_embedding: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        spatial = spatial_embedding.unsqueeze(0).unsqueeze(0)
        spatial = self.mlp_se(spatial)
        day = F.one_hot(time_embedding[..., 0].to(dtype=torch.long) % 7, 7).to(dtype=spatial.dtype)
        slot = F.one_hot(time_embedding[..., 1].to(dtype=torch.long) % self.slots_per_day, self.slots_per_day)
        slot = slot.to(dtype=spatial.dtype)
        temporal = torch.cat((day, slot), dim=-1).unsqueeze(2)
        temporal = self.mlp_te(temporal)
        return spatial + temporal


class TransformerSelfOutput(nn.Module):
    def __init__(self, insize: int, outsize: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(insize, outsize)
        self.layer_norm = nn.LayerNorm(outsize)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.layer_norm(hidden_states + input_tensor)


class TransformerIntermediate(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.intermediate_act_fn(self.dense(hidden_states))


class PostProcess(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.intermediate = TransformerIntermediate(hidden_dim, hidden_dim)
        self.layer_output = TransformerSelfOutput(hidden_dim, hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.layer_output(self.intermediate(hidden_states), hidden_states)


class TransformAttention(nn.Module):
    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        hidden = num_heads * head_dim
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.mlp_q = MLPs(input_dims=hidden, units=hidden, activations=None)
        self.mlp_k = MLPs(input_dims=hidden, units=hidden, activations=None)
        self.mlp_v = MLPs(input_dims=hidden, units=hidden, activations=None)
        self.out = MLPs(input_dims=hidden, units=hidden, activations=F.relu)

    def forward(self, values: torch.Tensor, history: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = values.shape[0]
        query = self.mlp_q(target)
        key = self.mlp_k(history)
        value = self.mlp_v(values)
        query = torch.cat(torch.split(query, self.num_heads, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.num_heads, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.num_heads, dim=-1), dim=0)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        attention = torch.matmul(query, key)
        attention = attention / (self.head_dim ** 0.5)
        attention = F.softmax(attention, dim=-1)
        out = torch.matmul(attention, value)
        out = out.permute(0, 2, 1, 3)
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        return self.out(out)
