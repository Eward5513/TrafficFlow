"""Graph WaveNet modules matching reference/Graph-WaveNet/model.py.

Public tensors stay ``[B, T, V, C]``. Internally the original layout
``[B, C, V, T]`` is used. Official ``engine.trainer.train`` left-pads the
time axis by 1 before the network; with ``blocks=4, layers=2, kernel=2``
the receptive field is 13, so a 12-step window becomes 13 zeros-on-the-left.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CODE_VERSION = "0.1.0"
ADAPTIVE_EMBED_DIM = 10


def receptive_field(blocks: int, layers: int, kernel_size: int) -> int:
    """Original ``gwnet.__init__`` accumulation of ``additional_scope``."""
    field = 1
    for _ in range(int(blocks)):
        additional_scope = int(kernel_size) - 1
        for _ in range(int(layers)):
            field += additional_scope
            additional_scope *= 2
    return int(field)


class NConv(nn.Module):
    """Original ``nconv``: ``einsum('ncvl,vw->ncwl')``."""

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ncvl,vw->ncwl", x, adj).contiguous()


class LinearConv(nn.Module):
    """Original ``linear``: 1x1 Conv2d."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GCN(nn.Module):
    """Original ``gcn`` of order 2: ``[x, A x, A^2 x]`` per support, then 1x1."""

    def __init__(self, c_in: int, c_out: int, dropout: float, support_len: int = 3, order: int = 2) -> None:
        super().__init__()
        self.nconv = NConv()
        self.order = int(order)
        self.support_len = int(support_len)
        self.dropout = float(dropout)
        expanded = (self.order * self.support_len + 1) * int(c_in)
        self.mlp = LinearConv(expanded, c_out)

    def forward(self, x: torch.Tensor, support: list[torch.Tensor]) -> torch.Tensor:
        if len(support) != self.support_len:
            raise RuntimeError(f"GCN expected {self.support_len} supports, got {len(support)}")
        out = [x]
        for adj in support:
            x1 = self.nconv(x, adj)
            out.append(x1)
            xk = x1
            for _ in range(2, self.order + 1):
                xk = self.nconv(xk, adj)
                out.append(xk)
        hidden = torch.cat(out, dim=1)
        hidden = self.mlp(hidden)
        return F.dropout(hidden, self.dropout, training=self.training)


class GraphWaveNet(nn.Module):
    """Faithful port of ``gwnet`` with a ``[B, T, V, C]`` public interface."""

    def __init__(
        self,
        num_nodes: int,
        dropout: float = 0.3,
        supports: list[torch.Tensor] | list | None = None,
        gcn_bool: bool = True,
        addaptadj: bool = True,
        aptinit: torch.Tensor | None = None,
        in_dim: int = 1,
        out_dim: int = 1,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 256,
        end_channels: int = 512,
        kernel_size: int = 2,
        blocks: int = 4,
        layers: int = 2,
        gcn_order: int = 2,
        engine_left_pad: int = 1,
    ) -> None:
        super().__init__()
        if not gcn_bool:
            raise ValueError("official Graph WaveNet command enables --gcn_bool")
        if not addaptadj:
            raise ValueError("official Graph WaveNet command enables --addaptadj")
        self.num_nodes = int(num_nodes)
        self.dropout = float(dropout)
        self.blocks = int(blocks)
        self.layers = int(layers)
        self.gcn_bool = True
        self.addaptadj = True
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.kernel_size = int(kernel_size)
        self.gcn_order = int(gcn_order)
        self.engine_left_pad = int(engine_left_pad)
        self.receptive_field = receptive_field(self.blocks, self.layers, self.kernel_size)
        self.adaptive_embed_dim = ADAPTIVE_EMBED_DIM

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(self.in_dim, residual_channels, kernel_size=(1, 1))

        support_tensors: list[torch.Tensor] = []
        if supports is not None:
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
                support_tensors.append(tensor)
        self.num_static_supports = len(support_tensors)
        if self.num_static_supports == 0:
            raise ValueError("Graph WaveNet needs the official static doubletransition supports")

        self.supports_len = self.num_static_supports
        if aptinit is None:
            self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, ADAPTIVE_EMBED_DIM))
            self.nodevec2 = nn.Parameter(torch.randn(ADAPTIVE_EMBED_DIM, self.num_nodes))
        else:
            init = torch.as_tensor(aptinit, dtype=torch.float32)
            left, singular, right = torch.svd(init)
            initemb1 = torch.mm(left[:, :ADAPTIVE_EMBED_DIM], torch.diag(singular[:ADAPTIVE_EMBED_DIM] ** 0.5))
            initemb2 = torch.mm(
                torch.diag(singular[:ADAPTIVE_EMBED_DIM] ** 0.5),
                right[:, :ADAPTIVE_EMBED_DIM].t(),
            )
            self.nodevec1 = nn.Parameter(initemb1)
            self.nodevec2 = nn.Parameter(initemb2)
        self.supports_len += 1

        for _block in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for _layer in range(self.layers):
                # Original filter_convs are Conv2d. gate/residual/skip were Conv1d with
                # kernel_size=(1, k) on 4-D tensors; that is a Conv2d layout and is the
                # only runnable meaning in current PyTorch.
                self.filter_convs.append(
                    nn.Conv2d(
                        residual_channels,
                        dilation_channels,
                        kernel_size=(1, self.kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        residual_channels,
                        dilation_channels,
                        kernel_size=(1, self.kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(dilation_channels, residual_channels, kernel_size=(1, 1))
                )
                self.skip_convs.append(
                    nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1))
                )
                self.bn.append(nn.BatchNorm2d(residual_channels))
                self.gconv.append(
                    GCN(
                        dilation_channels,
                        residual_channels,
                        self.dropout,
                        support_len=self.supports_len,
                        order=self.gcn_order,
                    )
                )
                new_dilation *= 2
                additional_scope *= 2

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, self.out_dim, kernel_size=(1, 1), bias=True)

    def support_list(self) -> list[torch.Tensor]:
        return [getattr(self, f"support_{index}") for index in range(self.num_static_supports)]

    def adaptive_adjacency(self) -> torch.Tensor:
        return F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

    def parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.parameters()))

    def forward(self, inputs: torch.Tensor, return_trace: bool = False):
        if inputs.ndim != 4:
            raise ValueError(f"GWN inputs must be [B, T, V, C], got {tuple(inputs.shape)}")
        batch, seq_len, nodes, channels = inputs.shape
        if (nodes, channels) != (self.num_nodes, self.in_dim):
            raise ValueError(
                f"GWN input {tuple(inputs.shape)} != [B, T, {self.num_nodes}, {self.in_dim}]"
            )
        x = inputs.permute(0, 3, 2, 1).contiguous()
        engine_pad = int(self.engine_left_pad)
        if engine_pad:
            x = F.pad(x, (engine_pad, 0, 0, 0))
        in_len = int(x.size(3))
        model_pad = 0
        if in_len < self.receptive_field:
            model_pad = self.receptive_field - in_len
            x = F.pad(x, (model_pad, 0, 0, 0))
        x = self.start_conv(x)
        skip: torch.Tensor | None = None
        static = self.support_list()
        adp = self.adaptive_adjacency()
        if not torch.isfinite(adp).all():
            raise RuntimeError("adaptive adjacency is not finite")
        new_supports = static + [adp]
        layer_trace = []
        for index in range(self.blocks * self.layers):
            residual = x
            residual_t = int(residual.size(3))
            filt = torch.tanh(self.filter_convs[index](residual))
            gate = torch.sigmoid(self.gate_convs[index](residual))
            x = filt * gate
            gated_t = int(x.size(3))
            skip_branch = self.skip_convs[index](x)
            if skip is None:
                skip = skip_branch
            else:
                skip = skip[:, :, :, -skip_branch.size(3) :] + skip_branch
            x = self.gconv[index](x, new_supports)
            x = x + residual[:, :, :, -x.size(3) :]
            x = self.bn[index](x)
            layer_trace.append(
                {
                    "layer": index,
                    "dilation": int(self.filter_convs[index].dilation[1]),
                    "kernel_size": self.kernel_size,
                    "residual_time_length": residual_t,
                    "input_time_length": residual_t,
                    "gated_time_length": gated_t,
                    "output_time_length": int(x.size(3)),
                    "skip_time_length": int(skip_branch.size(3)),
                }
            )
        if skip is None:
            raise RuntimeError("Graph WaveNet produced no skip path")
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        if x.size(3) != 1:
            raise RuntimeError(f"GWN time dimension after end conv is {x.size(3)}, expected 1")
        outputs = x.permute(0, 3, 2, 1).contiguous()
        expected = (batch, 1, self.num_nodes, self.out_dim)
        if outputs.shape != expected:
            raise RuntimeError(f"GWN output {tuple(outputs.shape)} != {expected}")
        if return_trace:
            trace = {
                "input": list(inputs.shape),
                "internal_bcvt_after_permute": [batch, channels, nodes, seq_len],
                "engine_left_pad": engine_pad,
                "model_extra_left_pad": model_pad,
                "padded_time_length": int(seq_len + engine_pad + model_pad),
                "receptive_field": self.receptive_field,
                "layers": layer_trace,
                "adaptive_shape": list(adp.shape),
                "static_support_count": self.num_static_supports,
                "support_count_with_adaptive": len(new_supports),
                "final_skip_time": int(skip.size(3)),
                "output": list(outputs.shape),
            }
            return outputs, trace
        return outputs
