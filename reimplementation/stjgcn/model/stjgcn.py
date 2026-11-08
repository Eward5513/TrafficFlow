"""PeMSD7 STJGCN, with ``Q=1`` and an output width of ``C_out``.

The four STJGC layers keep the reference assignments. Distinct names replace
the original ``z2 = STJGC(...)`` overwrite so the residual source stays visible:

* layer 1 reads ``x[:, ::2]`` and ``x[:, 1::2]`` with graphs ``[1, 0]``;
  residual is ``x[:, 1::2] + h1`` (positions 1,3,5,7,9,11);
* layer 2 splits that residual with graphs ``[2, 0]``;
  residual is ``h1[:, 1::2] + h2`` (positions 3,7,11);
* layer 3 uses ``[:, :-1]`` and ``[:, 1:]`` with graphs ``[4, 0]``;
  residual is ``h2[:, 1:] + h3`` (positions 7,11);
* layer 4 uses the same offset split and graphs ``[4, 0]`` (position 11).

``trace_time_positions`` applies those slices to ``arange(12)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from reimplementation.common.errors import ReimplementationError
from reimplementation.stjgcn.model.layers import AdaptiveGraph, Conv2dTF, FC, glorot_uniform_


def trace_time_positions(history: int = 12) -> dict[str, list[int]]:
    """Timestamps labeled by the PeMSD7 comments, derived from the same slices."""
    index = torch.arange(int(history))
    layer1 = index[1::2]
    layer2 = layer1[1::2]
    layer3 = layer2[1:]
    layer4 = layer3[1:]
    return {
        "layer1": layer1.tolist(),
        "layer2": layer2.tolist(),
        "layer3": layer3.tolist(),
        "layer4": layer4.tolist(),
        "graph_index": [[1, 0], [2, 0], [4, 0], [4, 0]],
    }


class Embedding(nn.Module):
    def __init__(self, num_nodes: int, hidden: int, slots_per_day: int = 288):
        super().__init__()
        self.slots_per_day = int(slots_per_day)
        self.spatial = nn.Parameter(torch.empty(1, 1, int(num_nodes), int(hidden)))
        glorot_uniform_(self.spatial)
        self.spatial_fc = FC(hidden, [hidden, hidden], [F.relu, None], use_bias=True)
        self.temporal_fc = FC(self.slots_per_day + 7, [hidden, hidden], [F.relu, None], use_bias=True)
        self.fusion_fc = FC(hidden, [hidden, hidden], [F.relu, None], use_bias=True)

    def forward(self, time_index: torch.Tensor) -> torch.Tensor:
        day = F.one_hot(time_index[..., 0].to(dtype=torch.long), 7).to(dtype=self.spatial.dtype)
        slot = F.one_hot(time_index[..., 1].to(dtype=torch.long), self.slots_per_day).to(dtype=self.spatial.dtype)
        temporal = torch.cat((day, slot), dim=-1).unsqueeze(2)
        return self.fusion_fc(self.spatial_fc(self.spatial) + self.temporal_fc(temporal))


class PredefinedBranch(nn.Module):
    def __init__(self, hidden: int, *, use_bias: bool):
        super().__init__()
        self.fc = Conv2dTF(hidden, hidden, use_bias=use_bias, activation=F.relu)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # z[b,t,i] = sum_j A[i,j] x[b,t,j]
        mixed = torch.einsum("ij,btjd->btid", adjacency, x)
        return self.fc(mixed)


class STJGC(nn.Module):
    """One PeMSD7 STJGC call. ``K=2``, so there are exactly two input branches."""

    def __init__(self, hidden: int, *, delta_adt: float, chunk_nodes: int):
        super().__init__()
        self.delta_adt = float(delta_adt)
        self.pdf_fw_early = PredefinedBranch(hidden, use_bias=False)
        self.pdf_bw_early = PredefinedBranch(hidden, use_bias=False)
        self.pdf_fw_last = PredefinedBranch(hidden, use_bias=False)
        self.pdf_bw_last = PredefinedBranch(hidden, use_bias=True)
        self.adt_forward = AdaptiveGraph(hidden, use_bias=False, chunk_nodes=chunk_nodes)
        self.adt_backward = AdaptiveGraph(hidden, use_bias=False, chunk_nodes=chunk_nodes)
        self.adt_last = AdaptiveGraph(hidden, use_bias=True, chunk_nodes=chunk_nodes)
        self.gate_pdf = Conv2dTF(hidden, hidden, use_bias=False, activation=None)
        self.gate_adt = Conv2dTF(hidden, hidden, use_bias=True, activation=None)
        self.fuse = FC(hidden, [hidden, hidden], [F.relu, None], use_bias=True)

    def forward(
        self,
        branches: list[torch.Tensor],
        adjacency_fw: list[torch.Tensor],
        adjacency_bw: list[torch.Tensor],
        embeddings: list[torch.Tensor],
        interaction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(branches) != 2:
            raise ReimplementationError("PeMSD7 STJGC is built for K=2")
        early, last = branches
        embed_early, embed_last = embeddings
        pdf = self.pdf_fw_early(early, adjacency_fw[0]) + self.pdf_bw_early(early, adjacency_bw[0])
        pdf = pdf + self.pdf_fw_last(last, adjacency_fw[1]) + self.pdf_bw_last(last, adjacency_bw[1])
        adaptive = self.adt_forward(early, embed_early, embed_last, interaction, self.delta_adt)
        adaptive = adaptive + self.adt_backward(early, embed_last, embed_early, interaction, self.delta_adt)
        adaptive = adaptive + self.adt_last(last, embed_last, embed_last, interaction, self.delta_adt)
        gate = torch.sigmoid(self.gate_pdf(pdf) + self.gate_adt(adaptive))
        fused = gate * pdf + (1.0 - gate) * adaptive
        return self.fuse(fused), embed_last


class MultiRangeAttention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.score_hidden = Conv2dTF(hidden, hidden, use_bias=True, activation=torch.tanh)
        self.score = Conv2dTF(hidden, 1, use_bias=False, activation=None)
        self.output = FC(hidden, [hidden, hidden], [F.relu, None], use_bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 4, N, d] -> [B, N, 1, d]
        hidden = x.permute(0, 2, 1, 3).contiguous()
        weight = self.score(self.score_hidden(hidden))
        weight = torch.softmax(weight.permute(0, 1, 3, 2), dim=-1)
        context = weight.matmul(hidden)
        return self.output(context)


class STJGCN(nn.Module):
    def __init__(
        self,
        *,
        num_nodes: int,
        input_channels: int,
        output_channels: int,
        hidden: int = 64,
        history: int = 12,
        slots_per_day: int = 288,
        kernel: int = 2,
        delta_adt: float = 0.7,
        chunk_nodes: int = 64,
    ):
        super().__init__()
        if int(kernel) != 2:
            raise ReimplementationError("PeMSD7 STJGCN requires K=2")
        if int(history) != 12:
            raise ReimplementationError("PeMSD7 layer slices are defined for P=12")
        self.num_nodes = int(num_nodes)
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.hidden = int(hidden)
        self.history = int(history)
        self.embedding = Embedding(self.num_nodes, self.hidden, slots_per_day=int(slots_per_day))
        self.interaction = nn.Parameter(torch.empty(1, 1, self.hidden, self.hidden))
        glorot_uniform_(self.interaction)
        self.input_fc = FC(self.input_channels, [self.hidden, self.hidden], [F.relu, None], use_bias=True)
        self.layers = nn.ModuleList(
            [STJGC(self.hidden, delta_adt=float(delta_adt), chunk_nodes=int(chunk_nodes)) for _ in range(4)]
        )
        self.attention = MultiRangeAttention(self.hidden)
        self.prediction = FC(self.hidden, [self.hidden, self.output_channels], [F.relu, None], use_bias=True)
        self.graph_index = ((1, 0), (2, 0), (4, 0), (4, 0))

    @classmethod
    def from_config(cls, config: dict) -> "STJGCN":
        return cls(
            num_nodes=int(config["num_nodes"]),
            input_channels=int(config["input_channels"]),
            output_channels=int(config["output_channels"]),
            hidden=int(config.get("d", 64)),
            history=int(config.get("P", config.get("n_his", 12))),
            slots_per_day=int(config.get("T", config.get("slots_per_day", 288))),
            kernel=int(config.get("K", 2)),
            delta_adt=float(config.get("delta_adt", 0.7)),
            chunk_nodes=int(config.get("attention_node_chunk", 64)),
        )

    def _graphs(self, adjacency_fw: torch.Tensor, adjacency_bw: torch.Tensor, pair: tuple[int, int]):
        return [adjacency_fw[pair[0]], adjacency_fw[pair[1]]], [adjacency_bw[pair[0]], adjacency_bw[pair[1]]]

    def forward(self, x: torch.Tensor, time_index: torch.Tensor, adjacency_fw: torch.Tensor, adjacency_bw: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or int(x.shape[1]) != self.history or int(x.shape[-1]) != self.input_channels:
            raise ReimplementationError(f"STJGCN x {tuple(x.shape)} is not [B, 12, N, C_in]")
        if time_index.ndim != 3 or int(time_index.shape[1]) != self.history or int(time_index.shape[-1]) != 2:
            raise ReimplementationError(f"STJGCN time index {tuple(time_index.shape)} is not [B, 12, 2]")
        if adjacency_fw.shape != adjacency_bw.shape or int(adjacency_fw.shape[0]) < 5:
            raise ReimplementationError("predefined STJG tensors must cover graph indexes 0, 1, 2, and 4")
        positions = trace_time_positions(self.history)
        embedding = self.embedding(time_index)
        hidden = self.input_fc(x)
        older, newer = hidden[:, ::2], hidden[:, 1::2]
        embed_older, embed_newer = embedding[:, ::2], embedding[:, 1::2]
        fw, bw = self._graphs(adjacency_fw, adjacency_bw, self.graph_index[0])
        h1, embed = self.layers[0]([older, newer], fw, bw, [embed_older, embed_newer], self.interaction)
        if int(h1.shape[1]) != len(positions["layer1"]):
            raise ReimplementationError("STJGC layer 1 time length drifted from the PeMSD7 slices")
        residual = newer + h1
        fw, bw = self._graphs(adjacency_fw, adjacency_bw, self.graph_index[1])
        h2, embed = self.layers[1]([residual[:, ::2], residual[:, 1::2]], fw, bw, [embed[:, ::2], embed[:, 1::2]], self.interaction)
        if int(h2.shape[1]) != len(positions["layer2"]):
            raise ReimplementationError("STJGC layer 2 time length drifted from the PeMSD7 slices")
        residual = h1[:, 1::2] + h2
        fw, bw = self._graphs(adjacency_fw, adjacency_bw, self.graph_index[2])
        h3, embed = self.layers[2]([residual[:, :-1], residual[:, 1:]], fw, bw, [embed[:, :-1], embed[:, 1:]], self.interaction)
        if int(h3.shape[1]) != len(positions["layer3"]):
            raise ReimplementationError("STJGC layer 3 time length drifted from the PeMSD7 slices")
        residual = h2[:, 1:] + h3
        fw, bw = self._graphs(adjacency_fw, adjacency_bw, self.graph_index[3])
        h4, _ = self.layers[3]([residual[:, :-1], residual[:, 1:]], fw, bw, [embed[:, :-1], embed[:, 1:]], self.interaction)
        if int(h4.shape[1]) != len(positions["layer4"]):
            raise ReimplementationError("STJGC layer 4 time length drifted from the PeMSD7 slices")
        stacked = torch.cat((h1[:, -1:], h2[:, -1:], h3[:, -1:], h4[:, -1:]), dim=1)
        context = self.attention(stacked)
        prediction = self.prediction(context).permute(0, 2, 1, 3).contiguous()
        if prediction.ndim != 4 or int(prediction.shape[1]) != 1 or int(prediction.shape[-1]) != self.output_channels:
            raise ReimplementationError(f"STJGCN output {tuple(prediction.shape)} is not [B, 1, N, C_out]")
        return prediction
