"""Input embedding: value projection plus road, time-of-day, and day-of-week embeddings.

``value_embedding_mode="joint_linear"`` (default) maps every input channel with
one ``Linear(C_in, value_dim)`` stored as ``value_projection``.
``value_embedding_mode="separate_mlp"`` requires four channels and concat
fusion. Each channel keeps its own ``Linear(1, width) -> SiLU -> Linear(width, width)``,
``width = value_dim / 4``, and the four outputs are concatenated back to
``value_dim``. Slice order is the dataset channel axis: inflow, outflow,
accumulation_start, accumulation_end. The MLPs share no parameters. Dropout
stays on the finished base embedding; the MLPs themselves have none.

``fusion="add"`` (v0.1): every part maps to ``d_model`` and the parts are
summed. ``fusion="concat"`` (v0.2 default): each part has its own width,
is broadcast to ``[B, T, N, d_i]``, and the parts are concatenated in the
order value, node, TOD, DOW; the widths must sum to ``d_model``. A disabled
part contributes zeros of its width so the backbone input stays ``d_model``.
Parameter names are the same in both modes. TOD/DOW are known calendar
features passed as integer indexes; they are not traffic channels and do not
count toward ``in_channels``.
"""

from __future__ import annotations

import torch
from torch import nn

FUSION_MODES = ("add", "concat")
VALUE_EMBEDDING_MODES = ("joint_linear", "separate_mlp")
_SEPARATE_VALUE_CHANNELS = 4


def resolve_embedding_dims(d_model, base_dim=None, adaptive_dim=None, *, adaptive=False, fusion="add"):
    """Resolve base/table/backbone widths without changing legacy additive models."""
    if fusion not in FUSION_MODES:
        raise ValueError(f"unknown adaptive fusion {fusion!r}")
    base = int(d_model if base_dim is None else base_dim)
    table = int(base if adaptive_dim is None else adaptive_dim)
    output = base + table if adaptive and fusion == "concat" else base
    if base < 1 or table < 1 or output != int(d_model):
        raise ValueError(f"embedding base={base}, adaptive={table}, fusion={fusion}, enabled={adaptive} != backbone={d_model}")
    if adaptive and fusion == "add" and base != table:
        raise ValueError("add adaptive embedding requires equal base and table widths")
    return base, table


class InputEmbedding(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        d_model: int,
        num_nodes: int,
        steps_per_day: int = 288,
        use_node_embedding: bool = True,
        use_tod: bool = True,
        use_dow: bool = True,
        dropout: float = 0.0,
        fusion: str = "add",
        value_embedding_mode: str = "joint_linear",
        value_dim: int | None = None,
        node_dim: int | None = None,
        tod_dim: int | None = None,
        dow_dim: int | None = None,
    ) -> None:
        super().__init__()
        if fusion not in FUSION_MODES:
            raise ValueError(f"embedding fusion {fusion!r} not in {FUSION_MODES}")
        self.fusion = fusion
        self.in_channels = int(in_channels)
        self.num_nodes = int(num_nodes)
        self.steps_per_day = int(steps_per_day)
        self.d_model = int(d_model)
        self.widths = embedding_widths(fusion, d_model, value_dim, node_dim, tod_dim, dow_dim)
        if value_embedding_mode not in VALUE_EMBEDDING_MODES:
            raise ValueError(f"value_embedding_mode {value_embedding_mode!r} not in {VALUE_EMBEDDING_MODES}")
        self.value_embedding_mode = value_embedding_mode
        self.value_branch_dim: int | None = None
        if value_embedding_mode == "separate_mlp":
            self.value_mlps = self._separate_value_mlps()
        else:
            self.value_projection = nn.Linear(self.in_channels, self.widths["value"])
        self.node_embedding = nn.Embedding(self.num_nodes, self.widths["node"]) if use_node_embedding else None
        self.tod_embedding = nn.Embedding(self.steps_per_day, self.widths["tod"]) if use_tod else None
        self.dow_embedding = nn.Embedding(7, self.widths["dow"]) if use_dow else None
        self.dropout = nn.Dropout(dropout)
        for table in (self.node_embedding, self.tod_embedding, self.dow_embedding):
            if table is not None:
                nn.init.trunc_normal_(table.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        time_of_day: torch.Tensor | None = None,
        day_of_week: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``x`` [B, T, N, C_in]; ``time_of_day``/``day_of_week`` [B, T] long -> [B, T, N, D]."""
        if x.ndim != 4 or int(x.shape[-1]) != self.in_channels:
            raise ValueError(f"expected [B, T, N, {self.in_channels}], got {tuple(x.shape)}")
        batch, steps, nodes = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        if self.value_embedding_mode == "joint_linear":
            value = self.value_projection(x)
        elif self.value_embedding_mode == "separate_mlp":
            # One shared MLP per input channel. Index 0..3 follows STATE_CHANNELS.
            value = torch.cat(
                [mlp(x[..., channel:channel + 1]) for channel, mlp in enumerate(self.value_mlps)],
                dim=-1,
            )
        else:
            raise ValueError(f"unknown value_embedding_mode {self.value_embedding_mode!r}")
        node = tod = dow = None
        if self.node_embedding is not None:
            if nodes != self.num_nodes:
                raise ValueError(f"node axis {nodes} != embedding table {self.num_nodes}")
            node = self.node_embedding.weight[None, None, :, :]
        if self.tod_embedding is not None:
            tod = self.tod_embedding(_check_index(time_of_day, "time_of_day", batch, steps, self.steps_per_day))[:, :, None, :]
        if self.dow_embedding is not None:
            dow = self.dow_embedding(_check_index(day_of_week, "day_of_week", batch, steps, 7))[:, :, None, :]
        if self.fusion == "add":
            hidden = value
            for part in (node, tod, dow):
                if part is not None:
                    hidden = hidden + part
            return self.dropout(hidden)
        shape = (batch, steps, nodes)
        parts = [value]
        for name, part in (("node", node), ("tod", tod), ("dow", dow)):
            if part is None:
                parts.append(value.new_zeros(*shape, self.widths[name]))
            else:
                parts.append(part.expand(*shape, self.widths[name]))
        return self.dropout(torch.cat(parts, dim=-1))

    def _separate_value_mlps(self) -> nn.ModuleList:
        if self.in_channels != _SEPARATE_VALUE_CHANNELS:
            raise ValueError(
                f"separate_mlp value embedding requires in_channels={_SEPARATE_VALUE_CHANNELS}, got {self.in_channels}"
            )
        if self.fusion != "concat":
            raise ValueError(f"separate_mlp value embedding requires fusion='concat', got {self.fusion!r}")
        value_dim = int(self.widths["value"])
        if value_dim % _SEPARATE_VALUE_CHANNELS != 0 or value_dim // _SEPARATE_VALUE_CHANNELS < 1:
            raise ValueError(
                "separate_mlp value width must be a positive multiple of "
                f"{_SEPARATE_VALUE_CHANNELS}, got {value_dim}"
            )
        branch_dim = value_dim // _SEPARATE_VALUE_CHANNELS
        self.value_branch_dim = branch_dim
        # A new Sequential on every iteration. Reusing one module would tie the four channels together.
        mlps = [_separate_value_mlp(branch_dim) for _ in range(_SEPARATE_VALUE_CHANNELS)]
        if len({id(mlp) for mlp in mlps}) != _SEPARATE_VALUE_CHANNELS:
            raise RuntimeError("separate value MLPs must be distinct module objects")
        return nn.ModuleList(mlps)

    def value_embedding_summary(self) -> dict[str, object]:
        """Mode, total value width, and the separate-MLP layout when that mode is built."""
        summary: dict[str, object] = {
            "mode": self.value_embedding_mode,
            "output_dim": int(self.widths["value"]),
        }
        if self.value_embedding_mode == "joint_linear":
            projection = self.value_projection
            summary.update(
                module="value_projection",
                in_features=int(projection.in_features),
                out_features=int(projection.out_features),
                bias=projection.bias is not None,
            )
            return summary
        mlps = list(self.value_mlps)
        descriptions = [_mlp_spec(mlp) for mlp in mlps]
        branch_widths = [int(mlp[-1].out_features) for mlp in mlps]
        hidden_widths = [int(mlp[0].out_features) for mlp in mlps]
        parameter_ptrs = [int(param.data_ptr()) for mlp in mlps for param in mlp.parameters()]
        summary.update(
            branch_count=len(mlps),
            independent_parameters=(
                len(mlps) == _SEPARATE_VALUE_CHANNELS
                and len({id(mlp) for mlp in mlps}) == len(mlps)
                and len(parameter_ptrs) == len(set(parameter_ptrs))
            ),
            branch_width=branch_widths[0] if len(set(branch_widths)) == 1 else branch_widths,
            hidden_width=hidden_widths[0] if len(set(hidden_widths)) == 1 else hidden_widths,
            mlp=descriptions[0] if len(set(descriptions)) == 1 else descriptions,
            parameter_scope="one MLP per input channel, shared across batch, time, and roads",
            bias=all(layer.bias is not None for mlp in mlps for layer in mlp if isinstance(layer, nn.Linear)),
        )
        return summary


def _separate_value_mlp(branch_dim: int) -> nn.Sequential:
    width = int(branch_dim)
    return nn.Sequential(
        nn.Linear(1, width),
        nn.SiLU(),
        nn.Linear(width, width),
    )


def _mlp_spec(module: nn.Sequential) -> str:
    parts: list[str] = []
    for layer in module:
        if isinstance(layer, nn.Linear):
            parts.append(f"Linear({int(layer.in_features)},{int(layer.out_features)})")
        elif isinstance(layer, nn.SiLU):
            parts.append("SiLU")
        else:
            parts.append(type(layer).__name__)
    return " -> ".join(parts)


def embedding_widths(
    fusion: str,
    d_model: int,
    value_dim: int | None,
    node_dim: int | None,
    tod_dim: int | None,
    dow_dim: int | None,
) -> dict[str, int]:
    """Per-part output width; ``add`` ignores the configured dims and uses ``d_model`` for all."""
    d_model = int(d_model)
    if fusion == "add":
        return {"value": d_model, "node": d_model, "tod": d_model, "dow": d_model}
    widths = {"value": value_dim, "node": node_dim, "tod": tod_dim, "dow": dow_dim}
    missing = [name for name, width in widths.items() if width is None]
    if missing:
        raise ValueError(f"concat embedding needs explicit widths for {missing}")
    widths = {name: int(width) for name, width in widths.items()}
    bad = {name: width for name, width in widths.items() if width < 1}
    if bad:
        raise ValueError(f"concat embedding widths must be positive, got {bad}")
    total = sum(widths.values())
    if total != d_model:
        raise ValueError(
            f"concat embedding widths value={widths['value']} + node={widths['node']} + "
            f"tod={widths['tod']} + dow={widths['dow']} = {total} != d_model={d_model}"
        )
    return widths


def _check_index(index: torch.Tensor | None, name: str, batch: int, steps: int, size: int) -> torch.Tensor:
    if index is None:
        raise ValueError(f"{name} is required when its embedding is enabled")
    if tuple(index.shape) != (batch, steps):
        raise ValueError(f"{name} must be [B, T]=({batch}, {steps}), got {tuple(index.shape)}")
    if index.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be an integer tensor")
    if bool((index < 0).any()) or bool((index >= size).any()):
        raise ValueError(f"{name} outside [0, {size})")
    return index.long()
