"""MSTT v0.4: encoder-only Modern Spatio-Temporal Transformer.

``forward(x [B, T, N, C_in], time_of_day [B, T], day_of_week [B, T]) -> [B, N, C_out]``.
The model reads no files or tasks; graph and optional target scaler are injected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Mapping

import torch
from torch import nn

from MSTT.model.adaptive_embedding import AdaptiveEmbedding
from MSTT.model.embedding import InputEmbedding, resolve_embedding_dims
from MSTT.model.dual_history_tsf import DualPathHistoryTSF
from MSTT.model.graph import SpatialGraph
from MSTT.model.ffn import SwiGLU
from MSTT.model.head import PredictionHead
from MSTT.model.history_fusion import HistoryPredictionFusion
from MSTT.model.joint_attention import JointSTAttention
from MSTT.model.conservation import ConservationProjection
from MSTT.model.state_branch import StateTSFBranch
from MSTT.model.state_context import SharedStateContextEmbedding
from MSTT.model.state_fusion_head import StateTokenTransformerHead
from MSTT.model.transformer_block import JointSTBlock, build_factorized_block

STATE_BRANCH_COUNT = 4
_DIVISIBLE_WIDTHS = (
    "d_model",
    "base_embedding_dim",
    "adaptive_embedding_dim",
    "value_embedding_dim",
    "node_embedding_dim",
    "tod_embedding_dim",
    "dow_embedding_dim",
)


@dataclass
class MSTTConfig:
    num_nodes: int
    in_channels: int
    out_channels: int
    input_window: int = 12
    d_model: int = 128
    num_heads: int = 4
    temporal_num_heads: int | None = None
    spatial_num_heads: int | None = None
    attention_head_dim: int | None = None
    num_layers: int = 1
    ffn_hidden_dim: int = 176
    dropout: float = 0.1
    attention_dropout: float = 0.1
    embedding_dropout: float = 0.1
    use_qk_norm: bool = True
    use_rope: bool = True
    rope_base: float = 10000.0
    use_gated_attention: bool = False
    spatial_attention: str = "global"
    use_edge_type_bias: bool = False
    spatial_chunk_size: int | None = 64
    spatial_checkpoint: bool = True
    use_tod: bool = True
    use_dow: bool = True
    use_node_embedding: bool = True
    embedding_fusion: str = "concat"
    value_embedding_mode: str = "joint_linear"
    value_embedding_dim: int = 32
    node_embedding_dim: int = 16
    tod_embedding_dim: int = 8
    dow_embedding_dim: int = 8
    use_adaptive_embedding: bool = True
    base_embedding_dim: int | None = 64
    adaptive_embedding_dim: int | None = 64
    adaptive_embedding_fusion: str = "concat"
    ffn_placement: str = "after_spatial"
    attention_order: str = "TS"
    attention_layout: str = "factorized_ts"
    joint_temporal_window: int = 3
    joint_temporal_alignment: str = "centered"
    joint_query_chunk: int = 256
    joint_checkpoint: bool = True
    steps_per_day: int = 288
    norm_eps: float = 1e-6
    head_type: str = "flatten"
    conservation_mode: str = "none"
    projection_mlp_hidden_dim: int = 32
    backbone_mode: str = "shared"
    state_context_sharing: str = "independent"
    state_node_sharing: str = "independent"
    fusion_mlp_hidden_dim: int = 128
    fusion_token_dim: int = 128
    fusion_num_heads: int = 4
    fusion_ffn_hidden_dim: int = 176
    fusion_chunk_size: int | None = None
    fusion_checkpoint: bool = False
    # Off by default so an older checkpoint has no extra parameters and consumes no extra RNG.
    temporal_aux_enabled: bool = False
    # Off by default. Older auxiliary runs load as ``none`` and keep their parameter set.
    history_fusion_mode: str = "none"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], **overrides: Any) -> "MSTTConfig":
        names = {item.name for item in fields(cls)}
        values = {key: payload[key] for key in names if key in payload}
        values.update(overrides)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cfg_get(config: MSTTConfig | Mapping[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config[key] if key in config else default
    return getattr(config, key, default)


def validate_independent_backbone(config: MSTTConfig | Mapping[str, Any]) -> None:
    """Widths and structure for four independent encoders.

    ``d_model`` and the embedding widths are totals. Each branch uses one
    quarter. ``ffn_hidden_dim`` and ``attention_head_dim`` are not quartered:
    the FFN width is the width inside one branch, and a set head dimension is
    the per-head size.
    """
    mode = str(_cfg_get(config, "value_embedding_mode", "joint_linear"))
    if mode in {"state_aux_gate", "state_aux_gate_split_mlp"}:
        raise ValueError(f"backbone_mode=state_independent cannot use value_embedding_mode={mode}")
    if mode != "separate_mlp":
        raise ValueError(f"backbone_mode=state_independent requires value_embedding_mode='separate_mlp', got {mode!r}")
    requirements = {
        "embedding_fusion": "concat",
        "adaptive_embedding_fusion": "concat",
        "use_adaptive_embedding": True,
        "attention_layout": "factorized_ts",
        "attention_order": "TS",
        "ffn_placement": "after_spatial",
        "num_layers": 1,
        "use_node_embedding": True,
        "use_tod": True,
        "use_dow": True,
    }
    for key, expected in requirements.items():
        actual = _cfg_get(config, key, None)
        if actual != expected:
            raise ValueError(f"backbone_mode=state_independent requires {key}={expected!r}, got {actual!r}")
    for key in ("in_channels", "out_channels"):
        actual = _cfg_get(config, key, None)
        if actual is not None and int(actual) != STATE_BRANCH_COUNT:
            raise ValueError(f"backbone_mode=state_independent requires {key}={STATE_BRANCH_COUNT}, got {actual}")
    totals: dict[str, int] = {}
    for key in _DIVISIBLE_WIDTHS:
        actual = _cfg_get(config, key, None)
        if actual is None:
            raise ValueError(f"backbone_mode=state_independent requires {key} to be set")
        value = int(actual)
        if value < 1 or value % STATE_BRANCH_COUNT != 0:
            raise ValueError(
                f"backbone_mode=state_independent requires {key} to be a positive multiple of "
                f"{STATE_BRANCH_COUNT}, got {value}"
            )
        totals[key] = value
    branch = {key: value // STATE_BRANCH_COUNT for key, value in totals.items()}
    base_sum = branch["value_embedding_dim"] + branch["node_embedding_dim"] + branch["tod_embedding_dim"] + branch["dow_embedding_dim"]
    if base_sum != branch["base_embedding_dim"]:
        raise ValueError(
            "state_independent branch widths must satisfy "
            f"value+node+tod+dow={base_sum} == base={branch['base_embedding_dim']}"
        )
    if branch["base_embedding_dim"] + branch["adaptive_embedding_dim"] != branch["d_model"]:
        raise ValueError(
            "state_independent branch widths must satisfy "
            f"base+adaptive={branch['base_embedding_dim'] + branch['adaptive_embedding_dim']} "
            f"== branch d_model={branch['d_model']}"
        )
    ffn = _cfg_get(config, "ffn_hidden_dim", None)
    if ffn is not None and int(ffn) < 1:
        raise ValueError(f"ffn_hidden_dim must be positive, got {ffn}")
    branch_d = branch["d_model"]
    head_dim_override = _cfg_get(config, "attention_head_dim", None)
    use_rope = bool(_cfg_get(config, "use_rope", False))
    for label, key in (("temporal", "temporal_num_heads"), ("spatial", "spatial_num_heads")):
        override = _cfg_get(config, key, None)
        heads = int(_cfg_get(config, "num_heads")) if override is None else int(override)
        if heads < 1:
            raise ValueError(f"{key} must be a positive head count, got {heads}")
        if head_dim_override is None:
            if branch_d % heads != 0:
                raise ValueError(
                    f"state_independent branch d_model={branch_d} must be divisible by {label} heads={heads}"
                )
            head_dim = branch_d // heads
        else:
            head_dim = int(head_dim_override)
            if head_dim < 1:
                raise ValueError(f"attention_head_dim must be positive, got {head_dim}")
        if use_rope and head_dim % 2 != 0:
            raise ValueError(
                f"RoPE needs an even {label} head_dim on each state branch, got {head_dim} "
                f"(branch d_model={branch_d}, heads={heads})"
            )


_HISTORY_FUSION_MODES = (
    "none",
    "gated_cross_attention",
    "dual_path_history_tsf",
    "two_stage_projected_tsf",
)
_FLATTEN_HISTORY_MODES = ("gated_cross_attention", "dual_path_history_tsf")


def validate_history_fusion(config: MSTTConfig | Mapping[str, Any]) -> None:
    """Active history modes keep the observed encoder at width 128.

    ``gated_cross_attention`` and ``dual_path_history_tsf`` read the 11-step
    auxiliary head and require ``head_type='flatten'``.
    ``two_stage_projected_tsf`` rebuilds all 12 steps, runs a second TSF on
    those projected values, and requires ``head_type='last'``.
    ``none`` builds no extra parameters.
    """
    mode = str(_cfg_get(config, "history_fusion_mode", "none"))
    if mode not in _HISTORY_FUSION_MODES:
        raise ValueError(
            "history_fusion_mode must be 'none', 'gated_cross_attention', "
            "'dual_path_history_tsf', or 'two_stage_projected_tsf', "
            f"got {mode!r}"
        )
    if mode == "none":
        return
    if not bool(_cfg_get(config, "temporal_aux_enabled", False)):
        raise ValueError(f"history_fusion_mode={mode!r} requires temporal_aux_enabled")
    window = _cfg_get(config, "input_window", None)
    if window is None or int(window) != 12:
        raise ValueError(f"history_fusion_mode={mode!r} requires input_window=12")
    out_channels = _cfg_get(config, "out_channels", None)
    if out_channels is not None and int(out_channels) != 4:
        raise ValueError(
            f"history_fusion_mode={mode!r} requires 4 output states, got out_channels={out_channels}"
        )
    if str(_cfg_get(config, "backbone_mode", "shared")) != "shared":
        raise ValueError(f"history_fusion_mode={mode!r} requires backbone_mode='shared'")
    if str(_cfg_get(config, "conservation_mode", "none")) != "projected":
        raise ValueError(f"history_fusion_mode={mode!r} requires conservation_mode='projected'")
    d_model = _cfg_get(config, "d_model", None)
    if d_model is None or int(d_model) != 128:
        raise ValueError(f"history_fusion_mode={mode!r} requires d_model=128")
    head = str(_cfg_get(config, "head_type", "flatten"))
    if mode in _FLATTEN_HISTORY_MODES:
        if head != "flatten":
            raise ValueError(f"history_fusion_mode={mode!r} requires head_type='flatten'")
        return
    if head != "last":
        raise ValueError("history_fusion_mode='two_stage_projected_tsf' requires head_type='last'")
    layers = _cfg_get(config, "num_layers", None)
    if layers is None or int(layers) != 1:
        raise ValueError("history_fusion_mode='two_stage_projected_tsf' requires num_layers=1")
    if str(_cfg_get(config, "attention_layout", "factorized_ts")) != "factorized_ts":
        raise ValueError("history_fusion_mode='two_stage_projected_tsf' requires attention_layout='factorized_ts'")
    if str(_cfg_get(config, "value_embedding_mode", "joint_linear")) != "separate_mlp":
        raise ValueError("history_fusion_mode='two_stage_projected_tsf' requires value_embedding_mode='separate_mlp'")
    if not bool(_cfg_get(config, "use_adaptive_embedding", False)):
        raise ValueError("history_fusion_mode='two_stage_projected_tsf' requires use_adaptive_embedding")


def validate_state_token_transformer(config: MSTTConfig | Mapping[str, Any]) -> None:
    """The 48-token fusion head is only the four-state Node16-style TSF stack.

    ``fusion_token_dim`` is the fusion width. It does not replace backbone
    ``d_model``. QK-Norm stays on and this head does not use RoPE; the TSF
    blocks keep their own RoPE setting.
    """
    head = str(_cfg_get(config, "head_type", "flatten"))
    if head != "state_token_transformer":
        return
    backbone = str(_cfg_get(config, "backbone_mode", "shared"))
    if backbone != "state_independent":
        raise ValueError("head_type='state_token_transformer' requires backbone_mode='state_independent'")
    mode = str(_cfg_get(config, "value_embedding_mode", "joint_linear"))
    if mode != "separate_mlp":
        raise ValueError(
            "head_type='state_token_transformer' requires value_embedding_mode='separate_mlp', "
            f"got {mode!r}"
        )
    window = _cfg_get(config, "input_window", None)
    if window is not None and int(window) != 12:
        raise ValueError("head_type='state_token_transformer' requires input_window=12")
    for key in ("in_channels", "out_channels"):
        actual = _cfg_get(config, key, None)
        if actual is not None and int(actual) != STATE_BRANCH_COUNT:
            raise ValueError(
                f"head_type='state_token_transformer' requires {key}={STATE_BRANCH_COUNT}, got {actual}"
            )
    token_dim = int(_cfg_get(config, "fusion_token_dim", 0) or 0)
    heads = int(_cfg_get(config, "fusion_num_heads", 0) or 0)
    ffn = int(_cfg_get(config, "fusion_ffn_hidden_dim", 0) or 0)
    if token_dim < 1 or heads < 1 or token_dim % heads != 0:
        raise ValueError(
            f"fusion_token_dim={token_dim} must be a positive multiple of fusion_num_heads={heads}"
        )
    if ffn < 1:
        raise ValueError(f"fusion_ffn_hidden_dim must be positive, got {ffn}")
    if not bool(_cfg_get(config, "use_qk_norm", False)):
        raise ValueError("head_type='state_token_transformer' keeps fusion QK-Norm on")
    chunk = _cfg_get(config, "fusion_chunk_size", None)
    if chunk is not None and int(chunk) < 1:
        raise ValueError("fusion_chunk_size must be a positive integer or null")


def validate_state_node_sharing(config: MSTTConfig | Mapping[str, Any]) -> None:
    """Assign the road table. Missing values stay one private table per branch.

    ``shared`` is one parent ``[N, node_dim]`` read in full by every branch.
    It is only valid when four TSFs already share the TOD/DOW/adaptive tables.
    """
    sharing = str(_cfg_get(config, "state_node_sharing", "independent"))
    if sharing not in ("independent", "shared"):
        raise ValueError(f"state_node_sharing must be 'independent' or 'shared', got {sharing!r}")
    if sharing != "shared":
        return
    backbone = str(_cfg_get(config, "backbone_mode", "shared"))
    context = str(_cfg_get(config, "state_context_sharing", "independent"))
    if backbone != "state_independent" or context != "shared":
        raise ValueError(
            "state_node_sharing='shared' requires backbone_mode='state_independent' "
            "and state_context_sharing='shared'"
        )


def validate_state_context_sharing(config: MSTTConfig | Mapping[str, Any]) -> None:
    """Assign TOD, DOW, and adaptive tables. It does not change widths.

    ``independent`` keeps one table set inside each state branch.
    ``shared`` keeps one parent table set and feeds every branch the full
    vectors. ``backbone_mode=shared`` is one TSF and does not use the switch;
    only the placeholder ``independent`` is accepted there.
    """
    sharing = str(_cfg_get(config, "state_context_sharing", "independent"))
    if sharing not in ("independent", "shared"):
        raise ValueError(f"state_context_sharing must be 'independent' or 'shared', got {sharing!r}")
    backbone = str(_cfg_get(config, "backbone_mode", "shared"))
    if backbone != "state_independent" and sharing != "independent":
        raise ValueError(
            "state_context_sharing='shared' requires backbone_mode='state_independent' "
            "(four TSFs sharing TOD/DOW/adaptive tables). backbone_mode='shared' "
            "only accepts the unused placeholder 'independent'"
        )


def _assert_distinct_parameters(modules) -> None:
    pointers = [param.data_ptr() for module in modules for param in module.parameters()]
    if len(pointers) != len(set(pointers)):
        raise RuntimeError("state branches must not share parameter storage")


def _assert_parameters_disjoint(module: nn.Module, others) -> None:
    owned = [param.data_ptr() for param in module.parameters()]
    if len(owned) != len(set(owned)):
        raise RuntimeError("shared context parameters are duplicated inside the context module")
    other = {param.data_ptr() for branch in others for param in branch.parameters()}
    if set(owned) & other:
        raise RuntimeError("shared context parameters are also registered on a state branch")


class MSTT(nn.Module):
    def __init__(self, config: MSTTConfig, graph: SpatialGraph | None, *, target_mean=None, target_std=None) -> None:
        super().__init__()
        self.config = config
        if config.conservation_mode not in ("none", "soft", "projected"):
            raise ValueError(f"unknown conservation_mode {config.conservation_mode!r}")
        if config.conservation_mode != "none" and (
            config.out_channels != 4 or config.in_channels not in (4, 6)
        ):
            raise ValueError(
                "soft/projected conservation requires four output states and four or six input channels, "
                f"got in={config.in_channels} out={config.out_channels}"
            )
        if config.projection_mlp_hidden_dim < 1:
            raise ValueError("projection_mlp_hidden_dim must be positive")
        self.graph_direction = None if graph is None else str(graph.direction)
        self.graph_edges = None if graph is None else int(graph.num_edges)
        if graph is None:
            raise ValueError("MSTT needs a SpatialGraph to check the node count")
        if int(graph.num_nodes) != int(config.num_nodes):
            raise ValueError(f"graph has {graph.num_nodes} nodes, config {config.num_nodes}")
        base_dim, adaptive_dim = resolve_embedding_dims(
            config.d_model, config.base_embedding_dim, config.adaptive_embedding_dim,
            adaptive=config.use_adaptive_embedding, fusion=config.adaptive_embedding_fusion,
        )
        if config.num_layers != 1:
            raise ValueError("MSTT has exactly one block")
        layout = str(config.attention_layout)
        if layout == "factorized_ts":
            if config.attention_order != "TS" or config.ffn_placement != "after_spatial":
                raise ValueError("factorized_ts requires one T→S→FFN block")
        elif layout == "joint_st":
            if config.joint_temporal_alignment != "centered" or int(config.joint_temporal_window) % 2 != 1:
                raise ValueError("joint_st requires a centered odd temporal window")
        else:
            raise ValueError(f"unknown attention_layout {layout!r}")
        mode = str(config.backbone_mode)
        if mode not in ("shared", "state_independent"):
            raise ValueError(f"unknown backbone_mode {mode!r}")
        validate_state_context_sharing(config)
        validate_state_node_sharing(config)
        validate_state_token_transformer(config)
        if mode == "state_independent":
            validate_independent_backbone(config)
            branch_cfg = replace(
                config,
                in_channels=1,
                d_model=config.d_model // STATE_BRANCH_COUNT,
                base_embedding_dim=int(config.base_embedding_dim) // STATE_BRANCH_COUNT,
                adaptive_embedding_dim=int(config.adaptive_embedding_dim) // STATE_BRANCH_COUNT,
                value_embedding_dim=config.value_embedding_dim // STATE_BRANCH_COUNT,
                node_embedding_dim=config.node_embedding_dim // STATE_BRANCH_COUNT,
                tod_embedding_dim=config.tod_embedding_dim // STATE_BRANCH_COUNT,
                dow_embedding_dim=config.dow_embedding_dim // STATE_BRANCH_COUNT,
            )
            external_context = config.state_context_sharing == "shared"
            external_node = config.state_node_sharing == "shared"
            self.branches = nn.ModuleList(
                StateTSFBranch(
                    branch_cfg,
                    graph,
                    external_context=external_context,
                    external_node=external_node,
                )
                for _ in range(STATE_BRANCH_COUNT)
            )
            _assert_distinct_parameters(list(self.branches))
            if external_context:
                self.shared_state_context = SharedStateContextEmbedding(
                    steps_per_day=int(config.steps_per_day),
                    tod_dim=int(config.tod_embedding_dim) // STATE_BRANCH_COUNT,
                    dow_dim=int(config.dow_embedding_dim) // STATE_BRANCH_COUNT,
                    input_window=int(config.input_window),
                    num_nodes=int(config.num_nodes),
                    adaptive_dim=int(config.adaptive_embedding_dim) // STATE_BRANCH_COUNT,
                    node_dim=(int(config.node_embedding_dim) // STATE_BRANCH_COUNT) if external_node else None,
                )
                _assert_parameters_disjoint(self.shared_state_context, self.branches)
        else:
            temporal_heads = config.num_heads if config.temporal_num_heads is None else int(config.temporal_num_heads)
            self.embedding = InputEmbedding(
                in_channels=config.in_channels,
                d_model=base_dim,
                num_nodes=config.num_nodes,
                steps_per_day=config.steps_per_day,
                use_node_embedding=config.use_node_embedding,
                use_tod=config.use_tod,
                use_dow=config.use_dow,
                dropout=config.embedding_dropout,
                fusion=config.embedding_fusion,
                value_embedding_mode=config.value_embedding_mode,
                value_dim=config.value_embedding_dim,
                node_dim=config.node_embedding_dim,
                tod_dim=config.tod_embedding_dim,
                dow_dim=config.dow_embedding_dim,
            )
            if layout == "factorized_ts":
                block = build_factorized_block(config, graph)
            else:
                block = JointSTBlock(
                    config.d_model,
                    joint=JointSTAttention(
                        config.d_model,
                        temporal_heads,
                        window=int(config.joint_temporal_window),
                        alignment=str(config.joint_temporal_alignment),
                        query_chunk=int(config.joint_query_chunk),
                        use_checkpoint=bool(config.joint_checkpoint),
                        attention_dropout=config.attention_dropout,
                        use_qk_norm=config.use_qk_norm,
                        use_rope=config.use_rope,
                        rope_base=config.rope_base,
                        use_gated_attention=config.use_gated_attention,
                        norm_eps=config.norm_eps,
                        head_dim=config.attention_head_dim,
                    ),
                    ffn=SwiGLU(config.d_model, config.ffn_hidden_dim),
                    dropout=config.dropout,
                    norm_eps=config.norm_eps,
                )
            self.blocks = nn.ModuleList([block])
        if config.head_type == "state_token_transformer":
            self.head = StateTokenTransformerHead(
                config.d_model // STATE_BRANCH_COUNT,
                config.input_window,
                config.out_channels,
                token_dim=int(config.fusion_token_dim),
                num_heads=int(config.fusion_num_heads),
                ffn_hidden_dim=int(config.fusion_ffn_hidden_dim),
                dropout=float(config.dropout),
                attention_dropout=float(config.attention_dropout),
                use_qk_norm=bool(config.use_qk_norm),
                norm_eps=float(config.norm_eps),
                chunk_size=None if config.fusion_chunk_size is None else int(config.fusion_chunk_size),
                use_checkpoint=bool(config.fusion_checkpoint),
            )
        elif config.head_type in ("flatten", "last", "flatten_mlp"):
            self.head = PredictionHead(
                config.d_model,
                config.input_window,
                config.out_channels,
                head_type=config.head_type,
                norm_eps=config.norm_eps,
                fusion_mlp_hidden_dim=config.fusion_mlp_hidden_dim,
            )
        else:
            raise ValueError(f"unknown head_type {config.head_type!r}")
        if mode == "shared":
            # After the head so a shared run keeps embedding → block → head → adaptive.
            self.adaptive = (
                AdaptiveEmbedding(input_window=config.input_window, num_nodes=config.num_nodes,
                                  d_model=adaptive_dim, fusion=config.adaptive_embedding_fusion)
                if config.use_adaptive_embedding
                else None
            )
        self.projection = None
        if config.conservation_mode == "projected":
            if target_mean is None or target_std is None:
                raise ValueError("projected mode requires the training-only target scaler at initialization")
            self.projection = ConservationProjection(
                config.d_model, config.projection_mlp_hidden_dim, target_mean, target_std,
            )
        # After every existing module, so the same seed keeps the main-path initialization.
        self.temporal_aux_head = None
        if bool(config.temporal_aux_enabled):
            if int(config.out_channels) != 4 or int(config.input_window) != 12:
                raise ValueError("temporal_aux_enabled requires 12 input steps and 4 output states")
            self.temporal_aux_head = nn.Linear(int(config.d_model), 4)
        # After the auxiliary head, so a shared seed still matches every older parameter.
        self.history_fusion = None
        self.dual_history = None
        self.stage2_block = None
        self.stage1_projection = None
        validate_history_fusion(config)
        if self.temporal_aux_head is None or self.projection is None:
            if str(config.history_fusion_mode) != "none":
                raise ValueError("history fusion requires the temporal auxiliary head and projected conservation")
        elif str(config.history_fusion_mode) == "gated_cross_attention":
            self.history_fusion = HistoryPredictionFusion()
        elif str(config.history_fusion_mode) == "dual_path_history_tsf":
            self.dual_history = DualPathHistoryTSF(config, graph)
        elif str(config.history_fusion_mode) == "two_stage_projected_tsf":
            if not isinstance(self.head, PredictionHead) or self.head.head_type != "last":
                raise ValueError("two_stage_projected_tsf uses the last-step head as its residual readout")
            nn.init.zeros_(self.head.proj.weight)
            nn.init.zeros_(self.head.proj.bias)
            self.stage2_block = build_factorized_block(config, graph)
            self.stage1_projection = ConservationProjection(
                config.d_model, config.projection_mlp_hidden_dim, target_mean, target_std,
            )
            _assert_parameters_disjoint(self.stage2_block, self.blocks)
            _assert_parameters_disjoint(self.stage1_projection, [self.projection])

    def forward(
        self,
        x: torch.Tensor,
        time_of_day: torch.Tensor | None = None,
        day_of_week: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.projection is not None:
            return self.forward_with_aux(x, time_of_day, day_of_week)["final_pred"]
        return self.head(self._encode(x, time_of_day, day_of_week))

    def forward_with_aux(self, x, time_of_day=None, day_of_week=None) -> dict[str, torch.Tensor]:
        if self.stage2_block is not None:
            return self._forward_two_stage(x, time_of_day, day_of_week)
        hidden = self._encode(x, time_of_day, day_of_week)
        base = self.head(hidden)
        temporal_aux = None if self.temporal_aux_head is None else self.temporal_aux_head(hidden)
        raw = base
        branches = None
        if self.history_fusion is not None:
            if temporal_aux is None:
                raise RuntimeError("history fusion requires temporal_aux_pred from this forward")
            # Standardized history, earliest first. The 12th auxiliary step is not a key.
            history = temporal_aux[:, :-1].permute(0, 2, 1, 3).contiguous()
            raw = base + self.history_fusion(hidden[:, -1], history)
        elif self.dual_history is not None:
            if temporal_aux is None:
                raise RuntimeError("dual path history TSF requires temporal_aux_pred from this forward")
            branches = self.dual_history(hidden, temporal_aux, base)
            raw = branches["raw_pred"]
        if self.projection is None:
            outputs = {"raw_pred": raw, "final_pred": raw}
        else:
            outputs = self.projection(hidden, raw)
        if temporal_aux is not None:
            outputs["temporal_aux_pred"] = temporal_aux
        if branches is not None:
            outputs["branch_a_pred"] = branches["branch_a_pred"]
            outputs["branch_b_pred"] = branches["branch_b_pred"]
            outputs["branch_gate"] = branches["branch_gate"]
        return outputs

    def _embed_shared(self, values, time_of_day=None, day_of_week=None) -> torch.Tensor:
        """One embedding call: value MLP, road, TOD, DOW, then adaptive."""
        hidden = self.embedding(values, time_of_day, day_of_week)
        if self.adaptive is not None:
            hidden = self.adaptive(hidden)
        return hidden

    def _forward_two_stage(self, x, time_of_day=None, day_of_week=None) -> dict[str, torch.Tensor]:
        """Rebuild 12 projected steps, then correct only the last step."""
        if self.temporal_aux_head is None or self.stage1_projection is None or self.projection is None:
            raise RuntimeError("two-stage forward requires the step head and both projections")
        if self.stage2_block is None:
            raise RuntimeError("two-stage forward requires an independent second TSF")
        hidden = self._encode(x, time_of_day, day_of_week)
        raw_steps = self.temporal_aux_head(hidden)
        stage1 = self.stage1_projection.project_aligned(hidden, raw_steps)
        projected = stage1["final_pred"]
        second = self.stage2_block(self._embed_shared(projected, time_of_day, day_of_week))
        raw2 = projected[:, -1] + self.head(second)
        outputs = self.projection(second, raw2)
        outputs["temporal_aux_pred"] = projected
        return outputs

    def _encode(self, x, time_of_day=None, day_of_week=None) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must be [B, T, N, C_in], got {tuple(x.shape)}")
        if int(x.shape[1]) != self.config.input_window:
            raise ValueError(f"x has T={x.shape[1]}, model input_window={self.config.input_window}")
        if self.config.backbone_mode == "state_independent":
            if int(x.shape[-1]) != STATE_BRANCH_COUNT:
                raise ValueError("state_independent expects four state channels")
            if self.config.state_context_sharing == "shared":
                context = self.shared_state_context(x, time_of_day, day_of_week)
                hidden_parts = [
                    branch(x[..., channel:channel + 1], time_of_day, day_of_week, context=context)
                    for channel, branch in enumerate(self.branches)
                ]
            else:
                hidden_parts = [
                    branch(x[..., channel:channel + 1], time_of_day, day_of_week)
                    for channel, branch in enumerate(self.branches)
                ]
            hidden = torch.cat(hidden_parts, dim=-1)
            if int(hidden.shape[-1]) != int(self.config.d_model):
                raise RuntimeError(
                    f"concatenated state branches have width {hidden.shape[-1]}, expected {self.config.d_model}"
                )
            return hidden
        hidden = self._embed_shared(x, time_of_day, day_of_week)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden

    def parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.parameters() if param.requires_grad))

    def architecture_summary(self) -> dict[str, Any]:
        fusion_ids = set()
        if isinstance(self.head, StateTokenTransformerHead):
            fusion_ids = {id(module) for module in self.head.modules() if isinstance(module, SwiGLU)}
        history_ffn_ids = set()
        if self.dual_history is not None:
            history_ffn_ids = {id(module) for module in self.dual_history.modules() if isinstance(module, SwiGLU)}
        stage2_ffn_ids = set()
        if self.stage2_block is not None:
            stage2_ffn_ids = {id(module) for module in self.stage2_block.modules() if isinstance(module, SwiGLU)}
        excluded_ffn_ids = fusion_ids | history_ffn_ids | stage2_ffn_ids
        backbone_ffns = [
            module for module in self.modules() if isinstance(module, SwiGLU) and id(module) not in excluded_ffn_ids
        ]
        fusion_ffns = [module for module in self.modules() if isinstance(module, SwiGLU) and id(module) in fusion_ids]
        layout = str(self.config.attention_layout)
        summary = {
            "backbone_mode": self.config.backbone_mode,
            "embedding": self.embedding_summary(),
            "head": self.head.describe(),
            "attention_layout": layout,
            "d_model": int(self.config.d_model),
            "temporal_causal_mask": False,
            "independent_ffn_count": len(backbone_ffns),
            "backbone_ffn_count": len(backbone_ffns),
            "fusion_ffn_count": len(fusion_ffns),
            "ffn_hidden_dims": [int(ffn.w_gate.out_features) for ffn in backbone_ffns],
            "fusion_ffn_hidden_dims": [int(ffn.w_gate.out_features) for ffn in fusion_ffns],
            "total_parameters": self.parameter_count(),
        }
        if isinstance(self.head, StateTokenTransformerHead):
            summary.update(
                branch_d_model=int(self.config.d_model) // STATE_BRANCH_COUNT,
                fusion_token_dim=int(self.head.token_dim),
                fusion_tokens=int(self.head.num_tokens),
                fusion_num_heads=int(self.head.num_heads),
                fusion_head_dim=int(self.head.head_dim),
                projection_condition_dim=int(self.config.d_model),
            )
        if self.projection is not None:
            first, last = self.projection.mlp[0], self.projection.mlp[-1]
            summary["projection_mlp"] = {
                "input_dim": int(first.in_features),
                "hidden_dim": int(first.out_features),
                "output_dim": int(last.out_features),
            }
        if self.temporal_aux_head is None:
            summary["temporal_aux_head"] = {"enabled": False}
        else:
            layer = self.temporal_aux_head
            supervised = "0..11" if self.stage2_block is not None else "0..10"
            summary["temporal_aux_head"] = {
                "enabled": True,
                "in_features": int(layer.in_features),
                "out_features": int(layer.out_features),
                "shared_across_time_and_roads": True,
                "parameters": int(layer.weight.numel() + layer.bias.numel()),
                "supervised_steps": supervised,
                "supervised_values": "projected_12step_target_zscore" if self.stage2_block is not None else "auxiliary_head",
            }
        if self.dual_history is not None:
            summary["history_fusion"] = {
                "enabled": False,
                "mode": "dual_path_history_tsf",
                "parameters": 0,
                "replaced_by": "history_tsf",
            }
            history = self.dual_history.describe()
            history["branch_a_head"] = self.head.describe()
            summary["history_tsf"] = history
        elif self.history_fusion is None:
            summary["history_fusion"] = {"enabled": False, "mode": "none", "parameters": 0}
        else:
            summary["history_fusion"] = self.history_fusion.describe()
        if self.config.backbone_mode == "state_independent":
            branches = [branch.summary() for branch in self.branches]
            embedding = summary["embedding"]
            summary.update(
                branch_count=len(self.branches),
                branch_d_model=int(self.config.d_model) // STATE_BRANCH_COUNT,
                state_context_sharing=embedding["state_context_sharing"],
                state_node_sharing=embedding["state_node_sharing"],
                state_channels=list(embedding["state_channels"]),
                context_tables=embedding["context_tables"],
                node_tables=embedding["node_tables"],
                joint_attention_count=0,
                temporal_attention_count=len(self.branches),
                spatial_attention_count=len(self.branches),
                per_branch=branches,
                block_order=["T", "S", "FFN"],
            )
            return summary
        if layout == "joint_st":
            joint = self.blocks[0].joint
            summary.update(
                branch_count=1,
                joint_attention_count=1,
                temporal_attention_count=0,
                spatial_attention_count=0,
                joint_num_heads=int(joint.num_heads),
                joint_head_dim=int(joint.head_dim),
                attention_inner_dim=int(joint.inner_dim),
                joint_temporal_window=int(joint.window),
                joint_temporal_alignment=str(joint.alignment),
                joint_boundary="crop_valid_times_no_token_repeat",
                joint_spatial_scope="all_nodes_including_self",
                block_order=["JointST", "FFN"],
            )
            return summary
        summary.update(
            branch_count=1,
            joint_attention_count=0,
            temporal_attention_count=1,
            spatial_attention_count=1,
            temporal_num_heads=int(self.blocks[0].temporal.num_heads),
            spatial_num_heads=int(self.blocks[0].spatial.num_heads),
            temporal_head_dim=int(self.blocks[0].temporal.head_dim),
            spatial_head_dim=int(self.blocks[0].spatial.head_dim),
            attention_inner_dim=int(self.blocks[0].temporal.inner_dim),
            block_order=["T", "S", "FFN"],
        )
        if self.stage2_block is not None and self.stage1_projection is not None:
            stage2_ffns = [module for module in self.stage2_block.modules() if isinstance(module, SwiGLU)]
            first, last = self.stage1_projection.mlp[0], self.stage1_projection.mlp[-1]
            summary.update(
                temporal_attention_count=2,
                spatial_attention_count=2,
                stage2_ffn_count=len(stage2_ffns),
                stage2_ffn_hidden_dims=[int(ffn.w_gate.out_features) for ffn in stage2_ffns],
                readout=(
                    "stage1 linear head on every step, stage1 conservation projection, "
                    "stage2 TSF on the projected values, zero-initialized last-step residual "
                    "added to the projected last step, then the stage2 conservation projection"
                ),
            )
            summary["two_stage"] = {
                "mode": "two_stage_projected_tsf",
                "stage1_input": "partial_observation",
                "stage2_input": "projected_12step_target_zscore",
                "shared_embedding": True,
                "shared_value_mlps": True,
                "stage1_blocks": len(self.blocks),
                "stage2_blocks": 1,
                "backbone_ffn_count": summary["backbone_ffn_count"],
                "stage2_ffn_count": len(stage2_ffns),
                "stage1_projection_mlp": {
                    "input_dim": int(first.in_features),
                    "hidden_dim": int(first.out_features),
                    "output_dim": int(last.out_features),
                },
                "stage2_projection_mlp": summary.get("projection_mlp"),
                "residual_head": self.head.describe(),
                "residual_head_input_width": int(self.head.proj.in_features),
                "aux_supervised_steps": "0..11",
                "total_parameters": summary["total_parameters"],
            }
        return summary

    def spatial_summary(self) -> dict[str, Any]:
        """Which attention mixes roads, and what restricts its keys."""
        branch_count = STATE_BRANCH_COUNT if self.config.backbone_mode == "state_independent" else 1
        if str(self.config.attention_layout) == "joint_st":
            joint = self.blocks[0].joint
            return {
                "type": "joint_st",
                "branch_count": branch_count,
                "graph_mask": False,
                "keys_per_query": "N times the valid centered window",
                "time_window": int(joint.window),
                "alignment": str(joint.alignment),
                "boundary": "crop_valid_times_no_token_repeat",
                "spatial_scope": "all_nodes_including_self",
                "includes_self": True,
                "causal_mask": False,
                "query_chunk": int(joint.query_chunk),
                "checkpoint": bool(joint.use_checkpoint),
                "stored_spatial_attention_used": False,
            }
        name = self.config.spatial_attention
        if name == "global":
            summary = {
                "type": "global",
                "branch_count": branch_count,
                "keys_per_query": int(self.config.num_nodes),
                "graph_mask": False,
                "edge_type_bias": "not_applicable",
                "spatial_direction": "not_applicable",
                "chunk_size": self.config.spatial_chunk_size,
                "checkpoint": bool(self.config.spatial_checkpoint),
            }
        else:
            summary = {
                "type": name,
                "branch_count": branch_count,
                "graph_mask": True,
                "graph_edges": self.graph_edges,
                "edge_type_bias": bool(self.config.use_edge_type_bias),
                "spatial_direction": self.graph_direction,
                "chunk_size": self.config.spatial_chunk_size,
                "checkpoint": bool(self.config.spatial_checkpoint),
            }
        if branch_count > 1:
            summary["independent_copies"] = (
                f"{branch_count} independent {name} spatial attentions, one per state channel"
            )
        if self.stage2_block is not None:
            summary["stage_copies"] = 2
            summary["stage2"] = "independent second TSF; same spatial attention settings, separate parameters"
        return summary

    def _context_table_summary(self) -> dict[str, Any]:
        """Owned TOD/DOW/adaptive tables. A shared table is counted once."""
        sharing = str(self.config.state_context_sharing)
        probe = self.branches[0]
        widths = probe.embedding.widths
        adaptive_shape = [int(size) for size in probe.context_adaptive_shape]
        per_branch = {
            "value": int(widths["value"]),
            "node": int(widths["node"]),
            "tod": int(widths["tod"]),
            "dow": int(widths["dow"]),
            "base": int(probe.embedding.d_model),
            "adaptive": int(adaptive_shape[-1]),
            "d_model": int(probe.block.temporal.d_model),
        }
        if per_branch["base"] + per_branch["adaptive"] != per_branch["d_model"]:
            raise RuntimeError(
                f"branch base {per_branch['base']} + adaptive {per_branch['adaptive']} "
                f"!= d_model {per_branch['d_model']}"
            )
        tod_shape = [int(self.config.steps_per_day), per_branch["tod"]]
        dow_shape = [7, per_branch["dow"]]
        per_copy = tod_shape[0] * tod_shape[1] + dow_shape[0] * dow_shape[1]
        per_copy += adaptive_shape[0] * adaptive_shape[1] * adaptive_shape[2]
        if sharing == "shared":
            copies = 1
            tables = [
                self.shared_state_context.tod_embedding.weight,
                self.shared_state_context.dow_embedding.weight,
                self.shared_state_context.adaptive.adaptive_embedding,
            ]
        else:
            copies = STATE_BRANCH_COUNT
            tables = []
            for branch in self.branches:
                tables.extend((
                    branch.embedding.tod_embedding.weight,
                    branch.embedding.dow_embedding.weight,
                    branch.adaptive.adaptive_embedding,
                ))
        pointers = [table.data_ptr() for table in tables]
        if len(pointers) != len(set(pointers)):
            raise RuntimeError("context tables do not have distinct storage")
        owned = int(sum(table.numel() for table in tables))
        if owned != per_copy * copies:
            raise RuntimeError(f"context parameter count {owned} != {per_copy} x {copies}")
        return {
            "sharing": sharing,
            "copies": copies,
            "tod_shape": tod_shape,
            "dow_shape": dow_shape,
            "adaptive_shape": adaptive_shape,
            "per_branch_feature_width": per_branch,
            "concat_feature_width": {
                "tod": per_branch["tod"] * STATE_BRANCH_COUNT,
                "dow": per_branch["dow"] * STATE_BRANCH_COUNT,
                "adaptive": per_branch["adaptive"] * STATE_BRANCH_COUNT,
                "d_model": per_branch["d_model"] * STATE_BRANCH_COUNT,
            },
            "parameters_per_copy": int(per_copy),
            "parameters_total": owned,
        }

    def _node_table_summary(self) -> dict[str, Any]:
        """Owned road tables. A shared table is counted once and read at full width."""
        sharing = str(self.config.state_node_sharing)
        width = int(self.branches[0].embedding.widths["node"])
        shape = [int(self.config.num_nodes), width]
        per_copy = shape[0] * shape[1]
        if any(int(branch.embedding.widths["node"]) != width for branch in self.branches):
            raise RuntimeError("node widths differ across state branches")
        if sharing == "shared":
            table = self.shared_state_context.node_embedding
            if table is None:
                raise RuntimeError("shared node mode has no parent node table")
            if any(branch.embedding.node_embedding is not None for branch in self.branches):
                raise RuntimeError("shared node mode allocated a private node table")
            if not all(branch.embedding.external_node_embedding for branch in self.branches):
                raise RuntimeError("shared node mode left a branch without an external node")
            tables = [table.weight]
            copies = 1
            source = "shared_parameter"
        else:
            tables = []
            for branch in self.branches:
                if branch.embedding.external_node_embedding or branch.embedding.node_embedding is None:
                    raise RuntimeError("independent node mode is missing a private node table")
                tables.append(branch.embedding.node_embedding.weight)
            copies = STATE_BRANCH_COUNT
            source = "parameter"
        pointers = [item.data_ptr() for item in tables]
        if len(pointers) != len(set(pointers)):
            raise RuntimeError("node tables do not have distinct storage")
        owned = int(sum(item.numel() for item in tables))
        if owned != per_copy * copies:
            raise RuntimeError(f"node parameter count {owned} != {per_copy} x {copies}")
        if any(tuple(item.shape) != tuple(shape) for item in tables):
            raise RuntimeError(f"node table shape is not {shape}")
        return {
            "enabled": True,
            "source": source,
            "shared": sharing == "shared",
            "sharing": sharing,
            "copies": copies,
            "shape": shape,
            "parameters_per_copy": int(per_copy),
            "parameters_total": owned,
        }

    def embedding_summary(self) -> dict[str, Any]:
        """Fusion mode and the widths actually built (``add`` uses ``d_model`` for every part)."""
        if self.config.backbone_mode == "state_independent":
            from MSTT.config import STATE_CHANNELS

            context = self._context_table_summary()
            node = self._node_table_summary()
            return {
                "backbone_mode": "state_independent",
                "state_context_sharing": context["sharing"],
                "state_node_sharing": node["sharing"],
                "state_channels": list(STATE_CHANNELS),
                "branch_count": STATE_BRANCH_COUNT,
                "parameter_sharing": {
                    "value_mlp": "independent",
                    "node_embedding": node["sharing"],
                    "tod": context["sharing"],
                    "dow": context["sharing"],
                    "adaptive": context["sharing"],
                    "tsf": "independent",
                },
                "concatenation": "channel axis after each branch TSF",
                "output_dim": int(self.config.d_model),
                "per_branch_feature_width": context["per_branch_feature_width"],
                "context_tables": context,
                "node_tables": node,
                "branches": [branch.summary() for branch in self.branches],
            }
        embedding = self.embedding
        return {
            "backbone_mode": "shared",
            "fusion": embedding.fusion,
            "widths": dict(embedding.widths),
            "value_embedding": embedding.value_embedding_summary(),
            "enabled": {
                "value": True,
                "node": embedding.node_embedding is not None,
                "tod": embedding.tod_embedding is not None,
                "dow": embedding.dow_embedding is not None,
            },
            "output_dim": int(self.config.d_model),
            "base_output_dim": embedding.d_model,
            "adaptive": {
                "enabled": self.adaptive is not None,
                "fusion": f"{self.config.adaptive_embedding_fusion}_after_embedding_dropout",
                "shape": None if self.adaptive is None else list(self.adaptive.adaptive_embedding.shape),
                "parameters": 0 if self.adaptive is None else int(self.adaptive.adaptive_embedding.numel()),
            },
        }
