"""Tasks, penetration rates, default configuration, and CLI overlays for MSTT.

Paths are resolved through ``data_root.py``. The four-state windows exist
only in the 3-second-phase package, so that package and its sibling results
root are the defaults, the same place the 9-task baselines read and write.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from data_root import xuancheng26_model_results_3s_dir, xuancheng26_training_3s_root, xuancheng26_sampling_root, sampling_window_root

STATE_CHANNELS = ("inflow", "outflow", "accumulation_start", "accumulation_end")
# Input-only extension. It is not a prediction target and not a baseline state type.
ALL_SP_C_CHANNELS = STATE_CHANNELS + ("average_speed_raw", "speed_observation_count")
RATES = ("p05", "p10", "p20", "p30", "p40", "p50", "p70")


def summary_rate_sort_key(rate: str) -> tuple[int, int, str]:
    """Order summary rows without treating ``RATES`` as the set of legal keys.

    The seven fixed penetrations stay in their declared order. An observation
    key such as ``p10_p20`` is accepted by ``rate_tag`` from the dataset
    manifest and sorts by its text after those seven.
    """
    text = str(rate)
    if text in RATES:
        return (0, RATES.index(text), "")
    return (1, len(RATES), text)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    input_type: str
    target_type: str

    @property
    def input_channels(self) -> tuple[str, ...]:
        if self.input_type == "all":
            return STATE_CHANNELS
        if self.input_type == "all_sp_c":
            return ALL_SP_C_CHANNELS
        return (self.input_type,)

    @property
    def output_channels(self) -> tuple[str, ...]:
        return STATE_CHANNELS if self.target_type == "all" else (self.target_type,)

    @property
    def in_channels(self) -> int:
        return len(self.input_channels)

    @property
    def out_channels(self) -> int:
        return len(self.output_channels)

    @property
    def baseline_leaf(self) -> str:
        """Directory name the baselines use for the same input/target pair."""
        return f"in_{self.input_type}__out_{self.target_type}"


TASKS: dict[str, TaskSpec] = {
    spec.name: spec
    for spec in (
        TaskSpec("I2I", "inflow", "inflow"),
        TaskSpec("O2O", "outflow", "outflow"),
        TaskSpec("AS2AS", "accumulation_start", "accumulation_start"),
        TaskSpec("AE2AE", "accumulation_end", "accumulation_end"),
        TaskSpec("All2All", "all", "all"),
        TaskSpec("All2I", "all", "inflow"),
        TaskSpec("All2O", "all", "outflow"),
        TaskSpec("All2AS", "all", "accumulation_start"),
        TaskSpec("All2AE", "all", "accumulation_end"),
        TaskSpec("AllSpC2All", "all_sp_c", "all"),
        TaskSpec("AllSpC2I", "all_sp_c", "inflow"),
    )
}


def task_spec(name: str) -> TaskSpec:
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; choose from {list(TASKS)}")
    return TASKS[name]


def rate_tag(rate: str | int, *, allowed: tuple[str, ...] | list[str] | None = None) -> str:
    """Delegate to the shared checker so ``p10_p20`` is not parsed as an integer."""
    from reimplementation.common.data.r_only_npz_dataset import rate_tag as shared_rate_tag

    return shared_rate_tag(rate, allowed=allowed)


MODEL_DEFAULTS: dict[str, Any] = {
    "d_model": 64,
    "num_heads": 4,
    "temporal_num_heads": None,
    "spatial_num_heads": None,
    "attention_head_dim": None,
    "num_layers": 2,
    "input_window": 12,
    "ffn_hidden_dim": None,
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "embedding_dropout": 0.1,
    "use_qk_norm": True,
    "use_rope": True,
    "rope_base": 10000.0,
    "use_gated_attention": False,
    "spatial_attention": "sparse",
    "spatial_direction": "bidirectional",
    "use_edge_type_bias": True,
    "spatial_chunk_size": 64,
    "spatial_checkpoint": True,
    "use_tod": True,
    "use_dow": True,
    "use_node_embedding": True,
    "embedding_fusion": "concat",
    "value_embedding_mode": "joint_linear",
    "value_embedding_dim": 32,
    "node_embedding_dim": 16,
    "tod_embedding_dim": 8,
    "dow_embedding_dim": 8,
    "use_adaptive_embedding": False,
    "base_embedding_dim": None,
    "adaptive_embedding_dim": None,
    "adaptive_embedding_fusion": "add",
    "ffn_placement": "after_spatial",
    "attention_order": "TS",
    "attention_layout": "factorized_ts",
    "joint_temporal_window": 3,
    "joint_temporal_alignment": "centered",
    "joint_query_chunk": 256,
    "joint_checkpoint": True,
    "steps_per_day": 288,
    "norm_eps": 1e-6,
    "head_type": "flatten",
    "conservation_mode": "none",
    "projection_mlp_hidden_dim": 32,
    "backbone_mode": "shared",
    "state_context_sharing": "independent",
    # A missing road-table switch keeps one private node embedding on each branch.
    "state_node_sharing": "independent",
    "fusion_mlp_hidden_dim": 128,
    # Inactive unless head_type is state_token_transformer.
    "fusion_token_dim": 128,
    "fusion_num_heads": 4,
    "fusion_ffn_hidden_dim": 176,
    "fusion_chunk_size": None,
    "fusion_checkpoint": False,
    "temporal_aux_enabled": False,
    # ``none`` keeps the auxiliary head out of the last-step prediction.
    "history_fusion_mode": "none",
}

# Meaning of keys that older run directories never stored: v0.1 predates
# ``embedding_fusion`` (it summed), v0.1/v0.2 predate the adaptive embedding.
# A missing ``value_embedding_mode`` is the original joint Linear, including
# inside a nested ``model_config``. Fill these in before rebuilding an old
# model from its resolved_config.json.
LEGACY_DEFAULTS: dict[str, Any] = {
    "attention_order": "TS",
    "attention_layout": "factorized_ts",
    "joint_temporal_window": 3,
    "joint_temporal_alignment": "centered",
    "joint_query_chunk": 256,
    "joint_checkpoint": True,
    "temporal_num_heads": None, "spatial_num_heads": None,
    "attention_head_dim": None,
    "base_embedding_dim": None, "adaptive_embedding_dim": None,
    "adaptive_embedding_fusion": "add", "ffn_placement": "after_spatial",
    "embedding_fusion": "add", "use_adaptive_embedding": False,
    "value_embedding_mode": "joint_linear",
    "conservation_mode": "none", "projection_mlp_hidden_dim": 32,
    "backbone_mode": "shared",
    # Runs saved before this field keep a private TOD/DOW/adaptive table on each branch.
    "state_context_sharing": "independent",
    # Runs saved before this field keep a private road table on each branch.
    "state_node_sharing": "independent",
    "fusion_mlp_hidden_dim": 128,
    # Runs saved before the 48-token head did not use these. Old heads ignore them.
    "fusion_token_dim": 128,
    "fusion_num_heads": 4,
    "fusion_ffn_hidden_dim": 176,
    "fusion_chunk_size": None,
    "fusion_checkpoint": False,
    # Checkpoints written before the temporal auxiliary head did not have one.
    "temporal_aux_enabled": False,
    # Checkpoints written before history fusion did not correct the last step with it.
    "history_fusion_mode": "none",
}

TRAIN_DEFAULTS: dict[str, Any] = {
    "conservation_lambda": 0.1,
    "projection_raw_loss_weight": 0.5,
    "projection_consistency_weight": 0.0,
    # "all" is original C. "inflow_only" supervises projected inflow plus the raw residual.
    "projection_supervision": "all",
    # Inactive unless temporal_aux_enabled. The first experiment uses the first 11 steps at weight 0.5.
    "temporal_aux_weight": 0.5,
    "temporal_aux_scope": "history_except_last",
    # Inactive unless history_fusion_mode is dual_path_history_tsf. Old runs omit them.
    "branch_a_loss_weight": 0.25,
    "branch_b_loss_weight": 0.25,
    "seed": 42,
    "device": "auto",
    "batch_size": 16,
    "eval_batch_size": 32,
    "epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "lr_scheduler": "cosine",
    "lr_warmup_epochs": 5,
    "lr_warmup_init": 1e-6,
    "lr_min": 1e-6,
    "loss": "mae_raw",
    "huber_delta": 1.0,
    "channel_balance": "none",
    "selection_metric": "val_loss",
    "early_stop_patience": 20,
    "clip_grad_norm": 5.0,
    "amp": "none",
    "allow_tf32": False,
    "num_workers": 0,
    "save_test_predictions": True,
    "save_validation_predictions": False,
    "max_train_batches": None,
    "max_eval_batches": None,
}

DATA_DEFAULTS: dict[str, Any] = {
    # baseline plus a conservation mode is stored as conservation_<mode>.
    # Any other id is kept, so a new run does not share that directory.
    "experiment_id": "baseline",
    "dataset_id": "fixed_uniform",
    "sampling_root": str(xuancheng26_sampling_root()),
    "model_seed": 42,
    "train_sampling_seed": 42,
    "evaluation_sampling_seeds": [42, 43, 44, 45, 46],
    "data_root": sampling_window_root(xuancheng26_sampling_root()).as_posix(),
    "graph_path": (xuancheng26_training_3s_root() / "graph" / "r_graph.npz").as_posix(),
    "r_nodes": (xuancheng26_training_3s_root() / "graph" / "r_nodes.csv").as_posix(),
    "output_root": xuancheng26_model_results_3s_dir("MSTT").as_posix(),
    "num_nodes": 1406,
    "target_mode": "last-observed-step",
}


# Frozen from the completed best experiment, including its nonstandard FFN width.
_FROZEN = json.loads((Path(__file__).parent / "configs/default.json").read_text(encoding="utf-8"))
MODEL_DEFAULTS.update({k: v for k, v in _FROZEN.items() if k in MODEL_DEFAULTS})
TRAIN_DEFAULTS.update({k: v for k, v in _FROZEN.items() if k in TRAIN_DEFAULTS})

def default_config() -> dict[str, Any]:
    return copy.deepcopy({**DATA_DEFAULTS, **MODEL_DEFAULTS, **TRAIN_DEFAULTS})


def load_config(path: Path | None) -> dict[str, Any]:
    config = default_config()
    if path is not None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = sorted(set(payload) - set(config))
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {unknown}")
        config.update(payload)
    return config


def _parse_bool(text: str) -> bool:
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {text!r}")


def _parse_optional_int(text: str) -> int | None:
    return None if str(text).lower() in {"none", "null", ""} else int(text)


_OPTIONAL_INT_KEYS = {"ffn_hidden_dim", "max_train_batches", "max_eval_batches", "early_stop_patience",
                      "base_embedding_dim", "adaptive_embedding_dim",
                      "temporal_num_heads", "spatial_num_heads", "attention_head_dim",
                      "fusion_chunk_size"}
_CHOICES = {
    "attention_order": ("TS",),
    "attention_layout": ("factorized_ts", "joint_st"),
    "joint_temporal_alignment": ("centered",),
    "adaptive_embedding_fusion": ("add", "concat"),
    "ffn_placement": ("after_spatial",),
    "conservation_mode": ("none", "soft", "projected"),
    "projection_supervision": ("all", "inflow_only"),
    "temporal_aux_scope": ("history_except_last", "all_steps"),
    "history_fusion_mode": ("none", "gated_cross_attention", "dual_path_history_tsf", "two_stage_projected_tsf"),
    "spatial_attention": ("sparse", "global"),
    "spatial_direction": ("forward", "backward", "bidirectional"),
    "head_type": ("flatten", "last", "flatten_mlp", "state_token_transformer"),
    "backbone_mode": ("shared", "state_independent"),
    "state_context_sharing": ("independent", "shared"),
    "state_node_sharing": ("independent", "shared"),
    "embedding_fusion": ("add", "concat"),
    "value_embedding_mode": ("joint_linear", "separate_mlp", "state_aux_gate"),
    "loss": ("mae_raw", "huber_raw", "mse_raw"),
    "channel_balance": ("none", "target_std"),
    "selection_metric": ("mean_validation_mae",),
    "amp": ("none", "bf16", "fp16"),
    "lr_scheduler": ("cosine", "none"),
}


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """One flag per config key, e.g. ``--d-model`` (``--d_model`` also works)."""
    for key, value in default_config().items():
        flags = [f"--{key.replace('_', '-')}"]
        if "_" in key:
            flags.append(f"--{key}")
        kwargs: dict[str, Any] = {"dest": key, "default": argparse.SUPPRESS}
        if key in _OPTIONAL_INT_KEYS:
            kwargs["type"] = _parse_optional_int
        elif isinstance(value, list):
            kwargs["type"] = int
            kwargs["nargs"] = "+"
        elif isinstance(value, bool):
            kwargs["type"] = _parse_bool
        elif isinstance(value, int):
            kwargs["type"] = int
        elif isinstance(value, float):
            kwargs["type"] = float
        else:
            kwargs["type"] = str
        if key in _CHOICES:
            kwargs["choices"] = _CHOICES[key]
        parser.add_argument(*flags, **kwargs)


def overlay_args(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    passed = [key for key in default_config() if hasattr(args, key)]
    for key in passed:
        config[key] = getattr(args, key)
    config["explicit_path_keys"] = [key for key in passed if key in {"dataset_id", "sampling_root"}]
    if hasattr(args, "seed") and not hasattr(args, "model_seed"):
        config["model_seed"] = args.seed
    from reimplementation.common.data.sampling_seeds import configure_sampling
    return configure_sampling(config, model_name="MSTT")


def resolve_ffn_hidden_dim(d_model: int, value: int | None) -> int:
    """SwiGLU default: 8/3 * d_model rounded up to a multiple of 8 (FFN params ~ 4d ReLU)."""
    if value is not None:
        return int(value)
    hidden = int(round(8 * int(d_model) / 3))
    return ((hidden + 7) // 8) * 8


def validate_config(config: Mapping[str, Any]) -> None:
    if int(config["num_layers"]) != 1:
        raise ValueError("MSTT has exactly one block")
    layout = str(config.get("attention_layout", "factorized_ts"))
    if layout == "factorized_ts":
        if config["attention_order"] != "TS" or config["ffn_placement"] != "after_spatial":
            raise ValueError("factorized_ts requires attention_order=TS and ffn_placement=after_spatial")
    elif layout == "joint_st":
        window = int(config["joint_temporal_window"])
        if window < 1 or window % 2 == 0:
            raise ValueError("joint_temporal_window must be a positive odd integer")
        if int(config["joint_query_chunk"]) < 1:
            raise ValueError("joint_query_chunk must be positive")
        if str(config["joint_temporal_alignment"]) != "centered":
            raise ValueError("joint_temporal_alignment must be centered")
    for key, choices in _CHOICES.items():
        if str(config.get(key, MODEL_DEFAULTS.get(key))) not in choices:
            raise ValueError(f"{key}={config[key]!r} not in {choices}")
    for key in ("conservation_lambda", "projection_raw_loss_weight", "projection_consistency_weight"):
        value = float(config.get(key, TRAIN_DEFAULTS[key]))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and non-negative")
    if int(config.get("projection_mlp_hidden_dim", 32)) < 1:
        raise ValueError("projection_mlp_hidden_dim must be positive")
    if int(config.get("fusion_mlp_hidden_dim", MODEL_DEFAULTS["fusion_mlp_hidden_dim"])) < 1:
        raise ValueError("fusion_mlp_hidden_dim must be positive")
    if config.get("conservation_mode", "none") != "none":
        if config["loss"] != "mae_raw" or config["channel_balance"] != "none":
            raise ValueError("soft/projected requires unbalanced raw-scale MAE (mae_raw, channel_balance=none)")
        if config["selection_metric"] not in ("mean_validation_mae",):
            raise ValueError("soft/projected selects mean_validation_mae")
        if config["embedding_fusion"] != "concat" or int(config["input_window"]) != 12:
            raise ValueError("soft/projected requires concat embeddings and a 12-step window")
        if config["spatial_attention"] not in ("sparse", "global"):
            raise ValueError("soft/projected spatial attention must be sparse or global")
        if config["use_adaptive_embedding"] and config.get("adaptive_embedding_fusion") != "concat":
            raise ValueError("soft/projected adaptive embedding must be concatenated, not added")
    supervision = str(config.get("projection_supervision", "all"))
    if supervision == "inflow_only":
        if config.get("conservation_mode", "none") != "projected":
            raise ValueError("projection_supervision=inflow_only requires conservation_mode=projected")
        raw_weight = float(config.get("projection_raw_loss_weight", TRAIN_DEFAULTS["projection_raw_loss_weight"]))
        if raw_weight != 0.0:
            raise ValueError(
                "projection_supervision=inflow_only does not add raw MAE; set projection_raw_loss_weight to 0"
            )
    aux_weight = float(config.get("temporal_aux_weight", TRAIN_DEFAULTS["temporal_aux_weight"]))
    if not math.isfinite(aux_weight) or aux_weight < 0:
        raise ValueError("temporal_aux_weight must be finite and non-negative")
    aux_scope = str(config.get("temporal_aux_scope", TRAIN_DEFAULTS["temporal_aux_scope"]))
    if aux_scope not in _CHOICES["temporal_aux_scope"]:
        raise ValueError(f"temporal_aux_scope={aux_scope!r} not in {_CHOICES['temporal_aux_scope']}")
    if bool(config.get("temporal_aux_enabled", False)) and int(config["input_window"]) != 12:
        raise ValueError("temporal_aux_enabled requires input_window=12")
    from MSTT.model.mstt import validate_history_fusion

    validate_history_fusion(config)
    validate_two_stage_training(config)
    if str(config.get("history_fusion_mode", "none")) == "dual_path_history_tsf":
        if float(config.get("projection_raw_loss_weight", TRAIN_DEFAULTS["projection_raw_loss_weight"])) != 0.0:
            raise ValueError("history_fusion_mode='dual_path_history_tsf' requires projection_raw_loss_weight=0")
        if float(config.get("projection_consistency_weight", TRAIN_DEFAULTS["projection_consistency_weight"])) != 0.0:
            raise ValueError("history_fusion_mode='dual_path_history_tsf' requires projection_consistency_weight=0")
        for key in ("branch_a_loss_weight", "branch_b_loss_weight"):
            value = float(config.get(key, TRAIN_DEFAULTS[key]))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and non-negative")
    if str(config["target_mode"]) != "last-observed-step":
        raise ValueError("MSTT reconstructs the last observed slot; target_mode must be last-observed-step")
    backbone = str(config.get("backbone_mode", "shared"))
    from MSTT.model.mstt import (
        validate_independent_backbone,
        validate_state_context_sharing,
        validate_state_node_sharing,
        validate_state_token_transformer,
    )

    validate_state_context_sharing(config)
    validate_state_node_sharing(config)
    validate_state_token_transformer(config)
    if backbone == "state_independent":
        validate_independent_backbone(config)
    else:
        if int(config["d_model"]) % int(config["num_heads"]) != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if bool(config["use_rope"]) and (int(config["d_model"]) // int(config["num_heads"])) % 2 != 0:
            raise ValueError("RoPE needs an even head dimension")
        for key in ("temporal_num_heads", "spatial_num_heads"):
            heads = attention_head_count(config, key)
            if heads < 1 or int(config["d_model"]) % heads != 0:
                raise ValueError(f"{key}={heads} must be a positive divisor of d_model={config['d_model']}")
        temporal_heads = attention_head_count(config, "temporal_num_heads")
        head_dim = config.get("attention_head_dim")
        if head_dim is not None and int(head_dim) < 1:
            raise ValueError("attention_head_dim must be positive")
        resolved_head_dim = int(config["d_model"]) // temporal_heads if head_dim is None else int(head_dim)
        if bool(config["use_rope"]) and resolved_head_dim % 2 != 0:
            raise ValueError("RoPE needs an even temporal head dimension")
    if int(config["input_window"]) < 1 or int(config["num_layers"]) < 1:
        raise ValueError("input_window and num_layers must be positive")
    if str(config["spatial_attention"]) == "global" and bool(config["use_edge_type_bias"]):
        raise ValueError("use_edge_type_bias does not apply to spatial_attention=global; pass --use-edge-type-bias 0")
    value_mode = str(config.get("value_embedding_mode", "joint_linear"))
    if value_mode == "separate_mlp":
        if str(config.get("embedding_fusion", "add")) != "concat":
            raise ValueError("value_embedding_mode=separate_mlp requires embedding_fusion=concat")
        value_dim = int(config.get("value_embedding_dim", 0))
        if value_dim < 1:
            raise ValueError("value_embedding_mode=separate_mlp requires a positive value_embedding_dim")
    elif value_mode == "state_aux_gate":
        if str(config.get("embedding_fusion", "add")) != "concat":
            raise ValueError("value_embedding_mode=state_aux_gate requires embedding_fusion=concat")
        if int(config.get("value_embedding_dim", 0)) != 32:
            raise ValueError("value_embedding_mode=state_aux_gate requires value_embedding_dim=32")
    from MSTT.model.embedding import resolve_embedding_dims

    base_dim, _ = resolve_embedding_dims(
        config["d_model"], config.get("base_embedding_dim"), config.get("adaptive_embedding_dim"),
        adaptive=config["use_adaptive_embedding"], fusion=config.get("adaptive_embedding_fusion", "add"),
    )
    if str(config["embedding_fusion"]) == "concat":
        from MSTT.model.embedding import embedding_widths

        embedding_widths(
            "concat",
            base_dim,
            config["value_embedding_dim"],
            config["node_embedding_dim"],
            config["tod_embedding_dim"],
            config["dow_embedding_dim"],
        )


def attention_head_count(config: Mapping[str, Any], key: str) -> int:
    """Missing or null per-attention head counts keep the shared ``num_heads``."""
    value = config.get(key)
    return int(config["num_heads"] if value is None else value)


def with_legacy_defaults(saved: Mapping[str, Any]) -> dict[str, Any]:
    """A stored run config with the old meaning of keys it predates."""
    payload = dict(saved)
    for key, value in LEGACY_DEFAULTS.items():
        payload.setdefault(key, value)
    if isinstance(payload.get("model_config"), Mapping):
        model = dict(payload["model_config"])
        for key, value in LEGACY_DEFAULTS.items():
            model.setdefault(key, value)
        payload["model_config"] = model
    return payload


_CONSERVATION_TASKS = frozenset({"All2All", "AllSpC2All"})


def validate_two_stage_training(config: Mapping[str, Any]) -> None:
    """Loss constraints for the two-stage readout. Other modes keep their own scope."""
    mode = str(config.get("history_fusion_mode", "none"))
    scope = str(config.get("temporal_aux_scope", TRAIN_DEFAULTS["temporal_aux_scope"]))
    if scope == "all_steps" and mode != "two_stage_projected_tsf":
        raise ValueError(
            "temporal_aux_scope='all_steps' requires history_fusion_mode='two_stage_projected_tsf'"
        )
    if mode != "two_stage_projected_tsf":
        return
    if scope != "all_steps":
        raise ValueError(
            "history_fusion_mode='two_stage_projected_tsf' requires temporal_aux_scope='all_steps'"
        )
    weight = float(config.get("temporal_aux_weight", TRAIN_DEFAULTS["temporal_aux_weight"]))
    if weight != 0.5:
        raise ValueError("history_fusion_mode='two_stage_projected_tsf' requires temporal_aux_weight=0.5")
    if str(config.get("projection_supervision", TRAIN_DEFAULTS["projection_supervision"])) != "all":
        raise ValueError("history_fusion_mode='two_stage_projected_tsf' requires projection_supervision='all'")
    for key in (
        "projection_raw_loss_weight",
        "projection_consistency_weight",
        "branch_a_loss_weight",
        "branch_b_loss_weight",
    ):
        value = float(config.get(key, TRAIN_DEFAULTS[key]))
        if value != 0.0:
            raise ValueError(f"history_fusion_mode='two_stage_projected_tsf' requires {key}=0")


def validate_history_fusion_task(config: Mapping[str, Any], task: TaskSpec) -> None:
    """History modes are four-state last-step paths on top of the auxiliary head."""
    from MSTT.model.mstt import validate_history_fusion

    validate_history_fusion(config)
    validate_two_stage_training(config)
    mode = str(config.get("history_fusion_mode", "none"))
    if mode == "none":
        return
    if task.output_channels != STATE_CHANNELS:
        raise ValueError(
            f"history_fusion_mode={mode!r} requires output channels "
            f"{list(STATE_CHANNELS)} in that order, got task {task.name} {list(task.output_channels)}"
        )
    if mode == "two_stage_projected_tsf" and task.input_channels != STATE_CHANNELS:
        raise ValueError(
            "history_fusion_mode='two_stage_projected_tsf' requires input channels "
            f"{list(STATE_CHANNELS)} in that order, got task {task.name} {list(task.input_channels)}"
        )


def validate_temporal_aux_task(config: Mapping[str, Any], task: TaskSpec) -> None:
    """The auxiliary head supervises four states. It stays off unless the config asks for it."""
    if not bool(config.get("temporal_aux_enabled", False)):
        return
    if task.output_channels != STATE_CHANNELS:
        raise ValueError(
            "temporal_aux_enabled requires output channels "
            f"{list(STATE_CHANNELS)} in that order, got task {task.name} {list(task.output_channels)}"
        )
    if int(config.get("input_window", 0)) != 12:
        raise ValueError("temporal_aux_enabled requires input_window=12")


def validate_conservation_task(config: Mapping[str, Any], task: TaskSpec) -> None:
    mode = config.get("conservation_mode", "none")
    if mode == "none":
        return
    if task.name not in _CONSERVATION_TASKS:
        raise ValueError(
            f"conservation_mode={mode} is only allowed for {sorted(_CONSERVATION_TASKS)}, got {task.name}"
        )
    if task.output_channels != STATE_CHANNELS:
        raise ValueError(
            "conservation requires four-state outputs "
            f"{list(STATE_CHANNELS)} in that order, got {list(task.output_channels)}"
        )


def validate_value_embedding_task(config: Mapping[str, Any], task: TaskSpec) -> None:
    """Channel count and order are known once the task is. Fusion stays in ``validate_config``."""
    mode = str(config.get("value_embedding_mode", "joint_linear"))
    if mode == "state_aux_gate":
        if task.input_channels != ALL_SP_C_CHANNELS:
            raise ValueError(
                "value_embedding_mode=state_aux_gate requires input channels "
                f"{list(ALL_SP_C_CHANNELS)}, task {task.name} has {list(task.input_channels)}"
            )
        return
    if mode != "separate_mlp":
        return
    allowed = (STATE_CHANNELS, ALL_SP_C_CHANNELS)
    if task.input_channels not in allowed:
        raise ValueError(
            "value_embedding_mode=separate_mlp requires input channels "
            f"{[list(item) for item in allowed]}, task {task.name} has {list(task.input_channels)}"
        )
    channels = len(task.input_channels)
    value_dim = int(config.get("value_embedding_dim", 0))
    if value_dim % channels != 0 or value_dim // channels < 1:
        raise ValueError(
            "value_embedding_mode=separate_mlp requires value_embedding_dim to be a positive "
            f"multiple of the {channels} input channels, got {value_dim}"
        )


def validate_backbone_task(config: Mapping[str, Any], task: TaskSpec) -> None:
    """Task constraints for ``state_independent``. Shared runs are unchanged."""
    from MSTT.model.mstt import (
        validate_independent_backbone,
        validate_state_context_sharing,
        validate_state_node_sharing,
    )

    validate_state_context_sharing(config)
    validate_state_node_sharing(config)
    mode = str(config.get("backbone_mode", "shared"))
    if mode == "shared":
        return
    if mode != "state_independent":
        raise ValueError(f"unknown backbone_mode {mode!r}")
    value_mode = str(config.get("value_embedding_mode", "joint_linear"))
    if value_mode == "state_aux_gate":
        raise ValueError("backbone_mode=state_independent cannot use value_embedding_mode=state_aux_gate")
    if value_mode != "separate_mlp":
        raise ValueError(
            "backbone_mode=state_independent requires value_embedding_mode='separate_mlp', "
            f"got {value_mode!r}"
        )
    if task.input_channels != STATE_CHANNELS or task.output_channels != STATE_CHANNELS:
        raise ValueError(
            "backbone_mode=state_independent requires input and output channels "
            f"{list(STATE_CHANNELS)} in that order, got task {task.name} "
            f"in={list(task.input_channels)} out={list(task.output_channels)}"
        )
    validate_independent_backbone(config)


def model_values_equal(left: Any, right: Any) -> bool:
    """Equal for checkpoint identity, including ``176`` and ``176.0``."""
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return left == right


def model_config_from_saved(saved: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the model identity stored in a resolved config.

    Nested ``model_config`` is the built model. A top-level field fills a key
    the nested config omitted. Keys missing from both take ``LEGACY_DEFAULTS``.
    An explicit disagreement raises. Current defaults are not used to invent a
    value the checkpoint never recorded.
    """
    from MSTT.model.mstt import MSTTConfig

    names = set(MSTTConfig.__dataclass_fields__)
    raw_model = saved.get("model_config")
    if isinstance(raw_model, Mapping):
        conflicts = []
        for key in sorted(names):
            if key not in raw_model or key not in saved:
                continue
            if saved[key] is None or raw_model[key] is None:
                continue
            if not model_values_equal(saved[key], raw_model[key]):
                conflicts.append(f"{key}: top-level={saved[key]!r} model_config={raw_model[key]!r}")
        if conflicts:
            raise ValueError("saved model_config conflicts with top-level fields: " + "; ".join(conflicts))
        model = {key: raw_model[key] for key in names if key in raw_model}
        for key in names:
            if key not in model and key in saved and saved[key] is not None:
                model[key] = saved[key]
    else:
        model = {key: saved[key] for key in names if key in saved and saved[key] is not None}
        if "in_channels" not in model and saved.get("input_channels") is not None:
            model["in_channels"] = saved["input_channels"]
        if "out_channels" not in model and saved.get("output_channels") is not None:
            model["out_channels"] = saved["output_channels"]
        if "input_window" not in model and saved.get("n_his") is not None:
            model["input_window"] = saved["n_his"]
    for key, value in LEGACY_DEFAULTS.items():
        if key in names:
            model.setdefault(key, value)
    if model.get("ffn_hidden_dim") is None and model.get("d_model") is not None:
        width = int(model["d_model"])
        if str(model.get("backbone_mode", "shared")) == "state_independent":
            if width % 4 != 0:
                raise ValueError(f"state_independent d_model={width} is not divisible by 4")
            width //= 4
        model["ffn_hidden_dim"] = resolve_ffn_hidden_dim(width, None)
    missing = [key for key in sorted(names) if key not in model]
    if missing:
        raise ValueError(
            "saved model identity is incomplete; missing "
            f"{missing}. Refusing to fill them from the current defaults."
        )
    return model


def restore_saved_model_fields(config: dict[str, Any], raw_resolved: Mapping[str, Any]) -> None:
    """Copy the built model fields onto a flat config before evaluation rebuilds it."""
    for key, value in model_config_from_saved(raw_resolved).items():
        config[key] = value


def conservation_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Loss identity checked against checkpoints. A missing supervision key is original C (``all``)."""
    return {key: config.get(key, default_config()[key]) for key in (
        "conservation_mode", "conservation_lambda", "projection_raw_loss_weight",
        "projection_consistency_weight", "projection_supervision", "projection_mlp_hidden_dim",
    )}


def temporal_aux_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Auxiliary-loss identity. A checkpoint that predates either switch is recorded as off."""
    defaults = default_config()
    settings = {
        "temporal_aux_enabled": bool(config.get("temporal_aux_enabled", defaults["temporal_aux_enabled"])),
        "temporal_aux_weight": float(config.get("temporal_aux_weight", defaults["temporal_aux_weight"])),
        "temporal_aux_scope": str(config.get("temporal_aux_scope", defaults["temporal_aux_scope"])),
        "history_fusion_mode": str(config.get("history_fusion_mode", defaults["history_fusion_mode"])),
    }
    if settings["history_fusion_mode"] in {"dual_path_history_tsf", "two_stage_projected_tsf"}:
        settings["branch_a_loss_weight"] = float(config.get("branch_a_loss_weight", defaults["branch_a_loss_weight"]))
        settings["branch_b_loss_weight"] = float(config.get("branch_b_loss_weight", defaults["branch_b_loss_weight"]))
    return settings


def conservation_output_root(config: Mapping[str, Any], root: Path, *, explicit: bool = False) -> Path:
    """Opt-in experiment roots; an omitted switch preserves historical Mode A paths."""
    mode = config.get("conservation_mode", "none")
    root = Path(root).expanduser()
    base = Path(DATA_DEFAULTS["output_root"])
    if (mode != "none" or explicit) and root.resolve() == base.resolve():
        root = root / f"v0.2_concat_A2A_{mode}"
    for part in root.parts:
        if part.startswith("v0.2_concat_A2A_") and part != f"v0.2_concat_A2A_{mode}":
            raise ValueError(f"output root {root} belongs to a different conservation mode")
        if mode != "none" and part.startswith(("v0.1", "v0.2", "v0.3", "v0.4")) and not part.startswith("v0.2_concat_A2A_"):
            raise ValueError(f"conservation experiments must not write into legacy experiment {root}")
    # Check the whole experiment root, including other rates/tasks, before any writes.
    for path in root.glob("p*/*/seed_*/resolved_config.json"):
        stored_mode = json.loads(path.read_text(encoding="utf-8")).get("conservation_mode", "none")
        if stored_mode != mode:
            raise ValueError(f"{root} already contains conservation_mode={stored_mode}; use an independent root")
    return root
