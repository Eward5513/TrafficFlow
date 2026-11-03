"""Tasks, penetration rates, default configuration, and CLI overlays for MSTT.

Paths are resolved through ``data_root.py``. The four-state windows exist
only in the 3-second-phase package, so that package and its sibling results
root are the defaults, the same place the 9-task baselines read and write.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from data_root import xuancheng26_model_results_3s_dir, xuancheng26_training_3s_root

STATE_CHANNELS = ("inflow", "outflow", "accumulation_start", "accumulation_end")
RATES = ("p05", "p10", "p20", "p30", "p40", "p50", "p70")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    input_type: str
    target_type: str

    @property
    def input_channels(self) -> tuple[str, ...]:
        return STATE_CHANNELS if self.input_type == "all" else (self.input_type,)

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
    )
}


def task_spec(name: str) -> TaskSpec:
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; choose from {list(TASKS)}")
    return TASKS[name]


def rate_tag(rate: str | int) -> str:
    text = str(rate).strip()
    if text.startswith("p"):
        text = text[1:]
    tag = f"p{int(text):02d}"
    if tag not in RATES:
        raise ValueError(f"penetration {tag} is not one of {list(RATES)}")
    return tag


MODEL_DEFAULTS: dict[str, Any] = {
    "d_model": 64,
    "num_heads": 4,
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
    "steps_per_day": 288,
    "norm_eps": 1e-6,
    "head_type": "flatten",
}

TRAIN_DEFAULTS: dict[str, Any] = {
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
    "data_root": (xuancheng26_training_3s_root() / "window_data").as_posix(),
    "graph_path": (xuancheng26_training_3s_root() / "graph" / "r_graph.npz").as_posix(),
    "r_nodes": (xuancheng26_training_3s_root() / "graph" / "r_nodes.csv").as_posix(),
    "output_root": xuancheng26_model_results_3s_dir("MSTT").as_posix(),
    "num_nodes": 1406,
    "target_mode": "last-observed-step",
}


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


_OPTIONAL_INT_KEYS = {"ffn_hidden_dim", "max_train_batches", "max_eval_batches", "early_stop_patience"}
_CHOICES = {
    "spatial_attention": ("sparse",),
    "spatial_direction": ("forward", "backward", "bidirectional"),
    "head_type": ("flatten", "last"),
    "loss": ("mae_raw", "huber_raw", "mse_raw"),
    "channel_balance": ("none", "target_std"),
    "selection_metric": ("val_loss", "val_mae", "val_scaled_mae"),
    "amp": ("none", "bf16", "fp16"),
    "lr_scheduler": ("cosine", "none"),
}


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """One flag per config key, e.g. ``--d-model`` (``--d_model`` also works)."""
    for key, value in default_config().items():
        flags = [f"--{key.replace('_', '-')}"]
        if "_" in key:
            flags.append(f"--{key}")
        kwargs: dict[str, Any] = {"dest": key, "default": None}
        if key in _OPTIONAL_INT_KEYS:
            kwargs["type"] = _parse_optional_int
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
    for key in default_config():
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    return config


def resolve_ffn_hidden_dim(d_model: int, value: int | None) -> int:
    """SwiGLU default: 8/3 * d_model rounded up to a multiple of 8 (FFN params ~ 4d ReLU)."""
    if value is not None:
        return int(value)
    hidden = int(round(8 * int(d_model) / 3))
    return ((hidden + 7) // 8) * 8


def validate_config(config: Mapping[str, Any]) -> None:
    for key, choices in _CHOICES.items():
        if str(config[key]) not in choices:
            raise ValueError(f"{key}={config[key]!r} not in {choices}")
    if str(config["target_mode"]) != "last-observed-step":
        raise ValueError("MSTT reconstructs the last observed slot; target_mode must be last-observed-step")
    if int(config["d_model"]) % int(config["num_heads"]) != 0:
        raise ValueError("d_model must be divisible by num_heads")
    if bool(config["use_rope"]) and (int(config["d_model"]) // int(config["num_heads"])) % 2 != 0:
        raise ValueError("RoPE needs an even head dimension")
    if int(config["input_window"]) < 1 or int(config["num_layers"]) < 1:
        raise ValueError("input_window and num_layers must be positive")
