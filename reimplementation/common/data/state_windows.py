"""Aligned canonical dated state windows; channels and scaler semantics are shared.

All model entry points default to seed42 training and five-seed evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from reimplementation.common.data.r_only_npz_dataset import (
    SPLITS,
    declared_observation_keys,
    invert_target,
    load_json,
    load_npz,
    rate_tag,
)
from reimplementation.common.errors import ReimplementationError

STATE_TYPES = (
    "inflow",
    "outflow",
    "accumulation_start",
    "accumulation_end",
    "all",
)
CHANNELS = STATE_TYPES[:4]
_INCOMPATIBLE_ALL = {
    "pdformer": (
        "PDFormer keeps a one-channel traffic stream: pattern keys were built for "
        "output_dim=1 and feature_dim - ext_dim must equal output_dim. "
        "inflow, outflow, accumulation_start, and accumulation_end are accepted. "
        "`all` is not, because that would require new pattern keys and a different output head."
    ),
}


def state_types_selected(config: Mapping[str, Any]) -> bool:
    return bool(config.get("input_type") or config.get("target_type"))


def channel_count(kind: str) -> int:
    if kind not in STATE_TYPES:
        raise ReimplementationError(f"unknown state type {kind}")
    return 4 if kind == "all" else 1


def add_state_type_arguments(parser) -> None:
    parser.add_argument("--dataset-id", dest="dataset_id", default=None)
    parser.add_argument("--sampling-root", type=Path, default=None)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--train-sampling-seed", type=int, default=42)
    parser.add_argument("--evaluation-sampling-seeds", type=int, nargs="+", default=[42,43,44,45,46])
    parser.add_argument("--input-type", dest="input_type", choices=STATE_TYPES, default=None)
    parser.add_argument("--target-type", dest="target_type", choices=STATE_TYPES, default=None)


def apply_state_types(config: dict[str, Any], args, *, model_name: str) -> dict[str, Any]:
    """Resolve canonical sampling paths and task channels (inflow defaults)."""
    input_flag = getattr(args, "input_type", None)
    target_flag = getattr(args, "target_type", None)
    from reimplementation.common.data.sampling_seeds import configure_sampling
    config = dict(config)
    explicit: list[str] = []
    if getattr(args, "dataset_id", None):
        config["dataset_id"] = str(args.dataset_id)
        explicit.append("dataset_id")
        if getattr(args, "sampling_root", None) is None:
            config.pop("sampling_root", None)
            config.pop("data_root", None)
    for key in ("sampling_root", "model_seed", "train_sampling_seed", "evaluation_sampling_seeds"):
        if getattr(args, key, None) is not None:
            config[key] = str(getattr(args, key)) if key == "sampling_root" else getattr(args, key)
            if key == "sampling_root":
                explicit.append("sampling_root")
    if getattr(args, "seed", None) is not None and getattr(args, "model_seed", None) is None:
        config["model_seed"] = args.seed
    if getattr(args, "data_root", None) is not None:
        from data_root import sampling_root_from_window
        config["sampling_root"] = str(sampling_root_from_window(args.data_root))
        explicit.append("sampling_root")
    config["explicit_path_keys"] = explicit
    config = configure_sampling(config, model_name=model_name)
    input_type = input_flag or config.get("input_type") or "inflow"
    target_type = target_flag or config.get("target_type") or "inflow"
    if model_name in _INCOMPATIBLE_ALL and (input_type == "all" or target_type == "all"):
        raise ReimplementationError(_INCOMPATIBLE_ALL[model_name])
    config["input_type"] = input_type
    config["target_type"] = target_type
    in_channels = channel_count(input_type)
    out_channels = channel_count(target_type)
    config["input_channels"] = in_channels
    config["output_channels"] = out_channels
    blocks = config.get("blocks")
    if isinstance(blocks, list) and blocks and all(isinstance(item, list) and item for item in blocks):
        updated = [list(item) for item in blocks]
        updated[0][0] = in_channels
        config["blocks"] = updated
    if model_name == "stid":
        extras = 0
        if bool(config.get("if_T_i_D", config.get("if_time_in_day", True))):
            extras += 1
        if bool(config.get("if_D_i_W", config.get("if_day_in_week", True))):
            extras += 1
        config["input_dim"] = in_channels + extras
        config["output_dim"] = out_channels
    elif model_name == "staeformer":
        extras = 0
        if int(config.get("tod_embedding_dim", 0)) > 0:
            extras += 1
        if int(config.get("dow_embedding_dim", 0)) > 0:
            extras += 1
        config["input_dim"] = in_channels + extras
        config["output_dim"] = out_channels
    else:
        if "input_dim" in config:
            config["input_dim"] = in_channels
        if "output_dim" in config:
            config["output_dim"] = out_channels
    # output_root stays the results pointer. sampling_rate_dir adds dataset,
    # physical version, task, observation key, experiment, model seed, and run id once.
    return config


def assert_physical_package(data_root: Path) -> None:
    """Reject cumsum-era labels and unfinished packages on the official 3-second tree."""
    from data_root import PHYSICAL_DEFINITION_VERSION

    root = Path(data_root)
    path = root / "state_normalization.json"
    if not path.is_file():
        return
    payload = load_json(path)
    official = "training_data_3s" in root.parts or bool(payload.get("physical_definition_version"))
    if not official:
        return
    version = payload.get("physical_definition_version")
    if version != PHYSICAL_DEFINITION_VERSION:
        raise ReimplementationError(
            f"refusing {root}: physical_definition_version={version!r}, expected {PHYSICAL_DEFINITION_VERSION}"
        )
    if payload.get("dataset_status") in {"building", "failed"}:
        raise ReimplementationError(f"refusing {root}: dataset_status={payload.get('dataset_status')}")
    report_path = root / "validation_summary.json"
    if report_path.is_file():
        report = load_json(report_path)
        if report.get("dataset_status") in {"building", "failed"} or report.get("overall_validation_passed") is False:
            raise ReimplementationError(f"refusing {root}: validation_summary is not ready")


def read_state_target_scaler(data_root: Path, target_type: str) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    payload = load_json(Path(data_root) / "state_normalization.json")
    if payload.get("fit_split") != "train":
        raise ReimplementationError("state_normalization.json was not fit on training only")
    target = payload.get("target") or {}
    if target_type == "all":
        missing = [name for name in CHANNELS if name not in target]
        if missing:
            raise ReimplementationError(f"state_normalization.json missing target {missing}")
        mean = np.asarray([float(target[name]["mean"]) for name in CHANNELS], dtype=np.float64)
        std = np.asarray([float(target[name]["std"]) for name in CHANNELS], dtype=np.float64)
    else:
        item = target.get(target_type)
        if not item:
            raise ReimplementationError(f"state_normalization.json missing target {target_type}")
        mean = float(item["mean"])
        std = float(item["std"])
    std_array = np.asarray(std, dtype=np.float64)
    if not np.isfinite(np.asarray(mean)).all() or not np.isfinite(std_array).all() or np.any(std_array <= 0.0):
        raise ReimplementationError(f"invalid state target scaler for {target_type}")
    return mean, std


def finalize_state_scaler(data_info: Mapping[str, Any], config: Mapping[str, Any], data_root: Path) -> dict[str, Any]:
    if not state_types_selected(config):
        return dict(data_info)
    target_type = str(config.get("target_type") or "inflow")
    mean, std = read_state_target_scaler(data_root, target_type)
    info = dict(data_info)
    if np.ndim(mean) == 0:
        info["mean_y_full"] = float(mean)
        info["std_y_full"] = float(std)
    else:
        info["mean_y_full"] = [float(value) for value in np.asarray(mean)]
        info["std_y_full"] = [float(value) for value in np.asarray(std)]
    info["input_type"] = str(config.get("input_type") or "inflow")
    info["target_type"] = target_type
    return info


def _history_steps(config: Mapping[str, Any]) -> int:
    for key in ("n_his", "seq_len", "seq_length", "input_len", "input_window", "in_steps"):
        if config.get(key) is not None:
            return int(config[key])
    return 12


def _output_steps(config: Mapping[str, Any]) -> int:
    for key in ("output_steps", "horizon", "output_window", "out_steps", "output_len"):
        if config.get(key) is not None:
            return int(config[key])
    return 1


class AlignedStateWindowDataset(Dataset):
    """``x`` comes from the input-type NPZ and ``y`` from the target-type NPZ."""

    def __init__(
        self,
        data_root: Path,
        rate: str,
        split: str,
        *,
        input_type: str,
        target_type: str,
        n_his: int = 12,
        num_nodes: int = 1406,
        output_steps: int = 1,
        expected_count: int | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ReimplementationError(f"unknown split {split}")
        self.data_root = Path(data_root)
        assert_physical_package(self.data_root)
        self.rate = rate_tag(rate, allowed=declared_observation_keys(self.data_root))
        self.split = split
        self.input_type = input_type
        self.target_type = target_type
        manifest = load_json(self.data_root / "split_manifest.json")
        days = [str(day) for day in manifest[split]]
        if not days:
            raise ReimplementationError(f"{split} manifest has no dates")
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        x_raws: list[np.ndarray] = []
        y_raws: list[np.ndarray] = []
        day_index: list[np.ndarray] = []
        starts: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        input_channels = channel_count(input_type)
        output_channels = channel_count(target_type)
        for day in days:
            directory = self.data_root / self.rate / split / day
            incoming = load_npz(directory / f"{input_type}.npz")
            outgoing = incoming if input_type == target_type else load_npz(directory / f"{target_type}.npz")
            _assert_same_windows(directory, incoming, outgoing)
            if int(incoming["x"].shape[-1]) != input_channels:
                raise ReimplementationError(
                    f"{directory / input_type}.npz x channels {incoming['x'].shape[-1]} != {input_channels}"
                )
            if int(outgoing["y"].shape[-1]) != output_channels:
                raise ReimplementationError(
                    f"{directory / target_type}.npz y channels {outgoing['y'].shape[-1]} != {output_channels}"
                )
            xs.append(incoming["x"])
            x_raws.append(incoming["x_raw"])
            ys.append(outgoing["y"])
            y_raws.append(outgoing["y_raw"])
            day_index.append(outgoing["day_index"])
            starts.append(outgoing["window_start_slot"])
            targets.append(outgoing["target_slot"])
        self.x = np.concatenate(xs, axis=0)
        self.x_raw = np.concatenate(x_raws, axis=0)
        self.y = np.concatenate(ys, axis=0)
        self.y_raw = np.concatenate(y_raws, axis=0)
        self.day_index = np.concatenate(day_index, axis=0)
        self.window_start_slot = np.concatenate(starts, axis=0)
        self.target_slot = np.concatenate(targets, axis=0)
        if expected_count is not None and int(self.x.shape[0]) != int(expected_count):
            raise ReimplementationError(
                f"{self.rate}/{split} sample count {self.x.shape[0]} != manifest {expected_count}"
            )
        expected_x = (int(self.x.shape[0]), int(n_his), int(num_nodes), input_channels)
        expected_y = (int(self.y.shape[0]), int(output_steps), int(num_nodes), output_channels)
        if tuple(self.x.shape) != expected_x or tuple(self.x_raw.shape) != expected_x:
            raise ReimplementationError(f"{self.rate}/{split} x shape {self.x.shape} != {expected_x}")
        if tuple(self.y.shape) != expected_y or tuple(self.y_raw.shape) != expected_y:
            raise ReimplementationError(f"{self.rate}/{split} y shape {self.y.shape} != {expected_y}")
        if not np.array_equal(self.target_slot, self.window_start_slot + (int(n_his) - 1)):
            raise ReimplementationError(f"{self.rate}/{split} target_slot is not the last history slot")
        mean, std = read_state_target_scaler(self.data_root, target_type)
        recovered = invert_target(self.y[:8], mean, std)
        if float(np.max(np.abs(np.asarray(recovered) - self.y_raw[:8]))) > 1e-3:
            raise ReimplementationError(
                f"{self.rate}/{split} {target_type} scaler does not recover y_raw"
            )

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(np.array(self.x[index], copy=True)),
            "y": torch.from_numpy(np.array(self.y[index], copy=True)),
            "x_raw": torch.from_numpy(np.array(self.x_raw[index], copy=True)),
            "y_raw": torch.from_numpy(np.array(self.y_raw[index], copy=True)),
            "day_index": torch.tensor(int(self.day_index[index]), dtype=torch.int64),
            "window_start_slot": torch.tensor(int(self.window_start_slot[index]), dtype=torch.int64),
            "target_slot": torch.tensor(int(self.target_slot[index]), dtype=torch.int64),
            "sample_index": torch.tensor(int(index), dtype=torch.int64),
        }


def _assert_same_windows(directory: Path, incoming: Mapping[str, np.ndarray], outgoing: Mapping[str, np.ndarray]) -> None:
    if int(incoming["x"].shape[0]) != int(outgoing["y"].shape[0]):
        raise ReimplementationError(f"{directory} input/target sample counts differ")
    if int(incoming["x"].shape[2]) != int(outgoing["y"].shape[2]):
        raise ReimplementationError(f"{directory} input/target node counts differ")
    for key in ("window_start_slot", "target_slot", "day_index"):
        if not np.array_equal(incoming[key], outgoing[key]):
            raise ReimplementationError(f"{directory} input/target {key} differ")


def open_split_dataset(
    data_root: Path,
    rate: str,
    split: str,
    config: Mapping[str, Any],
    *,
    n_his: int | None = None,
    num_nodes: int | None = None,
    expected_count: int | None = None,
) -> Dataset:
    """Read the canonical dated package; training is restricted to sampling42."""
    from data_root import sampling_root_from_window
    sampling_root_from_window(data_root)
    if split == "train" and Path(data_root).parent.parent.name != "seed_42":
        raise ReimplementationError("training uses sampling seed42 only")

    steps = int(n_his if n_his is not None else _history_steps(config))
    nodes = int(num_nodes if num_nodes is not None else config["num_nodes"])
    return AlignedStateWindowDataset(
        Path(data_root),
        rate,
        split,
        input_type=str(config.get("input_type") or "inflow"),
        target_type=str(config.get("target_type") or "inflow"),
        n_his=steps,
        num_nodes=nodes,
        output_steps=_output_steps(config),
        expected_count=expected_count,
    )
