"""Read prepared R-only STGCN NPZ windows without refitting normalization."""

from __future__ import annotations

import contextvars
import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from reimplementation.common.errors import ReimplementationError

NPZ_KEYS = (
    "x",
    "y",
    "x_raw",
    "y_raw",
    "day_index",
    "window_start_slot",
    "target_slot",
)
RATE_TAGS = ("p05", "p10", "p20", "p30", "p40", "p50", "p70")
SPLITS = ("train", "validation", "test")
_OBSERVATION_KEY = re.compile(r"^p\d+(?:p\d+)?_p\d+(?:p\d+)?$")
_ALLOWED_KEYS: contextvars.ContextVar[tuple[str, ...] | None] = contextvars.ContextVar(
    "xuancheng_observation_keys",
    default=None,
)


def bind_observation_keys(keys: tuple[str, ...] | list[str] | None) -> None:
    """Remember the selected dataset's observation keys for later path checks."""
    _ALLOWED_KEYS.set(None if keys is None else tuple(str(item) for item in keys))


def rate_tag(rate: int | str, *, allowed: tuple[str, ...] | list[str] | None = None) -> str:
    """Fixed penetrations stay the seven ``pXX`` tags.

    A range such as ``p10_p20`` is accepted only when the selected dataset
    manifest declares it. It is never parsed with ``int``.
    """
    if allowed is None:
        allowed = _ALLOWED_KEYS.get()
    allowed_set = None if allowed is None else {str(item) for item in allowed}
    text = str(rate).strip()
    if re.fullmatch(r"p?\d+", text):
        number = int(text[1:] if text.startswith("p") else text)
        tag = f"p{number:02d}"
        if tag not in RATE_TAGS:
            raise ValueError(f"penetration {text!r} is not one of {RATE_TAGS}")
        if allowed_set is not None and tag not in allowed_set:
            raise ValueError(f"penetration {tag} is not declared for this dataset")
        return tag
    if _OBSERVATION_KEY.fullmatch(text):
        # Keys are not known until the dataset manifest is bound. Callers that
        # parse the CLI before configure_sampling must not reject p10_p20 here.
        if allowed_set is None:
            return text
        if text in RATE_TAGS or text not in allowed_set:
            raise ValueError(
                f"observation key {text} is not declared by the selected dataset"
            )
        return text
    raise ValueError(f"invalid penetration or observation key: {text!r}")


def dataset_declares(tag: str) -> bool:
    """Whether the bound dataset, or the fixed seven when nothing is bound, includes ``tag``."""
    text = str(tag)
    allowed = _ALLOWED_KEYS.get()
    if allowed is None:
        return text in RATE_TAGS
    return text in {str(item) for item in allowed}


def is_bound_observation_key(tag: str) -> bool:
    """True when ``tag`` is an observation key of the dataset bound in this process."""
    text = str(tag)
    if not _OBSERVATION_KEY.fullmatch(text):
        return False
    allowed = _ALLOWED_KEYS.get()
    return allowed is not None and text in {str(item) for item in allowed}


def declared_observation_keys(data_root: Path) -> tuple[str, ...]:
    """Keys a window directory is allowed to open. Missing manifest means the fixed seven."""
    path = Path(data_root) / "state_window_manifest.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        rates = payload.get("rates")
        if rates:
            keys = tuple(str(item) for item in rates)
            for key in keys:
                rate_tag(key, allowed=keys)
            return keys
    return RATE_TAGS


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReimplementationError(f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ReimplementationError(f"missing NPZ: {path}")
    with np.load(path, allow_pickle=False) as handle:
        missing = [key for key in NPZ_KEYS if key not in handle.files]
        if missing:
            raise ReimplementationError(f"{path} missing fields {missing}")
        return {key: np.array(handle[key]) for key in handle.files}


def _scale_is_scalar(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return True
    if torch.is_tensor(value):
        return value.ndim == 0 or int(value.numel()) == 1
    array = np.asarray(value)
    return array.ndim == 0 or array.size == 1


def loss_scale(value: Any) -> float | np.ndarray:
    """Keep a one-number scaler as float. A channel vector stays shape ``(C,)``."""
    if _scale_is_scalar(value):
        if torch.is_tensor(value):
            return float(value.detach().cpu().reshape(-1)[0])
        return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _broadcast_scale(value: Any, reference: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    shape = (1,) * (int(reference.ndim) - 1) + (int(array.shape[0]),)
    if torch.is_tensor(reference):
        return torch.as_tensor(array, dtype=reference.dtype, device=reference.device).reshape(shape)
    return array.reshape(shape)


def invert_target(normalized: np.ndarray | torch.Tensor, mean: Any, std: Any) -> np.ndarray | torch.Tensor:
    """Invert the target scaler. A scalar applies to every element; a vector is per channel."""
    if _scale_is_scalar(mean) and _scale_is_scalar(std):
        return normalized * std + mean
    return normalized * _broadcast_scale(std, normalized) + _broadcast_scale(mean, normalized)


class ROnlyNPZDataset(Dataset):
    """One split of one penetration rate.

    Sample dict tensors keep the public layout:
        x, y, x_raw, y_raw : [T, V, C] / [T_out, V, C]
    """

    def __init__(
        self,
        npz_path: Path,
        *,
        n_his: int = 12,
        num_nodes: int | None = None,
        input_channels: int = 1,
        output_channels: int = 1,
        output_steps: int = 1,
        expected_count: int | None = None,
    ) -> None:
        self.npz_path = Path(npz_path)
        arrays = load_npz(self.npz_path)
        self.x = arrays["x"]
        self.y = arrays["y"]
        self.x_raw = arrays["x_raw"]
        self.y_raw = arrays["y_raw"]
        self.day_index = arrays["day_index"]
        self.window_start_slot = arrays["window_start_slot"]
        self.target_slot = arrays["target_slot"]
        if num_nodes is None:
            num_nodes = int(self.x.shape[2])
        self._validate_shapes(
            n_his=n_his,
            num_nodes=num_nodes,
            input_channels=input_channels,
            output_channels=output_channels,
            output_steps=output_steps,
            expected_count=expected_count,
        )

    def _validate_shapes(
        self,
        *,
        n_his: int,
        num_nodes: int,
        input_channels: int,
        output_channels: int,
        output_steps: int,
        expected_count: int | None,
    ) -> None:
        n_samples = int(self.x.shape[0])
        if expected_count is not None and n_samples != int(expected_count):
            raise ReimplementationError(
                f"{self.npz_path} sample count {n_samples} != manifest {expected_count}"
            )
        expected_x = (n_samples, n_his, num_nodes, input_channels)
        expected_y = (n_samples, output_steps, num_nodes, output_channels)
        for name, array, expected in (
            ("x", self.x, expected_x),
            ("x_raw", self.x_raw, expected_x),
            ("y", self.y, expected_y),
            ("y_raw", self.y_raw, expected_y),
        ):
            if tuple(array.shape) != expected:
                raise ReimplementationError(
                    f"{self.npz_path} {name} shape {array.shape} != {expected}"
                )
            if not np.isfinite(array).all():
                raise ReimplementationError(f"{self.npz_path} {name} contains NaN or inf")
        if not np.array_equal(self.target_slot, self.window_start_slot + (n_his - 1)):
            raise ReimplementationError(
                f"{self.npz_path} target_slot != window_start_slot + {n_his - 1}; "
                "this task is last-observed-step reconstruction, not next-slot forecasting"
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


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        generator=generator if shuffle else None,
        collate_fn=_collate_batch,
    )


def _collate_batch(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = samples[0].keys()
    return {key: torch.stack([sample[key] for sample in samples], dim=0) for key in keys}


def read_target_scaler(normalization: Mapping[str, Any]) -> tuple[float, float]:
    scaler = normalization.get("target_scaler") or {}
    if "mean_y_full" not in scaler or "std_y_full" not in scaler:
        raise ReimplementationError("normalization.json is missing target_scaler.mean_y_full/std_y_full")
    mean = float(scaler["mean_y_full"])
    std = float(scaler["std_y_full"])
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0.0:
        raise ReimplementationError(f"invalid target scaler mean={mean} std={std}")
    return mean, std


def expected_split_counts(split_manifest: Mapping[str, Any]) -> dict[str, int]:
    counts = split_manifest.get("actual_sample_counts") or split_manifest.get("theoretical_sample_counts") or {}
    result: dict[str, int] = {}
    for split in SPLITS:
        if split not in counts:
            raise ReimplementationError(f"split_manifest missing sample count for {split}")
        result[split] = int(counts[split])
    return result
