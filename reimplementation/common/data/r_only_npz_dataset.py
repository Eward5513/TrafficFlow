"""Read prepared R-only STGCN NPZ windows without refitting normalization."""

from __future__ import annotations

import json
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


def rate_tag(rate: int | str) -> str:
    if isinstance(rate, str):
        text = rate.strip()
        if text.startswith("p"):
            return f"p{int(text[1:]):02d}"
        return f"p{int(text):02d}"
    return f"p{int(rate):02d}"


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


def invert_target(normalized: np.ndarray | torch.Tensor, mean: float, std: float) -> np.ndarray | torch.Tensor:
    """Invert the shared full-flow target scaler. Never use the input scaler."""
    return normalized * std + mean


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
        num_nodes: int = 56,
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
