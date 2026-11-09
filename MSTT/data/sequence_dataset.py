"""Xuancheng26 12->12 sequence reconstruction windows for MSTT.

``x[s]`` is the existing 12->1 input: member ``x`` of
``{rate}/{split}/{date}/{input_type}.npz`` (``partial[s:s+12]``, already
z-scored with that rate's input scaler). ``y[s]`` is ``full[s:s+12]`` of the
same date, cut from ``sequence_targets_12step/{split}/{date}.npz`` (one
rate-independent ``full_raw [288, N, 4]`` table per date, written by
``dataset/xuancheng26/model_prep/generate_r_level_sequence_targets.py``) and
z-scored with the 12->1 full-state target scaler. ``y[:, t]`` reconstructs the
same slot as ``x[:, t]``; nothing is forecast.

Both arrays are memory-mapped from uncompressed NPZ members, so forked or
spawned DataLoader workers read the shared page cache instead of each holding
a private copy. The 12->1 loaders and files are not touched.
"""

from __future__ import annotations

import os
import struct
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_root import sampling_root_from_window

def shared_sequence_root(data_root):
    """Prefer the dataset shared link, then the authoritative training-root package."""
    from data_root import sequence_targets_dir

    sampling = sampling_root_from_window(data_root)
    linked = sampling.parent / "shared" / "sequence_targets_12step"
    if linked.exists():
        return linked
    for parent in sampling.parents:
        if parent.name == "training_data_3s":
            return sequence_targets_dir(parent)
    return sequence_targets_dir()

from MSTT.config import STATE_CHANNELS, TASKS, TaskSpec, rate_tag, task_spec
from reimplementation.common.data.r_only_npz_dataset import (
    SPLITS,
    build_dataloader,
    expected_split_counts,
    invert_target,
    load_json,
)
from reimplementation.common.data.state_windows import read_state_target_scaler
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.utils.hashing import sha256_file

SEQUENCE_DIRNAME = "sequence_targets_12step"
MANIFEST_NAME = "sequence_manifest.json"
FORMAT_VERSION = 1
STEPS_PER_DAY = 288


def npz_member_memmap(path: Path, key: str) -> np.memmap | None:
    """Read-only memmap of ``key`` inside an uncompressed ``.npz``; ``None`` if it is compressed."""
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(f"{key}.npy")
    if info.compress_type != zipfile.ZIP_STORED:
        return None
    with open(path, "rb") as handle:
        handle.seek(info.header_offset)
        local = handle.read(30)
        if local[:4] != b"PK\x03\x04":
            raise ReimplementationError(f"{path}:{key} has no local zip header")
        name_length, extra_length = struct.unpack("<HH", local[26:30])
        handle.seek(info.header_offset + 30 + name_length + extra_length)
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        offset = handle.tell()
    if dtype.hasobject:
        raise ReimplementationError(f"{path}:{key} holds Python objects")
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, order="F" if fortran else "C", offset=offset)


def _member(path: Path, key: str) -> np.ndarray:
    mapped = npz_member_memmap(path, key)
    if mapped is not None:
        return mapped
    with np.load(path, allow_pickle=False) as payload:
        return np.array(payload[key])


def _small_members(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ReimplementationError(f"missing NPZ: {path}")
    with np.load(path, allow_pickle=False) as payload:
        missing = [key for key in keys if key not in payload.files]
        if missing:
            raise ReimplementationError(f"{path} missing {missing}")
        return {key: np.array(payload[key]) for key in keys}


def read_sequence_manifest(data_root: Path) -> dict[str, Any]:
    manifest = load_json(shared_sequence_root(data_root) / MANIFEST_NAME)
    if int(manifest.get("format_version", -1)) != FORMAT_VERSION or not manifest.get("complete"):
        raise ReimplementationError(f"{SEQUENCE_DIRNAME}/{MANIFEST_NAME} is not a complete v{FORMAT_VERSION} package")
    if tuple(manifest.get("channels", ())) != STATE_CHANNELS:
        raise ReimplementationError(f"sequence channel order {manifest.get('channels')} != {STATE_CHANNELS}")
    return manifest


class SequenceStateWindowDataset(Dataset):
    """One rate, one split, one task. ``x``/``y``/``y_raw`` are ``[12, N, C]``."""

    def __init__(
        self,
        data_root: Path,
        rate: str,
        split: str,
        task: TaskSpec | str,
        *,
        n_his: int = 12,
        num_nodes: int = 1406,
        return_y_raw: bool = True,
        return_x_raw: bool = False,
        check_last_step: bool = True,
    ) -> None:
        if split not in SPLITS:
            raise ReimplementationError(f"unknown split {split}")
        self.data_root = Path(data_root)
        self.rate = rate_tag(rate)
        self.split = split
        self.task = task if isinstance(task, TaskSpec) else task_spec(str(task))
        self.n_his = int(n_his)
        self.num_nodes = int(num_nodes)
        self.return_y_raw = bool(return_y_raw)
        self.return_x_raw = bool(return_x_raw)

        manifest = read_sequence_manifest(self.data_root)
        split_manifest = load_json(self.data_root / "split_manifest.json")
        self.days = [str(day) for day in split_manifest[split]]
        if self.days != [str(day) for day in manifest["splits"][split]]:
            raise ReimplementationError(f"{split} dates differ between split_manifest and {MANIFEST_NAME}")
        if int(manifest["n_his"]) != self.n_his or int(manifest["output_steps"]) != self.n_his:
            raise ReimplementationError(f"sequence package is {manifest['n_his']}->{manifest['output_steps']}, not {self.n_his}->{self.n_his}")
        if int(manifest["num_nodes"]) != self.num_nodes or int(manifest["steps_per_day"]) != STEPS_PER_DAY:
            raise ReimplementationError("sequence package node count or day length differs")
        from data_root import PHYSICAL_DEFINITION_VERSION

        if manifest.get("physical_definition_version") != PHYSICAL_DEFINITION_VERSION:
            raise ReimplementationError(
                f"sequence targets physical_definition_version={manifest.get('physical_definition_version')!r} "
                f"!= {PHYSICAL_DEFINITION_VERSION}"
            )
        normalization = load_json(self.data_root / "state_normalization.json")
        if normalization.get("physical_definition_version") != PHYSICAL_DEFINITION_VERSION:
            raise ReimplementationError("state_normalization.json is not the current physical definition")
        if manifest.get("normalization", {}).get("target") != normalization.get("target"):
            raise ReimplementationError("sequence target scaler does not match state_normalization.json")
        self.samples_per_day = int(manifest["samples_per_day"])
        if self.samples_per_day != STEPS_PER_DAY - self.n_his + 1:
            raise ReimplementationError(f"samples_per_day {self.samples_per_day} is not 288 - n_his + 1")

        self.output_index = [STATE_CHANNELS.index(name) for name in self.task.output_channels]
        self._out = slice(self.output_index[0], self.output_index[-1] + 1)
        mean, std = read_state_target_scaler(self.data_root, "all")
        self.target_mean = np.asarray(mean, dtype=np.float64)[self.output_index]
        self.target_std = np.asarray(std, dtype=np.float64)[self.output_index]
        self._mean32 = self.target_mean.astype(np.float32)
        self._std32 = self.target_std.astype(np.float32)

        self.weekday = np.zeros(len(self.days), dtype=np.int64)
        for position, day in enumerate(self.days):
            self.weekday[position] = self._check_day(day, manifest, check_last_step)
        count = len(self.days) * self.samples_per_day
        self.sample_day = np.repeat(np.arange(len(self.days), dtype=np.int64), self.samples_per_day)
        self.sample_start = np.tile(np.arange(self.samples_per_day, dtype=np.int64), len(self.days))
        if self.sample_day.shape != (count,):
            raise ReimplementationError("sample index drifted")
        self._maps: tuple[list, list, list] | None = None
        self._pid = -1

    @property
    def in_channels(self) -> int:
        return self.task.in_channels

    @property
    def out_channels(self) -> int:
        return self.task.out_channels

    def _paths(self, day: str) -> tuple[Path, Path, Path]:
        window_dir = self.data_root / self.rate / self.split / day
        return (
            window_dir / f"{self.task.input_type}.npz",
            shared_sequence_root(self.data_root) / self.split / f"{day}.npz",
            window_dir / f"{self.task.target_type}.npz",
        )

    def _check_day(self, day: str, manifest: Mapping[str, Any], check_last_step: bool) -> int:
        x_path, seq_path, old_target_path = self._paths(day)
        entry = manifest["dates"].get(day) or {}
        if entry.get("split") != self.split or entry.get("file") != f"{self.split}/{day}.npz":
            raise ReimplementationError(f"{MANIFEST_NAME} has no {self.split} entry for {day}")
        index = _small_members(x_path, ("day_index", "window_start_slot", "target_slot"))
        starts = np.arange(self.samples_per_day)
        if not np.array_equal(index["window_start_slot"], starts):
            raise ReimplementationError(f"{x_path} window_start_slot is not 0..{self.samples_per_day - 1}")
        if not np.array_equal(index["target_slot"], starts + self.n_his - 1):
            raise ReimplementationError(f"{x_path} target_slot is not window_start_slot + {self.n_his - 1}")
        weekdays = np.unique(index["day_index"])
        meta = _small_members(seq_path, ("day_index", "date", "split", "channels"))
        if weekdays.shape != (1,) or int(weekdays[0]) != int(meta["day_index"]) or int(meta["day_index"]) != int(entry["day_index"]):
            raise ReimplementationError(f"{day} weekday differs between {x_path.name} and {seq_path.name}")
        if str(meta["date"]) != day or str(meta["split"]) != self.split:
            raise ReimplementationError(f"{seq_path} metadata is not {self.split}/{day}")
        if tuple(str(item) for item in meta["channels"].tolist()) != STATE_CHANNELS:
            raise ReimplementationError(f"{seq_path} channel order differs")
        x = _member(x_path, "x")
        expected_x = (self.samples_per_day, self.n_his, self.num_nodes, self.task.in_channels)
        if tuple(x.shape) != expected_x or x.dtype != np.float32:
            raise ReimplementationError(f"{x_path} x {x.shape} {x.dtype} != {expected_x} float32")
        full = _member(seq_path, "full_raw")
        if tuple(full.shape) != (STEPS_PER_DAY, self.num_nodes, len(STATE_CHANNELS)) or full.dtype != np.float32:
            raise ReimplementationError(f"{seq_path} full_raw {full.shape} {full.dtype}")
        if check_last_step:
            old = _small_members(old_target_path, ("y_raw",))["y_raw"]
            if not np.array_equal(old[:, 0], full[self.n_his - 1 :, :, self._out]):
                raise ReimplementationError(f"{seq_path} last step differs from {old_target_path} y_raw")
        return int(meta["day_index"])

    def _open(self) -> tuple[list, list, list]:
        if self._maps is None or self._pid != os.getpid():
            xs, fulls, x_raws = [], [], []
            for day in self.days:
                x_path, seq_path, _ = self._paths(day)
                xs.append(_member(x_path, "x"))
                fulls.append(_member(seq_path, "full_raw"))
                x_raws.append(_member(x_path, "x_raw") if self.return_x_raw else None)
            self._maps = (xs, fulls, x_raws)
            self._pid = os.getpid()
        return self._maps

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_maps"] = None
        state["_pid"] = -1
        return state

    def __len__(self) -> int:
        return int(self.sample_day.shape[0])

    def window_slots(self, index: int) -> np.ndarray:
        start = int(self.sample_start[index])
        return np.arange(start, start + self.n_his, dtype=np.int64)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        index = int(index)
        if index < 0:
            index += len(self)
        day = int(self.sample_day[index])
        start = int(self.sample_start[index])
        xs, fulls, x_raws = self._open()
        x = np.array(xs[day][start], dtype=np.float32)
        y_raw = np.array(fulls[day][start : start + self.n_his, :, self._out], dtype=np.float32)
        y = ((y_raw - self._mean32) / self._std32).astype(np.float32, copy=False)
        slots = self.window_slots(index)
        weekday = int(self.weekday[day])
        sample = {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "day_index": torch.tensor(weekday, dtype=torch.int64),
            "window_start_slot": torch.tensor(start, dtype=torch.int64),
            "target_slot": torch.tensor(start + self.n_his - 1, dtype=torch.int64),
            "time_of_day": torch.from_numpy(slots),
            "day_of_week": torch.full((self.n_his,), weekday, dtype=torch.int64),
            "date_position": torch.tensor(day, dtype=torch.int64),
            "sample_index": torch.tensor(index, dtype=torch.int64),
        }
        if self.return_y_raw:
            sample["y_raw"] = torch.from_numpy(y_raw)
        if self.return_x_raw:
            sample["x_raw"] = torch.from_numpy(np.array(x_raws[day][start], dtype=np.float32))
        return sample

    def inverse_transform(self, y: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Per output channel ``y * std[c] + mean[c]`` on a ``[..., C_out]`` array."""
        return invert_target(y, self.target_mean, self.target_std)


def open_sequence_datasets(
    data_root: Path,
    rate: str,
    task: TaskSpec | str,
    *,
    splits: tuple[str, ...] = SPLITS,
    **kwargs: Any,
) -> dict[str, SequenceStateWindowDataset]:
    counts = expected_split_counts(load_json(Path(data_root) / "split_manifest.json"))
    datasets: dict[str, SequenceStateWindowDataset] = {}
    for split in splits:
        dataset = SequenceStateWindowDataset(Path(data_root), rate, split, task, **kwargs)
        if len(dataset) != int(counts[split]):
            raise ReimplementationError(f"{rate_tag(rate)}/{split} has {len(dataset)} samples, expected {counts[split]}")
        datasets[split] = dataset
    return datasets


def build_sequence_loader(
    dataset: SequenceStateWindowDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    return build_dataloader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, seed=seed)


__all__ = [
    "SEQUENCE_DIRNAME",
    "SequenceStateWindowDataset",
    "TASKS",
    "build_sequence_loader",
    "npz_member_memmap",
    "open_sequence_datasets",
    "read_sequence_manifest",
]
