"""Xuancheng26 four-state windows for MSTT, read through the shared baseline loader.

``AlignedStateWindowDataset`` (``reimplementation/common/data/state_windows.py``)
takes ``x`` from ``{rate}/{split}/{date}/{input_type}.npz`` and ``y`` from the
``target_type`` file of the same rate. Nothing here refits a scaler or
rewrites a file. Scalers come from ``state_normalization.json`` (fit on the
20 training days): inputs are already z-scored per rate and channel, targets
are inverted with the shared full-state scaler.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from MSTT.config import ALL_SP_C_CHANNELS, STATE_CHANNELS, TaskSpec, rate_tag
from reimplementation.common.data.r_only_npz_dataset import (
    SPLITS,
    build_dataloader,
    expected_split_counts,
    load_json,
)
from reimplementation.common.data.state_windows import (
    ALL_SP_C_CHANNELS as WINDOW_ALL_SP_C_CHANNELS,
    AlignedStateWindowDataset,
    open_split_dataset,
    read_all_sp_c_input_scaler,
    read_state_target_scaler,
)
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.utils.hashing import sha256_file

if WINDOW_ALL_SP_C_CHANNELS != ALL_SP_C_CHANNELS:
    raise ReimplementationError(
        f"all_sp_c channel order drifted: windows {WINDOW_ALL_SP_C_CHANNELS} != MSTT {ALL_SP_C_CHANNELS}"
    )


def validate_state_package(data_root: Path, *, num_nodes: int, n_his: int) -> dict[str, Any]:
    root = Path(data_root)
    manifest = load_json(root / "split_manifest.json")
    state_manifest = load_json(root / "state_window_manifest.json")
    state_norm = load_json(root / "state_normalization.json")
    report = load_json(root / "validation_summary.json")
    counts = expected_split_counts(manifest)
    if report.get("failures") or report.get("overall_validation_passed") is False:
        raise ReimplementationError(f"{root}/validation_summary.json reports failures")
    if tuple(state_manifest.get("channels", ())) != STATE_CHANNELS:
        raise ReimplementationError(f"state channel order {state_manifest.get('channels')} != {STATE_CHANNELS}")
    if tuple(state_norm.get("channels", ())) != STATE_CHANNELS:
        raise ReimplementationError("state_normalization.json channel order differs")
    if state_norm.get("fit_split") != "train":
        raise ReimplementationError("state_normalization.json was not fit on the training split")
    days = {split: [str(day) for day in manifest[split]] for split in SPLITS}
    if [str(day) for day in state_norm.get("training_days", [])] != days["train"]:
        raise ReimplementationError("state scaler training days differ from the split manifest")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(days[left]) & set(days[right]):
            raise ReimplementationError(f"{left} and {right} days overlap")
    for key in ("target_mode",):
        if str(manifest.get(key)) != "last-observed-step" or str(state_manifest.get(key)) != "last-observed-step":
            raise ReimplementationError("windows are not last-observed-step reconstruction")
    if int(manifest.get("n_his", n_his)) != int(n_his) or int(state_manifest.get("n_his", n_his)) != int(n_his):
        raise ReimplementationError(f"prepared n_his != {n_his}")
    if int(state_manifest.get("num_nodes", num_nodes)) != int(num_nodes):
        raise ReimplementationError(f"prepared num_nodes != {num_nodes}")
    return {
        "sample_counts": counts,
        "days": days,
        "split_manifest_sha256": sha256_file(root / "split_manifest.json"),
        "state_normalization_sha256": sha256_file(root / "state_normalization.json"),
        "node_mapping": (root / "node_mapping.csv").as_posix(),
    }


def input_scaler(data_root: Path, rate: str, task: TaskSpec) -> tuple[np.ndarray, np.ndarray]:
    if task.input_type == "all_sp_c":
        if task.input_channels != ALL_SP_C_CHANNELS:
            raise ReimplementationError(f"all_sp_c input order {task.input_channels} != {ALL_SP_C_CHANNELS}")
        if any(name in task.output_channels for name in ALL_SP_C_CHANNELS[len(STATE_CHANNELS) :]):
            raise ReimplementationError("speed and snapshot count are not prediction targets")
        return read_all_sp_c_input_scaler(data_root, rate)
    payload = load_json(Path(data_root) / "state_normalization.json")
    table = (payload.get("input") or {}).get(rate_tag(rate))
    if not table:
        raise ReimplementationError(f"state_normalization.json has no input scaler for {rate}")
    mean = np.asarray([float(table[name]["mean"]) for name in task.input_channels], dtype=np.float64)
    std = np.asarray([float(table[name]["std"]) for name in task.input_channels], dtype=np.float64)
    return mean, std


def target_scaler(data_root: Path, task: TaskSpec) -> tuple[np.ndarray, np.ndarray]:
    mean, std = read_state_target_scaler(Path(data_root), task.target_type)
    return np.atleast_1d(np.asarray(mean, dtype=np.float64)), np.atleast_1d(np.asarray(std, dtype=np.float64))


def open_task_datasets(
    data_root: Path,
    rate: str,
    task: TaskSpec,
    *,
    n_his: int,
    num_nodes: int,
    sample_counts: Mapping[str, int],
    splits: tuple[str, ...] = SPLITS,
) -> dict[str, AlignedStateWindowDataset]:
    tag = rate_tag(rate)
    selector = {"input_type": task.input_type, "target_type": task.target_type, "num_nodes": num_nodes}
    in_mean, in_std = input_scaler(data_root, tag, task)
    datasets: dict[str, AlignedStateWindowDataset] = {}
    for split in splits:
        dataset = open_split_dataset(
            Path(data_root),
            tag,
            split,
            selector,
            n_his=n_his,
            num_nodes=num_nodes,
            expected_count=int(sample_counts[split]),
        )
        if not isinstance(dataset, AlignedStateWindowDataset):
            raise ReimplementationError("expected the dated four-state loader")
        for name in ("x", "y", "x_raw", "y_raw"):
            if not np.isfinite(getattr(dataset, name)).all():
                raise ReimplementationError(f"{tag}/{split} {name} has NaN or inf")
        probe = dataset.x[:8].astype(np.float64) * in_std + in_mean
        if float(np.max(np.abs(probe - dataset.x_raw[:8]))) > 1e-3:
            raise ReimplementationError(f"{tag}/{split} x is not z-scored with the {tag} input scaler")
        datasets[split] = dataset
    return datasets


def train_data_sha256(data_root: Path, rate: str, task: TaskSpec, days: list[str]) -> str:
    """One digest over every training-day file this task reads (input then target)."""
    digest = hashlib.sha256()
    kinds = [task.input_type] if task.input_type == task.target_type else [task.input_type, task.target_type]
    for kind in kinds:
        for day in days:
            path = Path(data_root) / rate_tag(rate) / "train" / day / f"{kind}.npz"
            digest.update(f"{kind}/{day}:".encode("utf-8"))
            digest.update(sha256_file(path).encode("utf-8"))
    return digest.hexdigest()


def build_loader(dataset, *, batch_size: int, shuffle: bool, seed: int, num_workers: int = 0) -> DataLoader:
    return build_dataloader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, seed=seed)


def time_features(
    batch: Mapping[str, torch.Tensor], n_his: int, *, steps_per_day: int = 288
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-step ``time_of_day = window_start_slot + t`` and ``day_of_week = day_index % 7``.

    ``day_index`` is the Monday=0 weekday (split_manifest ``day_index_encoding``).
    Windows never cross midnight, so every step shares the weekday. Same
    convention as the STAEformer and TLAST loaders.
    """
    start = batch["window_start_slot"].long().reshape(-1)
    offsets = torch.arange(int(n_his), device=start.device, dtype=torch.long)
    slots = start[:, None] + offsets[None, :]
    if bool((slots < 0).any()) or bool((slots >= int(steps_per_day)).any()):
        raise ReimplementationError("history slots leave the day; windows must not cross midnight")
    if not torch.equal(slots[:, -1], batch["target_slot"].long().reshape(-1)):
        raise ReimplementationError("target_slot must be the last history slot")
    weekday = batch["day_index"].long().reshape(-1)
    if bool((weekday < 0).any()):
        raise ReimplementationError("day_index must be non-negative")
    weekday = torch.remainder(weekday, 7)[:, None].expand(-1, int(n_his))
    return slots, weekday
