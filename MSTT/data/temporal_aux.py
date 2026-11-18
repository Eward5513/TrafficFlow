"""Train-only 12-step raw labels for the temporal auxiliary head.

The 12->1 window dataset stays the source of ``x`` and the last-step ``y_raw``.
Each training sample also reads ``full_raw[window_start_slot:window_start_slot+12]``
from the shared sequence package. Validation and test stay on the last step.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from data_root import PHYSICAL_DEFINITION_VERSION
from MSTT.config import STATE_CHANNELS
from MSTT.data.sequence_dataset import _member, shared_sequence_root
from reimplementation.common.data.r_only_npz_dataset import load_json
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.utils.hashing import sha256_file

SLICE_DEFINITION = {
    "member": "full_raw",
    "window": "full_raw[window_start_slot:window_start_slot+12]",
    "input_window": 12,
    "aux_supervised_steps": [0, 10],
    "last_step_index": 11,
    "last_step_equals": "y_raw[0]",
    "channels": list(STATE_CHANNELS),
}


def _node_order(data_root: Path) -> list[str]:
    path = Path(data_root) / "node_mapping.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "edge_id" not in rows[0]:
        raise ReimplementationError(f"{path} has no edge_id column")
    return [str(row["edge_id"]) for row in rows]


def _target_table(normalization: dict[str, Any]) -> dict[str, dict[str, float]]:
    target = normalization.get("target") or {}
    table = {}
    for name in STATE_CHANNELS:
        entry = target.get(name) or {}
        if "mean" not in entry or "std" not in entry:
            raise ReimplementationError(f"state_normalization.json target.{name} has no mean/std")
        table[name] = {"mean": entry["mean"], "std": entry["std"]}
    return table


def _day_counts(starts: np.ndarray, days: list[str]) -> list[int]:
    boundaries = np.flatnonzero(np.asarray(starts) == 0)
    if len(boundaries) != len(days) or (len(boundaries) and int(boundaries[0]) != 0):
        raise ReimplementationError(
            f"training windows do not start a new day at slot 0 for each of {len(days)} dates "
            f"(found {len(boundaries)} slot-0 rows)"
        )
    ends = [int(item) for item in boundaries[1:]] + [int(starts.shape[0])]
    counts = []
    for begin, end in zip((int(item) for item in boundaries), ends):
        expected = np.arange(end - begin)
        if not np.array_equal(np.asarray(starts[begin:end]), expected):
            raise ReimplementationError(f"window_start_slot[{begin}:{end}] is not 0..{end - begin - 1}")
        counts.append(end - begin)
    return counts


class TemporalHistoryLabelDataset(Dataset):
    """Delegate the 12->1 sample and add ``y_seq_raw`` of shape ``[12, N, 4]``."""

    def __init__(self, base) -> None:
        if getattr(base, "split", None) != "train":
            raise ReimplementationError("temporal history labels attach to the training split only")
        self.base = base
        self.data_root = Path(base.data_root)
        self.split = str(base.split)
        self.rate = str(base.rate)
        self.input_type = str(base.input_type)
        self.target_type = str(base.target_type)
        manifest_days = [str(day) for day in load_json(self.data_root / "split_manifest.json")[self.split]]
        counts = _day_counts(np.asarray(base.window_start_slot), manifest_days)
        if sum(counts) != len(base):
            raise ReimplementationError("day window counts do not cover the training dataset")
        self.days = manifest_days
        self.counts = counts
        self.sample_day = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
        self.sequence_root = shared_sequence_root(self.data_root)
        self._maps: list[np.ndarray] | None = None
        self._pid = -1
        self.audit = self._audit()

    def __getattr__(self, name: str):
        return getattr(self.base, name)

    def __len__(self) -> int:
        return len(self.base)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_maps"] = None
        state["_pid"] = -1
        return state

    def _open(self) -> list[np.ndarray]:
        if self._maps is None or self._pid != os.getpid():
            self._maps = [_member(self.sequence_root / self.split / f"{day}.npz", "full_raw") for day in self.days]
            self._pid = os.getpid()
        return self._maps

    def _audit(self) -> dict[str, Any]:
        normalization_path = self.data_root / "state_normalization.json"
        normalization = load_json(normalization_path)
        current_norm_sha = sha256_file(normalization_path)
        if normalization.get("physical_definition_version") != PHYSICAL_DEFINITION_VERSION:
            raise ReimplementationError(
                f"{normalization_path} physical_definition_version="
                f"{normalization.get('physical_definition_version')!r}"
            )
        target = _target_table(normalization)
        manifest_path = self.sequence_root / "sequence_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        recorded_norm = (manifest.get("normalization") or {}).get("sha256")
        recorded_target = (manifest.get("normalization") or {}).get("target")
        if manifest:
            if manifest.get("physical_definition_version") != PHYSICAL_DEFINITION_VERSION:
                raise ReimplementationError(
                    f"sequence manifest physical version {manifest.get('physical_definition_version')!r} "
                    f"!= {PHYSICAL_DEFINITION_VERSION}"
                )
            if tuple(manifest.get("channels") or ()) != STATE_CHANNELS:
                raise ReimplementationError(f"sequence manifest channels {manifest.get('channels')}")
            if recorded_target != target:
                raise ReimplementationError(
                    "sequence manifest target mean/std differ from this dataset's state_normalization.json"
                )
        node_order = _node_order(self.data_root)
        if len(node_order) != int(self.base.y_raw.shape[-2]):
            raise ReimplementationError(
                f"node_mapping has {len(node_order)} roads, labels have {self.base.y_raw.shape[-2]}"
            )
        node_sha = hashlib.sha256("\n".join(node_order).encode("utf-8")).hexdigest()
        manifest_node = (manifest.get("node_order") or {}).get("edge_ids_sha256")
        y_last = np.asarray(self.base.y_raw[:, 0])
        days = []
        raw_digest = hashlib.sha256()
        file_digest = hashlib.sha256()
        offset = 0
        for day, count in zip(self.days, self.counts):
            path = self.sequence_root / self.split / f"{day}.npz"
            if not path.is_file():
                raise ReimplementationError(f"missing sequence labels {path}")
            file_sha = sha256_file(path)
            with np.load(path, allow_pickle=False) as payload:
                full = np.array(payload["full_raw"], dtype=np.float32, copy=True)
                edge_ids = np.array(payload["edge_ids"])
                channels = tuple(str(item) for item in np.asarray(payload["channels"]).tolist())
                date = str(np.asarray(payload["date"]))
                split = str(np.asarray(payload["split"]))
            if date != day or split != self.split:
                raise ReimplementationError(f"{path} metadata is {split}/{date}, expected {self.split}/{day}")
            if channels != STATE_CHANNELS:
                raise ReimplementationError(f"{path} channels {channels}")
            if tuple(full.shape) != (288, len(node_order), 4) or full.dtype != np.float32:
                raise ReimplementationError(f"{path} full_raw {full.shape} {full.dtype}")
            if not np.isfinite(full).all():
                raise ReimplementationError(f"{path} full_raw is not finite")
            edges = [str(item) for item in edge_ids.tolist()]
            if edges != node_order:
                raise ReimplementationError(f"{path} edge_ids differ from node_mapping.csv")
            expected = full[11:11 + count]
            got = y_last[offset:offset + count]
            if got.shape != expected.shape or not np.array_equal(got, expected):
                raise ReimplementationError(f"{day} sequence step 11 does not match training y_raw[:, 0]")
            raw_sha = hashlib.sha256(np.ascontiguousarray(full).tobytes()).hexdigest()
            recorded = (manifest.get("dates") or {}).get(day) or {}
            recorded_sha = recorded.get("sha256")
            if recorded_sha is not None and recorded_sha != file_sha:
                raise ReimplementationError(
                    f"{path} sha256 {file_sha} != sequence manifest {recorded_sha}"
                )
            entry = {
                "date": day,
                "file": f"{self.split}/{day}.npz",
                "file_sha256": file_sha,
                "manifest_file_sha256": recorded_sha,
                "full_raw_sha256": raw_sha,
                "source_csv_sha256": recorded.get("source_csv_sha256"),
                "shape": [288, len(node_order), 4],
                "finite": True,
                "last_step_matches_y_raw": True,
            }
            days.append(entry)
            raw_digest.update(f"{day}:{raw_sha}\n".encode("utf-8"))
            file_digest.update(f"{day}:{file_sha}\n".encode("utf-8"))
            offset += count
        if manifest_node is not None and manifest_node != node_sha:
            raise ReimplementationError(
                f"node order sha256 {node_sha} != sequence manifest {manifest_node}"
            )
        return {
            "sequence_root": self.sequence_root.as_posix(),
            "train_days": list(self.days),
            "physical_definition_version": PHYSICAL_DEFINITION_VERSION,
            "node_order_sha256": node_sha,
            "manifest_node_order_sha256": manifest_node,
            "target_mean": [target[name]["mean"] for name in STATE_CHANNELS],
            "target_std": [target[name]["std"] for name in STATE_CHANNELS],
            "target_scaler": target,
            "state_normalization_sha256": current_norm_sha,
            "manifest_state_normalization_sha256": recorded_norm,
            "normalization_file_hash_matches_manifest": recorded_norm == current_norm_sha,
            "slice": dict(SLICE_DEFINITION),
            "full_raw_content_sha256": raw_digest.hexdigest(),
            "file_content_sha256": file_digest.hexdigest(),
            "days": days,
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        index = int(index)
        sample = dict(self.base[index])
        start = int(sample["window_start_slot"])
        full = self._open()[int(self.sample_day[index])]
        window = np.array(full[start:start + 12], dtype=np.float32, copy=True)
        if window.shape != (12, full.shape[1], 4):
            raise ReimplementationError(f"sequence window {window.shape} at index {index} slot {start}")
        sample["y_seq_raw"] = torch.from_numpy(window)
        return sample


def attach_temporal_history_labels(dataset) -> tuple[TemporalHistoryLabelDataset, dict[str, Any]]:
    wrapped = TemporalHistoryLabelDataset(dataset)
    return wrapped, wrapped.audit
