"""Assemble PDFormer input channels from prepared NPZ metadata.

Traffic values stay the already-normalized ``x``. Time-of-day is the fraction
of a 288-slot day (equivalent to original ``(ts - day) / 1 day``). Day-of-week
is used only when a real calendar mapping is supplied. ``day_index % 7`` is
rejected.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from reimplementation.common.data.r_only_npz_dataset import ROnlyNPZDataset, load_json
from reimplementation.common.errors import ReimplementationError

SLOTS_PER_DAY = 288
MINUTES_PER_DAY = 1440
MINUTES_PER_SLOT = 5


def slot_series(window_start_slot: np.ndarray | torch.Tensor, n_his: int) -> np.ndarray:
    starts = np.asarray(window_start_slot, dtype=np.int64).reshape(-1)
    offsets = np.arange(int(n_his), dtype=np.int64)
    return starts[:, None] + offsets[None, :]


def assert_same_day_window(window_start_slot: np.ndarray | torch.Tensor, n_his: int) -> np.ndarray:
    slots = slot_series(window_start_slot, n_his)
    if np.any(slots < 0) or np.any(slots >= SLOTS_PER_DAY):
        raise ReimplementationError(
            "input window crosses midnight or leaves [0, 288); PDFormer time-of-day "
            "cannot wrap. Prepared data already forbids cross-day windows."
        )
    return slots


def time_in_day_fraction(slots: np.ndarray) -> np.ndarray:
    """Original LibCity: minutes-since-midnight / 1440."""
    return (slots.astype(np.float32) * MINUTES_PER_SLOT) / float(MINUTES_PER_DAY)


def time_in_day_index(slots: np.ndarray) -> np.ndarray:
    minutes = np.rint(slots.astype(np.float64) * MINUTES_PER_SLOT).astype(np.int64)
    if np.any(minutes < 0) or np.any(minutes >= MINUTES_PER_DAY):
        raise ReimplementationError("time-of-day minute index is outside [0, 1440)")
    return minutes


def parse_weekday(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        weekday = int(value)
        if weekday < 0 or weekday > 6:
            raise ReimplementationError(f"weekday must be in [0, 6] (Monday=0), got {weekday}")
        return weekday
    text = str(value).strip()
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(text).date()
        except ValueError as exc:
            raise ReimplementationError(
                f"cannot parse calendar date {value!r}; refusing to invent a weekday"
            ) from exc
    return int(parsed.weekday())


def load_weekday_mapping(path: Path | None) -> dict[str, int] | None:
    if path is None:
        return None
    payload = load_json(Path(path))
    if not isinstance(payload, Mapping):
        raise ReimplementationError("weekday mapping must be a JSON object")
    mapping: dict[str, int] = {}
    for key, value in payload.items():
        mapping[str(key)] = parse_weekday(value)
    if not mapping:
        raise ReimplementationError("weekday mapping is empty")
    return mapping


def weekday_ids_from_mapping(
    day_index: np.ndarray | torch.Tensor,
    *,
    mapping: Mapping[str, int],
    all_days: Sequence[str],
) -> np.ndarray:
    indices = np.asarray(day_index, dtype=np.int64).reshape(-1)
    names = [str(item) for item in all_days]
    weekdays = np.empty(indices.shape, dtype=np.int64)
    for position, index in enumerate(indices.tolist()):
        if index < 0 or index >= len(names):
            raise ReimplementationError(f"day_index {index} is outside all_days")
        name = names[index]
        if name not in mapping and str(index) not in mapping:
            raise ReimplementationError(
                f"no calendar weekday for {name} (day_index={index}); "
                "refusing day_index % 7"
            )
        weekdays[position] = int(mapping.get(name, mapping.get(str(index))))
    return weekdays


def weekday_onehot(weekdays: np.ndarray, n_his: int, num_nodes: int) -> np.ndarray:
    batch = int(weekdays.shape[0])
    onehot = np.zeros((batch, n_his, num_nodes, 7), dtype=np.float32)
    for i, weekday in enumerate(weekdays.tolist()):
        onehot[i, :, :, int(weekday)] = 1.0
    return onehot


def assemble_model_input(
    x: np.ndarray | torch.Tensor,
    *,
    window_start_slot: np.ndarray | torch.Tensor,
    day_index: np.ndarray | torch.Tensor | None,
    n_his: int,
    add_time_in_day: bool,
    add_day_in_week: bool,
    weekday_mapping: Mapping[str, int] | None = None,
    all_days: Sequence[str] | None = None,
) -> torch.Tensor:
    if torch.is_tensor(x):
        traffic = x.detach().cpu().numpy()
    else:
        traffic = np.asarray(x)
    if traffic.ndim != 4:
        raise ReimplementationError(f"x must be [B,T,N,C], got {traffic.shape}")
    batch, time, nodes, channels = traffic.shape
    if int(time) != int(n_his):
        raise ReimplementationError(f"x time {time} != n_his {n_his}")
    if int(channels) != 1:
        raise ReimplementationError("R-only traffic channel count must be 1")
    slots = assert_same_day_window(window_start_slot, n_his)
    if slots.shape[0] != batch:
        raise ReimplementationError("window_start_slot batch does not match x")
    pieces = [traffic.astype(np.float32, copy=False)]
    if add_time_in_day:
        fractions = time_in_day_fraction(slots)
        time_channel = np.repeat(fractions[:, :, None, None], nodes, axis=2).astype(np.float32)
        pieces.append(time_channel)
    if add_day_in_week:
        if weekday_mapping is None:
            raise ReimplementationError(
                "add_day_in_week=true but no calendar weekday mapping was provided. "
                "Simulation days are labeled day_NN without dates; day_index % 7 is forbidden."
            )
        if all_days is None:
            raise ReimplementationError("weekday reconstruction needs split_manifest all_days")
        if day_index is None:
            raise ReimplementationError("weekday reconstruction needs per-sample day_index")
        weekdays = weekday_ids_from_mapping(day_index, mapping=weekday_mapping, all_days=all_days)
        pieces.append(weekday_onehot(weekdays, n_his, nodes))
    assembled = np.concatenate(pieces, axis=-1)
    return torch.from_numpy(np.array(assembled, copy=True))


class PDFormerBatchAdapter:
    """Wrap ``ROnlyNPZDataset`` items with PDFormer extra channels. Does not rewrite NPZ."""

    def __init__(
        self,
        dataset: ROnlyNPZDataset,
        *,
        n_his: int,
        add_time_in_day: bool,
        add_day_in_week: bool,
        weekday_mapping: Mapping[str, int] | None = None,
        all_days: Sequence[str] | None = None,
    ) -> None:
        self.dataset = dataset
        self.n_his = int(n_his)
        self.add_time_in_day = bool(add_time_in_day)
        self.add_day_in_week = bool(add_day_in_week)
        self.weekday_mapping = weekday_mapping
        self.all_days = None if all_days is None else [str(item) for item in all_days]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[index]
        model_x = assemble_model_input(
            sample["x"].unsqueeze(0),
            window_start_slot=sample["window_start_slot"].unsqueeze(0),
            day_index=sample["day_index"].unsqueeze(0),
            n_his=self.n_his,
            add_time_in_day=self.add_time_in_day,
            add_day_in_week=self.add_day_in_week,
            weekday_mapping=self.weekday_mapping,
            all_days=self.all_days,
        )
        sample["model_x"] = model_x.squeeze(0)
        return sample
