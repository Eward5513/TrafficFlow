"""Time-of-day and day-of-week identities for R-only STID.

Original PeMS preprocessing (``scripts/data_preparation/PEMS04/generate_training_data.py``)::

    time_of_day[i] = (i % 288) / 288
    day_of_week[i] = (i // 288) % 7 / 7

The model then recovers integer rows with truncation::

    index = (frac * size).type(torch.LongTensor)

This is **not** calendar Monday=0. The official PeMS script uses sequential
day-of-series modulo 7 from the start of the concatenated array. R-only
``day_index`` is the same kind of 0-based simulation-day counter, so the
original formula is ``day_index % 7`` and is recorded explicitly as
``original_sequential_index_mod_7``.

A separate ``calendar`` source exists for tests with known dates. It is not
the official default. Missing calendar dates raise; they are never invented.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from reimplementation.common.errors import ReimplementationError

SLOTS_PER_DAY = 288
DAY_OF_WEEK_SIZE = 7
ORIGINAL_DAY_SOURCE = "original_sequential_index_mod_7"
CALENDAR_DAY_SOURCE = "calendar"
ALLOWED_DAY_SOURCES = {ORIGINAL_DAY_SOURCE, CALENDAR_DAY_SOURCE}


def slot_series(window_start_slot: np.ndarray | torch.Tensor, n_his: int) -> np.ndarray:
    starts = np.asarray(window_start_slot, dtype=np.int64).reshape(-1)
    offsets = np.arange(int(n_his), dtype=np.int64)
    return starts[:, None] + offsets[None, :]


def assert_same_day_window(window_start_slot: np.ndarray | torch.Tensor, n_his: int) -> np.ndarray:
    slots = slot_series(window_start_slot, n_his)
    if np.any(slots < 0) or np.any(slots >= SLOTS_PER_DAY):
        raise ReimplementationError(
            "input window crosses midnight or leaves [0, 288); STID time-of-day "
            "cannot wrap. Prepared data already forbids cross-day windows."
        )
    return slots


def last_history_slot(window_start_slot: np.ndarray | torch.Tensor, n_his: int) -> np.ndarray:
    slots = assert_same_day_window(window_start_slot, n_his)
    return slots[:, -1]


def assert_target_is_last_history(
    window_start_slot: np.ndarray | torch.Tensor,
    target_slot: np.ndarray | torch.Tensor,
    n_his: int,
) -> np.ndarray:
    last = last_history_slot(window_start_slot, n_his)
    target = np.asarray(target_slot, dtype=np.int64).reshape(-1)
    if last.shape != target.shape or not np.array_equal(last, target):
        raise ReimplementationError(
            "target_slot must equal window_start_slot + n_his - 1 "
            "(last observed step, not s+12 / next-step forecasting)"
        )
    if np.any(target < 0) or np.any(target >= SLOTS_PER_DAY):
        raise ReimplementationError("target_slot is outside [0, 288)")
    return target


def time_of_day_fraction(slots: np.ndarray) -> np.ndarray:
    """Original: ``i % 288 / 288``. Slots are already in ``[0, 288)``."""
    return slots.astype(np.float32) / float(SLOTS_PER_DAY)


def long_index_from_fraction(
    fraction: np.ndarray | torch.Tensor,
    size: int,
) -> np.ndarray | torch.Tensor:
    """Match original ``(frac * size).type(torch.LongTensor)`` truncation."""
    if int(size) < 1:
        raise ReimplementationError("identity table size must be positive")
    if torch.is_tensor(fraction):
        return (fraction * int(size)).to(dtype=torch.long)
    array = np.asarray(fraction, dtype=np.float64) * int(size)
    return array.astype(np.int64)


def sequential_day_of_week_index(day_index: np.ndarray | torch.Tensor) -> np.ndarray:
    """Original PeMS: ``(i // 288) % 7``. Explicit, not a silent calendar guess."""
    indices = np.asarray(day_index, dtype=np.int64).reshape(-1)
    if np.any(indices < 0):
        raise ReimplementationError("day_index must be non-negative")
    return indices % DAY_OF_WEEK_SIZE


def sequential_day_of_week_fraction(day_index: np.ndarray | torch.Tensor) -> np.ndarray:
    return sequential_day_of_week_index(day_index).astype(np.float32) / float(DAY_OF_WEEK_SIZE)


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


def load_weekday_mapping(path: Any) -> dict[str, int]:
    from pathlib import Path

    from reimplementation.common.data.r_only_npz_dataset import load_json

    payload = load_json(Path(path))
    if not isinstance(payload, Mapping):
        raise ReimplementationError("weekday mapping must be a JSON object")
    mapping: dict[str, int] = {}
    for key, value in payload.items():
        mapping[str(key)] = parse_weekday(value)
    if not mapping:
        raise ReimplementationError("weekday mapping is empty")
    return mapping


def calendar_day_of_week_index(
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
                "refusing to invent day_index % 7 as Monday-Sunday"
            )
        weekdays[position] = parse_weekday(mapping.get(name, mapping.get(str(index))))
    if np.any(weekdays < 0) or np.any(weekdays > 6):
        raise ReimplementationError("calendar weekday is outside [0, 6]")
    return weekdays


def day_of_week_index(
    day_index: np.ndarray | torch.Tensor,
    *,
    source: str,
    weekday_mapping: Mapping[str, int] | None = None,
    all_days: Sequence[str] | None = None,
) -> np.ndarray:
    kind = str(source)
    if kind not in ALLOWED_DAY_SOURCES:
        raise ReimplementationError(f"unknown day_of_week_source {source!r}")
    if kind == ORIGINAL_DAY_SOURCE:
        return sequential_day_of_week_index(day_index)
    if weekday_mapping is None or all_days is None:
        raise ReimplementationError(
            "calendar day-of-week requires weekday_mapping and all_days; "
            "simulation days have no dates by default"
        )
    return calendar_day_of_week_index(day_index, mapping=weekday_mapping, all_days=all_days)


def day_of_week_fraction(
    day_index: np.ndarray | torch.Tensor,
    *,
    source: str,
    weekday_mapping: Mapping[str, int] | None = None,
    all_days: Sequence[str] | None = None,
) -> np.ndarray:
    return day_of_week_index(
        day_index,
        source=source,
        weekday_mapping=weekday_mapping,
        all_days=all_days,
    ).astype(np.float32) / float(DAY_OF_WEEK_SIZE)
