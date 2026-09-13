"""Per-step time-of-day and day-of-week features for R-only STAEformer.

Official PeMS04 data (Torch-MTS ``generate_training_data.py``, npz branch)::

    tod[i] = (i % 288) / 288
    dow[i] = (i // 288) % 7          # integer 0..6, not a fraction

The model then does ``(tod * 288).long()`` and ``dow.long()``.
That npz weekday is sequential day-of-series modulo 7, not calendar Monday.
R-only ``day_index`` is the same kind of 0-based simulation-day counter, so
the official PeMS04 formula is ``day_index % 7`` and is recorded as
``original_pems_npz_sequential_index_mod_7``.

Identity lookup uses integer slots so float32 ``slot/288*288`` truncation
cannot skip a row. Channel 1 of the assembled 3-feature tensor still stores
the official fraction for ``input_proj`` (official ``input_dim=3``).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from reimplementation.common.errors import ReimplementationError

SLOTS_PER_DAY = 288
DAY_OF_WEEK_SIZE = 7
ORIGINAL_PEMS_NPZ_DOW = "original_pems_npz_sequential_index_mod_7"
CALENDAR_DAY_SOURCE = "calendar"
ALLOWED_DAY_SOURCES = {ORIGINAL_PEMS_NPZ_DOW, CALENDAR_DAY_SOURCE}


def _as_int64_vector(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
    return np.asarray(value, dtype=np.int64).reshape(-1)


def slot_series(window_start_slot: np.ndarray | torch.Tensor, n_his: int) -> np.ndarray:
    starts = _as_int64_vector(window_start_slot)
    offsets = np.arange(int(n_his), dtype=np.int64)
    return starts[:, None] + offsets[None, :]


def assert_same_day_window(window_start_slot: np.ndarray | torch.Tensor, n_his: int) -> np.ndarray:
    slots = slot_series(window_start_slot, n_his)
    if np.any(slots < 0) or np.any(slots >= SLOTS_PER_DAY):
        raise ReimplementationError(
            "input window crosses midnight or leaves [0, 288); STAEformer "
            "time-of-day cannot wrap. Prepared data already forbids cross-day windows."
        )
    return slots


def assert_target_is_last_history(
    window_start_slot: np.ndarray | torch.Tensor,
    target_slot: np.ndarray | torch.Tensor,
    n_his: int,
) -> np.ndarray:
    slots = assert_same_day_window(window_start_slot, n_his)
    last = slots[:, -1]
    target = _as_int64_vector(target_slot)
    if last.shape != target.shape or not np.array_equal(last, target):
        raise ReimplementationError(
            "target_slot must equal window_start_slot + n_his - 1 "
            "(last observed step, not s+12)"
        )
    return target


def time_of_day_fraction(slots: np.ndarray) -> np.ndarray:
    return slots.astype(np.float32) / float(SLOTS_PER_DAY)


def long_index_from_fraction(fraction: np.ndarray | torch.Tensor, size: int) -> np.ndarray:
    """Official ``(tod * steps_per_day).long()`` truncation. Not used in production."""
    if torch.is_tensor(fraction):
        values = fraction.detach().to(dtype=torch.float32)
    else:
        values = torch.as_tensor(np.asarray(fraction), dtype=torch.float32)
    return (values * int(size)).long().cpu().numpy()


def sequential_day_of_week_index(day_index: np.ndarray | torch.Tensor) -> np.ndarray:
    indices = _as_int64_vector(day_index)
    if np.any(indices < 0):
        raise ReimplementationError("day_index must be non-negative")
    return indices % DAY_OF_WEEK_SIZE


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
    mapping = {str(key): parse_weekday(value) for key, value in payload.items()}
    if not mapping:
        raise ReimplementationError("weekday mapping is empty")
    return mapping


def calendar_day_of_week_index(
    day_index: np.ndarray | torch.Tensor,
    *,
    mapping: Mapping[str, int],
    all_days: Sequence[str],
) -> np.ndarray:
    indices = _as_int64_vector(day_index)
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
    if kind == ORIGINAL_PEMS_NPZ_DOW:
        return sequential_day_of_week_index(day_index)
    if weekday_mapping is None or all_days is None:
        raise ReimplementationError("calendar day-of-week requires weekday_mapping and all_days")
    return calendar_day_of_week_index(day_index, mapping=weekday_mapping, all_days=all_days)
