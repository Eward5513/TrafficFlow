"""Deterministic variable-penetration schedules and depart-group sampling.

Target rates are continuous draws from U(p_min, p_max). A vehicle is kept or
dropped once, using the segment that contains its actual departure. Later
road or 5-minute bins do not resample it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from vehicle_sampling import daily_seed, permute

DATE_PATTERN = re.compile(r"^\d{4}_\d{2}_\d{2}$")

ALGORITHM_VERSION = "xuancheng26_variable_penetration_v1"
SCHEMA_VERSION = "xuancheng26_variable_windows_v1"
DAY_SECONDS = 86400
SEGMENT_SECONDS = {"30min": 1800, "1h": 3600, "2h": 7200, "day": 86400}


class PopulationConflict(RuntimeError):
    """Trip departures do not match the within-day sampling口径."""


def stable_unit(algorithm_version: str, schedule_seed: int, mode: str, date: str, segment_index: int) -> float:
    """Uniform draw in [0, 1). Independent of process count, hash randomization, and date order."""
    payload = "\0".join((algorithm_version, str(int(schedule_seed)), mode, date, str(int(segment_index))))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def segment_bounds(segment_seconds: int, day_seconds: int = DAY_SECONDS) -> list[tuple[int, int, int]]:
    if segment_seconds <= 0 or day_seconds % segment_seconds != 0:
        raise ValueError(f"segment_seconds must divide {day_seconds}, got {segment_seconds}")
    count = day_seconds // segment_seconds
    return [(index, index * segment_seconds, (index + 1) * segment_seconds) for index in range(count)]


def segment_index_for(depart: float, segment_seconds: int, day_seconds: int = DAY_SECONDS) -> int:
    """Left-closed, right-open. A departure at a boundary belongs to the later segment."""
    if depart < 0 or depart >= day_seconds:
        raise ValueError(f"depart {depart} is outside [0, {day_seconds})")
    return int(depart // segment_seconds)


def map_probability(u: float, p_min: float, p_max: float) -> float:
    probability = float(p_min) + (float(p_max) - float(p_min)) * float(u)
    if probability < p_min:
        probability = float(p_min)
    if probability > p_max:
        probability = float(p_max)
    return probability


def canonical_json(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "schedule_sha256"}
    return json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def schedule_sha256(payload: dict) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def validate_dates(dates: list[str]) -> list[str]:
    """Reject blank, duplicate, or non ``YYYY_MM_DD`` labels."""
    if not dates:
        raise ValueError("dates must be non-empty")
    cleaned = [str(item) for item in dates]
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"duplicate dates: {cleaned}")
    illegal = [item for item in cleaned if not DATE_PATTERN.fullmatch(item)]
    if illegal:
        raise ValueError(f"illegal dates: {illegal}")
    return cleaned


def validate_sampling_seeds(seeds: list[int], *, expected: tuple[int, ...] | None = None) -> list[int]:
    """Seeds must be unique non-negative integers. ``expected`` pins the official list."""
    try:
        values = [int(item) for item in seeds]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"illegal sampling seeds: {seeds}") from exc
    if any(item < 0 or item > 2**31 - 1 for item in values):
        raise ValueError(f"illegal sampling seeds: {values}")
    if len(values) != len(set(values)) or values != sorted(values):
        raise ValueError(f"sampling seeds must be strictly increasing and unique: {values}")
    if expected is not None and values != [int(item) for item in expected]:
        raise ValueError(f"sampling seeds must be {list(expected)}, got {values}")
    return values


def validate_dataset_spec(dataset_id: str, mode: str, p_min: float, p_max: float, segment_seconds: int) -> str:
    """Return the observation key. The dataset id and segment length must match the range."""
    from data_root import observation_key, variable_dataset_id

    key = observation_key(p_min, p_max)
    expected = variable_dataset_id(mode, p_min, p_max)
    if str(dataset_id) != expected:
        raise ValueError(f"dataset_id {dataset_id} != {expected}")
    expected_seconds = SEGMENT_SECONDS.get(mode)
    if expected_seconds is None or int(segment_seconds) != expected_seconds:
        raise ValueError(f"{mode} must use {expected_seconds} second segments, got {segment_seconds}")
    return key


def validate_generator_config(payload: dict) -> dict:
    """Check one range config. Observation keys come only from ``observation_key``."""
    from data_root import SAMPLING_SEEDS, TRAIN_SAMPLING_SEED

    low = float(payload["p_min"])
    high = float(payload["p_max"])
    if payload.get("algorithm_version") != ALGORITHM_VERSION:
        raise ValueError(f"algorithm_version {payload.get('algorithm_version')!r} != {ALGORITHM_VERSION}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version {payload.get('schema_version')!r} != {SCHEMA_VERSION}")
    seeds = validate_sampling_seeds(list(payload["sampling_seeds"]), expected=SAMPLING_SEEDS)
    payload["sampling_seeds"] = seeds
    if int(payload["fit_sampling_seed"]) != int(TRAIN_SAMPLING_SEED):
        raise ValueError("fit_sampling_seed stays 42; it is not the model seed")
    if int(payload["schedule_seed"]) != 42:
        raise ValueError("schedule_seed stays 42; it does not resample vehicles")
    if int(payload["fit_sampling_seed"]) not in seeds:
        raise ValueError("fit_sampling_seed is not one of sampling_seeds")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("config datasets must be a non-empty list")
    seen: set[str] = set()
    for item in datasets:
        key = validate_dataset_spec(
            str(item["dataset_id"]), str(item["mode"]), low, high, int(item["segment_seconds"]),
        )
        if item["dataset_id"] in seen:
            raise ValueError(f"duplicate dataset_id {item['dataset_id']}")
        seen.add(item["dataset_id"])
        item["observation_key"] = key
        item["p_min"] = low
        item["p_max"] = high
    if payload.get("dates") is not None:
        payload["dates"] = validate_dates(list(payload["dates"]))
    return payload


def verify_depart_prefix(
    eligible_ids: list[str],
    depart_of: dict[str, float],
    *,
    date: str,
    sampling_seed: int,
    segments: list[dict],
    segment_seconds: int,
    chosen: list[str],
    p_min: float | None = None,
    p_max: float | None = None,
) -> dict:
    """Check ``chosen`` against one permutation's depart-group prefixes.

    Day mode also derives the ``p_min`` and ``p_max`` prefixes in memory.
    Segment modes do not inherit that day-level nesting.
    """
    ordered = list(eligible_ids)
    if ordered != sorted(ordered):
        raise RuntimeError("eligible ids must already be sorted")
    rows = sorted(segments, key=lambda item: int(item["segment_index"]))
    derived = daily_seed(date, sampling_seed)
    permutation = permute(ordered, derived)
    chosen_sorted = sorted(chosen)
    if len(rows) == 1 and int(segment_seconds) == DAY_SECONDS:
        expected, selected, ratio = take_prefix(permutation, rows[0]["p"])
        if chosen_sorted != sorted(expected):
            raise RuntimeError(f"{date} seed {sampling_seed} day sample is not the permutation prefix")
        if p_min is not None and p_max is not None:
            low, _low_n, _low_ratio = take_prefix(permutation, p_min)
            high, _high_n, _high_ratio = take_prefix(permutation, p_max)
            chosen_set = set(expected)
            if not set(low) <= chosen_set <= set(high):
                raise RuntimeError(
                    f"{date} seed {sampling_seed} day sample is outside the in-memory endpoint prefixes"
                )
        return {"check": "day_prefix", "selected_count": selected, "realized_ratio": ratio, "derived_seed": derived}
    groups: list[list[str]] = [[] for _ in rows]
    for vehicle_id in permutation:
        groups[segment_index_for(depart_of[vehicle_id], segment_seconds)].append(vehicle_id)
    expected: list[str] = []
    checked = []
    for segment, group in zip(rows, groups):
        take, selected, ratio = take_prefix(group, segment["p"])
        if selected != (math.floor(len(group) * float(segment["p"])) if group else 0):
            raise RuntimeError(f"{date} segment {segment['segment_index']} prefix length drifted")
        expected.extend(take)
        checked.append({
            "segment_index": int(segment["segment_index"]),
            "eligible_count": len(group),
            "selected_count": selected,
            "realized_ratio": ratio,
        })
    if len(expected) != len(set(expected)):
        raise RuntimeError(f"{date} seed {sampling_seed} segment sample has duplicate ids")
    if chosen_sorted != sorted(expected):
        raise RuntimeError(f"{date} seed {sampling_seed} segment sample is not the depart-group prefix")
    return {"check": "segment_prefix", "segments": checked, "derived_seed": derived}


def build_schedule(
    *,
    dataset_id: str,
    mode: str,
    p_min: float,
    p_max: float,
    segment_seconds: int,
    schedule_seed: int,
    dates: list[str],
    observation_key: str,
    algorithm_version: str = ALGORITHM_VERSION,
) -> dict:
    key = validate_dataset_spec(dataset_id, mode, p_min, p_max, segment_seconds)
    if observation_key != key:
        raise ValueError(f"observation_key {observation_key} != {key}")
    dates = validate_dates(list(dates))
    if int(schedule_seed) < 0:
        raise ValueError(f"illegal schedule_seed {schedule_seed}")
    bounds = segment_bounds(segment_seconds)
    segments = []
    for date in sorted(dates):
        for index, start, end in bounds:
            draw = stable_unit(algorithm_version, schedule_seed, mode, date, index)
            probability = map_probability(draw, p_min, p_max)
            if not p_min <= probability <= p_max:
                raise RuntimeError(f"target p {probability} left [{p_min}, {p_max}]")
            segments.append({
                "date": date,
                "segment_index": index,
                "start_s": start,
                "end_s": end,
                "u": draw,
                "p": probability,
            })
    payload = {
        "algorithm_version": algorithm_version,
        "dataset_id": dataset_id,
        "mode": mode,
        "observation_key": observation_key,
        "distribution": "uniform_continuous",
        "p_min": float(p_min),
        "p_max": float(p_max),
        "p_mapping": "p_min + (p_max - p_min) * u",
        "schedule_seed": int(schedule_seed),
        "segment_seconds": int(segment_seconds),
        "day_seconds": DAY_SECONDS,
        "dates": sorted(dates),
        "segments": segments,
    }
    payload["schedule_sha256"] = schedule_sha256(payload)
    covered = {}
    for item in segments:
        covered.setdefault(item["date"], []).append((item["start_s"], item["end_s"]))
    for date, spans in covered.items():
        spans = sorted(spans)
        if spans[0][0] != 0 or spans[-1][1] != DAY_SECONDS:
            raise RuntimeError(f"{date} does not cover the day")
        for left, right in zip(spans, spans[1:]):
            if left[1] != right[0]:
                raise RuntimeError(f"{date} segments overlap or leave a gap")
    return payload


def segments_for_date(schedule: dict, date: str) -> list[dict]:
    rows = [dict(item) for item in schedule["segments"] if item["date"] == date]
    rows.sort(key=lambda item: item["segment_index"])
    expected = DAY_SECONDS // int(schedule["segment_seconds"])
    if len(rows) != expected or [item["segment_index"] for item in rows] != list(range(expected)):
        raise RuntimeError(f"{date} does not have {expected} schedule segments")
    return rows


def take_prefix(group: list[str], probability: float) -> tuple[list[str], int, float | None]:
    count = len(group)
    selected = math.floor(count * float(probability)) if count else 0
    if selected < 0 or selected > count:
        raise RuntimeError(f"prefix {selected} is outside 0..{count}")
    ratio = None if count == 0 else selected / count
    return list(group[:selected]), selected, ratio


def sample_permutation(
    permutation: list[str],
    depart_of: dict[str, float],
    segments: list[dict],
    *,
    segment_seconds: int,
) -> tuple[list[str], list[dict]]:
    """Keep each departure group's original relative order and take its prefix."""
    if len(segments) == 1 and int(segment_seconds) == DAY_SECONDS:
        chosen, selected, ratio = take_prefix(permutation, segments[0]["p"])
        return chosen, [{
            "segment_index": 0,
            "start_s": 0,
            "end_s": DAY_SECONDS,
            "target_p": segments[0]["p"],
            "eligible_count": len(permutation),
            "selected_count": selected,
            "realized_ratio": ratio,
        }]
    groups: list[list[str]] = [[] for _ in segments]
    for vehicle_id in permutation:
        groups[segment_index_for(depart_of[vehicle_id], segment_seconds)].append(vehicle_id)
    chosen: list[str] = []
    stats = []
    for segment, group in zip(segments, groups):
        take, selected, ratio = take_prefix(group, segment["p"])
        chosen.extend(take)
        stats.append({
            "segment_index": int(segment["segment_index"]),
            "start_s": int(segment["start_s"]),
            "end_s": int(segment["end_s"]),
            "target_p": segment["p"],
            "eligible_count": len(group),
            "selected_count": selected,
            "realized_ratio": ratio,
        })
    if len(chosen) != len(set(chosen)):
        raise RuntimeError("sampled ids are not unique")
    return chosen, stats


def sample_day(
    eligible_ids: list[str],
    depart_of: dict[str, float],
    *,
    date: str,
    sampling_seed: int,
    segments: list[dict],
    segment_seconds: int,
) -> tuple[list[str], list[dict], int]:
    ordered = list(eligible_ids)
    if ordered != sorted(ordered):
        raise RuntimeError("eligible ids must already be sorted")
    segments = sorted(segments, key=lambda item: int(item["segment_index"]))
    seed = daily_seed(date, sampling_seed)
    permutation = permute(ordered, seed)
    chosen, stats = sample_permutation(
        permutation, depart_of, segments, segment_seconds=segment_seconds,
    )
    return chosen, stats, seed


def population_from_tripinfo(path: Path) -> dict:
    """Eligible vehicles are those with a real departure inside [0, 86400)."""
    import gzip
    from xml.parsers import expat

    seen: dict[str, int] = {}
    depart_of: dict[str, float] = {}
    counts = {
        "rows": 0,
        "duplicate_ids": 0,
        "missing_depart": 0,
        "non_numeric_depart": 0,
        "undeparted": 0,
        "within_day": 0,
        "depart_at_or_after_86400": 0,
        "missing_or_negative_arrival": 0,
        "arrival_crosses_2h_boundary": 0,
        "depart_exactly_7200": 0,
    }
    examples: list[dict] = []

    def start(name: str, attrs) -> None:
        if name != "tripinfo":
            return
        counts["rows"] += 1
        vehicle_id = attrs.get("id", "")
        if not vehicle_id:
            counts["missing_depart"] += 1
            return
        seen[vehicle_id] = seen.get(vehicle_id, 0) + 1
        if seen[vehicle_id] > 1:
            counts["duplicate_ids"] += 1
            return
        if "depart" not in attrs or attrs.get("depart", "") == "":
            counts["missing_depart"] += 1
            return
        try:
            depart = float(attrs["depart"])
        except ValueError:
            counts["non_numeric_depart"] += 1
            return
        if depart < 0:
            counts["undeparted"] += 1
            return
        if depart >= DAY_SECONDS:
            counts["depart_at_or_after_86400"] += 1
            return
        counts["within_day"] += 1
        depart_of[vehicle_id] = depart
        if depart == 7200:
            counts["depart_exactly_7200"] += 1
        arrival_text = attrs.get("arrival", "")
        try:
            arrival = float(arrival_text) if arrival_text != "" else None
        except ValueError:
            arrival = None
        if arrival is None or arrival < 0:
            counts["missing_or_negative_arrival"] += 1
            return
        if int(depart // 7200) != int(min(arrival, DAY_SECONDS - 1e-9) // 7200) and len(examples) < 8:
            counts["arrival_crosses_2h_boundary"] += 1
            examples.append({"vehicle_id": vehicle_id, "depart": depart, "arrival": arrival})
        elif int(depart // 7200) != int(min(arrival, DAY_SECONDS - 1e-9) // 7200):
            counts["arrival_crosses_2h_boundary"] += 1

    parser = expat.ParserCreate()
    parser.StartElementHandler = start
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as handle:
        parser.ParseFile(handle)
    conflicts = []
    if counts["duplicate_ids"]:
        conflicts.append("duplicate_ids")
    if counts["missing_depart"] or counts["non_numeric_depart"]:
        conflicts.append("illegal_depart")
    if counts["depart_at_or_after_86400"]:
        conflicts.append("depart_outside_day")
    return {
        "ids": sorted(depart_of),
        "depart": depart_of,
        "all_ids": set(seen),
        "counts": counts,
        "conflicts": conflicts,
        "cross_boundary_examples": examples,
    }
