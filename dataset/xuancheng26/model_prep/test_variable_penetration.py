"""Schedule, boundary, and observation-key checks. No SUMO and no full FCD."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_root import observation_key, resolve_xuancheng_dataset, variable_dataset_id
from reimplementation.common.data.r_only_npz_dataset import bind_observation_keys, rate_tag
from variable_penetration import (
    ALGORITHM_VERSION,
    build_schedule,
    sample_day,
    schedule_sha256,
    segment_index_for,
    stable_unit,
)


def test_observation_key_accepts_decimal_bounds() -> None:
    assert observation_key(0.10, 0.20) == "p10_p20"
    assert variable_dataset_id("day", 0.10, 0.20) == "variable_day_p10_p20"
    assert variable_dataset_id("2h", 0.125, 0.175) == "variable_2h_p12p5_p17p5"
    with pytest.raises(Exception):
        observation_key(0.20, 0.10)


def test_schedule_is_stable_and_covers_the_day() -> None:
    dates = ["2023_04_02", "2023_04_01"]
    kwargs = dict(dataset_id="variable_2h_p10_p20", mode="2h", p_min=0.10, p_max=0.20,
                  segment_seconds=7200, schedule_seed=42, observation_key="p10_p20")
    left = build_schedule(dates=dates, **kwargs)
    right = build_schedule(dates=list(reversed(dates)), **kwargs)
    assert left == right
    assert left["schedule_sha256"] == schedule_sha256(left)
    assert len(left["segments"]) == 2 * 12
    assert all(0.10 <= item["p"] <= 0.20 for item in left["segments"])
    assert segment_index_for(7199.9, 7200) == 0
    assert segment_index_for(7200.0, 7200) == 1
    assert segment_index_for(86399.9, 7200) == 11
    with pytest.raises(ValueError):
        segment_index_for(86400, 7200)
    again = stable_unit(ALGORITHM_VERSION, 42, "2h", "2023_04_01", 1)
    assert again == next(item["u"] for item in left["segments"] if item["date"] == "2023_04_01" and item["segment_index"] == 1)


def test_sampling_prefixes_and_boundary_identity() -> None:
    ids = [f"v{i:03d}" for i in range(100)]
    depart = {vehicle_id: (index * 1000) % 86400 for index, vehicle_id in enumerate(ids)}
    depart["v010"] = 7199.0
    depart["v011"] = 7200.0
    day = build_schedule(dataset_id="variable_day_p10_p20", mode="day", p_min=0.10, p_max=0.20,
                         segment_seconds=86400, schedule_seed=42, dates=["2023_04_01"], observation_key="p10_p20")
    chosen, stats, _seed = sample_day(ids, depart, date="2023_04_01", sampling_seed=42,
                                      segments=day["segments"], segment_seconds=86400)
    assert stats[0]["selected_count"] == len(chosen) == int(100 * day["segments"][0]["p"])
    two = build_schedule(dataset_id="variable_2h_p10_p20", mode="2h", p_min=0.10, p_max=0.20,
                         segment_seconds=7200, schedule_seed=42, dates=["2023_04_01"], observation_key="p10_p20")
    chosen_2h, stats_2h, _seed = sample_day(ids, depart, date="2023_04_01", sampling_seed=42,
                                            segments=two["segments"], segment_seconds=7200)
    assert "v010" in chosen_2h or stats_2h[0]["selected_count"] <= stats_2h[0]["eligible_count"]
    assert sum(item["selected_count"] for item in stats_2h) == len(chosen_2h) == len(set(chosen_2h))
    empty, empty_stats, _seed = sample_day([], {}, date="2023_04_01", sampling_seed=42,
                                           segments=day["segments"], segment_seconds=86400)
    assert empty == [] and empty_stats[0]["realized_ratio"] is None
    repeat, _, _ = sample_day(ids, depart, date="2023_04_01", sampling_seed=43,
                              segments=two["segments"], segment_seconds=7200)
    other_order, _, _ = sample_day(list(ids), dict(depart), date="2023_04_01", sampling_seed=43,
                                   segments=list(reversed(two["segments"])), segment_seconds=7200)
    assert repeat == other_order


def test_rate_tag_does_not_parse_ranges_as_fixed_penetrations(tmp_path: Path) -> None:
    bind_observation_keys(None)
    assert rate_tag("p05") == "p05"
    assert rate_tag(10) == "p10"
    with pytest.raises(ValueError):
        rate_tag("p15")
    with pytest.raises(ValueError):
        rate_tag("p10_p20")
    assert rate_tag("p10_p20", allowed=("p10_p20",)) == "p10_p20"
    with pytest.raises(ValueError):
        rate_tag("p15", allowed=("p10_p20",))
    root = tmp_path / "training_data_3s"
    dataset = root / "datasets" / "variable_day_p10_p20"
    (dataset / "sampling_seeds").mkdir(parents=True)
    (dataset / "dataset_manifest.json").write_text(json.dumps({
        "dataset_id": "variable_day_p10_p20",
        "observation_keys": ["p10_p20"],
    }), encoding="utf-8")
    (dataset / "penetration_schedule.json").write_text(json.dumps({"schedule_sha256": "abc"}), encoding="utf-8")
    spec = resolve_xuancheng_dataset(dataset_id="variable_day_p10_p20", training_root=root)
    assert spec["observation_keys"] == ("p10_p20",)
    assert spec["schedule_sha256"] == "abc"
    with pytest.raises(Exception):
        resolve_xuancheng_dataset(
            dataset_id="variable_day_p10_p20",
            sampling_root=root / "sampling_seeds",
            training_root=root,
        )
    bind_observation_keys(None)
