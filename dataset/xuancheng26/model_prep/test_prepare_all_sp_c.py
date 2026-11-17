#!/usr/bin/env python3
"""Spec, path, and raw-speed checks for prepare_all_sp_c.py.

The fixture does not need the production data disk. A mounted training tree
adds a read-only identity check and does not rewrite published files.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[3]
PREP = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(PREP))

import prepare_all_sp_c as prep  # noqa: E402
import prepare_rf_features as rf  # noqa: E402

PRODUCTION = Path("/data5/tzhang174/TrafficFlow/xuancheng26/training_data_3s")


def _manifest(dataset_id: str, key: str, **overrides) -> dict:
    config = prep.load_range_configs().get(dataset_id)
    payload = {
        "dataset_id": dataset_id,
        "observation_keys": [key],
        "sampling_seeds": [42, 43, 44, 45, 46],
        "physical_definition_version": "independent_fcd_v1",
        "mode": config["entry"]["mode"] if config else "30min",
        "segment_seconds": int(config["entry"]["segment_seconds"]) if config else 1800,
        "schedule_seed": 42,
        "schedule_sha256": "fixture-sha",
        "kind": "variable",
        "dataset_status": "ready",
    }
    payload.update(overrides)
    return payload


def _write_dataset(root: Path, dataset_id: str, key: str, **overrides) -> None:
    dataset = root / "datasets" / dataset_id
    dataset.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(dataset_id, key, **overrides)
    (dataset / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (dataset / "penetration_schedule.json").write_text(
        json.dumps({"schedule_sha256": manifest["schedule_sha256"]}),
        encoding="utf-8",
    )


class SpecPathTest(unittest.TestCase):
    def test_help_and_default_ids(self) -> None:
        parser = prep.build_parser()
        help_text = parser.format_help()
        self.assertIn("--dataset-ids", help_text)
        self.assertIn("--training-root", help_text)
        self.assertIn("--force", help_text)
        args = parser.parse_args([])
        self.assertIsNone(args.dataset_ids)
        self.assertEqual(prep.selected_dataset_ids(args), list(prep.DATASET_IDS))
        mixed = parser.parse_args([
            "--dataset-ids",
            "variable_30min_p5_p15",
            "variable_30min_p5_p15",
            "variable_1h_p5_p15",
            "variable_30min_p10_p20",
        ])
        self.assertEqual(
            prep.selected_dataset_ids(mixed),
            ["variable_30min_p5_p15", "variable_1h_p5_p15", "variable_30min_p10_p20"],
        )

    def test_specs_keep_each_range_on_its_own_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_dataset(root, "variable_30min_p10_p20", "p10_p20")
            _write_dataset(root, "variable_30min_p5_p15", "p5_p15")
            _write_dataset(root, "variable_1h_p5_p15", "p5_p15")
            p10 = prep.resolve_dataset_specs(root, ["variable_30min_p10_p20"])
            p5 = prep.resolve_dataset_specs(root, ["variable_30min_p5_p15"])
            mixed = prep.resolve_dataset_specs(root, [
                "variable_30min_p5_p15",
                "variable_30min_p10_p20",
                "variable_30min_p5_p15",
                "variable_1h_p5_p15",
            ])
            self.assertEqual([item["observation_key"] for item in p10], ["p10_p20"])
            self.assertEqual([item["observation_key"] for item in p5], ["p5_p15"])
            self.assertEqual(
                [(item["dataset_id"], item["observation_key"]) for item in mixed],
                [
                    ("variable_30min_p5_p15", "p5_p15"),
                    ("variable_30min_p10_p20", "p10_p20"),
                    ("variable_1h_p5_p15", "p5_p15"),
                ],
            )
            day = "2023_04_01"
            narrow = prep.planned_paths(p5[0], 42, day, "train")
            wide = prep.planned_paths(p10[0], 42, day, "train")
            mixed_narrow = prep.planned_paths(mixed[0], 42, day, "train")
            mixed_wide = prep.planned_paths(mixed[1], 42, day, "train")
            self.assertEqual(narrow, mixed_narrow)
            self.assertEqual(wide, mixed_wide)
            self.assertNotEqual(narrow["members"], wide["members"])
            self.assertNotEqual(narrow["daily"], wide["daily"])
            self.assertNotEqual(narrow["scaler"], wide["scaler"])
            self.assertNotEqual(narrow["window_all_sp_c"], wide["window_all_sp_c"])
            for path in narrow.values():
                self.assertNotIn("p10_p20", Path(path).parts)
                self.assertIn("variable_30min_p5_p15", Path(path).parts)
            self.assertIn("p5_p15", narrow["members"].parts)
            self.assertIn("p10_p20", wide["members"].parts)
            self.assertEqual(p5[0]["sampling_seeds"], prep.SEEDS)
            self.assertEqual(p5[0]["manifest"]["physical_definition_version"], "independent_fcd_v1")

    def test_manifest_must_declare_one_key_and_the_physical_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_dataset(
                root,
                "variable_30min_p5_p15",
                "p5_p15",
                observation_keys=["p5_p15", "p10_p20"],
            )
            with self.assertRaises(rf.FeatureError):
                prep.resolve_dataset_specs(root, ["variable_30min_p5_p15"])
            _write_dataset(
                root,
                "variable_30min_p5_p15",
                "p5_p15",
                physical_definition_version="other",
            )
            with self.assertRaises(rf.FeatureError):
                prep.resolve_dataset_specs(root, ["variable_30min_p5_p15"])


class RawSpeedTest(unittest.TestCase):
    def test_snapshot_mean_filters_and_two_vehicle_sets(self) -> None:
        xml = """<fcd-export>
<timestep time="0.0">
  <vehicle id="veh_20230401_000000001" lane="laneA" speed="10"/>
  <vehicle id="veh_20230401_000000001" lane="laneA" speed="0"/>
  <vehicle id="veh_20230401_000000002" lane="laneA" speed="20"/>
  <vehicle id="veh_20230401_000000003" lane="laneB" speed="-1"/>
  <vehicle id="veh_20230401_000000003" lane="laneB" speed="nan"/>
  <vehicle id="veh_20230401_000000003" lane="laneB" speed="inf"/>
  <vehicle id="veh_20230401_000000004" lane="internalA" speed="5"/>
</timestep>
<timestep time="299.9">
  <vehicle id="veh_20230401_000000001" lane="laneA" speed="10"/>
</timestep>
<timestep time="300.0">
  <vehicle id="veh_20230401_000000001" lane="laneA" speed="50"/>
  <vehicle id="veh_20230401_000000004" lane="laneB" speed="0"/>
</timestep>
<timestep time="86400.0">
  <vehicle id="veh_20230401_000000002" lane="laneA" speed="1"/>
</timestep>
</fcd-export>
"""
        lane_to_index = {"laneA": 0, "laneB": 1}
        base = np.asarray([10.0, 10.0], dtype=np.float64)
        body = [xml.encode("utf-8")]

        def parse(groups: list[list[int]]):
            sums, counts, stats = rf.parse_fcd_stream(
                body,
                mask=rf.build_mask(groups),
                lane_to_index=lane_to_index,
                internal_lanes={"internalA"},
                base_ms=base,
                n_rates=len(groups),
                n_edges=2,
                until_time=None,
                progress_label="fixture",
            )
            self.assertFalse(stats["stopped_early"])
            return [prep.raw_speed_and_count(sum_row, count_row) for sum_row, count_row in zip(sums, counts)], stats

        alone, alone_stats = parse([[1, 2, 3]])
        mixed, mixed_stats = parse([[1, 2, 3], [4]])
        self.assertGreater(alone_stats["negative_speed"], 0)
        self.assertGreater(alone_stats["nonfinite_speed"], 0)
        self.assertGreater(alone_stats["extreme_speed"], 0)
        self.assertGreater(mixed_stats["internal_lane"], 0)
        speed, count = mixed[0]
        np.testing.assert_allclose(speed, alone[0][0])
        np.testing.assert_array_equal(count, alone[0][1])
        self.assertEqual(int(count[0, 0]), 4)
        self.assertAlmostEqual(float(speed[0, 0]), 36.0, places=5)
        self.assertEqual(int(count[1, 0]), 1)
        self.assertAlmostEqual(float(speed[1, 0]), 180.0, places=5)
        self.assertEqual(int(count[287, 0]), 1)
        self.assertAlmostEqual(float(speed[287, 0]), 3.6, places=5)
        self.assertEqual(int(count[0, 1]), 0)
        self.assertEqual(float(speed[0, 1]), 0.0)
        other_speed, other_count = mixed[1]
        self.assertEqual(int(other_count[1, 1]), 1)
        self.assertEqual(float(other_speed[1, 1]), 0.0)
        self.assertEqual(int(other_count[0, 0]), 0)
        self.assertTrue(np.all(other_speed[other_count == 0] == 0))
        self.assertTrue(np.all(count >= 0))
        self.assertEqual(count.dtype, np.int32)
        self.assertEqual(speed.dtype, np.float32)

    def test_history_uses_the_same_twelve_bins(self) -> None:
        speed = np.arange(rf.N_BINS, dtype=np.float32)[:, None] + np.zeros((1, prep.windows.N_NODES), dtype=np.float32)
        count = np.ones((rf.N_BINS, prep.windows.N_NODES), dtype=np.int32)
        count[0, 0] = 0
        speed[0, 0] = 0
        starts = np.arange(prep.windows.N_WINDOWS, dtype=np.int32)
        extra = prep.slice_extra(speed, count, starts)
        self.assertEqual(extra.shape, (prep.windows.N_WINDOWS, prep.windows.N_HIS, prep.windows.N_NODES, 2))
        np.testing.assert_array_equal(extra[0, :, 1, 0], np.arange(12, dtype=np.float32))
        np.testing.assert_array_equal(extra[1, :, 1, 0], np.arange(1, 13, dtype=np.float32))
        np.testing.assert_array_equal(extra[-1, :, 1, 0], np.arange(276, 288, dtype=np.float32))
        self.assertEqual(float(extra[0, 0, 0, 0]), 0.0)
        self.assertEqual(float(extra[0, 0, 0, 1]), 0.0)
        with self.assertRaises(rf.FeatureError):
            prep.slice_extra(speed, count, starts + 1)


class PublishedReadOnlyTest(unittest.TestCase):
    def test_mounted_ranges_do_not_share_lists_or_scalers(self) -> None:
        manifest = PRODUCTION / "datasets" / "variable_30min_p5_p15" / "dataset_manifest.json"
        if not manifest.is_file():
            self.skipTest("production training data is not mounted")
        specs = prep.resolve_dataset_specs(PRODUCTION, [
            "variable_30min_p5_p15",
            "variable_1h_p5_p15",
            "variable_2h_p5_p15",
            "variable_day_p5_p15",
            "variable_30min_p10_p20",
        ])
        self.assertEqual([item["observation_key"] for item in specs[:4]], ["p5_p15"] * 4)
        self.assertEqual(specs[4]["observation_key"], "p10_p20")
        day = "2023_04_01"
        p5 = prep.planned_paths(specs[0], 42, day, "train")
        p10 = prep.planned_paths(specs[4], 42, day, "train")
        self.assertTrue(p5["members"].is_file())
        self.assertTrue(p5["window_all"].is_file())
        self.assertTrue(p10["daily"].is_file())
        self.assertNotEqual(prep.sha256_file(p5["members"]), prep.sha256_file(p10["members"]))
        self.assertNotEqual(p5["scaler"], p10["scaler"])
        self.assertFalse(p5["daily"].is_file())
        with np.load(p10["daily"], mmap_mode="r", allow_pickle=False) as payload:
            self.assertEqual(str(payload["observation_key"]), "p10_p20")
            self.assertEqual(str(payload["dataset_id"]), "variable_30min_p10_p20")
            self.assertEqual(str(payload["definition_version"]), prep.DEFINITION_VERSION)
            count = payload["speed_observation_count"]
            speed = payload["average_speed_raw"]
            self.assertTrue(np.isfinite(speed).all())
            self.assertGreaterEqual(int(count.min()), 0)
            self.assertTrue(bool(np.all(speed[count == 0] == 0)))
        groups = prep.groups_for(
            PRODUCTION,
            day,
            split="train",
            edges=["placeholder"],
            specs=specs[:1],
            node_order_sha256="unchecked-until-load-edges",
        )
        self.assertEqual(len(groups), 5)
        self.assertTrue(all(group["key"] == "p5_p15" for group in groups))
        self.assertTrue(all("p10_p20" not in group["members"] for group in groups))
        self.assertEqual([group["sampling_seed"] for group in groups], list(prep.SEEDS))


if __name__ == "__main__":
    unittest.main()
