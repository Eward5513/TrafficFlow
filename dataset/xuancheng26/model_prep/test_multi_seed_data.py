"""Small regression fixtures; no production writes, GPU, or SUMO required."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import multi_seed_common as common
import generate_multi_seed_flow_data as flow
import vehicle_sampling as sampling


class MultiSeedTests(unittest.TestCase):
    def test_sampling_and_no_clobber(self):
        eligible = [f"v{i:04}" for i in range(100)]
        daily = sampling.daily_seed("2023_04_01", 43)
        first = sampling.permute(eligible, daily)
        self.assertEqual(first, sampling.permute(eligible, daily))
        sampling.validate_samples(eligible, sampling.sample_prefixes(first, len(eligible)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dest = root / "sampling_seeds/test.csv"
            with common.new_file(dest, root) as temp:
                temp.write_text("original", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                with common.new_file(dest, root):
                    self.fail("existing file must never be writable")
            self.assertEqual(dest.read_text(), "original")

    def test_history_resolver_does_not_reuse_wrong_seed_alias(self):
        root = Path("/tmp/training_data_3s")
        self.assertIsNone(common.historical(root, 43, "p10", "test", "2023_04_26", "window"))
        self.assertIsNone(common.historical(root, 43, "p05", "train", "2023_04_01", "window"))
        source = common.historical(root, 43, "p05", "test", "2023_04_26", "window")
        self.assertIsNone(source)

    def test_pending_rate_gaps_match_original_event_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            day = "2023_04_01"
            date_dir = root / day
            date_dir.mkdir()
            fcd = date_dir / f"fcd_{day}.xml"
            fcd.write_text(
                '<fcd-export><timestep time="0"><vehicle id="a" lane="A_0"/>'
                '<vehicle id="b" lane="A_0"/></timestep><timestep time="301">'
                '<vehicle id="a" lane=":i_0"/><vehicle id="b" lane="B_0"/></timestep>'
                '<timestep time="302"><vehicle id="a" lane="B_0"/></timestep></fcd-export>', encoding="utf-8")
            (date_dir / f"tripinfo_{day}.xml").write_text(
                '<tripinfos><tripinfo id="a" depart="0" arrival="600"/>'
                '<tripinfo id="b" depart="0" arrival="650"/></tripinfos>', encoding="utf-8")
            edges = ["A", "B"] + [f"edge{i}" for i in range(1404)]
            lanes = {"A_0": "A", "B_0": "B"}
            ids = {(s, r): (["a"] if i < 3 else ["a", "b"])
                   for s in common.SEEDS for i, r in enumerate(common.RATES)}
            pending = {(43, "p05"), (43, "p20"), (43, "p70"), (44, "p10"), (44, "p50")}
            with patch.dict(flow.CONTEXT, fcd=root, edges=edges, lanes=lanes, internal={":i_0"}):
                counts, stats = flow.count_pending(day, ids, pending)
            self.assertEqual(stats["scans"], 1)
            for seed, rate in pending:
                entry, exit_flow, _, _ = flow.full.stream_fcd(
                    fcd, edges, lanes, {":i_0"}, {"a": 600, "b": 650}, day, include=set(ids[seed, rate]))
                si, ri = common.SEEDS.index(seed), common.RATES.index(rate)
                np.testing.assert_array_equal(counts[0, si, ri], entry)
                np.testing.assert_array_equal(counts[1, si, ri], exit_flow)

    def test_window_contract_and_tampering(self):
        shape = (288, 2)
        base = np.broadcast_to(np.arange(288)[:, None], shape).copy()
        complete = {name: base + i for i, name in enumerate(common.CHANNELS)}
        partial = {name: values // 5 for name, values in complete.items()}
        norm = {"input": {"p05": {n: {"mean": 2., "std": 3.} for n in common.CHANNELS}},
                "target": {n: {"mean": 4., "std": 5.} for n in common.CHANNELS}}
        with patch.object(flow.windows, "N_NODES", 2), patch.dict(flow.CONTEXT, normalization=norm):
            arrays = flow.expected_arrays(partial, complete, "p05", "2023_04_01", common.CHANNELS)
        self.assertEqual(arrays["x"].shape, (277, 12, 2, 4))
        self.assertEqual(arrays["y"].shape, (277, 1, 2, 4))
        for i, name in enumerate(common.CHANNELS):
            np.testing.assert_array_equal(arrays["y_raw"][:, 0, :, i], complete[name][11:])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "all.npz"
            np.savez(path, **arrays)
            flow.verify_npz(path, arrays)
            bad = dict(arrays, target_slot=arrays["target_slot"] + 1)
            np.savez(path, **bad)
            with self.assertRaisesRegex(RuntimeError, "target_slot"):
                flow.verify_npz(path, arrays)


if __name__ == "__main__":
    unittest.main()
