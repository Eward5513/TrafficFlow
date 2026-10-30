"""Scalar and per-channel target scalers both finish official evaluation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reimplementation.common.evaluation import evaluate_loader
from reimplementation.s2s_gcn_lstm.losses import MAERawLoss


def _batch(y_norm: np.ndarray, y_raw: np.ndarray) -> dict[str, torch.Tensor]:
    count = int(y_norm.shape[0])
    start = np.arange(count, dtype=np.int64)
    return {
        "x": torch.from_numpy(y_norm),
        "y": torch.from_numpy(y_norm.astype(np.float32)),
        "x_raw": torch.from_numpy(y_raw.astype(np.float32)),
        "y_raw": torch.from_numpy(y_raw.astype(np.float32)),
        "day_index": torch.zeros(count, dtype=torch.int64),
        "window_start_slot": torch.from_numpy(start),
        "target_slot": torch.from_numpy(start + 11),
        "sample_index": torch.from_numpy(start),
    }


class ChannelScalerEvaluationTests(unittest.TestCase):
    def test_scalar_and_four_channel_evaluation_both_finish(self) -> None:
        cases = {
            "scalar": (
                np.array([2.0], dtype=np.float64),
                np.array([4.0], dtype=np.float64),
                np.array([[[[0.5]]], [[[-1.0]]]], dtype=np.float32),
            ),
            "all": (
                np.array([11.0, 11.0, 1.4, 1.4], dtype=np.float64),
                np.array([17.0, 17.0, 4.0, 4.0], dtype=np.float64),
                np.array(
                    [
                        [[[0.2, -0.1, 0.5, 1.0]]],
                        [[[-0.4, 0.3, -1.0, 0.0]]],
                    ],
                    dtype=np.float32,
                ),
            ),
        }
        for name, (mean, std, y_norm) in cases.items():
            with self.subTest(name=name):
                y_raw = y_norm * std.reshape(1, 1, 1, -1) + mean.reshape(1, 1, 1, -1)
                batch = _batch(y_norm, y_raw.astype(np.float32))
                loss = MAERawLoss(mean, std)
                value = loss(batch["y"], batch["y_raw"])
                self.assertTrue(torch.isfinite(value))
                self.assertAlmostEqual(float(value), 0.0, places=5)
                with tempfile.TemporaryDirectory() as directory:
                    report = evaluate_loader(
                        [batch],
                        lambda item: item["y"],
                        mean_y=mean,
                        std_y=std,
                        device=torch.device("cpu"),
                        output_dir=Path(directory),
                        split="test",
                        expected_count=2,
                        n_his=12,
                    )
                self.assertEqual(report["sample_count"], 2)
                self.assertAlmostEqual(report["metrics"]["mae"], 0.0, places=5)
                self.assertTrue(np.isfinite(report["metrics"]["wape"]))
                if name == "all":
                    for channel in ("inflow", "outflow", "accumulation_start", "accumulation_end"):
                        self.assertAlmostEqual(report["metrics"][f"{channel}_mae"], 0.0, places=5)
                else:
                    self.assertNotIn("inflow_mae", report["metrics"])


if __name__ == "__main__":
    unittest.main()
