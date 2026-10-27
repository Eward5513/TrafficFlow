"""DCRNN and Graph WaveNet accept one official penetration rate."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reimplementation.common.errors import ReimplementationError
from reimplementation.common.utils.structured_logging import JsonlLogger
from reimplementation.dcrnn.engine import run_all_rates as run_dcrnn
from reimplementation.graph_wavenet.engine import run_all_rates as run_gwn


def _call(runner, rates: list[str], resume: Path | None = None) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with JsonlLogger(root / "train.jsonl") as logger:
            runner(
                {"seq_len": 12, "num_nodes": 56},
                project_root=root,
                rates=rates,
                data_root=root,
                output_root=root,
                adjacency_path=root / "adj.npy",
                pickle_path=root / "adj.pkl",
                r_nodes_path=root / "nodes.csv",
                logger=logger,
                resume=resume,
            )


class SingleRateEntryTests(unittest.TestCase):
    def test_one_official_rate_passes_the_gate(self) -> None:
        for module, runner in (
            ("reimplementation.dcrnn.engine", run_dcrnn),
            ("reimplementation.graph_wavenet.engine", run_gwn),
        ):
            with self.subTest(module=module):
                with patch(f"{module}.validate_prepared_data", side_effect=RuntimeError("past-rate-gate")):
                    with self.assertRaisesRegex(RuntimeError, "past-rate-gate"):
                        _call(runner, ["p05"])

    def test_illegal_rate_is_still_rejected(self) -> None:
        for runner in (run_dcrnn, run_gwn):
            with self.subTest(runner=runner.__module__):
                with self.assertRaises(ReimplementationError):
                    _call(runner, ["p60"])

    def test_resume_still_refuses_multiple_rates(self) -> None:
        for runner in (run_dcrnn, run_gwn):
            with self.subTest(runner=runner.__module__):
                with self.assertRaisesRegex(ReimplementationError, "do not share checkpoints"):
                    _call(runner, ["p05", "p10"], resume=Path("checkpoint.pt"))


if __name__ == "__main__":
    unittest.main()
