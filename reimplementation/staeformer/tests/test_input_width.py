"""STAEformer accepts one traffic channel or the four-state stack."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reimplementation.common.data.state_windows import apply_state_types
from reimplementation.common.errors import ReimplementationError
from reimplementation.staeformer.data import assemble_history_and_indices
from reimplementation.staeformer.model.staeformer import STAEformer
from reimplementation.staeformer.train_r_only_staeformer import _assert_input_width, _assert_task


def _tiny(**overrides: int) -> STAEformer:
    kwargs = {
        "num_nodes": 4,
        "in_steps": 12,
        "out_steps": 1,
        "steps_per_day": 288,
        "input_dim": 3,
        "output_dim": 1,
        "input_embedding_dim": 8,
        "tod_embedding_dim": 8,
        "dow_embedding_dim": 8,
        "spatial_embedding_dim": 0,
        "adaptive_embedding_dim": 8,
        "feed_forward_dim": 16,
        "num_heads": 4,
        "num_layers": 1,
    }
    kwargs.update(overrides)
    return STAEformer(**kwargs)


def _indices(batch: int = 2, time: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    slots = torch.arange(time).view(1, time).expand(batch, time)
    days = torch.zeros(batch, dtype=torch.long)
    return slots, days


def _task_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "target_mode": "last-observed-step",
        "out_steps": 1,
        "in_steps": 12,
        "steps_per_day": 288,
        "spatial_embedding_dim": 0,
        "tod_embedding_dim": 24,
        "dow_embedding_dim": 24,
        "day_of_week_source": "original_pems_npz_sequential_index_mod_7",
        "input_dim": 3,
        "input_type": "inflow",
    }
    config.update(overrides)
    return config


class InputWidthTests(unittest.TestCase):
    def test_single_and_all_share_attention_width(self) -> None:
        single = _tiny(input_dim=3, output_dim=1)
        wide = _tiny(input_dim=6, output_dim=1)
        four_out = _tiny(input_dim=6, output_dim=4)
        self.assertEqual(single.model_dim, wide.model_dim)
        self.assertEqual(tuple(single.input_proj.weight.shape), (8, 3))
        self.assertEqual(tuple(wide.input_proj.weight.shape), (8, 6))
        self.assertEqual(tuple(four_out.output_proj.weight.shape), (4, 12 * single.model_dim))

        tod, dow = _indices()
        one = torch.randn(2, 12, 4, 1)
        pred_one = single(one, time_of_day_index=tod, day_of_week_index=dow)
        self.assertEqual(tuple(pred_one.shape), (2, 1, 4, 1))

        four = torch.randn(2, 12, 4, 4)
        pred_four = wide(four, time_of_day_index=tod, day_of_week_index=dow)
        self.assertEqual(tuple(pred_four.shape), (2, 1, 4, 1))
        pred_all = four_out(four, time_of_day_index=tod, day_of_week_index=dow)
        self.assertEqual(tuple(pred_all.shape), (2, 1, 4, 4))

    def test_preassembled_all_reads_time_channels_at_the_end(self) -> None:
        model = _tiny(input_dim=6, output_dim=1)
        history = torch.full((2, 12, 4, 6), 9.0)
        history[..., 4] = torch.arange(12, dtype=torch.float32).view(1, 12, 1) / 288.0
        history[..., 5] = 3.0
        pred = model(history)
        self.assertEqual(tuple(pred.shape), (2, 1, 4, 1))
        self.assertTrue(torch.isfinite(pred).all())

    def test_assemble_places_time_channels_after_four_states(self) -> None:
        traffic = np.arange(2 * 12 * 3 * 4, dtype=np.float32).reshape(2, 12, 3, 4)
        packed = assemble_history_and_indices(
            traffic,
            window_start_slot=np.asarray([0, 20], dtype=np.int64),
            day_index=np.asarray([1, 8], dtype=np.int64),
            n_his=12,
            input_dim=6,
            tod_embedding_dim=24,
            dow_embedding_dim=24,
            target_slot=np.asarray([11, 31], dtype=np.int64),
        )
        history = packed["history"].numpy()
        self.assertEqual(history.shape, (2, 12, 3, 6))
        self.assertTrue(np.allclose(history[..., :4], traffic))
        self.assertTrue(np.allclose(history[0, :, 0, 4], np.arange(12) / 288.0))
        self.assertTrue(np.all(history[0, ..., 5] == 1))
        self.assertTrue(np.all(history[1, ..., 5] == 1))

    def test_guard_accepts_single_and_all(self) -> None:
        _assert_task(_task_config())
        _assert_task(_task_config(input_type="all", input_dim=6))
        _assert_input_width(_task_config(input_type="outflow", input_dim=3))
        with self.assertRaises(ReimplementationError):
            _assert_input_width(_task_config(input_type="all", input_dim=3))
        config = {
            "output_root": "/tmp/staeformer_width_test",
            "tod_embedding_dim": 24,
            "dow_embedding_dim": 24,
            "input_dim": 3,
            "output_dim": 1,
        }
        args = SimpleNamespace(input_type="all", target_type="inflow")
        updated = apply_state_types(config, args, model_name="staeformer")
        self.assertEqual(updated["input_dim"], 6)
        self.assertEqual(updated["output_dim"], 1)
        _assert_input_width(updated)


if __name__ == "__main__":
    unittest.main()
