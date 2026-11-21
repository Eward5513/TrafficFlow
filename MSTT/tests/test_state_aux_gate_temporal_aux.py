"""CPU checks for the speed-gate backbone plus the first-11-step auxiliary loss.

Run from the repository root:

    python MSTT/tests/test_state_aux_gate_temporal_aux.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MSTT.config import load_config, task_spec, validate_config  # noqa: E402
from MSTT.model.graph import build_spatial_graph  # noqa: E402
from MSTT.model.mstt import MSTT  # noqa: E402
from MSTT.trainer import (  # noqa: E402
    attach_temporal_history_if_loaded,
    model_config,
    selects_inflow_mae,
)
from MSTT.utils.losses import RawScaleLoss, conservation_objective, temporal_aux_terms  # noqa: E402
from MSTT.run_mstt_state_aux_gate_temporal_aux_p5_p15 import child_env  # noqa: E402
from MSTT.utils.threads import MAX_THREADS, parse_thread_cap  # noqa: E402

CONFIG = PROJECT_ROOT / "MSTT/configs/experiment_state_aux_gate_temporal_aux_history11_projected.json"
PARENT = PROJECT_ROOT / "MSTT/configs/all_sp_c_state_aux_gate_d128_projected_all.json"
MEAN = torch.tensor([1.0, 2.0, 3.0, 4.0])
STD = torch.tensor([2.0, 2.0, 2.0, 2.0])
AUX_KEYS = (
    "experiment_id",
    "temporal_aux_enabled",
    "temporal_aux_scope",
    "temporal_aux_weight",
)


def _graph(num_nodes: int):
    return build_spatial_graph(np.zeros((num_nodes, num_nodes), dtype=np.uint8), "bidirectional")


def _built(path: Path, num_nodes: int, task: str = "AllSpC2All"):
    config = load_config(path)
    config["num_nodes"] = num_nodes
    config["dropout"] = 0.0
    config["attention_dropout"] = 0.0
    config["embedding_dropout"] = 0.0
    validate_config(config)
    return config, model_config(config, task_spec(task))


def _model(num_nodes: int, *, path: Path = CONFIG, seed: int = 3) -> tuple[dict, MSTT]:
    config, built = _built(path, num_nodes)
    torch.manual_seed(seed)
    model = MSTT(built, _graph(num_nodes), target_mean=MEAN.clone(), target_std=STD.clone())
    return config, model


def _batch(batch: int, nodes: int, channels: int = 6):
    x = torch.randn(batch, 12, nodes, channels)
    tod = torch.randint(0, 288, (batch, 12))
    dow = torch.randint(0, 7, (batch, 12))
    return x, tod, dow


class SpeedGateTemporalAuxTest(unittest.TestCase):
    def test_resolved_config_only_adds_the_auxiliary_switch(self) -> None:
        parent = load_config(PARENT)
        child = load_config(CONFIG)
        self.assertEqual(child["experiment_id"], "state_aux_gate_d128_temporal_aux11_l05_projected_all")
        self.assertIs(child["temporal_aux_enabled"], True)
        self.assertEqual(child["temporal_aux_scope"], "history_except_last")
        self.assertEqual(child["temporal_aux_weight"], 0.5)
        self.assertEqual(child["history_fusion_mode"], "none")
        self.assertEqual(child["value_embedding_mode"], "state_aux_gate")
        self.assertTrue(selects_inflow_mae(task_spec("AllSpC2All")))
        differing = sorted(key for key in set(parent) | set(child) if parent.get(key) != child.get(key))
        self.assertEqual(differing, ["experiment_id", "temporal_aux_enabled"])
        for key in AUX_KEYS:
            self.assertIn(key, child)

    def test_shapes_loss_and_backward(self) -> None:
        config, model = _model(4, seed=2)
        model.train()
        self.assertIsNone(model.history_fusion)
        self.assertIsNone(model.dual_history)
        self.assertIsNone(model.stage2_block)
        x, tod, dow = _batch(2, 4)
        outputs = model.forward_with_aux(x, tod, dow)
        self.assertEqual(tuple(outputs["final_pred"].shape), (2, 4, 4))
        self.assertEqual(tuple(outputs["raw_pred"].shape), (2, 4, 4))
        self.assertEqual(tuple(outputs["temporal_aux_pred"].shape), (2, 12, 4, 4))
        criterion = RawScaleLoss(MEAN, STD, kind="mae_raw")
        y_raw = torch.randn(2, 4, 4)
        y_seq = torch.randn(2, 12, 4, 4)
        loss, _parts = conservation_objective(criterion, outputs, y_raw, config)
        aux, weighted = temporal_aux_terms(criterion, outputs, y_seq, config)
        total = loss + weighted
        self.assertTrue(torch.isfinite(total))
        aux.backward()
        head_grad = model.temporal_aux_head.weight.grad
        encoder_grad = model.embedding.value_mlps[0][0].weight.grad
        self.assertIsNotNone(head_grad)
        self.assertTrue(torch.isfinite(head_grad).all())
        self.assertGreater(float(head_grad.abs().max()), 0.0)
        self.assertIsNotNone(encoder_grad)
        self.assertTrue(torch.isfinite(encoder_grad).all())
        self.assertGreater(float(encoder_grad.abs().max()), 0.0)

    def test_aux_head_does_not_change_the_final_prediction(self) -> None:
        _config, model = _model(3, seed=5)
        model.eval()
        x, tod, dow = _batch(2, 3)
        with torch.no_grad():
            before = model.forward_with_aux(x, tod, dow)
            model.temporal_aux_head.weight.add_(0.7)
            model.temporal_aux_head.bias.add_(0.3)
            after = model.forward_with_aux(x, tod, dow)
        self.assertTrue(torch.equal(before["final_pred"], after["final_pred"]))
        self.assertTrue(torch.equal(before["raw_pred"], after["raw_pred"]))
        self.assertFalse(torch.equal(before["temporal_aux_pred"], after["temporal_aux_pred"]))

    def test_last_auxiliary_target_is_outside_the_loss(self) -> None:
        config, _model_unused = _model(2, seed=1)
        criterion = RawScaleLoss(MEAN, STD, kind="mae_raw")
        prediction = torch.zeros(2, 12, 3, 4)
        target = MEAN.view(1, 1, 1, 4).expand(2, 12, 3, 4).clone()
        target[:, :11] = target[:, :11] + 4.0
        outputs = {"temporal_aux_pred": prediction}
        aux, weighted = temporal_aux_terms(criterion, outputs, target, config)
        self.assertAlmostEqual(float(aux), 4.0, places=5)
        self.assertAlmostEqual(float(weighted), 2.0, places=5)
        shifted = target.clone()
        shifted[:, 11] += 9.0
        aux_shifted, _weighted_shifted = temporal_aux_terms(criterion, outputs, shifted, config)
        self.assertTrue(torch.equal(aux_shifted, aux))

    def test_full_width_parameter_count(self) -> None:
        _parent_config, parent = _model(1406, path=PARENT, seed=0)
        _child_config, child = _model(1406, path=CONFIG, seed=0)
        self.assertEqual(parent.parameter_count(), 1316000)
        self.assertEqual(child.parameter_count() - parent.parameter_count(), 128 * 4 + 4)
        self.assertEqual(child.parameter_count(), 1316516)

    def test_eval_splits_do_not_require_train_labels(self) -> None:
        datasets = {"test": object()}
        audit = attach_temporal_history_if_loaded(datasets, True)
        self.assertIsNone(audit)
        self.assertNotIn("train", datasets)

    def test_thread_cap_rejects_the_default_pool(self) -> None:
        self.assertEqual(MAX_THREADS, 15)
        self.assertEqual(parse_thread_cap("15"), 15)
        self.assertIsNone(parse_thread_cap(None))
        with self.assertRaises(SystemExit):
            parse_thread_cap("64")
        with self.assertRaises(SystemExit):
            parse_thread_cap("0")
        env = child_env("GPU-test", base={"OMP_NUM_THREADS": "64", "PATH": "/usr/bin"})
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "MSTT_NUM_THREADS"):
            self.assertEqual(env[key], "15", key)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-test")


if __name__ == "__main__":
    unittest.main()
