"""CPU checks for the 12-step temporal auxiliary head.

Run from the repository root:

    python MSTT/tests/test_temporal_aux.py
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

from MSTT.config import load_config, model_config_from_saved, task_spec, validate_config  # noqa: E402
from MSTT.model.graph import build_spatial_graph  # noqa: E402
from MSTT.model.mstt import MSTT  # noqa: E402
from MSTT.trainer import model_config, predict_with_aux  # noqa: E402
from MSTT.utils.losses import RawScaleLoss, temporal_aux_terms  # noqa: E402

CONFIG = PROJECT_ROOT / "MSTT/configs/experiment_temporal_aux_history11_projected.json"
MEAN = torch.tensor([1.0, 2.0, 3.0, 4.0])
STD = torch.tensor([2.0, 2.0, 2.0, 2.0])


def _graph(num_nodes: int):
    return build_spatial_graph(np.zeros((num_nodes, num_nodes), dtype=np.uint8), "bidirectional")


def _model(num_nodes: int, *, enabled: bool, seed: int = 3) -> MSTT:
    config = load_config(CONFIG)
    config["num_nodes"] = num_nodes
    config["temporal_aux_enabled"] = enabled
    validate_config(config)
    torch.manual_seed(seed)
    built = model_config(config, task_spec("All2All"))
    return MSTT(built, _graph(num_nodes), target_mean=MEAN.clone(), target_std=STD.clone())


class TemporalAuxTest(unittest.TestCase):
    def test_parameter_delta_and_single_encode(self) -> None:
        disabled = _model(8, enabled=False, seed=3)
        enabled = _model(8, enabled=True, seed=3)
        delta = enabled.parameter_count() - disabled.parameter_count()
        self.assertEqual(delta, 128 * 4 + 4)
        self.assertEqual(enabled.temporal_aux_head.weight.shape, (4, 128))
        shared = {key: value for key, value in disabled.state_dict().items()}
        for key, value in shared.items():
            self.assertTrue(torch.equal(value, enabled.state_dict()[key]), key)
        calls = {"n": 0}
        original = enabled._encode

        def counted(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        enabled._encode = counted
        enabled.eval()
        x = torch.randn(2, 12, 8, 4)
        tod = torch.randint(0, 288, (2, 12))
        dow = torch.randint(0, 7, (2, 12))
        outputs = enabled.forward_with_aux(x, tod, dow)
        self.assertEqual(calls["n"], 1)
        self.assertEqual(tuple(outputs["final_pred"].shape), (2, 8, 4))
        self.assertEqual(tuple(outputs["temporal_aux_pred"].shape), (2, 12, 8, 4))

    def test_aux_head_does_not_change_the_final_step(self) -> None:
        torch.manual_seed(5)
        source = _model(6, enabled=False, seed=5)
        torch.manual_seed(5)
        target = _model(6, enabled=True, seed=5)
        copied = {key: value.detach().clone() for key, value in source.state_dict().items()}
        target.load_state_dict(copied, strict=False)
        source.eval()
        target.eval()
        x = torch.randn(2, 12, 6, 4)
        tod = torch.randint(0, 288, (2, 12))
        dow = torch.randint(0, 7, (2, 12))
        with torch.no_grad():
            left = source.forward_with_aux(x, tod, dow)
            right = target.forward_with_aux(x, tod, dow)
        self.assertTrue(torch.equal(left["final_pred"], right["final_pred"]))
        self.assertTrue(torch.equal(left["raw_pred"], right["raw_pred"]))
        self.assertNotIn("temporal_aux_pred", left)
        self.assertEqual(tuple(right["temporal_aux_pred"].shape), (2, 12, 6, 4))

    def test_disabled_checkpoint_loads_strictly(self) -> None:
        old = _model(4, enabled=False, seed=9)
        saved = {
            "model_config": {
                key: value for key, value in old.config.to_dict().items() if key != "temporal_aux_enabled"
            }
        }
        restored = model_config_from_saved(saved)
        self.assertIs(restored["temporal_aux_enabled"], False)
        rebuilt = MSTT(
            type(old.config).from_mapping(restored),
            _graph(4),
            target_mean=MEAN.clone(),
            target_std=STD.clone(),
        )
        rebuilt.load_state_dict(old.state_dict(), strict=True)

    def test_predict_with_aux_keeps_twelve_steps(self) -> None:
        model = _model(4, enabled=True, seed=1)
        model.eval()
        batch = {
            "x": torch.randn(2, 12, 4, 4),
            "window_start_slot": torch.tensor([3, 8]),
            "target_slot": torch.tensor([14, 19]),
            "day_index": torch.tensor([1, 2]),
        }
        outputs = predict_with_aux(model, batch, amp="none", device=torch.device("cpu"))
        self.assertEqual(tuple(outputs["final_pred"].shape), (2, 1, 4, 4))
        self.assertEqual(tuple(outputs["raw_pred"].shape), (2, 1, 4, 4))
        self.assertEqual(tuple(outputs["correction_weights"].shape), (2, 1, 4, 4))
        self.assertEqual(tuple(outputs["temporal_aux_pred"].shape), (2, 12, 4, 4))

    def test_history_loss_ignores_the_last_step(self) -> None:
        criterion = RawScaleLoss(MEAN, STD, kind="mae_raw")
        prediction = torch.zeros(2, 12, 3, 4)
        target = MEAN.view(1, 1, 1, 4).expand(2, 12, 3, 4).clone()
        target[:, :11] = target[:, :11] + 4.0
        config = {"temporal_aux_enabled": True, "temporal_aux_weight": 0.5, "temporal_aux_scope": "history_except_last"}
        outputs = {"temporal_aux_pred": prediction}
        aux, weighted = temporal_aux_terms(criterion, outputs, target, config)
        # Denormalized prediction is the scaler mean, so each error is 4. Mean over B*11*N*4 stays 4.
        self.assertAlmostEqual(float(aux), 4.0, places=5)
        self.assertAlmostEqual(float(weighted), 2.0, places=5)
        shifted_last_pred = prediction.clone()
        shifted_last_pred[:, 11] += 9.0
        shifted_last_target = target.clone()
        shifted_last_target[:, 11] += 7.0
        aux_pred, _ = temporal_aux_terms(criterion, {"temporal_aux_pred": shifted_last_pred}, target, config)
        aux_target, _ = temporal_aux_terms(criterion, outputs, shifted_last_target, config)
        self.assertTrue(torch.equal(aux_pred, aux))
        self.assertTrue(torch.equal(aux_target, aux))
        changed = prediction.clone()
        changed[:, 0] -= 1.0
        aux_changed, _ = temporal_aux_terms(criterion, {"temporal_aux_pred": changed}, target, config)
        self.assertGreater(float(aux_changed), float(aux))
        # One step of error 4 and ten steps of error 0 averages to 4/11, not 4.
        sparse_target = MEAN.view(1, 1, 1, 4).expand(1, 12, 1, 4).clone()
        sparse_target[:, 0] = sparse_target[:, 0] + 4.0
        sparse, _ = temporal_aux_terms(
            criterion,
            {"temporal_aux_pred": torch.zeros(1, 12, 1, 4)},
            sparse_target,
            config,
        )
        self.assertAlmostEqual(float(sparse), 4.0 / 11.0, places=5)

    def test_aux_backward_reaches_the_encoder(self) -> None:
        model = _model(4, enabled=True, seed=2)
        model.train()
        criterion = RawScaleLoss(MEAN, STD, kind="mae_raw")
        x = torch.randn(2, 12, 4, 4)
        tod = torch.randint(0, 288, (2, 12))
        dow = torch.randint(0, 7, (2, 12))
        outputs = model.forward_with_aux(x, tod, dow)
        target = torch.randn(2, 12, 4, 4)
        config = {"temporal_aux_enabled": True, "temporal_aux_weight": 0.5, "temporal_aux_scope": "history_except_last"}
        aux, _ = temporal_aux_terms(criterion, outputs, target, config)
        aux.backward()
        head_grad = model.temporal_aux_head.weight.grad
        encoder_grad = model.embedding.value_proj.weight.grad if hasattr(model.embedding, "value_proj") else None
        if encoder_grad is None:
            encoder_grad = next(model.embedding.parameters()).grad
        self.assertIsNotNone(head_grad)
        self.assertTrue(torch.isfinite(head_grad).all())
        self.assertGreater(float(head_grad.abs().max()), 0.0)
        self.assertIsNotNone(encoder_grad)
        self.assertTrue(torch.isfinite(encoder_grad).all())
        self.assertGreater(float(encoder_grad.abs().max()), 0.0)

    def test_full_width_parameter_count(self) -> None:
        disabled = _model(1406, enabled=False, seed=0)
        enabled = _model(1406, enabled=True, seed=0)
        self.assertEqual(disabled.parameter_count(), 1314848)
        self.assertEqual(enabled.parameter_count(), 1315364)


if __name__ == "__main__":
    unittest.main()
