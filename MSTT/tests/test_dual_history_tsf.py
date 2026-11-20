"""CPU checks for the dual-path history TSF.

Run from the repository root:

    python MSTT/tests/test_dual_history_tsf.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reimplementation.common.errors import ReimplementationError  # noqa: E402
from MSTT.config import load_config, task_spec, temporal_aux_settings, validate_config, validate_history_fusion_task  # noqa: E402
from MSTT.model.conservation import conservation_residual  # noqa: E402
from MSTT.model.dual_history_tsf import mix_branch_predictions  # noqa: E402
from MSTT.model.ffn import SwiGLU  # noqa: E402
from MSTT.model.graph import build_spatial_graph  # noqa: E402
from MSTT.model.mstt import MSTT  # noqa: E402
from MSTT.trainer import assert_run_dir_matches, model_config, predict_with_aux  # noqa: E402
from MSTT.utils.losses import RawScaleLoss, add_dual_path_terms, conservation_objective, objective_description  # noqa: E402

CONFIG = PROJECT_ROOT / "MSTT/configs/experiment_dual_path_history_tsf_projected.json"
PARENT = PROJECT_ROOT / "MSTT/configs/experiment_temporal_aux_history11_projected.json"
FUSION = PROJECT_ROOT / "MSTT/configs/experiment_temporal_aux_history11_gated_cross_attention_projected.json"
MEAN = torch.tensor([1.0, 2.0, 3.0, 4.0])
STD = torch.tensor([2.0, 2.0, 2.0, 2.0])
TASK = task_spec("All2All")


def _graph(num_nodes: int):
    return build_spatial_graph(np.zeros((num_nodes, num_nodes), dtype=np.uint8), "bidirectional")


def _config(num_nodes: int) -> dict:
    config = load_config(CONFIG)
    config["num_nodes"] = num_nodes
    validate_config(config)
    return config


def _model(num_nodes: int, *, seed: int) -> MSTT:
    torch.manual_seed(seed)
    return MSTT(
        model_config(_config(num_nodes), TASK),
        _graph(num_nodes),
        target_mean=MEAN.clone(),
        target_std=STD.clone(),
    )


def _batch(batch: int, nodes: int, *, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, 12, nodes, 4, generator=generator)
    tod = torch.randint(0, 288, (batch, 12), generator=generator)
    dow = torch.randint(0, 7, (batch, 12), generator=generator)
    return x, tod, dow


def _open_gate(module, *, seed: int = 5) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        module.gate_out.weight.copy_(torch.randn(module.gate_out.weight.shape, generator=generator) * 0.05)
        module.gate_out.bias.copy_(torch.randn(module.gate_out.bias.shape, generator=generator) * 0.05)


class DualPathHistoryTSFTest(unittest.TestCase):
    def test_shapes_initial_gate_and_one_encode(self) -> None:
        model = _model(4, seed=3)
        model.eval()
        self.assertIsNone(model.history_fusion)
        self.assertIsNotNone(model.dual_history)
        self.assertTrue(torch.equal(model.dual_history.gate_out.weight, torch.zeros_like(model.dual_history.gate_out.weight)))
        self.assertTrue(torch.equal(model.dual_history.gate_out.bias, torch.zeros_like(model.dual_history.gate_out.bias)))
        encode_calls = {"n": 0}
        history_calls = {"n": 0}
        real_encode = model._encode
        real_history = model.dual_history.encode_history

        def counted_encode(*args, **kwargs):
            encode_calls["n"] += 1
            return real_encode(*args, **kwargs)

        def counted_history(history):
            history_calls["n"] += 1
            return real_history(history)

        model._encode = counted_encode
        model.dual_history.encode_history = counted_history
        x, tod, dow = _batch(2, 4, seed=8)
        outputs = model.forward_with_aux(x, tod, dow)
        self.assertEqual(encode_calls["n"], 1)
        self.assertEqual(history_calls["n"], 1)
        self.assertEqual(tuple(outputs["branch_a_pred"].shape), (2, 4, 4))
        self.assertEqual(tuple(outputs["branch_b_pred"].shape), (2, 4, 4))
        self.assertEqual(tuple(outputs["branch_gate"].shape), (2, 4, 4))
        self.assertEqual(tuple(outputs["raw_pred"].shape), (2, 4, 4))
        self.assertEqual(tuple(outputs["final_pred"].shape), (2, 4, 4))
        self.assertEqual(tuple(outputs["temporal_aux_pred"].shape), (2, 12, 4, 4))
        self.assertNotIn("history_state", outputs)
        self.assertTrue(torch.equal(outputs["branch_gate"], torch.full_like(outputs["branch_gate"], 0.5)))
        expected = mix_branch_predictions(outputs["branch_a_pred"], outputs["branch_b_pred"], outputs["branch_gate"])
        torch.testing.assert_close(outputs["raw_pred"], expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            outputs["raw_pred"],
            0.5 * (outputs["branch_a_pred"] + outputs["branch_b_pred"]),
            atol=1e-6,
            rtol=1e-6,
        )
        summary = model.architecture_summary()
        self.assertEqual(summary["backbone_ffn_count"], 1)
        self.assertEqual(summary["independent_ffn_count"], 1)
        self.assertEqual(summary["ffn_hidden_dims"], [176])
        self.assertEqual(summary["history_tsf"]["num_layers"], 1)
        self.assertEqual(summary["history_tsf"]["d_model"], 64)
        self.assertEqual(summary["history_tsf"]["temporal_num_heads"], 4)
        self.assertEqual(summary["history_tsf"]["spatial_num_heads"], 4)
        self.assertEqual(summary["history_tsf"]["ffn_hidden_dim"], 176)
        self.assertEqual(summary["history_tsf"]["ffn_count"], 1)
        self.assertEqual(summary["history_tsf"]["branch_a_head"]["in_features"], 1536)
        self.assertEqual(summary["history_tsf"]["branch_b_head"]["in_features"], 704)
        history_ffns = [module for module in model.dual_history.modules() if isinstance(module, SwiGLU)]
        backbone_ffns = [
            module for module in model.modules()
            if isinstance(module, SwiGLU) and module not in history_ffns
        ]
        self.assertEqual(len(history_ffns), 1)
        self.assertEqual(len(backbone_ffns), 1)
        self.assertEqual(summary["total_parameters"], model.parameter_count())

    def test_gate_algebra_and_trainer_time_axis(self) -> None:
        branch_a = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
        branch_b = torch.tensor([[[8.0, 7.0, 6.0, 5.0], [4.0, 3.0, 2.0, 1.0]]])
        zeros = torch.zeros_like(branch_a)
        ones = torch.ones_like(branch_a)
        self.assertTrue(torch.equal(mix_branch_predictions(branch_a, branch_b, zeros), branch_a))
        self.assertTrue(torch.equal(mix_branch_predictions(branch_a, branch_b, ones), branch_b))
        gate = torch.tensor([[[0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.7]]])
        mixed = mix_branch_predictions(branch_a, branch_b, gate)
        torch.testing.assert_close(mixed, (1.0 - gate) * branch_a + gate * branch_b, atol=0.0, rtol=0.0)

        model = _model(3, seed=4)
        model.eval()
        target = torch.tensor([0.25, 0.5, 0.75, 0.125])
        logit = torch.log(target / (1.0 - target))
        with torch.no_grad():
            model.dual_history.gate_out.weight.zero_()
            model.dual_history.gate_out.bias.copy_(logit)
        x, tod, dow = _batch(2, 3, seed=9)
        outputs = model.forward_with_aux(x, tod, dow)
        expected_gate = target.view(1, 1, 4).expand_as(outputs["branch_gate"])
        torch.testing.assert_close(outputs["branch_gate"], expected_gate, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            outputs["raw_pred"],
            mix_branch_predictions(outputs["branch_a_pred"], outputs["branch_b_pred"], outputs["branch_gate"]),
            atol=1e-6,
            rtol=1e-6,
        )
        slots = torch.tensor([4, 6])
        observed = {
            "x": x,
            "window_start_slot": slots,
            "target_slot": slots + 11,
            "day_index": torch.tensor([1, 2]),
        }
        packed = predict_with_aux(model, observed, amp="none", device=torch.device("cpu"))
        self.assertEqual(tuple(packed["branch_a_pred"].shape), (2, 1, 3, 4))
        self.assertEqual(tuple(packed["branch_b_pred"].shape), (2, 1, 3, 4))
        self.assertEqual(tuple(packed["branch_gate"].shape), (2, 1, 3, 4))
        self.assertEqual(tuple(packed["raw_pred"].shape), (2, 1, 3, 4))
        self.assertEqual(tuple(packed["final_pred"].shape), (2, 1, 3, 4))
        self.assertEqual(tuple(packed["temporal_aux_pred"].shape), (2, 12, 3, 4))
        self.assertNotIn("history_state", packed)

    def test_history_path_ignores_context_last_aux_step_and_labels(self) -> None:
        model = _model(4, seed=6)
        model.eval()
        _open_gate(model.dual_history, seed=12)
        x, tod, dow = _batch(2, 4, seed=2)
        with torch.no_grad():
            hidden = model._encode(x, tod, dow)
            auxiliary = model.temporal_aux_head(hidden)
            branch_a = model.head(hidden)
        history = auxiliary[:, :11]
        first_b = model.dual_history.history_head(model.dual_history.encode_history(history))
        second_b = model.dual_history.history_head(model.dual_history.encode_history(history.clone()))
        self.assertTrue(torch.equal(first_b, second_b))
        changed_hidden = hidden.clone()
        changed_hidden[:, -1] = changed_hidden[:, -1] + 0.7
        changed_a = branch_a + 1.5
        same_b = model.dual_history(changed_hidden, auxiliary, changed_a)["branch_b_pred"]
        base = model.dual_history(hidden, auxiliary, branch_a)
        self.assertTrue(torch.equal(same_b, base["branch_b_pred"]))
        self.assertFalse(torch.equal(base["branch_gate"], model.dual_history(changed_hidden, auxiliary, changed_a)["branch_gate"]))
        shifted = auxiliary.clone()
        shifted[:, 11] = shifted[:, 11] + 3.0
        shifted_out = model.dual_history(hidden, shifted, branch_a)
        self.assertTrue(torch.equal(shifted_out["branch_b_pred"], base["branch_b_pred"]))
        self.assertTrue(torch.equal(shifted_out["raw_pred"], base["raw_pred"]))
        clean = model.forward_with_aux(x, tod, dow)
        labeled = model.forward_with_aux(x, tod, dow)
        self.assertTrue(torch.equal(clean["final_pred"], labeled["final_pred"]))
        slots = torch.tensor([2, 5])
        observed = {"x": x, "window_start_slot": slots, "target_slot": slots + 11, "day_index": torch.tensor([0, 3])}
        without = predict_with_aux(model, observed, amp="none", device=torch.device("cpu"))
        with_labels = dict(observed)
        with_labels["y_seq_raw"] = torch.randn(2, 12, 4, 4)
        labeled_pack = predict_with_aux(model, with_labels, amp="none", device=torch.device("cpu"))
        for key in ("raw_pred", "final_pred", "branch_a_pred", "branch_b_pred", "temporal_aux_pred"):
            self.assertTrue(torch.equal(without[key], labeled_pack[key]), key)

    def test_spatial_history_reaches_another_road(self) -> None:
        model = _model(4, seed=7)
        module = model.dual_history
        module.eval()
        history = torch.zeros(1, 11, 4, 4)
        changed = history.clone()
        changed[0, 3, 0, 1] = 1.0
        base = module.encode_history(history)
        moved = module.encode_history(changed)
        self.assertGreater(float((moved[:, 3, 1] - base[:, 3, 1]).abs().max()), 0.0)
        leaf = changed.detach().requires_grad_(True)
        encoded = module.encode_history(leaf)
        encoded[0, 3, 2].sum().backward()
        self.assertGreater(float(leaf.grad[0, 3, 0].abs().sum()), 0.0)

    def test_weighted_physical_loss_and_gradients(self) -> None:
        criterion = RawScaleLoss(torch.zeros(4), torch.ones(4))
        final = torch.tensor([[[[1.0, -1.0, 0.5, 0.0]]]])
        branch_a = torch.tensor([[[[2.0, 0.0, -2.0, 1.0]]]])
        branch_b = torch.tensor([[[[0.0, 4.0, 1.0, -1.0]]]])
        auxiliary = torch.zeros(1, 12, 1, 4)
        auxiliary[:, :11] = torch.tensor([1.0, 0.0, -1.0, 0.5])
        target = torch.zeros(1, 1, 1, 4)
        sequence = torch.zeros(1, 12, 1, 4)
        manual_final = (1.0 + 1.0 + 0.5 + 0.0) / 4.0
        manual_a = (2.0 + 0.0 + 2.0 + 1.0) / 4.0
        manual_b = (0.0 + 4.0 + 1.0 + 1.0) / 4.0
        manual_aux = (1.0 + 0.0 + 1.0 + 0.5) / 4.0
        manual_total = manual_final + 0.25 * manual_a + 0.25 * manual_b + 0.5 * manual_aux
        outputs = {
            "final_pred": final.clone().requires_grad_(True),
            "raw_pred": final.clone(),
            "branch_a_pred": branch_a.clone().requires_grad_(True),
            "branch_b_pred": branch_b.clone().requires_grad_(True),
            "branch_gate": torch.full((1, 1, 1, 4), 0.5),
            "temporal_aux_pred": auxiliary.clone().requires_grad_(True),
        }
        config = _config(1)
        base, parts = conservation_objective(criterion, outputs, target, config)
        total, parts = add_dual_path_terms(criterion, outputs, target, sequence, config, base, parts)
        self.assertAlmostEqual(float(parts["final_mae"]), manual_final, places=6)
        self.assertAlmostEqual(float(parts["branch_a_mae"]), manual_a, places=6)
        self.assertAlmostEqual(float(parts["branch_b_mae"]), manual_b, places=6)
        self.assertAlmostEqual(float(parts["temporal_aux_mae"]), manual_aux, places=6)
        self.assertAlmostEqual(float(total), manual_total, places=6)
        self.assertAlmostEqual(float(parts["weighted_branch_a_loss"]), 0.25 * manual_a, places=6)
        self.assertAlmostEqual(float(parts["weighted_branch_b_loss"]), 0.25 * manual_b, places=6)
        self.assertAlmostEqual(float(parts["weighted_temporal_aux_loss"]), 0.5 * manual_aux, places=6)
        total.backward()
        self.assertGreater(float(outputs["branch_b_pred"].grad.abs().sum()), 0.0)

        model = _model(3, seed=8)
        model.train()
        x, tod, dow = _batch(2, 3, seed=1)
        only_b = model.forward_with_aux(x, tod, dow)
        branch_loss = criterion(only_b["branch_b_pred"], torch.zeros(2, 3, 4))
        branch_loss.backward()
        for name, param in (
            ("history_head", model.dual_history.history_head.proj.weight),
            ("history_block", model.dual_history.block.temporal.qkv.weight),
            ("aux_head", model.temporal_aux_head.weight),
        ):
            self.assertIsNotNone(param.grad, name)
            self.assertGreater(float(param.grad.abs().max()), 0.0, name)
        self.assertTrue(model.dual_history.q_proj.weight.grad is None or float(model.dual_history.q_proj.weight.grad.abs().max()) == 0.0)

        model.zero_grad(set_to_none=True)
        _open_gate(model.dual_history, seed=9)
        opened = model.forward_with_aux(x, tod, dow)
        self.assertFalse(torch.equal(opened["branch_a_pred"], opened["branch_b_pred"]))
        final_loss = criterion(opened["final_pred"], torch.randn(2, 3, 4))
        final_loss.backward()
        for name, param in (
            ("gate", model.dual_history.gate_out.weight),
            ("q_proj", model.dual_history.q_proj.weight),
            ("k_proj", model.dual_history.k_proj.weight),
            ("v_proj", model.dual_history.v_proj.weight),
        ):
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
            self.assertGreater(float(param.grad.abs().max()), 0.0, name)
        physical = opened["final_pred"].detach() * STD + MEAN
        self.assertLess(float(conservation_residual(physical).abs().max()), 1e-4)

    def test_config_identity_and_old_modes_stay_unchanged(self) -> None:
        loaded = load_config(CONFIG)
        validate_config(loaded)
        validate_history_fusion_task(loaded, TASK)
        self.assertEqual(loaded["experiment_id"], "dual_path_history_tsf_gate4_projected_v1")
        self.assertEqual(loaded["projection_raw_loss_weight"], 0.0)
        self.assertEqual(loaded["branch_a_loss_weight"], 0.25)
        self.assertEqual(loaded["branch_b_loss_weight"], 0.25)
        self.assertIsNone(loaded["early_stop_patience"])
        self.assertEqual(loaded["epochs"], 100)
        settings = temporal_aux_settings(loaded)
        self.assertEqual(settings["branch_a_loss_weight"], 0.25)
        self.assertNotIn("branch_a_loss_weight", temporal_aux_settings(load_config(FUSION)))
        self.assertNotIn("branch_a_loss_weight", temporal_aux_settings(load_config(PARENT)))
        text = objective_description(RawScaleLoss(MEAN, STD), loaded)
        self.assertIn("0.25 * branch_a_mae", text)
        self.assertIn("0.25 * branch_b_mae", text)
        self.assertIn("0.5 * temporal_aux_mae[0:11]", text)
        old_text = objective_description(RawScaleLoss(MEAN, STD), load_config(FUSION))
        self.assertNotIn("branch_a_mae", old_text)
        self.assertIn("0.5 * raw_mae_raw", old_text)

        rejected = dict(loaded)
        rejected["projection_raw_loss_weight"] = 0.5
        with self.assertRaises(ValueError):
            validate_config(rejected)
        rejected = dict(loaded)
        rejected["branch_b_loss_weight"] = -0.1
        with self.assertRaises(ValueError):
            validate_config(rejected)
        for key, value in (
            ("temporal_aux_enabled", False),
            ("head_type", "last"),
            ("backbone_mode", "state_independent"),
            ("d_model", 64),
        ):
            broken = dict(loaded)
            broken[key] = value
            with self.assertRaises(ValueError):
                validate_config(broken)

        torch.manual_seed(7)
        plain = MSTT(
            model_config({**load_config(PARENT), "num_nodes": 4, "history_fusion_mode": "none"}, TASK),
            _graph(4),
            target_mean=MEAN.clone(),
            target_std=STD.clone(),
        )
        torch.manual_seed(7)
        dual = _model(4, seed=7)
        shared = plain.state_dict()
        for key, value in shared.items():
            self.assertTrue(torch.equal(value, dual.state_dict()[key]), key)
        self.assertIsNone(plain.history_fusion)
        self.assertIsNone(plain.dual_history)

        config = _config(4)
        stored = {
            "batch_size": config["batch_size"],
            "conservation_mode": config["conservation_mode"],
            "conservation_lambda": config["conservation_lambda"],
            "projection_raw_loss_weight": 0.0,
            "projection_consistency_weight": 0.0,
            "projection_supervision": config["projection_supervision"],
            "projection_mlp_hidden_dim": config["projection_mlp_hidden_dim"],
            "temporal_aux_enabled": True,
            "temporal_aux_weight": 0.5,
            "temporal_aux_scope": "history_except_last",
            "history_fusion_mode": "dual_path_history_tsf",
            "branch_a_loss_weight": 0.25,
            "branch_b_loss_weight": 0.25,
            "model_config": dual.config.to_dict(),
        }
        other = dict(config)
        other["branch_a_loss_weight"] = 0.1
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resolved_config.json"
            path.write_text(json.dumps(stored), encoding="utf-8")
            assert_run_dir_matches(Path(tmp), config, TASK)
            with self.assertRaises(ReimplementationError):
                assert_run_dir_matches(Path(tmp), other, TASK)

    def test_full_width_parameter_count(self) -> None:
        model = _model(1406, seed=0)
        summary = model.architecture_summary()["history_tsf"]
        self.assertEqual(summary["parameters"], model.dual_history.parameter_count())
        self.assertEqual(model.parameter_count(), 1315364 + model.dual_history.parameter_count())
        print(
            f"dual_history_parameters={model.dual_history.parameter_count()} "
            f"total_N1406={model.parameter_count()}",
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
