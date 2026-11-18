"""CPU checks for gated cross-attention over the 11 history predictions.

Run from the repository root:

    python MSTT/tests/test_history_fusion.py
"""

from __future__ import annotations

import io
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
from MSTT.config import load_config, model_config_from_saved, task_spec, validate_config, validate_history_fusion_task  # noqa: E402
from MSTT.model.conservation import conservation_residual  # noqa: E402
from MSTT.model.graph import build_spatial_graph  # noqa: E402
from MSTT.model.history_fusion import HistoryPredictionFusion, expected_parameter_count  # noqa: E402
from MSTT.model.mstt import MSTT, MSTTConfig  # noqa: E402
from MSTT.trainer import assert_model_checkpoint_matches, assert_run_dir_matches, model_config, predict_with_aux  # noqa: E402

PARENT = PROJECT_ROOT / "MSTT/configs/experiment_temporal_aux_history11_projected.json"
FUSION = PROJECT_ROOT / "MSTT/configs/experiment_temporal_aux_history11_gated_cross_attention_projected.json"
MEAN = torch.tensor([1.0, 2.0, 3.0, 4.0])
STD = torch.tensor([2.0, 2.0, 2.0, 2.0])
TASK = task_spec("All2All")


def _graph(num_nodes: int):
    return build_spatial_graph(np.zeros((num_nodes, num_nodes), dtype=np.uint8), "bidirectional")


def _flat(num_nodes: int, mode: str) -> dict:
    config = load_config(FUSION if mode == "gated_cross_attention" else PARENT)
    config["num_nodes"] = num_nodes
    config["history_fusion_mode"] = mode
    config["temporal_aux_enabled"] = True
    validate_config(config)
    return config


def _model(num_nodes: int, mode: str, *, seed: int) -> MSTT:
    torch.manual_seed(seed)
    return MSTT(
        model_config(_flat(num_nodes, mode), TASK),
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


def _open_residual(module: HistoryPredictionFusion, *, seed: int = 11) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        module.res_out.weight.copy_(torch.randn(module.res_out.weight.shape, generator=generator))
        module.res_out.bias.copy_(torch.randn(module.res_out.bias.shape, generator=generator))


class HistoryFusionTest(unittest.TestCase):
    def test_initial_output_matches_the_aux_only_model(self) -> None:
        plain = _model(5, "none", seed=7)
        fused = _model(5, "gated_cross_attention", seed=7)
        self.assertEqual(expected_parameter_count(), 39132)
        self.assertEqual(fused.history_fusion.parameter_count(), 39132)
        self.assertEqual(fused.parameter_count() - plain.parameter_count(), 39132)
        self.assertTrue(torch.equal(fused.history_fusion.res_out.weight, torch.zeros(4, 32)))
        self.assertTrue(torch.equal(fused.history_fusion.res_out.bias, torch.zeros(4)))
        self.assertGreater(float(fused.history_fusion.position.abs().sum()), 0.0)
        shared = plain.state_dict()
        for key, value in shared.items():
            self.assertTrue(torch.equal(value, fused.state_dict()[key]), key)
        summary = fused.architecture_summary()["history_fusion"]
        self.assertEqual(summary["query_source"], "last_step_hidden")
        self.assertEqual(summary["key_source"], "standardized_history_prediction_plus_position")
        self.assertEqual(summary["value_source"], "standardized_history_prediction_plus_position")
        self.assertEqual(summary["num_heads"], 4)
        self.assertEqual(summary["head_dim"], 16)
        self.assertEqual(summary["position_dim"], 8)
        self.assertEqual(summary["gate_hidden_dim"], 32)
        self.assertEqual(summary["residual_hidden_dim"], 32)
        self.assertEqual(summary["parameters"], 39132)
        self.assertEqual(summary["attention_dropout"], 0.0)
        self.assertEqual(plain.architecture_summary()["history_fusion"]["mode"], "none")
        plain.eval()
        fused.eval()
        calls = {"encode": 0, "aux": 0}
        encode = fused._encode
        aux = fused.temporal_aux_head.forward
        fuse = fused.history_fusion.forward

        def counted_encode(*args, **kwargs):
            calls["encode"] += 1
            return encode(*args, **kwargs)

        def counted_aux(*args, **kwargs):
            calls["aux"] += 1
            return aux(*args, **kwargs)

        def counted_fuse(h, p_hist):
            calls["delta"] = fuse(h, p_hist)
            return calls["delta"]

        fused._encode = counted_encode
        fused.temporal_aux_head.forward = counted_aux
        fused.history_fusion.forward = counted_fuse
        x, tod, dow = _batch(2, 5, seed=19)
        with torch.no_grad():
            left = plain.forward_with_aux(x, tod, dow)
            right = fused.forward_with_aux(x, tod, dow)
            direct = fused(x, tod, dow)
        self.assertEqual(calls["encode"], 2)
        self.assertEqual(calls["aux"], 2)
        self.assertTrue(torch.equal(calls["delta"], torch.zeros_like(calls["delta"])))
        for key in ("raw_pred", "final_pred", "temporal_aux_pred", "correction_weights"):
            self.assertTrue(torch.equal(left[key], right[key]), key)
        self.assertTrue(torch.equal(direct, right["final_pred"]))
        self.assertNotIn("alpha", right)
        self.assertNotIn("g", right)
        self.assertNotIn("delta", right)

    def test_attention_is_per_road_and_ignores_the_last_aux_step(self) -> None:
        module = HistoryPredictionFusion()
        self.assertFalse(any(isinstance(layer, torch.nn.Dropout) for layer in module.modules()))
        for batch, nodes in ((2, 5), (3, 7), (1, 4)):
            h = torch.randn(batch, nodes, 128)
            history = torch.randn(batch, nodes, 11, 4)
            details = module.forward_details(h, history)
            self.assertEqual(tuple(details["alpha"].shape), (batch, nodes, 4, 11))
            self.assertEqual(tuple(details["g"].shape), (batch, nodes, 128))
            self.assertEqual(tuple(details["delta"].shape), (batch, nodes, 4))
            torch.testing.assert_close(details["alpha"].sum(dim=-1), torch.ones(batch, nodes, 4))
            self.assertTrue(torch.all(details["g"] >= 0))
            self.assertTrue(torch.all(details["g"] <= 1))
            self.assertTrue(torch.equal(details["delta"], torch.zeros_like(details["delta"])))
        with self.assertRaises(ValueError):
            module(torch.randn(2, 5, 128), torch.randn(2, 5, 10, 4))

        _open_residual(module)
        h = torch.randn(2, 6, 128)
        history = torch.randn(2, 6, 11, 4)
        original = module(h, history)
        changed = history.clone()
        changed[0, 3, 4, 1] += 2.5
        moved = module(h, changed)
        self.assertFalse(torch.allclose(original[0, 3], moved[0, 3]))
        self.assertTrue(torch.equal(original[0, :3], moved[0, :3]))
        self.assertTrue(torch.equal(original[0, 4:], moved[0, 4:]))
        self.assertTrue(torch.equal(original[1], moved[1]))
        other_sample = history.clone()
        other_sample[0, :, :, :] += 1.5
        sample_moved = module(h, other_sample)
        self.assertFalse(torch.allclose(original[0], sample_moved[0]))
        self.assertTrue(torch.equal(original[1], sample_moved[1]))
        shifted_h = h.clone()
        shifted_h[1, 2] += 0.7
        from_h = module(shifted_h, history)
        self.assertFalse(torch.allclose(original[1, 2], from_h[1, 2]))
        self.assertTrue(torch.equal(original[1, :2], from_h[1, :2]))
        self.assertTrue(torch.equal(original[1, 3:], from_h[1, 3:]))
        self.assertTrue(torch.equal(original[0], from_h[0]))

        model = _model(5, "gated_cross_attention", seed=4)
        model.eval()
        _open_residual(model.history_fusion, seed=12)
        hidden = torch.randn(2, 12, 5, 128)
        prediction = torch.randn(2, 12, 5, 4)
        holder = {"P": prediction}

        def encode(*args, **kwargs):
            del args, kwargs
            return hidden

        def aux(features):
            del features
            return holder["P"]

        model._encode = encode
        model.temporal_aux_head.forward = aux
        x, tod, dow = _batch(2, 5, seed=1)
        with torch.no_grad():
            first = model.forward_with_aux(x, tod, dow)
            holder["P"] = prediction.clone()
            holder["P"][:, 11] += 4.0
            second = model.forward_with_aux(x, tod, dow)
            holder["P"] = prediction.clone()
            holder["P"][:, 0, 2, 0] += 3.0
            third = model.forward_with_aux(x, tod, dow)
        self.assertTrue(torch.equal(first["raw_pred"], second["raw_pred"]))
        self.assertTrue(torch.equal(first["final_pred"], second["final_pred"]))
        self.assertFalse(torch.equal(first["temporal_aux_pred"], second["temporal_aux_pred"]))
        self.assertFalse(torch.allclose(first["raw_pred"][:, 2], third["raw_pred"][:, 2]))
        self.assertTrue(torch.equal(first["raw_pred"][:, :2], third["raw_pred"][:, :2]))
        self.assertTrue(torch.equal(first["raw_pred"][:, 3:], third["raw_pred"][:, 3:]))

    def test_last_step_loss_reaches_the_history_path_after_one_update(self) -> None:
        model = _model(4, "gated_cross_attention", seed=6)
        model.eval()
        x, tod, dow = _batch(2, 4, seed=8)
        target = torch.full((2, 4, 4), 0.3)

        def last_step_loss():
            raw = model.forward_with_aux(x, tod, dow)["raw_pred"]
            return (raw - target).abs().mean()

        loss = last_step_loss()
        loss.backward()
        self.assertGreater(float(model.history_fusion.res_out.weight.grad.abs().max()), 0.0)
        closed = (
            model.history_fusion.q_proj.weight,
            model.history_fusion.k_proj.weight,
            model.history_fusion.v_proj.weight,
            model.history_fusion.gate_in.weight,
            model.history_fusion.gate_out.weight,
            model.temporal_aux_head.weight,
        )
        for param in closed:
            self.assertIsNotNone(param.grad)
            self.assertEqual(float(param.grad.abs().max()), 0.0)
        before = model.history_fusion.res_out.weight.detach().clone()
        optimizer = torch.optim.SGD(
            [model.history_fusion.res_out.weight, model.history_fusion.res_out.bias],
            lr=0.05,
        )
        optimizer.step()
        self.assertFalse(torch.equal(model.history_fusion.res_out.weight, before))
        model.zero_grad(set_to_none=True)
        last_step_loss().backward()
        opened = (
            ("q_proj", model.history_fusion.q_proj.weight),
            ("k_proj", model.history_fusion.k_proj.weight),
            ("v_proj", model.history_fusion.v_proj.weight),
            ("gate_in", model.history_fusion.gate_in.weight),
            ("gate_out", model.history_fusion.gate_out.weight),
            ("temporal_aux_head", model.temporal_aux_head.weight),
        )
        for name, param in opened:
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
            self.assertGreater(float(param.grad.abs().max()), 0.0, name)

    def test_predictions_ignore_labels_and_stay_conservative(self) -> None:
        for batch, nodes in ((2, 6), (1, 8)):
            model = _model(nodes, "gated_cross_attention", seed=3)
            model.eval()
            x, tod, dow = _batch(batch, nodes, seed=15)
            with torch.no_grad():
                first = model.forward_with_aux(x, tod, dow)
                second = model.forward_with_aux(x, tod, dow)
                hidden = model._encode(x, tod, dow)
                base = model.head(hidden)
                auxiliary = model.temporal_aux_head(hidden)
                delta = model.history_fusion(hidden[:, -1], auxiliary[:, :-1].permute(0, 2, 1, 3).contiguous())
            self.assertTrue(torch.equal(first["final_pred"], second["final_pred"]))
            self.assertTrue(torch.equal(first["raw_pred"], base + delta))
            self.assertTrue(torch.equal(first["temporal_aux_pred"], auxiliary))
            physical = first["final_pred"] * STD + MEAN
            self.assertLess(float(conservation_residual(physical).abs().max()), 1e-4)
            slots = torch.tensor([3] if batch == 1 else [3, 8])
            observed = {
                "x": x,
                "window_start_slot": slots,
                "target_slot": slots + 11,
                "day_index": torch.arange(batch) + 1,
            }
            clean = predict_with_aux(model, observed, amp="none", device=torch.device("cpu"))
            replaced = dict(observed)
            replaced["y_seq_raw"] = torch.randn(batch, 12, nodes, 4)
            with_labels = predict_with_aux(model, replaced, amp="none", device=torch.device("cpu"))
            removed = dict(replaced)
            del removed["y_seq_raw"]
            without_labels = predict_with_aux(model, removed, amp="none", device=torch.device("cpu"))
            self.assertEqual(tuple(clean["raw_pred"].shape), (batch, 1, nodes, 4))
            self.assertEqual(tuple(clean["final_pred"].shape), (batch, 1, nodes, 4))
            self.assertEqual(tuple(clean["temporal_aux_pred"].shape), (batch, 12, nodes, 4))
            self.assertNotIn("alpha", clean)
            self.assertNotIn("g", clean)
            for key in ("raw_pred", "final_pred", "temporal_aux_pred"):
                self.assertTrue(torch.equal(clean[key], with_labels[key]), key)
                self.assertTrue(torch.equal(clean[key], without_labels[key]), key)

    def test_old_checkpoint_loads_and_new_mode_roundtrips(self) -> None:
        old = _model(4, "none", seed=9)
        omitted = {
            "model_config": {
                key: value for key, value in old.config.to_dict().items() if key != "history_fusion_mode"
            }
        }
        restored = model_config_from_saved(omitted)
        self.assertEqual(restored["history_fusion_mode"], "none")
        rebuilt = MSTT(
            MSTTConfig.from_mapping(restored),
            _graph(4),
            target_mean=MEAN.clone(),
            target_std=STD.clone(),
        )
        rebuilt.load_state_dict(old.state_dict(), strict=True)
        fused = _model(4, "gated_cross_attention", seed=10)
        blob = io.BytesIO()
        torch.save(fused.state_dict(), blob)
        blob.seek(0)
        clone = _model(4, "gated_cross_attention", seed=11)
        clone.load_state_dict(torch.load(blob, map_location="cpu", weights_only=True), strict=True)
        for key, value in fused.state_dict().items():
            self.assertTrue(torch.equal(value, clone.state_dict()[key]), key)
        assert_model_checkpoint_matches(
            {"config": {"model_config": fused.config.to_dict(), "history_fusion_mode": "gated_cross_attention"}},
            fused.config.to_dict(),
        )
        with self.assertRaises(ReimplementationError):
            assert_model_checkpoint_matches({"config": {"model_config": old.config.to_dict()}}, fused.config.to_dict())

        config = _flat(4, "none")
        payload = {
            "batch_size": config["batch_size"],
            "conservation_mode": config["conservation_mode"],
            "conservation_lambda": config["conservation_lambda"],
            "projection_raw_loss_weight": config["projection_raw_loss_weight"],
            "projection_consistency_weight": config["projection_consistency_weight"],
            "projection_supervision": config["projection_supervision"],
            "projection_mlp_hidden_dim": config["projection_mlp_hidden_dim"],
            "temporal_aux_enabled": True,
            "temporal_aux_weight": config["temporal_aux_weight"],
            "temporal_aux_scope": config["temporal_aux_scope"],
            "model_config": {
                key: value for key, value in old.config.to_dict().items() if key != "history_fusion_mode"
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "resolved_config.json").write_text(json.dumps(payload), encoding="utf-8")
            assert_run_dir_matches(run_dir, config, TASK)
            with self.assertRaises(ReimplementationError):
                assert_run_dir_matches(run_dir, _flat(4, "gated_cross_attention"), TASK)

    def test_config_rejects_an_incompatible_backbone(self) -> None:
        loaded = load_config(FUSION)
        validate_config(loaded)
        validate_history_fusion_task(loaded, TASK)
        self.assertEqual(loaded["experiment_id"], "separate_value_mlp_temporal_aux11_l05_gated_cross_attention_projected")
        self.assertEqual(loaded["history_fusion_mode"], "gated_cross_attention")
        self.assertEqual(loaded["temporal_aux_weight"], 0.5)
        self.assertEqual(loaded["batch_size"], 16)
        self.assertEqual(loaded["epochs"], 100)
        self.assertIsNone(loaded["early_stop_patience"])
        for key, value in (
            ("temporal_aux_enabled", False),
            ("head_type", "last"),
            ("conservation_mode", "none"),
            ("backbone_mode", "state_independent"),
            ("d_model", 64),
        ):
            config = load_config(FUSION)
            config[key] = value
            with self.assertRaises(ValueError):
                validate_config(config)
        config = load_config(FUSION)
        config["history_fusion_mode"] = "other"
        with self.assertRaises(ValueError):
            validate_config(config)
        with self.assertRaises(ValueError):
            validate_history_fusion_task(load_config(FUSION), task_spec("All2I"))

    def test_full_width_parameter_count(self) -> None:
        plain = _model(1406, "none", seed=0)
        fused = _model(1406, "gated_cross_attention", seed=0)
        self.assertEqual(plain.parameter_count(), 1315364)
        self.assertEqual(fused.parameter_count(), 1354496)
        self.assertEqual(fused.architecture_summary()["total_parameters"], 1354496)
        print(
            f"history_fusion_parameters={fused.history_fusion.parameter_count()} "
            f"total_N1406={fused.parameter_count()}",
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
