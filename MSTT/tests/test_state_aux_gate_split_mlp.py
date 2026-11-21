"""CPU checks for the split speed/count gate.

Run from the repository root:

    python MSTT/tests/test_state_aux_gate_split_mlp.py
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MSTT.config import load_config, task_spec, validate_config  # noqa: E402
from MSTT.model.auxiliary_gate import AuxiliaryStateGate, SplitAuxiliaryStateGate  # noqa: E402
from MSTT.model.graph import build_spatial_graph  # noqa: E402
from MSTT.model.mstt import MSTT  # noqa: E402
from MSTT.tests.test_state_token_transformer import MEAN, STD, _sample  # noqa: E402
from MSTT.trainer import (  # noqa: E402
    assert_conservation_checkpoint,
    assert_model_checkpoint_matches,
    model_config,
)
from MSTT.run_mstt_state_aux_gate_split_mlp_p5_p15 import baseline_blocks_claim  # noqa: E402
from reimplementation.common.errors import ReimplementationError  # noqa: E402

import numpy as np

PARENT = PROJECT_ROOT / "MSTT/configs/all_sp_c_state_aux_gate_d128_projected_all.json"
CONFIG = PROJECT_ROOT / "MSTT/configs/all_sp_c_state_aux_gate_split_mlp_d128_projected_all.json"


def _count(module: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters()))


def _graph(num_nodes: int):
    return build_spatial_graph(np.zeros((num_nodes, num_nodes), dtype=np.uint8), "bidirectional")


def _model(path: Path, num_nodes: int) -> MSTT:
    config = load_config(path)
    config["num_nodes"] = num_nodes
    config["dropout"] = 0.0
    config["attention_dropout"] = 0.0
    config["embedding_dropout"] = 0.0
    validate_config(config)
    built = model_config(config, task_spec("AllSpC2All"))
    return MSTT(built, _graph(num_nodes), target_mean=MEAN.clone(), target_std=STD.clone())


class SplitGateModuleTest(unittest.TestCase):
    def test_identity_init_shape_and_independent_encoders(self) -> None:
        gate = SplitAuxiliaryStateGate()
        self.assertIsNot(gate.speed_encoder, gate.count_encoder)
        speed_ptrs = {param.data_ptr() for param in gate.speed_encoder.parameters()}
        count_ptrs = {param.data_ptr() for param in gate.count_encoder.parameters()}
        self.assertTrue(speed_ptrs.isdisjoint(count_ptrs))
        self.assertEqual(_count(gate.speed_encoder), 88)
        self.assertEqual(_count(gate.count_encoder), 88)
        self.assertEqual(_count(gate.fusion), 544)
        self.assertEqual(_count(gate.last), 1056)
        self.assertEqual(_count(gate), 1776)
        self.assertEqual(_count(AuxiliaryStateGate()), 1152)
        aux = torch.randn(2, 12, 5, 2)
        initial = gate(aux)
        self.assertEqual(tuple(initial.shape), (2, 12, 5, 32))
        self.assertTrue(torch.allclose(initial, torch.ones_like(initial)))
        self.assertTrue(torch.allclose(gate(aux + 4.0), torch.ones_like(initial)))
        with self.assertRaises(ValueError):
            gate(torch.randn(2, 3))

    def test_each_channel_moves_the_gate_after_the_last_layer_opens(self) -> None:
        gate = SplitAuxiliaryStateGate()
        with torch.no_grad():
            gate.speed_encoder[0].weight.fill_(0.2)
            gate.speed_encoder[0].bias.zero_()
            gate.speed_encoder[2].weight.copy_(torch.eye(8))
            gate.speed_encoder[2].bias.zero_()
            gate.count_encoder[0].weight.fill_(0.2)
            gate.count_encoder[0].bias.zero_()
            gate.count_encoder[2].weight.copy_(torch.eye(8))
            gate.count_encoder[2].bias.zero_()
            gate.fusion.weight.zero_()
            gate.fusion.bias.zero_()
            gate.fusion.weight[:8, :8] = torch.eye(8)
            gate.fusion.weight[8:16, 8:] = torch.eye(8)
            gate.last.weight.copy_(torch.eye(32))
            gate.last.bias.zero_()
        aux = torch.randn(3, 4, 2)
        base = gate(aux)
        faster = aux.clone()
        faster[..., 0] += 1.5
        busier = aux.clone()
        busier[..., 1] += 1.5
        self.assertFalse(torch.allclose(gate(faster), base))
        self.assertFalse(torch.allclose(gate(busier), base))


class SplitGateModelTest(unittest.TestCase):
    def test_config_changes_only_the_mode_and_experiment(self) -> None:
        parent = json.loads(PARENT.read_text(encoding="utf-8"))
        child = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(child["value_embedding_mode"], "state_aux_gate_split_mlp")
        self.assertEqual(child["experiment_id"], "state_aux_gate_split_mlp_d128_projected_all")
        parent["value_embedding_mode"] = child["value_embedding_mode"]
        parent["experiment_id"] = child["experiment_id"]
        self.assertEqual(parent, child)
        loaded = load_config(CONFIG)
        self.assertEqual(loaded["num_layers"], 1)
        self.assertEqual(loaded["backbone_mode"], "shared")
        self.assertEqual(loaded["head_type"], "flatten")
        self.assertFalse(loaded["temporal_aux_enabled"])
        self.assertEqual(loaded["history_fusion_mode"], "none")
        self.assertFalse(loaded["use_gated_attention"])
        self.assertEqual(loaded["epochs"], 100)
        self.assertIsNone(loaded["early_stop_patience"])
        validate_config(loaded)

    def test_two_steps_reach_both_encoders_and_checkpoints_stay_apart(self) -> None:
        model = _model(CONFIG, 3)
        gate = model.embedding.auxiliary_gate
        self.assertIsInstance(gate, SplitAuxiliaryStateGate)
        summary = model.embedding.value_embedding_summary()
        self.assertEqual(summary["mode"], "state_aux_gate_split_mlp")
        self.assertEqual(summary["value_width"], 32)
        self.assertTrue(summary["encoders_independent"])
        self.assertEqual(model.architecture_summary()["d_model"], 128)
        self.assertEqual(tuple(model.embedding.widths.values())[:1], (32,))
        x, tod, dow = _sample(2, 3, 6)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
        model.train()
        encoder_grads = []
        for _step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = model(x, tod, dow)
            self.assertEqual(tuple(output.shape), (2, 3, 4))
            output.square().mean().backward()
            self.assertGreater(float(gate.last.weight.grad.abs().sum()), 0.0)
            encoder_grads.append(
                float(gate.speed_encoder[0].weight.grad.abs().sum())
                + float(gate.count_encoder[0].weight.grad.abs().sum())
                + float(gate.fusion.weight.grad.abs().sum())
            )
            optimizer.step()
        self.assertEqual(encoder_grads[0], 0.0)
        self.assertGreater(encoder_grads[1], 0.0)

        rebuilt = _model(CONFIG, 3)
        rebuilt.load_state_dict(model.state_dict(), strict=True)
        old = _model(PARENT, 3)
        with self.assertRaises(RuntimeError):
            old.load_state_dict(model.state_dict(), strict=True)
        with self.assertRaises(RuntimeError):
            rebuilt.load_state_dict(old.state_dict(), strict=True)
        current = model_config(load_config(CONFIG), task_spec("AllSpC2All")).to_dict()
        previous = model_config(load_config(PARENT), task_spec("AllSpC2All")).to_dict()
        assert_model_checkpoint_matches({"config": {"model_config": current}}, current)
        with self.assertRaises(ReimplementationError):
            assert_model_checkpoint_matches({"config": {"model_config": previous}}, current)
        with self.assertRaises(ReimplementationError):
            assert_model_checkpoint_matches({"config": {"model_config": current}}, previous)
        with self.assertRaises(ReimplementationError):
            assert_conservation_checkpoint(
                {"conservation_mode": "projected", "config": load_config(PARENT)},
                load_config(CONFIG),
            )

    def test_parameter_count_at_full_road_count(self) -> None:
        old = _model(PARENT, 1406)
        new = _model(CONFIG, 1406)
        self.assertEqual(old.parameter_count(), 1_316_000)
        self.assertEqual(new.parameter_count(), 1_316_624)
        self.assertEqual(new.parameter_count() - old.parameter_count(), 624)
        self.assertEqual(_count(new.embedding.auxiliary_gate), 1776)
        self.assertEqual(new.embedding.auxiliary_gate.output_dim, 32)
        self.assertEqual(new.config.d_model, 128)


class BaselineCapTest(unittest.TestCase):
    def test_a_freed_card_is_available_only_while_the_cap_stays_full(self) -> None:
        self.assertFalse(baseline_blocks_claim(4, 2, 66))
        self.assertFalse(baseline_blocks_claim(3, 2, 66))
        self.assertFalse(baseline_blocks_claim(2, 2, 66))
        self.assertTrue(baseline_blocks_claim(1, 2, 66))
        self.assertTrue(baseline_blocks_claim(0, 2, 12))
        self.assertFalse(baseline_blocks_claim(0, 2, 0))
        self.assertFalse(baseline_blocks_claim(0, 0, 12))


if __name__ == "__main__":
    unittest.main()
