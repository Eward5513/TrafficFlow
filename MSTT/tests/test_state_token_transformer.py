"""CPU checks for the 48-token fusion head and the auxiliary gate path.

Run from the repository root:

    python MSTT/tests/test_state_token_transformer.py
"""

from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MSTT.config import load_config, model_config_from_saved, task_spec, validate_config  # noqa: E402
from MSTT.model.conservation import conservation_residual  # noqa: E402
from MSTT.model.graph import build_spatial_graph  # noqa: E402
from MSTT.model.mstt import MSTT, validate_state_token_transformer  # noqa: E402
from MSTT.model.state_fusion_head import (  # noqa: E402
    StateTokenTransformerHead,
    arrange_state_tokens,
    token_index,
)
from MSTT.train import apply_explicit_run_id, build_parser  # noqa: E402
from MSTT.trainer import assert_model_checkpoint_matches, model_config  # noqa: E402
from reimplementation.common.errors import ReimplementationError  # noqa: E402

FUSION_CONFIG = (
    PROJECT_ROOT
    / "MSTT/configs/experiment_state_independent_full_context_shared_node16_tsf_token_transformer_projected.json"
)
NODE16_CONFIG = (
    PROJECT_ROOT / "MSTT/configs/experiment_state_independent_full_context_shared_node16_tsf_mlp_projected.json"
)
GATE_CONFIG = PROJECT_ROOT / "MSTT/configs/all_sp_c_state_aux_gate_d128_projected_all.json"
MEAN = torch.tensor([0.2, -1.5, 3.0, 8.0])
STD = torch.tensor([0.5, 1.25, 2.0, 4.5])


def _graph(num_nodes: int):
    return build_spatial_graph(np.zeros((num_nodes, num_nodes), dtype=np.uint8), "bidirectional")


def _fusion_model(num_nodes: int, **overrides) -> MSTT:
    config = load_config(FUSION_CONFIG)
    config["num_nodes"] = num_nodes
    config.update(overrides)
    validate_config(config)
    built = model_config(config, task_spec("All2All"))
    return MSTT(built, _graph(num_nodes), target_mean=MEAN.clone(), target_std=STD.clone())


def _sample(batch: int, num_nodes: int, channels: int):
    torch.manual_seed(7)
    x = torch.randn(batch, 12, num_nodes, channels)
    tod = torch.randint(0, 288, (batch, 12))
    dow = torch.randint(0, 7, (batch, 12))
    return x, tod, dow


class FusionHeadTest(unittest.TestCase):
    def test_token_order_and_last_indexes(self) -> None:
        self.assertEqual([token_index(state, 11, 12) for state in range(4)], [11, 23, 35, 47])
        parts = []
        for state in range(4):
            part = torch.zeros(1, 12, 2, 3)
            part[0, 5, 1, 0] = float(state + 1)
            parts.append(part)
        flat = arrange_state_tokens(parts).reshape(1, 2, 48, 3)
        for state in range(4):
            self.assertEqual(flat[0, 1, state * 12 + 5, 0].item(), float(state + 1))
            self.assertEqual(flat[0, 0, state * 12 + 5, 0].item(), 0.0)
        head = StateTokenTransformerHead(4, 12, 4, token_dim=16, num_heads=4, ffn_hidden_dim=8, dropout=0.0, attention_dropout=0.0)
        self.assertEqual(head.last_token_indices(), [11, 23, 35, 47])

    def test_road_isolation_chunk_checkpoint_and_cross_token_grad(self) -> None:
        torch.manual_seed(3)
        head = StateTokenTransformerHead(
            8, 12, 4, token_dim=16, num_heads=4, ffn_hidden_dim=12,
            dropout=0.0, attention_dropout=0.0, chunk_size=None, use_checkpoint=False,
        )
        other = StateTokenTransformerHead(
            8, 12, 4, token_dim=16, num_heads=4, ffn_hidden_dim=12,
            dropout=0.0, attention_dropout=0.0, chunk_size=3, use_checkpoint=True,
        )
        other.load_state_dict(head.state_dict())
        hidden = torch.randn(2, 12, 5, 32)
        head.eval()
        shifted = hidden.clone()
        shifted[:, :, 2, :] += 4.0
        base = head(hidden)
        changed = head(shifted)
        self.assertTrue(torch.equal(base[:, :2], changed[:, :2]))
        self.assertTrue(torch.equal(base[:, 3:], changed[:, 3:]))
        self.assertFalse(torch.equal(base[:, 2], changed[:, 2]))

        left = hidden.detach().clone().requires_grad_(True)
        right = hidden.detach().clone().requires_grad_(True)
        head.train()
        other.train()
        out_left = head(left)
        out_right = other(right)
        torch.testing.assert_close(out_left, out_right, atol=1e-6, rtol=1e-6)
        out_left.sum().backward()
        out_right.sum().backward()
        torch.testing.assert_close(left.grad, right.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(head.projectors[0].weight.grad, other.projectors[0].weight.grad, atol=1e-5, rtol=1e-5)

        head.zero_grad(set_to_none=True)
        leaf = torch.randn(1, 12, 1, 32, requires_grad=True)
        output = head(leaf)
        output[0, 0, 0].backward()
        self.assertTrue(torch.isfinite(leaf.grad).all())
        self.assertGreater(leaf.grad[0, :11, 0, :8].abs().sum().item(), 0.0)
        self.assertGreater(leaf.grad[0, :, 0, 8:].abs().sum().item(), 0.0)
        head.eval()
        with torch.no_grad():
            weights = head.attention_map(leaf.detach(), [0])
        self.assertEqual(tuple(weights.shape), (1, 1, 4, 48, 48))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 1, 4, 48), atol=1e-5, rtol=1e-5)
        self.assertGreater(weights[0, 0, 0, 11, :11].sum().item(), 0.0)
        self.assertGreater(weights[0, 0, 0, 11, 12:].sum().item(), 0.0)

    def test_model_summary_conservation_grads_and_checkpoint_identity(self) -> None:
        model = _fusion_model(3, dropout=0.0, attention_dropout=0.0, embedding_dropout=0.0, fusion_chunk_size=4, fusion_checkpoint=True)
        self.assertEqual(model.branches[0].block.temporal.head_dim, 24)
        self.assertEqual(model.head.attention.head_dim, 32)
        self.assertIsNone(model.head.attention.rope)
        self.assertIsNotNone(model.branches[0].block.temporal.rope)
        self.assertEqual(model.projection.mlp[0].in_features, 420)
        summary = model.architecture_summary()
        self.assertEqual(summary["d_model"], 416)
        self.assertEqual(summary["branch_d_model"], 104)
        self.assertEqual(summary["fusion_token_dim"], 128)
        self.assertEqual(summary["fusion_tokens"], 48)
        self.assertEqual(summary["fusion_num_heads"], 4)
        self.assertEqual(summary["fusion_head_dim"], 32)
        self.assertEqual(summary["backbone_ffn_count"], 4)
        self.assertEqual(summary["fusion_ffn_count"], 1)
        self.assertEqual(summary["independent_ffn_count"], 4)
        self.assertEqual(summary["ffn_hidden_dims"], [176, 176, 176, 176])
        self.assertEqual(summary["fusion_ffn_hidden_dims"], [176])
        self.assertEqual(summary["projection_mlp"]["input_dim"], 420)
        self.assertEqual(summary["projection_condition_dim"], 416)
        self.assertEqual(summary["head"]["last_token_indices"], [11, 23, 35, 47])
        self.assertFalse(summary["head"]["use_rope"])

        x, tod, dow = _sample(2, 3, 4)
        seen = {}

        def _hook(_module, inputs, _output):
            seen["features"] = int(inputs[0].shape[-1])

        handle = model.projection.mlp[0].register_forward_hook(_hook)
        try:
            aux = model.forward_with_aux(x, tod, dow)
        finally:
            handle.remove()
        self.assertEqual(seen["features"], 420)
        self.assertEqual(tuple(aux["final_pred"].shape), (2, 3, 4))
        physical = aux["final_pred"] * STD + MEAN
        self.assertLess(float(conservation_residual(physical).abs().max()), 1e-4)
        self.assertTrue(torch.isfinite(aux["final_pred"]).all())

        model.train()
        loss = model(x, tod, dow).square().mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for state, projector in enumerate(model.head.projectors):
            self.assertGreater(float(projector.weight.grad.abs().sum()), 0.0, f"projector {state}")
            self.assertGreater(float(model.head.readouts[state].weight.grad.abs().sum()), 0.0, f"readout {state}")
            self.assertGreater(float(model.branches[state].block.temporal.qkv.weight.grad.abs().sum()), 0.0, f"branch {state}")
        self.assertGreater(float(model.head.attention.qkv.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.head.ffn.w_gate.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.head.state_id.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.head.time_id.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.shared_state_context.tod_embedding.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.shared_state_context.node_embedding.weight.grad.abs().sum()), 0.0)

        model.eval()
        clone = _fusion_model(3, dropout=0.0, attention_dropout=0.0, embedding_dropout=0.0, fusion_chunk_size=4, fusion_checkpoint=True)
        clone.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(clone(x, tod, dow), model(x, tod, dow))

        task = task_spec("All2All")
        current = model_config(load_config(FUSION_CONFIG), task).to_dict()
        previous = model_config(load_config(NODE16_CONFIG), task).to_dict()
        assert_model_checkpoint_matches({"config": {"model_config": current}}, current)
        with self.assertRaises(ReimplementationError):
            assert_model_checkpoint_matches({"config": {"model_config": previous}}, current)
        with self.assertRaises(ReimplementationError):
            assert_model_checkpoint_matches({"config": {"model_config": current}}, previous)
        narrowed = dict(current)
        narrowed["fusion_token_dim"] = 64
        with self.assertRaises(ReimplementationError):
            assert_model_checkpoint_matches({"config": {"model_config": current}}, narrowed)
        legacy = {"model_config": {key: value for key, value in previous.items() if not str(key).startswith("fusion_")}}
        restored = model_config_from_saved(legacy)
        self.assertEqual(restored["fusion_token_dim"], 128)
        assert_model_checkpoint_matches({"config": {"model_config": restored}}, previous)
        old_model = _fusion_model(3, head_type="flatten_mlp", dropout=0.0, attention_dropout=0.0, embedding_dropout=0.0)
        with self.assertRaises(RuntimeError):
            old_model.load_state_dict(model.state_dict(), strict=True)
        with self.assertRaises(ValueError):
            validate_state_token_transformer({"head_type": "state_token_transformer", "backbone_mode": "shared"})

        counted = _fusion_model(1406)
        self.assertEqual(counted.parameter_count(), counted.architecture_summary()["total_parameters"])
        print(f"fusion parameters={counted.parameter_count()}")

    def test_cli_accepts_fusion_fields_and_run_id(self) -> None:
        from MSTT.config import overlay_args

        help_buffer = io.StringIO()
        with self.assertRaises(SystemExit):
            with redirect_stdout(help_buffer):
                build_parser().parse_args(["--help"])
        text = help_buffer.getvalue()
        self.assertIn("state_token_transformer", text)
        self.assertIn("--fusion-token-dim", text)
        self.assertIn("--run-id", text)
        args = build_parser().parse_args([
            "--config", str(FUSION_CONFIG),
            "--task", "All2All",
            "--penetration", "p5_p15",
            "--dataset-id", "variable_1h_p5_p15",
            "--run-id", "fusion_run_1",
            "--device", "cuda:0",
        ])
        config = apply_explicit_run_id(overlay_args(load_config(args.config), args), args)
        validate_config(config)
        self.assertEqual(config["run_id"], "fusion_run_1")
        self.assertEqual(config["head_type"], "state_token_transformer")
        self.assertEqual(config["dataset_id"], "variable_1h_p5_p15")
        self.assertEqual(config["fusion_token_dim"], 128)
        self.assertEqual(config["fusion_num_heads"], 4)
        self.assertEqual(config["fusion_ffn_hidden_dim"], 176)
        self.assertEqual(config["d_model"], 416)
        self.assertEqual(config["attention_head_dim"], 24)
        self.assertIsNone(config["max_train_batches"])
        self.assertEqual(config["epochs"], 100)
        payload = json.loads(FUSION_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(payload["experiment_id"], "state_independent_full_context_shared_node16_tsf_token_transformer_projected")


class AuxiliaryGateTest(unittest.TestCase):
    def test_gate_starts_at_one_and_learns(self) -> None:
        config = load_config(GATE_CONFIG)
        config["num_nodes"] = 3
        config["dropout"] = 0.0
        config["attention_dropout"] = 0.0
        config["embedding_dropout"] = 0.0
        validate_config(config)
        built = model_config(config, task_spec("AllSpC2All"))
        self.assertEqual(built.in_channels, 6)
        self.assertEqual(built.out_channels, 4)
        self.assertEqual(built.d_model, 128)
        self.assertEqual(built.backbone_mode, "shared")
        self.assertEqual(built.head_type, "flatten")
        model = MSTT(built, _graph(3), target_mean=MEAN.clone(), target_std=STD.clone())
        gate = model.embedding.auxiliary_gate
        with torch.no_grad():
            initial = gate(torch.randn(2, 12, 3, 2))
        self.assertTrue(torch.allclose(initial, torch.ones_like(initial)))
        x, tod, dow = _sample(2, 3, 6)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
        model.train()
        first_grads = []
        for _step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = model(x, tod, dow)
            self.assertEqual(tuple(output.shape), (2, 3, 4))
            output.square().mean().backward()
            first_grads.append(float(gate.first.weight.grad.abs().sum()))
            self.assertGreater(float(gate.last.weight.grad.abs().sum()), 0.0)
            optimizer.step()
        self.assertEqual(first_grads[0], 0.0)
        self.assertGreater(first_grads[1], 0.0)
        model.eval()
        with torch.no_grad():
            opened = gate(x[..., 4:])
            shifted = gate(x[..., 4:] + 0.8)
        self.assertFalse(torch.allclose(opened, shifted))
        self.assertEqual(model.architecture_summary()["d_model"], 128)
        self.assertEqual(model.projection.mlp[0].in_features, 132)


if __name__ == "__main__":
    unittest.main()
