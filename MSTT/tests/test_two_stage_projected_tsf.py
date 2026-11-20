"""CPU checks for the two-stage projected TSF.

Run from the repository root:

    python MSTT/tests/test_two_stage_projected_tsf.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reimplementation.common.errors import ReimplementationError  # noqa: E402
from reimplementation.common.sampling_evaluation import aggregate_sampling_metrics  # noqa: E402
from MSTT.config import (  # noqa: E402
    load_config,
    overlay_args,
    task_spec,
    temporal_aux_settings,
    validate_config,
    validate_history_fusion_task,
)
from MSTT.model.conservation import conservation_residual, project_conservation  # noqa: E402
from MSTT.model.graph import build_spatial_graph  # noqa: E402
from MSTT.model.mstt import MSTT  # noqa: E402
from MSTT.train import apply_explicit_run_id, build_parser  # noqa: E402
from MSTT.trainer import (  # noqa: E402
    _save_split,
    _stream_two_stage_inflow,
    assert_model_checkpoint_matches,
    assert_temporal_aux_checkpoint,
    model_config,
    predict_with_aux,
    train_step,
)
from MSTT.utils.losses import RawScaleLoss, conservation_objective, objective_description, temporal_aux_terms  # noqa: E402

CONFIG = PROJECT_ROOT / "MSTT/configs/experiment_two_stage_projected_tsf.json"
PARENT = PROJECT_ROOT / "MSTT/configs/experiment_temporal_aux_history11_projected.json"
DUAL = PROJECT_ROOT / "MSTT/configs/experiment_dual_path_history_tsf_projected.json"
MEAN = torch.tensor([1.0, 2.0, 3.0, 4.0])
STD = torch.tensor([2.0, 2.0, 2.0, 2.0])
TASK = task_spec("All2All")


def _graph(num_nodes: int):
    return build_spatial_graph(np.zeros((num_nodes, num_nodes), dtype=np.uint8), "bidirectional")


def _config(num_nodes: int) -> dict:
    config = load_config(CONFIG)
    config["num_nodes"] = num_nodes
    validate_config(config)
    validate_history_fusion_task(config, TASK)
    return config


def _model(num_nodes: int, *, seed: int, path: Path | None = None) -> MSTT:
    torch.manual_seed(seed)
    config = load_config(path or CONFIG)
    config["num_nodes"] = num_nodes
    validate_config(config)
    return MSTT(
        model_config(config, TASK),
        _graph(num_nodes),
        target_mean=MEAN.clone(),
        target_std=STD.clone(),
    )


def _silence_dropout(model: MSTT) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0


def _observations(batch: int, nodes: int, *, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, 12, nodes, 4, generator=generator)
    tod = torch.randint(0, 288, (batch, 12), generator=generator)
    dow = torch.randint(0, 7, (batch, 12), generator=generator)
    return x, tod, dow


def _train_batch(batch: int, nodes: int, *, seed: int = 0) -> dict[str, torch.Tensor]:
    x, _tod, _dow = _observations(batch, nodes, seed=seed)
    sequence = torch.arange(batch * 12 * nodes * 4, dtype=torch.float32).reshape(batch, 12, nodes, 4)
    return {
        "x": x,
        "y": torch.zeros(batch, 1, nodes, 4),
        "y_raw": sequence[:, -1:].clone(),
        "y_seq_raw": sequence,
        "day_index": torch.zeros(batch, dtype=torch.long),
        "window_start_slot": torch.zeros(batch, dtype=torch.long),
        "target_slot": torch.full((batch,), 11, dtype=torch.long),
        "sample_index": torch.arange(batch),
    }


class _Run:
    def __init__(self, config, criterion):
        self.config = config
        self.criterion = criterion
        self.device = torch.device("cpu")
        self.last_train_components = {}


class TwoStageProjectedTSFTest(unittest.TestCase):
    def test_shared_modules_and_independent_stages(self) -> None:
        model = _model(4, seed=3)
        model.eval()
        self.assertIsNone(model.history_fusion)
        self.assertIsNone(model.dual_history)
        self.assertIsNotNone(model.stage2_block)
        self.assertIsNotNone(model.stage1_projection)
        self.assertTrue(torch.equal(model.head.proj.weight, torch.zeros_like(model.head.proj.weight)))
        self.assertTrue(torch.equal(model.head.proj.bias, torch.zeros_like(model.head.proj.bias)))
        stage1_params = {param.data_ptr() for param in model.blocks.parameters()}
        stage2_params = {param.data_ptr() for param in model.stage2_block.parameters()}
        self.assertTrue(stage1_params)
        self.assertTrue(stage2_params.isdisjoint(stage1_params))
        first_projection = {param.data_ptr() for param in model.stage1_projection.mlp.parameters()}
        second_projection = {param.data_ptr() for param in model.projection.mlp.parameters()}
        self.assertTrue(first_projection.isdisjoint(second_projection))
        torch.testing.assert_close(model.stage1_projection.target_mean, model.projection.target_mean)
        torch.testing.assert_close(model.stage1_projection.target_std, model.projection.target_std)
        self.assertNotEqual(model.stage1_projection.target_mean.data_ptr(), model.projection.target_mean.data_ptr())

        calls = {"embed": [], "adaptive": [], "mlps": []}
        real_embed = model.embedding.forward
        real_adaptive = model.adaptive.forward

        def counted_embed(values, tod=None, dow=None, **kwargs):
            calls["embed"].append((id(model.embedding), values.detach().clone()))
            return real_embed(values, tod, dow, **kwargs)

        def counted_adaptive(hidden):
            calls["adaptive"].append(id(model.adaptive))
            return real_adaptive(hidden)

        model.embedding.forward = counted_embed
        model.adaptive.forward = counted_adaptive
        wrapped = []
        for index, mlp in enumerate(model.embedding.value_mlps):
            real = mlp.forward

            def counted_mlp(values, real=real, index=index, mlp=mlp):
                calls["mlps"].append((index, id(mlp)))
                return real(values)

            mlp.forward = counted_mlp
            wrapped.append(mlp)
        head_calls = {"n": 0}
        real_head = model.head.forward

        def counted_head(hidden):
            head_calls["n"] += 1
            self.assertEqual(tuple(hidden.shape), (2, 12, 4, 128))
            return real_head(hidden)

        model.head.forward = counted_head
        x, tod, dow = _observations(2, 4, seed=8)
        outputs = model.forward_with_aux(x, tod, dow)
        self.assertEqual(len(calls["embed"]), 2)
        self.assertEqual(calls["embed"][0][0], calls["embed"][1][0])
        self.assertEqual(calls["adaptive"], [id(model.adaptive), id(model.adaptive)])
        self.assertEqual([item[0] for item in calls["mlps"]], [0, 1, 2, 3, 0, 1, 2, 3])
        self.assertEqual(len({item[1] for item in calls["mlps"]}), 4)
        self.assertEqual(head_calls["n"], 1)
        torch.testing.assert_close(calls["embed"][0][1], x)
        torch.testing.assert_close(calls["embed"][1][1], outputs["temporal_aux_pred"])
        self.assertEqual(tuple(outputs["temporal_aux_pred"].shape), (2, 12, 4, 4))
        self.assertEqual(tuple(outputs["raw_pred"].shape), (2, 4, 4))
        self.assertEqual(tuple(outputs["final_pred"].shape), (2, 4, 4))
        torch.testing.assert_close(outputs["raw_pred"], outputs["temporal_aux_pred"][:, -1])
        raw_physical = outputs["raw_pred"] * STD + MEAN
        final_physical = outputs["final_pred"] * STD + MEAN
        torch.testing.assert_close(final_physical, raw_physical, atol=1e-5, rtol=1e-5)
        self.assertLess(float(conservation_residual(final_physical).abs().max()), 1e-4)
        public = model(x, tod, dow)
        torch.testing.assert_close(public, outputs["final_pred"])

    def test_vectorized_projection_matches_twelve_steps(self) -> None:
        model = _model(3, seed=4)
        model.eval()
        captured = {}
        real_encode = model._encode
        real_aux = model.temporal_aux_head.forward

        def counted_encode(*args, **kwargs):
            hidden = real_encode(*args, **kwargs)
            captured["h1"] = hidden
            return hidden

        def counted_aux(hidden):
            raw = real_aux(hidden)
            captured["r1"] = raw
            return raw

        model._encode = counted_encode
        model.temporal_aux_head.forward = counted_aux
        x, tod, dow = _observations(2, 3, seed=9)
        outputs = model.forward_with_aux(x, tod, dow)
        vector = model.stage1_projection.project_aligned(captured["h1"], captured["r1"])["final_pred"]
        steps = []
        for step in range(12):
            one = model.stage1_projection.project_aligned(captured["h1"][:, step], captured["r1"][:, step])
            steps.append(one["final_pred"])
            physical = one["final_pred"] * STD + MEAN
            self.assertLess(float(conservation_residual(physical).abs().max()), 1e-4)
        torch.testing.assert_close(vector, torch.stack(steps, dim=1))
        torch.testing.assert_close(outputs["temporal_aux_pred"], vector)
        physical = vector * STD + MEAN
        self.assertLess(float(conservation_residual(physical).abs().max()), 1e-4)

    def test_uniform_projection_example(self) -> None:
        physical = torch.tensor([9.0, 4.0, 3.0, 6.0])
        weights = torch.full((4,), 0.25)
        corrected = project_conservation(physical, weights)
        torch.testing.assert_close(corrected, torch.tensor([8.5, 4.5, 2.5, 6.5]))
        self.assertLess(float(conservation_residual(corrected).abs()), 1e-6)
        module = _model(2, seed=1).stage1_projection
        aligned = module.project_aligned(
            torch.zeros(2, 128),
            physical.view(1, 4).expand(2, 4).contiguous(),
        )
        # Zero-initialized last layer makes the weights uniform. Mean/std then undo the example's raw scale.
        raw_physical = physical * STD + MEAN
        residual = float(conservation_residual(raw_physical))
        weights = torch.full((4,), 0.25)
        expected = project_conservation(raw_physical, weights)
        got = aligned["final_pred"][0] * STD + MEAN
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
        self.assertNotEqual(residual, 0.0)

    def test_all_steps_scope_and_label_independence(self) -> None:
        criterion = RawScaleLoss(torch.zeros(4), torch.ones(4))
        prediction = torch.zeros(2, 12, 3, 4)
        target = torch.zeros(2, 12, 3, 4)
        prediction[:, 11, :, 0] = 2.0
        outputs = {"temporal_aux_pred": prediction}
        all_steps = {"temporal_aux_enabled": True, "temporal_aux_scope": "all_steps", "temporal_aux_weight": 0.5}
        history = {"temporal_aux_enabled": True, "temporal_aux_scope": "history_except_last", "temporal_aux_weight": 0.5}
        aux, weighted = temporal_aux_terms(criterion, outputs, target, all_steps)
        old, old_weighted = temporal_aux_terms(criterion, outputs, target, history)
        self.assertAlmostEqual(float(old), 0.0, places=6)
        self.assertAlmostEqual(float(old_weighted), 0.0, places=6)
        manual = 2.0 * (2 * 3) / (2 * 12 * 3 * 4)
        self.assertAlmostEqual(float(aux), manual, places=6)
        self.assertAlmostEqual(float(weighted), 0.5 * manual, places=6)
        self.assertGreater(float(aux), 0.0)

        model = _model(3, seed=6)
        model.eval()
        batch = _train_batch(2, 3, seed=2)
        other = dict(batch)
        other["y_raw"] = batch["y_raw"] + 9
        other["y_seq_raw"] = batch["y_seq_raw"] + 4
        first = predict_with_aux(model, batch, amp="none", device=torch.device("cpu"))
        second = predict_with_aux(model, other, amp="none", device=torch.device("cpu"))
        torch.testing.assert_close(first["final_pred"], second["final_pred"])
        torch.testing.assert_close(first["temporal_aux_pred"], second["temporal_aux_pred"])
        self.assertEqual(tuple(first["temporal_aux_pred"].shape), (2, 12, 3, 4))
        self.assertEqual(tuple(first["final_pred"].shape), (2, 1, 3, 4))

    def test_gradients_train_step_and_checkpoint_identity(self) -> None:
        model = _model(3, seed=7)
        _silence_dropout(model)
        criterion = RawScaleLoss(MEAN, STD)
        x, tod, dow = _observations(2, 3, seed=5)
        outputs = model.forward_with_aux(x, tod, dow)
        target = torch.full_like(outputs["final_pred"], 10.0)
        criterion(outputs["final_pred"], target).backward()
        self.assertIsNotNone(model.head.proj.weight.grad)
        self.assertGreater(float(model.head.proj.weight.grad.abs().sum()), 0.0)

        opened = _model(3, seed=8)
        _silence_dropout(opened)
        with torch.no_grad():
            opened.head.proj.weight.normal_(0.0, 0.02)
            opened.head.proj.bias.normal_(0.0, 0.02)
        holder = {}
        real_project = opened.stage1_projection.project_aligned

        def remembered(features, raw):
            projected = real_project(features, raw)
            projected["final_pred"].retain_grad()
            holder["p1"] = projected["final_pred"]
            return projected

        opened.stage1_projection.project_aligned = remembered
        opened.zero_grad(set_to_none=True)
        opened_out = opened.forward_with_aux(x, tod, dow)
        criterion(opened_out["final_pred"], torch.full_like(opened_out["final_pred"], 10.0)).backward()
        self.assertGreater(float(holder["p1"].grad[:, :-1].abs().sum()), 0.0)
        stage2_grad = [
            float(param.grad.abs().sum())
            for param in opened.stage2_block.parameters()
            if param.grad is not None
        ]
        self.assertTrue(stage2_grad)
        self.assertGreater(max(stage2_grad), 0.0)

        config = _config(3)
        run = _Run(config, criterion)
        batch = _train_batch(2, 3, seed=11)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        model.zero_grad(set_to_none=True)
        loss, _norm = train_step(run, model, batch, optimizer, scaler, amp="none", clip=5.0, epoch=0)
        self.assertTrue(np.isfinite(loss))
        final = run.last_train_components["supervised_mae"]
        aux = run.last_train_components["temporal_aux_mae"]
        self.assertAlmostEqual(run.last_train_components["total_loss"], final + 0.5 * aux, delta=1e-4)
        self.assertEqual(
            objective_description(criterion, config),
            "final_mae_raw + 0.5 * stage1_projected_mae[0:12]",
        )
        broken = _train_batch(2, 3, seed=12)
        broken["y_raw"] = broken["y_raw"] + 1
        with self.assertRaises(ReimplementationError):
            train_step(run, model, broken, optimizer, scaler, amp="none", clip=0.0, epoch=0)

        restored = _model(3, seed=1)
        restored.load_state_dict(model.state_dict(), strict=True)
        model.eval()
        restored.eval()
        with torch.no_grad():
            left = model.forward_with_aux(x, tod, dow)["final_pred"]
            right = restored.forward_with_aux(x, tod, dow)["final_pred"]
        torch.testing.assert_close(left, right)
        audit = {
            "full_raw_content_sha256": "abc",
            "file_content_sha256": "def",
            "physical_definition_version": "independent_fcd_v1",
            "node_order_sha256": "ghi",
            "target_mean": [1, 2, 3, 4],
            "target_std": [2, 2, 2, 2],
            "slice": {"window": "full"},
            "train_days": ["2026-01-01"],
        }
        stored = dict(config)
        stored["model_config"] = model.config.to_dict()
        payload = {"config": stored, "temporal_aux_label_audit": audit}
        assert_temporal_aux_checkpoint(payload, config, audit)
        assert_model_checkpoint_matches(payload, model.config.to_dict())
        changed = dict(config)
        changed["temporal_aux_scope"] = "history_except_last"
        with self.assertRaises(ReimplementationError) as raised:
            assert_temporal_aux_checkpoint(payload, changed, audit)
        self.assertIn("temporal_aux_scope", str(raised.exception))
        other_head = dict(model.config.to_dict())
        other_head["head_type"] = "flatten"
        with self.assertRaises(ReimplementationError) as raised_head:
            assert_model_checkpoint_matches({"config": {"model_config": other_head}}, model.config.to_dict())
        self.assertIn("head_type", str(raised_head.exception))

    def test_stage_metrics_round_trip_and_cli(self) -> None:
        criterion = RawScaleLoss(torch.zeros(4), torch.ones(4))
        outputs = {
            "temporal_aux_pred": torch.zeros(2, 12, 2, 4),
            "raw_pred": torch.zeros(2, 1, 2, 4),
            "final_pred": torch.zeros(2, 1, 2, 4),
        }
        outputs["temporal_aux_pred"][:, -1, :, 0] = 2.0
        outputs["raw_pred"][..., 0] = 3.0
        outputs["final_pred"][..., 0] = 4.0
        batch = {"y_raw": torch.zeros(2, 1, 2, 4)}
        totals = {}
        _stream_two_stage_inflow(outputs, batch, criterion, totals)
        self.assertAlmostEqual(totals["stage1_last"] / totals["stage1_last_count"], 2.0)
        self.assertAlmostEqual(totals["raw"] / totals["raw_count"], 3.0)
        self.assertAlmostEqual(totals["final"] / totals["final_count"], 4.0)
        per_seed = {}
        for offset, seed in enumerate((42, 43, 44, 45, 46)):
            per_seed[str(seed)] = {
                "mae": 1.0 + offset,
                "inflow_mae": 4.0 + offset,
                "stage1_last_inflow_mae": 2.0 + offset,
                "raw_inflow_mae": 3.0 + offset,
            }
        report = aggregate_sampling_metrics(per_seed)
        packed = {
            "sampling_metrics": report,
            "sampling_predictions": {str(seed): {"pred_raw": np.zeros((1, 1, 1, 4), dtype=np.float32)} for seed in range(42, 47)},
            "metrics": report["mean"],
        }

        class RunDir:
            def __init__(self, path: Path):
                self.run_dir = path

        with tempfile.TemporaryDirectory() as directory:
            run = RunDir(Path(directory))
            _save_split(run, "test", dict(report["mean"]), packed, False)
            saved = json.loads((run.run_dir / "test_stage_inflow.json").read_text(encoding="utf-8"))
            sampling = json.loads((run.run_dir / "test_sampling_metrics.json").read_text(encoding="utf-8"))
        for key in ("stage1_last_inflow_mae", "raw_inflow_mae", "inflow_mae"):
            self.assertIn(key, saved["mean"])
            self.assertIn(key, saved["std"])
            self.assertIn(key, saved["per_seed"]["42"])
            self.assertIn(key, sampling["mean"])
            self.assertEqual(saved["mean"][key], sampling["mean"][key])

        args = build_parser().parse_args([
            "--task", "All2All",
            "--penetration", "p5_p15",
            "--dataset-id", "variable_30min_p5_p15",
            "--config", str(CONFIG),
            "--run-id", "cpu_two_stage_identity",
        ])
        loaded = apply_explicit_run_id(overlay_args(load_config(args.config), args), args)
        validate_config(loaded)
        validate_history_fusion_task(loaded, TASK)
        self.assertEqual(loaded["experiment_id"], "two_stage_projected_tsf_shared_embedding_v1")
        self.assertEqual(loaded["history_fusion_mode"], "two_stage_projected_tsf")
        self.assertEqual(loaded["temporal_aux_scope"], "all_steps")
        self.assertEqual(loaded["head_type"], "last")
        self.assertEqual(loaded["backbone_mode"], "shared")
        self.assertEqual(temporal_aux_settings(loaded)["branch_a_loss_weight"], 0.0)
        self.assertEqual(loaded["run_id"], "cpu_two_stage_identity")
        self.assertEqual(loaded["dataset_id"], "variable_30min_p5_p15")

    def test_old_modes_keep_their_checks(self) -> None:
        parent = load_config(PARENT)
        parent["num_nodes"] = 3
        validate_config(parent)
        self.assertEqual(parent["temporal_aux_scope"], "history_except_last")
        parent["temporal_aux_scope"] = "all_steps"
        with self.assertRaises(ValueError):
            validate_config(parent)
        dual = load_config(DUAL)
        validate_config(dual)
        validate_history_fusion_task(dual, TASK)
        plain = _model(3, seed=2, path=PARENT)
        self.assertIsNone(plain.stage2_block)
        self.assertEqual(plain.architecture_summary()["temporal_aux_head"]["supervised_steps"], "0..10")
        x, tod, dow = _observations(1, 3, seed=1)
        outputs = plain.forward_with_aux(x, tod, dow)
        self.assertEqual(tuple(outputs["temporal_aux_pred"].shape), (1, 12, 3, 4))
        summary = _model(3, seed=2).architecture_summary()
        self.assertEqual(summary["backbone_ffn_count"], 1)
        self.assertEqual(summary["stage2_ffn_count"], 1)
        self.assertEqual(summary["temporal_attention_count"], 2)
        self.assertEqual(summary["temporal_aux_head"]["supervised_steps"], "0..11")
        self.assertEqual(summary["two_stage"]["residual_head_input_width"], 128)
        self.assertEqual(summary["two_stage"]["stage1_blocks"], 1)
        self.assertEqual(summary["two_stage"]["stage2_blocks"], 1)
        rejected = _config(3)
        rejected["head_type"] = "flatten"
        with self.assertRaises(ValueError):
            validate_config(rejected)
        with self.assertRaises(ValueError):
            validate_history_fusion_task(load_config(CONFIG), task_spec("All2I"))
        full = _model(1406, seed=0)
        self.assertEqual(full.parameter_count(), 1513288)
        self.assertEqual(full.architecture_summary()["total_parameters"], 1513288)

    def test_objective_uses_each_term_once(self) -> None:
        model = _model(3, seed=9)
        model.eval()
        criterion = RawScaleLoss(MEAN, STD)
        batch = _train_batch(2, 3, seed=4)
        outputs = predict_with_aux(model, batch, amp="none", device=torch.device("cpu"))
        config = _config(3)
        loss, parts = conservation_objective(criterion, outputs, batch["y_raw"], config)
        aux, weighted = temporal_aux_terms(criterion, outputs, batch["y_seq_raw"], config)
        manual_final = criterion(outputs["final_pred"], batch["y_raw"])
        manual_stage = criterion(outputs["temporal_aux_pred"], batch["y_seq_raw"])
        torch.testing.assert_close(parts["supervised_mae"], manual_final)
        torch.testing.assert_close(aux, manual_stage)
        torch.testing.assert_close(loss + weighted, manual_final + 0.5 * manual_stage)


if __name__ == "__main__":
    unittest.main()
