"""CPU checks for full-context state branches, shared or independent.

Run from the repository root:

    python MSTT/tests/test_state_context_sharing.py
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MSTT.config import (  # noqa: E402
    STATE_CHANNELS,
    load_config,
    model_config_from_saved,
    task_spec,
    validate_backbone_task,
    validate_config,
    validate_conservation_task,
    validate_value_embedding_task,
)
from MSTT.model.conservation import conservation_residual  # noqa: E402
from MSTT.model.embedding import InputEmbedding  # noqa: E402
from MSTT.model.graph import build_spatial_graph  # noqa: E402
from MSTT.model.mstt import MSTT, MSTTConfig, validate_independent_backbone, validate_state_node_sharing  # noqa: E402
from MSTT.trainer import (  # noqa: E402
    assert_model_checkpoint_matches,
    assert_run_dir_matches,
    build_optimizer,
    merge_summary,
    model_config,
    run_directory,
)
from reimplementation.common.errors import ReimplementationError  # noqa: E402

MEAN = torch.tensor([0.2, -1.5, 3.0, 8.0])
STD = torch.tensor([0.5, 1.25, 2.0, 4.5])
OLD_INDEPENDENT = PROJECT_ROOT / "MSTT/configs/experiment_state_independent_tsf_mlp_projected.json"
OLD_SHARED = PROJECT_ROOT / "MSTT/configs/experiment_shared_tsf_mlp_projected.json"
NEW_SHARED = PROJECT_ROOT / "MSTT/configs/experiment_state_independent_full_context_shared_tsf_mlp_projected.json"
NEW_INDEPENDENT = PROJECT_ROOT / "MSTT/configs/experiment_state_independent_full_context_independent_tsf_mlp_projected.json"
NODE16_SHARED = PROJECT_ROOT / "MSTT/configs/experiment_state_independent_full_context_shared_node16_tsf_mlp_projected.json"
NODE16_INDEPENDENT = PROJECT_ROOT / "MSTT/configs/experiment_state_independent_full_context_independent_node16_tsf_mlp_projected.json"
CONTEXT_TABLES = 288 * 8 + 7 * 8 + 12 * 1406 * 64
NODE16_CONTEXT_COPY = 1406 * 16 + CONTEXT_TABLES


def _graph(num_nodes: int):
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.uint8)
    return build_spatial_graph(adjacency, "bidirectional")


def _full_config(num_nodes: int, sharing: str, **overrides) -> dict:
    config = {
        "num_nodes": num_nodes,
        "in_channels": 4,
        "out_channels": 4,
        "input_window": 12,
        "d_model": 368,
        "num_heads": 4,
        "temporal_num_heads": 4,
        "spatial_num_heads": 4,
        "attention_head_dim": 24,
        "num_layers": 1,
        "ffn_hidden_dim": 176,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "embedding_dropout": 0.0,
        "use_qk_norm": True,
        "use_rope": True,
        "use_gated_attention": False,
        "spatial_attention": "global",
        "use_edge_type_bias": False,
        "spatial_checkpoint": True,
        "use_tod": True,
        "use_dow": True,
        "use_node_embedding": True,
        "embedding_fusion": "concat",
        "value_embedding_mode": "separate_mlp",
        "value_embedding_dim": 32,
        "node_embedding_dim": 16,
        "tod_embedding_dim": 32,
        "dow_embedding_dim": 32,
        "use_adaptive_embedding": True,
        "base_embedding_dim": 112,
        "adaptive_embedding_dim": 256,
        "adaptive_embedding_fusion": "concat",
        "ffn_placement": "after_spatial",
        "attention_order": "TS",
        "attention_layout": "factorized_ts",
        "steps_per_day": 288,
        "head_type": "flatten_mlp",
        "conservation_mode": "projected",
        "projection_mlp_hidden_dim": 32,
        "backbone_mode": "state_independent",
        "state_context_sharing": sharing,
        "fusion_mlp_hidden_dim": 128,
    }
    config.update(overrides)
    return config


def _build(num_nodes: int, sharing: str, **overrides) -> MSTT:
    config = MSTTConfig.from_mapping(_full_config(num_nodes, sharing, **overrides))
    return MSTT(config, _graph(num_nodes), target_mean=MEAN.clone(), target_std=STD.clone())


def _legacy_independent_config(num_nodes: int, **overrides) -> dict:
    config = _full_config(
        num_nodes,
        "independent",
        d_model=128,
        temporal_num_heads=None,
        spatial_num_heads=None,
        attention_head_dim=None,
        tod_embedding_dim=8,
        dow_embedding_dim=8,
        base_embedding_dim=64,
        adaptive_embedding_dim=64,
    )
    config.update(overrides)
    return config


def _build_from_mapping(payload: dict) -> MSTT:
    return MSTT(
        MSTTConfig.from_mapping(payload),
        _graph(int(payload["num_nodes"])),
        target_mean=MEAN.clone(),
        target_std=STD.clone(),
    )


def _sample(num_nodes: int, seed: int = 123):
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(2, 12, num_nodes, 4, generator=generator)
    inputs[..., 0] *= 0.3
    inputs[..., 1] += 1.7
    inputs[..., 2] *= -0.8
    inputs[..., 3] += 0.4
    tod = torch.randint(0, 288, (2, 12), generator=generator)
    dow = torch.randint(0, 7, (2, 12), generator=generator)
    target = torch.randn(2, num_nodes, 4, generator=generator)
    target[..., 0] += 2.0
    target[..., 3] -= 1.3
    return inputs, tod, dow, target


def _branch_outputs(model: MSTT, inputs, tod, dow):
    if model.config.state_context_sharing == "shared":
        context = model.shared_state_context(inputs, tod, dow)
        return [
            branch(inputs[..., channel:channel + 1], tod, dow, context=context)
            for channel, branch in enumerate(model.branches)
        ]
    return [
        branch(inputs[..., channel:channel + 1], tod, dow)
        for channel, branch in enumerate(model.branches)
    ]


def _context_parameter(name: str) -> bool:
    return (
        name.endswith("tod_embedding.weight")
        or name.endswith("dow_embedding.weight")
        or name.endswith("adaptive_embedding")
    )


def _align(shared: MSTT, independent: MSTT) -> None:
    shared_state = shared.state_dict()
    independent_state = independent.state_dict()
    copied = set()
    with torch.no_grad():
        for key, value in shared_state.items():
            if key.startswith("shared_state_context."):
                continue
            if key not in independent_state:
                raise AssertionError(f"shared parameter {key} is missing on the independent model")
            independent_state[key].copy_(value)
            copied.add(key)
        tod = shared.shared_state_context.tod_embedding.weight
        dow = shared.shared_state_context.dow_embedding.weight
        adaptive = shared.shared_state_context.adaptive.adaptive_embedding
        for index, _branch in enumerate(independent.branches):
            for suffix, source in (
                ("embedding.tod_embedding.weight", tod),
                ("embedding.dow_embedding.weight", dow),
                ("adaptive.adaptive_embedding", adaptive),
            ):
                key = f"branches.{index}.{suffix}"
                independent_state[key].copy_(source)
                copied.add(key)
    missing = set(independent_state) - copied
    if missing:
        raise AssertionError(f"independent parameters were not copied: {sorted(missing)}")


class StateContextSharingTest(unittest.TestCase):
    def test_configs_and_rejected_combinations(self) -> None:
        shared = json.loads(NEW_SHARED.read_text(encoding="utf-8"))
        independent = json.loads(NEW_INDEPENDENT.read_text(encoding="utf-8"))
        self.assertEqual(shared["experiment_id"], "state_independent_full_context_shared_tsf_mlp_projected")
        self.assertEqual(independent["experiment_id"], "state_independent_full_context_independent_tsf_mlp_projected")
        self.assertEqual(shared["state_context_sharing"], "shared")
        self.assertEqual(independent["state_context_sharing"], "independent")
        self.assertNotIn("state_node_sharing", shared)
        self.assertNotIn("state_node_sharing", independent)
        self.assertEqual(shared["dataset_id"], "variable_30min_p10_p20")
        left = {key: value for key, value in shared.items() if key not in {"experiment_id", "state_context_sharing"}}
        right = {key: value for key, value in independent.items() if key not in {"experiment_id", "state_context_sharing"}}
        self.assertEqual(left, right)

        loaded_shared = load_config(NEW_SHARED)
        loaded_independent = load_config(NEW_INDEPENDENT)
        for config, sharing in ((loaded_shared, "shared"), (loaded_independent, "independent")):
            validate_config(config)
            validate_backbone_task(config, task_spec("All2All"))
            built = model_config(config, task_spec("All2All"))
            self.assertEqual(built.state_context_sharing, sharing)
            self.assertEqual(built.state_node_sharing, "independent")
            self.assertEqual(built.d_model, 368)
            self.assertEqual(built.attention_head_dim, 24)
            self.assertEqual(built.ffn_hidden_dim, 176)
            self.assertEqual(built.tod_embedding_dim, 32)

        old = load_config(OLD_INDEPENDENT)
        self.assertEqual(old["state_context_sharing"], "independent")
        self.assertEqual(old["d_model"], 128)
        self.assertEqual(old["tod_embedding_dim"], 8)
        frozen = load_config(None)
        self.assertEqual(frozen["backbone_mode"], "shared")
        self.assertEqual(frozen["state_context_sharing"], "independent")
        self.assertEqual(frozen["state_node_sharing"], "independent")
        self.assertEqual(frozen["head_type"], "flatten")

        with self.assertRaises(ValueError):
            validate_independent_backbone(_full_config(4, "independent", attention_head_dim=None))
        rejected = dict(frozen)
        rejected["state_context_sharing"] = "shared"
        with self.assertRaises(ValueError):
            validate_config(rejected)
        rejected_node = dict(frozen)
        rejected_node["state_node_sharing"] = "shared"
        with self.assertRaises(ValueError):
            validate_config(rejected_node)
        with self.assertRaises(ValueError):
            validate_state_node_sharing(_full_config(4, "independent", state_node_sharing="shared"))
        with self.assertRaises(ValueError):
            validate_backbone_task(rejected, task_spec("I2I"))
        with self.assertRaises(ValueError):
            validate_backbone_task(loaded_shared, task_spec("AllSpC2All"))
        with self.assertRaises(ValueError):
            InputEmbedding(
                in_channels=1,
                d_model=28,
                num_nodes=4,
                fusion="concat",
                value_embedding_mode="separate_mlp",
                value_dim=8,
                node_dim=4,
                tod_dim=8,
                dow_dim=8,
                use_tod=False,
                use_dow=True,
                external_calendar_embeddings=True,
            )

    def test_old_models_roundtrip_when_the_field_is_missing(self) -> None:
        old = _build_from_mapping(_legacy_independent_config(4))
        self.assertFalse(hasattr(old, "shared_state_context"))
        self.assertEqual(old.branches[0].block.temporal.d_model, 32)
        self.assertEqual(tuple(old.branches[0].embedding.tod_embedding.weight.shape), (288, 2))
        self.assertEqual(old.branches[0].summary()["adaptive"]["parameters"] > 0, True)
        raw = {"model_config": {key: value for key, value in old.config.to_dict().items() if key != "state_context_sharing"}}
        restored = model_config_from_saved(raw)
        self.assertEqual(restored["state_context_sharing"], "independent")
        rebuilt = _build_from_mapping(restored)
        rebuilt.load_state_dict(old.state_dict())
        self.assertEqual(rebuilt.branches[0].block.temporal.head_dim, 8)
        summary = rebuilt.architecture_summary()
        self.assertEqual(summary["head"]["input_dim"], 1536)
        self.assertEqual(summary["projection_mlp"]["input_dim"], 132)
        self.assertEqual(summary["context_tables"]["copies"], 4)
        self.assertEqual(summary["context_tables"]["tod_shape"], [288, 2])
        self.assertTrue(all(value == "independent" for value in summary["embedding"]["parameter_sharing"].values()))

        legacy_shared = _build_from_mapping(_legacy_independent_config(
            4,
            backbone_mode="shared",
            state_context_sharing="independent",
            d_model=128,
            value_embedding_dim=32,
            node_embedding_dim=16,
            tod_embedding_dim=8,
            dow_embedding_dim=8,
            base_embedding_dim=64,
            adaptive_embedding_dim=64,
            in_channels=4,
            out_channels=4,
        ))
        omitted = {
            "model_config": {
                key: value for key, value in legacy_shared.config.to_dict().items() if key != "state_context_sharing"
            }
        }
        restored_shared = model_config_from_saved(omitted)
        self.assertEqual(restored_shared["backbone_mode"], "shared")
        self.assertEqual(restored_shared["state_context_sharing"], "independent")
        reloaded_shared = _build_from_mapping(restored_shared)
        reloaded_shared.load_state_dict(legacy_shared.state_dict())
        with self.assertRaises(ValueError):
            _build_from_mapping(_legacy_independent_config(4, backbone_mode="shared", state_context_sharing="shared", d_model=128))

    def test_forward_alignment_gradients_and_conservation(self) -> None:
        num_nodes = 4
        torch.manual_seed(1)
        shared = _build(num_nodes, "shared")
        torch.manual_seed(2)
        independent = _build(num_nodes, "independent")
        inputs, tod, dow, target = _sample(num_nodes)
        self._assert_shapes_and_summaries(shared, independent, inputs, tod, dow)
        self._assert_identity(shared, independent)
        self._assert_finite_backward(shared, independent, inputs, tod, dow, target)

        shared.eval()
        independent.eval()
        self.assertFalse(torch.allclose(shared(inputs, tod, dow), independent(inputs, tod, dow), atol=1e-3, rtol=1e-3))
        _align(shared, independent)
        shared_parts = _branch_outputs(shared, inputs, tod, dow)
        independent_parts = _branch_outputs(independent, inputs, tod, dow)
        for left, right in zip(shared_parts, independent_parts):
            torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)
        shared_aux = shared.forward_with_aux(inputs, tod, dow)
        independent_aux = independent.forward_with_aux(inputs, tod, dow)
        torch.testing.assert_close(shared_aux["raw_pred"], independent_aux["raw_pred"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(shared_aux["final_pred"], independent_aux["final_pred"], atol=1e-5, rtol=1e-5)

        shared.zero_grad(set_to_none=True)
        independent.zero_grad(set_to_none=True)
        shared.train()
        independent.train()
        shared_loss = (shared(inputs, tod, dow) * target).square().mean()
        independent_loss = (independent(inputs, tod, dow) * target).square().mean()
        torch.testing.assert_close(shared_loss, independent_loss, atol=1e-5, rtol=1e-5)
        shared_loss.backward()
        independent_loss.backward()
        self._assert_context_grad_is_sum(shared, independent)

        self._assert_channel_perturbation(shared, inputs, tod, dow)
        self._assert_channel_perturbation(independent, inputs, tod, dow)
        self._assert_physical_conservation(shared, inputs, tod, dow)
        self._assert_physical_conservation(independent, inputs, tod, dow)
        self._assert_strict_reload(shared, inputs, tod, dow)
        self._assert_strict_reload(independent, inputs, tod, dow)
        fresh = _build(num_nodes, "shared")
        with self.assertRaises(RuntimeError):
            fresh.load_state_dict(independent.state_dict())
        old = _build_from_mapping(_legacy_independent_config(num_nodes))
        with self.assertRaises(RuntimeError):
            shared.load_state_dict(old.state_dict())

    def test_checkpoint_and_run_directory_identity(self) -> None:
        task = task_spec("All2All")
        independent_cfg = load_config(NEW_INDEPENDENT)
        shared_cfg = load_config(NEW_SHARED)
        independent_model = model_config(independent_cfg, task).to_dict()
        shared_model = model_config(shared_cfg, task).to_dict()
        self.assertEqual(
            {key for key in shared_model if shared_model[key] != independent_model.get(key)},
            {"state_context_sharing"},
        )
        payload = {"config": {"model_config": shared_model, "state_context_sharing": "shared"}}
        assert_model_checkpoint_matches(payload, shared_model)
        with self.assertRaises(ReimplementationError) as mismatch:
            assert_model_checkpoint_matches(payload, independent_model)
        self.assertIn("state_context_sharing", str(mismatch.exception))

        old_cfg = load_config(OLD_INDEPENDENT)
        old_model = model_config(old_cfg, task).to_dict()
        old_payload = {"config": dict(json.loads(OLD_INDEPENDENT.read_text(encoding="utf-8")))}
        old_payload["config"]["model_config"] = {
            key: value for key, value in old_model.items() if key != "state_context_sharing"
        }
        assert_model_checkpoint_matches(old_payload, old_model)
        with self.assertRaises(ReimplementationError) as width_mismatch:
            assert_model_checkpoint_matches(old_payload, independent_model)
        self.assertIn("d_model", str(width_mismatch.exception))

        kept = dict(shared_model)
        del kept["state_context_sharing"]
        restored = model_config_from_saved({"state_context_sharing": "shared", "model_config": kept})
        self.assertEqual(restored["state_context_sharing"], "shared")
        with self.assertRaises(ValueError):
            model_config_from_saved({"state_context_sharing": "shared", "model_config": independent_model})

        stored = json.loads(NEW_INDEPENDENT.read_text(encoding="utf-8"))
        stored["model_config"] = dict(independent_model)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "resolved_config.json").write_text(json.dumps(stored), encoding="utf-8")
            assert_run_dir_matches(run_dir, independent_cfg, task)
            with self.assertRaises(ReimplementationError) as resume_mismatch:
                assert_run_dir_matches(run_dir, shared_cfg, task)
            self.assertIn("state_context_sharing", str(resume_mismatch.exception))

        legacy = json.loads(OLD_INDEPENDENT.read_text(encoding="utf-8"))
        legacy["model_config"] = {key: value for key, value in old_model.items() if key != "state_context_sharing"}
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "resolved_config.json").write_text(json.dumps(legacy), encoding="utf-8")
            assert_run_dir_matches(run_dir, old_cfg, task)
            conflict = dict(legacy)
            conflict["state_context_sharing"] = "shared"
            (run_dir / "resolved_config.json").write_text(json.dumps(conflict), encoding="utf-8")
            with self.assertRaises(ReimplementationError):
                assert_run_dir_matches(run_dir, old_cfg, task)

    def test_full_network_parameter_delta(self) -> None:
        self.assertEqual(CONTEXT_TABLES, 1_082_168)
        counts = {}
        summaries = {}
        for label, path in (("old_independent", OLD_INDEPENDENT), ("old_shared", OLD_SHARED), ("new_shared", NEW_SHARED), ("new_independent", NEW_INDEPENDENT)):
            config = load_config(path)
            built = model_config(config, task_spec("All2All"))
            model = MSTT(built, _graph(int(config["num_nodes"])), target_mean=MEAN.clone(), target_std=STD.clone())
            counts[label] = model.parameter_count()
            summaries[label] = model.architecture_summary()
            state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            del model
            reloaded = MSTT(built, _graph(int(config["num_nodes"])), target_mean=MEAN.clone(), target_std=STD.clone())
            reloaded.load_state_dict(state)
            del reloaded
        self.assertEqual(counts["old_independent"], 1_407_648)
        self.assertEqual(counts["old_shared"], 1_505_952)
        self.assertEqual(counts["new_independent"] - counts["new_shared"], 3 * CONTEXT_TABLES)
        shared_tables = summaries["new_shared"]["context_tables"]
        independent_tables = summaries["new_independent"]["context_tables"]
        self.assertEqual(shared_tables["copies"], 1)
        self.assertEqual(independent_tables["copies"], 4)
        self.assertEqual(shared_tables["parameters_per_copy"], CONTEXT_TABLES)
        self.assertEqual(shared_tables["parameters_total"], CONTEXT_TABLES)
        self.assertEqual(independent_tables["parameters_total"], 4 * CONTEXT_TABLES)
        self.assertEqual(shared_tables["tod_shape"], [288, 8])
        self.assertEqual(shared_tables["dow_shape"], [7, 8])
        self.assertEqual(shared_tables["adaptive_shape"], [12, 1406, 64])
        self.assertEqual(shared_tables["per_branch_feature_width"]["d_model"], 92)
        self.assertEqual(shared_tables["concat_feature_width"]["d_model"], 368)
        self.assertEqual(summaries["new_shared"]["node_tables"]["sharing"], "independent")
        self.assertEqual(summaries["new_shared"]["node_tables"]["copies"], 4)
        self.assertEqual(summaries["new_shared"]["node_tables"]["shape"], [1406, 4])
        self.assertEqual(summaries["new_shared"]["node_tables"]["parameters_total"], 4 * 1406 * 4)
        self.assertEqual(summaries["new_independent"]["node_tables"]["copies"], 4)
        self.assertEqual(summaries["new_independent"]["node_tables"]["shape"], [1406, 4])
        self.assertEqual(sum(branch["adaptive"]["parameters"] for branch in summaries["new_shared"]["per_branch"]), 0)
        self.assertEqual(
            sum(branch["adaptive"]["parameters"] for branch in summaries["new_independent"]["per_branch"]),
            4 * 12 * 1406 * 64,
        )
        self.assertEqual(counts["new_shared"], 2_161_760)
        self.assertEqual(counts["new_independent"], 5_408_264)

    def test_cli_help_and_dataset_manifest(self) -> None:
        help_text = subprocess.run(
            [sys.executable, "MSTT/train.py", "--help"],
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        ).stdout
        self.assertIn("--state-context-sharing", help_text)
        self.assertIn("--state-node-sharing", help_text)
        from MSTT.train import build_parser

        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                build_parser().parse_args(["--task", "All2All", "--penetration", "p10_p20", "--state-context-sharing", "both"])
        from MSTT.config import overlay_args

        for path, sharing in ((NEW_SHARED, "shared"), (NEW_INDEPENDENT, "independent")):
            args = build_parser().parse_args([
                "--config", str(path),
                "--task", "All2All",
                "--penetration", "p10_p20",
                "--dataset-id", "variable_30min_p10_p20",
                "--device", "cuda:0",
            ])
            config = overlay_args(load_config(args.config), args)
            validate_config(config)
            validate_conservation_task(config, task_spec("All2All"))
            validate_backbone_task(config, task_spec("All2All"))
            validate_value_embedding_task(config, task_spec("All2All"))
            self.assertEqual(config["dataset_id"], "variable_30min_p10_p20")
            self.assertIn("p10_p20", config["observation_keys"])
            self.assertEqual(config["state_context_sharing"], sharing)
            self.assertEqual(config["state_node_sharing"], "independent")
            self.assertEqual(config["device"], "cuda:0")
            resolved = run_directory(Path(config["output_root"]), "p10_p20", "All2All", int(config["seed"]), config)
            text = resolved.as_posix()
            self.assertIn("datasets/variable_30min_p10_p20/", text)
            self.assertIn(f"physical_{config['physical_definition_version']}/MSTT/", text)
            self.assertIn("/in_all__out_all/p10_p20/", text)
            self.assertIn(f"/{config['experiment_id']}/seed_42/", text)

    def _assert_shapes_and_summaries(self, shared, independent, inputs, tod, dow) -> None:
        calls = {"n": 0}
        original = shared.shared_state_context.forward

        def wrapped(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        shared.shared_state_context.forward = wrapped
        try:
            output = shared(inputs, tod, dow)
        finally:
            shared.shared_state_context.forward = original
        self.assertEqual(calls["n"], 1)
        self.assertEqual(tuple(output.shape), (2, 4, 4))
        for model, copies, source in ((shared, 1, "external"), (independent, 4, "parameter")):
            model.eval()
            parts = _branch_outputs(model, inputs, tod, dow)
            self.assertEqual(len(parts), 4)
            for part in parts:
                self.assertEqual(tuple(part.shape), (2, 12, 4, 92))
            summary = model.architecture_summary()
            self.assertEqual(summary["state_channels"], list(STATE_CHANNELS))
            self.assertEqual(summary["temporal_attention_count"], 4)
            self.assertEqual(summary["independent_ffn_count"], 4)
            self.assertEqual(summary["ffn_hidden_dims"], [176, 176, 176, 176])
            self.assertEqual(summary["head"]["input_dim"], 4416)
            self.assertEqual(summary["head"]["hidden_dim"], 128)
            self.assertEqual(summary["head"]["output_dim"], 4)
            self.assertEqual(summary["projection_mlp"]["input_dim"], 372)
            self.assertEqual(summary["context_tables"]["copies"], copies)
            widths = summary["context_tables"]["per_branch_feature_width"]
            self.assertEqual(
                [widths[key] for key in ("value", "node", "tod", "dow", "base", "adaptive", "d_model")],
                [8, 4, 8, 8, 28, 64, 92],
            )
            self.assertEqual(summary["embedding"]["parameter_sharing"]["tsf"], "independent")
            self.assertEqual(summary["embedding"]["parameter_sharing"]["tod"], model.config.state_context_sharing)
            self.assertNotIn("shared_parameters", summary["embedding"])
            for branch in model.branches:
                block = branch.block
                self.assertEqual(block.temporal.qkv.in_features, 92)
                self.assertEqual(block.temporal.qkv.out_features, 3 * 96)
                self.assertEqual(block.temporal.out_proj.in_features, 96)
                self.assertEqual(block.temporal.out_proj.out_features, 92)
                self.assertEqual(block.temporal.head_dim, 24)
                self.assertEqual(block.spatial.head_dim, 24)
                self.assertEqual(block.temporal.inner_dim, 96)
                mlp = branch.embedding.value_mlps[0]
                self.assertEqual((mlp[0].in_features, mlp[0].out_features), (1, 8))
                self.assertIsInstance(mlp[1], torch.nn.SiLU)
                self.assertEqual((mlp[2].in_features, mlp[2].out_features), (8, 8))
                self.assertEqual(tuple(branch.embedding.node_embedding.weight.shape), (4, 4))
                self.assertEqual(branch.summary()["calendar"]["tod"]["source"], source)
                self.assertTrue(branch.summary()["enabled"]["tod"])
                self.assertTrue(branch.summary()["enabled"]["dow"])

    def _assert_identity(self, shared, independent) -> None:
        shared_tables = [
            parameter
            for name, parameter in shared.named_parameters()
            if _context_parameter(name)
        ]
        self.assertEqual(len(shared_tables), 3)
        branch_pointers = {parameter.data_ptr() for branch in shared.branches for parameter in branch.parameters()}
        self.assertTrue(all(parameter.data_ptr() not in branch_pointers for parameter in shared_tables))
        for suffix in ("tod_embedding.weight", "dow_embedding.weight", "adaptive.adaptive_embedding"):
            tables = [parameter for name, parameter in independent.named_parameters() if name.endswith(suffix)]
            self.assertEqual(len(tables), 4)
            self.assertEqual(len({parameter.data_ptr() for parameter in tables}), 4)
            self.assertEqual(len({id(parameter) for parameter in tables}), 4)
        for model in (shared, independent):
            optimizer = build_optimizer(model, {"learning_rate": 1e-3, "weight_decay": 1e-4})
            identifiers = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            self.assertEqual(len(identifiers), len(set(identifiers)))
            self.assertEqual(len(identifiers), len(trainable))
        shared_private = sum(parameter.numel() for name, parameter in shared.named_parameters() if not _context_parameter(name))
        independent_private = sum(
            parameter.numel() for name, parameter in independent.named_parameters() if not _context_parameter(name)
        )
        self.assertEqual(shared_private, independent_private)
        self.assertEqual(independent.parameter_count() - shared.parameter_count(), 3 * shared.architecture_summary()["context_tables"]["parameters_per_copy"])
        for parameter in shared.parameters():
            self.assertNotEqual(list(parameter.shape), [12, 4, 256])

    def _assert_finite_backward(self, shared, independent, inputs, tod, dow, target) -> None:
        for model in (shared, independent):
            model.zero_grad(set_to_none=True)
            model.train()
            loss = (model(inputs, tod, dow) * target).square().mean()
            loss.backward()
            for parameter in model.parameters():
                if not parameter.requires_grad:
                    continue
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        for parameter in (
            shared.shared_state_context.tod_embedding.weight,
            shared.shared_state_context.dow_embedding.weight,
            shared.shared_state_context.adaptive.adaptive_embedding,
        ):
            self.assertGreater(float(parameter.grad.abs().max()), 0.0)
        shared.zero_grad(set_to_none=True)
        independent.zero_grad(set_to_none=True)

    def _assert_context_grad_is_sum(self, shared, independent) -> None:
        pairs = [
            (
                shared.shared_state_context.tod_embedding.weight.grad,
                [branch.embedding.tod_embedding.weight.grad for branch in independent.branches],
            ),
            (
                shared.shared_state_context.dow_embedding.weight.grad,
                [branch.embedding.dow_embedding.weight.grad for branch in independent.branches],
            ),
            (
                shared.shared_state_context.adaptive.adaptive_embedding.grad,
                [branch.adaptive.adaptive_embedding.grad for branch in independent.branches],
            ),
        ]
        for shared_grad, parts in pairs:
            total = torch.stack(parts).sum(dim=0)
            mean = total / 4
            torch.testing.assert_close(shared_grad, total, rtol=1e-4, atol=1e-5)
            self.assertGreater(float((shared_grad - mean).abs().max()), 1e-5)

    def _assert_channel_perturbation(self, model, inputs, tod, dow) -> None:
        model.eval()
        for channel in range(4):
            baseline = _branch_outputs(model, inputs, tod, dow)
            perturbed = inputs.clone()
            perturbed[..., channel] += 1.7
            changed = _branch_outputs(model, perturbed, tod, dow)
            self.assertFalse(torch.equal(baseline[channel], changed[channel]))
            for other in range(4):
                if other == channel:
                    continue
                self.assertTrue(torch.equal(baseline[other], changed[other]))

    def _assert_physical_conservation(self, model, inputs, tod, dow) -> None:
        model.eval()
        final = model.forward_with_aux(inputs, tod, dow)["final_pred"]
        mean = model.projection.target_mean
        std = model.projection.target_std
        self.assertEqual(len(set(std.tolist())), 4)
        physical = final * std + mean
        self.assertLess(float(conservation_residual(physical).abs().max()), 1e-5)
        self.assertGreater(float(conservation_residual(final).abs().max()), 1e-2)

    def _assert_strict_reload(self, model, inputs, tod, dow) -> None:
        model.eval()
        expected = model(inputs, tod, dow).detach()
        blob = io.BytesIO()
        torch.save(model.state_dict(), blob)
        blob.seek(0)
        fresh = _build(int(model.config.num_nodes), model.config.state_context_sharing)
        fresh.load_state_dict(torch.load(blob, map_location="cpu", weights_only=True))
        fresh.eval()
        torch.testing.assert_close(fresh(inputs, tod, dow), expected)


def _node16_config(num_nodes: int, node_sharing: str, **overrides) -> dict:
    context_sharing = "shared" if node_sharing == "shared" else "independent"
    config = _full_config(
        num_nodes,
        context_sharing,
        d_model=416,
        value_embedding_dim=32,
        node_embedding_dim=64,
        tod_embedding_dim=32,
        dow_embedding_dim=32,
        base_embedding_dim=160,
        adaptive_embedding_dim=256,
        state_node_sharing=node_sharing,
    )
    config.update(overrides)
    return config


def _build_node16(num_nodes: int, node_sharing: str, **overrides) -> MSTT:
    return _build_from_mapping(_node16_config(num_nodes, node_sharing, **overrides))


def _align_node16(shared: MSTT, independent: MSTT) -> None:
    shared_state = shared.state_dict()
    independent_state = independent.state_dict()
    copied = set()
    with torch.no_grad():
        for key, value in shared_state.items():
            if key.startswith("shared_state_context."):
                continue
            if key not in independent_state:
                raise AssertionError(f"shared parameter {key} is missing on the independent model")
            independent_state[key].copy_(value)
            copied.add(key)
        sources = {
            "embedding.node_embedding.weight": shared.shared_state_context.node_embedding.weight,
            "embedding.tod_embedding.weight": shared.shared_state_context.tod_embedding.weight,
            "embedding.dow_embedding.weight": shared.shared_state_context.dow_embedding.weight,
            "adaptive.adaptive_embedding": shared.shared_state_context.adaptive.adaptive_embedding,
        }
        for index, _branch in enumerate(independent.branches):
            for suffix, source in sources.items():
                key = f"branches.{index}.{suffix}"
                independent_state[key].copy_(source)
                copied.add(key)
    missing = set(independent_state) - copied
    if missing:
        raise AssertionError(f"independent parameters were not copied: {sorted(missing)}")


class Node16ContextTest(unittest.TestCase):
    def test_configs_reject_partial_node_sharing(self) -> None:
        shared = json.loads(NODE16_SHARED.read_text(encoding="utf-8"))
        independent = json.loads(NODE16_INDEPENDENT.read_text(encoding="utf-8"))
        self.assertEqual(shared["experiment_id"], "state_independent_full_context_shared_node16_tsf_mlp_projected")
        self.assertEqual(independent["experiment_id"], "state_independent_full_context_independent_node16_tsf_mlp_projected")
        self.assertEqual(shared["dataset_id"], "variable_30min_p5_p15")
        self.assertEqual(shared["state_context_sharing"], "shared")
        self.assertEqual(shared["state_node_sharing"], "shared")
        self.assertEqual(independent["state_context_sharing"], "independent")
        self.assertEqual(independent["state_node_sharing"], "independent")
        self.assertEqual(shared["attention_head_dim"], 24)
        self.assertEqual(shared["ffn_hidden_dim"], 176)
        self.assertEqual(shared["node_embedding_dim"], 64)
        self.assertEqual(shared["d_model"], 416)
        ignored = {"experiment_id", "state_context_sharing", "state_node_sharing"}
        self.assertEqual(
            {key: value for key, value in shared.items() if key not in ignored},
            {key: value for key, value in independent.items() if key not in ignored},
        )
        for path, context_sharing, node_sharing in (
            (NODE16_SHARED, "shared", "shared"),
            (NODE16_INDEPENDENT, "independent", "independent"),
        ):
            config = load_config(path)
            validate_config(config)
            validate_backbone_task(config, task_spec("All2All"))
            built = model_config(config, task_spec("All2All"))
            self.assertEqual(built.state_context_sharing, context_sharing)
            self.assertEqual(built.state_node_sharing, node_sharing)
        with self.assertRaises(ValueError):
            validate_config(load_config(NODE16_INDEPENDENT) | {"state_node_sharing": "shared"})

    def test_widths_sharing_alignment_and_gradients(self) -> None:
        num_nodes = 4
        torch.manual_seed(1)
        shared = _build_node16(num_nodes, "shared")
        torch.manual_seed(2)
        independent = _build_node16(num_nodes, "independent")
        inputs, tod, dow, target = _sample(num_nodes)
        self.assertEqual(shared.config.embedding_dropout, 0.0)
        self.assertEqual(independent.config.embedding_dropout, 0.0)
        self._assert_structure(shared, independent, inputs, tod, dow)
        self._assert_shared_node_identity(shared, inputs, tod, dow)
        self._assert_optimizer(shared)
        self._assert_optimizer(independent)

        shared.eval()
        independent.eval()
        self.assertFalse(torch.allclose(shared(inputs, tod, dow), independent(inputs, tod, dow), atol=1e-3, rtol=1e-3))
        _align_node16(shared, independent)
        for left, right in zip(_branch_outputs(shared, inputs, tod, dow), _branch_outputs(independent, inputs, tod, dow)):
            torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)
        shared_aux = shared.forward_with_aux(inputs, tod, dow)
        independent_aux = independent.forward_with_aux(inputs, tod, dow)
        torch.testing.assert_close(shared_aux["raw_pred"], independent_aux["raw_pred"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(shared_aux["final_pred"], independent_aux["final_pred"], atol=1e-5, rtol=1e-5)

        shared.zero_grad(set_to_none=True)
        independent.zero_grad(set_to_none=True)
        shared.train()
        independent.train()
        shared_loss = (shared(inputs, tod, dow) * target).square().mean()
        independent_loss = (independent(inputs, tod, dow) * target).square().mean()
        self.assertTrue(torch.isfinite(shared_loss))
        torch.testing.assert_close(shared_loss, independent_loss, atol=1e-5, rtol=1e-5)
        shared_loss.backward()
        independent_loss.backward()
        self._assert_gradients_sum(shared, independent)
        for model in (shared, independent):
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

        shared.eval()
        independent.eval()
        self._assert_channel_isolation(shared, inputs, tod, dow)
        self._assert_channel_isolation(independent, inputs, tod, dow)
        self._assert_conservation(shared, inputs, tod, dow)
        self._assert_conservation(independent, inputs, tod, dow)
        self._assert_reload(shared, inputs, tod, dow)
        self._assert_reload(independent, inputs, tod, dow)
        with self.assertRaises(RuntimeError):
            _build_node16(num_nodes, "shared").load_state_dict(independent.state_dict())

    def test_dropout_masks_stay_on_each_branch(self) -> None:
        model = _build_node16(4, "shared")
        model.train()
        self.assertEqual(len({id(branch.embedding.dropout) for branch in model.branches}), 4)
        for branch in model.branches:
            branch.embedding.dropout.p = 0.5
        inputs, tod, dow, _target = _sample(4, seed=9)
        context = model.shared_state_context(inputs, tod, dow)
        slices = []
        for channel, branch in enumerate(model.branches):
            hidden = branch.embedding(
                inputs[..., channel:channel + 1],
                tod,
                dow,
                tod_features=context.tod,
                dow_features=context.dow,
                node_features=context.node,
            )
            slices.append(hidden[..., 8:24])
        self.assertFalse(all(torch.equal(slices[0], other) for other in slices[1:]))

    def test_full_network_parameter_delta(self) -> None:
        self.assertEqual(NODE16_CONTEXT_COPY, 1_104_664)
        self.assertEqual(3 * NODE16_CONTEXT_COPY, 3_313_992)
        counts = {}
        for label, path in (("shared", NODE16_SHARED), ("independent", NODE16_INDEPENDENT)):
            config = load_config(path)
            model = MSTT(
                model_config(config, task_spec("All2All")),
                _graph(int(config["num_nodes"])),
                target_mean=MEAN.clone(),
                target_std=STD.clone(),
            )
            counts[label] = model.parameter_count()
            summary = model.architecture_summary()
            self.assertEqual(summary["node_tables"]["shape"], [1406, 16])
            self.assertEqual(summary["context_tables"]["per_branch_feature_width"]["d_model"], 104)
            self.assertEqual(summary["head"]["input_dim"], 4992)
            self.assertEqual(summary["projection_mlp"]["input_dim"], 420)
            self.assertNotEqual(counts[label], 2_161_760)
            self.assertNotEqual(counts[label], 5_408_264)
            del model
        self.assertEqual(counts["independent"] - counts["shared"], 3_313_992)
        print(f"node16 parameters shared={counts['shared']} independent={counts['independent']}")

    def test_checkpoint_field_and_summary_isolation(self) -> None:
        task = task_spec("All2All")
        shared_model = model_config(load_config(NODE16_SHARED), task).to_dict()
        independent_model = model_config(load_config(NODE16_INDEPENDENT), task).to_dict()
        self.assertEqual(
            {key for key in shared_model if shared_model[key] != independent_model.get(key)},
            {"state_context_sharing", "state_node_sharing"},
        )
        payload = {"config": {"model_config": shared_model, "state_node_sharing": "shared"}}
        assert_model_checkpoint_matches(payload, shared_model)
        with self.assertRaises(ReimplementationError) as mismatch:
            assert_model_checkpoint_matches(payload, independent_model)
        self.assertIn("state_node_sharing", str(mismatch.exception))

        old = _build_from_mapping(_legacy_independent_config(4))
        omitted = {"model_config": {key: value for key, value in old.config.to_dict().items() if key != "state_node_sharing"}}
        restored = model_config_from_saved(omitted)
        self.assertEqual(restored["state_node_sharing"], "independent")
        rebuilt = _build_from_mapping(restored)
        rebuilt.load_state_dict(old.state_dict())
        self.assertEqual(tuple(rebuilt.branches[0].embedding.node_embedding.weight.shape), (4, 4))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = {"task": "All2All", "rate": "p5_p15", "seed": 42}
            merge_summary(root, [{**base, "experiment_id": "node16_d", "dataset_id": "variable_30min_p5_p15", "run_dir": "d30"}])
            merge_summary(root, [{**base, "experiment_id": "node16_d", "dataset_id": "variable_1h_p5_p15", "run_dir": "d1h"}])
            merge_summary(root, [{**base, "experiment_id": "node16_e", "dataset_id": "variable_30min_p5_p15", "run_dir": "e30"}])
            merge_summary(root, [{**base, "experiment_id": "node16_d", "dataset_id": "variable_30min_p5_p15", "run_dir": "d30b"}])
            merge_summary(root, [{"task": "I2I", "rate": "p05", "seed": 42, "run_dir": "old-a"}])
            merge_summary(root, [{"task": "I2I", "rate": "p05", "seed": 42, "run_dir": "old-b"}])
            merged = json.loads((root / "overall_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted(row["run_dir"] for row in merged), ["d1h", "d30b", "e30", "old-b"])

    def test_cli_resolves_p5_p15_paths(self) -> None:
        from MSTT.config import overlay_args
        from MSTT.train import build_parser

        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                build_parser().parse_args(["--task", "All2All", "--penetration", "p5_p15", "--state-node-sharing", "both"])
        for path, node_sharing in ((NODE16_SHARED, "shared"), (NODE16_INDEPENDENT, "independent")):
            args = build_parser().parse_args([
                "--config", str(path),
                "--task", "All2All",
                "--penetration", "p5_p15",
                "--dataset-id", "variable_1h_p5_p15",
                "--device", "cuda:0",
            ])
            config = overlay_args(load_config(args.config), args)
            validate_config(config)
            self.assertEqual(config["dataset_id"], "variable_1h_p5_p15")
            self.assertEqual(config["state_node_sharing"], node_sharing)
            self.assertIn("p5_p15", config["observation_keys"])
            resolved = run_directory(Path(config["output_root"]), "p5_p15", "All2All", int(config["seed"]), config)
            text = resolved.as_posix()
            self.assertIn("datasets/variable_1h_p5_p15/", text)
            self.assertIn("/in_all__out_all/p5_p15/", text)
            self.assertIn(f"/{config['experiment_id']}/seed_42/", text)
            self.assertNotIn("_smoke", text)
            from MSTT.train import resolve_output_root

            smoke_config = dict(config)
            smoke_config["max_train_batches"] = 1
            smoke_config["max_eval_batches"] = 1
            smoke_config["epochs"] = 1
            smoke_config["experiment_id"] = f"{config['experiment_id']}_smoke"
            smoke_root = resolve_output_root(smoke_config)
            smoke_dir = run_directory(smoke_root, "p5_p15", "All2All", int(smoke_config["seed"]), smoke_config)
            smoke_text = smoke_dir.as_posix()
            self.assertIn("/_smoke/datasets/", smoke_text)
            self.assertIn(f"/{smoke_config['experiment_id']}/seed_42/", smoke_text)
            self.assertNotIn(f"/p5_p15/{config['experiment_id']}/seed_", smoke_text)
            formal_root = resolve_output_root(config)
            formal_dir = run_directory(formal_root, "p5_p15", "All2All", int(config["seed"]), config)
            formal_text = formal_dir.as_posix()
            self.assertIn("datasets/variable_1h_p5_p15/", formal_text)
            self.assertIn("/in_all__out_all/p5_p15/", formal_text)
            self.assertIn(f"/{config['experiment_id']}/seed_42/", formal_text)
            self.assertNotIn("_smoke", formal_text)
            self.assertNotIn("v0.2_concat_A2A_", formal_text)

    def _assert_structure(self, shared, independent, inputs, tod, dow) -> None:
        self.assertEqual(tuple(shared(inputs, tod, dow).shape), (2, 4, 4))
        for model, node_copies, context_copies, node_source in (
            (shared, 1, 1, "external"),
            (independent, 4, 4, "parameter"),
        ):
            parts = _branch_outputs(model, inputs, tod, dow)
            self.assertEqual(len(parts), 4)
            for part in parts:
                self.assertEqual(tuple(part.shape), (2, 12, 4, 104))
            summary = model.architecture_summary()
            widths = summary["context_tables"]["per_branch_feature_width"]
            self.assertEqual(
                [widths[key] for key in ("value", "node", "tod", "dow", "base", "adaptive", "d_model")],
                [8, 16, 8, 8, 40, 64, 104],
            )
            self.assertEqual(summary["context_tables"]["copies"], context_copies)
            self.assertEqual(summary["node_tables"]["copies"], node_copies)
            self.assertEqual(summary["node_tables"]["shape"], [4, 16])
            self.assertEqual(summary["node_tables"]["enabled"], True)
            self.assertEqual(summary["node_tables"]["parameters_total"], node_copies * 4 * 16)
            self.assertEqual(summary["head"]["input_dim"], 4992)
            self.assertEqual(summary["head"]["hidden_dim"], 128)
            self.assertEqual(summary["head"]["output_dim"], 4)
            self.assertEqual(summary["projection_mlp"]["input_dim"], 420)
            self.assertEqual(summary["projection_mlp"]["hidden_dim"], 32)
            self.assertEqual(summary["embedding"]["parameter_sharing"]["value_mlp"], "independent")
            self.assertEqual(summary["embedding"]["parameter_sharing"]["tsf"], "independent")
            self.assertEqual(summary["embedding"]["parameter_sharing"]["node_embedding"], model.config.state_node_sharing)
            value_ptrs = [parameter.data_ptr() for branch in model.branches for parameter in branch.embedding.value_mlps.parameters()]
            block_ptrs = [parameter.data_ptr() for branch in model.branches for parameter in branch.block.parameters()]
            self.assertEqual(len(value_ptrs), len(set(value_ptrs)))
            self.assertEqual(len(block_ptrs), len(set(block_ptrs)))
            for branch in model.branches:
                block = branch.block
                self.assertEqual(block.temporal.qkv.in_features, 104)
                self.assertEqual(block.temporal.qkv.out_features, 3 * 96)
                self.assertEqual(block.temporal.out_proj.in_features, 96)
                self.assertEqual(block.temporal.out_proj.out_features, 104)
                self.assertEqual(block.temporal.head_dim, 24)
                self.assertEqual(block.spatial.head_dim, 24)
                self.assertEqual(branch.summary()["enabled"]["node"], True)
                self.assertEqual(branch.summary()["node"]["source"], node_source)
                self.assertEqual(branch.summary()["node"]["shape"], [4, 16])
                mlp = branch.embedding.value_mlps[0]
                self.assertEqual((mlp[0].in_features, mlp[0].out_features, mlp[2].out_features), (1, 8, 8))

    def _assert_shared_node_identity(self, shared, inputs, tod, dow) -> None:
        seen = []

        def capture(_module, _args, kwargs):
            seen.append(kwargs["node_features"])

        handles = [branch.embedding.register_forward_pre_hook(capture, with_kwargs=True) for branch in shared.branches]
        try:
            shared.eval()
            shared(inputs, tod, dow)
        finally:
            for handle in handles:
                handle.remove()
        parent = shared.shared_state_context.node_embedding.weight
        self.assertEqual(len(seen), 4)
        self.assertEqual(len({id(parent)}), 1)
        for features in seen:
            self.assertEqual(tuple(features.shape), (1, 1, 4, 16))
            self.assertEqual(features.data_ptr(), parent.data_ptr())
        names = [name for name, _parameter in shared.named_parameters() if name.endswith("node_embedding.weight")]
        self.assertEqual(names, ["shared_state_context.node_embedding.weight"])
        independent_names = [name for name, _parameter in _build_node16(4, "independent").named_parameters() if name.endswith("node_embedding.weight")]
        self.assertEqual(len(independent_names), 4)
        self.assertEqual(len({name.split(".")[1] for name in independent_names}), 4)

    def _assert_optimizer(self, model) -> None:
        optimizer = build_optimizer(model, {"learning_rate": 1e-3, "weight_decay": 1e-4})
        identifiers = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(len(identifiers), len(trainable))

    def _assert_gradients_sum(self, shared, independent) -> None:
        pairs = [
            (
                shared.shared_state_context.node_embedding.weight.grad,
                [branch.embedding.node_embedding.weight.grad for branch in independent.branches],
            ),
            (
                shared.shared_state_context.tod_embedding.weight.grad,
                [branch.embedding.tod_embedding.weight.grad for branch in independent.branches],
            ),
            (
                shared.shared_state_context.dow_embedding.weight.grad,
                [branch.embedding.dow_embedding.weight.grad for branch in independent.branches],
            ),
            (
                shared.shared_state_context.adaptive.adaptive_embedding.grad,
                [branch.adaptive.adaptive_embedding.grad for branch in independent.branches],
            ),
        ]
        for shared_grad, parts in pairs:
            self.assertTrue(all(part is not None for part in parts))
            total = torch.stack(parts).sum(dim=0)
            torch.testing.assert_close(shared_grad, total, rtol=1e-4, atol=1e-5)
            self.assertGreater(float(shared_grad.abs().max()), 0.0)

    def _assert_channel_isolation(self, model, inputs, tod, dow) -> None:
        model.eval()
        for channel in range(4):
            baseline = _branch_outputs(model, inputs, tod, dow)
            perturbed = inputs.clone()
            perturbed[..., channel] += 1.7
            changed = _branch_outputs(model, perturbed, tod, dow)
            self.assertFalse(torch.equal(baseline[channel], changed[channel]))
            for other in range(4):
                if other == channel:
                    continue
                self.assertTrue(torch.equal(baseline[other], changed[other]))

    def _assert_conservation(self, model, inputs, tod, dow) -> None:
        model.eval()
        final = model.forward_with_aux(inputs, tod, dow)["final_pred"]
        physical = final * model.projection.target_std + model.projection.target_mean
        self.assertLess(float(conservation_residual(physical).abs().max()), 1e-5)

    def _assert_reload(self, model, inputs, tod, dow) -> None:
        model.eval()
        expected = model(inputs, tod, dow).detach()
        blob = io.BytesIO()
        torch.save(model.state_dict(), blob)
        blob.seek(0)
        fresh = _build_node16(int(model.config.num_nodes), model.config.state_node_sharing)
        fresh.load_state_dict(torch.load(blob, map_location="cpu", weights_only=True))
        fresh.eval()
        torch.testing.assert_close(fresh(inputs, tod, dow), expected)


if __name__ == "__main__":
    unittest.main()
