"""C_inflow_aux loss, gradients, original C identity, and inflow checkpoint selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from MSTT.config import (
    conservation_settings,
    default_config,
    load_config,
    task_spec,
    validate_config,
    validate_conservation_task,
)
from MSTT.model import MSTT, MSTTConfig
from MSTT.model.conservation import conservation_residual
from MSTT.model.graph import build_spatial_graph
from MSTT.trainer import TaskRun, assert_conservation_checkpoint, gradient_report, train_step
from MSTT.utils.losses import RawScaleLoss, conservation_objective, objective_description
from reimplementation.common.errors import ReimplementationError

ROOT = Path(__file__).resolve().parents[1]
PROJECTED = ROOT / "configs" / "conservation_best_backbone" / "projected.json"
AUX = ROOT / "configs" / "conservation_best_backbone" / "C_inflow_aux.json"
LEGACY_C = "final_mae_raw + 0.5 * raw_mae_raw + 0 * raw_conservation_mae"


def _criterion(mean, std):
    return RawScaleLoss(mean, std, kind="mae_raw", channel_balance="none")


def _manual_inflow_loss(final_physical, raw_physical, target, weight=0.5):
    inflow = (final_physical[..., 0] - target[..., 0]).abs().mean()
    residual = conservation_residual(raw_physical).abs().mean()
    return inflow + weight * residual, inflow, residual


def test_inflow_only_loss_matches_hand_calculation_on_physical_scale():
    mean = torch.tensor([10.0, -2.0, 0.5, 1.5])
    std = torch.tensor([2.0, 3.0, 0.4, 0.8])
    raw = torch.tensor(
        [[[1.0, 4.0, 0.2, 3.0], [2.0, 0.5, 1.0, 0.0]], [[0.0, 1.0, 2.0, 5.0], [3.0, 3.0, 0.0, 1.0]]]
    )
    final = raw + torch.tensor([0.4, -0.2, 0.1, -0.3])
    target = torch.tensor(
        [[[12.0, 8.0, 1.0, 0.0], [9.0, 1.0, 4.0, 2.0]], [[11.0, 0.0, 0.5, 6.0], [7.0, 5.0, 1.5, 1.0]]]
    )
    criterion = _criterion(mean, std)
    config = {
        "conservation_mode": "projected",
        "projection_supervision": "inflow_only",
        "projection_raw_loss_weight": 0.0,
        "projection_consistency_weight": 0.5,
    }
    loss, parts = conservation_objective(
        criterion,
        {"raw_pred": (raw - mean) / std, "final_pred": (final - mean) / std},
        target,
        config,
    )
    expected, inflow, raw_residual = _manual_inflow_loss(final, raw, target, 0.5)
    final_residual = conservation_residual(final).abs().mean()
    assert not torch.allclose(raw_residual, final_residual)
    assert torch.allclose(loss, expected)
    assert torch.allclose(parts["inflow_final_mae"], inflow)
    assert torch.allclose(parts["raw_conservation_mae"], raw_residual)
    assert loss is parts["total_loss"]
    assert set(parts) == {"inflow_final_mae", "raw_conservation_mae", "total_loss"}
    wrong = inflow + 0.5 * final_residual
    assert not torch.allclose(loss, wrong)

    shifted = target.clone()
    shifted[..., 1:] += 7.0
    unchanged, _ = conservation_objective(
        criterion,
        {"raw_pred": (raw - mean) / std, "final_pred": (final - mean) / std},
        shifted,
        config,
    )
    assert torch.allclose(unchanged, loss)
    supervised = target.clone()
    supervised[..., 0] += 1.5
    changed, _ = conservation_objective(
        criterion,
        {"raw_pred": (raw - mean) / std, "final_pred": (final - mean) / std},
        supervised,
        config,
    )
    assert not torch.allclose(changed, loss)


def test_only_final_inflow_and_raw_residual_receive_direct_gradients():
    mean = torch.tensor([10.0, -2.0, 0.5, 1.5])
    std = torch.tensor([2.0, 3.0, 0.4, 0.8])
    raw = torch.randn(2, 3, 4)
    final = raw + torch.tensor([0.4, -0.2, 0.1, -0.3])
    target = torch.randn(2, 3, 4) + 4.0
    raw_n = ((raw - mean) / std).detach().requires_grad_(True)
    final_n = ((final - mean) / std).detach().requires_grad_(True)
    criterion = _criterion(mean, std)
    loss, _ = conservation_objective(
        criterion,
        {"raw_pred": raw_n, "final_pred": final_n},
        target,
        {
            "conservation_mode": "projected",
            "projection_supervision": "inflow_only",
            "projection_raw_loss_weight": 0.0,
            "projection_consistency_weight": 0.5,
        },
    )
    loss.backward()
    assert torch.isfinite(final_n.grad).all() and torch.isfinite(raw_n.grad).all()
    assert final_n.grad[..., 0].abs().sum() > 0
    assert final_n.grad[..., 1:].abs().max() == 0
    assert raw_n.grad.abs().sum() > 0
    assert raw_n.grad[..., 1:].abs().sum() > 0


def test_original_projected_loss_is_unchanged():
    mean = torch.tensor([1.0, 2.0, 3.0, 4.0])
    std = torch.tensor([0.5, 1.5, 2.0, 0.8])
    raw_n = torch.randn(4, 2, 5, 4, requires_grad=True)
    final_n = torch.randn(4, 2, 5, 4, requires_grad=True)
    target = torch.randn(4, 2, 5, 4)
    criterion = _criterion(mean, std)
    config = default_config()
    config.update(conservation_mode="projected", projection_supervision="all",
                  projection_raw_loss_weight=0.5, projection_consistency_weight=0.0)
    loss, parts = conservation_objective(criterion, {"raw_pred": raw_n, "final_pred": final_n}, target, config)
    supervised = criterion(final_n, target)
    raw_mae = criterion(raw_n, target)
    raw_cons = conservation_residual(criterion.denormalize(raw_n)).abs().mean()
    legacy = supervised + 0.5 * raw_mae + 0.0 * raw_cons
    assert torch.equal(loss, legacy)
    assert torch.equal(parts["total_loss"], legacy)
    assert "supervised_mae" in parts and "raw_mae" in parts
    assert objective_description(criterion, config) == LEGACY_C
    loaded = load_config(PROJECTED)
    assert loaded["projection_supervision"] == "all"
    assert "projection_supervision" not in json.loads(PROJECTED.read_text(encoding="utf-8"))
    assert objective_description(criterion, loaded) == LEGACY_C


def test_projection_mlp_and_backbone_gradients_and_exact_projection():
    torch.manual_seed(7)
    cfg = default_config()
    cfg.update(
        num_nodes=4,
        conservation_mode="projected",
        projection_supervision="inflow_only",
        projection_raw_loss_weight=0.0,
        projection_consistency_weight=0.5,
        device="cpu",
    )
    validate_config(cfg)
    mean = torch.tensor([1.0, -1.0, 0.25, 0.5])
    std = torch.tensor([2.0, 1.5, 0.5, 0.75])
    model = MSTT(
        MSTTConfig.from_mapping(cfg, in_channels=4, out_channels=4),
        build_spatial_graph(torch.eye(4)),
        target_mean=mean,
        target_std=std,
    )
    assert model.projection is not None
    assert model.config.attention_layout == "factorized_ts"
    batch = {
        "x": torch.randn(2, 12, 4, 4),
        "y": torch.zeros(2, 1, 4, 4),
        "y_raw": torch.rand(2, 1, 4, 4) + 0.3,
        "day_index": torch.zeros(2, dtype=torch.long),
        "window_start_slot": torch.zeros(2, dtype=torch.long),
        "target_slot": torch.full((2,), 11, dtype=torch.long),
        "sample_index": torch.arange(2),
    }
    model.train()
    preview = model.forward_with_aux(batch["x"], torch.arange(12).expand(2, 12), torch.zeros(2, 12, dtype=torch.long))
    physical_final = preview["final_pred"].float() * std + mean
    physical_raw = preview["raw_pred"].float() * std + mean
    assert conservation_residual(physical_final).abs().max() < 1e-4
    assert conservation_residual(physical_raw).abs().mean() > 1e-3

    class _Run:
        config = cfg
        device = torch.device("cpu")
        criterion = _criterion(mean, std)

    run = _Run()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loss, _ = train_step(run, model, batch, optimizer, scaler, amp="none", clip=0.0, epoch=0)
    assert torch.isfinite(torch.tensor(loss))
    report = gradient_report(model)
    assert report["missing"] == [] and report["nonfinite"] == []
    assert model.projection.mlp[-1].weight.grad.abs().sum() > 0
    assert model.projection.mlp[-1].bias.grad.abs().sum() > 0
    for module in (model.embedding, model.blocks, model.head, model.adaptive):
        assert any(param.grad is not None and param.grad.abs().sum() > 0 for param in module.parameters())
    assert set(run.last_train_components) == {"inflow_final_mae", "raw_conservation_mae", "total_loss"}
    assert run.last_train_components["total_loss"] == pytest.approx(
        run.last_train_components["inflow_final_mae"] + 0.5 * run.last_train_components["raw_conservation_mae"]
    )


def test_aux_config_checkpoint_and_inflow_selection():
    aux = load_config(AUX)
    projected = load_config(PROJECTED)
    validate_config(aux)
    validate_config(projected)
    validate_conservation_task(aux, task_spec("All2All"))
    with pytest.raises(ValueError, match="All2All"):
        validate_conservation_task(aux, task_spec("I2I"))
    settings = conservation_settings(aux)
    assert settings["conservation_mode"] == "projected"
    assert settings["projection_supervision"] == "inflow_only"
    assert settings["projection_raw_loss_weight"] == 0.0
    assert settings["projection_consistency_weight"] == 0.5
    assert aux["attention_layout"] == "factorized_ts"
    assert aux["output_root"].endswith("MSTT/C_inflow_aux")
    assert aux["epochs"] == 100 and aux["model_seed"] == 42 and aux["train_sampling_seed"] == 42
    assert aux["evaluation_sampling_seeds"] == [42, 43, 44, 45, 46]
    assert aux["selection_metric"] == "mean_validation_mae"
    assert objective_description(_criterion([0, 0, 0, 0], [1, 1, 1, 1]), aux) == (
        "inflow_final_mae + 0.5 * raw_conservation_mae"
    )

    old = dict(projected)
    old.pop("projection_supervision")
    assert conservation_settings(old)["projection_supervision"] == "all"
    assert_conservation_checkpoint(
        {"conservation_mode": "projected", "config": old, "normalization_sha256": "sha"},
        projected,
        normalization_sha256="sha",
    )
    with pytest.raises(ReimplementationError, match="conservation"):
        assert_conservation_checkpoint(
            {"conservation_mode": "projected", "config": old, "normalization_sha256": "sha"},
            aux,
            normalization_sha256="sha",
        )
    assert_conservation_checkpoint(
        {"conservation_mode": "projected", "config": aux, "normalization_sha256": "sha"},
        aux,
        normalization_sha256="sha",
    )

    rejected = dict(aux)
    rejected["projection_raw_loss_weight"] = 0.5
    with pytest.raises(ValueError, match="projection_raw_loss_weight"):
        validate_config(rejected)

    run = TaskRun.__new__(TaskRun)
    run.task = task_spec("All2All")
    assert run.selection_value({"mae": 9.0, "inflow_mae": 1.25}) == 1.25
    run.task = task_spec("I2I")
    assert run.selection_value({"mae": 3.0}) == 3.0
