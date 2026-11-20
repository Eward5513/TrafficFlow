"""Training objectives on the raw count scale.

Default ``mae_raw`` is the unmasked raw-scale MAE that STID, Graph WaveNet,
DCRNN, and Seq2seq GCN-LSTM train and select on: predictions are inverted
with the per-channel target scaler and compared with ``y_raw``, zeros
included (they are real counts).

``channel_balance="target_std"`` divides each channel's error by its target
std before averaging. For All2All this stops inflow/outflow (std ~17) from
outweighing accumulation (std ~4); for MAE it equals MAE in target z-space.
It changes the objective, so it is off by default.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from MSTT.model.conservation import conservation_residual


class RawScaleLoss(nn.Module):
    def __init__(
        self,
        mean: Any,
        std: Any,
        *,
        kind: str = "mae_raw",
        huber_delta: float = 1.0,
        channel_balance: str = "none",
    ) -> None:
        super().__init__()
        if kind not in ("mae_raw", "huber_raw", "mse_raw"):
            raise ValueError(f"unknown loss {kind!r}")
        if channel_balance not in ("none", "target_std"):
            raise ValueError(f"unknown channel_balance {channel_balance!r}")
        mean_t = torch.as_tensor(np.atleast_1d(np.asarray(mean, dtype=np.float64)), dtype=torch.float32)
        std_t = torch.as_tensor(np.atleast_1d(np.asarray(std, dtype=np.float64)), dtype=torch.float32)
        if mean_t.shape != std_t.shape or bool((std_t <= 0).any()):
            raise ValueError("target scaler mean/std must match and std must be positive")
        self.register_buffer("mean", mean_t, persistent=False)
        self.register_buffer("std", std_t, persistent=False)
        self.kind = kind
        self.huber_delta = float(huber_delta)
        self.channel_balance = channel_balance

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        return normalized.float() * self.std + self.mean

    def elementwise(self, prediction_normalized: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
        if prediction_normalized.shape != target_raw.shape:
            raise ValueError(f"prediction {tuple(prediction_normalized.shape)} != target {tuple(target_raw.shape)}")
        if int(target_raw.shape[-1]) != int(self.mean.numel()):
            raise ValueError("channel axis does not match the target scaler")
        error = self.denormalize(prediction_normalized) - target_raw.float()
        if self.channel_balance == "target_std":
            error = error / self.std
        if self.kind == "mae_raw":
            return error.abs()
        if self.kind == "mse_raw":
            return error.square()
        delta = self.huber_delta
        magnitude = error.abs()
        return torch.where(magnitude <= delta, 0.5 * magnitude.square(), delta * magnitude - 0.5 * delta * delta)

    def forward(self, prediction_normalized: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
        return self.elementwise(prediction_normalized, target_raw).mean()

    def describe(self) -> str:
        name = self.kind if self.kind != "huber_raw" else f"huber_raw_delta_{self.huber_delta:g}"
        return name if self.channel_balance == "none" else f"{name}__balance_{self.channel_balance}"


def _projected_inflow_only_objective(criterion: RawScaleLoss, outputs, target_raw, config):
    """Projected inflow MAE plus the pre-projection conservation residual. No other state MAE."""
    if criterion.kind != "mae_raw" or criterion.channel_balance != "none":
        raise ValueError("inflow_only projection loss requires unbalanced raw-scale MAE")
    if float(config.get("projection_raw_loss_weight", 0.5)) != 0.0:
        raise ValueError("inflow_only projection loss does not add raw MAE; projection_raw_loss_weight must be 0")
    final_physical = criterion.denormalize(outputs["final_pred"])
    raw_physical = criterion.denormalize(outputs["raw_pred"])
    target = target_raw.float()
    if final_physical.shape != target.shape or raw_physical.shape != target.shape or int(target.shape[-1]) != 4:
        raise ValueError("inflow_only projection loss expects four-state raw and final predictions matching the target")
    inflow_final_mae = (final_physical[..., 0] - target[..., 0]).abs().mean()
    raw_conservation_mae = conservation_residual(raw_physical).abs().mean()
    total = inflow_final_mae + float(config.get("projection_consistency_weight", 0.0)) * raw_conservation_mae
    return total, {
        "inflow_final_mae": inflow_final_mae,
        "raw_conservation_mae": raw_conservation_mae,
        "total_loss": total,
    }


def conservation_objective(criterion: RawScaleLoss, outputs, target_raw, config):
    """All components remain tensors on the training device until after backward."""
    mode = config.get("conservation_mode", "none")
    supervision = str(config.get("projection_supervision", "all"))
    if mode == "projected" and supervision == "inflow_only":
        return _projected_inflow_only_objective(criterion, outputs, target_raw, config)
    if supervision not in ("all", "inflow_only"):
        raise ValueError(f"unknown projection_supervision {supervision!r}")
    supervised = criterion(outputs["final_pred"], target_raw)
    total = supervised
    parts = {"supervised_mae": supervised}
    if mode != "none":
        raw_cons = conservation_residual(criterion.denormalize(outputs["raw_pred"])).abs().mean()
        parts["conservation_loss"] = raw_cons
        if mode == "soft":
            total = supervised + float(config.get("conservation_lambda", 0.1)) * raw_cons
        elif mode == "projected":
            raw_mae = criterion(outputs["raw_pred"], target_raw)
            parts["raw_mae"] = raw_mae
            total = (supervised + float(config.get("projection_raw_loss_weight", 0.5)) * raw_mae
                     + float(config.get("projection_consistency_weight", 0.0)) * raw_cons)
        else:
            raise ValueError(f"unknown conservation_mode {mode}")
    parts["total_loss"] = total
    return total, parts


def temporal_aux_terms(criterion: RawScaleLoss, outputs, target_seq, config):
    """Physical MAE of the supervised auxiliary tensor.

    ``history_except_last`` uses steps 0..10. ``all_steps`` uses the full 12-step
    tensor. Each scope averages every element in its own slice once.
    """
    if not bool(config.get("temporal_aux_enabled", False)):
        raise ValueError("temporal_aux_terms requires temporal_aux_enabled")
    scope = str(config.get("temporal_aux_scope", "history_except_last"))
    prediction = outputs["temporal_aux_pred"]
    target = target_seq
    if prediction.ndim != 4 or target.ndim != 4 or prediction.shape != target.shape:
        raise ValueError(
            f"temporal aux prediction {tuple(prediction.shape)} != sequence target {tuple(target.shape)}"
        )
    if int(prediction.shape[1]) != 12:
        raise ValueError(f"temporal aux prediction must have 12 steps, got {prediction.shape[1]}")
    if scope == "history_except_last":
        aux = criterion(prediction[:, :11], target[:, :11])
    elif scope == "all_steps":
        aux = criterion(prediction, target)
    else:
        raise ValueError(f"unsupported temporal_aux_scope {scope!r}")
    weight = float(config.get("temporal_aux_weight", 0.5))
    weighted = aux * weight
    return aux, weighted


def add_dual_path_terms(criterion: RawScaleLoss, outputs, target_raw, target_seq, config, loss, parts):
    """Add the two branch MAEs and the existing 11-step auxiliary MAE. One returned total."""
    if str(config.get("history_fusion_mode", "none")) != "dual_path_history_tsf":
        raise ValueError("add_dual_path_terms requires history_fusion_mode='dual_path_history_tsf'")
    for key in ("branch_a_pred", "branch_b_pred", "branch_gate"):
        if key not in outputs:
            raise ValueError(f"dual path outputs are missing {key}")
    branch_a = criterion(outputs["branch_a_pred"], target_raw)
    branch_b = criterion(outputs["branch_b_pred"], target_raw)
    weight_a = float(config.get("branch_a_loss_weight", 0.25))
    weight_b = float(config.get("branch_b_loss_weight", 0.25))
    weighted_a = branch_a * weight_a
    weighted_b = branch_b * weight_b
    aux, weighted_aux = temporal_aux_terms(criterion, outputs, target_seq, config)
    total = loss + weighted_a + weighted_b + weighted_aux
    updated = dict(parts)
    updated["final_mae"] = updated.get("supervised_mae", updated.get("inflow_final_mae"))
    updated["branch_a_mae"] = branch_a
    updated["branch_b_mae"] = branch_b
    updated["weighted_branch_a_loss"] = weighted_a
    updated["weighted_branch_b_loss"] = weighted_b
    updated["temporal_aux_mae"] = aux
    updated["weighted_temporal_aux_loss"] = weighted_aux
    gate = outputs["branch_gate"].detach()
    gate_mean = gate.float().mean(dim=tuple(range(gate.ndim - 1)))
    from MSTT.config import STATE_CHANNELS

    if int(gate_mean.numel()) != len(STATE_CHANNELS):
        raise ValueError(f"branch gate has {int(gate_mean.numel())} states, expected {len(STATE_CHANNELS)}")
    for name, value in zip(STATE_CHANNELS, gate_mean):
        updated[f"gate_{name}_mean"] = value
    updated["total_loss"] = total
    return total, updated


def objective_description(criterion: RawScaleLoss, config) -> str:
    if str(config.get("history_fusion_mode", "none")) == "two_stage_projected_tsf":
        weight = float(config.get("temporal_aux_weight", 0.5))
        return f"final_mae_raw + {weight:g} * stage1_projected_mae[0:12]"
    mode = config.get("conservation_mode", "none")
    if mode == "soft":
        text = f"mae_raw + {float(config.get('conservation_lambda', .1)):g} * conservation_mae"
    elif mode == "projected" and str(config.get("projection_supervision", "all")) == "inflow_only":
        text = (f"inflow_final_mae + {float(config.get('projection_consistency_weight', 0.)):g}"
                " * raw_conservation_mae")
    elif mode == "projected":
        text = (f"final_mae_raw + {float(config.get('projection_raw_loss_weight', .5)):g} * raw_mae_raw"
                f" + {float(config.get('projection_consistency_weight', 0.)):g} * raw_conservation_mae")
    else:
        text = criterion.describe()
    if str(config.get("history_fusion_mode", "none")) == "dual_path_history_tsf":
        text += (
            f" + {float(config.get('branch_a_loss_weight', 0.25)):g} * branch_a_mae"
            f" + {float(config.get('branch_b_loss_weight', 0.25)):g} * branch_b_mae"
        )
    if bool(config.get("temporal_aux_enabled", False)):
        scope = str(config.get("temporal_aux_scope", "history_except_last"))
        if scope == "all_steps":
            text += f" + {float(config.get('temporal_aux_weight', 0.5)):g} * temporal_aux_mae[0:12]"
        else:
            text += f" + {float(config.get('temporal_aux_weight', 0.5)):g} * temporal_aux_mae[0:11]"
    return text
