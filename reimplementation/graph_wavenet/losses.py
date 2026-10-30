"""Graph WaveNet training loss: inverse-transform then unmasked MAE.

Official R-only semantics: ``flow = 0`` is a real empty 5-minute window, not a
PeMS missing detector. Training therefore uses

    mean(abs(pred_raw - y_raw))

after inverse-transform. Zero targets participate in the loss and gradient.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from reimplementation.common.data.r_only_npz_dataset import invert_target, loss_scale


class UnmaskedMAERawLoss(nn.Module):
    def __init__(self, mean: float, std: float, null_val: float | None = None) -> None:
        super().__init__()
        self.mean = loss_scale(mean)
        self.std = loss_scale(std)
        self.null_val = null_val

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        return invert_target(normalized, self.mean, self.std)

    def forward(self, prediction_normalized: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
        if prediction_normalized.shape != target_raw.shape:
            raise ValueError(
                f"prediction {tuple(prediction_normalized.shape)} != "
                f"target {tuple(target_raw.shape)}"
            )
        preds = self.denormalize(prediction_normalized)
        return torch.mean(torch.abs(preds - target_raw))


# Historical name kept so existing imports continue to resolve to the unmasked loss.
MaskedMAERawLoss = UnmaskedMAERawLoss


def mae_raw_all(prediction_raw: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
    if prediction_raw.shape != target_raw.shape:
        raise ValueError("mae_raw_all shape mismatch")
    return torch.mean(torch.abs(prediction_raw - target_raw))


def mae_raw_nonzero(prediction_raw: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
    if prediction_raw.shape != target_raw.shape:
        raise ValueError("mae_raw_nonzero shape mismatch")
    mask = target_raw != 0
    if not bool(mask.any()):
        return prediction_raw.new_zeros(())
    return torch.mean(torch.abs(prediction_raw[mask] - target_raw[mask]))
