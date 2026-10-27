"""Unmasked MAE on the raw vehicle-count scale.

Prepared targets are global z-scores, not values in ``[0, 1]``. The network
emits the normalized target and this loss inverts it with the training-only
full-flow scaler before MAE. A raw count of zero is a real empty interval,
so it is included.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from reimplementation.common.data.r_only_npz_dataset import invert_target
from reimplementation.common.errors import ReimplementationError


def mae_all(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ReimplementationError(
            f"prediction {tuple(prediction.shape)} != target {tuple(target.shape)}"
        )
    if prediction.numel() == 0:
        raise ReimplementationError("cannot compute MAE on an empty batch")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ReimplementationError("MAE inputs contain NaN or inf")
    return torch.mean(torch.abs(prediction - target))


class MAERawLoss(nn.Module):
    def __init__(self, mean: float, std: float) -> None:
        super().__init__()
        if not math.isfinite(float(mean)) or not math.isfinite(float(std)) or float(std) <= 0.0:
            raise ReimplementationError(f"invalid target scaler mean={mean} std={std}")
        self.mean = float(mean)
        self.std = float(std)

    def forward(self, prediction_normalized: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
        return mae_all(invert_target(prediction_normalized, self.mean, self.std), target_raw)
