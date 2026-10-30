"""Unmasked MAE on the raw vehicle-count scale.

Prepared targets are global z-scores, not values in ``[0, 1]``. The network
emits the normalized target and this loss inverts it with the training-only
full-flow scaler before MAE. A raw count of zero is a real empty interval,
so it is included.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from reimplementation.common.data.r_only_npz_dataset import invert_target, loss_scale
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


def _require_target_scaler(mean: Any, std: Any) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Accept one scaler or one scaler per target channel."""
    scaled_mean = loss_scale(mean)
    scaled_std = loss_scale(std)
    mean_array = np.asarray(scaled_mean, dtype=np.float64).reshape(-1)
    std_array = np.asarray(scaled_std, dtype=np.float64).reshape(-1)
    if (
        mean_array.size == 0
        or mean_array.shape != std_array.shape
        or not np.isfinite(mean_array).all()
        or not np.isfinite(std_array).all()
        or np.any(std_array <= 0.0)
    ):
        raise ReimplementationError(f"invalid target scaler mean={mean} std={std}")
    return scaled_mean, scaled_std


class MAERawLoss(nn.Module):
    def __init__(self, mean: Any, std: Any) -> None:
        super().__init__()
        self.mean, self.std = _require_target_scaler(mean, std)

    def forward(self, prediction_normalized: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
        return mae_all(invert_target(prediction_normalized, self.mean, self.std), target_raw)
