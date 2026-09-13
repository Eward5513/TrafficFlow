"""Unmasked MAE on the raw vehicle-count scale.

Original PeMS04 STID trains ``masked_mae`` after ``ZScoreScaler`` inverse
transform (``RESCALE=True``). ``NULL_VAL=0.0`` in PeMS marks **missing**
detector readings. Prepared R-only ``y_raw == 0`` is a real zero count.

Necessary adaptation (does not change the STID architecture): compute MAE
on every target element, including zeros. This is ``masked_mae`` with
``null_val=nan`` and no NaNs, i.e. ordinary MAE.
"""

from __future__ import annotations

import torch
import torch.nn as nn

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
        self.mean = float(mean)
        self.std = float(std)

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        return invert_target(normalized, self.mean, self.std)

    def forward(self, prediction_normalized: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
        prediction_raw = self.denormalize(prediction_normalized)
        return mae_all(prediction_raw, target_raw)
