"""Huber loss on the raw vehicle-count scale.

Released TLAST trains Huber with ``delta=2`` after inverting the shared
z-score (``run.py``, ``loss_type='huber'``). Zeros in this dataset are real
counts, so the loss is unmasked. The architecture is unchanged.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from reimplementation.common.data.r_only_npz_dataset import invert_target, loss_scale
from reimplementation.common.errors import ReimplementationError

AUTHOR_HUBER_DELTA = 2.0


def huber_loss(prediction: torch.Tensor, target: torch.Tensor, delta: float = AUTHOR_HUBER_DELTA) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ReimplementationError(f"prediction {tuple(prediction.shape)} != target {tuple(target.shape)}")
    if prediction.numel() == 0:
        raise ReimplementationError("cannot compute Huber on an empty batch")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ReimplementationError("Huber inputs contain NaN or inf")
    residual = torch.abs(prediction - target)
    width = float(delta)
    small = 0.5 * torch.square(residual)
    large = width * residual - 0.5 * width * width
    return torch.mean(torch.where(residual <= width, small, large))


class HuberRawLoss(nn.Module):
    def __init__(self, mean: Any, std: Any, delta: float = AUTHOR_HUBER_DELTA) -> None:
        super().__init__()
        self.mean = loss_scale(mean)
        self.std = loss_scale(std)
        self.delta = float(delta)

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        return invert_target(normalized, self.mean, self.std)

    def forward(self, prediction_normalized: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
        return huber_loss(self.denormalize(prediction_normalized), target_raw, delta=self.delta)
