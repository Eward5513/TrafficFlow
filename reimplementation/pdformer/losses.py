"""Huber loss on the raw vehicle-count scale.

Original PeMS04/08 config: ``set_loss='huber'``, ``huber_delta=2``.
``libcity.model.loss.huber_loss`` is unmasked::

    r = |pred - label|
    0.5 r^2                 if r <= delta
    delta * r - 0.5 delta^2 otherwise

Original applies this after inverse-transform. Prepared R-only ``y`` is already
z-scored, so this module inverse-transforms the prediction with the public
target scaler and compares it to ``y_raw``.

Real zeros are valid counts. This is **not** ``masked_huber`` / ``masked_mae``
and does not treat ``target == 0`` as missing.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from reimplementation.common.data.r_only_npz_dataset import invert_target
from reimplementation.common.errors import ReimplementationError


def huber_loss(preds: torch.Tensor, labels: torch.Tensor, delta: float = 2.0) -> torch.Tensor:
    residual = torch.abs(preds - labels)
    condition = torch.le(residual, delta)
    small_res = 0.5 * torch.square(residual)
    large_res = delta * residual - 0.5 * delta * delta
    return torch.mean(torch.where(condition, small_res, large_res))


class HuberRawLoss(nn.Module):
    def __init__(self, mean: float, std: float, delta: float = 2.0) -> None:
        super().__init__()
        self.mean = float(mean)
        self.std = float(std)
        self.delta = float(delta)

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        return invert_target(normalized, self.mean, self.std)

    def forward(self, prediction_normalized: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
        if prediction_normalized.shape != target_raw.shape:
            raise ReimplementationError(
                f"prediction {tuple(prediction_normalized.shape)} != "
                f"target {tuple(target_raw.shape)}"
            )
        prediction_raw = self.denormalize(prediction_normalized)
        return huber_loss(prediction_raw, target_raw, delta=self.delta)
