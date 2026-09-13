"""Huber loss on the raw vehicle-count scale.

Official PeMS04 training (``reference/STAEformer/model/train.py``)::

    criterion = nn.HuberLoss()  # PyTorch default delta=1.0
    loss = criterion(SCALER.inverse_transform(pred), y)

``MaskedMAELoss`` is only used for METRLA/PEMSBAY. The R-only traffic target
matches PeMS04 (flow), so this port uses unmasked Huber.

Prepared ``y`` is already z-scored. Inverse-transform predictions with the
public target scaler and compare to ``y_raw``. Real zeros are valid counts,
not PeMS missing readings. This is a data-semantics adaptation, not a change
to the STAEformer architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from reimplementation.common.data.r_only_npz_dataset import invert_target
from reimplementation.common.errors import ReimplementationError

OFFICIAL_HUBER_DELTA = 1.0


def huber_loss(preds: torch.Tensor, labels: torch.Tensor, delta: float = OFFICIAL_HUBER_DELTA) -> torch.Tensor:
    if preds.shape != labels.shape:
        raise ReimplementationError(f"prediction {tuple(preds.shape)} != target {tuple(labels.shape)}")
    if preds.numel() == 0:
        raise ReimplementationError("cannot compute Huber on an empty batch")
    if not torch.isfinite(preds).all() or not torch.isfinite(labels).all():
        raise ReimplementationError("Huber inputs contain NaN or inf")
    residual = torch.abs(preds - labels)
    small = 0.5 * torch.square(residual)
    large = float(delta) * residual - 0.5 * float(delta) * float(delta)
    return torch.mean(torch.where(residual <= float(delta), small, large))


class HuberRawLoss(nn.Module):
    def __init__(self, mean: float, std: float, delta: float = OFFICIAL_HUBER_DELTA) -> None:
        super().__init__()
        self.mean = float(mean)
        self.std = float(std)
        self.delta = float(delta)

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        return invert_target(normalized, self.mean, self.std)

    def forward(self, prediction_normalized: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
        prediction_raw = self.denormalize(prediction_normalized)
        return huber_loss(prediction_raw, target_raw, delta=self.delta)
