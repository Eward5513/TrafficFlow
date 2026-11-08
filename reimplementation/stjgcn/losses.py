"""PeMSD7 training loss on the raw count scale.

``MAE + beta * 100 * abs(error) / (label + 1)``. Labels equal to zero are
masked, and the mask is divided by the fraction of valid entries before the
mean. An all-zero target returns a finite zero that still depends on the
prediction, so backward does not see a NaN from dividing by that fraction.

Evaluation does not use this mask. Public MAE and RMSE include valid zeros.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from reimplementation.asttn.losses import _broadcast, _scale_buffer


class STJGCNRawLoss(nn.Module):
    def __init__(self, mean, std, *, beta: float = 0.5, epsilon: float = 1.0):
        super().__init__()
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.register_buffer("mean", _scale_buffer(mean))
        self.register_buffer("std", _scale_buffer(std))

    def forward(self, prediction_norm: torch.Tensor, label_raw: torch.Tensor) -> torch.Tensor:
        prediction = prediction_norm * _broadcast(self.std, prediction_norm) + _broadcast(self.mean, prediction_norm)
        return stjgcn_loss(prediction, label_raw, beta=self.beta, epsilon=self.epsilon)


def stjgcn_loss(prediction: torch.Tensor, label: torch.Tensor, *, beta: float = 0.5, epsilon: float = 1.0) -> torch.Tensor:
    if prediction.shape != label.shape:
        raise ValueError(f"prediction {tuple(prediction.shape)} != label {tuple(label.shape)}")
    valid = label != 0
    if label.numel() == 0 or not bool(valid.any().item()):
        return prediction.sum() * 0.0
    mask = valid.to(dtype=prediction.dtype)
    mask = mask / mask.mean()
    absolute = torch.abs(prediction - label)
    loss = (absolute + float(beta) * (100.0 * absolute / (label + float(epsilon)))) * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)
