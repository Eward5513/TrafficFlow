"""Reference ``masked_mae`` on the raw target scale.

``null_val`` stays ``NaN``, so a true zero count is a valid label. The public
metrics are separate: MAE and RMSE keep those zeros, and MAPE uses
``MAPE_nonzero``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def masked_mae(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.to(dtype=preds.dtype)
    mask = mask / torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


class MaskedMAERawLoss(nn.Module):
    def __init__(self, mean, std, null_val: float = np.nan):
        super().__init__()
        self.null_val = float(null_val)
        self.register_buffer("mean", _scale_buffer(mean))
        self.register_buffer("std", _scale_buffer(std))

    def forward(self, prediction_norm: torch.Tensor, label_raw: torch.Tensor) -> torch.Tensor:
        prediction = prediction_norm * _broadcast(self.std, prediction_norm) + _broadcast(self.mean, prediction_norm)
        return masked_mae(prediction, label_raw, null_val=self.null_val)


def _scale_buffer(value) -> torch.Tensor:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0 or array.size == 1:
        return torch.tensor(float(array.reshape(-1)[0]), dtype=torch.float32)
    return torch.tensor(array.reshape(-1).copy(), dtype=torch.float32)


def _broadcast(scale: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if scale.ndim == 0:
        return scale
    view = [1] * int(reference.ndim)
    view[-1] = int(scale.shape[0])
    return scale.view(*view)
