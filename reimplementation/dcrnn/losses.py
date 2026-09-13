"""Original DCRNN training loss: inverse-transform then masked MAE.

``reference/dcrnn/lib/metrics.py`` ``masked_mae_loss(scaler, null_val=0)``:

1. inverse-transform predictions and labels with the scaler
2. mask positions where the *raw* label equals ``null_val`` (0)
3. rescale the mask by ``1 / mean(mask)``
4. replace NaNs with 0
5. return the mean absolute error

Normalized ``y == 0`` is **not** a raw-flow zero, so this loss always uses
``y_raw`` after denormalizing the prediction with the shared target scaler.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskedMAERawLoss(nn.Module):
    def __init__(self, mean: float, std: float, null_val: float = 0.0) -> None:
        super().__init__()
        self.mean = float(mean)
        self.std = float(std)
        self.null_val = float(null_val)

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        return normalized * self.std + self.mean

    def forward(self, prediction_normalized: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
        if prediction_normalized.shape != target_raw.shape:
            raise ValueError(
                f"prediction {tuple(prediction_normalized.shape)} != "
                f"target {tuple(target_raw.shape)}"
            )
        preds = self.denormalize(prediction_normalized)
        labels = target_raw
        if self.null_val != self.null_val:  # NaN
            mask = ~torch.isnan(labels)
        else:
            mask = labels != self.null_val
        mask = mask.to(dtype=preds.dtype)
        denom = torch.mean(mask)
        mask = torch.where(denom > 0, mask / denom, torch.zeros_like(mask))
        mask = torch.nan_to_num(mask, nan=0.0)
        loss = torch.abs(preds - labels) * mask
        loss = torch.nan_to_num(loss, nan=0.0)
        return torch.mean(loss)


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
