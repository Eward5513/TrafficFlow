"""Huber / Smooth-L1 loss matching ``reference/STSGCN/models/stsgcn.py::huber_loss``.

Original training compares the network output with the data iterator's ``label``.
In that repository ``generate_data`` yields normalized ``x`` and **raw** ``y``,
so Huber is computed on the original (vehicle-count) scale with ``rho=1``.

This port keeps the public model output in the prepared normalized ``y`` space
and inverse-transforms with ``mean_y_full`` / ``std_y_full`` before Huber::

    prediction_raw = prediction_normalized * target_std + target_mean
    loss = Huber(prediction_raw, y_raw, rho=1)

MXNet ``MakeLoss`` reduces by mean. Piecewise definition with ``rho=1``::

    |e| > 1  ->  |e| - 0.5
    else     ->  0.5 * e^2
"""

from __future__ import annotations

import torch
import torch.nn as nn

from reimplementation.common.data.r_only_npz_dataset import invert_target


class HuberRawLoss(nn.Module):
    def __init__(self, mean: float, std: float, rho: float = 1.0) -> None:
        super().__init__()
        self.mean = float(mean)
        self.std = float(std)
        self.rho = float(rho)

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        return invert_target(normalized, self.mean, self.std)

    def forward(self, prediction_normalized: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
        if prediction_normalized.shape != target_raw.shape:
            raise ValueError(
                f"prediction {tuple(prediction_normalized.shape)} != "
                f"target {tuple(target_raw.shape)}"
            )
        prediction_raw = self.denormalize(prediction_normalized)
        error = torch.abs(prediction_raw - target_raw)
        quadratic = (0.5 / self.rho) * torch.square(error)
        linear = error - 0.5 * self.rho
        return torch.where(error > self.rho, linear, quadratic).mean()
