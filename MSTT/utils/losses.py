"""Training objectives on the raw count scale.

Default ``mae_raw`` is the unmasked raw-scale MAE that STID, Graph WaveNet,
DCRNN, and Seq2seq GCN-LSTM train and select on: predictions are inverted
with the per-channel target scaler and compared with ``y_raw``, zeros
included (they are real counts).

``channel_balance="target_std"`` divides each channel's error by its target
std before averaging. For All2All this stops inflow/outflow (std ~17) from
outweighing accumulation (std ~4); for MAE it equals MAE in target z-space.
It changes the objective, so it is off by default.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


class RawScaleLoss(nn.Module):
    def __init__(
        self,
        mean: Any,
        std: Any,
        *,
        kind: str = "mae_raw",
        huber_delta: float = 1.0,
        channel_balance: str = "none",
    ) -> None:
        super().__init__()
        if kind not in ("mae_raw", "huber_raw", "mse_raw"):
            raise ValueError(f"unknown loss {kind!r}")
        if channel_balance not in ("none", "target_std"):
            raise ValueError(f"unknown channel_balance {channel_balance!r}")
        mean_t = torch.as_tensor(np.atleast_1d(np.asarray(mean, dtype=np.float64)), dtype=torch.float32)
        std_t = torch.as_tensor(np.atleast_1d(np.asarray(std, dtype=np.float64)), dtype=torch.float32)
        if mean_t.shape != std_t.shape or bool((std_t <= 0).any()):
            raise ValueError("target scaler mean/std must match and std must be positive")
        self.register_buffer("mean", mean_t, persistent=False)
        self.register_buffer("std", std_t, persistent=False)
        self.kind = kind
        self.huber_delta = float(huber_delta)
        self.channel_balance = channel_balance

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        return normalized.float() * self.std + self.mean

    def elementwise(self, prediction_normalized: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
        if prediction_normalized.shape != target_raw.shape:
            raise ValueError(f"prediction {tuple(prediction_normalized.shape)} != target {tuple(target_raw.shape)}")
        if int(target_raw.shape[-1]) != int(self.mean.numel()):
            raise ValueError("channel axis does not match the target scaler")
        error = self.denormalize(prediction_normalized) - target_raw.float()
        if self.channel_balance == "target_std":
            error = error / self.std
        if self.kind == "mae_raw":
            return error.abs()
        if self.kind == "mse_raw":
            return error.square()
        delta = self.huber_delta
        magnitude = error.abs()
        return torch.where(magnitude <= delta, 0.5 * magnitude.square(), delta * magnitude - 0.5 * delta * delta)

    def forward(self, prediction_normalized: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
        return self.elementwise(prediction_normalized, target_raw).mean()

    def describe(self) -> str:
        name = self.kind if self.kind != "huber_raw" else f"huber_raw_delta_{self.huber_delta:g}"
        return name if self.channel_balance == "none" else f"{name}__balance_{self.channel_balance}"
