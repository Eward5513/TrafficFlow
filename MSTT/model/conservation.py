"""Differentiable four-state conservation in vehicle counts, ordered I/O/N0/N1."""

from __future__ import annotations

import torch
from torch import nn


def conservation_residual(physical: torch.Tensor) -> torch.Tensor:
    if physical.shape[-1] != 4:
        raise ValueError("conservation requires four channels [I, O, N_start, N_end]")
    return physical[..., 3] - physical[..., 2] - physical[..., 0] + physical[..., 1]


def project_conservation(physical: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """y' = y - (a^T y) (a * w); w is a probability vector, not an average of y."""
    if weights.shape != physical.shape:
        raise ValueError("correction weights must have the same shape as predictions")
    a = physical.new_tensor((-1.0, 1.0, -1.0, 1.0))
    return physical - conservation_residual(physical).unsqueeze(-1) * (a * weights)


class ConservationProjection(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, mean, std) -> None:
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32).clone()
        std = torch.as_tensor(std, dtype=torch.float32).clone()
        if mean.shape != (4,) or std.shape != (4,):
            raise ValueError("projection needs four independent target mean/std values")
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
            raise ValueError("projection target scaler must be finite with positive std")
        self.register_buffer("target_mean", mean)
        self.register_buffer("target_std", std)
        self.mlp = nn.Sequential(nn.Linear(d_model + 4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 4))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def project_aligned(self, features: torch.Tensor, raw_normalized: torch.Tensor) -> dict[str, torch.Tensor]:
        """Project ``raw_normalized`` with weights from ``cat(features, raw)``.

        Leading dimensions of ``features[..., D]`` and ``raw_normalized[..., 4]``
        must already match. The last-step caller slices time before this method.
        A vectorized ``[B, T, N]`` call is the same map as one call per step.
        """
        if features.ndim < 1 or raw_normalized.ndim < 1:
            raise ValueError("projection features and raw predictions need a channel axis")
        if int(features.shape[-1]) + 4 != int(self.mlp[0].in_features):
            raise ValueError(
                f"projection features width {features.shape[-1]} does not match "
                f"MLP input {self.mlp[0].in_features}"
            )
        if int(raw_normalized.shape[-1]) != 4:
            raise ValueError(f"projection raw predictions must end in 4 channels, got {tuple(raw_normalized.shape)}")
        if features.shape[:-1] != raw_normalized.shape[:-1]:
            raise ValueError(
                f"projection features {tuple(features.shape)} are not aligned with raw {tuple(raw_normalized.shape)}"
            )
        # FP32 even inside AMP: independent channel scales must not be rounded to bf16/fp16.
        with torch.autocast(device_type=raw_normalized.device.type, enabled=False):
            raw_fp32 = raw_normalized.float()
            weights = self.mlp(torch.cat((features.float(), raw_fp32), dim=-1)).softmax(dim=-1)
            raw_physical = raw_fp32 * self.target_std + self.target_mean
            final_physical = project_conservation(raw_physical, weights)
            final_normalized = (final_physical - self.target_mean) / self.target_std
        return {
            "raw_pred": raw_normalized,
            "final_pred": final_normalized,
            "correction_weights": weights,
            "raw_conservation_residual": conservation_residual(raw_physical),
        }

    def forward(self, hidden: torch.Tensor, raw_normalized: torch.Tensor) -> dict[str, torch.Tensor]:
        if hidden.ndim < 2:
            raise ValueError(f"projection hidden must include a time axis, got {tuple(hidden.shape)}")
        return self.project_aligned(hidden[:, -1], raw_normalized)
