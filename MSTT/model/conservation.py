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

    def forward(self, hidden: torch.Tensor, raw_normalized: torch.Tensor) -> dict[str, torch.Tensor]:
        # FP32 even inside AMP: independent channel scales must not be rounded to bf16/fp16.
        with torch.autocast(device_type=raw_normalized.device.type, enabled=False):
            features = torch.cat((hidden[:, -1].float(), raw_normalized.float()), dim=-1)
            weights = self.mlp(features).softmax(dim=-1)
            raw_physical = raw_normalized.float() * self.target_std + self.target_mean
            final_physical = project_conservation(raw_physical, weights)
            final_normalized = (final_physical - self.target_mean) / self.target_std
        return {
            "raw_pred": raw_normalized,
            "final_pred": final_normalized,
            "correction_weights": weights,
            "raw_conservation_residual": conservation_residual(raw_physical),
        }
