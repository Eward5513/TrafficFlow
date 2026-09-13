"""Diffusion convolution matching original DCGRUCell._gconv.

Original (reference/dcrnn/model/dcrnn_cell.py):

    x0 = transpose(x, [1, 2, 0])                 # (N, F, B)
    x0 = reshape(x0, [N, F * B])
    concat x0
    for support in supports:
        x1 = support @ x0
        concat x1
        for k in 2 .. max_diffusion_step:
            x2 = 2 * support @ x1 - x0
            concat x2
            x1, x0 = x2, x1
    num_matrices = len(supports) * max_diffusion_step + 1

Supports are already stored as (D^{-1} A).T (or the reverse analogue).
This module must not transpose them again.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def diffusion_feature_count(num_supports: int, max_diffusion_step: int) -> int:
    if max_diffusion_step < 0:
        raise ValueError(f"max_diffusion_step must be >= 0, got {max_diffusion_step}")
    return int(num_supports) * int(max_diffusion_step) + 1


def build_diffusion_features(
    x: torch.Tensor,
    supports: list[torch.Tensor],
    max_diffusion_step: int,
) -> torch.Tensor:
    """Return the concatenated Chebyshev-style diffusion features.

    Args:
        x: ``[B, N, F]``
        supports: each ``[N, N]``, already in original DCGRU multiply layout
        max_diffusion_step: K; 0 keeps only the identity term

    Returns:
        ``[B * N, F * num_matrices]``
    """
    if x.ndim != 3:
        raise ValueError(f"diffusion input must be [B, N, F], got {tuple(x.shape)}")
    batch_size, num_nodes, input_size = x.shape
    x0 = x.permute(1, 2, 0).reshape(num_nodes, input_size * batch_size)
    terms = [x0]
    if max_diffusion_step > 0:
        for support in supports:
            x1 = torch.matmul(support, x0)
            terms.append(x1)
            prev, curr = x0, x1
            for _ in range(2, max_diffusion_step + 1):
                nxt = 2.0 * torch.matmul(support, curr) - prev
                terms.append(nxt)
                prev, curr = curr, nxt
    stacked = torch.stack(terms, dim=0)
    num_matrices = stacked.shape[0]
    expected = diffusion_feature_count(len(supports), max_diffusion_step)
    if num_matrices != expected:
        raise RuntimeError(
            f"diffusion term count {num_matrices} != {expected} "
            f"(supports={len(supports)}, K={max_diffusion_step})"
        )
    stacked = stacked.view(num_matrices, num_nodes, input_size, batch_size)
    stacked = stacked.permute(3, 1, 2, 0)
    return stacked.reshape(batch_size * num_nodes, input_size * num_matrices)


class DiffusionLinear(nn.Module):
    """Linear map after diffusion features. Xavier weights; constant bias."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_supports: int,
        max_diffusion_step: int,
        *,
        bias_start: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.num_matrices = diffusion_feature_count(num_supports, max_diffusion_step)
        self.weights = nn.Parameter(
            torch.empty(self.input_size * self.num_matrices, self.output_size)
        )
        self.biases = nn.Parameter(torch.empty(self.output_size))
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, float(bias_start))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(features, self.weights.t(), self.biases)
