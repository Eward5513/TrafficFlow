"""STGCN training loss matching ``tf.nn.l2_loss`` on prediction error.

Original ``build_model`` (base_model.py)::

    train_loss = tf.nn.l2_loss(y - inputs[:, n_his:n_his+1, :, :])

``tf.nn.l2_loss(t)`` is ``sum(t ** 2) / 2``. Weight tensors are collected with
the same ``l2_loss`` into ``weight_decay``, but the original optimizer
minimizes **only** ``train_loss``. Biases and LayerNorm parameters are not
in that collection.

``copy_loss`` compared the last history frame to the *next* future frame. This
project does not use future labels, so that term is not computed.
"""

from __future__ import annotations

import torch
from torch import nn


def tf_l2_loss(residual: torch.Tensor) -> torch.Tensor:
    """Equivalent to TensorFlow ``tf.nn.l2_loss``: ``sum(x^2) / 2``."""
    return 0.5 * torch.sum(residual * residual)


class STGCNPredictionLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction shape {tuple(prediction.shape)} != target shape {tuple(target.shape)}"
            )
        return tf_l2_loss(prediction - target)


def collected_weight_l2(weights: list[torch.Tensor]) -> torch.Tensor:
    if not weights:
        return torch.zeros((), dtype=torch.float32)
    total = weights[0].new_zeros(())
    for tensor in weights:
        total = total + 0.5 * tensor.pow(2).sum()
    return total
