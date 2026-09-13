"""Masked MAE should use raw zeros, not normalized zeros."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reimplementation.dcrnn.losses import MaskedMAERawLoss, mae_raw_all, mae_raw_nonzero


class LossTests(unittest.TestCase):
    def test_mask_uses_raw_zeros(self) -> None:
        mean = 5.0
        std = 2.0
        loss_fn = MaskedMAERawLoss(mean, std, null_val=0.0)
        pred_norm = torch.tensor([[[[0.0]], [[1.0]]]])
        target_raw = torch.tensor([[[[0.0]], [[7.0]]]])
        # prediction raw = [5, 7]; masked abs err at non-zero label: |7-7|=0
        value = float(loss_fn(pred_norm, target_raw))
        self.assertAlmostEqual(value, 0.0, places=5)

    def test_all_vs_nonzero(self) -> None:
        pred = torch.tensor([1.0, 2.0, 3.0])
        true = torch.tensor([0.0, 2.0, 6.0])
        self.assertAlmostEqual(float(mae_raw_all(pred, true)), (1.0 + 0.0 + 3.0) / 3.0)
        self.assertAlmostEqual(float(mae_raw_nonzero(pred, true)), (0.0 + 3.0) / 2.0)

    def test_rejects_broadcast(self) -> None:
        loss_fn = MaskedMAERawLoss(0.0, 1.0)
        with self.assertRaises(ValueError):
            loss_fn(torch.zeros(2, 1, 4, 1), torch.zeros(2, 1, 4))
