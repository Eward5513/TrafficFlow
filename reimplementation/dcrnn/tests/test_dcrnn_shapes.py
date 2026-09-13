"""DCRNN shape, GO-symbol, and gradient tests. Tiny synthetic graph only."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reimplementation.dcrnn.graph import build_dcrnn_supports
from reimplementation.dcrnn.model.dcrnn_model import DCRNNModel


def _tiny_model(n_nodes: int = 4) -> DCRNNModel:
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for index in range(n_nodes - 1):
        adj[index, index + 1] = 0.5
    supports = build_dcrnn_supports(adj, "dual_random_walk")
    return DCRNNModel(
        supports,
        num_nodes=n_nodes,
        input_dim=1,
        output_dim=1,
        rnn_units=8,
        num_rnn_layers=2,
        max_diffusion_step=2,
        seq_len=12,
        horizon=1,
        use_curriculum_learning=True,
        cl_decay_steps=2000,
    )


class DCRNNShapeTests(unittest.TestCase):
    def test_output_shape_and_go_independence(self) -> None:
        model = _tiny_model()
        x = torch.randn(3, 12, 4, 1)
        y = torch.randn(3, 1, 4, 1)
        model.train()
        pred_a, trace = model(x, y, global_step=0, return_trace=True)
        pred_b = model(x, torch.ones_like(y), global_step=0)
        self.assertEqual(tuple(pred_a.shape), (3, 1, 4, 1))
        self.assertTrue(torch.allclose(pred_a, pred_b, atol=1e-5))
        self.assertFalse(trace["used_label_on_first_step"])
        self.assertTrue(trace["discarded_step_ran"])
        self.assertEqual(trace["go_shape"], [3, 4])

    def test_eval_does_not_need_labels(self) -> None:
        model = _tiny_model().eval()
        x = torch.randn(2, 12, 4, 1)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 1, 4, 1))

    def test_supports_are_buffers(self) -> None:
        model = _tiny_model()
        buffers = dict(model.named_buffers())
        self.assertIn("support_0", buffers)
        self.assertIn("support_1", buffers)
        self.assertNotIn("support_0", dict(model.named_parameters()))
        self.assertGreater(model.parameter_count(), 0)

    def test_gradients_reach_encoder_decoder_projection(self) -> None:
        model = _tiny_model()
        x = torch.randn(2, 12, 4, 1)
        y = torch.randn(2, 1, 4, 1)
        pred = model(x, y, global_step=1)
        pred.sum().backward()
        names = []
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad)
            if float(param.grad.abs().max()) > 0:
                names.append(name)
        self.assertTrue(any(item.startswith("encoder.") for item in names))
        self.assertTrue(any(item.startswith("decoder.") for item in names))
        self.assertTrue(any(item.endswith("proj_w") for item in names))
        self.assertTrue(any("gate_linear.weights" in item for item in names))
