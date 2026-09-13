"""Directed support construction tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reimplementation.dcrnn.graph import build_dcrnn_supports
from reimplementation.dcrnn.validation import chain_direction_ok


class SupportTests(unittest.TestCase):
    def test_dual_random_walk_chain(self) -> None:
        result = chain_direction_ok([])
        self.assertEqual(result["status"], "ok")

    def test_asymmetric_graph_yields_distinct_supports(self) -> None:
        adj = np.array([[0.0, 0.7, 0.0], [0.0, 0.0, 0.4], [0.2, 0.0, 0.0]], dtype=np.float32)
        supports = build_dcrnn_supports(adj, "dual_random_walk")
        self.assertEqual(len(supports), 2)
        self.assertFalse(np.allclose(supports[0], supports[1]))
        self.assertTrue(np.isfinite(supports[0]).all())
        self.assertTrue(np.isfinite(supports[1]).all())

    def test_zero_out_degree_row_is_stable(self) -> None:
        adj = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
        supports = build_dcrnn_supports(adj, "random_walk")
        self.assertEqual(len(supports), 1)
        self.assertTrue(np.isfinite(supports[0]).all())
