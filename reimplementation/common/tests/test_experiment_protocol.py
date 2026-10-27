"""LibCity cosine scheduler and official rate helpers."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reimplementation.common.experiment_protocol import (
    OFFICIAL_RATES,
    assert_official_rates,
    libcity_cosine_lr,
)


class ProtocolTests(unittest.TestCase):
    def test_official_rates_forbid_p60(self) -> None:
        self.assertEqual(list(OFFICIAL_RATES), ["p05", "p10", "p20", "p30", "p40", "p50", "p70"])
        with self.assertRaises(Exception):
            assert_official_rates(["p05", "p60"])

    def test_cosine_warmup_prefix_false(self) -> None:
        kwargs = dict(
            base_lr=1e-3,
            t_initial=200,
            lr_min=1e-4,
            warmup_t=5,
            warmup_lr_init=1e-6,
            warmup_prefix=False,
        )
        self.assertAlmostEqual(libcity_cosine_lr(0, **kwargs), 1e-6, places=12)
        self.assertAlmostEqual(
            libcity_cosine_lr(4, **kwargs),
            1e-6 + 4.0 * (1e-3 - 1e-6) / 5.0,
            places=12,
        )
        at_five = libcity_cosine_lr(5, **kwargs)
        expected = 1e-4 + 0.5 * (1e-3 - 1e-4) * (1.0 + math.cos(math.pi * 5.0 / 200.0))
        self.assertAlmostEqual(at_five, expected, places=12)
        restarted = 1e-4 + 0.5 * (1e-3 - 1e-4) * (1.0 + math.cos(0.0))
        self.assertNotAlmostEqual(at_five, restarted, places=6)


if __name__ == "__main__":
    unittest.main()
