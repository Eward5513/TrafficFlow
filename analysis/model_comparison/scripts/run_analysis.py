"""Regenerate the completed-experiment audit.

Reads result directories and writes only under analysis/model_comparison/.
Does not train and does not use a GPU.

    python3 analysis/model_comparison/scripts/run_analysis.py
"""

from __future__ import annotations

import build_inventory
import build_metrics
import error_distribution
import render_reports


def main() -> None:
    build_inventory.main()
    build_metrics.main()
    error_distribution.main()
    render_reports.build()


if __name__ == "__main__":
    main()
