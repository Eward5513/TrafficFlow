"""Process entry point. Thread limits are set before NumPy is imported."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main(config_path: str) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    code_root = "/home/tzhang174/TrafficFlow"
    if code_root not in sys.path:
        sys.path.insert(0, code_root)
    import json

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    from reimplementation.RF.pipeline import train_penetration

    train_penetration(config)


if __name__ == "__main__":
    main(sys.argv[1])
