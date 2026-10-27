"""Import ``data_root`` when simulation scripts run from the repository root."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data_root import (  # noqa: E402
    node_table,
    simulation_data_dir,
    subgraph_file,
    subgraph_trajectories_dir,
)

__all__ = [
    "node_table",
    "simulation_data_dir",
    "subgraph_file",
    "subgraph_trajectories_dir",
]
