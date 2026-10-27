"""OCR script I/O lives on the data disk; this package is code-only."""

from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from data_root import code_root, ocr_data_dir

OCR_DATA = ocr_data_dir()
GEOJSON = code_root() / "analysis" / "road_network" / "basemap.geojson"
