"""Local reference snapshot used for the ASTTN port.

``reference/ASTTN_pytorch`` has no ``LICENSE`` file and no independent git
metadata. The hashes below are the files this port was transcribed from.
Training does not import that directory.
"""

from __future__ import annotations

SOURCE_REPO = "https://github.com/yokifly/ASTTN_pytorch"
SOURCE_PAPER = (
    "Adaptive Graph Spatial-Temporal Transformer Network for Traffic Forecasting, "
    "CIKM 2022"
)
SOURCE_LOCAL = "reference/ASTTN_pytorch"
SOURCE_BRANCH = "Model_Adp (main.py default model=adp)"
CODE_VERSION = "xuancheng26-asttn-0.1.0"

REFERENCE_SHA256 = {
    "ASTTN_pytorch/main.py": "323bd070f9ee61acbf03e8ce8ef1cae1d1602d251f97f3f9d59b2a5d582271db",
    "ASTTN_pytorch/model/model_adp.py": "1f5e9dad9c162d1a4c8d29a44a7e1d88bda128279f4974287eed382a98e0200e",
    "ASTTN_pytorch/model/model_utils.py": "08159bab27a0d106ac8fcc2b887f970a508d543f97a0b393b76247720b05a1a9",
    "ASTTN_pytorch/model/train.py": "7ef6e505077c3b7102bacc4eb715c8e0d3012639a519623c4a86c95082500702",
    "ASTTN_pytorch/utils.py": "98a8f5deb1ac42be0a605c84b1526793dbc5929e1ee8a76c123590cc00259e3d",
}

LICENSE_NOTE = (
    "The local ASTTN_pytorch snapshot does not include a LICENSE file. "
    "Retain the repository URL, paper citation, and file hashes above."
)
