"""Local reference snapshot used for the STJGCN port.

``reference/STJGCN`` has no ``LICENSE`` file and no independent git metadata.
Only the PeMSD7 scripts were transcribed. Training does not import that
directory and does not load its PeMS arrays or checkpoints.
"""

from __future__ import annotations

SOURCE_REPO = "https://github.com/zhengchuanpan/STJGCN"
SOURCE_PAPER = (
    "Spatio-Temporal Joint Graph Convolutional Networks for Traffic Forecasting, "
    "IEEE TKDE 2024"
)
SOURCE_LOCAL = "reference/STJGCN/PeMSD7"
SOURCE_BRANCH = "PeMSD7 model.py / tf_utils.py / train.py / utils.py"
CODE_VERSION = "xuancheng26-stjgcn-0.1.0"

REFERENCE_SHA256 = {
    "STJGCN/PeMSD7/model.py": "b8e2e9cb9f2804ddabd7f928d7feb560e9cd7966729b61df715638e1faa324fd",
    "STJGCN/PeMSD7/tf_utils.py": "2d70a322feeaf152dd17feb41f3959af22b6fae2958bdc8c46571ac6f8785ffc",
    "STJGCN/PeMSD7/train.py": "7e279d92339f123c099507483cca5cea071d9afda65a18384389073d20e5d170",
    "STJGCN/PeMSD7/utils.py": "49592c97c0b80f2b3497dc13994a1a6ba88d269964e42b75cdb64f83cf0341d2",
}

LICENSE_NOTE = (
    "The local STJGCN snapshot does not include a LICENSE file. "
    "Retain the repository URL, paper citation, and file hashes above."
)
