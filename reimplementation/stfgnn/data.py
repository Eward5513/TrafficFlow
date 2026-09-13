"""R-only NPZ access for STFGNN. Does not refit scalers or rebuild windows."""

from __future__ import annotations

from reimplementation.common.data.r_only_npz_dataset import (
    NPZ_KEYS,
    RATE_TAGS,
    SPLITS,
    ROnlyNPZDataset,
    build_dataloader,
    expected_split_counts,
    invert_target,
    load_json,
    load_npz,
    rate_tag,
    read_target_scaler,
)

__all__ = [
    "NPZ_KEYS",
    "RATE_TAGS",
    "SPLITS",
    "ROnlyNPZDataset",
    "build_dataloader",
    "expected_split_counts",
    "invert_target",
    "load_json",
    "load_npz",
    "rate_tag",
    "read_target_scaler",
]
