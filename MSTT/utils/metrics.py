"""Raw-scale metrics through the shared ``traffic_metrics`` (MAE, RMSE, MAPE_nonzero, WAPE).

``traffic_metrics`` already adds ``inflow_mae`` ... ``accumulation_end_wape``
when the channel axis has 4 entries. ``channel_summary`` adds one aggregate
that no channel dominates: ``scaled_mae = mean_c(MAE_c / std_c)``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from reimplementation.common.metrics.traffic_metrics import traffic_metrics

CORE = ("mae", "rmse", "mape_nonzero", "wape")


def raw_metrics(pred_raw: np.ndarray, y_raw: np.ndarray, channel_names: Sequence[str], target_std: Any) -> dict[str, float]:
    metrics = traffic_metrics(pred_raw, y_raw, clip_negative=False)
    std = np.atleast_1d(np.asarray(target_std, dtype=np.float64))
    if len(channel_names) == 1:
        name = channel_names[0]
        for key in CORE:
            metrics[f"{name}_{key}"] = metrics[key]
        metrics["scaled_mae"] = metrics["mae"] / float(std[0])
    else:
        metrics["scaled_mae"] = float(np.mean([metrics[f"{name}_mae"] / s for name, s in zip(channel_names, std)]))
        metrics["mean_channel_wape"] = float(np.mean([metrics[f"{name}_wape"] for name in channel_names]))
    metrics["channels"] = list(channel_names)
    return metrics


def flat_core(metrics: Mapping[str, Any], channel_names: Sequence[str], prefix: str) -> dict[str, float]:
    row = {f"{prefix}_{key}": float(metrics[key]) for key in CORE + ("scaled_mae",)}
    for name in channel_names:
        for key in CORE:
            row[f"{prefix}_{name}_{key}"] = float(metrics[f"{name}_{key}"])
    return row
