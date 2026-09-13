"""MAE / RMSE / MAPE_nonzero / WAPE on the raw vehicle-count scale.

MAPE is computed only where ``target > 0``. No epsilon is added to the
denominator. Negative predictions are not clipped.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def traffic_metrics(
    prediction: Any,
    target: Any,
    *,
    clip_negative: bool = False,
) -> dict[str, float]:
    pred = _as_numpy(prediction)
    true = _as_numpy(target)
    if pred.shape != true.shape:
        raise ValueError(f"prediction shape {pred.shape} != target shape {true.shape}")
    pred_eval = np.maximum(pred, 0.0) if clip_negative else pred
    abs_err = np.abs(pred_eval - true)
    sq_err = (pred_eval - true) ** 2
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(sq_err)))
    positive = true > 0.0
    nonzero_count = int(positive.sum())
    zero_count = int((~positive).sum())
    if nonzero_count:
        mape = float(np.mean(abs_err[positive] / true[positive]) * 100.0)
    else:
        mape = float("nan")
    denom = float(np.sum(np.abs(true)))
    wape = float(np.sum(abs_err) / denom * 100.0) if denom > 0.0 else float("nan")
    negative = pred < 0.0
    return {
        "mae": mae,
        "rmse": rmse,
        "mape_nonzero": mape,
        "wape": wape,
        "nonzero_target_count": float(nonzero_count),
        "zero_target_count": float(zero_count),
        "zero_target_fraction": float(zero_count / true.size) if true.size else float("nan"),
        "negative_prediction_count": float(int(negative.sum())),
        "negative_prediction_fraction": float(negative.mean()) if pred.size else float("nan"),
        "minimum_prediction": float(pred.min()) if pred.size else float("nan"),
        "clipped_negative_predictions": float(clip_negative),
    }
