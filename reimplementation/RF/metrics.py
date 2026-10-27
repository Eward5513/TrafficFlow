"""Metrics on the raw 5-minute vehicle-count scale.

MAE and RMSE use the same definitions as
``reimplementation.common.metrics.traffic_metrics``: the mean is over every
evaluated road-time sample, including zeros, and predictions are not clipped
unless a separate clipped result is requested.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from reimplementation.common.metrics.traffic_metrics import traffic_metrics


def geh_statistics(prediction: Any, target: Any) -> dict[str, float | str | None]:
    """Mean GEH where the hourly-style formula is defined.

    Samples with a negative prediction, or with prediction + target <= 0, are
    excluded. These 5-minute counts are not the paper's hourly flows.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    denominator = pred + true
    valid = np.isfinite(pred) & np.isfinite(true) & (pred >= 0.0) & (denominator > 0.0)
    if not np.any(valid):
        mean = None
        within_5 = None
    else:
        values = np.sqrt(2.0 * (pred[valid] - true[valid]) ** 2 / denominator[valid])
        mean = float(values.mean())
        within_5 = float(np.mean(values <= 5.0))
    return {
        "geh_mean": mean,
        "geh_fraction_le_5": within_5,
        "geh_valid_fraction": float(valid.mean()) if pred.size else float("nan"),
        "geh_rule": (
            "sqrt(2 (yhat-y)^2 / (yhat+y)) only where yhat >= 0 and yhat+y > 0; "
            "5-minute counts are not comparable to the paper's hourly GEH"
        ),
    }


def regression_report(prediction: Any, target: Any) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"prediction shape {pred.shape} != target shape {true.shape}")
    if pred.size == 0:
        raise ValueError("cannot score an empty array")
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError("prediction or target contains NaN or Inf")
    raw = traffic_metrics(pred, true, clip_negative=False)
    clipped = traffic_metrics(pred, true, clip_negative=True)
    raw["bias"] = float(np.mean(pred - true))
    clipped_pred = np.maximum(pred, 0.0)
    clipped_summary = {
        "mae": clipped["mae"],
        "rmse": clipped["rmse"],
        "bias": float(np.mean(clipped_pred - true)),
        "mape_nonzero": clipped["mape_nonzero"],
        "wape": clipped["wape"],
    }
    report: dict[str, Any] = {
        "mae": raw["mae"],
        "rmse": raw["rmse"],
        "bias": raw["bias"],
        "mape_nonzero": raw["mape_nonzero"],
        "wape": raw["wape"],
        "sample_count": int(pred.size),
        "negative_prediction_count": int(raw["negative_prediction_count"]),
        "negative_prediction_fraction": raw["negative_prediction_fraction"],
        "minimum_prediction": raw["minimum_prediction"],
        "maximum_prediction": float(pred.max()),
        "raw": raw,
        "clipped_at_zero": clipped_summary,
    }
    report.update(geh_statistics(pred, true))
    return report
