"""Four-state diagnostics computed on the inference device before artifact transfer."""

from __future__ import annotations

import numpy as np
import torch

from MSTT.config import STATE_CHANNELS
from MSTT.model.conservation import conservation_residual
from MSTT.utils.metrics import raw_metrics
from reimplementation.common.evaluation import save_predictions_npz, verify_prediction_npz
from reimplementation.common.utils.atomic_io import atomic_write_npz


@torch.no_grad()
def collect_conservation_predictions(loader, predict_fn, *, criterion, device):
    chunks = {}
    residuals = {}
    for incoming in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in incoming.items()}
        outputs = predict_fn(batch)
        final = outputs["final_pred"].float()
        if final.shape != batch["y"].shape:
            raise ValueError(f"prediction {final.shape} != target {batch['y'].shape}")
        physical = criterion.denormalize(final)
        payload = {
            "y_pred_normalized": final, "y_true_normalized": batch["y"],
            "pred_raw": physical, "y_raw": batch["y_raw"],
            **{k: batch[k] for k in ("day_index", "window_start_slot", "target_slot", "sample_index")},
        }
        variants = {"": physical}
        if "correction_weights" in outputs:
            raw = criterion.denormalize(outputs["raw_pred"])
            payload.update({"raw_pred_normalized": outputs["raw_pred"].float(),
                            "raw_pred_physical": raw, "correction_weights": outputs["correction_weights"]})
            variants["raw_"] = raw
        for prefix, pred in variants.items():
            # Conservation is always evaluated with torch on the prediction device.
            r = conservation_residual(pred).abs()
            totals = residuals.setdefault(prefix, [torch.zeros((), dtype=torch.float64, device=device),
                                                   torch.zeros((), device=device), 0])
            totals[0] += r.double().sum()
            totals[1] = torch.maximum(totals[1], r.max())
            totals[2] += r.numel()
        for key, tensor in payload.items():
            chunks.setdefault(key, []).append(tensor.detach().cpu().numpy())
    if not chunks:
        raise ValueError("evaluation loader produced zero batches")
    packed = {k: np.concatenate(v, axis=0) for k, v in chunks.items()}
    order = np.argsort(packed["sample_index"], kind="stable")
    packed = {k: v[order] for k, v in packed.items()}
    packed["y_pred_raw"] = packed["pred_raw"]
    packed["y_true_raw"] = packed["y_raw"]
    if any(not np.isfinite(v).all() for v in packed.values()):
        raise ValueError("non-finite conservation evaluation output")
    diagnostics = {}
    for prefix, (total, maximum, count) in residuals.items():
        diagnostics[prefix + "conservation_mae"] = float(total / count)
        diagnostics[prefix + "conservation_max_abs"] = float(maximum)
        pred = packed["raw_pred_physical" if prefix else "pred_raw"]
        diagnostics.update({prefix + k: v for k, v in negative_statistics(pred).items()})
    if "correction_weights" in packed:
        diagnostics["raw_metrics"] = raw_metrics(packed["raw_pred_physical"], packed["y_raw"], STATE_CHANNELS,
                                                   criterion.std.cpu().numpy())
        diagnostics["raw_mae"] = diagnostics["raw_metrics"]["mae"]
        correction = packed["pred_raw"] - packed["raw_pred_physical"]
        diagnostics["projection_correction_mae"] = float(np.abs(correction).mean())
        weights = packed["correction_weights"].reshape(-1, 4).astype(np.float64)
        diagnostics["correction_weights"] = {
            name: {"mean": float(w.mean()), "std": float(w.std()), "min": float(w.min()),
                   "max": float(w.max()), "p05": float(np.quantile(w, .05)),
                   "p50": float(np.quantile(w, .5)), "p95": float(np.quantile(w, .95))}
            for name, w in zip(STATE_CHANNELS, weights.T)
        }
    return packed, diagnostics


def negative_statistics(pred):
    negative = pred < 0
    magnitude = -pred[negative]
    return {"negative_fraction": float(negative.mean()),
            "negative_mean_abs": float(magnitude.mean()) if magnitude.size else 0.0,
            "negative_max_abs": float(magnitude.max()) if magnitude.size else 0.0,
            "negative_by_channel": {name: {
                "fraction": float((pred[..., i] < 0).mean()),
                "max_abs": float(np.maximum(-pred[..., i], 0).max()),
            } for i, name in enumerate(STATE_CHANNELS)}}


def save_mstt_predictions(path, packed):
    """Keep common names as FINAL predictions and persist C's pre-projection values too."""
    if "raw_pred_physical" not in packed:
        save_predictions_npz(path, packed)
    else:
        atomic_write_npz(path, **packed)
        verify_prediction_npz(path, expected_count=len(packed["pred_raw"]))
