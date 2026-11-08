"""Five independent sampling realizations; unweighted metrics, sample std (ddof=1)."""
from __future__ import annotations

from functools import wraps
from numbers import Real
from pathlib import Path

import numpy as np

from data_root import SAMPLING_SEEDS
from reimplementation.common.data.sampling_seeds import is_sampling_loader, iter_sampling_loaders
from reimplementation.common.utils.atomic_io import atomic_write_json, atomic_write_npz


def aggregate_sampling_metrics(per_seed: dict) -> dict:
    if tuple(int(s) for s in per_seed) != SAMPLING_SEEDS:
        raise ValueError("evaluation must contain seeds 42–46 in order")
    common = set.intersection(*(set(m) for m in per_seed.values()))
    keys = sorted(k for k in common if all(isinstance(m[k], Real) for m in per_seed.values()))
    mean, std = {}, {}
    for key in keys:
        values = np.asarray([m[key] for m in per_seed.values()], dtype=np.float64)
        mean[key] = float(values.mean())
        std[key] = float(values.std(ddof=1))
    if not np.isfinite(mean.get("mae", np.nan)):
        raise ValueError("validation/test MAE must be finite")
    return dict(sampling_seeds=list(SAMPLING_SEEDS), aggregation="unweighted_mean_of_per_seed_metrics",
                std_ddof=1, per_seed=per_seed, mean=mean, std=std)


def sampling_history(report: dict) -> dict:
    return {"mean_validation_mae": report["mean"]["mae"], "std_validation_mae": report["std"]["mae"],
            **{f"validation_seed_{s}_mae": m["mae"] for s, m in report["per_seed"].items()}}


def evaluate_sampling(loader, evaluate_one, *, output_dir=None, split=None):
    """Callback owns prediction/loss math. Datasets are evaluated sequentially on CPU-backed loaders.

    Keep prediction arrays only when requested by a legacy model-specific output writer.
    """
    per_seed, predictions = {}, {}
    first = None
    for seed, single in iter_sampling_loaders(loader):
        packed = evaluate_one(single)
        metrics = packed["metrics"]
        per_seed[str(seed)] = metrics
        if first is None:
            first = packed
        predictions[str(seed)] = packed
    report = aggregate_sampling_metrics(per_seed)
    result = dict(first)
    result["metrics"] = {**first["metrics"], **report["mean"]}
    result["sampling_metrics"] = report
    result["sampling_predictions"] = predictions
    if output_dir is not None:
        save_sampling_outputs(Path(output_dir), split or loader.dataset.split, result)
    return result


def sampling_epoch(fn):
    """Extend a model's existing pure evaluation routine without changing its training loop."""
    @wraps(fn)
    def wrapped(model, loader, *args, **kwargs):
        if not is_sampling_loader(loader) or kwargs.pop("_single_sampling_seed", False):
            return fn(model, loader, *args, **kwargs)
        return evaluate_sampling(loader, lambda single: fn(model, single, *args, **kwargs))
    return wrapped


def save_sampling_outputs(output_dir: Path, split: str, packed: dict) -> None:
    report = packed.get("sampling_metrics")
    if report is None:
        return
    atomic_write_json(output_dir / f"{split}_sampling_metrics.json", report)
    for seed, prediction in packed["sampling_predictions"].items():
        arrays = {k: v for k, v in prediction.items() if isinstance(v, np.ndarray)}
        atomic_write_npz(output_dir / "predictions" / f"seed_{seed}" / f"{split}.npz", **arrays)


def evaluate_sampling_predictor(loader, predict_fn, *, mean_y, std_y, device):
    from reimplementation.common.evaluation import collect_split_predictions
    from reimplementation.common.metrics.traffic_metrics import traffic_metrics
    def one(single):
        packed = collect_split_predictions(single, predict_fn, mean_y=mean_y, std_y=std_y, device=device)
        packed["metrics"] = traffic_metrics(packed["pred_raw"], packed["y_raw"])
        return packed
    return evaluate_sampling(loader, one)
