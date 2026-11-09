"""Train and evaluate one penetration. Imported after thread limits are set."""

from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from reimplementation.RF.data_loader import (
    N_TARGETS,
    STATES,
    XuanchengRFData,
    feature_names_for,
    flow_column_for,
    validate_task,
)
from reimplementation.RF.metrics import regression_report
from reimplementation.RF.model import N_CLUSTERS, N_ESTIMATORS, RFEstimator

RESULTS_ROOT = Path(
    os.environ.get(
        "XUANCHENG_RF_RESULTS_ROOT",
        str(__import__("data_root").xuancheng26_model_results_3s_dir("RF")),
    )
).expanduser().resolve()
CODE_ROOT = Path("/home/tzhang174/TrafficFlow")


class RunError(RuntimeError):
    """One penetration run failed."""


def _log(handle, message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    line = f"{stamp} {message}"
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def _rss_mib() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def _assert_output(path: Path) -> None:
    resolved = path.resolve()
    code = CODE_ROOT.resolve()
    if resolved == code or code in resolved.parents:
        raise RunError(f"refusing to write into the git tree: {resolved}")
    legacy = RESULTS_ROOT.resolve()
    from data_root import xuancheng26_training_results_3s_root

    canonical = xuancheng26_training_results_3s_root().resolve()
    under_legacy = resolved == legacy or legacy in resolved.parents
    under_canonical = "RF" in resolved.parts and (resolved == canonical or canonical in resolved.parents)
    if not under_legacy and not under_canonical:
        raise RunError(f"refusing to write outside {legacy} or {canonical}: {resolved}")


_FITTED_FILENAMES = (
    "scaler.joblib",
    "kmeans.joblib",
    "random_forest.joblib",
    "regression_coefficients.npz",
)
_KEPT_MODEL_FILENAMES = ("model_config.json",)


def _model_files(model_dir: Path) -> list[Path]:
    return [model_dir / name for name in _FITTED_FILENAMES]


def _model_dirs(rate_dir: Path, multi: bool) -> list[Path]:
    if multi:
        return [rate_dir / "model" / state for state in STATES]
    return [rate_dir / "model"]


def _head_dirs(rate_dir: Path, multi: bool, heads: list[str]) -> list[tuple[str, Path]]:
    if multi:
        return [(state, rate_dir / "model" / state) for state in heads]
    return [(heads[0], rate_dir / "model")]


def _outputs_ready(rate_dir: Path) -> bool:
    required = (
        rate_dir / "metrics" / "test_metrics.json",
        rate_dir / "metrics" / "validation_metrics.json",
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    return all(any((rate_dir / "predictions" / f"seed_{seed}" / "test").glob("*.npz")) for seed in range(42,47))


def _fitted_complete(rate_dir: Path, multi: bool) -> bool:
    for model_dir in _model_dirs(rate_dir, multi):
        for path in _model_files(model_dir):
            if not path.is_file() or path.stat().st_size <= 0:
                return False
    return True


def _any_fitted(rate_dir: Path, multi: bool) -> bool:
    for model_dir in _model_dirs(rate_dir, multi):
        for path in _model_files(model_dir):
            if path.is_file() and path.stat().st_size > 0:
                return True
    return False


def _read_discard_record(rate_dir: Path) -> dict | None:
    path = rate_dir / "model_discarded.json"
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("discarded"):
        return None
    return payload


def _complete(rate_dir: Path, multi: bool) -> bool:
    return (_outputs_ready(rate_dir) and _fitted_complete(rate_dir, multi)
            and (rate_dir / "test_sampling_metrics.json").is_file()
            and (rate_dir / "validation_sampling_metrics.json").is_file())


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _regression_parameters(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            status = np.asarray(payload["optimizer_status"]).tolist()
            return {
                "alpha": np.asarray(payload["alpha"], dtype=float).tolist(),
                "beta": np.asarray(payload["beta"], dtype=float).tolist(),
                "cluster_sample_counts": np.asarray(payload["cluster_sample_counts"], dtype=int).tolist(),
                "cluster_training_mae": np.asarray(payload["cluster_training_mae"], dtype=float).tolist(),
                "optimizer_status": [str(item) for item in status],
                "n_clusters": int(payload["n_clusters"]) if "n_clusters" in payload.files else None,
            }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _kmeans_parameters(path: Path) -> dict | None:
    """Record cluster centers only. The joblib also holds every training label."""
    if not path.is_file():
        return None
    try:
        import joblib

        model = joblib.load(path)
        payload = {}
        centers = getattr(model, "cluster_centers_", None)
        if centers is not None:
            payload["cluster_centers"] = np.asarray(centers, dtype=float).tolist()
        n_iter = getattr(model, "n_iter_", None)
        if n_iter is not None:
            payload["n_iter"] = int(n_iter)
        inertia = getattr(model, "inertia_", None)
        if inertia is not None and np.isfinite(inertia):
            payload["inertia"] = float(inertia)
        return payload
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _forest_structure(estimator: RFEstimator) -> dict:
    trees = list(estimator.forest.estimators_)
    node_counts = [int(tree.tree_.node_count) for tree in trees]
    depths = [int(tree.tree_.max_depth) for tree in trees]
    n_classes = getattr(estimator.forest, "n_classes_", None)
    if n_classes is not None:
        n_classes = int(np.max(np.asarray(n_classes)))
    return {
        "n_trees": len(node_counts),
        "n_classes": n_classes,
        "node_count_sum": int(sum(node_counts)),
        "node_count_min": int(min(node_counts)) if node_counts else 0,
        "node_count_max": int(max(node_counts)) if node_counts else 0,
        "max_depth_max": int(max(depths)) if depths else 0,
    }


def _state_record(state: str, model_dir: Path, estimator: RFEstimator | None) -> dict:
    files = []
    total = 0
    for name in _FITTED_FILENAMES + _KEPT_MODEL_FILENAMES:
        path = model_dir / name
        present = path.is_file()
        size = path.stat().st_size if present else 0
        files.append(
            {
                "name": name,
                "bytes": size,
                "removed": name in _FITTED_FILENAMES and present,
            }
        )
        if name in _FITTED_FILENAMES:
            total += size
    parameters = None
    config_path = model_dir / "model_config.json"
    if config_path.is_file():
        try:
            parameters = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            parameters = {"error": str(exc)}
    record = {
        "state": state,
        "directory": str(model_dir),
        "total_bytes": total,
        "files": files,
        "parameters": parameters,
        "regression": _regression_parameters(model_dir / "regression_coefficients.npz"),
        "kmeans": _kmeans_parameters(model_dir / "kmeans.joblib"),
    }
    if estimator is not None:
        try:
            record["forest"] = _forest_structure(estimator)
        except Exception as exc:
            record["forest"] = {"error": f"{type(exc).__name__}: {exc}"}
    return record


def _unlink_fitted(model_dir: Path) -> None:
    for path in _model_files(model_dir):
        if path.is_file() or path.is_symlink():
            path.unlink()


def _discard_fitted_models(
    rate_dir: Path,
    multi: bool,
    heads: list[str],
    *,
    input_type: str,
    target_type: str,
    estimators: list[tuple[str, RFEstimator]] | None = None,
) -> dict:
    """Write sizes and parameters, then delete the fitted forest files.

    Predictions, metrics, ``config.json``, and ``model_config.json`` stay.
    A second call only removes files left behind by an interrupted delete.
    """
    existing = _read_discard_record(rate_dir)
    if _any_fitted(rate_dir, multi) and existing is not None and not _fitted_complete(rate_dir, multi):
        for _state, model_dir in _head_dirs(rate_dir, multi, heads):
            _unlink_fitted(model_dir)
        return existing
    if not _any_fitted(rate_dir, multi):
        if existing is not None:
            return existing
        raise RunError(f"no fitted forest to remove under {rate_dir}")
    estimator_map = dict(estimators or [])
    by_state = {}
    total = 0
    for state, model_dir in _head_dirs(rate_dir, multi, heads):
        record = _state_record(state, model_dir, estimator_map.get(state))
        by_state[state] = record
        total += int(record["total_bytes"])
    summary = {
        "discarded": True,
        "input_type": input_type,
        "target_type": target_type,
        "penetration": rate_dir.name,
        "total_bytes": total,
        "kept": [
            "predictions/",
            "metrics/",
            "config.json",
            "active_edge_mask.npy",
            "model/model_config.json",
        ],
        "states": by_state,
    }
    _write_payload(rate_dir / "model_discarded.json", summary)
    for _state, model_dir in _head_dirs(rate_dir, multi, heads):
        _unlink_fitted(model_dir)
    return summary


def _predict_day(estimators: list[tuple[str, RFEstimator]], cube: np.ndarray, multi: bool) -> np.ndarray:
    flat = cube.reshape(-1, cube.shape[-1])
    channels = [
        estimator.predict(flat).reshape(N_TARGETS, cube.shape[1])
        for _name, estimator in estimators
    ]
    if not multi:
        return channels[0]
    return np.stack(channels, axis=-1)


def _score_split(
    estimators: list[tuple[str, RFEstimator]],
    data: XuanchengRFData,
    penetration: str,
    days: list[str],
    prediction_dir: Path | None,
    *,
    input_type: str,
    target_type: str,
) -> dict:
    multi = target_type == "all"
    predictions = []
    targets = []
    state_predictions = {name: [] for name, _estimator in estimators} if multi else {}
    state_targets = {name: [] for name, _estimator in estimators} if multi else {}
    daily = []
    for day_name in days:
        cube, target = data.day_cube(penetration, day_name, input_type, target_type)
        predicted = _predict_day(estimators, cube, multi)
        report = regression_report(predicted, target)
        daily.append(
            {
                "date": day_name,
                "mae": report["mae"],
                "rmse": report["rmse"],
                "bias": report["bias"],
                "negative_prediction_fraction": report["negative_prediction_fraction"],
                "sample_count": report["sample_count"],
            }
        )
        if prediction_dir is not None:
            prediction_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "prediction": predicted.astype(np.float32),
                "ground_truth": np.asarray(target, dtype=np.float32),
                "edge_ids": np.asarray(data.edges, dtype="U32"),
                "time_bins": np.arange(11, 11 + N_TARGETS, dtype=np.int16),
                "date": np.asarray(day_name),
                "penetration": np.asarray(penetration),
                "input_type": np.asarray(input_type),
                "target_type": np.asarray(target_type),
                "supervised_bin_start": np.int16(11),
                "supervised_bin_end_inclusive": np.int16(287),
            }
            if multi:
                payload["states"] = np.asarray(STATES, dtype="U32")
            np.savez_compressed(prediction_dir / f"{day_name}.npz", **payload)
        predictions.append(predicted.reshape(-1))
        targets.append(np.asarray(target).reshape(-1))
        if multi:
            for index, (name, _estimator) in enumerate(estimators):
                state_predictions[name].append(predicted[:, :, index].reshape(-1))
                state_targets[name].append(np.asarray(target)[:, :, index].reshape(-1))
    overall = regression_report(np.concatenate(predictions), np.concatenate(targets))
    result = {"overall": overall, "daily": daily}
    if multi:
        result["states"] = {
            name: regression_report(np.concatenate(state_predictions[name]), np.concatenate(state_targets[name]))
            for name, _estimator in estimators
        }
    return result


def _active_mask(target: np.ndarray, n_days: int, n_edges: int) -> np.ndarray:
    summed = target.reshape(n_days, N_TARGETS, n_edges).sum(axis=(0, 1))
    return summed > 0


def _task_types(config: dict) -> tuple[str, str, bool, list[str]]:
    input_type = str(config.get("input_type", "inflow"))
    target_type = str(config.get("target_type", "inflow"))
    validate_task(input_type, target_type)
    multi = target_type == "all"
    heads = list(STATES) if multi else [target_type]
    return input_type, target_type, multi, heads


def _fit_head(
    data: XuanchengRFData,
    config: dict,
    penetration: str,
    input_type: str,
    state: str,
    model_dir: Path,
    log,
) -> tuple[RFEstimator, np.ndarray, float]:
    load_started = time.perf_counter()
    features, target = data.stack(penetration, data.days["train"], input_type, state)
    names = feature_names_for(input_type)
    column = flow_column_for(input_type, state)
    log(
        f"{state} train matrix {features.shape} flow_column={column} "
        f"load_seconds={time.perf_counter() - load_started:.1f} rss_mib={_rss_mib()}"
    )
    estimator = RFEstimator(
        n_clusters=int(config["n_clusters"]),
        n_estimators=int(config["n_estimators"]),
        seed=int(config["seed"]),
        n_jobs=int(config["n_jobs"]),
        kmeans_n_init=int(config["kmeans_n_init"]),
    )
    fit_started = time.perf_counter()
    estimator.fit(features, target, flow_column=column, feature_names=names)
    fit_seconds = time.perf_counter() - fit_started
    active = _active_mask(target, len(data.days["train"]), len(data.edges))
    estimator.save(model_dir)
    counts = estimator.cluster_samples
    log(
        f"{state} fit_seconds={fit_seconds:.1f} kmeans_iter={estimator.kmeans_iterations} "
        f"inertia={estimator.kmeans_inertia:.6g} "
        f"cluster_min={int(counts.min())} cluster_max={int(counts.max())} "
        f"empty={int(np.count_nonzero(counts == 0))} rss_mib={_rss_mib()}"
    )
    for cluster_id, status in enumerate(estimator.cluster_status):
        if status not in {"converged", "duplicate_flow", "expanded_bounds"}:
            log(f"{state} cluster {cluster_id} status={status} samples={int(counts[cluster_id])}")
    del features
    return estimator, active, fit_seconds


def _active_report(rate_dir: Path, days: list[str], multi: bool):
    active_path = rate_dir / "active_edge_mask.npy"
    if not active_path.is_file():
        return None
    mask = np.load(active_path)
    pieces_p = []
    pieces_t = []
    per_state = {state: ([], []) for state in STATES} if multi else None
    for day_name in days:
        with np.load(rate_dir / "predictions" / "seed_42" / "test" / f"{day_name}.npz") as payload:
            prediction = payload["prediction"]
            truth = payload["ground_truth"]
        if multi:
            for index, state in enumerate(STATES):
                per_state[state][0].append(prediction[:, mask[index], index].reshape(-1))
                per_state[state][1].append(truth[:, mask[index], index].reshape(-1))
                pieces_p.append(per_state[state][0][-1])
                pieces_t.append(per_state[state][1][-1])
        else:
            pieces_p.append(prediction[:, mask].reshape(-1))
            pieces_t.append(truth[:, mask].reshape(-1))
    report = regression_report(np.concatenate(pieces_p), np.concatenate(pieces_t))
    if not multi:
        return report
    return {
        "overall": report,
        "states": {
            state: regression_report(np.concatenate(values[0]), np.concatenate(values[1]))
            for state, values in per_state.items()
        },
    }


def _confirm_reload(
    estimators: list[tuple[str, RFEstimator]],
    data: XuanchengRFData,
    penetration: str,
    sample_day: str,
    rate_dir: Path,
    input_type: str,
    target_type: str,
) -> None:
    multi = target_type == "all"
    cube, _target = data.day_cube(penetration, sample_day, input_type, target_type)
    recomputed = _predict_day(estimators, cube, multi)
    with np.load(rate_dir / "predictions" / "seed_42" / "test" / f"{sample_day}.npz") as payload:
        saved = payload["prediction"]
    if not np.allclose(recomputed, saved, atol=1e-4, rtol=1e-4):
        raise RunError("reloaded model does not reproduce the saved test prediction")


def _rf_data(config: dict, sampling_seed: int = 42):
    kwargs = {"dataset_id": str(config.get("dataset_id") or "fixed_uniform")}
    if config.get("training_root"):
        kwargs["training_root"] = Path(config["training_root"])
    return XuanchengRFData(sampling_seed, **kwargs)


def _aggregate_state_reports(samples: list[dict]) -> tuple[dict, dict]:
    if not samples:
        return {}, {}
    names = list(samples[0])
    mean: dict = {}
    std: dict = {}
    for name in names:
        numeric = [
            key for key, value in samples[0][name].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        mean[name] = {}
        std[name] = {}
        for key in numeric:
            values = np.asarray([float(item[name][key]) for item in samples], dtype=np.float64)
            mean[name][key] = float(values.mean())
            std[name][key] = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return mean, std


def train_penetration(config: dict) -> None:
    penetration = str(config["penetration"])
    input_type, target_type, multi, heads = _task_types(config)
    rate_dir = Path(config["rate_dir"]) if config.get("rate_dir") else Path(config["run_dir"]) / penetration
    from reimplementation.common.data.sampling_seeds import protect_historical_results
    protect_historical_results(rate_dir)
    _assert_output(rate_dir)
    rate_dir.mkdir(parents=True, exist_ok=True)
    _write_payload(rate_dir / "resolved_config.json", {
        **config,
        "model_name": "RF",
        "rates": [penetration],
        "model_seed": int(config["seed"]),
        "train_sampling_seed": 42,
        "evaluation_sampling_seeds": [42, 43, 44, 45, 46],
        "checkpoint_selection": "mean_validation_mae",
    })
    log_path = rate_dir / "logs" / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as handle:
        def log(message: str) -> None:
            _log(handle, f"[{penetration} {input_type}->{target_type}] {message}")

        try:
            log(f"start pid={os.getpid()} ppid={os.getppid()} n_jobs={config['n_jobs']}")
            if _complete(rate_dir, multi) and not config["force"] and not config["evaluate_only"]:
                log("SKIP completed five-sampling run")
                return
            data = _rf_data(config)
            if config["evaluate_only"]:
                if not _fitted_complete(rate_dir, multi):
                    if _read_discard_record(rate_dir) is not None:
                        raise RunError("evaluate-only found predictions but the fitted forest was removed")
                    raise RunError("evaluate-only found no saved model")
                estimators = [
                    (state, RFEstimator.load(rate_dir / "model" / state if multi else rate_dir / "model"))
                    for state in heads
                ]
                log("loaded saved model")
            else:
                estimators = []
                masks = []
                fit_seconds = {}
                names = feature_names_for(input_type)
                for state in heads:
                    model_dir = rate_dir / "model" / state if multi else rate_dir / "model"
                    estimator, active, seconds = _fit_head(
                        data, config, penetration, input_type, state, model_dir, log
                    )
                    estimators.append((state, estimator))
                    masks.append(active)
                    fit_seconds[state] = seconds
                active_mask = np.stack(masks, axis=0) if multi else masks[0]
                np.save(rate_dir / "active_edge_mask.npy", active_mask)
                config_payload = {
                    "penetration": penetration,
                    "input_type": input_type,
                    "target_type": target_type,
                    "feature_names": list(names),
                    "flow_columns": {
                        state: flow_column_for(input_type, state) for state in heads
                    },
                    "seed": int(config["seed"]),
                    "n_clusters": int(config["n_clusters"]),
                    "n_estimators": int(config["n_estimators"]),
                    "kmeans_n_init": int(config["kmeans_n_init"]),
                    "fit_seconds": fit_seconds if multi else fit_seconds[heads[0]],
                    "train_days": data.days["train"],
                    "adaptations": [
                        "K-Means uses every model feature; the paper does not specify this.",
                        "StandardScaler is fit on the training penetration only.",
                        "Functional class is replaced by num_lanes.",
                        "Capacity is replaced by the SUMO speed limit in km/h.",
                        "Supervised bins are 11..287.",
                        "The univariate line uses the partial count of the target state.",
                        "all output is four independent single-state estimators.",
                    ],
                }
                (rate_dir / "config.json").write_text(
                    json.dumps(config_payload, indent=2) + "\n",
                    encoding="utf-8",
                )
            score_kwargs = {"input_type": input_type, "target_type": target_type}
            from reimplementation.common.sampling_evaluation import aggregate_sampling_metrics
            from data_root import SAMPLING_SEEDS
            reports = {}
            seed42 = {}
            state_samples = {"validation": [], "test": []}
            for split in ("validation", "test"):
                per_seed = {}
                for sampling_seed in SAMPLING_SEEDS:
                    observed = _rf_data(config, sampling_seed)
                    scored = _score_split(estimators, observed, penetration, observed.days[split],
                                          rate_dir / "predictions" / f"seed_{sampling_seed}" / split, **score_kwargs)
                    flat = dict(scored["overall"])
                    for state, metrics in scored.get("states", {}).items():
                        flat.update({f"{state}_{key}": value for key, value in metrics.items() if isinstance(value, (int, float))})
                    if scored.get("states"):
                        state_samples[split].append(scored["states"])
                    per_seed[str(sampling_seed)] = flat
                    if sampling_seed == 42:
                        seed42[split] = scored
                reports[split] = aggregate_sampling_metrics(per_seed)
                _write_payload(rate_dir / f"{split}_sampling_metrics.json", reports[split])
            validation, test = seed42["validation"], seed42["test"]
            validation["overall"].update(reports["validation"]["mean"])
            test["overall"].update(reports["test"]["mean"])
            for row in validation["daily"]:
                row["sampling_seed"] = 42
            for row in test["daily"]:
                row["sampling_seed"] = 42
            active_report = _active_report(rate_dir, data.days["test"], multi)
            if isinstance(active_report, dict):
                active_report = {**active_report, "sampling_seed": 42}
            metrics_dir = rate_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            _write_daily(metrics_dir / "validation_daily.csv", validation["daily"])
            _write_daily(metrics_dir / "test_daily.csv", test["daily"])
            validation_payload = dict(validation["overall"])
            validation_payload["daily_sampling_seed"] = 42
            if "states" in validation:
                validation_payload["states_sampling_seed_42"] = validation["states"]
                state_mean, state_std = _aggregate_state_reports(state_samples["validation"])
                validation_payload["states"] = state_mean
                validation_payload["states_std"] = state_std
            (metrics_dir / "validation_metrics.json").write_text(
                json.dumps(validation_payload, indent=2) + "\n",
                encoding="utf-8",
            )
            test_payload = {
                "overall": test["overall"],
                "active_edges_train_defined": active_report,
                "daily_sampling_seed": 42,
            }
            if "states" in test:
                test_payload["states_sampling_seed_42"] = test["states"]
                state_mean, state_std = _aggregate_state_reports(state_samples["test"])
                test_payload["states"] = state_mean
                test_payload["states_std"] = state_std
            (metrics_dir / "test_metrics.json").write_text(
                json.dumps(test_payload, indent=2) + "\n",
                encoding="utf-8",
            )
            reloaded = [
                (state, RFEstimator.load(rate_dir / "model" / state if multi else rate_dir / "model"))
                for state in heads
            ]
            _confirm_reload(
                reloaded,
                data,
                penetration,
                data.days["test"][0],
                rate_dir,
                input_type,
                target_type,
            )
            import torch
            checkpoint = {"model_name": "RF", "model_seed": int(config["seed"]),
                          "train_sampling_seed": 42, "evaluation_sampling_seeds": list(SAMPLING_SEEDS),
                          "model_paths": [str(path) for _, path in _head_dirs(rate_dir, multi, heads)],
                          "checkpoint_selection": "single_fit_no_epoch_selection"}
            torch.save(checkpoint, rate_dir / "best_checkpoint.pt")
            torch.save(checkpoint, rate_dir / "last_checkpoint.pt")
            _write_payload(rate_dir / "training_history.json", [{"fit": 0, "mean_validation_mae": reports["validation"]["mean"]["mae"]}])
            _write_payload(rate_dir / "run_metadata.json", {**checkpoint, "status": "completed",
                           "speed_features": "per dataset, observation key, and sampling seed; not shared across seeds",
                           "scaler_fit": "seed42 training only", "optimizer": "RandomForest/KMeans/LAD"})
            elapsed = time.perf_counter() - started
            log(f"PASS seconds={elapsed:.1f} test_mae={test['overall']['mae']:.6f} rss_mib={_rss_mib()}")
            _write_status(rate_dir, "PASS", elapsed, "")
        except Exception:
            detail = traceback.format_exc()
            log("FAIL\n" + detail)
            _write_status(rate_dir, "FAIL", time.perf_counter() - started, detail)
            raise


def _write_daily(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_status(rate_dir: Path, status: str, seconds: float, error: str) -> None:
    payload = {"status": status, "seconds": seconds, "error": error[-4000:]}
    (rate_dir / "status.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def default_estimator_settings() -> dict:
    return {
        "n_clusters": N_CLUSTERS,
        "n_estimators": N_ESTIMATORS,
        "kmeans_n_init": 10,
    }
