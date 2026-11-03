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
        "/data5/tzhang174/TrafficFlow/xuancheng26/training_results/RF",
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
    if RESULTS_ROOT.resolve() not in resolved.parents and resolved != RESULTS_ROOT.resolve():
        raise RunError(f"refusing to write outside {RESULTS_ROOT}: {resolved}")
    if CODE_ROOT.resolve() == resolved or CODE_ROOT.resolve() in resolved.parents:
        raise RunError(f"refusing to write into the git tree: {resolved}")


def _model_files(model_dir: Path) -> list[Path]:
    return [
        model_dir / "scaler.joblib",
        model_dir / "kmeans.joblib",
        model_dir / "random_forest.joblib",
        model_dir / "regression_coefficients.npz",
    ]


def _complete(rate_dir: Path, multi: bool) -> bool:
    model_dirs = [rate_dir / "model" / state for state in STATES] if multi else [rate_dir / "model"]
    required = [path for model_dir in model_dirs for path in _model_files(model_dir)]
    required.extend(
        [
            rate_dir / "metrics" / "test_metrics.json",
            rate_dir / "metrics" / "validation_metrics.json",
        ]
    )
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


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
        with np.load(rate_dir / "predictions" / f"{day_name}.npz") as payload:
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
    with np.load(rate_dir / "predictions" / f"{sample_day}.npz") as payload:
        saved = payload["prediction"]
    if not np.allclose(recomputed, saved, atol=1e-4, rtol=1e-4):
        raise RunError("reloaded model does not reproduce the saved test prediction")


def train_penetration(config: dict) -> None:
    penetration = str(config["penetration"])
    input_type, target_type, multi, heads = _task_types(config)
    rate_dir = Path(config["rate_dir"]) if config.get("rate_dir") else Path(config["run_dir"]) / penetration
    _assert_output(rate_dir)
    rate_dir.mkdir(parents=True, exist_ok=True)
    log_path = rate_dir / "logs" / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as handle:
        def log(message: str) -> None:
            _log(handle, f"[{penetration} {input_type}->{target_type}] {message}")

        try:
            log(f"start pid={os.getpid()} ppid={os.getppid()} n_jobs={config['n_jobs']}")
            if _complete(rate_dir, multi) and not config["force"] and not config["evaluate_only"]:
                log("SKIP existing model and test metrics")
                _write_status(rate_dir, "SKIP", time.perf_counter() - started, "")
                return
            data = XuanchengRFData()
            if config["evaluate_only"]:
                if not _complete(rate_dir, multi):
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
            log("validation")
            validation = _score_split(
                estimators, data, penetration, data.days["validation"], None, **score_kwargs
            )
            log(
                f"val mae={validation['overall']['mae']:.6f} "
                f"rmse={validation['overall']['rmse']:.6f} bias={validation['overall']['bias']:.6f}"
            )
            log("test")
            test = _score_split(
                estimators,
                data,
                penetration,
                data.days["test"],
                rate_dir / "predictions",
                **score_kwargs,
            )
            active_report = _active_report(rate_dir, data.days["test"], multi)
            metrics_dir = rate_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            _write_daily(metrics_dir / "validation_daily.csv", validation["daily"])
            _write_daily(metrics_dir / "test_daily.csv", test["daily"])
            validation_payload = dict(validation["overall"])
            if "states" in validation:
                validation_payload["states"] = validation["states"]
            (metrics_dir / "validation_metrics.json").write_text(
                json.dumps(validation_payload, indent=2) + "\n",
                encoding="utf-8",
            )
            test_payload = {"overall": test["overall"], "active_edges_train_defined": active_report}
            if "states" in test:
                test_payload["states"] = test["states"]
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
