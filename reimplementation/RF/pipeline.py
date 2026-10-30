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
    FEATURE_NAMES,
    N_TARGETS,
    XuanchengRFData,
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


def _complete(rate_dir: Path) -> bool:
    required = [
        rate_dir / "model" / "scaler.joblib",
        rate_dir / "model" / "kmeans.joblib",
        rate_dir / "model" / "random_forest.joblib",
        rate_dir / "model" / "regression_coefficients.npz",
        rate_dir / "metrics" / "test_metrics.json",
        rate_dir / "metrics" / "validation_metrics.json",
    ]
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def _score_split(estimator: RFEstimator, data: XuanchengRFData, penetration: str, days: list[str], prediction_dir: Path | None) -> dict:
    predictions = []
    targets = []
    daily = []
    for day_name in days:
        cube, target = data.day_cube(penetration, day_name)
        flat = cube.reshape(-1, cube.shape[-1])
        predicted = estimator.predict(flat).reshape(N_TARGETS, cube.shape[1])
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
            np.savez_compressed(
                prediction_dir / f"{day_name}.npz",
                prediction=predicted.astype(np.float32),
                ground_truth=target.astype(np.float32),
                edge_ids=np.asarray(data.edges, dtype="U32"),
                time_bins=np.arange(11, 11 + N_TARGETS, dtype=np.int16),
                date=np.asarray(day_name),
                penetration=np.asarray(penetration),
                supervised_bin_start=np.int16(11),
                supervised_bin_end_inclusive=np.int16(287),
            )
        predictions.append(predicted.reshape(-1))
        targets.append(target.reshape(-1))
    overall = regression_report(np.concatenate(predictions), np.concatenate(targets))
    return {"overall": overall, "daily": daily}


def _active_mask(target: np.ndarray, n_days: int, n_edges: int) -> np.ndarray:
    summed = target.reshape(n_days, N_TARGETS, n_edges).sum(axis=(0, 1))
    return summed > 0


def train_penetration(config: dict) -> None:
    penetration = str(config["penetration"])
    rate_dir = Path(config["run_dir"]) / penetration
    _assert_output(rate_dir)
    rate_dir.mkdir(parents=True, exist_ok=True)
    log_path = rate_dir / "logs" / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as handle:
        def log(message: str) -> None:
            _log(handle, f"[{penetration}] {message}")

        try:
            log(f"start pid={os.getpid()} ppid={os.getppid()} n_jobs={config['n_jobs']}")
            if _complete(rate_dir) and not config["force"] and not config["evaluate_only"]:
                log("SKIP existing model and test metrics")
                _write_status(rate_dir, "SKIP", time.perf_counter() - started, "")
                return
            data = XuanchengRFData()
            model_dir = rate_dir / "model"
            if config["evaluate_only"]:
                if not _complete(rate_dir) and not (model_dir / "random_forest.joblib").is_file():
                    raise RunError("evaluate-only found no saved model")
                estimator = RFEstimator.load(model_dir)
                log("loaded saved model")
            else:
                load_started = time.perf_counter()
                features, target = data.stack(penetration, data.days["train"])
                log(
                    f"train matrix {features.shape} dtype={features.dtype} "
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
                estimator.fit(features, target)
                fit_seconds = time.perf_counter() - fit_started
                active = _active_mask(target, len(data.days["train"]), len(data.edges))
                np.save(rate_dir / "active_edge_mask.npy", active)
                estimator.save(model_dir)
                counts = estimator.cluster_samples
                log(
                    f"fit_seconds={fit_seconds:.1f} kmeans_iter={estimator.kmeans_iterations} "
                    f"inertia={estimator.kmeans_inertia:.6g} "
                    f"cluster_min={int(counts.min())} cluster_max={int(counts.max())} "
                    f"empty={int(np.count_nonzero(counts == 0))} rss_mib={_rss_mib()}"
                )
                for cluster_id, status in enumerate(estimator.cluster_status):
                    if status not in {"converged", "duplicate_flow", "expanded_bounds"}:
                        log(f"cluster {cluster_id} status={status} samples={int(counts[cluster_id])}")
                del features, target
                config_payload = {
                    "penetration": penetration,
                    "feature_names": list(FEATURE_NAMES),
                    "seed": int(config["seed"]),
                    "n_clusters": int(config["n_clusters"]),
                    "n_estimators": int(config["n_estimators"]),
                    "kmeans_n_init": int(config["kmeans_n_init"]),
                    "fit_seconds": fit_seconds,
                    "train_days": data.days["train"],
                    "adaptations": [
                        "K-Means uses the same seven features as the forest; the paper does not specify this.",
                        "StandardScaler is fit on the training penetration only.",
                        "Functional class is replaced by num_lanes.",
                        "Capacity is replaced by the SUMO speed limit in km/h.",
                        "Supervised bins are 11..287.",
                        "The target is the current 5-minute full entry count.",
                    ],
                }
                (rate_dir / "config.json").write_text(
                    json.dumps(config_payload, indent=2) + "\n",
                    encoding="utf-8",
                )
            log("validation")
            validation = _score_split(estimator, data, penetration, data.days["validation"], None)
            log(
                f"val mae={validation['overall']['mae']:.6f} "
                f"rmse={validation['overall']['rmse']:.6f} bias={validation['overall']['bias']:.6f}"
            )
            log("test")
            test = _score_split(
                estimator,
                data,
                penetration,
                data.days["test"],
                rate_dir / "predictions",
            )
            active_path = rate_dir / "active_edge_mask.npy"
            active_report = None
            if active_path.is_file():
                mask = np.load(active_path)
                pieces_p = []
                pieces_t = []
                for day_name in data.days["test"]:
                    with np.load(rate_dir / "predictions" / f"{day_name}.npz") as payload:
                        pieces_p.append(payload["prediction"][:, mask])
                        pieces_t.append(payload["ground_truth"][:, mask])
                active_report = regression_report(np.concatenate(pieces_p), np.concatenate(pieces_t))
            metrics_dir = rate_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            _write_daily(metrics_dir / "validation_daily.csv", validation["daily"])
            _write_daily(metrics_dir / "test_daily.csv", test["daily"])
            (metrics_dir / "validation_metrics.json").write_text(
                json.dumps(validation["overall"], indent=2) + "\n",
                encoding="utf-8",
            )
            test_payload = {"overall": test["overall"], "active_edges_train_defined": active_report}
            (metrics_dir / "test_metrics.json").write_text(
                json.dumps(test_payload, indent=2) + "\n",
                encoding="utf-8",
            )
            # Reload and confirm one saved test day reproduces.
            restored = RFEstimator.load(model_dir)
            sample_day = data.days["test"][0]
            cube, _target = data.day_cube(penetration, sample_day)
            recomputed = restored.predict(cube.reshape(-1, cube.shape[-1])).reshape(cube.shape[0], cube.shape[1])
            with np.load(rate_dir / "predictions" / f"{sample_day}.npz") as payload:
                saved = payload["prediction"]
            if not np.allclose(recomputed, saved, atol=1e-4, rtol=1e-4):
                raise RunError("reloaded model does not reproduce the saved test prediction")
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
