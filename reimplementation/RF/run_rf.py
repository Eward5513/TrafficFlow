#!/usr/bin/env python3
"""Train the seven Xuancheng26 RF penetrations in detached worker processes."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

CODE_ROOT = Path("/home/tzhang174/TrafficFlow")
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

RESULTS_ROOT = Path("/data5/tzhang174/TrafficFlow/xuancheng26/training_results/RF")
DEFAULT_RATES = ["p05", "p10", "p20", "p30", "p40", "p50", "p70"]


def _log(message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"{stamp} {message}", flush=True)


def _parse_rates(values: list[str] | None) -> list[str]:
    if not values:
        return list(DEFAULT_RATES)
    rates: list[str] = []
    for value in values:
        for part in value.split(","):
            name = part.strip()
            if name:
                rates.append(name)
    unknown = [name for name in rates if name not in DEFAULT_RATES]
    if unknown or len(rates) != len(set(rates)):
        raise SystemExit(f"penetrations must be a subset of {DEFAULT_RATES}, got {rates}")
    return rates


def _run_dir(run_id: str | None) -> Path:
    now = datetime.now().astimezone()
    folder = RESULTS_ROOT / now.strftime("%Y_%m_%d")
    name = run_id or now.strftime("%H%M%S")
    path = folder / name
    resolved = path.resolve()
    if RESULTS_ROOT.resolve() not in resolved.parents:
        raise SystemExit(f"run directory escapes {RESULTS_ROOT}: {resolved}")
    if CODE_ROOT.resolve() in resolved.parents or resolved == CODE_ROOT.resolve():
        raise SystemExit(f"run directory is inside the git tree: {resolved}")
    return path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _summarize(run_dir: Path, rates: list[str], elapsed: float, workers: int) -> None:
    rows = []
    for rate in rates:
        metrics_path = run_dir / rate / "metrics" / "test_metrics.json"
        validation_path = run_dir / rate / "metrics" / "validation_metrics.json"
        status_path = run_dir / rate / "status.json"
        config_path = run_dir / rate / "config.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {"status": "MISSING", "seconds": None}
        if not metrics_path.is_file() or not validation_path.is_file():
            rows.append(
                {
                    "model": "RF",
                    "penetration": rate,
                    "status": status.get("status", "MISSING"),
                    "val_mae": None,
                    "test_mae": None,
                    "test_rmse": None,
                    "test_bias": None,
                    "train_seconds": status.get("seconds"),
                }
            )
            continue
        test_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["overall"]
        validation_metrics = json.loads(validation_path.read_text(encoding="utf-8"))
        fit_seconds = None
        if config_path.is_file():
            fit_seconds = json.loads(config_path.read_text(encoding="utf-8")).get("fit_seconds")
        rows.append(
            {
                "model": "RF",
                "penetration": rate,
                "status": status.get("status"),
                "val_mae": validation_metrics["mae"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_bias": test_metrics["bias"],
                "train_seconds": fit_seconds if fit_seconds is not None else status.get("seconds"),
                "negative_prediction_fraction": test_metrics["negative_prediction_fraction"],
                "test_samples": test_metrics["sample_count"],
            }
        )
    summary = {
        "model": "RF",
        "elapsed_seconds": elapsed,
        "parallel_models": workers,
        "rates": rows,
    }
    _write_json(run_dir / "summary.json", summary)
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "model",
            "penetration",
            "status",
            "val_mae",
            "test_mae",
            "test_rmse",
            "test_bias",
            "train_seconds",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _log("summary")
    for row in rows:
        _log(
            f"{row['penetration']} status={row['status']} val_mae={row['val_mae']} "
            f"test_mae={row['test_mae']} test_rmse={row['test_rmse']} test_bias={row['test_bias']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Xuancheng26 RF baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--penetrations", nargs="*", default=None)
    parser.add_argument("--parallel-models", type=int, default=7)
    parser.add_argument("--n-jobs-per-model", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--kmeans-n-init", type=int, default=10)
    parser.add_argument("--n-clusters", type=int, default=50)
    args = parser.parse_args(argv)
    if args.self_test:
        from reimplementation.RF.model import self_test

        self_test()
        return 0
    if args.parallel_models < 1 or args.n_jobs_per_model < 1:
        parser.error("parallelism must be >= 1")
    if args.n_clusters < 2:
        parser.error("--n-clusters must be >= 2")
    rates = _parse_rates(args.penetrations)
    from reimplementation.RF.data_loader import XuanchengRFData

    _log("auditing RF inputs")
    audit = XuanchengRFData().audit()
    _log(f"audit {audit}")
    if args.audit_only:
        return 0

    run_dir = _run_dir(args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    experiment = {
        "model": "RF",
        "paper": "Othman et al., IEEE ITSC 2022",
        "paper_settings": {
            "kmeans_clusters": 50,
            "kmeans_clusters_this_run": args.n_clusters,
            "random_forest_trees": 50,
            "random_forest_criterion": "gini",
            "regression": "cluster-specific univariate absolute-error line on FCD flow",
            "voting": "soft",
        },
        "project_adaptations": [
            "K-Means features are the same seven inputs; the paper does not state this.",
            "StandardScaler is fit on each penetration's training rows.",
            "Functional class is replaced by number of lanes.",
            "Capacity is replaced by the SUMO speed limit.",
            "Base speed is the SUMO speed limit.",
            "Samples are 5-minute directed-edge counts, supervised bins 11-287.",
            "Split is the existing Xuancheng26 20/4/5 manifest.",
            "The target is the current full entry count, not a future bin.",
        ],
        "penetrations": rates,
        "seed": args.seed,
        "parallel_models": min(args.parallel_models, len(rates)),
        "n_jobs_per_model": args.n_jobs_per_model,
        "kmeans_n_init": args.kmeans_n_init,
        "n_clusters": args.n_clusters,
        "force": args.force,
        "evaluate_only": args.evaluate_only,
        "run_dir": str(run_dir),
    }
    _write_json(run_dir / "experiment_config.json", experiment)
    workers = min(args.parallel_models, len(rates))
    _log(
        f"run_dir={run_dir} rates={rates} clusters={args.n_clusters} "
        f"parallel={workers} n_jobs={args.n_jobs_per_model}"
    )
    started = time.perf_counter()
    context = mp.get_context("spawn")
    processes = []
    for rate in rates:
        config = {
            "penetration": rate,
            "run_dir": str(run_dir),
            "seed": args.seed,
            "n_jobs": args.n_jobs_per_model,
            "n_clusters": args.n_clusters,
            "n_estimators": 50,
            "kmeans_n_init": args.kmeans_n_init,
            "force": args.force,
            "evaluate_only": args.evaluate_only,
        }
        config_path = run_dir / f"worker_{rate}.json"
        _write_json(config_path, config)
        process = context.Process(target=_spawn_worker, args=(str(config_path),), name=f"rf-{rate}")
        processes.append(process)
    running = []
    exit_codes = {}
    next_index = 0
    while next_index < len(processes) or running:
        while next_index < len(processes) and len(running) < workers:
            process = processes[next_index]
            process.start()
            running.append((rates[next_index], process))
            next_index += 1
        still = []
        for rate, process in running:
            process.join(timeout=1.0)
            if process.is_alive():
                still.append((rate, process))
            else:
                exit_codes[rate] = process.exitcode
                _log(f"{rate} exit={process.exitcode}")
        running = still
    elapsed = time.perf_counter() - started
    _summarize(run_dir, rates, elapsed, workers)
    failed = [rate for rate, code in exit_codes.items() if code != 0]
    _log(f"done elapsed_seconds={elapsed:.1f} failed={failed}")
    return 1 if failed else 0


def _spawn_worker(config_path: str) -> None:
    from reimplementation.RF.train_worker import main as worker_main

    try:
        worker_main(config_path)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
