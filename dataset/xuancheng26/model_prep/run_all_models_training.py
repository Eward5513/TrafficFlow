#!/usr/bin/env python3
"""Schedule Xuancheng26 formal training on GPUs 0, 3, and 4.

One model runs all seven penetration rates on one GPU. A finished rate is
skipped only when its SUCCESS marker exists. That is not checkpoint resume.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

CODE = Path("/home/tzhang174/TrafficFlow")
DATA = Path("/data5/tzhang174/TrafficFlow/xuancheng26/training_data_3s")
RESULTS = Path("/data5/tzhang174/TrafficFlow/xuancheng26/training_results_3s")
PYTHON = Path("/home/tzhang174/miniforge3/envs/trafficflow/bin/python3")
LAUNCHER_LOG = RESULTS / "training_launcher.log"
GPUS = (0, 3, 4)
RATES = (5, 10, 20, 30, 40, 50, 70)
MIN_FREE_BYTES = 80 * 1024**3

JOBS = (
    {
        "name": "pdformer",
        "script": "reimplementation/pdformer/train_r_only_pdformer.py",
        "config": DATA / "config/pdformer/r_only_pdformer.json",
        "result": "pdformer",
    },
    {
        "name": "staeformer",
        "script": "reimplementation/staeformer/train_r_only_staeformer.py",
        "config": DATA / "config/staeformer/r_only_staeformer.json",
        "result": "staeformer",
    },
    {
        "name": "stfgnn",
        "script": "reimplementation/stfgnn/train_r_only_stfgnn.py",
        "config": DATA / "config/stfgnn/r_only_stfgnn.json",
        "result": "stfgnn",
    },
    {
        "name": "stsgcn",
        "script": "reimplementation/stsgcn/train_r_only_stsgcn.py",
        "config": DATA / "config/stsgcn/r_only_stsgcn.json",
        "result": "stsgcn",
    },
    {
        "name": "dcrnn",
        "script": "reimplementation/dcrnn/train_r_only_dcrnn.py",
        "config": DATA / "config/dcrnn/r_only_dcrnn.json",
        "result": "dcrnn",
    },
    {
        "name": "graph_wavenet",
        "script": "reimplementation/graph_wavenet/train_r_only_gwn.py",
        "config": DATA / "config/gwn/r_only_gwn.json",
        "result": "graph_wavenet",
    },
    {
        "name": "stgcn",
        "script": "reimplementation/stgcn/train_r_only_stgcn.py",
        "config": DATA / "config/stgcn/r_only_stgcn.json",
        "result": "stgcn",
    },
    {
        "name": "stid",
        "script": "reimplementation/stid/train_r_only_stid.py",
        "config": DATA / "config/stid/r_only_stid.json",
        "result": "stid",
    },
)

LOG_LOCK = threading.Lock()
STATUS_LOCK = threading.Lock()
STATUS: dict[int, dict[str, str]] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_line(message: str) -> None:
    line = f"{utc_now()} {message}\n"
    with LOG_LOCK:
        LAUNCHER_LOG.parent.mkdir(parents=True, exist_ok=True)
        with LAUNCHER_LOG.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
    print(line, end="", flush=True)


def disk_free_bytes() -> int:
    return os.statvfs("/data5").f_bavail * os.statvfs("/data5").f_frsize


def rate_tag(rate: int) -> str:
    return f"p{int(rate):02d}"


def gpu_is_free(gpu: int) -> bool:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    uuid = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=uuid", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    for row in completed.stdout.splitlines():
        if row.strip().startswith(uuid):
            return False
    return True


def verify_rate(rate_dir: Path) -> dict[str, object]:
    best = rate_dir / "best_checkpoint.pt"
    metrics_path = rate_dir / "test_metrics.json"
    predictions = rate_dir / "test_predictions.npz"
    missing = [path.name for path in (best, metrics_path, predictions) if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise RuntimeError(f"missing outputs: {missing}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in ("mae", "rmse", "mape_nonzero"):
        value = float(metrics[key])
        if value != value or value in (float("inf"), float("-inf")):
            raise RuntimeError(f"non-finite {key}")
    import numpy as np

    with np.load(predictions, allow_pickle=False) as payload:
        pred = payload["pred_raw"]
        true = payload["y_raw"]
    if pred.shape != true.shape or pred.shape[0] != 1385 or pred.shape[-2:] != (1406, 1):
        raise RuntimeError(f"prediction shape {tuple(pred.shape)}")
    history = json.loads((rate_dir / "training_history.json").read_text(encoding="utf-8"))
    best_epoch = None
    if isinstance(history, list) and history:
        key = "val_huber" if "val_huber" in history[0] else "val_mae" if "val_mae" in history[0] else None
        if key is not None:
            best_epoch = min(history, key=lambda row: float(row[key])).get("epoch")
    return {"mae": float(metrics["mae"]), "rmse": float(metrics["rmse"]), "mape_nonzero": float(metrics["mape_nonzero"]), "best_epoch": best_epoch}


def run_rate(gpu: int, job: dict, rate: int) -> int:
    tag = rate_tag(rate)
    result_dir = RESULTS / job["result"]
    rate_log_dir = result_dir / tag
    rate_log_dir.mkdir(parents=True, exist_ok=True)
    success = rate_log_dir / "SUCCESS"
    if success.is_file():
        log_line(f"GPU {gpu} {job['name']} {tag} SKIP existing SUCCESS")
        return 0
    config = json.loads(Path(job["config"]).read_text(encoding="utf-8"))
    log_path = rate_log_dir / "train.log"
    header = (
        f"start_time={utc_now()}\n"
        f"physical_gpu={gpu}\n"
        f"visible_device=cuda:0\n"
        f"model={job['name']}\n"
        f"penetration={tag}\n"
        f"config={job['config']}\n"
        f"batch_size={config.get('batch_size')}\n"
        f"seed={config.get('seed')}\n"
    )
    log_line(f"GPU {gpu} {job['name']} {tag} START batch={config.get('batch_size')} log={log_path}")
    command = [
        str(PYTHON),
        str(CODE / job["script"]),
        "--config",
        str(job["config"]),
        "--rates",
        str(rate),
        "--device",
        "cuda:0",
        "--overwrite",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(CODE)
    env["PYTHONUNBUFFERED"] = "1"
    with STATUS_LOCK:
        STATUS[gpu] = {
            "model": job["name"],
            "penetration": tag,
            "batch_size": str(config.get("batch_size")),
            "log": str(log_path),
            "state": "running",
        }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(header)
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=CODE,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        handle.write(f"\nfinish_time={utc_now()}\nexit_code={completed.returncode}\n")
        handle.flush()
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        log_line(
            f"GPU {gpu} {job['name']} {tag} FAILED exit={completed.returncode} "
            f"tail={' | '.join(tail[-5:])}"
        )
        return completed.returncode
    rate_dir = result_dir / tag / "seed_42"
    try:
        checked = verify_rate(rate_dir)
    except Exception as exc:
        log_line(f"GPU {gpu} {job['name']} {tag} FAILED output_check {exc}")
        return 1
    success.write_text(
        json.dumps({"finish_time": utc_now(), "best_epoch": checked["best_epoch"], "test_mae": checked["mae"], "test_rmse": checked["rmse"], "test_mape_nonzero": checked["mape_nonzero"]}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    log_line(
        f"GPU {gpu} {job['name']} {tag} SUCCESS best_epoch={checked['best_epoch']} "
        f"test_mae={checked['mae']:.6f} free_gib={disk_free_bytes() / 1024**3:.1f}"
    )
    return 0


def run_model(gpu: int, job: dict) -> None:
    log_line(f"GPU {gpu} {job['name']} MODEL_START free_gib={disk_free_bytes() / 1024**3:.1f}")
    if disk_free_bytes() < MIN_FREE_BYTES:
        log_line(f"GPU {gpu} {job['name']} FAILED low_disk")
        return
    for rate in RATES:
        code = run_rate(gpu, job, rate)
        if code != 0:
            log_line(f"GPU {gpu} {job['name']} MODEL_FAILED at {rate_tag(rate)} exit={code}")
            with STATUS_LOCK:
                STATUS[gpu] = {"model": job["name"], "penetration": rate_tag(rate), "state": "failed"}
            return
        if disk_free_bytes() < MIN_FREE_BYTES:
            log_line(f"GPU {gpu} {job['name']} MODEL_FAILED low_disk after {rate_tag(rate)}")
            return
    log_line(f"GPU {gpu} {job['name']} MODEL_SUCCESS")
    with STATUS_LOCK:
        STATUS[gpu] = {"model": "", "penetration": "", "state": "idle"}


def worker(gpu: int, jobs: queue.Queue) -> None:
    while True:
        job = jobs.get()
        try:
            if job is None:
                return
            if not gpu_is_free(gpu):
                log_line(f"GPU {gpu} occupied by another process; return {job['name']} to the queue")
                jobs.put(job)
                time.sleep(30)
                continue
            run_model(gpu, job)
        finally:
            jobs.task_done()


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    free = [gpu for gpu in GPUS if gpu_is_free(gpu)]
    busy = [gpu for gpu in GPUS if gpu not in free]
    log_line(
        f"LAUNCHER_START pid={os.getpid()} free_gpus={free} busy_gpus={busy} "
        f"disk_free_gib={disk_free_bytes() / 1024**3:.1f}"
    )
    if not free:
        log_line("LAUNCHER_STOP no allowed GPU is free")
        return 1
    jobs: queue.Queue = queue.Queue()
    for job in JOBS:
        jobs.put(job)
    threads = []
    for gpu in free:
        thread = threading.Thread(target=worker, args=(gpu, jobs), name=f"gpu-{gpu}", daemon=False)
        thread.start()
        threads.append(thread)
    jobs.join()
    for _ in threads:
        jobs.put(None)
    for thread in threads:
        thread.join()
    log_line("LAUNCHER_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
