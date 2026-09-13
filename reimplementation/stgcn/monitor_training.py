"""Read-only training monitor. Never starts, kills, or repairs a training run."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor an R-only STGCN training directory.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def _pid_alive(pid: int | None) -> str:
    if pid is None:
        return "unknown"
    try:
        os.kill(pid, 0)
        return "running"
    except OSError:
        return "not_running"


def _read_jsonl_tail(path: Path, limit: int = 20) -> list[dict]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _checkpoint_mtime(root: Path) -> str | None:
    times = []
    for path in root.rglob("best_checkpoint.pt"):
        times.append(path.stat().st_mtime)
    for path in root.rglob("last_checkpoint.pt"):
        times.append(path.stat().st_mtime)
    if not times:
        return None
    return datetime.fromtimestamp(max(times), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _completed_rates(root: Path) -> list[str]:
    found = []
    for path in sorted(root.glob("p*/seed_*/best_checkpoint.pt")):
        found.append(path.parent.parent.name)
    return found


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _eta_seconds(output_root: Path, last_train: dict, completed_rates: list[str]) -> float | None:
    config = _load_json(output_root / "resolved_config.json")
    rates = [str(item) for item in config.get("rates") or []]
    epochs = int(config.get("epochs") or 0)
    epoch = last_train.get("epoch")
    elapsed_epoch = last_train.get("elapsed_seconds")
    current_rate = str(last_train.get("penetration_rate") or "")
    if epoch is None or elapsed_epoch is None or epochs <= 0:
        return None
    remaining_this = max(epochs - int(epoch) - 1, 0)
    done = set(str(item) for item in completed_rates)
    if current_rate:
        done.add(current_rate)
    remaining_rates = len([rate for rate in rates if rate not in done]) if rates else 0
    return float(elapsed_epoch) * (remaining_this + remaining_rates * epochs)


def report(output_root: Path, pid: int | None, started: float) -> dict:
    log_path = output_root / "training_log.jsonl"
    records = _read_jsonl_tail(log_path, 50)
    latest = records[-1] if records else {}
    train_records = [item for item in records if item.get("stage") == "training"]
    last_train = train_records[-1] if train_records else {}
    elapsed = time.time() - started
    epoch = last_train.get("epoch")
    status = latest.get("status", "unknown")
    error = latest.get("latest_error")
    stalled = False
    if log_path.is_file():
        age = time.time() - log_path.stat().st_mtime
        stalled = age > 600 and _pid_alive(pid) == "running"
    completed = _completed_rates(output_root)
    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": elapsed,
        "process": _pid_alive(pid),
        "pid": pid,
        "current_rate": last_train.get("penetration_rate") or latest.get("penetration_rate"),
        "current_epoch": epoch,
        "latest_train_loss": last_train.get("total_loss") or last_train.get("prediction_loss"),
        "latest_validation_mae_raw": last_train.get("validation_mae_raw"),
        "best_epoch": last_train.get("best_epoch"),
        "best_metric": last_train.get("best_metric"),
        "completed_rates": completed,
        "checkpoint_updated_at": _checkpoint_mtime(output_root),
        "gpu_memory_allocated": last_train.get("gpu_memory_allocated"),
        "status": status,
        "latest_error": error,
        "stalled": stalled,
        "latest_stage": latest.get("stage"),
        "estimated_remaining_seconds": _eta_seconds(output_root, last_train, completed),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    while True:
        payload = report(args.output_root, args.pid, started)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        if args.once:
            return
        time.sleep(max(args.interval, 1.0))


if __name__ == "__main__":
    main()
