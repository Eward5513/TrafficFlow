"""Read-only progress monitor for count_observed_edge_flow.py.

Does not write observed CSVs, lock files, kill the worker, or start another
count. Prints one status block per interval until the worker PID exits.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def tail_text(path: Path, n_lines: int = 8) -> str:
    if not path.is_file():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n_lines:])


def count_csvs(root: Path, subdir: str) -> int:
    total = 0
    if not root.is_dir():
        return 0
    for day_dir in root.iterdir():
        folder = day_dir / subdir
        if folder.is_dir():
            total += sum(1 for path in folder.glob("edge_flow_5min_p*.csv") if path.is_file())
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only monitor for observed edge-flow counting.")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-subdir", default="observed_edge_flow_5min")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--days-total", type=int, default=20)
    parser.add_argument("--rates", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    last_log_mtime = None
    last_csv_count = -1
    stall_rounds = 0
    while True:
        alive = pid_alive(args.pid)
        status = read_json(args.status_file)
        csv_count = count_csvs(args.data_root, args.output_subdir)
        log_path = args.log
        log_mtime = log_path.stat().st_mtime if log_path.is_file() else None
        elapsed = int(time.time() - started)
        days_done = int(status.get("days_done") or 0)
        days_total = int(status.get("days_total") or args.days_total)
        remaining_days = max(days_total - days_done, 0)
        per_day = (elapsed / days_done) if days_done else None
        eta = int(per_day * remaining_days) if per_day else None
        stalled = csv_count == last_csv_count and log_mtime == last_log_mtime
        if stalled:
            stall_rounds += 1
        else:
            stall_rounds = 0
        last_csv_count = csv_count
        last_log_mtime = log_mtime
        print(
            "\n".join(
                [
                    f"=== monitor {utc_now()} ===",
                    f"elapsed_s={elapsed}",
                    f"worker_pid={args.pid} alive={alive}",
                    f"current_day={status.get('current_day')}",
                    f"current_stage={status.get('current_stage')}",
                    f"current_rate={status.get('current_rate')}",
                    f"days_done={days_done}/{days_total}",
                    f"csv_written_status={status.get('csv_written')} csv_on_disk={csv_count}",
                    f"log_mtime={log_mtime}",
                    f"stalled_rounds={stall_rounds}",
                    f"eta_s={eta}",
                    "latest_log:",
                    tail_text(log_path) or "(no log)",
                    "",
                ]
            ),
            flush=True,
        )
        if not alive:
            print("worker exited; monitor stopping", flush=True)
            return
        time.sleep(max(args.interval, 1))


if __name__ == "__main__":
    main()
