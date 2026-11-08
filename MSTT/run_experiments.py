"""Schedule MSTT runs over tasks x penetrations. Nothing starts without an explicit selection.

    python MSTT/run_experiments.py --all_tasks --all_penetrations --dry_run
    python MSTT/run_experiments.py --tasks I2I All2All --penetrations p05 p70 --gpus 5 6
    python MSTT/run_experiments.py --summarize

Each job is a ``train.py`` subprocess pinned with ``CUDA_VISIBLE_DEVICES``.
``--gpus`` gives one worker per listed GPU (repeat an id to run two jobs on
it); without it jobs run one at a time on ``--device`` (default ``auto``).
Completed runs are skipped by ``train.py`` unless ``--overwrite`` is passed
through. Logs: ``<output_root>/_logs/<rate>_<task>.log``.
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MSTT.config import RATES, TASKS, load_config, rate_tag  # noqa: E402

TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"


def parse_args(argv=None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Batch-run MSTT experiments.", allow_abbrev=False)
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=None)
    parser.add_argument("--all_tasks", "--all-tasks", dest="all_tasks", action="store_true")
    parser.add_argument("--penetrations", "--rates", dest="penetrations", nargs="+", default=None)
    parser.add_argument("--all_penetrations", "--all-penetrations", dest="all_penetrations", action="store_true")
    parser.add_argument("--gpus", nargs="+", type=int, default=None)
    parser.add_argument("--device", default="auto", help="used when --gpus is not given")
    parser.add_argument("--order", choices=("rate_major", "task_major"), default="rate_major")
    parser.add_argument("--dry_run", "--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--summarize", action="store_true", help="rebuild overall_summary from finished runs")
    parser.add_argument("--output_root", "--output-root", dest="output_root", default=None)
    args, passthrough = parser.parse_known_args(argv)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return args, passthrough


def job_list(args: argparse.Namespace) -> list[tuple[str, str]]:
    tasks = list(TASKS) if args.all_tasks else list(args.tasks or [])
    rates = list(RATES) if args.all_penetrations else [rate_tag(item) for item in (args.penetrations or [])]
    if not tasks or not rates:
        raise SystemExit("select tasks (--tasks / --all_tasks) and penetrations (--penetrations / --all_penetrations)")
    if args.order == "rate_major":
        return [(task, rate) for rate in rates for task in tasks]
    return [(task, rate) for task in tasks for rate in rates]


def summarize(output_root: Path) -> None:
    from MSTT.trainer import merge_summary, run_is_complete, summary_from_disk

    rows = []
    for meta in sorted(output_root.glob("in_*__out_*/p*/seed_*/run_metadata.json")):
        if run_is_complete(meta.parent):
            rows.append(summary_from_disk(meta.parent))
    merged = merge_summary(output_root, rows) if rows else []
    print(f"{len(merged)} rows -> {output_root / 'overall_summary.csv'}")


def run_job(task: str, rate: str, gpu: int | None, device: str, passthrough: list[str], log_dir: Path) -> int:
    command = [sys.executable, str(TRAIN_SCRIPT), "--task", task, "--penetration", rate, *passthrough]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command += ["--device", "cuda:0"]
    else:
        command += ["--device", device]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{rate}_{task}.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] gpu={gpu} cmd={' '.join(command)}\n")
        handle.flush()
        status = subprocess.run(command, cwd=str(PROJECT_ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
        handle.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] exit={status}\n")
    print(f"[run_experiments] {task} {rate} gpu={gpu} exit={status} log={log_path}", flush=True)
    return status


def main(argv=None) -> int:
    args, passthrough = parse_args(argv)
    output_root = Path(args.output_root or load_config(None)["output_root"])
    if args.output_root:
        passthrough += ["--output-root", str(output_root)]
    if args.summarize:
        summarize(output_root)
        return 0
    jobs = job_list(args)
    print(f"[run_experiments] {len(jobs)} jobs, workers={args.gpus or [args.device]}, extra={passthrough}")
    for task, rate in jobs:
        print(f"  {rate} {task}")
    if args.dry_run:
        return 0
    log_dir = output_root / "_logs"
    pending: "queue.Queue[tuple[str, str]]" = queue.Queue()
    for job in jobs:
        pending.put(job)
    failures: list[tuple[str, str, int]] = []
    lock = threading.Lock()

    def worker(gpu: int | None) -> None:
        while True:
            try:
                task, rate = pending.get_nowait()
            except queue.Empty:
                return
            status = run_job(task, rate, gpu, args.device, passthrough, log_dir)
            if status != 0:
                with lock:
                    failures.append((task, rate, status))

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in (args.gpus or [None])]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    summarize(output_root)
    if failures:
        print(f"[run_experiments] {len(failures)} failed: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
