"""Queue three single-head variants of the finished single-layer STF run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MSTT.ablation_100ep import (
    ROOT, config_for, exact_match, read, write, run_dir, process_identity, alive, summarize, source_hashes,
)
from MSTT.config import attention_head_count
from MSTT.sftf_experiment import idle_cards

NAMES = (
    "single_ts_ffn_spatial_head1_100ep",
    "single_ts_ffn_temporal_head1_100ep",
    "single_ts_ffn_both_head1_100ep",
)
EXPECTED = {
    "single_ts_ffn_spatial_head1_100ep": (1, 4),
    "single_ts_ffn_temporal_head1_100ep": (4, 1),
    "single_ts_ffn_both_head1_100ep": (1, 1),
}
STATE = ROOT / "head1_queue_state.json"
LOCK = ROOT / ".head1_supervisor.lock"
POLL_SECONDS = 30
MIN_FREE_MB = 16000


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f"[{timestamp()}] {message}", flush=True)


def experiment_state(name):
    return ROOT / name / "launcher_state.json"


def update_job(name, **changes):
    path = experiment_state(name)
    state = read(path, {})
    state.update(changes, updated_at=timestamp(), experiment=name)
    write(path, state)
    return state


def validate_cpu():
    import torch
    from MSTT.config import task_spec
    from MSTT.model import MSTT
    from MSTT.model.graph import build_spatial_graph
    from MSTT.trainer import model_config

    torch.set_num_threads(2)
    reports = {}
    for name, (spatial_heads, temporal_heads) in EXPECTED.items():
        cfg = config_for(name)
        assert cfg["num_layers"] == 1 and cfg["d_model"] == 64 and cfg["num_heads"] == 4
        assert cfg["use_adaptive_embedding"] is False and cfg["spatial_attention"] == "global"
        assert attention_head_count(cfg, "spatial_num_heads") == spatial_heads
        assert attention_head_count(cfg, "temporal_num_heads") == temporal_heads
        torch.manual_seed(42)
        full = MSTT(model_config(cfg, task_spec("I2I")), build_spatial_graph(torch.zeros(1406, 1406)))
        architecture = full.architecture_summary()
        assert architecture["spatial_num_heads"] == spatial_heads
        assert architecture["temporal_num_heads"] == temporal_heads
        assert architecture["spatial_head_dim"] == 64 // spatial_heads
        assert architecture["temporal_head_dim"] == 64 // temporal_heads
        assert architecture["block_order"] == ["T", "S", "FFN"]
        assert architecture["independent_ffn_count"] == 1
        del full
        small = {**cfg, "num_nodes": 7}
        model = MSTT(model_config(small, task_spec("I2I")), build_spatial_graph(torch.zeros(7, 7)))
        calendar = torch.zeros(2, 12, dtype=torch.long)
        prediction = model(torch.randn(2, 12, 7, 1), calendar, calendar)
        assert list(prediction.shape) == [2, 7, 1]
        assert model.blocks[0].spatial.num_heads == spatial_heads
        assert model.blocks[0].temporal.num_heads == temporal_heads
        prediction.abs().mean().backward()
        reports[name] = {
            "status": "passed",
            "spatial_num_heads": spatial_heads,
            "temporal_num_heads": temporal_heads,
            "parameter_count": architecture["total_parameters"],
            "architecture": architecture,
            "config": cfg,
        }
    return reports


def gpu_preflight(name):
    import torch
    from MSTT.trainer import TaskRun, build_optimizer, build_scheduler, train_step, gradient_report, predict
    from reimplementation.common.utils.reproducibility import seed_everything

    cfg = config_for(name)
    seed_everything(42)
    run = TaskRun(cfg, task="I2I", rate="p05", output_root=ROOT / name, splits=("train",))
    model = run.build_model()
    spatial_heads, temporal_heads = EXPECTED[name]
    assert model.blocks[0].spatial.num_heads == spatial_heads
    assert model.blocks[0].temporal.num_heads == temporal_heads
    model.train()
    optimizer = build_optimizer(model, cfg)
    build_scheduler(optimizer, cfg)
    batch = next(iter(run.loaders["train"]))
    assert list(batch["x"].shape) == [16, 12, 1406, 1]
    torch.cuda.reset_peak_memory_stats()
    loss, norm = train_step(
        run, model, batch, optimizer, torch.amp.GradScaler("cuda", enabled=False),
        amp="none", clip=cfg["clip_grad_norm"], epoch=0,
    )
    gradients = gradient_report(model)
    assert not gradients["missing"] and not gradients["nonfinite"], gradients
    eval_batch = {k: torch.cat((v, v), dim=0).to(run.device) for k, v in batch.items()}
    model.eval()
    with torch.no_grad():
        assert list(predict(model, eval_batch, amp="none", device=run.device).shape) == [32, 1, 1406, 1]
    torch.cuda.synchronize()
    report = {
        "status": "passed",
        "loss": loss,
        "gradient_norm": norm,
        "spatial_num_heads": spatial_heads,
        "temporal_num_heads": temporal_heads,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "architecture": model.architecture_summary(),
    }
    write(ROOT / name / "preflight.json", report)
    print(f"[preflight] {name}: {json.dumps(report)}", flush=True)
    del run, model, optimizer, batch, eval_batch
    gc.collect()
    torch.cuda.empty_cache()


def refresh_summary():
    summarize(output_stem="comparison_with_head1")


def worker(name):
    from MSTT.trainer import train_task
    gpu = int(os.environ["CUDA_VISIBLE_DEVICES"])
    if gpu not in idle_cards():
        log(f"{name}: GPU {gpu} became busy before preflight; returning to queue")
        return 75
    try:
        exact_match(name)
        update_job(name, status="preflight", training_pid=os.getpid(), training_process_start=process_identity(os.getpid()), gpu=gpu)
        gpu_preflight(name)
        update_job(name, status="training", training_started_at=timestamp())
        log(f"Formal training starts: {name} GPU {gpu} PID {os.getpid()} heads S={EXPECTED[name][0]} T={EXPECTED[name][1]}")
        train_task(config_for(name), task="I2I", rate="p05", output_root=ROOT / name)
        update_job(name, status="completed", completed_at=timestamp())
        refresh_summary()
        return 0
    except Exception as exc:
        update_job(name, status="failed", error=repr(exc), failed_at=timestamp())
        traceback.print_exc()
        refresh_summary()
        return 1


def supervise(lock_fd):
    from MSTT.trainer import run_is_complete
    jobs = {}
    for name in NAMES:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
        update_job(name, status="waiting_gpu", waiting_pid=os.getpid(), gpu=None, training_pid=None)
        jobs[name] = {"status": "queued"}
    log("Waiting for independent GPUs; three single-head STF runs, one GPU each, poll=30s, no timeout")
    children = {}
    while True:
        for name, child in list(children.items()):
            code = child["proc"].poll()
            if code is None:
                continue
            child["handle"].close()
            child["lock"].close()
            children.pop(name)
            if code == 75:
                jobs[name]["status"] = "queued"
                update_job(name, status="waiting_gpu", training_pid=None, gpu=None)
                log(f"{name}: card taken before start; still waiting")
            else:
                state = read(experiment_state(name), {})
                ok = code == 0 and run_is_complete(run_dir(name))
                jobs[name]["status"] = "completed" if ok else "failed"
                if not ok and state.get("status") != "failed":
                    update_job(name, status="failed", exit_code=code, error=state.get("error", "Worker exited without completed outputs"))
                elif ok:
                    update_job(name, status="completed", exit_code=0)
                log(f"{name} ended exit={code} status={jobs[name]['status']}; no relaunch")
        if all(job["status"] in ("completed", "failed") for job in jobs.values()) and not children:
            refresh_summary()
            log("All three single-head runs have finished or failed")
            return 0 if all(job["status"] == "completed" for job in jobs.values()) else 1
        try:
            cards = idle_cards()
        except RuntimeError as exc:
            log(f"GPU query unavailable; still waiting: {exc}")
            time.sleep(POLL_SECONDS)
            continue
        launched = False
        for name in NAMES:
            if jobs[name]["status"] != "queued" or not cards:
                continue
            while cards:
                gpu = cards.pop(0)
                gpu_lock = (ROOT / f".gpu_{gpu}.lock").open("a")
                try:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    gpu_lock.close()
                    continue
                try:
                    if gpu not in idle_cards():
                        continue
                    exact_match(name)
                    log_path = ROOT / name / "train.log"
                    handle = log_path.open("a", encoding="utf-8")
                    child = subprocess.Popen(
                        [sys.executable, str(Path(__file__).resolve()), "--worker", name],
                        cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
                             "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
                        start_new_session=True, pass_fds=(lock_fd, gpu_lock.fileno()),
                    )
                    children[name] = {"proc": child, "handle": handle, "lock": gpu_lock}
                    jobs[name] = {"status": "running", "gpu": gpu, "pid": child.pid}
                    update_job(name, status="starting", gpu=gpu, training_pid=child.pid, launched_at=timestamp())
                    log(f"Reserved GPU {gpu} for {name}, worker PID {child.pid}")
                    launched = True
                    gpu_lock = None
                    break
                finally:
                    if gpu_lock is not None:
                        gpu_lock.close()
        if not launched and any(job["status"] == "queued" for job in jobs.values()):
            log("Waiting: not enough independent GPUs for the remaining single-head runs")
        refresh_summary()
        time.sleep(POLL_SECONDS)


def detach():
    ROOT.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"already_active": True, "state": read(STATE, {})}, indent=2))
            return 0
        reports = validate_cpu()
        for name, report in reports.items():
            (ROOT / name).mkdir(parents=True, exist_ok=True)
            write(ROOT / name / "cpu_validation.json", report)
            (ROOT / name / "train.log").touch(exist_ok=True)
        write(ROOT / "head1_source_sha256.json", source_hashes())
        with (ROOT / "head1_wait.log").open("a", encoding="utf-8") as output:
            child = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--supervise", str(lock.fileno())],
                cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(lock.fileno(),),
                env={**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
            )
        print(json.dumps({
            "waiting_pid": child.pid,
            "wait_log": str(ROOT / "head1_wait.log"),
            "parameter_counts": {name: report["parameter_count"] for name, report in reports.items()},
            "heads": {name: {"spatial": report["spatial_num_heads"], "temporal": report["temporal_num_heads"]}
                      for name, report in reports.items()},
        }, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--detach", action="store_true")
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--supervise", type=int, help=argparse.SUPPRESS)
    mode.add_argument("--worker", choices=NAMES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.validate:
        print(json.dumps(validate_cpu(), indent=2))
        return 0
    if args.detach:
        return detach()
    if args.worker:
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            parser.error("worker requires CUDA_VISIBLE_DEVICES")
        return worker(args.worker)
    try:
        return supervise(args.supervise)
    except Exception as exc:
        log(f"supervisor failed: {exc!r}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
