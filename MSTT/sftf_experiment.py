"""Validate and queue only single_sftf_100ep; never launch the other ablations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
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
    ROOT, config_for, exact_match, read, write, run_dir, preflight,
    process_identity, alive, summarize, source_hashes,
)

NAME = "single_sftf_100ep"
EXPERIMENT = ROOT / NAME
STATE = EXPERIMENT / "launcher_state.json"
POLL_SECONDS = 30
MIN_FREE_MB = 16000


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f"[{timestamp()}] {message}", flush=True)


def update(**changes):
    state = read(STATE, {})
    state.update(changes, updated_at=timestamp())
    write(STATE, state)
    return state


def validate_cpu():
    import torch
    from MSTT.config import task_spec, overlay_args, load_config
    from MSTT.train import build_parser
    from MSTT.trainer import model_config, build_optimizer, gradient_report
    from MSTT.model import MSTT
    from MSTT.model.graph import build_spatial_graph

    torch.set_num_threads(2)
    cfg = config_for(NAME)
    assert cfg["num_layers"] == 1 and cfg["attention_order"] == "ST"
    args = build_parser().parse_args(["--task", "I2I", "--penetration", "p05", "--config",
        str(PROJECT_ROOT / "MSTT/configs/ablation_100ep" / f"{NAME}.json"), "--early-stop-patience", "null"])
    assert overlay_args(load_config(args.config), args) == cfg
    torch.manual_seed(42)
    full = MSTT(model_config(cfg, task_spec("I2I")), build_spatial_graph(torch.zeros(1406, 1406)))
    architecture = full.architecture_summary()
    assert architecture["block_order"] == ["S", "FFN_S", "T", "FFN_T"]
    assert architecture["temporal_attention_count"] == architecture["spatial_attention_count"] == 1
    assert architecture["independent_ffn_count"] == 2 and architecture["ffn_hidden_dims"] == [176, 176]
    del full
    small_cfg = {**cfg, "num_nodes": 7}
    model = MSTT(model_config(small_cfg, task_spec("I2I")), build_spatial_graph(torch.zeros(7, 7)))
    block = model.blocks[0]
    assert not ({id(p) for p in block.ffn.parameters()} & {id(p) for p in block.temporal_ffn.parameters()})
    order = []
    hooks = [module.register_forward_hook(lambda m, a, o, label=label: order.append(label))
             for label, module in (("S", block.spatial), ("FFN_S", block.ffn),
                                   ("T", block.temporal), ("FFN_T", block.temporal_ffn))]
    x = torch.randn(2, 12, 7, 1)
    calendar = torch.zeros(2, 12, dtype=torch.long)
    optimizer = build_optimizer(model, cfg)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if "ffn" in n}
    prediction = model(x, calendar, calendar)
    assert list(prediction.shape) == [2, 7, 1]
    assert order == ["S", "FFN_S", "T", "FFN_T"], order
    prediction.abs().mean().backward()
    gradients = gradient_report(model)
    assert not any(gradients.values()), gradients
    optimizer.step()
    for n, p in model.named_parameters():
        if n in before:
            assert not torch.equal(before[n], p), n
    for hook in hooks:
        hook.remove()
    # Verify the four Pre-RMSNorm residual equations, with dropout off for exact comparison.
    model.eval()
    with torch.no_grad():
        h = model.embedding(x, calendar, calendar)
        expected = h + block.spatial(block.spatial_norm(h))
        expected = expected + block.ffn(block.ffn_norm(expected))
        expected = expected + block.temporal(block.temporal_norm(expected))
        expected = expected + block.temporal_ffn(block.temporal_ffn_norm(expected))
        assert torch.equal(block(h), expected)
        assert torch.equal(model(x, calendar, calendar), model.head(expected))
    # Exercise the actual trainer's early-stop condition at arbitrarily large wait.
    import ast
    tree = ast.parse((PROJECT_ROOT / "MSTT/trainer.py").read_text())
    stop = next(n.test for n in ast.walk(tree) if isinstance(n, ast.If)
                and ast.unparse(n.test) == "patience is not None and wait >= int(patience)")
    expr = compile(ast.Expression(stop), "trainer_early_stop", "eval")
    assert not eval(expr, {"patience": cfg["early_stop_patience"], "wait": 10000})
    assert eval(expr, {"patience": 0, "wait": 0})  # zero is not disabled
    return {"status": "passed", "architecture": architecture, "observed_order": order,
            "input_shape": list(x.shape), "output_shape": list(prediction.shape),
            "independent_ffns": True, "gradients": gradients, "optimizer_step": "passed",
            "pre_norm_residual_equations": "passed", "early_stop_null": "disabled", "config": cfg}


def scheduler_reserved_gpus():
    """GPUs the baseline scheduler has already assigned, even before CUDA shows up."""
    payload = read(Path("/tmp/retry_three_models_scheduler/task_status.json"), {})
    reserved = set()
    for key, job in (payload.get("running") or {}).items():
        try:
            reserved.add(int(job.get("gpu", key)))
        except (TypeError, ValueError):
            continue
    return reserved


def idle_cards():
    from MSTT.utils.device import busy_physical_gpus, _nvidia_smi
    busy = busy_physical_gpus()
    busy.update(scheduler_reserved_gpus())
    # Respect reservations recorded by the existing four-run supervisor too.
    for job in read(ROOT / "queue_state.json", {}).get("jobs", {}).values():
        if alive(job):
            busy.add(job["gpu"])
    cards = []
    for line in _nvidia_smi("gpu=index,memory.free"):
        gpu, free = [int(v.strip()) for v in line.split(",")]
        if gpu not in busy and free >= MIN_FREE_MB:
            cards.append(gpu)
    return sorted(cards)


def refresh_summary():
    # The already-running old supervisor still writes the original comparison.
    # This separate combined file cannot be overwritten by that old process.
    summarize(output_stem="comparison_with_sftf")


def worker(gpu):
    from MSTT.trainer import train_task
    if gpu not in idle_cards():
        log(f"GPU {gpu} became busy before preflight; returning to queue")
        return 75
    try:
        exact_match(NAME)
        update(status="preflight", training_pid=os.getpid(), training_process_start=process_identity(os.getpid()), gpu=gpu)
        preflight(NAME)
        update(status="training", training_started_at=timestamp())
        log(f"Formal training starts: GPU {gpu}, PID {os.getpid()}, seed42, 100 epochs, early_stop=null")
        train_task(config_for(NAME), task="I2I", rate="p05", output_root=EXPERIMENT)
        update(status="completed", completed_at=timestamp())
        refresh_summary()
        return 0
    except Exception as exc:
        update(status="failed", error=repr(exc), failed_at=timestamp())
        traceback.print_exc()
        refresh_summary()
        return 1


def supervise(lock_fd):
    from MSTT.trainer import run_is_complete
    update(status="waiting_gpu", waiting_pid=os.getpid(), waiting_process_start=process_identity(os.getpid()),
           poll_seconds=POLL_SECONDS, minimum_free_memory_mb=MIN_FREE_MB,
           waiting_started_at=timestamp(), training_pid=None, gpu=None)
    log("Waiting for an independent GPU; poll=30s, no timeout, minimum free=16000MiB")
    refresh_summary()
    while True:
        try:
            cards = idle_cards()
        except RuntimeError as exc:
            update(last_gpu_check_error=str(exc))
            log(f"GPU query unavailable; still waiting: {exc}")
            time.sleep(POLL_SECONDS)
            continue
        launched = False
        for gpu in cards:
            gpu_lock = (ROOT / f".gpu_{gpu}.lock").open("a")
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                gpu_lock.close()
                continue
            try:
                if gpu not in idle_cards():
                    continue
                exact_match(NAME)
                if config_for(NAME) != read(EXPERIMENT / "cpu_validation.json")["config"]:
                    raise RuntimeError("Queued configuration changed after CPU validation")
                log(f"Reserved GPU {gpu}; launching one worker")
                with (EXPERIMENT / "train.log").open("a", encoding="utf-8") as output:
                    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(gpu)],
                        cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
                             "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
                        start_new_session=True, pass_fds=(lock_fd, gpu_lock.fileno()))
                launched = True
                log(f"Worker PID {child.pid}, GPU {gpu}; train.log records validation and training")
                while child.poll() is None:
                    refresh_summary()
                    time.sleep(POLL_SECONDS)
                code = child.returncode
                if code == 75:
                    update(status="waiting_gpu", training_pid=None, gpu=None)
                    log("Card taken before worker started; continuing to wait")
                    break
                state = read(STATE, {})
                if code != 0 or not run_is_complete(run_dir(NAME)):
                    update(status="failed", exit_code=code, error=state.get("error", "Worker exited without completed outputs"))
                else:
                    update(status="completed", exit_code=0)
                refresh_summary()
                log(f"Worker ended with exit={code}; status={read(STATE)['status']}; no relaunch")
                return 0 if read(STATE)["status"] == "completed" else 1
            finally:
                gpu_lock.close()
        update(last_gpu_poll_at=timestamp())
        if not launched:
            log("Waiting: no independent GPU with sufficient free memory")
        refresh_summary()
        time.sleep(POLL_SECONDS)


def detach():
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    with (EXPERIMENT / ".launcher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"already_active": True, **read(STATE, {})}, indent=2))
            return 0
        state = read(STATE, {})
        if state.get("status") in ("failed", "completed"):
            print(json.dumps({"not_restarted": True, **state}, indent=2))
            return 1 if state["status"] == "failed" else 0
        exact_match(NAME)
        # Never silently resume a new invocation after a crashed formal run.
        if (run_dir(NAME) / "last_checkpoint.pt").exists():
            raise RuntimeError("Existing formal checkpoint; inspect interrupted run before explicitly resuming")
        report = validate_cpu()
        write(EXPERIMENT / "cpu_validation.json", report)
        write(EXPERIMENT / "source_sha256.json", source_hashes())
        (EXPERIMENT / "train.log").touch(exist_ok=True)
        with (EXPERIMENT / "wait.log").open("a", encoding="utf-8") as output:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--supervise", str(lock.fileno())],
                cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(lock.fileno(),),
                env={**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
        print(json.dumps({"waiting_pid": child.pid, "wait_log": str(EXPERIMENT / "wait.log"),
                          "parameter_count": report["architecture"]["total_parameters"]}, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--detach", action="store_true")
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--supervise", type=int, help=argparse.SUPPRESS)
    mode.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.validate:
        print(json.dumps(validate_cpu(), indent=2))
        return 0
    if args.detach:
        return detach()
    if args.worker is not None:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.worker):
            parser.error("worker GPU must match CUDA_VISIBLE_DEVICES")
        return worker(args.worker)
    try:
        return supervise(args.supervise)
    except Exception as exc:
        update(status="failed", error=repr(exc), failed_at=timestamp())
        traceback.print_exc()
        refresh_summary()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
