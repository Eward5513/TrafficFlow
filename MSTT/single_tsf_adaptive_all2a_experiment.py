"""Queue one All2All TSF block: 64-d base embedding concatenated with a 64-d adaptive table."""
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
    ROOT, config_for, read, write, process_identity, alive, source_hashes,
)
from MSTT.config import with_legacy_defaults
from MSTT.sftf_experiment import idle_cards

NAME = "single_tsf_adaptive128_all2a_100ep"
TASK = "All2All"
EXPERIMENT = ROOT / NAME
STATE = EXPERIMENT / "launcher_state.json"
POLL_SECONDS = 30
MIN_FREE_MB = 16000


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f"[{timestamp()}] {message}", flush=True)


def result_dir() -> Path:
    return EXPERIMENT / "p05" / TASK / "seed_42"


def update(**changes):
    state = read(STATE, {})
    for key, value in changes.items():
        if value is None:
            state.pop(key, None)
        else:
            state[key] = value
    state["updated_at"] = timestamp()
    write(STATE, state)
    return state


def exact_match():
    """Resume only the same All2All configuration. The other adaptive runs stay in their own directories."""
    saved = read(result_dir() / "resolved_config.json")
    if saved is None:
        if any(result_dir().glob("*checkpoint.pt")):
            raise RuntimeError(f"{NAME}: checkpoint without resolved_config.json")
        return
    saved = with_legacy_defaults(saved)
    cfg = config_for(NAME)
    diff = {k: [saved.get(k), v] for k, v in cfg.items() if saved.get(k) != v}
    if diff or saved.get("task") != TASK or saved.get("penetration") != "p05" or saved.get("smoke"):
        raise RuntimeError(f"{NAME}: refusing mismatched existing run: {diff}")


def validate_cpu():
    import torch
    from MSTT.config import task_spec
    from MSTT.model import MSTT
    from MSTT.model.graph import build_spatial_graph
    from MSTT.trainer import model_config

    torch.set_num_threads(2)
    cfg = config_for(NAME)
    task = task_spec(TASK)
    assert task.input_type == "all" and task.target_type == "all"
    assert task.in_channels == 4 and task.out_channels == 4
    widths = (
        cfg["value_embedding_dim"], cfg["node_embedding_dim"],
        cfg["tod_embedding_dim"], cfg["dow_embedding_dim"],
    )
    assert widths == (32, 16, 8, 8) and sum(widths) == 64
    assert cfg["base_embedding_dim"] == 64 and cfg["adaptive_embedding_dim"] == 64
    assert cfg["d_model"] == 128 and cfg["use_adaptive_embedding"] is True
    assert cfg["adaptive_embedding_fusion"] == "concat" and cfg["embedding_fusion"] == "concat"
    assert cfg["num_layers"] == 1 and cfg["attention_order"] == "TS"
    assert cfg["ffn_placement"] == "after_spatial" and cfg["spatial_attention"] == "global"
    torch.manual_seed(42)
    full = MSTT(model_config(cfg, task), build_spatial_graph(torch.zeros(1406, 1406)))
    architecture = full.architecture_summary()
    assert full.adaptive is not None and tuple(full.adaptive.adaptive_embedding.shape) == (12, 1406, 64)
    assert architecture["block_order"] == ["T", "S", "FFN"]
    assert architecture["embedding"]["base_output_dim"] == 64
    assert architecture["embedding"]["output_dim"] == 128
    assert architecture["embedding"]["adaptive"]["enabled"] is True
    assert architecture["embedding"]["adaptive"]["shape"] == [12, 1406, 64]
    assert architecture["embedding"]["widths"] == {"value": 32, "node": 16, "tod": 8, "dow": 8}
    assert architecture["temporal_attention_count"] == architecture["spatial_attention_count"] == 1
    assert architecture["independent_ffn_count"] == 1 and architecture["ffn_hidden_dims"] == [176]
    parameter_count = architecture["total_parameters"]
    del full
    small_cfg = {**cfg, "num_nodes": 7}
    model = MSTT(model_config(small_cfg, task), build_spatial_graph(torch.zeros(7, 7)))
    order = []
    hooks = []
    block = model.blocks[0]
    assert model.adaptive is not None and block.temporal_ffn is None
    for label, module in (("T", block.temporal), ("S", block.spatial), ("FFN", block.ffn)):
        hooks.append(module.register_forward_hook(lambda m, a, o, label=label: order.append(label)))
    calendar = torch.zeros(2, 12, dtype=torch.long)
    x = torch.randn(2, 12, 7, 4)
    base = model.embedding(x, calendar, calendar)
    fused = model.adaptive(base)
    prediction = model(x, calendar, calendar)
    assert list(base.shape) == [2, 12, 7, 64]
    assert list(fused.shape) == [2, 12, 7, 128]
    assert list(prediction.shape) == [2, 7, 4]
    assert order == ["T", "S", "FFN"], order
    prediction.abs().mean().backward()
    assert model.adaptive.adaptive_embedding.grad is not None
    assert torch.isfinite(model.adaptive.adaptive_embedding.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    for hook in hooks:
        hook.remove()
    return {
        "status": "passed",
        "task": TASK,
        "architecture": architecture,
        "parameter_count": parameter_count,
        "base_shape": list(base.shape),
        "fused_shape": list(fused.shape),
        "output_shape": list(prediction.shape),
        "observed_order": order,
        "adaptive_enabled": True,
        "config": cfg,
    }


def gpu_preflight():
    import torch
    from MSTT.trainer import TaskRun, build_optimizer, build_scheduler, train_step, gradient_report, predict
    from reimplementation.common.utils.reproducibility import seed_everything

    cfg = config_for(NAME)
    seed_everything(42)
    run = TaskRun(cfg, task=TASK, rate="p05", output_root=EXPERIMENT, splits=("train",))
    model = run.build_model()
    assert model.adaptive is not None and tuple(model.adaptive.adaptive_embedding.shape) == (12, 1406, 64)
    assert model.architecture_summary()["block_order"] == ["T", "S", "FFN"]
    model.train()
    optimizer = build_optimizer(model, cfg)
    build_scheduler(optimizer, cfg)
    batch = next(iter(run.loaders["train"]))
    assert list(batch["x"].shape) == [16, 12, 1406, 4], list(batch["x"].shape)
    assert list(batch["y"].shape) == [16, 1, 1406, 4], list(batch["y"].shape)
    shapes = {}
    hook = model.embedding.register_forward_hook(lambda m, a, o: shapes.update(base=list(o.shape)))
    adaptive_hook = model.adaptive.register_forward_hook(lambda m, a, o: shapes.update(fused=list(o.shape)))
    torch.cuda.reset_peak_memory_stats()
    loss, norm = train_step(
        run, model, batch, optimizer, torch.amp.GradScaler("cuda", enabled=False),
        amp="none", clip=cfg["clip_grad_norm"], epoch=0,
    )
    hook.remove()
    adaptive_hook.remove()
    gradients = gradient_report(model)
    assert shapes["base"] == [16, 12, 1406, 64], shapes
    assert shapes["fused"] == [16, 12, 1406, 128], shapes
    assert "adaptive.adaptive_embedding" not in gradients["all_zero"]
    assert not gradients["missing"] and not gradients["nonfinite"], gradients
    eval_batch = {k: torch.cat((v, v), dim=0).to(run.device) for k, v in batch.items()}
    model.eval()
    with torch.no_grad():
        assert list(predict(model, eval_batch, amp="none", device=run.device).shape) == [32, 1, 1406, 4]
    torch.cuda.synchronize()
    report = {
        "status": "passed",
        "task": TASK,
        "shapes": shapes,
        "loss": loss,
        "gradient_norm": norm,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "architecture": model.architecture_summary(),
    }
    write(EXPERIMENT / "preflight.json", report)
    print(f"[preflight] {NAME}: {json.dumps(report)}", flush=True)
    del run, model, optimizer, batch, eval_batch
    gc.collect()
    torch.cuda.empty_cache()


def worker(gpu):
    from MSTT.trainer import train_task
    if gpu not in idle_cards():
        log(f"GPU {gpu} became busy before preflight; returning to queue")
        return 75
    try:
        exact_match()
        update(status="preflight", training_pid=os.getpid(), training_process_start=process_identity(os.getpid()), gpu=gpu)
        gpu_preflight()
        update(status="training", training_started_at=timestamp(), error=None, exit_code=None, failed_at=None)
        log(f"Formal training starts: GPU {gpu}, PID {os.getpid()}, All2All, one TSF block, base 64 plus adaptive 64")
        train_task(config_for(NAME), task=TASK, rate="p05", output_root=EXPERIMENT)
        update(status="completed", completed_at=timestamp())
        return 0
    except Exception as exc:
        update(status="failed", error=repr(exc), failed_at=timestamp())
        traceback.print_exc()
        return 1


def supervise(lock_fd):
    from MSTT.trainer import run_is_complete
    update(
        status="waiting_gpu", waiting_pid=os.getpid(), waiting_process_start=process_identity(os.getpid()),
        poll_seconds=POLL_SECONDS, minimum_free_memory_mb=MIN_FREE_MB,
        waiting_started_at=timestamp(), training_pid=None, gpu=None,
    )
    log("Waiting for an independent GPU; poll=30s, no timeout, minimum free=16000MiB")
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
                exact_match()
                if config_for(NAME) != read(EXPERIMENT / "cpu_validation.json")["config"]:
                    raise RuntimeError("Queued configuration changed after CPU validation")
                log(f"Reserved GPU {gpu}; launching one worker")
                with (EXPERIMENT / "train.log").open("a", encoding="utf-8") as output:
                    child = subprocess.Popen(
                        [sys.executable, str(Path(__file__).resolve()), "--worker", str(gpu)],
                        cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
                             "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
                        start_new_session=True, pass_fds=(lock_fd, gpu_lock.fileno()),
                    )
                launched = True
                log(f"Worker PID {child.pid}, GPU {gpu}")
                while child.poll() is None:
                    time.sleep(POLL_SECONDS)
                code = child.returncode
                if code == 75:
                    update(status="waiting_gpu", training_pid=None, gpu=None)
                    log("Card taken before worker started; continuing to wait")
                    break
                state = read(STATE, {})
                if code != 0 or not run_is_complete(result_dir()):
                    update(status="failed", exit_code=code, error=state.get("error", "Worker exited without completed outputs"))
                else:
                    update(status="completed", exit_code=0, error=None, failed_at=None)
                log(f"Worker ended with exit={code}; status={read(STATE)['status']}; no relaunch")
                return 0 if read(STATE)["status"] == "completed" else 1
            finally:
                gpu_lock.close()
        update(last_gpu_poll_at=timestamp())
        if not launched:
            log("Waiting: no independent GPU with sufficient free memory")
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
        if state.get("status") == "completed" or (
            state.get("status") == "failed" and (result_dir() / "last_checkpoint.pt").is_file()
        ):
            print(json.dumps({"not_restarted": True, **state}, indent=2))
            return 1 if state.get("status") == "failed" else 0
        if alive({"pid": state.get("waiting_pid"), "process_start": state.get("waiting_process_start")}):
            print(json.dumps({"already_active": True, **state}, indent=2))
            return 0
        exact_match()
        if (result_dir() / "last_checkpoint.pt").exists():
            raise RuntimeError("Existing formal checkpoint; inspect interrupted run before explicitly resuming")
        report = validate_cpu()
        write(EXPERIMENT / "cpu_validation.json", report)
        write(EXPERIMENT / "source_sha256.json", source_hashes())
        (EXPERIMENT / "train.log").touch(exist_ok=True)
        with (EXPERIMENT / "wait.log").open("a", encoding="utf-8") as output:
            child = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--supervise", str(lock.fileno())],
                cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(lock.fileno(),),
                env={**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
            )
        print(json.dumps({
            "waiting_pid": child.pid,
            "wait_log": str(EXPERIMENT / "wait.log"),
            "parameter_count": report["parameter_count"],
            "task": TASK,
            "run_dir": str(result_dir()),
        }, indent=2))
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
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
