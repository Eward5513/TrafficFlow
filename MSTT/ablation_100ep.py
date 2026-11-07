"""Fixed four-run p05/I2I/seed42 queue, preflight, and automatic comparison.

Run with the trafficflow Python environment. The supervisor never claims a GPU
with a compute process and never changes model/batch/precision to fit memory.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MSTT.config import load_config, task_spec, validate_config, with_legacy_defaults
from data_root import xuancheng26_model_results_3s_dir

NAMES = (
    "control_global_100ep", "adaptive_concat_128_100ep",
    "ffn_after_each_attention_100ep", "single_ts_ffn_100ep",
)
ROOT = xuancheng26_model_results_3s_dir("MSTT") / "ablation_100ep"
CONFIG_DIR = PROJECT_ROOT / "MSTT/configs/ablation_100ep"
LEGACY = ROOT.parent / "v0.4_global/p05/I2I/seed_42"


def read(path, fallback=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else fallback


def write(path, payload):
    from reimplementation.common.utils.atomic_io import atomic_write_json
    atomic_write_json(path, payload)


def config_for(name):
    cfg = load_config(CONFIG_DIR / f"{name}.json")
    cfg["output_root"] = str(ROOT / name)
    validate_config(cfg)
    assert cfg["seed"] == 42 and cfg["epochs"] == 100 and cfg["early_stop_patience"] is None
    assert cfg["ffn_hidden_dim"] == 176 and cfg["selection_metric"] == "val_mae"
    assert cfg["max_train_batches"] is None and cfg["max_eval_batches"] is None
    return cfg


def run_dir(name):
    return ROOT / name / "p05/I2I/seed_42"


def exact_match(name):
    """Resume only the same full resolved training configuration, not just shape."""
    saved = read(run_dir(name) / "resolved_config.json")
    if saved is None:
        if any(run_dir(name).glob("*checkpoint.pt")):
            raise RuntimeError(f"{name}: checkpoint without resolved_config.json")
        return
    saved = with_legacy_defaults(saved)
    cfg = config_for(name)
    diff = {k: [saved.get(k), v] for k, v in cfg.items() if saved.get(k) != v}
    if diff or saved.get("task") != "I2I" or saved.get("penetration") != "p05" or saved.get("smoke"):
        raise RuntimeError(f"{name}: refusing mismatched existing run: {diff}")


def source_hashes():
    paths = sorted((PROJECT_ROOT / "MSTT").rglob("*.py")) + sorted(CONFIG_DIR.glob("*.json"))
    return {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def prepare():
    """Build exact models on CPU; validate counts and legacy strict checkpoint load."""
    import torch
    from MSTT.trainer import model_config
    from MSTT.model import MSTT
    from MSTT.data.graph import load_r_graph
    from MSTT.train import build_parser
    from MSTT.config import overlay_args

    assert overlay_args(load_config(None), build_parser().parse_args(
        ["--task", "I2I", "--penetration", "p05", "--early-stop-patience", "none"]
    ))["early_stop_patience"] is None
    cfg = config_for(NAMES[0])
    graph, _ = load_r_graph(Path(cfg["graph_path"]), r_nodes_path=Path(cfg["r_nodes"]),
                           window_node_mapping=Path(cfg["data_root"]) / "node_mapping.csv",
                           expected_nodes=1406, direction=cfg["spatial_direction"])
    reports = {}
    old = read(LEGACY / "resolved_config.json")
    for name, counts in zip(NAMES, [(2, 2), (2, 2), (2, 4), (1, 1)]):
        exact_match(name)
        cfg = config_for(name)
        torch.manual_seed(42)
        model = MSTT(model_config(cfg, task_spec("I2I")), graph)
        report = model.architecture_summary()
        assert report["temporal_attention_count"] == report["spatial_attention_count"] == counts[0]
        assert report["independent_ffn_count"] == counts[1]
        assert report["ffn_hidden_dims"] == [176] * counts[1]
        assert report["embedding"]["base_output_dim"] == 64
        reports[name] = {"config": cfg, "architecture": report,
                         "differences_from_v0.4": {k: {"old": old.get(k), "new": v}
                             for k, v in cfg.items() if old.get(k) != v}}
    legacy_model = MSTT(model_config(with_legacy_defaults(old), task_spec("I2I")), graph)
    payload = torch.load(LEGACY / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    legacy_model.load_state_dict(payload["model_state_dict"], strict=True)
    reports["legacy_strict_checkpoint_load"] = "passed"
    reports["source_sha256"] = source_hashes()
    write(ROOT / "experiment_plan.json", reports)
    print(json.dumps({name: reports[name]["architecture"] for name in NAMES}, indent=2), flush=True)


def preflight(name):
    """One real training batch and one eval-shape pass; no test data opened."""
    import torch
    from MSTT.trainer import TaskRun, build_optimizer, build_scheduler, train_step, gradient_report, predict
    from reimplementation.common.utils.reproducibility import seed_everything
    cfg = config_for(name)
    seed_everything(42)
    run = TaskRun(cfg, task="I2I", rate="p05", output_root=ROOT / name, splits=("train",))
    model = run.build_model()
    model.train()
    optimizer = build_optimizer(model, cfg)
    build_scheduler(optimizer, cfg)
    batch = next(iter(run.loaders["train"]))
    assert list(batch["x"].shape) == [16, 12, 1406, 1]
    assert bool((batch["y_raw"] == 0).any()), "real zeros must remain in raw-scale loss"
    torch.cuda.reset_peak_memory_stats()
    shapes = {}
    hooks = [model.embedding.register_forward_hook(lambda m, a, o: shapes.update(base=list(o.shape)))]
    if model.adaptive is not None:
        hooks.append(model.adaptive.register_forward_hook(lambda m, a, o: shapes.update(adaptive=list(o.shape))))
    hooks.append(model.head.register_forward_hook(lambda m, a, o: shapes.update(head=list(o.shape))))
    new_params = {n: p.detach().clone() for n, p in model.named_parameters()
                  if n.startswith("adaptive.") or "temporal_ffn" in n}
    loss, norm = train_step(run, model, batch, optimizer, torch.amp.GradScaler("cuda", enabled=False),
                           amp="none", clip=cfg["clip_grad_norm"], epoch=0)
    gradients = gradient_report(model)
    assert not gradients["missing"] and not gradients["nonfinite"], gradients
    for n, p in model.named_parameters():
        if n in new_params:
            assert n not in gradients["all_zero"] and not torch.equal(p, new_params[n]), n
    assert shapes["base"] == [16, 12, 1406, 64]
    assert shapes["head"] == [16, 1406, 1]
    if name == NAMES[1]:
        assert shapes["adaptive"] == [16, 12, 1406, 128]
        assert list(model.adaptive.adaptive_embedding.shape) == [12, 1406, 64]
    for hook in hooks:
        hook.remove()
    # Eval batch size 32 from duplicated training examples, never the test set.
    eval_batch = {k: torch.cat((v, v), dim=0).to(run.device) for k, v in batch.items()}
    model.eval()
    with torch.no_grad():
        assert list(predict(model, eval_batch, amp="none", device=run.device).shape) == [32, 1, 1406, 1]
    torch.cuda.synchronize()
    report = {"status": "passed", "train_batch_shape": list(batch["x"].shape), "shapes": shapes,
              "loss": loss, "gradient_norm": norm, "gradients": gradients,
              "new_parameters_updated": sorted(new_params), "eval_batch_size": 32,
              "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
              "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
              "gpu": torch.cuda.get_device_name(), "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
              "architecture": model.architecture_summary()}
    write(ROOT / name / "preflight.json", report)
    print(f"[preflight] {name}: {json.dumps(report)}", flush=True)
    del run, model, optimizer, batch, eval_batch, new_params
    gc.collect()
    torch.cuda.empty_cache()


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]  # starttime (field 22)
    except (OSError, IndexError):
        return None


def alive(job):
    return job.get("pid") is not None and job.get("process_start") is not None and process_identity(job["pid"]) == job["process_start"]


def summarize(state=None, *, output_stem="comparison"):
    state = state or read(ROOT / "queue_state.json", {})
    # Summary-only extension: NAMES and the four-run scheduler remain unchanged.
    extra_names = tuple(
        name for name in (
            "single_sftf_100ep", "base_concat_128_100ep",
            "single_ts_ffn_spatial_head1_100ep", "single_ts_ffn_temporal_head1_100ep",
            "single_ts_ffn_both_head1_100ep",
            "single_ts_ffn_wide_head128_100ep",
            "two_layer_sftf_100ep",
            "single_layer_tfsf_100ep",
            "single_tsf_adaptive128_100ep",
            "ttfssf_adaptive128_100ep",
        )
        if (CONFIG_DIR / f"{name}.json").is_file()
    )
    extra_states = {name: read(ROOT / name / "launcher_state.json", {}) for name in extra_names}
    rows = []
    summary_names = (*NAMES, *extra_names)
    for name, directory in [(n, run_dir(n)) for n in summary_names] + [("v0.4_global_reference", LEGACY)]:
        history = read(directory / "training_history.json", [])
        meta = read(directory / "run_metadata.json", {})
        job = state.get("jobs", {}).get(name, {})
        if name in extra_states:
            job = {**extra_states[name], "pid": extra_states[name].get("training_pid")}
        complete = meta.get("status") == "completed"
        best = min(history, key=lambda r: (r["val_mae"], r["epoch"])) if history else {}
        val = read(directory / "validation_metrics.json", {}) if complete else {}
        test = read(directory / "test_metrics.json", {}) if complete else {}
        resolved = read(directory / "resolved_config.json", {})
        row = {"experiment": name, "status": "completed" if complete else job.get("status", "queued"),
               "epochs_completed": len(history), "full_100_epochs": [h["epoch"] for h in history] == list(range(100)),
               "best_epoch_0based": meta.get("best_epoch", best.get("epoch")),
               "best_validation_mae": val.get("mae", best.get("val_mae")),
               **{f"test_{k}": test.get(k) for k in ("mae", "rmse", "mape_nonzero", "wape")},
               "parameters": meta.get("parameter_count", resolved.get("architecture", {}).get("total_parameters")),
               "total_seconds": meta.get("total_experiment_wall_time_sec"),
               "peak_gpu_memory_allocated_mb": meta.get("peak_gpu_memory_allocated_mb",
                   max([h.get("peak_gpu_memory_allocated_mb") or 0 for h in history], default=None)),
               "peak_gpu_memory_reserved_mb": meta.get("peak_gpu_memory_reserved_mb",
                   max([h.get("peak_gpu_memory_reserved_mb") or 0 for h in history], default=None)),
               "gpu": job.get("gpu", meta.get("physical_gpu_id")), "pid": job.get("pid"),
               "run_dir": str(directory)}
        rows.append(row)
    control = rows[0]
    for row in rows:
        for metric in ("best_validation_mae", "test_mae", "test_rmse", "test_mape_nonzero", "test_wape"):
            row[f"delta_vs_A_{metric}"] = (row[metric] - control[metric]
                if row[metric] is not None and control[metric] is not None else None)
    write(ROOT / f"{output_stem}.json", rows)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    from reimplementation.common.utils.atomic_io import atomic_write_text
    atomic_write_text(ROOT / f"{output_stem}.csv", buffer.getvalue())
    return rows


def worker(name, gpu):
    from MSTT.utils.device import busy_physical_gpus
    from MSTT.trainer import train_task
    exact_match(name)
    if gpu in busy_physical_gpus():
        return 75  # Another training process took the card; return to queue.
    preflight(name)
    # train_task resets seed42 and builds a fresh model/optimizer, or restores an
    # exact-match interrupted formal run. Preflight weights never enter training.
    train_task(config_for(name), task="I2I", rate="p05", output_root=ROOT / name)
    return 0


def supervise():
    from MSTT.utils.device import busy_physical_gpus, _nvidia_smi
    from MSTT.trainer import run_is_complete
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / ".supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("An ablation supervisor already exists; no duplicate launched.", flush=True)
            return 0
        prepare()
        state = read(ROOT / "queue_state.json", {"jobs": {}})
        state.update(supervisor_pid=os.getpid(), source_sha256=source_hashes())
        children, handles, idle_since = {}, {}, {}
        for name in NAMES:
            job = state["jobs"].setdefault(name, {"status": "queued"})
            if not alive(job) and not run_is_complete(run_dir(name)):
                job["status"] = "queued"
        while True:
            for name, job in state["jobs"].items():
                child = children.get(name)
                code = child.poll() if child is not None else None
                if code is not None:
                    handles.pop(name).close()
                    children.pop(name)
                    job["exit_code"] = code
                    job["status"] = "completed" if code == 0 else "queued" if code == 75 else "failed"
                elif job["status"] == "running" and not alive(job):
                    job["status"] = "interrupted"
                if run_is_complete(run_dir(name)):
                    job["status"] = "completed"
            busy = busy_physical_gpus()
            reserved = {j["gpu"] for j in state["jobs"].values() if alive(j)}
            now = time.monotonic()
            cards = []
            for line in _nvidia_smi("gpu=index,memory.free,memory.total,name"):
                index, free, total, label = [v.strip() for v in line.split(",", 3)]
                index, free, total = int(index), int(free), int(total)
                if index in busy or index in reserved or free < 16000:
                    idle_since.pop(index, None)
                else:
                    idle_since.setdefault(index, now)
                    if now - idle_since[index] >= 30:
                        cards.append((index, free, total, label))
            for name in NAMES:
                job = state["jobs"][name]
                if job["status"] != "queued" or not cards:
                    continue
                gpu, free, total, label = cards.pop(0)
                exact_match(name)
                log_path = ROOT / name / "train.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                handle = log_path.open("a", encoding="utf-8")
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
                       "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", name, "--gpu", str(gpu)]
                child = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdin=subprocess.DEVNULL,
                                         stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                children[name], handles[name] = child, handle
                job.update(status="running", gpu=gpu, pid=child.pid, process_start=process_identity(child.pid),
                           log=str(log_path), run_dir=str(run_dir(name)), free_memory_mb_at_launch=free,
                           total_memory_mb=total, gpu_name=label, launched_unix=time.time())
                write(ROOT / "queue_state.json", state)
                print(f"[queue] {name} gpu={gpu} pid={child.pid} free={free}MiB log={log_path}", flush=True)
            state["updated_unix"] = time.time()
            write(ROOT / "queue_state.json", state)
            summarize(state)
            if all(j["status"] in ("completed", "failed", "interrupted") for j in state["jobs"].values()):
                return 0 if all(j["status"] == "completed" for j in state["jobs"].values()) else 1
            time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--summarize", action="store_true")
    mode.add_argument("--worker", choices=NAMES)
    mode.add_argument("--detach", action="store_true", help="start the durable queue supervisor and return its PID")
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.detach:
        ROOT.mkdir(parents=True, exist_ok=True)
        with (ROOT / "supervisor.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve())], cwd=PROJECT_ROOT,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     env={**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
                                     start_new_session=True)
        print(json.dumps({"supervisor_pid": child.pid, "log": str(ROOT / "supervisor.log")}), flush=True)
    elif args.prepare:
        prepare()
    elif args.summarize:
        print(json.dumps(summarize(), indent=2))
    elif args.worker:
        if args.gpu is None or os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
            parser.error("worker requires matching --gpu and CUDA_VISIBLE_DEVICES")
        return worker(args.worker, args.gpu)
    else:
        return supervise()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
