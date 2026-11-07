"""Train soft and projected conservation on the best single-layer TSF backbone.

The backbone matches the finished ``single_tsf_adaptive128_100ep`` run:
one T→S→FFN block, 64-d base embedding concatenated with a 64-d adaptive
table. The two modes differ only by ``conservation_mode``. They write to
``conservation_best_backbone/{soft,projected}/`` and do not touch other runs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MSTT.config import load_config, task_spec, validate_config  # noqa: E402
from data_root import xuancheng26_model_results_3s_dir  # noqa: E402

ROOT = xuancheng26_model_results_3s_dir("MSTT") / "conservation_best_backbone"
CONFIG_DIR = PROJECT_ROOT / "MSTT/configs/conservation_best_backbone"
MODES = ("soft", "projected")
PROTECTED_PIDS = (2473188, 2474713)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_mode(mode: str) -> dict:
    if mode not in MODES:
        raise ValueError(mode)
    cfg = load_config(CONFIG_DIR / f"{mode}.json")
    validate_config(cfg)
    if cfg["conservation_mode"] != mode:
        raise RuntimeError(f"{mode} config conservation_mode is {cfg['conservation_mode']}")
    if Path(cfg["output_root"]).resolve() != (ROOT / mode).resolve():
        raise RuntimeError(f"{mode} output_root {cfg['output_root']} is not {(ROOT / mode)}")
    if cfg["d_model"] != 128 or cfg["num_layers"] != 1 or cfg["attention_order"] != "TS":
        raise RuntimeError(f"{mode} is not the single-layer TSF-128 backbone")
    if cfg["base_embedding_dim"] != 64 or cfg["adaptive_embedding_dim"] != 64:
        raise RuntimeError(f"{mode} embedding widths are not 64/64")
    if cfg["adaptive_embedding_fusion"] != "concat" or cfg["use_adaptive_embedding"] is not True:
        raise RuntimeError(f"{mode} is not adaptive concatenation")
    if int(cfg["epochs"]) != 100 or cfg["early_stop_patience"] is not None or int(cfg["seed"]) != 42:
        raise RuntimeError(f"{mode} training schedule differs from the backbone run")
    if int(cfg["batch_size"]) != 16 or int(cfg["eval_batch_size"]) != 32:
        raise RuntimeError(f"{mode} batch size was changed")
    return cfg


def validate() -> dict:
    import torch
    from MSTT.data.dataset import target_scaler
    from MSTT.model import MSTT
    from MSTT.model.conservation import conservation_residual
    from MSTT.model.graph import build_spatial_graph
    from MSTT.trainer import assert_conservation_checkpoint, model_config
    from MSTT.utils.losses import conservation_objective, RawScaleLoss
    from reimplementation.common.errors import ReimplementationError

    torch.set_num_threads(2)
    configs = {mode: load_mode(mode) for mode in MODES}
    task = task_spec("All2All")
    mean, std = target_scaler(Path(configs["soft"]["data_root"]), task)
    if tuple(mean.shape) != (4,) or tuple(std.shape) != (4,):
        raise RuntimeError(f"All2All target scaler shape {mean.shape} {std.shape}")
    order = ("inflow", "outflow", "accumulation_start", "accumulation_end")
    if task.output_channels != order or task.input_channels != order:
        raise RuntimeError(f"channel order {task.input_channels} -> {task.output_channels}")

    def build(mode: str, nodes: int = 7):
        cfg = {**configs[mode], "num_nodes": nodes}
        model = MSTT(
            model_config(cfg, task),
            build_spatial_graph(torch.zeros(nodes, nodes)),
            target_mean=mean, target_std=std,
        )
        return cfg, model

    _, none_model = build("soft")
    none_cfg = {**configs["soft"], "conservation_mode": "none", "num_nodes": 7}
    none_model = MSTT(
        model_config(none_cfg, task), build_spatial_graph(torch.zeros(7, 7)),
        target_mean=mean, target_std=std,
    )
    _, soft = build("soft")
    _, projected = build("projected")
    if soft.projection is not None or none_model.projection is not None:
        raise RuntimeError("soft/none built a projection module")
    if projected.projection is None:
        raise RuntimeError("projected mode did not build the conservation MLP")
    first = projected.projection.mlp[0]
    if int(first.in_features) != 128 + 4 or int(projected.projection.mlp[-1].out_features) != 4:
        raise RuntimeError(f"projection MLP shape {first.in_features} -> {projected.projection.mlp[-1].out_features}")
    if int(projected.projection.mlp[0].out_features) != 32:
        raise RuntimeError("projection hidden dim is not 32")
    if float(projected.projection.mlp[-1].weight.abs().sum()) != 0.0 or float(projected.projection.mlp[-1].bias.abs().sum()) != 0.0:
        raise RuntimeError("projection output layer is not zero-initialized")
    if soft.parameter_count() != none_model.parameter_count():
        raise RuntimeError("soft conservation changed the backbone parameter count")
    if projected.parameter_count() <= soft.parameter_count():
        raise RuntimeError("projected mode did not add MLP parameters")

    torch.manual_seed(0)
    x = torch.randn(2, 12, 7, 4)
    calendar = torch.zeros(2, 12, dtype=torch.long)
    soft.eval()
    raw = soft(x, calendar, calendar)
    if list(raw.shape) != [2, 7, 4]:
        raise RuntimeError(f"soft output {tuple(raw.shape)}")
    criterion = RawScaleLoss(mean, std, kind="mae_raw")
    target = torch.rand(2, 7, 4)
    loss, parts = conservation_objective(criterion, {"raw_pred": raw, "final_pred": raw}, target, configs["soft"])
    loss.backward()
    head_with = soft.head.proj.weight.grad.detach().clone()
    soft.zero_grad(set_to_none=True)
    raw_again = soft(x, calendar, calendar)
    criterion(raw_again, target).backward()
    if torch.allclose(head_with, soft.head.proj.weight.grad):
        raise RuntimeError("soft conservation term did not change the prediction-head gradient")
    if "conservation_loss" not in parts:
        raise RuntimeError("soft objective did not record the conservation term")

    projected.eval()
    aux = projected.forward_with_aux(x, calendar, calendar)
    final_physical = aux["final_pred"].float() * projected.projection.target_std + projected.projection.target_mean
    residual = conservation_residual(final_physical).abs().max()
    if float(residual) > 1e-4:
        raise RuntimeError(f"projected residual {float(residual)} is not at float precision")
    projected.train()
    aux = projected.forward_with_aux(x, calendar, calendar)
    total, _ = conservation_objective(criterion, aux, target, configs["projected"])
    total.backward()
    if projected.projection.mlp[-1].weight.grad is None or not torch.isfinite(projected.projection.mlp[-1].weight.grad).all():
        raise RuntimeError("projected MLP did not receive a finite gradient")

    payload = {
        "conservation_mode": "soft",
        "config": configs["soft"],
        "normalization_sha256": "unit",
        "model_state_dict": soft.state_dict(),
    }
    restored = build("soft")[1]
    restored.load_state_dict(payload["model_state_dict"])
    assert_conservation_checkpoint(payload, configs["soft"], normalization_sha256="unit")
    try:
        assert_conservation_checkpoint(payload, configs["projected"], normalization_sha256="unit")
    except ReimplementationError:
        pass
    else:
        raise RuntimeError("a soft checkpoint was accepted as projected")
    none_forward = none_model(x, calendar, calendar)
    if none_model.projection is not None or list(none_forward.shape) != [2, 7, 4]:
        raise RuntimeError("none mode forward changed")
    for mode in MODES:
        result = ROOT / mode
        if result.exists() and any(result.rglob("*checkpoint.pt")):
            raise RuntimeError(f"{result} already contains a checkpoint")
    for pid in PROTECTED_PIDS:
        if not Path(f"/proc/{pid}").exists():
            raise RuntimeError(f"protected training pid {pid} is not running")
    return {
        "status": "passed",
        "soft_parameters": soft.parameter_count(),
        "projected_parameters": projected.parameter_count(),
        "projection_in_features": int(first.in_features),
        "projected_residual_max_abs": float(residual),
        "target_scaler_channels": 4,
    }


def free_gpus() -> list[int]:
    from MSTT.sftf_experiment import idle_cards
    return idle_cards()


def spawn(mode: str, gpu: int) -> int:
    log_dir = ROOT / mode
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"
    with log_path.open("a", encoding="utf-8") as output:
        output.write(f"start_time={timestamp()}\nphysical_gpu={gpu}\nmode={mode}\n")
        output.flush()
        child = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", mode, str(gpu)],
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
                 "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
            start_new_session=True,
        )
    state = {
        "status": "training",
        "mode": mode,
        "gpu": gpu,
        "pid": child.pid,
        "started_at": timestamp(),
        "log": str(log_path),
        "output_root": str(ROOT / mode),
    }
    (log_dir / "launcher_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return child.pid


def worker(mode: str, gpu: int) -> int:
    import gc
    import torch
    from MSTT.trainer import TaskRun, build_optimizer, train_step, train_task
    from reimplementation.common.utils.reproducibility import seed_everything

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise SystemExit("worker GPU does not match CUDA_VISIBLE_DEVICES")
    cfg = load_mode(mode)
    seed_everything(42)
    run = TaskRun(cfg, task="All2All", rate="p05", output_root=ROOT / mode, splits=("train",))
    model = run.build_model()
    if mode == "projected":
        if model.projection is None or int(model.projection.mlp[0].in_features) != 132:
            raise SystemExit(f"projected MLP input is not 132")
    else:
        if model.projection is not None:
            raise SystemExit("soft mode built a projection")
    optimizer = build_optimizer(model, cfg)
    batch = next(iter(run.loaders["train"]))
    if list(batch["x"].shape) != [16, 12, 1406, 4] or list(batch["y"].shape) != [16, 1, 1406, 4]:
        raise SystemExit(f"unexpected batch x={tuple(batch['x'].shape)} y={tuple(batch['y'].shape)}")
    loss, _norm = train_step(
        run, model, batch, optimizer, torch.amp.GradScaler("cuda", enabled=False),
        amp="none", clip=float(cfg["clip_grad_norm"]), epoch=0,
    )
    if not torch.isfinite(torch.tensor(loss)):
        raise SystemExit(f"non-finite preflight loss {loss}")
    print(f"[preflight] {mode} gpu={gpu} loss={loss:.6f} parameters={model.parameter_count()}", flush=True)
    del run, model, optimizer, batch
    gc.collect()
    torch.cuda.empty_cache()
    train_task(cfg, task="All2All", rate="p05", output_root=ROOT / mode)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--detach", action="store_true")
    mode.add_argument("--worker", nargs=2, metavar=("MODE", "GPU"))
    args = parser.parse_args()
    if args.worker is not None:
        return worker(args.worker[0], int(args.worker[1]))
    report = validate()
    print(json.dumps(report, indent=2), flush=True)
    if args.validate:
        return 0
    cards = free_gpus()
    if len(cards) < 2:
        raise SystemExit(f"need two free GPUs for soft and projected, found {cards}")
    assignment = dict(zip(MODES, cards[:2]))
    pids = {name: spawn(name, gpu) for name, gpu in assignment.items()}
    print(json.dumps({"gpus": assignment, "pids": pids, "output_root": str(ROOT)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
