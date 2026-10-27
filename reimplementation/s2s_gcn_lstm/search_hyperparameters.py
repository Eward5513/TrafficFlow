"""Random search for Seq2seq GCN-LSTM, following Table II of Abdelraouf et al. 2023.

This is not KerasTuner and not a TPE or Bayesian search. Each trial draws one
value uniformly from each discrete range below, then trains from scratch.

Learning rate
-------------
Table II gives ``[1e-4, 1e-2]`` with step ``0.5`` on a log scale. KerasTuner
1.1.2 rejects ``Float(sampling="log", step=0.5)`` because ``sampling`` and
``step`` cannot both be set. Its linear ``step`` grid is ``arange(min, max,
step)``; a step of 0.5 on ``[1e-4, 1e-2]`` leaves only ``1e-4``. Current
KerasTuner documents a log ``step`` as a multiplier (``step=10`` yields
decades). A multiplier of 0.5 moves below the lower bound, so it is not a
grid inside the published interval.

The search therefore uses the half-decade reading of "step 0.5 (log scale)":
uniform draws from

    1e-4, 10 ** -3.5, 1e-3, 10 ** -2.5, 1e-2

No other learning rates are used. Test metrics are never computed.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from data_root import xuancheng26_model_config
from reimplementation.common.data.r_only_npz_dataset import (
    ROnlyNPZDataset,
    build_dataloader,
    expected_split_counts,
    load_json,
    rate_tag,
    read_target_scaler,
)
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.utils.atomic_io import atomic_write_json
from reimplementation.common.utils.reproducibility import seed_everything
from reimplementation.s2s_gcn_lstm.engine import (
    _move_batch,
    _validation_mae,
    build_model,
    build_optimizer,
    build_scheduler,
)
from reimplementation.s2s_gcn_lstm.losses import MAERawLoss
from reimplementation.s2s_gcn_lstm.model.s2s_gcn_lstm import CODE_VERSION, MODEL_LABEL
from reimplementation.s2s_gcn_lstm.validation import assert_feature_order, assert_task_config
from reimplementation.stgcn.engine import resolve_device, resolve_path, resolve_training_output
from reimplementation.stgcn.validation import load_edge_ids

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = xuancheng26_model_config("s2s_gcn_lstm")

ENCODER_GCN_LAYERS = (1, 2, 3, 4, 5)
ENCODER_LSTM_LAYERS = (1, 2, 3, 4, 5)
DECODER_GCN_LAYERS = (1, 2, 3, 4, 5)
DECODER_LSTM_LAYERS = (1, 2, 3, 4, 5)
GCN_HIDDEN_SIZES = (4, 8, 12, 16, 20, 24, 28, 32)
LSTM_HIDDEN_SIZES = (128, 256, 384, 512)
DROPOUTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
BATCH_SIZES = (32, 64, 96, 128)
LEARNING_RATES = tuple(float(10 ** exponent) for exponent in (-4.0, -3.5, -3.0, -2.5, -2.0))

TRIAL_FIELDS = (
    "trial_id",
    "encoder_gcn_layers",
    "encoder_lstm_layers",
    "decoder_gcn_layers",
    "decoder_lstm_layers",
    "gcn_hidden_size",
    "lstm_hidden_size",
    "dropout",
    "batch_size",
    "learning_rate",
    "parameter_count",
    "best_epoch",
    "best_validation_mae",
    "training_time_sec",
    "status",
    "error",
)
FINISHED_STATUSES = {"completed", "failed_oom", "failed"}


def sample_trial(seed: int, trial_id: int) -> dict[str, Any]:
    """One independent uniform draw. Trial ``i`` does not depend on trial ``i-1``."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(trial_id), 2023]))
    return {
        "trial_id": int(trial_id),
        "encoder_gcn_layers": int(rng.choice(ENCODER_GCN_LAYERS)),
        "encoder_lstm_layers": int(rng.choice(ENCODER_LSTM_LAYERS)),
        "decoder_gcn_layers": int(rng.choice(DECODER_GCN_LAYERS)),
        "decoder_lstm_layers": int(rng.choice(DECODER_LSTM_LAYERS)),
        "gcn_hidden_size": int(rng.choice(GCN_HIDDEN_SIZES)),
        "lstm_hidden_size": int(rng.choice(LSTM_HIDDEN_SIZES)),
        "dropout": float(rng.choice(DROPOUTS)),
        "batch_size": int(rng.choice(BATCH_SIZES)),
        "learning_rate": float(rng.choice(LEARNING_RATES)),
    }


def search_directory(output_root: Path, penetration: str) -> Path:
    return Path(output_root) / rate_tag(penetration) / "hyperparameter_search"


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _read_trials(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_trials(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in TRIAL_FIELDS})
    os.replace(temporary, path)


def _completed_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    finished = []
    for row in rows:
        if str(row.get("status")) != "completed":
            continue
        finished.append(row)
    finished.sort(key=lambda row: (float(row["best_validation_mae"]), int(row["trial_id"])))
    return finished


def _write_summary(directory: Path, rows: list[Mapping[str, Any]]) -> None:
    ranked = _completed_rows(rows)
    lines = [
        "# Seq2seq GCN-LSTM random search",
        "",
        "Selection metric: validation MAE on raw counts, including zeros. Test was not evaluated.",
        "",
        f"Completed trials: {len(ranked)}",
        f"Failed trials: {sum(1 for row in rows if str(row.get('status')) not in {'completed', ''})}",
        "",
        "## Top validation MAE",
        "",
    ]
    if not ranked:
        lines.append("No completed trial yet.")
    else:
        lines.append("| Rank | Trial | Validation MAE | Parameters | Layers (GCN enc/LSTM enc/GCN dec/LSTM dec) | Hidden (GCN/LSTM) | Dropout | Batch | LR |")
        lines.append("|---:|---:|---:|---:|---|---|---:|---:|---:|")
        for rank, row in enumerate(ranked[:5], start=1):
            layers = (
                f"{row['encoder_gcn_layers']}/{row['encoder_lstm_layers']}/"
                f"{row['decoder_gcn_layers']}/{row['decoder_lstm_layers']}"
            )
            lines.append(
                "| {rank} | {trial} | {mae:.6f} | {params} | {layers} | {gcn}/{lstm} | {dropout} | {batch} | {lr:.6g} |".format(
                    rank=rank,
                    trial=row["trial_id"],
                    mae=float(row["best_validation_mae"]),
                    params=row["parameter_count"],
                    layers=layers,
                    gcn=row["gcn_hidden_size"],
                    lstm=row["lstm_hidden_size"],
                    dropout=row["dropout"],
                    batch=row["batch_size"],
                    lr=float(row["learning_rate"]),
                )
            )
    (directory / "search_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _publish_best(directory: Path, rows: list[Mapping[str, Any]]) -> None:
    ranked = _completed_rows(rows)
    _write_summary(directory, rows)
    if not ranked:
        return
    best = ranked[0]
    trial_dir = directory / "trials" / f"trial_{int(best['trial_id']):03d}"
    config_path = trial_dir / "config.json"
    checkpoint_path = trial_dir / "best_checkpoint.pt"
    if config_path.is_file():
        atomic_write_json(directory / "best_config.json", json.loads(config_path.read_text(encoding="utf-8")))
    if checkpoint_path.is_file():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        target = directory / "best_model.pt"
        temporary = target.with_name(".best_model.pt.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)


def _apply_trial(config: dict[str, Any], trial: Mapping[str, Any], *, max_epochs: int) -> dict[str, Any]:
    updated = dict(config)
    updated["encoder_gcn_layers"] = int(trial["encoder_gcn_layers"])
    updated["encoder_lstm_layers"] = int(trial["encoder_lstm_layers"])
    updated["decoder_gcn_layers"] = int(trial["decoder_gcn_layers"])
    updated["decoder_lstm_layers"] = int(trial["decoder_lstm_layers"])
    updated["gcn_hidden_dim"] = int(trial["gcn_hidden_size"])
    updated["lstm_hidden_dim"] = int(trial["lstm_hidden_size"])
    updated["dropout"] = float(trial["dropout"])
    updated["batch_size"] = int(trial["batch_size"])
    updated["learning_rate"] = float(trial["learning_rate"])
    updated["epochs"] = int(max_epochs)
    updated["output_steps"] = 1
    updated["n_his"] = 12
    updated["input_channels"] = 1
    updated["output_channels"] = 1
    updated["output_activation"] = "linear"
    updated["lr_factor"] = 0.5
    return updated


def _train_trial(
    trial_config: Mapping[str, Any],
    pack: Mapping[str, Any],
    *,
    device: torch.device,
    trial_dir: Path,
    patience: int,
) -> dict[str, Any]:
    seed_everything(int(trial_config["seed"]))
    model = build_model(trial_config, pack["adjacency"]).to(device)
    if int(model.output_window) != 1 or int(model.input_window) != 12:
        raise ReimplementationError("search trial left the 12-to-1 task")
    optimizer = build_optimizer(model, trial_config)
    scheduler = build_scheduler(optimizer, trial_config)
    criterion = MAERawLoss(float(pack["mean_y"]), float(pack["std_y"]))
    best_mae = float("inf")
    best_epoch = -1
    best_state = None
    wait = 0
    history: list[dict[str, float]] = []
    for epoch in range(int(trial_config["epochs"])):
        model.train()
        losses: list[float] = []
        for batch in pack["loaders"]["train"]:
            batch = _move_batch(batch, device)
            prediction = model(batch["x"])
            if prediction.shape != batch["y"].shape:
                raise ReimplementationError(
                    f"prediction {tuple(prediction.shape)} != target {tuple(batch['y'].shape)}"
                )
            loss = criterion(prediction, batch["y_raw"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_mae = _validation_mae(
            model,
            pack["loaders"]["validation"],
            mean_y=float(pack["mean_y"]),
            std_y=float(pack["std_y"]),
            device=device,
        )
        scheduler.step(val_mae)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mae": val_mae})
        print(
            f"trial={int(trial_config['trial_id']):03d} epoch={epoch} train_mae={history[-1]['train_loss']:.4f} val_mae={val_mae:.4f}",
            flush=True,
        )
        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch
            wait = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            wait += 1
            if patience > 0 and wait >= patience:
                break
    if best_state is None:
        raise ReimplementationError("trial produced no validation checkpoint")
    trial_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(trial_dir / "history.json", history)
    payload = {
        "model_state_dict": best_state,
        "trial_id": int(trial_config["trial_id"]),
        "best_epoch": best_epoch,
        "best_validation_mae": best_mae,
        "config": {key: value for key, value in trial_config.items() if key != "trial_id"},
        "code_version": CODE_VERSION,
        "model_label": MODEL_LABEL,
    }
    checkpoint = trial_dir / "best_checkpoint.pt"
    temporary = checkpoint.with_name(".best_checkpoint.pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint)
    reloaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(reloaded["model_state_dict"])
    return {
        "parameter_count": model.parameter_count(),
        "best_epoch": best_epoch,
        "best_validation_mae": best_mae,
        "epochs_ran": len(history),
    }


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _train_trial_isolated(
    trial_config: Mapping[str, Any],
    pack: Mapping[str, Any],
    *,
    device: torch.device,
    trial_dir: Path,
    patience: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """Run one trial in this frame so an OOM traceback does not outlive it."""
    try:
        return (
            _train_trial(
                trial_config,
                pack,
                device=device,
                trial_dir=trial_dir,
                patience=patience,
            ),
            None,
        )
    except Exception:
        return None, traceback.format_exc()


def _row_from_trial(trial: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    row = {key: trial.get(key, "") for key in TRIAL_FIELDS}
    row.update(extra)
    return row


def run_search(args: argparse.Namespace) -> None:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if str(args.dataset) != "xuancheng26":
        raise ReimplementationError(f"only the prepared xuancheng26 windows are supported, got {args.dataset}")
    config["target_mode"] = str(config.get("target_mode", "last-observed-step"))
    assert_task_config(config)
    if str(args.device).startswith("cpu") and not args.dry_run:
        raise ReimplementationError("refusing to train the search on CPU; pass --device cuda on the selected GPU")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise ReimplementationError("requested CUDA device is unavailable; refusing to use another device")
    output_root = resolve_training_output(PROJECT_ROOT, config["output_root"], model_name="s2s_gcn_lstm")
    directory = search_directory(output_root, args.penetration)
    directory.mkdir(parents=True, exist_ok=True)
    plan = [sample_trial(int(args.seed), trial_id) for trial_id in range(int(args.num_trials))]
    search_config = {
        "model_label": MODEL_LABEL,
        "code_version": CODE_VERSION,
        "dataset": args.dataset,
        "penetration": rate_tag(args.penetration),
        "seed": int(args.seed),
        "model_init_seed": int(config["seed"]),
        "num_trials": int(args.num_trials),
        "max_epochs": int(args.max_epochs),
        "early_stopping_patience": int(args.early_stopping_patience),
        "device": args.device,
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "sampler": "uniform_random_over_table_ii_discrete_sets",
        "learning_rate_values": list(LEARNING_RATES),
        "learning_rate_interpretation": (
            "Half-decade log10 grid. KerasTuner 1.1.2 cannot combine sampling=log with step=0.5, "
            "and a linear step of 0.5 does not fit inside [1e-4, 1e-2]. A log multiplier of 0.5 "
            "leaves the interval. See the module docstring."
        ),
        "uses_test_for_selection": False,
        "output_directory": directory.as_posix(),
    }
    atomic_write_json(directory / "search_config.json", search_config)
    if args.dry_run:
        _dry_run(config, plan, directory, project_root=PROJECT_ROOT, penetration=args.penetration, device_name=args.device)
        print(json.dumps({"status": "dry_run_ok", "directory": directory.as_posix(), "num_trials": len(plan)}, indent=2))
        return
    device = resolve_device(args.device)
    if device.type == "cuda" and torch.cuda.device_count() < 1:
        raise ReimplementationError("CUDA was requested but no device is visible")
    data_root = resolve_path(PROJECT_ROOT, config["data_root"])
    manifest = load_json(data_root / "split_manifest.json")
    normalization = load_json(data_root / "normalization.json")
    if str(manifest.get("target_mode")) != "last-observed-step":
        raise ReimplementationError("split manifest is not last-observed-step reconstruction")
    if bool(normalization.get("used_test_to_fit")) or bool(normalization.get("used_validation_to_fit")):
        raise ReimplementationError("normalization.json was not fit on training only")
    mean_y, std_y = read_target_scaler(normalization)
    counts = expected_split_counts(manifest)
    from reimplementation.s2s_gcn_lstm.engine import _graph_payload

    graph = _graph_payload(config, project_root=PROJECT_ROOT)
    assert_feature_order(load_edge_ids(data_root / "node_mapping.csv"), graph["node_ids"])
    adjacency = graph["adjacency"]
    tag = rate_tag(args.penetration)
    datasets = {
        split: ROnlyNPZDataset(
            data_root / tag / f"{split}.npz",
            n_his=12,
            num_nodes=int(config["num_nodes"]),
            output_steps=1,
            expected_count=int(counts[split]),
        )
        for split in ("train", "validation")
    }
    trials_path = directory / "trials.csv"
    existing = _read_trials(trials_path)
    by_id = {int(row["trial_id"]): row for row in existing}
    for trial in plan:
        trial_id = int(trial["trial_id"])
        previous = by_id.get(trial_id)
        if previous is not None and str(previous.get("status")) in FINISHED_STATUSES and args.resume:
            print(f"skip trial {trial_id:03d} status={previous['status']}", flush=True)
            continue
        if previous is not None and str(previous.get("status")) in FINISHED_STATUSES and not args.resume:
            raise ReimplementationError(f"{trials_path} already has trial {trial_id}; pass --resume")
        trial_config = _apply_trial(config, trial, max_epochs=int(args.max_epochs))
        trial_config["trial_id"] = trial_id
        trial_dir = directory / "trials" / f"trial_{trial_id:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(trial_dir / "config.json", {key: trial_config[key] for key in trial_config if key != "trial_id"})
        pack = {
            "adjacency": adjacency,
            "mean_y": mean_y,
            "std_y": std_y,
            "counts": counts,
            "loaders": {
                split: build_dataloader(
                    datasets[split],
                    batch_size=int(trial["batch_size"]),
                    shuffle=(split == "train"),
                    num_workers=0,
                    seed=int(config["seed"]),
                )
                for split in ("train", "validation")
            },
        }
        print(
            "start trial {trial_id:03d} gcn={gcn_hidden_size} lstm={lstm_hidden_size} "
            "layers={encoder_gcn_layers}/{encoder_lstm_layers}/{decoder_gcn_layers}/{decoder_lstm_layers} "
            "dropout={dropout} batch={batch_size} lr={learning_rate:.6g}".format(**trial),
            flush=True,
        )
        started = time.perf_counter()
        outcome, details = _train_trial_isolated(
            trial_config,
            pack,
            device=device,
            trial_dir=trial_dir,
            patience=int(args.early_stopping_patience),
        )
        _release_cuda()
        if details is None and outcome is not None:
            status = "completed"
            row = _row_from_trial(
                trial,
                parameter_count=outcome["parameter_count"],
                best_epoch=outcome["best_epoch"],
                best_validation_mae=f"{outcome['best_validation_mae']:.8f}",
                training_time_sec=f"{time.perf_counter() - started:.3f}",
                status=status,
                error="",
            )
        else:
            details = details or ""
            status = "failed_oom" if "out of memory" in details.lower() else "failed"
            row = _row_from_trial(
                trial,
                parameter_count="",
                best_epoch="",
                best_validation_mae="",
                training_time_sec=f"{time.perf_counter() - started:.3f}",
                status=status,
                error=details.strip().splitlines()[-1][:500] if details else "",
            )
            (trial_dir / "error.txt").write_text(details, encoding="utf-8")
        pack.clear()
        if status == "completed":
            print(
                f"done trial {trial_id:03d} val_mae={row['best_validation_mae']} params={row['parameter_count']}",
                flush=True,
            )
        else:
            print(f"trial {trial_id:03d} {status}: {row['error']}", flush=True)
            if status != "failed_oom":
                by_id[trial_id] = row
                ordered = [by_id[i] for i in range(int(args.num_trials)) if i in by_id]
                _write_trials(trials_path, ordered)
                _publish_best(directory, ordered)
                raise ReimplementationError(str(row["error"]))
        by_id[trial_id] = row
        ordered = [by_id[i] for i in range(int(args.num_trials)) if i in by_id]
        _write_trials(trials_path, ordered)
        _publish_best(directory, ordered)
        _release_cuda()
    print(json.dumps({"status": "search_finished", "directory": directory.as_posix()}, indent=2), flush=True)


def _dry_run(
    config: Mapping[str, Any],
    plan: list[dict[str, Any]],
    directory: Path,
    *,
    project_root: Path,
    penetration: str,
    device_name: str,
) -> None:
    from reimplementation.s2s_gcn_lstm.engine import _graph_payload

    graph = _graph_payload(dict(config), project_root=project_root)
    data_root = resolve_path(project_root, config["data_root"])
    manifest = load_json(data_root / "split_manifest.json")
    counts = expected_split_counts(manifest)
    tag = rate_tag(penetration)
    dataset = ROnlyNPZDataset(
        data_root / tag / "train.npz",
        n_his=12,
        num_nodes=int(config["num_nodes"]),
        output_steps=1,
        expected_count=int(counts["train"]),
    )
    sample = dataset[0]
    if tuple(sample["x"].shape) != (12, int(config["num_nodes"]), 1):
        raise ReimplementationError(f"dry-run input shape {tuple(sample['x'].shape)}")
    if tuple(sample["y"].shape) != (1, int(config["num_nodes"]), 1):
        raise ReimplementationError(f"dry-run target shape {tuple(sample['y'].shape)}")
    device = torch.device("cpu")
    if str(device_name).startswith("cuda"):
        if not torch.cuda.is_available():
            raise ReimplementationError("dry-run requested CUDA, but no CUDA device is visible")
        device = resolve_device(device_name)
    reports = []
    for trial in plan:
        trial_config = _apply_trial(dict(config), trial, max_epochs=1)
        print(
            "dry-run trial {trial_id:03d} layers={encoder_gcn_layers}/{encoder_lstm_layers}/{decoder_gcn_layers}/{decoder_lstm_layers} "
            "gcn={gcn_hidden_size} lstm={lstm_hidden_size}".format(**trial),
            flush=True,
        )
        model = build_model(trial_config, graph["adjacency"]).to(device)
        model.eval()
        with torch.no_grad():
            prediction, trace = model(sample["x"].unsqueeze(0).to(device), return_trace=True)
        if tuple(prediction.shape) != (1, 1, int(config["num_nodes"]), 1):
            raise ReimplementationError(f"dry-run prediction {tuple(prediction.shape)}")
        if int(trace["decoder_gcn_calls"]) != 1 or int(trace["encoder_gcn_steps"]) != 12:
            raise ReimplementationError(f"dry-run trace is not 12-to-1: {trace}")
        if len(model.encoder_gcn.layers) != int(trial["encoder_gcn_layers"]):
            raise ReimplementationError("encoder GCN depth does not match the sample")
        if len(model.decoder_gcn.layers) != int(trial["decoder_gcn_layers"]):
            raise ReimplementationError("decoder GCN depth does not match the sample")
        if int(model.encoder_lstm.num_layers) != int(trial["encoder_lstm_layers"]):
            raise ReimplementationError("encoder LSTM depth does not match the sample")
        if int(model.decoder_lstm.num_layers) != int(trial["decoder_lstm_layers"]):
            raise ReimplementationError("decoder LSTM depth does not match the sample")
        reports.append(
            {
                "trial_id": trial["trial_id"],
                "parameter_count": model.parameter_count(),
                "output_shape": list(prediction.shape),
                **{key: trial[key] for key in trial if key != "trial_id"},
            }
        )
        del model
    atomic_write_json(
        directory / "dry_run_report.json",
        {
            "status": "ok",
            "directory": directory.as_posix(),
            "train_samples": int(counts["train"]),
            "validation_samples": int(counts["validation"]),
            "did_not_open_test": True,
            "trials": reports,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random search for Seq2seq GCN-LSTM Table II ranges.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=str, default="xuancheng26")
    parser.add_argument("--penetration", type=str, default="p70")
    parser.add_argument("--num-trials", dest="num_trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-epochs", dest="max_epochs", type=int, default=200)
    parser.add_argument("--early-stopping-patience", dest="early_stopping_patience", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.num_trials) < 1:
        raise ReimplementationError("--num-trials must be positive")
    if int(args.max_epochs) < 1:
        raise ReimplementationError("--max-epochs must be positive")
    run_search(args)


if __name__ == "__main__":
    try:
        main()
    except ReimplementationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
