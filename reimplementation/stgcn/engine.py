"""Training and evaluation engine for R-only STGCN. Not a generic Trainer framework."""

from __future__ import annotations

import hashlib
import resource
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.optim import Adam, RMSprop
from torch.optim.lr_scheduler import StepLR

from reimplementation.common.data.r_only_npz_dataset import (
    ROnlyNPZDataset,
    SPLITS,
    build_dataloader,
    invert_target,
    rate_tag,
)
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.metrics.traffic_metrics import traffic_metrics
from reimplementation.common.utils.atomic_io import atomic_write_json, atomic_write_npz
from reimplementation.common.utils.checkpoint import (
    assert_checkpoint_compatible,
    load_checkpoint,
    save_checkpoint,
)
from reimplementation.common.utils.reproducibility import seed_everything
from reimplementation.common.utils.structured_logging import JsonlLogger
from reimplementation.stgcn.losses import STGCNPredictionLoss
from reimplementation.stgcn.model import CODE_VERSION, STGCN
from reimplementation.stgcn.validation import (
    validate_graph,
    validate_model_shapes,
    validate_prepared_data,
)


def atomic_write_csv_rows_impl(path: Path, fieldnames: list[str], rows: list[Mapping[str, Any]]) -> None:
    from reimplementation.common.utils.atomic_io import atomic_write_text

    import io
    import csv as csvlib

    buffer = io.StringIO()
    writer = csvlib.DictWriter(
        buffer,
        fieldnames=fieldnames,
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in fieldnames})
    atomic_write_text(path, buffer.getvalue())


# Late bind in case atomic_io grows a csv helper later.
atomic_write_csv_rows = atomic_write_csv_rows_impl


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, param in sorted(model.named_parameters(), key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        array = np.ascontiguousarray(param.detach().cpu().numpy())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _cpu_rss_bytes() -> float | None:
    try:
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if sys.platform != "darwin":
        rss *= 1024.0
    return rss


def build_model(config: Mapping[str, Any], cheb_kernel: np.ndarray) -> STGCN:
    dropout_p = float(config.get("dropout", 0.0))
    keep_prob = float(config.get("keep_prob", 1.0))
    if "dropout" not in config:
        dropout_p = 1.0 - keep_prob
    return STGCN(
        n_nodes=int(config["num_nodes"]),
        n_his=int(config["n_his"]),
        ks=int(config["Ks"]),
        kt=int(config["Kt"]),
        blocks=list(config["blocks"]),
        cheb_kernel=cheb_kernel,
        dropout_p=dropout_p,
        input_channels=int(config["input_channels"]),
        output_channels=int(config["output_channels"]),
        output_steps=int(config["output_steps"]),
    )


def build_optimizer(model: STGCN, config: Mapping[str, Any]):
    name = str(config.get("optimizer", "RMSProp")).lower()
    lr = float(config["learning_rate"])
    # Original RMSPropOptimizer uses decay=0.9 and epsilon=1e-10 and does not
    # add the collected L2 collection to the minimize() target.
    if name == "rmsprop":
        return RMSprop(
            model.parameters(),
            lr=lr,
            alpha=float(config.get("rmsprop_alpha", 0.9)),
            eps=float(config.get("rmsprop_eps", 1e-10)),
            weight_decay=0.0,
        )
    if name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=0.0)
    raise ReimplementationError(f'ERROR: optimizer "{name}" is not defined.')


def build_scheduler(optimizer, config: Mapping[str, Any], steps_per_epoch: int):
    """Match TF ``exponential_decay(..., decay_steps=5 * epoch_step, staircase=True)``.

    Original ``global_steps`` increases once per training batch, not per epoch.
    """
    decay_epochs = int(config.get("lr_decay_epochs", 5))
    return StepLR(
        optimizer,
        step_size=max(decay_epochs * int(steps_per_epoch), 1),
        gamma=float(config.get("lr_decay_rate", 0.7)),
    )


def build_dataloaders(
    data_root: Path,
    rate: str,
    config: Mapping[str, Any],
    split_counts: Mapping[str, int],
) -> dict[str, Any]:
    loaders = {}
    datasets = {}
    for split in SPLITS:
        dataset = ROnlyNPZDataset(
            data_root / rate / f"{split}.npz",
            n_his=int(config["n_his"]),
            num_nodes=int(config["num_nodes"]),
            expected_count=int(split_counts[split]),
        )
        datasets[split] = dataset
        loaders[split] = build_dataloader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=(split == "train"),
            num_workers=int(config.get("num_workers", 0)),
            seed=int(config["seed"]),
        )
    return {"loaders": loaders, "datasets": datasets}


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def train_one_epoch(
    model: STGCN,
    loader,
    optimizer,
    criterion: STGCNPredictionLoss,
    device: torch.device,
    apply_weight_decay: bool,
    scheduler=None,
    gradient_clip: float = 0.0,
) -> dict[str, float]:
    model.train()
    total_pred = 0.0
    total_reg = 0.0
    n_samples = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch["x"])
        if prediction.shape != batch["y"].shape:
            raise ReimplementationError(
                f"prediction {tuple(prediction.shape)} != target {tuple(batch['y'].shape)}"
            )
        pred_loss = criterion(prediction, batch["y"])
        reg_loss = model.regularization_loss()
        loss = pred_loss + reg_loss if apply_weight_decay else pred_loss
        loss.backward()
        if gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        batch_n = int(batch["x"].size(0))
        total_pred += float(pred_loss.detach().cpu())
        total_reg += float(reg_loss.detach().cpu())
        n_samples += batch_n
    return {
        "prediction_loss": total_pred / max(len(loader), 1),
        "regularization_loss": total_reg / max(len(loader), 1),
        "total_loss": (total_pred if not apply_weight_decay else total_pred + total_reg) / max(len(loader), 1),
        "train_sample_count": float(n_samples),
    }


@torch.no_grad()
def evaluate_one_epoch(
    model: STGCN,
    loader,
    criterion: STGCNPredictionLoss,
    device: torch.device,
    mean_y: float,
    std_y: float,
) -> dict[str, Any]:
    model.eval()
    predictions = []
    targets = []
    predictions_raw = []
    targets_raw = []
    days = []
    starts = []
    targets_slot = []
    sample_index = []
    total_pred = 0.0
    n_batches = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        pred = model(batch["x"])
        if pred.shape != batch["y"].shape:
            raise ReimplementationError(
                f"prediction {tuple(pred.shape)} != target {tuple(batch['y'].shape)}"
            )
        total_pred += float(criterion(pred, batch["y"]).cpu())
        n_batches += 1
        pred_cpu = pred.cpu().numpy()
        y_cpu = batch["y"].cpu().numpy()
        predictions.append(pred_cpu)
        targets.append(y_cpu)
        predictions_raw.append(np.asarray(invert_target(pred_cpu, mean_y, std_y)))
        targets_raw.append(batch["y_raw"].cpu().numpy())
        days.append(batch["day_index"].cpu().numpy())
        starts.append(batch["window_start_slot"].cpu().numpy())
        targets_slot.append(batch["target_slot"].cpu().numpy())
        sample_index.append(batch["sample_index"].cpu().numpy())
    y_pred = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(targets, axis=0)
    y_pred_raw = np.concatenate(predictions_raw, axis=0)
    y_true_raw = np.concatenate(targets_raw, axis=0)
    packed = {
        "y_pred_normalized": y_pred.astype(np.float32),
        "y_true_normalized": y_true.astype(np.float32),
        "y_pred_raw": y_pred_raw.astype(np.float32),
        "y_true_raw": y_true_raw.astype(np.float32),
        "day_index": np.concatenate(days, axis=0),
        "window_start_slot": np.concatenate(starts, axis=0),
        "target_slot": np.concatenate(targets_slot, axis=0),
        "sample_index": np.concatenate(sample_index, axis=0),
    }
    order = np.argsort(packed["sample_index"], kind="stable")
    packed = {key: value[order] for key, value in packed.items()}
    metrics = traffic_metrics(packed["y_pred_raw"], packed["y_true_raw"], clip_negative=False)
    metrics["prediction_loss"] = total_pred / max(n_batches, 1)
    packed["metrics"] = metrics
    return packed


def predict(model: STGCN, loader, device: torch.device, mean_y: float, std_y: float) -> dict[str, Any]:
    criterion = STGCNPredictionLoss()
    return evaluate_one_epoch(model, loader, criterion, device, mean_y, std_y)


def _gpu_stats(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"gpu_memory_allocated": None, "gpu_memory_reserved": None}
    return {
        "gpu_memory_allocated": float(torch.cuda.memory_allocated(device)),
        "gpu_memory_reserved": float(torch.cuda.memory_reserved(device)),
    }


def _process_stats(device: torch.device) -> dict[str, float | None]:
    stats = _gpu_stats(device)
    stats["cpu_memory_rss_bytes"] = _cpu_rss_bytes()
    return stats


def _write_split_outputs(
    output_dir: Path,
    split: str,
    result: Mapping[str, Any],
    *,
    write_tables: bool = False,
    node_ids: list[str] | None = None,
) -> None:
    metrics = dict(result["metrics"])
    atomic_write_json(output_dir / f"{split}_metrics.json", metrics)
    atomic_write_npz(
        output_dir / f"{split}_predictions.npz",
        y_pred_normalized=result["y_pred_normalized"],
        y_pred_raw=result["y_pred_raw"],
        y_true_normalized=result["y_true_normalized"],
        y_true_raw=result["y_true_raw"],
        day_index=result["day_index"],
        window_start_slot=result["window_start_slot"],
        target_slot=result["target_slot"],
        sample_index=result["sample_index"],
    )
    if not write_tables:
        return
    node_rows = []
    pred = result["y_pred_raw"]
    true = result["y_true_raw"]
    n_nodes = pred.shape[2]
    for node in range(n_nodes):
        stats = traffic_metrics(pred[:, :, node, :], true[:, :, node, :], clip_negative=False)
        stats["node_index"] = node
        stats["edge_id"] = node_ids[node] if node_ids is not None and node < len(node_ids) else ""
        node_rows.append(stats)
    atomic_write_csv_rows(
        output_dir / "node_metrics.csv",
        [
            "node_index",
            "edge_id",
            "mae",
            "rmse",
            "mape_nonzero",
            "wape",
            "nonzero_target_count",
            "zero_target_count",
            "zero_target_fraction",
            "negative_prediction_count",
            "negative_prediction_fraction",
            "minimum_prediction",
        ],
        node_rows,
    )
    day_rows = []
    days = result["day_index"]
    for day in sorted(set(int(item) for item in days.tolist())):
        mask = days == day
        stats = traffic_metrics(pred[mask], true[mask], clip_negative=False)
        stats["day_index"] = day
        day_rows.append(stats)
    atomic_write_csv_rows(
        output_dir / "day_metrics.csv",
        [
            "day_index",
            "mae",
            "rmse",
            "mape_nonzero",
            "wape",
            "nonzero_target_count",
            "zero_target_count",
            "zero_target_fraction",
            "negative_prediction_count",
            "negative_prediction_fraction",
            "minimum_prediction",
        ],
        day_rows,
    )


def run_p70_smoke(
    config: Mapping[str, Any],
    *,
    data_root: Path,
    graph_info: Mapping[str, Any],
    data_info: Mapping[str, Any],
    logger: JsonlLogger,
) -> dict[str, Any]:
    """One p70 training batch, then the model is discarded."""
    seed = int(config["seed"])
    seed_everything(seed)
    device = resolve_device(str(config.get("device", "auto")))
    loaders_pack = build_dataloaders(data_root, "p70", config, data_info["sample_counts"])
    model = build_model(config, graph_info["cheb_kernel"]).to(device)
    init_hash = parameter_sha256(model)
    shape_info = validate_model_shapes(model, batch_size=min(2, int(config["batch_size"])))
    optimizer = build_optimizer(model, config)
    criterion = STGCNPredictionLoss()
    batch = _move_batch(next(iter(loaders_pack["loaders"]["train"])), device)
    x = batch["x"]
    y = batch["y"]
    if tuple(x.shape[1:]) != (int(config["n_his"]), int(config["num_nodes"]), int(config["input_channels"])):
        raise ReimplementationError(f"smoke input shape {tuple(x.shape)}")
    model.train()
    pred, trace = model(x, return_trace=True)
    if pred.shape != y.shape:
        raise ReimplementationError(f"smoke prediction {tuple(pred.shape)} != target {tuple(y.shape)}")
    if not torch.isfinite(pred).all():
        raise ReimplementationError("smoke prediction is not finite")
    loss = criterion(pred, y)
    if not torch.isfinite(loss):
        raise ReimplementationError(f"smoke loss is not finite: {loss}")
    before = {name: tensor.detach().cpu().clone() for name, tensor in model.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    theta_grads = []
    for name, tensor in model.named_parameters():
        if tensor.grad is None or not torch.isfinite(tensor.grad).all():
            raise ReimplementationError(f"smoke missing or non-finite grad: {name}")
        if "spatial.theta" in name:
            theta_grads.append(float(tensor.grad.abs().max().cpu()))
    if not theta_grads or max(theta_grads) <= 0.0:
        raise ReimplementationError("graph convolution theta did not receive a non-zero gradient")
    if model.cheb_kernel.requires_grad or model.cheb_kernel.grad is not None:
        raise ReimplementationError("Chebyshev kernel must not participate in backprop as a parameter")
    optimizer.step()
    changed = 0
    for name, tensor in model.named_parameters():
        if not torch.equal(tensor.detach().cpu(), before[name]):
            changed += 1
    if changed == 0:
        raise ReimplementationError("optimizer step did not change parameters")
    payload = {
        "status": "ok",
        "penetration_rate": "p70",
        "batch_size": int(x.size(0)),
        "input_shape": list(x.shape),
        "target_shape": list(y.shape),
        "prediction_shape": list(pred.shape),
        "trace": trace,
        "prediction_loss": float(loss.detach().cpu()),
        "regularization_loss": float(model.regularization_loss().detach().cpu()),
        "parameter_count": model.parameter_count(),
        "init_parameter_sha256": init_hash,
        "nonzero_theta_grad_max": max(theta_grads),
        "parameters_changed": changed,
        "discarded_after_smoke": True,
        "shape_info": shape_info,
        **_process_stats(device),
    }
    logger.log(
        {
            "stage": "smoke_testing",
            "model_name": "stgcn",
            "penetration_rate": "p70",
            "seed": seed,
            "status": "ok",
            "prediction_loss": payload["prediction_loss"],
            **_process_stats(device),
        }
    )
    print(
        f"[stgcn] stage=smoke_testing rate=p70 loss={payload['prediction_loss']:.6f} status=ok",
        flush=True,
    )
    del model
    del optimizer
    del loaders_pack
    return payload


def run_single_rate(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    rate: str,
    data_root: Path,
    output_root: Path,
    graph_info: Mapping[str, Any],
    data_info: Mapping[str, Any],
    logger: JsonlLogger,
    resume_path: Path | None = None,
    smoke_only: bool = False,
) -> dict[str, Any]:
    tag = rate_tag(rate)
    seed = int(config["seed"])
    if bool(config.get("amp")):
        raise ReimplementationError("AMP is not implemented in this faithful STGCN port")
    seed_everything(seed)
    device = resolve_device(str(config.get("device", "auto")))
    rate_dir = output_root / tag / f"seed_{seed}"
    rate_dir.mkdir(parents=True, exist_ok=True)
    loaders_pack = build_dataloaders(data_root, tag, config, data_info["sample_counts"])
    model = build_model(config, graph_info["cheb_kernel"]).to(device)
    init_hash = parameter_sha256(model)
    shape_info = validate_model_shapes(model, batch_size=min(2, int(config["batch_size"])))
    optimizer = build_optimizer(model, config)
    train_len = data_info["sample_counts"]["train"]
    epoch_step = int(np.ceil(train_len / float(config["batch_size"])))
    scheduler = build_scheduler(optimizer, config, epoch_step)
    criterion = STGCNPredictionLoss()
    apply_decay = bool(config.get("weight_decay_in_optimizer", False))
    mean_y = float(data_info["mean_y_full"])
    std_y = float(data_info["std_y_full"])
    train_hash = data_info["file_hashes"][f"{tag}/train.npz"]
    start_epoch = 0
    best_metric = float("inf")
    best_epoch = -1
    if resume_path is not None:
        payload = load_checkpoint(resume_path, map_location=device)
        assert_checkpoint_compatible(
            payload,
            penetration_rate=tag,
            graph_sha256=str(graph_info["graph_sha256"]),
            data_file_sha256=train_hash,
            num_nodes=int(config["num_nodes"]),
            n_his=int(config["n_his"]),
            output_steps=int(config["output_steps"]),
            input_channels=int(config["input_channels"]),
            output_channels=int(config["output_channels"]),
        )
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if payload["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch = int(payload["epoch"]) + 1
        best_metric = float(payload["best_metric"])
        best_epoch = int(payload["epoch"])

    def checkpoint_payload(epoch: int, metric: float) -> dict[str, Any]:
        return {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": metric,
            "config": dict(config),
            "penetration_rate": tag,
            "seed": seed,
            "model_parameter_count": model.parameter_count(),
            "graph_sha256": graph_info["graph_sha256"],
            "data_file_sha256": train_hash,
            "normalization_sha256": data_info["normalization_sha256"],
            "code_version": CODE_VERSION,
        }

    if smoke_only:
        batch = _move_batch(next(iter(loaders_pack["loaders"]["train"])), device)
        model.train()
        pred = model(batch["x"])
        loss = criterion(pred, batch["y"])
        loss.backward()
        logger.log(
            {
                "stage": "smoke_testing",
                "model_name": "stgcn",
                "penetration_rate": tag,
                "seed": seed,
                "status": "ok",
                "prediction_loss": float(loss.detach().cpu()),
                **_process_stats(device),
            }
        )
        return {
            "status": "smoke_ok",
            "trace": shape_info["trace"],
            "parameter_count": model.parameter_count(),
            "init_parameter_sha256": init_hash,
        }

    history_rows: list[dict[str, Any]] = []
    patience = int(config.get("early_stopping_patience", 0))
    epochs_without_improve = 0
    epochs = int(config["epochs"])
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        train_stats = train_one_epoch(
            model,
            loaders_pack["loaders"]["train"],
            optimizer,
            criterion,
            device,
            apply_decay,
            scheduler=scheduler,
            gradient_clip=float(config.get("gradient_clip", 0.0)),
        )
        val_result = evaluate_one_epoch(
            model,
            loaders_pack["loaders"]["validation"],
            criterion,
            device,
            mean_y,
            std_y,
        )
        val_mae = float(val_result["metrics"]["mae"])
        improved = val_mae < best_metric
        if improved:
            best_metric = val_mae
            best_epoch = epoch
            epochs_without_improve = 0
            save_checkpoint(rate_dir / "best_checkpoint.pt", checkpoint_payload(epoch, best_metric))
        else:
            epochs_without_improve += 1
        save_checkpoint(rate_dir / "last_checkpoint.pt", checkpoint_payload(epoch, best_metric))
        lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_prediction_loss": train_stats["prediction_loss"],
            "train_regularization_loss": train_stats["regularization_loss"],
            "train_total_loss": train_stats["total_loss"],
            "validation_mae_raw": val_mae,
            "validation_rmse_raw": float(val_result["metrics"]["rmse"]),
            "validation_mape_nonzero": float(val_result["metrics"]["mape_nonzero"]),
            "validation_wape": float(val_result["metrics"]["wape"]),
            "learning_rate": lr,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "elapsed_seconds": time.time() - epoch_start,
        }
        history_rows.append(row)
        logger.log(
            {
                "stage": "training",
                "model_name": "stgcn",
                "penetration_rate": tag,
                "seed": seed,
                "epoch": epoch,
                "status": "ok",
                "device": str(device),
                **train_stats,
                "validation_mae_raw": val_mae,
                "validation_rmse_raw": row["validation_rmse_raw"],
                "learning_rate": lr,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "elapsed_seconds": row["elapsed_seconds"],
                **_process_stats(device),
            }
        )
        print(
            (
                f"[stgcn] stage=training rate={tag} epoch={epoch}/{epochs - 1} "
                f"train_loss={train_stats['prediction_loss']:.6f} "
                f"val_mae={val_mae:.6f} best_epoch={best_epoch} "
                f"elapsed={row['elapsed_seconds']:.1f}s"
            ),
            flush=True,
        )
        if patience > 0 and epochs_without_improve >= patience:
            break

    atomic_write_csv_rows(
        rate_dir / "training_history.csv",
        list(history_rows[0].keys()) if history_rows else ["epoch"],
        history_rows,
    )
    atomic_write_json(rate_dir / "training_history.json", history_rows)
    best_path = rate_dir / "best_checkpoint.pt"
    if best_path.is_file():
        payload = load_checkpoint(best_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
    val_best = evaluate_one_epoch(
        model, loaders_pack["loaders"]["validation"], criterion, device, mean_y, std_y
    )
    _write_split_outputs(
        rate_dir,
        "validation",
        val_best,
        write_tables=False,
        node_ids=list(data_info["node_ids"]),
    )
    test_best = evaluate_one_epoch(
        model, loaders_pack["loaders"]["test"], criterion, device, mean_y, std_y
    )
    dataset_test = loaders_pack["datasets"]["test"]
    if not np.allclose(test_best["y_true_raw"], dataset_test.y_raw, atol=1e-5):
        raise ReimplementationError("test y_true_raw does not match the original NPZ order")
    recovered = invert_target(test_best["y_pred_normalized"], mean_y, std_y)
    if np.max(np.abs(np.asarray(recovered) - test_best["y_pred_raw"])) > 1e-4:
        raise ReimplementationError("test denormalization does not recover y_pred_raw")
    _write_split_outputs(
        rate_dir,
        "test",
        test_best,
        write_tables=True,
        node_ids=list(data_info["node_ids"]),
    )
    atomic_write_json(
        rate_dir / "run_metadata.json",
        {
            "penetration_rate": tag,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "parameter_count": model.parameter_count(),
            "init_parameter_sha256": init_hash,
            "code_version": CODE_VERSION,
            "graph_sha256": graph_info["graph_sha256"],
            "data_file_sha256": train_hash,
            "target_mode": "last-observed-step",
            "did_not_use_test_for_selection": True,
            "checkpoint_monitor": config.get("checkpoint_monitor", "validation_mae_raw"),
        },
    )
    atomic_write_json(
        rate_dir / "run_validation.json",
        {
            "status": "ok",
            "remaining_time": model.remaining_time,
            "shape_trace": shape_info["trace"],
        },
    )
    return {
        "status": "trained",
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "parameter_count": model.parameter_count(),
        "init_parameter_sha256": init_hash,
        "validation_mae_raw": float(val_best["metrics"]["mae"]),
        "validation_rmse_raw": float(val_best["metrics"]["rmse"]),
        "validation_mape_nonzero": float(val_best["metrics"]["mape_nonzero"]),
        "validation_wape": float(val_best["metrics"]["wape"]),
        "test_mae_raw": float(test_best["metrics"]["mae"]),
        "test_rmse_raw": float(test_best["metrics"]["rmse"]),
        "test_mape_nonzero": float(test_best["metrics"]["mape_nonzero"]),
        "test_wape": float(test_best["metrics"]["wape"]),
        "output_dir": rate_dir,
    }


def run_all_rates(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    rates: Iterable[str],
    data_root: Path,
    output_root: Path,
    adjacency_path: Path,
    r_nodes_path: Path,
    logger: JsonlLogger,
    smoke_only: bool = False,
    resume: Path | None = None,
) -> dict[str, Any]:
    rate_list = [rate_tag(item) for item in rates]
    if resume is not None and len(rate_list) > 1:
        raise ReimplementationError("use --resume together with --rate; do not share checkpoints across penetrations")
    logger.log({"stage": "scanning", "model_name": "stgcn", "status": "ok"})
    data_info = validate_prepared_data(
        data_root,
        n_his=int(config["n_his"]),
        num_nodes=int(config["num_nodes"]),
        rates=rate_list,
    )
    logger.log({"stage": "validating_inputs", "model_name": "stgcn", "status": "ok"})
    graph_info = validate_graph(
        adjacency_path,
        int(config["Ks"]),
        int(config["num_nodes"]),
        data_info["node_ids"],
        r_nodes_path,
    )
    logger.log(
        {
            "stage": "building_graph",
            "model_name": "stgcn",
            "status": "ok",
            "graph_sha256": graph_info["graph_sha256"],
        }
    )
    data_preview = {key: value for key, value in data_info.items() if key != "node_ids"}
    graph_preview = {
        "graph_sha256": graph_info["graph_sha256"],
        "counts": graph_info["counts"],
        "did_not_reapply_gaussian": graph_info["did_not_reapply_gaussian"],
        "did_not_add_self_loops": graph_info["did_not_add_self_loops"],
        "t0_is_identity": True,
        "ks": int(config["Ks"]),
        "kernel_shape": [int(config["num_nodes"]), int(config["Ks"]) * int(config["num_nodes"])],
        "weight_min_nonzero": graph_info.get("weight_min_nonzero"),
        "weight_median_nonzero": graph_info.get("weight_median_nonzero"),
        "weight_max": graph_info.get("weight_max"),
        "sparse_matches_dense": graph_info.get("sparse_matches_dense"),
    }
    atomic_write_json(output_root / "data_runtime_validation.json", data_preview)
    atomic_write_json(output_root / "graph_runtime_validation.json", graph_preview)
    print(
        (
            f"[stgcn] stage=building_graph nodes={config['num_nodes']} "
            f"sha256={graph_info['graph_sha256'][:12]} weight_min={graph_info.get('weight_min_nonzero')}"
        ),
        flush=True,
    )
    smoke_result = None
    if resume is None:
        smoke_result = run_p70_smoke(
            config,
            data_root=data_root,
            graph_info=graph_info,
            data_info=data_info,
            logger=logger,
        )
        seed_everything(int(config["seed"]))
        atomic_write_json(output_root / "smoke_test.json", smoke_result)
        print("[stgcn] stage=smoke_testing discarded_model=yes wrote smoke_test.json", flush=True)
    if smoke_only:
        graph_summary = {
            "graph_sha256": graph_info["graph_sha256"],
            "counts": graph_info["counts"],
            "did_not_reapply_gaussian": graph_info["did_not_reapply_gaussian"],
            "did_not_add_self_loops": graph_info["did_not_add_self_loops"],
            "t0_is_identity": True,
            "ks": int(config["Ks"]),
            "kernel_shape": [int(config["num_nodes"]), int(config["Ks"]) * int(config["num_nodes"])],
            "weight_min_nonzero": graph_info.get("weight_min_nonzero"),
            "weight_median_nonzero": graph_info.get("weight_median_nonzero"),
            "weight_max": graph_info.get("weight_max"),
            "sparse_matches_dense": graph_info.get("sparse_matches_dense"),
        }
        data_summary = {key: value for key, value in data_info.items() if key != "node_ids"}
        return {
            "rates": [],
            "data_info": data_summary,
            "graph_info": graph_summary,
            "smoke": smoke_result,
        }
    summaries = []
    init_hashes = []
    for rate in rate_list:
        print(f"[stgcn] stage=training_start rate={rate_tag(rate)} seed={config['seed']}", flush=True)
        result = run_single_rate(
            config,
            project_root=project_root,
            rate=rate,
            data_root=data_root,
            output_root=output_root,
            graph_info=graph_info,
            data_info=data_info,
            logger=logger,
            resume_path=resume if resume is not None else None,
            smoke_only=False,
        )
        init_hashes.append(result.get("init_parameter_sha256"))
        summaries.append({"rate": rate_tag(rate), **{k: v for k, v in result.items() if k != "output_dir"}})
        print(
            (
                f"[stgcn] stage=rate_completed rate={rate_tag(rate)} "
                f"best_epoch={result.get('best_epoch')} val_mae={result.get('validation_mae_raw')} "
                f"test_mae={result.get('test_mae_raw')}"
            ),
            flush=True,
        )
    unique_hashes = set(item for item in init_hashes if item)
    if len(unique_hashes) > 1:
        raise ReimplementationError(
            f"initial parameter SHA256 differs across rates: {init_hashes}"
        )
    graph_summary = {
        "graph_sha256": graph_info["graph_sha256"],
        "counts": graph_info["counts"],
        "did_not_reapply_gaussian": graph_info["did_not_reapply_gaussian"],
        "did_not_add_self_loops": graph_info["did_not_add_self_loops"],
        "t0_is_identity": True,
        "ks": int(config["Ks"]),
        "kernel_shape": [int(config["num_nodes"]), int(config["Ks"]) * int(config["num_nodes"])],
        "weight_min_nonzero": graph_info.get("weight_min_nonzero"),
        "weight_median_nonzero": graph_info.get("weight_median_nonzero"),
        "weight_max": graph_info.get("weight_max"),
        "sparse_matches_dense": graph_info.get("sparse_matches_dense"),
        "init_parameter_sha256": next(iter(unique_hashes)) if unique_hashes else None,
        "init_parameter_sha256_identical_across_rates": True,
    }
    data_summary = {key: value for key, value in data_info.items() if key != "node_ids"}
    return {
        "rates": summaries,
        "data_info": data_summary,
        "graph_info": graph_summary,
        "smoke": smoke_result,
    }
