"""Training and evaluation engine for R-only STSGCN.

Official epoch training is implemented for a later run. This session only
uses ``run_p70_smoke`` (one training batch) and never writes the future
experiment directory.
"""

from __future__ import annotations

import hashlib
import resource
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.optim import Adam

from reimplementation.common.data.r_only_npz_dataset import (
    ROnlyNPZDataset,
    SPLITS,
    build_dataloader,
    expected_split_counts,
    invert_target,
    load_json,
    rate_tag,
    read_target_scaler,
)
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.metrics.traffic_metrics import traffic_metrics
from reimplementation.common.utils.atomic_io import atomic_write_json, atomic_write_npz
from reimplementation.common.utils.checkpoint import (
    assert_checkpoint_compatible,
    load_checkpoint,
    save_checkpoint,
)
from reimplementation.common.utils.hashing import sha256_file
from reimplementation.common.utils.reproducibility import seed_everything
from reimplementation.common.utils.structured_logging import JsonlLogger
from reimplementation.stgcn.engine import atomic_write_csv_rows, resolve_device, resolve_path
from reimplementation.stsgcn.losses import HuberRawLoss
from reimplementation.stsgcn.model.stsgcn import CODE_VERSION, STSGCN
from reimplementation.stsgcn.validation import (
    load_node_ids,
    validate_model_shapes,
    validate_prepared_data,
    validate_stsgcn_graph,
    validate_train_npz_only,
)


class PolyLRScheduler:
    """MXNet 1.4 ``PolyScheduler(pwr=2)`` with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        max_update: int,
        power: float = 2.0,
        warmup_steps: int = 0,
        final_lr: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.max_update = int(max_update)
        self.power = float(power)
        self.warmup_steps = int(warmup_steps)
        self.final_lr = float(final_lr)
        self.max_steps = max(self.max_update - self.warmup_steps, 1)
        self.num_update = 0

    def _lr(self, num_update: int) -> float:
        base = self.base_lrs[0]
        if self.warmup_steps > 0 and num_update < self.warmup_steps:
            return self.final_lr + (base - 0.0) * float(num_update) / float(self.warmup_steps)
        if num_update <= self.max_update:
            progress = 1.0 - float(num_update - self.warmup_steps) / float(self.max_steps)
            return self.final_lr + (base - self.final_lr) * (max(progress, 0.0) ** self.power)
        return self.final_lr

    def step(self) -> None:
        self.num_update += 1
        lr = self._lr(self.num_update)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def state_dict(self) -> dict[str, Any]:
        return {"num_update": self.num_update}

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        self.num_update = int(payload["num_update"])
        if self.num_update > 0:
            lr = self._lr(self.num_update)
            for group in self.optimizer.param_groups:
                group["lr"] = lr


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


def _process_stats(device: torch.device) -> dict[str, float | None]:
    stats: dict[str, float | None] = {"cpu_memory_rss_bytes": _cpu_rss_bytes()}
    if device.type == "cuda" and torch.cuda.is_available():
        stats["gpu_memory_allocated"] = float(torch.cuda.memory_allocated(device))
        stats["gpu_memory_reserved"] = float(torch.cuda.memory_reserved(device))
    else:
        stats["gpu_memory_allocated"] = None
        stats["gpu_memory_reserved"] = None
    return stats


def build_model(config: Mapping[str, Any], spatial_adj: np.ndarray) -> STSGCN:
    return STSGCN(
        spatial_adj,
        seq_len=int(config["seq_len"]),
        num_nodes=int(config["num_nodes"]),
        input_channels=int(config["input_channels"]),
        first_layer_embedding_size=int(config.get("first_layer_embedding_size", 64)),
        filters=list(config.get("filters", [[64, 64, 64]] * 4)),
        module_type=str(config.get("module_type", "individual")),
        activation=str(config.get("act_type", config.get("activation", "GLU"))),
        use_mask=bool(config.get("use_mask", True)),
        temporal_emb=bool(config.get("temporal_emb", config.get("use_temporal_embedding", True))),
        spatial_emb=bool(config.get("spatial_emb", config.get("use_spatial_embedding", True))),
        horizon=int(config.get("horizon", 1)),
        output_hidden=int(config.get("output_hidden", 128)),
        xavier_magnitude=float(config.get("xavier_magnitude", 0.0003)),
    )


def build_optimizer(model: STSGCN, config: Mapping[str, Any]) -> Adam:
    if str(config.get("optimizer", "adam")).lower() != "adam":
        raise ReimplementationError("original STSGCN training uses Adam")
    return Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )


def build_scheduler(
    optimizer: Adam,
    config: Mapping[str, Any],
    *,
    steps_per_epoch: int,
) -> PolyLRScheduler:
    epochs = int(config["epochs"])
    max_update = max(int(steps_per_epoch * epochs * float(config.get("max_update_factor", 1))), 1)
    warmup = max(int(steps_per_epoch), 0)
    return PolyLRScheduler(
        optimizer,
        max_update=max_update,
        power=float(config.get("poly_power", 2.0)),
        warmup_steps=warmup,
        final_lr=float(config.get("final_lr", 0.0)),
    )


def build_train_dataloader(
    data_root: Path,
    rate: str,
    config: Mapping[str, Any],
    split_counts: Mapping[str, int],
):
    tag = rate_tag(rate)
    dataset = ROnlyNPZDataset(
        data_root / tag / "train.npz",
        n_his=int(config["seq_len"]),
        num_nodes=int(config["num_nodes"]),
        expected_count=int(split_counts["train"]),
    )
    loader = build_dataloader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
        seed=int(config["seed"]),
    )
    return {"loader": loader, "dataset": dataset}


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
            data_root / rate_tag(rate) / f"{split}.npz",
            n_his=int(config["seq_len"]),
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
    model: STSGCN,
    loader,
    optimizer: Adam,
    criterion: HuberRawLoss,
    device: torch.device,
    scheduler: PolyLRScheduler | None = None,
) -> dict[str, float]:
    model.train()
    total = 0.0
    n_samples = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch["x"])
        if prediction.shape != batch["y"].shape:
            raise ReimplementationError(
                f"prediction {tuple(prediction.shape)} != target {tuple(batch['y'].shape)}"
            )
        loss = criterion(prediction, batch["y_raw"])
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += float(loss.detach().cpu())
        n_samples += int(batch["x"].size(0))
    return {
        "training_loss": total / max(len(loader), 1),
        "train_sample_count": float(n_samples),
    }


@torch.no_grad()
def evaluate_one_epoch(
    model: STSGCN,
    loader,
    criterion: HuberRawLoss,
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
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        pred = model(batch["x"])
        if pred.shape != batch["y"].shape:
            raise ReimplementationError(
                f"prediction {tuple(pred.shape)} != target {tuple(batch['y'].shape)}"
            )
        total_loss += float(criterion(pred, batch["y_raw"]).cpu())
        n_batches += 1
        pred_cpu = pred.cpu().numpy()
        predictions.append(pred_cpu)
        targets.append(batch["y"].cpu().numpy())
        predictions_raw.append(np.asarray(invert_target(pred_cpu, mean_y, std_y)))
        targets_raw.append(batch["y_raw"].cpu().numpy())
        days.append(batch["day_index"].cpu().numpy())
        starts.append(batch["window_start_slot"].cpu().numpy())
        targets_slot.append(batch["target_slot"].cpu().numpy())
        sample_index.append(batch["sample_index"].cpu().numpy())
    packed = {
        "y_pred_normalized": np.concatenate(predictions, axis=0).astype(np.float32),
        "y_true_normalized": np.concatenate(targets, axis=0).astype(np.float32),
        "y_pred_raw": np.concatenate(predictions_raw, axis=0).astype(np.float32),
        "y_true_raw": np.concatenate(targets_raw, axis=0).astype(np.float32),
        "day_index": np.concatenate(days, axis=0),
        "window_start_slot": np.concatenate(starts, axis=0),
        "target_slot": np.concatenate(targets_slot, axis=0),
        "sample_index": np.concatenate(sample_index, axis=0),
    }
    order = np.argsort(packed["sample_index"], kind="stable")
    packed = {key: value[order] for key, value in packed.items()}
    metrics = traffic_metrics(packed["y_pred_raw"], packed["y_true_raw"], clip_negative=False)
    metrics["training_loss"] = total_loss / max(n_batches, 1)
    packed["metrics"] = metrics
    return packed


def save_predictions(path: Path, packed: Mapping[str, Any]) -> None:
    atomic_write_npz(
        path,
        y_pred_normalized=packed["y_pred_normalized"],
        y_true_normalized=packed["y_true_normalized"],
        y_pred_raw=packed["y_pred_raw"],
        y_true_raw=packed["y_true_raw"],
        day_index=packed["day_index"],
        window_start_slot=packed["window_start_slot"],
        target_slot=packed["target_slot"],
        sample_index=packed["sample_index"],
    )


def _prediction_stats(pred: np.ndarray) -> dict[str, float]:
    return {
        "minimum_prediction": float(pred.min()) if pred.size else float("nan"),
        "maximum_prediction": float(pred.max()) if pred.size else float("nan"),
        "mean_prediction": float(pred.mean()) if pred.size else float("nan"),
    }


def write_split_tables(output_dir: Path, packed: Mapping[str, Any], node_ids: list[str]) -> None:
    pred = packed["y_pred_raw"]
    true = packed["y_true_raw"]
    node_rows = []
    for node in range(pred.shape[2]):
        stats = traffic_metrics(pred[:, :, node, :], true[:, :, node, :], clip_negative=False)
        stats.update(_prediction_stats(pred[:, :, node, :]))
        stats["node_index"] = node
        stats["edge_id"] = node_ids[node] if node < len(node_ids) else ""
        node_rows.append(stats)
    fields = [
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
        "maximum_prediction",
        "mean_prediction",
    ]
    atomic_write_csv_rows(output_dir / "node_metrics.csv", fields, node_rows)
    day_rows = []
    days = packed["day_index"]
    for day in sorted(set(int(item) for item in days.tolist())):
        mask = days == day
        stats = traffic_metrics(pred[mask], true[mask], clip_negative=False)
        stats.update(_prediction_stats(pred[mask]))
        stats["day_index"] = day
        day_rows.append(stats)
    atomic_write_csv_rows(
        output_dir / "day_metrics.csv",
        ["day_index"] + [name for name in fields if name not in {"node_index", "edge_id"}],
        day_rows,
    )


def _finite_nonzero_grad(tensor: torch.Tensor | None) -> bool:
    if tensor is None or tensor.grad is None:
        return False
    if not torch.isfinite(tensor.grad).all():
        return False
    return bool(tensor.grad.abs().max().item() > 0.0)


def run_p70_smoke(
    config: Mapping[str, Any],
    *,
    data_root: Path,
    graph_info: Mapping[str, Any],
    data_info: Mapping[str, Any],
    logger: JsonlLogger | None = None,
) -> dict[str, Any]:
    seed = int(config["seed"])
    seed_everything(seed)
    device = resolve_device(str(config.get("device", "auto")))
    train_pack = build_train_dataloader(data_root, "p70", config, data_info["sample_counts"])
    model = build_model(config, graph_info["spatial_topology"]).to(device)
    init_hash = parameter_sha256(model)
    shape_info = validate_model_shapes(
        model, batch_size=min(2, int(config["batch_size"])), seq_len=int(config["seq_len"])
    )
    optimizer = build_optimizer(model, config)
    criterion = HuberRawLoss(float(data_info["mean_y_full"]), float(data_info["std_y_full"]))
    batch = _move_batch(next(iter(train_pack["loader"])), device)
    x = batch["x"]
    y = batch["y"]
    model.train()
    pred, trace = model(x, return_trace=True)
    if pred.shape != y.shape:
        raise ReimplementationError(f"smoke prediction {tuple(pred.shape)} != target {tuple(y.shape)}")
    if not torch.isfinite(pred).all():
        raise ReimplementationError("smoke prediction is not finite")
    loss = criterion(pred, batch["y_raw"])
    if not torch.isfinite(loss):
        raise ReimplementationError(f"smoke loss is not finite: {loss}")
    before = {name: tensor.detach().cpu().clone() for name, tensor in model.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    flags = {
        "input_projection": False,
        "gcn": False,
        "stsgcm": False,
        "embedding": False,
        "mask": False,
        "output": False,
    }
    for name, tensor in model.named_parameters():
        if tensor.grad is None or not torch.isfinite(tensor.grad).all():
            raise ReimplementationError(f"smoke missing or non-finite grad: {name}")
        max_abs = float(tensor.grad.abs().max().cpu())
        if max_abs > 0.0:
            if name.startswith("input_projection"):
                flags["input_projection"] = True
            if "gcn_layers" in name:
                flags["gcn"] = True
                flags["stsgcm"] = True
            if "temporal_emb" in name or "spatial_emb" in name:
                flags["embedding"] = True
            if name == "adj_mask":
                flags["mask"] = True
            if name.startswith("output_heads"):
                flags["output"] = True
    missing = [key for key, ok in flags.items() if not ok]
    if missing:
        raise ReimplementationError(f"smoke missing gradients: {missing}")
    if model.localized_adj.grad is not None:
        raise ReimplementationError("localized_adj buffer must not receive gradients")
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
        "device": str(device),
        "seed": seed,
        "input_shape": list(x.shape),
        "target_shape": list(y.shape),
        "prediction_shape": list(pred.shape),
        "trace": trace,
        "training_loss": float(loss.detach().cpu()),
        "parameter_count": model.parameter_count(),
        "init_parameter_sha256": init_hash,
        "parameters_changed": changed,
        "gradient_status": flags,
        "discarded_after_smoke": True,
        "did_not_write_official_checkpoint": True,
        "did_not_read_validation_or_test": True,
        "shape_info": shape_info,
        **_process_stats(device),
    }
    if logger is not None:
        logger.log(
            {
                "stage": "smoke_testing",
                "model_name": "stsgcn",
                "penetration_rate": "p70",
                "seed": seed,
                "status": "ok",
                "training_loss": payload["training_loss"],
                "device": str(device),
            }
        )
    print(
        f"[stsgcn] stage=smoke_testing rate=p70 loss={payload['training_loss']:.6f} status=ok",
        flush=True,
    )
    del model
    del optimizer
    del train_pack
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def collect_data_info(data_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    manifest = load_json(data_root / "split_manifest.json")
    normalization = load_json(data_root / "normalization.json")
    data_report = load_json(data_root / "validation_summary.json")
    if data_report.get("failures"):
        raise ReimplementationError(f"data validation report has failures: {data_report['failures']}")
    if data_report.get("overall_validation_passed") is False:
        raise ReimplementationError("data validation report overall_validation_passed is false")
    mean_y, std_y = read_target_scaler(normalization)
    counts = expected_split_counts(manifest)
    node_ids = load_node_ids(data_root / "node_mapping.csv")
    return {
        "sample_counts": counts,
        "mean_y_full": mean_y,
        "std_y_full": std_y,
        "node_ids": node_ids,
        "normalization_sha256": sha256_file(data_root / "normalization.json"),
        "split_manifest_target_mode": manifest.get("target_mode"),
        "data_validation_status": data_report.get("status"),
        "file_hashes": {
            f"{tag}/train.npz": sha256_file(data_root / tag / "train.npz")
            for tag in [rate_tag(item) for item in config["rates"]]
        },
    }


def run_single_rate(
    config: Mapping[str, Any],
    *,
    rate: str,
    data_root: Path,
    output_root: Path,
    graph_info: Mapping[str, Any],
    data_info: Mapping[str, Any],
    logger: JsonlLogger,
    resume_path: Path | None = None,
) -> dict[str, Any]:
    tag = rate_tag(rate)
    seed = int(config["seed"])
    seed_everything(seed)
    device = resolve_device(str(config.get("device", "auto")))
    rate_dir = output_root / tag / f"seed_{seed}"
    rate_dir.mkdir(parents=True, exist_ok=True)
    loaders_pack = build_dataloaders(data_root, tag, config, data_info["sample_counts"])
    model = build_model(config, graph_info["spatial_topology"]).to(device)
    init_hash = parameter_sha256(model)
    optimizer = build_optimizer(model, config)
    steps_per_epoch = max(len(loaders_pack["loaders"]["train"]), 1)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch=steps_per_epoch)
    criterion = HuberRawLoss(float(data_info["mean_y_full"]), float(data_info["std_y_full"]))
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
            n_his=int(config["seq_len"]),
            output_steps=int(config["horizon"]),
            input_channels=int(config["input_channels"]),
            output_channels=int(config["output_channels"]),
        )
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if payload.get("scheduler_state_dict"):
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

    history_rows: list[dict[str, Any]] = []
    epochs = int(config["epochs"])
    print(
        (
            f"[stsgcn] stage=building_model rate={tag} params={model.parameter_count()} "
            f"init={init_hash[:16]} device={device}"
        ),
        flush=True,
    )
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_stats = train_one_epoch(
            model,
            loaders_pack["loaders"]["train"],
            optimizer,
            criterion,
            device,
            scheduler=scheduler,
        )
        val_packed = evaluate_one_epoch(
            model,
            loaders_pack["loaders"]["validation"],
            criterion,
            device,
            mean_y,
            std_y,
        )
        val_mae = float(val_packed["metrics"]["mae"])
        improved = val_mae < best_metric
        if improved:
            best_metric = val_mae
            best_epoch = epoch
            save_checkpoint(rate_dir / "best_checkpoint.pt", checkpoint_payload(epoch, val_mae))
        save_checkpoint(rate_dir / "last_checkpoint.pt", checkpoint_payload(epoch, best_metric))
        row = {
            "epoch": epoch,
            "training_loss": train_stats["training_loss"],
            "validation_mae_raw": val_mae,
            "validation_rmse_raw": val_packed["metrics"]["rmse"],
            "seconds": time.time() - t0,
            "is_best": improved,
        }
        history_rows.append(row)
        logger.log({"stage": "epoch", "model_name": "stsgcn", "penetration_rate": tag, **row})
        print(
            (
                f"[stsgcn] stage=epoch rate={tag} epoch={epoch} "
                f"train_loss={train_stats['training_loss']:.6f} val_mae={val_mae:.6f}"
            ),
            flush=True,
        )

    best_payload = load_checkpoint(rate_dir / "best_checkpoint.pt", map_location=device)
    model.load_state_dict(best_payload["model_state_dict"])
    test_packed = evaluate_one_epoch(
        model,
        loaders_pack["loaders"]["test"],
        criterion,
        device,
        mean_y,
        std_y,
    )
    val_packed = evaluate_one_epoch(
        model,
        loaders_pack["loaders"]["validation"],
        criterion,
        device,
        mean_y,
        std_y,
    )
    save_predictions(rate_dir / "validation_predictions.npz", val_packed)
    save_predictions(rate_dir / "test_predictions.npz", test_packed)
    write_split_tables(rate_dir, test_packed, data_info["node_ids"])
    atomic_write_json(rate_dir / "validation_metrics.json", val_packed["metrics"])
    atomic_write_json(rate_dir / "test_metrics.json", test_packed["metrics"])
    atomic_write_json(rate_dir / "training_history.json", history_rows)
    result = {
        "rate": tag,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "parameter_count": model.parameter_count(),
        "init_parameter_sha256": init_hash,
        "validation_mae_raw": float(val_packed["metrics"]["mae"]),
        "validation_rmse_raw": float(val_packed["metrics"]["rmse"]),
        "validation_mape_nonzero": float(val_packed["metrics"]["mape_nonzero"]),
        "validation_wape": float(val_packed["metrics"]["wape"]),
        "test_mae_raw": float(test_packed["metrics"]["mae"]),
        "test_rmse_raw": float(test_packed["metrics"]["rmse"]),
        "test_mape_nonzero": float(test_packed["metrics"]["mape_nonzero"]),
        "test_wape": float(test_packed["metrics"]["wape"]),
        "output_dir": rate_dir.as_posix(),
    }
    del model
    del optimizer
    return result


def run_all_rates(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    rates: list[str],
    data_root: Path,
    output_root: Path,
    topology_path: Path,
    metadata_path: Path,
    graph_validation_path: Path,
    r_nodes_path: Path,
    logger: JsonlLogger,
    smoke_only: bool,
    resume: Path | None = None,
) -> dict[str, Any]:
    node_ids = load_node_ids(data_root / "node_mapping.csv")
    graph_info = validate_stsgcn_graph(
        topology_path=topology_path,
        metadata_path=metadata_path,
        validation_path=graph_validation_path,
        r_nodes_path=r_nodes_path,
        node_ids=node_ids,
        expected_nodes=int(config["num_nodes"]),
    )
    data_info = collect_data_info(data_root, config)
    if not smoke_only:
        validate_prepared_data(
            data_root,
            n_his=int(config["seq_len"]),
            num_nodes=int(config["num_nodes"]),
            rates=[rate_tag(item) for item in rates],
        )
    else:
        data_info["train_npz"] = validate_train_npz_only(
            data_root,
            rate="p70",
            n_his=int(config["seq_len"]),
            num_nodes=int(config["num_nodes"]),
            expected_count=int(data_info["sample_counts"]["train"]),
        )
    data_preview = {key: value for key, value in data_info.items() if key != "node_ids"}
    graph_preview = {
        key: value
        for key, value in graph_info.items()
        if key not in {"spatial_topology", "localized_adj", "mask_init", "node_ids"}
    }
    smoke_result = run_p70_smoke(
        config,
        data_root=data_root,
        graph_info=graph_info,
        data_info=data_info,
        logger=logger,
    )
    seed_everything(int(config["seed"]))
    if smoke_only:
        return {
            "rates": [],
            "data_info": data_preview,
            "graph_info": graph_preview,
            "smoke": smoke_result,
        }
    summaries = []
    for rate in rates:
        print(f"[stsgcn] stage=training_start rate={rate_tag(rate)} seed={config['seed']}", flush=True)
        result = run_single_rate(
            config,
            rate=rate,
            data_root=data_root,
            output_root=output_root,
            graph_info=graph_info,
            data_info=data_info,
            logger=logger,
            resume_path=resume,
        )
        summaries.append({"rate": rate_tag(rate), **{k: v for k, v in result.items() if k != "output_dir"}})
    return {
        "rates": summaries,
        "data_info": data_preview,
        "graph_info": graph_preview,
        "smoke": smoke_result,
    }
