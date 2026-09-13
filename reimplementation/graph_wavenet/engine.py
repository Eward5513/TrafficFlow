"""Training and evaluation engine for R-only Graph WaveNet."""

from __future__ import annotations

import hashlib
import io
import resource
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.optim import Adam

from reimplementation.common.data.r_only_npz_dataset import (
    RATE_TAGS,
    ROnlyNPZDataset,
    SPLITS,
    build_dataloader,
    invert_target,
    load_json,
    rate_tag,
)
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.metrics.traffic_metrics import traffic_metrics
from reimplementation.common.utils.atomic_io import atomic_write_bytes, atomic_write_json, atomic_write_npz
from reimplementation.common.utils.checkpoint import (
    assert_checkpoint_compatible,
    load_checkpoint,
    save_checkpoint,
)
from reimplementation.common.utils.reproducibility import seed_everything
from reimplementation.common.utils.structured_logging import JsonlLogger
from reimplementation.graph_wavenet.losses import MaskedMAERawLoss, mae_raw_all, mae_raw_nonzero
from reimplementation.graph_wavenet.model.graph_wavenet import CODE_VERSION, GraphWaveNet
from reimplementation.graph_wavenet.validation import (
    validate_gwn_graph,
    validate_model_shapes,
    validate_prepared_data,
)
from reimplementation.stgcn.engine import atomic_write_csv_rows, resolve_device, resolve_path


def parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, param in sorted(model.named_parameters(), key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        array = np.ascontiguousarray(param.detach().cpu().numpy())
        digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_write_npy(path: Path, array: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array))
    atomic_write_bytes(path, buffer.getvalue())


def _cpu_rss_bytes() -> float | None:
    try:
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if sys.platform != "darwin":
        rss *= 1024.0
    return rss


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


def build_model(config: Mapping[str, Any], supports: list[np.ndarray]) -> GraphWaveNet:
    nhid = int(config.get("nhid", config.get("residual_channels", 32)))
    aptinit = None
    if not bool(config.get("randomadj", True)):
        aptinit = torch.as_tensor(supports[0], dtype=torch.float32)
    return GraphWaveNet(
        num_nodes=int(config["num_nodes"]),
        dropout=float(config.get("dropout", 0.3)),
        supports=supports,
        gcn_bool=bool(config.get("gcn_bool", True)),
        addaptadj=bool(config.get("addaptadj", True)),
        aptinit=aptinit,
        in_dim=int(config["input_channels"]),
        out_dim=int(config["output_channels"]),
        residual_channels=int(config.get("residual_channels", nhid)),
        dilation_channels=int(config.get("dilation_channels", nhid)),
        skip_channels=int(config.get("skip_channels", nhid * 8)),
        end_channels=int(config.get("end_channels", nhid * 16)),
        kernel_size=int(config.get("kernel_size", 2)),
        blocks=int(config.get("blocks", 4)),
        layers=int(config.get("layers", 2)),
        gcn_order=int(config.get("gcn_order", 2)),
        engine_left_pad=int(config.get("engine_left_pad", 1)),
    )


def build_optimizer(model: GraphWaveNet, config: Mapping[str, Any]) -> Adam:
    if str(config.get("optimizer", "adam")).lower() != "adam":
        raise ReimplementationError("original Graph WaveNet training uses Adam")
    return Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 1.0e-4)),
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


def _prediction_stats(pred: np.ndarray) -> dict[str, float]:
    flat = np.asarray(pred, dtype=np.float64)
    return {
        "minimum_prediction": float(flat.min()) if flat.size else float("nan"),
        "maximum_prediction": float(flat.max()) if flat.size else float("nan"),
        "mean_prediction": float(flat.mean()) if flat.size else float("nan"),
    }


def train_one_epoch(
    model: GraphWaveNet,
    loader,
    optimizer: Adam,
    criterion: MaskedMAERawLoss,
    device: torch.device,
    *,
    gradient_clip: float,
    rate: str = "",
    epoch: int = -1,
    epochs: int = -1,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_mae_all = 0.0
    total_mae_nz = 0.0
    n_batches = 0
    n_samples = 0
    epoch_t0 = time.time()
    for batch in loader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch["x"])
        if prediction.shape != batch["y"].shape:
            raise ReimplementationError(
                f"prediction {tuple(prediction.shape)} != target {tuple(batch['y'].shape)}"
            )
        loss = criterion(prediction, batch["y_raw"])
        if not torch.isfinite(loss):
            raise ReimplementationError(f"non-finite training loss at batch={n_batches}")
        loss.backward()
        if gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        pred_raw = criterion.denormalize(prediction.detach())
        total_loss += float(loss.detach().cpu())
        total_mae_all += float(mae_raw_all(pred_raw, batch["y_raw"]).detach().cpu())
        total_mae_nz += float(mae_raw_nonzero(pred_raw, batch["y_raw"]).detach().cpu())
        n_batches += 1
        n_samples += int(batch["x"].size(0))
        if n_batches == 1 or n_batches % 10 == 0:
            print(
                (
                    f"[gwn] stage=training_batch rate={rate} epoch={epoch}/{epochs - 1} "
                    f"batch={n_batches} samples={n_samples} "
                    f"loss={float(loss.detach().cpu()):.6f} "
                    f"elapsed={time.time() - epoch_t0:.1f}s"
                ),
                flush=True,
            )
    return {
        "training_loss": total_loss / max(n_batches, 1),
        "training_mae_raw_all": total_mae_all / max(n_batches, 1),
        "training_mae_raw_nonzero": total_mae_nz / max(n_batches, 1),
        "train_sample_count": float(n_samples),
        "batch_count": float(n_batches),
    }


@torch.no_grad()
def evaluate_one_epoch(
    model: GraphWaveNet,
    loader,
    criterion: MaskedMAERawLoss,
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
    metrics.update(_prediction_stats(packed["y_pred_raw"]))
    pred_t = torch.from_numpy(packed["y_pred_raw"])
    true_t = torch.from_numpy(packed["y_true_raw"])
    metrics["mae_raw_all"] = float(mae_raw_all(pred_t, true_t))
    metrics["mae_raw_nonzero"] = float(mae_raw_nonzero(pred_t, true_t))
    metrics["masked_mae_loss"] = total_loss / max(n_batches, 1)
    packed["metrics"] = metrics
    return packed


def _write_split_outputs(
    output_dir: Path,
    split: str,
    result: Mapping[str, Any],
    *,
    write_tables: bool,
    node_ids: list[str],
) -> None:
    atomic_write_json(output_dir / f"{split}_metrics.json", dict(result["metrics"]))
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
    pred = result["y_pred_raw"]
    true = result["y_true_raw"]
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
    days = result["day_index"]
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


def run_p70_smoke(
    config: Mapping[str, Any],
    *,
    data_root: Path,
    graph_info: Mapping[str, Any],
    data_info: Mapping[str, Any],
    logger: JsonlLogger,
) -> dict[str, Any]:
    seed = int(config["seed"])
    seed_everything(seed)
    device = resolve_device(str(config.get("device", "auto")))
    loaders_pack = build_dataloaders(data_root, "p70", config, data_info["sample_counts"])
    model = build_model(config, graph_info["supports"]).to(device)
    init_hash = parameter_sha256(model)
    shape_info = validate_model_shapes(
        model, batch_size=min(2, int(config["batch_size"])), seq_len=int(config["seq_len"])
    )
    optimizer = build_optimizer(model, config)
    criterion = MaskedMAERawLoss(float(data_info["mean_y_full"]), float(data_info["std_y_full"]))
    batch = _move_batch(next(iter(loaders_pack["loaders"]["train"])), device)
    x = batch["x"]
    y = batch["y"]
    model.train()
    pred, trace = model(x, return_trace=True)
    if pred.shape != y.shape:
        raise ReimplementationError(f"smoke prediction {tuple(pred.shape)} != target {tuple(y.shape)}")
    if not torch.isfinite(pred).all():
        raise ReimplementationError("smoke prediction is not finite")
    if int(trace["layers"][-1]["output_time_length"]) != 1:
        raise ReimplementationError("smoke final time length is not 1")
    loss = criterion(pred, batch["y_raw"])
    if not torch.isfinite(loss):
        raise ReimplementationError(f"smoke loss is not finite: {loss}")
    before = {name: tensor.detach().cpu().clone() for name, tensor in model.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    flags = {
        "filter": False,
        "gate": False,
        "skip": False,
        "gcn": False,
        "nodevec": False,
        "end": False,
    }
    n_layers = model.blocks * model.layers
    last_gconv = f"gconv.{n_layers - 1}."
    last_bn = f"bn.{n_layers - 1}."
    for name, tensor in model.named_parameters():
        if "residual_convs" in name or name.startswith(last_gconv) or name.startswith(last_bn):
            continue
        if tensor.grad is None or not torch.isfinite(tensor.grad).all():
            raise ReimplementationError(f"smoke missing or non-finite grad: {name}")
        max_abs = float(tensor.grad.abs().max().cpu())
        if max_abs > 0.0:
            if "filter_convs" in name:
                flags["filter"] = True
            if "gate_convs" in name:
                flags["gate"] = True
            if "skip_convs" in name:
                flags["skip"] = True
            if "gconv" in name:
                flags["gcn"] = True
            if name.startswith("nodevec"):
                flags["nodevec"] = True
            if name.startswith("end_conv"):
                flags["end"] = True
    missing = [key for key, ok in flags.items() if not ok]
    if missing:
        raise ReimplementationError(f"smoke missing gradients: {missing}")
    for name, tensor in model.named_buffers():
        if name.startswith("support_") and tensor.grad is not None:
            raise ReimplementationError("static supports must not receive gradients")
    optimizer.step()
    changed = 0
    for name, tensor in model.named_parameters():
        if not torch.equal(tensor.detach().cpu(), before[name]):
            changed += 1
    if changed == 0:
        raise ReimplementationError("optimizer step did not change parameters")
    adp = model.adaptive_adjacency().detach()
    row_sum = adp.sum(dim=1)
    if not torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-5):
        raise ReimplementationError("adaptive adjacency rows do not sum to 1")
    payload = {
        "status": "ok",
        "penetration_rate": "p70",
        "batch_size": int(x.size(0)),
        "input_shape": list(x.shape),
        "target_shape": list(y.shape),
        "prediction_shape": list(pred.shape),
        "trace": trace,
        "training_loss": float(loss.detach().cpu()),
        "parameter_count": model.parameter_count(),
        "init_parameter_sha256": init_hash,
        "parameters_changed": changed,
        "discarded_after_smoke": True,
        "shape_info": shape_info,
        "adaptive_row_sum_ok": True,
        "engine_left_pad": model.engine_left_pad,
        "receptive_field": model.receptive_field,
        **_process_stats(device),
    }
    logger.log(
        {
            "stage": "smoke_testing",
            "model_name": "graph_wavenet",
            "penetration_rate": "p70",
            "seed": seed,
            "status": "ok",
            "training_loss": payload["training_loss"],
            "device": str(device),
            **_process_stats(device),
        }
    )
    print(
        f"[gwn] stage=smoke_testing rate=p70 loss={payload['training_loss']:.6f} status=ok",
        flush=True,
    )
    del model
    del optimizer
    del loaders_pack
    return payload


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
    model = build_model(config, graph_info["supports"]).to(device)
    init_hash = parameter_sha256(model)
    shape_info = validate_model_shapes(
        model, batch_size=min(2, int(config["batch_size"])), seq_len=int(config["seq_len"])
    )
    optimizer = build_optimizer(model, config)
    criterion = MaskedMAERawLoss(float(data_info["mean_y_full"]), float(data_info["std_y_full"]))
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
        start_epoch = int(payload["epoch"]) + 1
        best_metric = float(payload["best_metric"])
        best_epoch = int(payload["epoch"])

    def checkpoint_payload(epoch: int, metric: float) -> dict[str, Any]:
        return {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,
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
            f"[gwn] stage=building_model rate={tag} params={model.parameter_count()} "
            f"init={init_hash[:16]} device={device} rf={model.receptive_field}"
        ),
        flush=True,
    )
    for epoch in range(start_epoch, epochs):
        lr = float(config["learning_rate"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        epoch_start = time.time()
        print(
            f"[gwn] stage=training_epoch_start rate={tag} epoch={epoch}/{epochs - 1} lr={lr:.6g}",
            flush=True,
        )
        train_stats = train_one_epoch(
            model,
            loaders_pack["loaders"]["train"],
            optimizer,
            criterion,
            device,
            gradient_clip=float(config.get("max_grad_norm", 5.0)),
            rate=tag,
            epoch=epoch,
            epochs=epochs,
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
        if val_mae < best_metric:
            best_metric = val_mae
            best_epoch = epoch
            save_checkpoint(rate_dir / "best_checkpoint.pt", checkpoint_payload(epoch, best_metric))
        save_checkpoint(rate_dir / "last_checkpoint.pt", checkpoint_payload(epoch, best_metric))
        row = {
            "epoch": epoch,
            "training_loss": train_stats["training_loss"],
            "training_mae_raw_all": train_stats["training_mae_raw_all"],
            "training_mae_raw_nonzero": train_stats["training_mae_raw_nonzero"],
            "validation_loss": float(val_result["metrics"]["masked_mae_loss"]),
            "validation_mae_raw_all": val_mae,
            "validation_mae_raw_nonzero": float(val_result["metrics"]["mae_raw_nonzero"]),
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
                "model_name": "graph_wavenet",
                "penetration_rate": tag,
                "seed": seed,
                "epoch": epoch,
                "batch_index": int(train_stats["batch_count"]) - 1,
                "train_sample_count": train_stats["train_sample_count"],
                "validation_sample_count": float(data_info["sample_counts"]["validation"]),
                "status": "ok",
                "device": str(device),
                "best_validation_mae": best_metric,
                **row,
                **_process_stats(device),
            }
        )
        print(
            (
                f"[gwn] stage=training rate={tag} epoch={epoch}/{epochs - 1} "
                f"train_loss={train_stats['training_loss']:.6f} "
                f"train_mae_all={train_stats['training_mae_raw_all']:.6f} "
                f"val_mae={val_mae:.6f} best_epoch={best_epoch} lr={lr:.6g} "
                f"elapsed={row['elapsed_seconds']:.1f}s"
            ),
            flush=True,
        )

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
    adp = model.adaptive_adjacency().detach().cpu().numpy().astype(np.float32)
    atomic_write_npy(rate_dir / "learned_adaptive_adjacency.npy", adp)
    logger.log({"stage": "validating", "model_name": "graph_wavenet", "penetration_rate": tag, "status": "ok"})
    val_best = evaluate_one_epoch(
        model, loaders_pack["loaders"]["validation"], criterion, device, mean_y, std_y
    )
    _write_split_outputs(rate_dir, "validation", val_best, write_tables=False, node_ids=list(data_info["node_ids"]))
    logger.log({"stage": "testing", "model_name": "graph_wavenet", "penetration_rate": tag, "status": "ok"})
    test_best = evaluate_one_epoch(
        model, loaders_pack["loaders"]["test"], criterion, device, mean_y, std_y
    )
    dataset_test = loaders_pack["datasets"]["test"]
    if not np.allclose(test_best["y_true_raw"], dataset_test.y_raw, atol=1e-5):
        raise ReimplementationError("test y_true_raw does not match the original NPZ order")
    recovered = invert_target(test_best["y_pred_normalized"], mean_y, std_y)
    if np.max(np.abs(np.asarray(recovered) - test_best["y_pred_raw"])) > 1e-4:
        raise ReimplementationError("test denormalization does not recover y_pred_raw")
    _write_split_outputs(rate_dir, "test", test_best, write_tables=True, node_ids=list(data_info["node_ids"]))
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
            "checkpoint_monitor": config.get("checkpoint_monitor", "validation_mae_raw_all"),
            "adjtype": config.get("adjtype"),
            "gcn_bool": True,
            "addaptadj": True,
            "randomadj": bool(config.get("randomadj", True)),
            "horizon": config.get("horizon"),
            "receptive_field": model.receptive_field,
            "device": str(device),
        },
    )
    atomic_write_json(
        rate_dir / "run_validation.json",
        {
            "status": "ok",
            "shape_trace": shape_info["trace"],
            "parameter_count": model.parameter_count(),
            "receptive_field": model.receptive_field,
            "adaptive_adjacency_path": (rate_dir / "learned_adaptive_adjacency.npy").as_posix(),
        },
    )
    logger.log({"stage": "validating_outputs", "model_name": "graph_wavenet", "penetration_rate": tag, "status": "ok"})
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
        "validation_mae_raw_all": float(val_best["metrics"]["mae_raw_all"]),
        "validation_mae_raw_nonzero": float(val_best["metrics"]["mae_raw_nonzero"]),
        "test_mae_raw": float(test_best["metrics"]["mae"]),
        "test_rmse_raw": float(test_best["metrics"]["rmse"]),
        "test_mape_nonzero": float(test_best["metrics"]["mape_nonzero"]),
        "test_wape": float(test_best["metrics"]["wape"]),
        "test_mae_raw_all": float(test_best["metrics"]["mae_raw_all"]),
        "test_mae_raw_nonzero": float(test_best["metrics"]["mae_raw_nonzero"]),
        "test_negative_prediction_count": float(test_best["metrics"]["negative_prediction_count"]),
        "test_negative_prediction_fraction": float(test_best["metrics"]["negative_prediction_fraction"]),
        "test_minimum_prediction": float(test_best["metrics"]["minimum_prediction"]),
        "test_maximum_prediction": float(test_best["metrics"]["maximum_prediction"]),
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
    pickle_path: Path,
    metadata_path: Path,
    graph_validation_path: Path,
    r_nodes_path: Path,
    logger: JsonlLogger,
    smoke_only: bool = False,
    resume: Path | None = None,
) -> dict[str, Any]:
    rate_list = [rate_tag(item) for item in rates]
    if resume is not None and len(rate_list) > 1:
        raise ReimplementationError("use --resume together with --rate; do not share checkpoints across penetrations")
    stgcn_manifest = project_root / "reimplementation" / "stgcn" / "experiments" / "r_only" / "experiment_manifest.json"
    if stgcn_manifest.is_file():
        stgcn_rates = [rate_tag(item) for item in load_json(stgcn_manifest).get("rates", [])]
        if not smoke_only and resume is None and rate_list != stgcn_rates:
            raise ReimplementationError(
                f"GWN rates {rate_list} must match STGCN R-only rates {stgcn_rates}"
            )
    if not smoke_only and resume is None and rate_list != list(RATE_TAGS):
        raise ReimplementationError(
            f"official R-only training requires the 7 STGCN rates {list(RATE_TAGS)}, got {rate_list}"
        )
    logger.log({"stage": "scanning", "model_name": "graph_wavenet", "status": "ok"})
    data_info = validate_prepared_data(
        data_root,
        n_his=int(config["seq_len"]),
        num_nodes=int(config["num_nodes"]),
        rates=rate_list if rate_list else list(RATE_TAGS),
    )
    logger.log({"stage": "validating_inputs", "model_name": "graph_wavenet", "status": "ok"})
    graph_info = validate_gwn_graph(
        adjacency_path=adjacency_path,
        pickle_path=pickle_path,
        metadata_path=metadata_path,
        validation_path=graph_validation_path,
        r_nodes_path=r_nodes_path,
        node_ids=list(data_info["node_ids"]),
        expected_nodes=int(config["num_nodes"]),
        adjtype=str(config.get("adjtype", "doubletransition")),
        sparse_path=adjacency_path.with_name("dcrnn_weighted_adjacency_sparse.npz"),
    )
    logger.log(
        {
            "stage": "building_static_supports",
            "model_name": "graph_wavenet",
            "status": "ok",
            "graph_sha256": graph_info["graph_sha256"],
            "support_count": graph_info["support_count"],
            "adjtype": graph_info["adjtype"],
        }
    )
    data_preview = {key: value for key, value in data_info.items() if key != "node_ids"}
    graph_preview = {key: value for key, value in graph_info.items() if key not in {"supports", "adjacency", "node_ids"}}
    atomic_write_json(output_root / "data_runtime_validation.json", data_preview)
    atomic_write_json(output_root / "graph_runtime_validation.json", graph_preview)
    print(
        (
            f"[gwn] stage=building_static_supports nodes={config['num_nodes']} "
            f"supports={graph_info['support_count']} adjtype={graph_info['adjtype']} "
            f"sha256={graph_info['graph_sha256'][:12]}"
        ),
        flush=True,
    )
    smoke_result = None
    if resume is None:
        logger.log({"stage": "building_adaptive_graph", "model_name": "graph_wavenet", "status": "ok"})
        smoke_result = run_p70_smoke(
            config,
            data_root=data_root,
            graph_info=graph_info,
            data_info=data_info,
            logger=logger,
        )
        seed_everything(int(config["seed"]))
        atomic_write_json(output_root / "smoke_test.json", smoke_result)
        print("[gwn] stage=smoke_testing discarded_model=yes wrote smoke_test.json", flush=True)
    if smoke_only:
        return {
            "rates": [],
            "data_info": data_preview,
            "graph_info": graph_preview,
            "smoke": smoke_result,
        }
    summaries = []
    init_hashes = []
    for rate in rate_list:
        print(f"[gwn] stage=training_start rate={rate_tag(rate)} seed={config['seed']}", flush=True)
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
        init_hashes.append(result.get("init_parameter_sha256"))
        summaries.append({"rate": rate_tag(rate), **{k: v for k, v in result.items() if k != "output_dir"}})
        print(
            (
                f"[gwn] stage=rate_completed rate={rate_tag(rate)} "
                f"best_epoch={result.get('best_epoch')} val_mae={result.get('validation_mae_raw')} "
                f"test_mae={result.get('test_mae_raw')}"
            ),
            flush=True,
        )
    unique_hashes = set(item for item in init_hashes if item)
    if len(unique_hashes) > 1:
        raise ReimplementationError(f"initial parameter SHA256 differs across rates: {init_hashes}")
    graph_preview["init_parameter_sha256"] = next(iter(unique_hashes)) if unique_hashes else None
    graph_preview["init_parameter_sha256_identical_across_rates"] = True
    return {
        "rates": summaries,
        "data_info": data_preview,
        "graph_info": graph_preview,
        "smoke": smoke_result,
    }
