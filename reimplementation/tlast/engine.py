"""Train one Xuancheng26 penetration with the released TLAST objective.

The training loss stays the reference Huber loss (delta=2). Checkpoint
selection uses the mean raw-scale MAE across sampling seeds 42–46.
Test metrics are the shared MAE / RMSE / MAPE_nonzero / WAPE after the
best checkpoint is reloaded, reported per seed and as mean and sample std.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import time
from reimplementation.common.data.sampling_seeds import sampling_rate_dir
from reimplementation.common.sampling_evaluation import evaluate_sampling_predictor, sampling_history

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.optim import AdamW

from reimplementation.common.data.r_only_npz_dataset import (
    SPLITS,
    build_dataloader,
    invert_target,
    loss_scale,
    rate_tag,
)
from reimplementation.common.data.state_windows import finalize_state_scaler, open_split_dataset
from reimplementation.common.evaluation import (
    assert_official_rate_outputs,
    evaluate_loader,
    reload_best_checkpoint,
    upsert_overall_summary,
)
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.experiment_protocol import (
    ExperimentClock,
    assert_official_rates,
    build_run_metadata,
    finalize_completed_rate,
    mark_rate_failed,
    reset_peak_memory,
    trainable_parameter_count,
)
from reimplementation.common.utils.atomic_io import atomic_write_json
from reimplementation.common.utils.checkpoint import (
    assert_checkpoint_compatible,
    load_checkpoint,
    save_checkpoint,
)
from reimplementation.common.utils.hashing import sha256_file
from reimplementation.common.utils.reproducibility import seed_everything
from reimplementation.common.utils.structured_logging import JsonlLogger
from reimplementation.stgcn.engine import atomic_write_csv_rows, resolve_device, resolve_path
from reimplementation.tlast.losses import AUTHOR_HUBER_DELTA, HuberRawLoss
from reimplementation.tlast.model.tlast import CODE_VERSION, TLAST
from reimplementation.tlast.residual import latest_residual
from reimplementation.tlast.scheduler import AuthorCosineLRScheduler
from reimplementation.tlast.temporal import time_index_from_batch
from reimplementation.tlast.validation import validate_node_order, validate_prepared_data


def parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, param in sorted(model.named_parameters(), key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        array = np.ascontiguousarray(param.detach().cpu().numpy())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def rate_is_complete(rate_dir: Path) -> bool:
    meta_path = rate_dir / "run_metadata.json"
    if not meta_path.is_file() or meta_path.stat().st_size <= 0:
        return False
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        return False
    required = (
        "best_checkpoint.pt",
        "test_metrics.json",
        "validation_metrics.json",
        "test_predictions.npz",
        "training_history.json",
    )
    return all((rate_dir / name).is_file() and (rate_dir / name).stat().st_size > 0 for name in required)


def build_model(config: Mapping[str, Any]) -> TLAST:
    return TLAST.from_config(config)


def build_optimizer(model: torch.nn.Module, config: Mapping[str, Any]) -> AdamW:
    name = str(config.get("optimizer", "AdamW")).lower()
    if name != "adamw":
        raise ReimplementationError(f"released TLAST optimizer is AdamW, got {name}")
    return AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.001)),
        weight_decay=float(config.get("weight_decay", 0.0001)),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: Mapping[str, Any]) -> AuthorCosineLRScheduler | None:
    name = str(config.get("lr_scheduler", "cosinelr")).lower()
    if name in {"none", "off"}:
        return None
    if name not in {"cosinelr", "cosine", "author_cosine"}:
        raise ReimplementationError(f"released TLAST scheduler is cosinelr, got {name}")
    return AuthorCosineLRScheduler(
        optimizer,
        t_initial=int(config.get("epochs", 100)),
        lr_min=float(config.get("lr_min", 0.0)),
        decay_rate=float(config.get("lr_decay_rate", 0.1)),
        warmup_t=int(config.get("lr_warmup_epochs", 5)),
        warmup_lr_init=float(config.get("lr_warmup_init", 1e-6)),
    )


def predict_batch(
    model: TLAST,
    batch: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    mean_y: Any,
    std_y: Any,
    *,
    return_attention: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    time_index = time_index_from_batch(batch, int(config.get("input_len", config.get("n_his", 12))))
    residual = latest_residual(batch, config, mean_y, std_y)
    return model(batch["x"], time_index, residual, return_attention=return_attention)


def build_dataloaders(
    data_root: Path,
    rate: str,
    config: Mapping[str, Any],
    split_counts: Mapping[str, int],
) -> dict[str, Any]:
    loaders = {}
    datasets = {}
    n_his = int(config.get("input_len", config.get("n_his", 12)))
    for split in SPLITS:
        dataset = open_split_dataset(
            data_root,
            rate,
            split,
            config,
            n_his=n_his,
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


def collect_data_info(
    data_root: Path,
    config: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    n_his = int(config.get("input_len", config.get("n_his", 12)))
    data_info = validate_prepared_data(
        data_root,
        n_his=n_his,
        num_nodes=int(config["num_nodes"]),
        rates=[rate_tag(item) for item in config["rates"]],
    )
    node_ids = validate_node_order(
        r_nodes_path=resolve_path(project_root, config["r_nodes"]),
        node_mapping=resolve_path(project_root, config["node_mapping"]),
        expected_nodes=int(config["num_nodes"]),
    )
    data_info["node_ids"] = node_ids
    data_info["did_not_read_adjacency"] = True
    state_norm = Path(data_root) / "state_normalization.json"
    if state_norm.is_file():
        data_info["state_normalization_sha256"] = sha256_file(state_norm)
    return data_info


def _assert_attention(attentions: list[dict[str, torch.Tensor]], model: TLAST, batch: int, steps: int) -> dict[str, list[int]]:
    if len(attentions) != model.num_layers:
        raise ReimplementationError(f"expected {model.num_layers} attention blocks, got {len(attentions)}")
    shapes: dict[str, list[int]] = {}
    for block in attentions:
        to_proxy = block["node_to_proxy"]
        to_node = block["proxy_to_node"]
        expected_proxy = (batch, steps, model.n_heads, model.proxy_num, model.num_nodes)
        expected_node = (batch, steps, model.n_heads, model.num_nodes, model.proxy_num)
        if tuple(to_proxy.shape) != expected_proxy:
            raise ReimplementationError(f"node->proxy attention {tuple(to_proxy.shape)} != {expected_proxy}")
        if tuple(to_node.shape) != expected_node:
            raise ReimplementationError(f"proxy->node attention {tuple(to_node.shape)} != {expected_node}")
        if model.proxy_num != model.num_nodes and (
            int(to_proxy.shape[-1]) == int(to_proxy.shape[-2]) or int(to_node.shape[-1]) == int(to_node.shape[-2])
        ):
            raise ReimplementationError("proxy attention collapsed to a square node-node matrix")
        if steps > 1 and torch.allclose(to_proxy[:, 0], to_proxy[:, 1]):
            raise ReimplementationError("history steps produced identical proxy attention")
        shapes = {"node_to_proxy": list(to_proxy.shape), "proxy_to_node": list(to_node.shape)}
    return shapes


def _assert_gradients(model: TLAST) -> None:
    required = {
        "data_encoding.linear1.weight": False,
        "data_encoding.linear2.weight": False,
        "time_lag_embedding.mlp.0.weight": False,
        "encoder_layers.0.readout.weight": False,
        "encoder_layers.0.node2proxy.query_projection.weight": False,
        "encoder_layers.0.proxy2node.query_projection.weight": False,
        "predictor.hidden.0.weight": False,
        "predictor.heads.0.weight": False,
    }
    for name, tensor in model.named_parameters():
        if tensor.grad is None:
            raise ReimplementationError(f"missing gradient: {name}")
        if not torch.isfinite(tensor.grad).all():
            raise ReimplementationError(f"non-finite gradient: {name}")
        if name in required and float(tensor.grad.abs().max().detach().cpu()) > 0.0:
            required[name] = True
    missing = [name for name, seen in required.items() if not seen]
    if missing:
        raise ReimplementationError(f"expected nonzero gradients, got zeros for {missing}")


def check_real_batch(
    config: Mapping[str, Any],
    *,
    data_root: Path,
    data_info: Mapping[str, Any],
    rate: str = "p70",
) -> dict[str, Any]:
    """One real training batch: shapes, attention, loss, backward, finite grads."""
    data_info = finalize_state_scaler(data_info, config, data_root)
    tag = rate_tag(rate)
    seed_everything(int(config["seed"]))
    device = resolve_device(str(config.get("device", "cpu")))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    counts = data_info["sample_counts"]
    pack = build_dataloaders(data_root, tag, config, counts)
    model = build_model(config).to(device)
    criterion = HuberRawLoss(
        loss_scale(data_info["mean_y_full"]),
        loss_scale(data_info["std_y_full"]),
        delta=float(config.get("huber_delta", AUTHOR_HUBER_DELTA)),
    )
    batch = _move_batch(next(iter(pack["loaders"]["train"])), device)
    model.eval()
    with torch.no_grad():
        inspected, attentions = predict_batch(
            model,
            batch,
            config,
            loss_scale(data_info["mean_y_full"]),
            loss_scale(data_info["std_y_full"]),
            return_attention=True,
        )
    if not isinstance(inspected, torch.Tensor):
        raise ReimplementationError("attention forward did not return a prediction tensor")
    if inspected.shape != batch["y"].shape:
        raise ReimplementationError(f"prediction {tuple(inspected.shape)} != label {tuple(batch['y'].shape)}")
    shapes = _assert_attention(attentions, model, int(inspected.shape[0]), int(config.get("input_len", 12)))
    del attentions, inspected
    model.train()
    prediction = predict_batch(
        model,
        batch,
        config,
        loss_scale(data_info["mean_y_full"]),
        loss_scale(data_info["std_y_full"]),
    )
    loss = criterion(prediction, batch["y_raw"])
    if not torch.isfinite(loss):
        raise ReimplementationError(f"smoke loss is not finite: {loss}")
    loss.backward()
    _assert_gradients(model)
    restored = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(restored)
    return {
        "status": "ok",
        "rate": tag,
        "input_type": config.get("input_type"),
        "target_type": config.get("target_type"),
        "prediction_shape": list(prediction.shape),
        "label_shape": list(batch["y"].shape),
        "attention_shapes": shapes,
        "loss": float(loss.detach().cpu()),
        "sample_counts": {split: int(counts[split]) for split in SPLITS},
        "parameter_count": model.parameter_count(),
    }


def _completed_summary(rate_dir: Path, tag: str) -> dict[str, Any]:
    metrics = json.loads((rate_dir / "test_metrics.json").read_text(encoding="utf-8"))
    validation = json.loads((rate_dir / "validation_metrics.json").read_text(encoding="utf-8"))
    metadata = json.loads((rate_dir / "run_metadata.json").read_text(encoding="utf-8"))
    return {
        "status": "completed",
        "skipped_existing": True,
        "rate": tag,
        "best_epoch": metadata.get("best_epoch"),
        "best_metric": metadata.get("best_metric"),
        "early_stopped": bool(metadata.get("early_stopped", False)),
        "epochs": metadata.get("max_epochs"),
        "epochs_executed": metadata.get("epochs_executed"),
        "parameter_count": metadata.get("parameter_count"),
        "init_parameter_sha256": metadata.get("init_parameter_sha256"),
        "validation_mae_raw": float(validation["mae"]),
        "validation_rmse_raw": float(validation["rmse"]),
        "validation_mape_nonzero": float(validation["mape_nonzero"]),
        "validation_wape": float(validation["wape"]),
        "test_mae_raw": float(metrics["mae"]),
        "test_rmse_raw": float(metrics["rmse"]),
        "test_mape_nonzero": float(metrics["mape_nonzero"]),
        "test_wape": float(metrics["wape"]),
        "test_negative_prediction_count": metrics.get("negative_prediction_count"),
        "test_negative_prediction_fraction": metrics.get("negative_prediction_fraction"),
        "train_wall_time_sec": metadata.get("train_wall_time_sec"),
        "test_inference_sec": metadata.get("test_inference_sec"),
        "peak_gpu_memory_allocated_mb": metadata.get("peak_gpu_memory_allocated_mb"),
        "did_not_use_test_for_selection": True,
    }


def run_single_rate(
    config: Mapping[str, Any],
    *,
    rate: str,
    data_root: Path,
    output_root: Path,
    data_info: Mapping[str, Any],
    logger: JsonlLogger,
    resume_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    data_info = finalize_state_scaler(data_info, config, data_root)
    tag = rate_tag(rate)
    seed = int(config["seed"])
    from reimplementation.common.data.sampling_seeds import protect_historical_results, sampling_rate_dir
    rate_dir = sampling_rate_dir(output_root, config, tag)
    protect_historical_results(rate_dir)
    if rate_is_complete(rate_dir) and not overwrite and resume_path is None:
        print(f"[tlast] stage=skip rate={tag} dir={rate_dir}", flush=True)
        return _completed_summary(rate_dir, tag)
    seed_everything(seed)
    device = resolve_device(str(config.get("device", "cpu")))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    rate_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(rate_dir / "resolved_config.json", dict(config))
    loaders_pack = build_dataloaders(data_root, tag, config, data_info["sample_counts"])
    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    mean_y = loss_scale(data_info["mean_y_full"])
    std_y = loss_scale(data_info["std_y_full"])
    criterion = HuberRawLoss(mean_y, std_y, delta=float(config.get("huber_delta", AUTHOR_HUBER_DELTA)))
    start_epoch = 0
    best_mae = float("inf")
    train_hash = data_info["file_hashes"][f"{tag}/train.npz"]
    n_his = int(config.get("input_len", config.get("n_his", 12)))
    input_channels = int(config.get("input_channels", config.get("input_dim", 1)))
    output_channels = int(config.get("output_channels", config.get("output_dim", 1)))
    auto_resume = rate_dir / "last_checkpoint.pt"
    if resume_path is None and not overwrite and auto_resume.is_file():
        resume_path = auto_resume
    if resume_path is not None:
        payload = load_checkpoint(resume_path, map_location="cpu")
        assert_checkpoint_compatible(
            payload,
            penetration_rate=tag,
            graph_sha256="",
            data_file_sha256=train_hash,
            num_nodes=int(config["num_nodes"]),
            n_his=n_his,
            output_steps=1,
            input_channels=input_channels,
            output_channels=output_channels,
        )
        saved = payload.get("config") or {}
        if saved.get("input_type") != config.get("input_type") or saved.get("target_type") != config.get("target_type"):
            raise ReimplementationError(
                "refusing to resume a checkpoint for a different input/target pair: "
                f"{saved.get('input_type')}->{saved.get('target_type')}"
            )
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and payload.get("scheduler_state_dict"):
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        best_mae = float(payload.get("best_metric", best_mae))

    def checkpoint_payload(epoch: int, metric: float) -> dict[str, Any]:
        return {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": metric,
            "config": dict(config),
            "penetration_rate": tag,
            "seed": seed,
            "model_parameter_count": model.parameter_count(),
            "graph_sha256": "",
            "data_file_sha256": train_hash,
            "normalization_sha256": data_info.get("state_normalization_sha256", data_info["normalization_sha256"]),
            "code_version": CODE_VERSION,
        }

    history: list[dict[str, Any]] = []
    if (rate_dir / "training_history.json").is_file() and start_epoch > 0:
        previous = json.loads((rate_dir / "training_history.json").read_text(encoding="utf-8"))
        if isinstance(previous, list):
            history = [row for row in previous if int(row.get("epoch", -1)) < start_epoch]
    epochs = int(config.get("epochs", 100))
    early = bool(config.get("early_stop", False))
    patience = int(config.get("early_stop_step", 20))
    early_stopped = False
    wait = 0
    best_epoch = -1
    if history:
        best_epoch = min(history, key=lambda row: float(row["val_mae"]))["epoch"]
    init_hash = parameter_sha256(model)
    param_count = trainable_parameter_count(model)
    reset_peak_memory(device)
    clock = ExperimentClock()
    clock.mark_experiment_start()
    clock.mark_train_start()
    epochs_this_process = 0
    for epoch in range(start_epoch, epochs):
        epochs_this_process += 1
        model.train()
        loss_sum = 0.0
        loss_count = 0
        for batch in loaders_pack["loaders"]["train"]:
            batch = _move_batch(batch, device)
            prediction = predict_batch(model, batch, config, mean_y, std_y)
            if prediction.shape != batch["y"].shape:
                raise ReimplementationError("prediction/target shape mismatch")
            loss = criterion(prediction, batch["y_raw"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if bool(config.get("clip_grad_norm", True)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
            optimizer.step()
            elements = int(prediction.numel())
            loss_sum += float(loss.detach().cpu()) * elements
            loss_count += elements
        if scheduler is not None:
            scheduler.step(epoch + 1)
        model.eval()
        def validation_predict(batch):
            return predict_batch(model, batch, config, mean_y, std_y)
        val_sampling = evaluate_sampling_predictor(
            loaders_pack["loaders"]["validation"], validation_predict,
            mean_y=loss_scale(data_info["mean_y_full"]), std_y=loss_scale(data_info["std_y_full"]), device=device,
        )
        val_mae = val_sampling["metrics"]["mae"]
        val_huber = float(np.mean([
            float(criterion(torch.from_numpy(p["y_pred_normalized"]).to(device),
                            torch.from_numpy(p["y_raw"]).to(device)).detach().cpu())
            for p in val_sampling["sampling_predictions"].values()
        ]))
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(loss_count, 1),
                "val_huber": val_huber,
                "val_mae": val_mae, **sampling_history(val_sampling["sampling_metrics"]),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch
            wait = 0
            save_checkpoint(rate_dir / "best_checkpoint.pt", checkpoint_payload(epoch, best_mae))
        elif early and (epoch + 1) >= 50:
            wait += 1
        save_checkpoint(rate_dir / "last_checkpoint.pt", checkpoint_payload(epoch, best_mae))
        atomic_write_json(rate_dir / "training_history.json", history)
        logger.log(
            {
                "stage": "train_epoch",
                "model_name": "tlast",
                "penetration_rate": tag,
                "input_type": config.get("input_type"),
                "target_type": config.get("target_type"),
                "epoch": epoch,
                "train_loss": history[-1]["train_loss"],
                "validation_huber_raw": val_huber,
                "validation_mae_raw_all": val_mae,
            }
        )
        print(
            (
                f"[tlast] rate={tag} in={config.get('input_type')} out={config.get('target_type')} "
                f"epoch={epoch} train_huber={history[-1]['train_loss']:.6f} "
                f"val_huber={val_huber:.6f} val_mae={val_mae:.6f}"
            ),
            flush=True,
        )
        if early and wait >= patience:
            early_stopped = True
            break
    clock.mark_train_end()
    atomic_write_json(rate_dir / "training_history.json", history)
    if not history:
        raise ReimplementationError("TLAST training produced no epochs")
    hist_best = min(history, key=lambda row: float(row["val_mae"]))
    ckpt = reload_best_checkpoint(model, rate_dir / "best_checkpoint.pt", device)
    if int(ckpt["epoch"]) != int(hist_best["epoch"]):
        raise ReimplementationError(
            f"best checkpoint epoch {ckpt['epoch']} != history min-mean-validation-MAE epoch {hist_best['epoch']}"
        )
    best_epoch = int(ckpt["epoch"])
    counts = data_info["sample_counts"]
    model.eval()

    def _predict(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return predict_batch(model, batch, config, mean_y, std_y)

    logger.log({"stage": "validating", "model_name": "tlast", "penetration_rate": tag, "status": "ok"})
    val_eval = evaluate_loader(
        loaders_pack["loaders"]["validation"],
        _predict,
        mean_y=mean_y,
        std_y=std_y,
        device=device,
        output_dir=rate_dir,
        split="validation",
        expected_count=int(counts["validation"]),
        n_his=n_his,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    wall_start = time.perf_counter()
    if inference_start is not None:
        inference_start.record()
    logger.log({"stage": "testing", "model_name": "tlast", "penetration_rate": tag, "status": "ok"})
    test_eval = evaluate_loader(
        loaders_pack["loaders"]["test"],
        _predict,
        mean_y=mean_y,
        std_y=std_y,
        device=device,
        output_dir=rate_dir,
        split="test",
        expected_count=int(counts["test"]),
        n_his=n_his,
    )
    if device.type == "cuda":
        inference_end = torch.cuda.Event(enable_timing=True)
        inference_end.record()
        torch.cuda.synchronize(device)
        test_inference_sec = float(inference_start.elapsed_time(inference_end) / 1000.0)
    else:
        test_inference_sec = float(time.perf_counter() - wall_start)
    output_check = assert_official_rate_outputs(
        rate_dir,
        expected_test_count=int(counts["test"]),
        expected_val_count=int(counts["validation"]),
        n_his=n_his,
    )
    clock.mark_experiment_end()
    metadata = build_run_metadata(
        model="tlast",
        penetration=tag,
        seed=seed,
        device=device,
        parameter_count=param_count,
        training_loss="unmasked_huber_raw_delta_2",
        optimizer="AdamW",
        initial_lr=float(config["learning_rate"]),
        scheduler="author_cosine_warmup5_t100_decay0.1",
        max_epochs=epochs,
        epochs_executed=epochs_this_process if epochs_this_process > 0 else len(history),
        best_epoch=best_epoch,
        best_metric=float(best_mae),
        early_stopping=early,
        early_stopping_patience=patience if early else None,
        selection_metric="mean_validation_mae",
        clock=clock,
        extra={
            "init_parameter_sha256": init_hash,
            "code_version": CODE_VERSION,
            "data_file_sha256": train_hash,
            "normalization_sha256": data_info.get(
                "state_normalization_sha256", data_info["normalization_sha256"]
            ),
            "state_normalization_sha256": data_info.get("state_normalization_sha256"),
            "checkpoint_monitor": "mean_validation_mae",
            "early_stopped": early_stopped,
            "weight_decay": float(config.get("weight_decay", 1e-4)),
            "max_grad_norm": float(config.get("max_grad_norm", 1.0)),
            "huber_delta": float(config.get("huber_delta", AUTHOR_HUBER_DELTA)),
            "input_type": config.get("input_type"),
            "target_type": config.get("target_type"),
            "use_input_residual": bool(config.get("use_input_residual", False)),
            "proxy_num": int(config.get("proxy_num", config.get("M", 8))),
            "hid_dim": int(config.get("hid_dim", 64)),
            "n_heads": int(config.get("n_heads", 1)),
            "num_layers": int(config.get("num_layers", 1)),
            "tau": int(config.get("tau", 3)),
            "test_inference_sec": test_inference_sec,
            "batch_size": int(config["batch_size"]),
        },
    )
    output_check = finalize_completed_rate(
        rate_dir,
        metadata=metadata,
        history=history,
        expected_test_count=int(counts["test"]),
        expected_val_count=int(counts["validation"]),
        n_his=n_his,
        history_metric_key="val_mae",
        history_rule="lt",
    )
    result = {
        "status": "completed",
        "skipped_existing": False,
        "rate": tag,
        "best_epoch": best_epoch,
        "best_metric": float(best_mae),
        "early_stopped": early_stopped,
        "epochs": epochs,
        "total_epochs": len(history),
        "epochs_executed": len(history),
        "parameter_count": param_count,
        "init_parameter_sha256": init_hash,
        "validation_mae_raw": float(val_eval["metrics"]["mae"]),
        "validation_rmse_raw": float(val_eval["metrics"]["rmse"]),
        "validation_mape_nonzero": float(val_eval["metrics"]["mape_nonzero"]),
        "validation_wape": float(val_eval["metrics"]["wape"]),
        "test_mae_raw": float(test_eval["metrics"]["mae"]),
        "test_rmse_raw": float(test_eval["metrics"]["rmse"]),
        "test_mape_nonzero": float(test_eval["metrics"]["mape_nonzero"]),
        "test_wape": float(test_eval["metrics"]["wape"]),
        "test_negative_prediction_count": test_eval["metrics"].get("negative_prediction_count"),
        "test_negative_prediction_fraction": test_eval["metrics"].get("negative_prediction_fraction"),
        "did_not_use_test_for_selection": True,
        "checkpoint_monitor": "mean_validation_mae",
        "target_mode": "last-observed-step",
        "train_wall_time_sec": metadata["train_wall_time_sec"],
        "average_epoch_time_sec": metadata["average_epoch_time_sec"],
        "test_inference_sec": test_inference_sec,
        "peak_gpu_memory_allocated_mb": metadata.get("peak_gpu_memory_allocated_mb"),
        "output_check": output_check,
    }
    logger.log(
        {
            "stage": "completed",
            "model_name": "tlast",
            "penetration_rate": tag,
            "status": "ok",
            "best_epoch": best_epoch,
            "test_mae_raw": result["test_mae_raw"],
        }
    )
    print(
        (
            f"[tlast] stage=rate_completed rate={tag} best_epoch={best_epoch} "
            f"val_mae={result['validation_mae_raw']:.6f} test_mae={result['test_mae_raw']:.6f} "
            f"inference_sec={test_inference_sec:.3f}"
        ),
        flush=True,
    )
    return result


def run_all_rates(
    config: Mapping[str, Any],
    *,
    data_root: Path,
    output_root: Path,
    data_info: Mapping[str, Any],
    logger: JsonlLogger,
    resume_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    summaries = []
    rates = assert_official_rates(config["rates"])
    for rate in rates:
        try:
            summary = run_single_rate(
                config,
                rate=rate,
                data_root=data_root,
                output_root=output_root,
                data_info=data_info,
                logger=logger,
                resume_path=resume_path,
                overwrite=overwrite,
            )
            if str(summary.get("status")) != "completed":
                raise ReimplementationError(f"{summary.get('rate')} did not complete official test evaluation")
        except Exception as exc:
            mark_rate_failed(
                sampling_rate_dir(output_root, config, rate),
                reason=str(exc),
            )
            raise
        summaries.append(summary)
        _merge_summaries(output_root, summaries)
    return {"rates": summaries}


SUMMARY_FIELDS = (
    "rate",
    "status",
    "best_epoch",
    "best_metric",
    "parameter_count",
    "init_parameter_sha256",
    "validation_mae_raw",
    "validation_rmse_raw",
    "validation_mape_nonzero",
    "validation_wape",
    "test_mae_raw",
    "test_rmse_raw",
    "test_mape_nonzero",
    "test_wape",
    "test_negative_prediction_count",
    "test_negative_prediction_fraction",
)


def _merge_summaries(output_root: Path, summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge rate rows under a lock so parallel penetrations do not drop each other."""
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".summary.lock"
    with lock_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            merged = upsert_overall_summary(output_root / "overall_summary.json", summaries)
            atomic_write_csv_rows(
                output_root / "overall_summary.csv",
                list(SUMMARY_FIELDS),
                [{key: row.get(key) for key in SUMMARY_FIELDS} for row in merged],
            )
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return merged
