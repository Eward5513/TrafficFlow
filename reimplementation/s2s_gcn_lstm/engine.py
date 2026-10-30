"""Train and evaluate one Xuancheng26 penetration at a time.

Checkpoint selection uses validation MAE on raw counts, including zeros.
The test split is scored once after the best checkpoint is reloaded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from reimplementation.common.data.r_only_npz_dataset import (
    SPLITS,
    ROnlyNPZDataset,
    build_dataloader,
    expected_split_counts,
    invert_target,
    loss_scale,
    load_json,
    load_npz,
    rate_tag,
    read_target_scaler,
)
from reimplementation.common.data.state_windows import finalize_state_scaler, open_split_dataset
from reimplementation.common.evaluation import evaluate_loader, reload_best_checkpoint, upsert_overall_summary
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
from reimplementation.s2s_gcn_lstm.losses import MAERawLoss
from reimplementation.s2s_gcn_lstm.model.s2s_gcn_lstm import CODE_VERSION, MODEL_LABEL, MODEL_NAME, S2SGCNLSTM
from reimplementation.s2s_gcn_lstm.validation import (
    assert_feature_order,
    validate_model_shapes,
    validate_prepared_data,
    validate_s2s_graph,
)
from reimplementation.stgcn.engine import resolve_device, resolve_path
from reimplementation.stgcn.validation import load_edge_ids


def parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, param in sorted(model.named_parameters(), key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        array = np.ascontiguousarray(param.detach().cpu().numpy())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def build_model(config: Mapping[str, Any], adjacency: np.ndarray) -> S2SGCNLSTM:
    return S2SGCNLSTM.from_config(config, adjacency)


def build_optimizer(model: torch.nn.Module, config: Mapping[str, Any]) -> Adam:
    if str(config.get("optimizer", "Adam")).lower() != "adam":
        raise ReimplementationError("this reimplementation trains with Adam, matching the paper's optimizer")
    return Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )


def build_scheduler(optimizer: Adam, config: Mapping[str, Any]) -> ReduceLROnPlateau:
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config.get("lr_factor", 0.5)),
        patience=int(config.get("lr_patience", 5)),
        threshold=float(config.get("lr_threshold", 1e-4)),
        min_lr=float(config.get("min_lr", 1e-6)),
    )


def build_dataloaders(
    data_root: Path,
    rate: str,
    config: Mapping[str, Any],
    split_counts: Mapping[str, int],
) -> dict[str, Any]:
    loaders = {}
    datasets = {}
    n_his = int(config.get("n_his", 12))
    tag = rate_tag(rate)
    for split in SPLITS:
        dataset = open_split_dataset(
            data_root,
            tag,
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


def _graph_payload(config: Mapping[str, Any], *, project_root: Path) -> dict[str, Any]:
    graph = validate_s2s_graph(dict(config), project_root=project_root)
    if not np.isfinite(graph["a_hat"]).all():
        raise ReimplementationError("normalized adjacency contains NaN or inf")
    return graph


def _weight_sha256(graph: Mapping[str, Any]) -> str:
    if graph["edge_weight"] in {"distance", "gaussian"}:
        digest = graph.get("distance_sha256")
        if not digest:
            raise ReimplementationError("distance adjacency has no SHA256")
        return str(digest)
    return str(graph["graph_sha256"])


def collect_smoke_data_info(
    data_root: Path,
    config: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    manifest = load_json(data_root / "split_manifest.json")
    normalization = load_json(data_root / "normalization.json")
    if str(manifest.get("target_mode")) != "last-observed-step":
        raise ReimplementationError("split manifest is not last-observed-step reconstruction")
    if bool(normalization.get("used_test_to_fit")) or bool(normalization.get("used_validation_to_fit")):
        raise ReimplementationError("normalization.json was not fit on training only")
    mean_y, std_y = read_target_scaler(normalization)
    counts = expected_split_counts(manifest)
    graph = _graph_payload(config, project_root=project_root)
    window_ids = load_edge_ids(data_root / "node_mapping.csv")
    assert_feature_order(window_ids, graph["node_ids"])
    train_path = data_root / "p70" / "train.npz"
    arrays = load_npz(train_path)
    if not np.array_equal(arrays["target_slot"], arrays["window_start_slot"] + 11):
        raise ReimplementationError("p70 train target is not the last observed slot")
    recovered = invert_target(arrays["y"][:4], mean_y, std_y)
    if float(np.max(np.abs(recovered - arrays["y_raw"][:4]))) > 1e-3:
        raise ReimplementationError("mean_y_full/std_y_full does not recover y_raw")
    return {
        "sample_counts": counts,
        "mean_y_full": mean_y,
        "std_y_full": std_y,
        "node_ids": window_ids,
        "adjacency": graph["adjacency"],
        "graph_sha256": _weight_sha256(graph),
        "edge_weight": graph["edge_weight"],
        "undirected_neighbor_count": graph["undirected_neighbor_count"],
        "normalization_sha256": sha256_file(data_root / "normalization.json"),
        "split_manifest_target_mode": manifest.get("target_mode"),
        "file_hashes": {"p70/train.npz": sha256_file(train_path)},
        "did_not_read_validation_or_test": True,
        "did_not_use_gaussian_kernel": bool(graph["did_not_use_gaussian_kernel"]),
    }


def collect_data_info(
    data_root: Path,
    config: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    data_info = validate_prepared_data(
        data_root,
        n_his=int(config.get("n_his", 12)),
        num_nodes=int(config["num_nodes"]),
        rates=[rate_tag(item) for item in config["rates"]],
    )
    graph = _graph_payload(config, project_root=project_root)
    assert_feature_order(list(data_info["node_ids"]), graph["node_ids"])
    data_info["adjacency"] = graph["adjacency"]
    data_info["graph_sha256"] = _weight_sha256(graph)
    data_info["edge_weight"] = graph["edge_weight"]
    data_info["undirected_neighbor_count"] = graph["undirected_neighbor_count"]
    data_info["did_not_use_gaussian_kernel"] = bool(graph["did_not_use_gaussian_kernel"])
    return data_info


def _gradient_flags(model: S2SGCNLSTM) -> dict[str, bool]:
    flags = {
        "encoder_gcn": False,
        "encoder_lstm": False,
        "decoder_gcn": False,
        "decoder_lstm": False,
        "fc": False,
    }
    for name, tensor in model.named_parameters():
        if tensor.grad is None or not torch.isfinite(tensor.grad).all():
            raise ReimplementationError(f"smoke missing or non-finite grad: {name}")
        if float(tensor.grad.detach().abs().sum().cpu()) <= 0.0:
            continue
        if name.startswith("encoder_gcn."):
            flags["encoder_gcn"] = True
        elif name.startswith("encoder_lstm."):
            flags["encoder_lstm"] = True
        elif name.startswith("decoder_gcn."):
            flags["decoder_gcn"] = True
        elif name.startswith("decoder_lstm."):
            flags["decoder_lstm"] = True
        elif name.startswith("fc."):
            flags["fc"] = True
    missing = [key for key, ok in flags.items() if not ok]
    if missing:
        raise ReimplementationError(f"smoke missing gradients: {missing}")
    return flags


def run_p70_smoke(
    config: Mapping[str, Any],
    *,
    data_root: Path,
    data_info: Mapping[str, Any],
    project_root: Path,
    logger: JsonlLogger | None = None,
) -> dict[str, Any]:
    seed_everything(int(config["seed"]))
    device = torch.device("cpu")
    dataset = ROnlyNPZDataset(
        data_root / "p70" / "train.npz",
        n_his=int(config.get("n_his", 12)),
        num_nodes=int(config["num_nodes"]),
        input_channels=1,
        output_channels=1,
        output_steps=1,
        expected_count=int(data_info["sample_counts"]["train"]),
    )
    loader = build_dataloader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0,
        seed=int(config["seed"]),
    )
    model = build_model(config, data_info["adjacency"]).to(device)
    shape_info = validate_model_shapes(model, batch_size=2)
    optimizer = build_optimizer(model, config)
    criterion = MAERawLoss(loss_scale(data_info["mean_y_full"]), loss_scale(data_info["std_y_full"]))
    probe = _move_batch(next(iter(loader)), device)
    model.train()
    prediction, trace = model(probe["x"], return_trace=True)
    if tuple(prediction.shape) != tuple(probe["y"].shape):
        raise ReimplementationError(
            f"smoke prediction {tuple(prediction.shape)} != target {tuple(probe['y'].shape)}"
        )
    if int(trace["encoder_gcn_steps"]) != 12 or int(trace["decoder_lstm_steps"]) != 1:
        raise ReimplementationError(f"smoke trace is not 12-to-1: {trace}")
    if int(trace["decoder_gcn_calls"]) != 1:
        raise ReimplementationError("smoke decoder GCN did not run exactly once")
    loss = criterion(prediction, probe["y_raw"])
    if not torch.isfinite(loss):
        raise ReimplementationError(f"smoke loss is not finite: {loss}")
    before = {name: tensor.detach().cpu().clone() for name, tensor in model.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    flags = _gradient_flags(model)
    clip = float(config.get("gradient_clip", 0.0))
    if clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    changed = sum(
        1
        for name, tensor in model.named_parameters()
        if not torch.equal(tensor.detach().cpu(), before[name])
    )
    if changed == 0:
        raise ReimplementationError("optimizer step did not change parameters")
    payload = {
        "status": "ok",
        "model_label": MODEL_LABEL,
        "penetration_rate": "p70",
        "batch_size": int(probe["x"].shape[0]),
        "device": str(device),
        "seed": int(config["seed"]),
        "input_shape": list(probe["x"].shape),
        "target_shape": list(probe["y"].shape),
        "prediction_shape": list(prediction.shape),
        "trace": trace,
        "training_loss": float(loss.detach().cpu()),
        "parameter_count": model.parameter_count(),
        "parameters_changed": changed,
        "gradient_status": flags,
        "edge_weight": data_info["edge_weight"],
        "undirected_neighbor_count": data_info["undirected_neighbor_count"],
        "a_hat_finite": bool(torch.isfinite(model.a_hat).all()),
        "discarded_after_smoke": True,
        "did_not_write_official_checkpoint": True,
        "did_not_read_validation_or_test": True,
        "shape_info": {"output_shape": shape_info["output_shape"], "parameter_count": shape_info["parameter_count"]},
        "code_version": CODE_VERSION,
        "project_root": project_root.as_posix(),
    }
    if logger is not None:
        logger.log({"stage": "smoke_testing", "model_name": MODEL_NAME, "status": "ok"})
    del model
    del optimizer
    return payload


def _validation_mae(
    model: S2SGCNLSTM,
    loader,
    *,
    mean_y: float,
    std_y: float,
    device: torch.device,
) -> float:
    total = torch.zeros((), dtype=torch.float64)
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            prediction = model(batch["x"])
            if prediction.shape != batch["y"].shape:
                raise ReimplementationError("prediction/target shape mismatch")
            raw = invert_target(prediction, mean_y, std_y)
            error = torch.abs(raw - batch["y_raw"]).detach().to(dtype=torch.float64)
            total += error.sum().cpu()
            count += int(error.numel())
    if count <= 0:
        raise ReimplementationError("validation loader produced zero elements")
    return float(total / count)


def run_single_rate(
    config: Mapping[str, Any],
    *,
    rate: str,
    data_root: Path,
    output_root: Path,
    data_info: Mapping[str, Any],
    logger: JsonlLogger,
    resume_path: Path | None = None,
) -> dict[str, Any]:
    data_info = finalize_state_scaler(data_info, config, data_root)
    tag = rate_tag(rate)
    seed = int(config["seed"])
    seed_everything(seed)
    device = resolve_device(str(config.get("device", "cpu")))
    rate_dir = output_root / tag / f"seed_{seed}"
    rate_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(rate_dir / "resolved_config.json", dict(config))
    loaders = build_dataloaders(data_root, tag, config, data_info["sample_counts"])
    model = build_model(config, data_info["adjacency"]).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    criterion = MAERawLoss(loss_scale(data_info["mean_y_full"]), loss_scale(data_info["std_y_full"]))
    start_epoch = 0
    best_mae = float("inf")
    best_epoch = -1
    wait = 0
    history: list[dict[str, Any]] = []
    train_hash = data_info["file_hashes"][f"{tag}/train.npz"]
    graph_sha = str(data_info["graph_sha256"])
    clock = ExperimentClock()
    clock.mark_experiment_start()
    init_hash = parameter_sha256(model)
    if resume_path is not None:
        payload = load_checkpoint(resume_path, map_location="cpu")
        assert_checkpoint_compatible(
            payload,
            penetration_rate=tag,
            graph_sha256=graph_sha,
            data_file_sha256=train_hash,
            num_nodes=int(config["num_nodes"]),
            n_his=int(config.get("n_his", 12)),
            output_steps=1,
            input_channels=1,
            output_channels=1,
        )
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if payload.get("scheduler_state_dict"):
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        best_mae = float(payload.get("best_metric", best_mae))
        prior_history = rate_dir / "training_history.json"
        if prior_history.is_file():
            loaded = load_json(prior_history)
            if not isinstance(loaded, list):
                raise ReimplementationError("training_history.json is not a list")
            history = [dict(row) for row in loaded]
            best_row = min(history, key=lambda row: (float(row["val_mae"]), int(row["epoch"])))
            best_mae = float(best_row["val_mae"])
            best_epoch = int(best_row["epoch"])
            wait = sum(1 for row in history if int(row["epoch"]) > best_epoch)
        else:
            best_epoch = int(payload.get("best_epoch", payload.get("epoch", -1)))

    def checkpoint_payload(epoch: int, metric: float) -> dict[str, Any]:
        return {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_epoch": int(best_epoch),
            "best_metric": metric,
            "config": dict(config),
            "penetration_rate": tag,
            "seed": seed,
            "model_parameter_count": model.parameter_count(),
            "graph_sha256": graph_sha,
            "data_file_sha256": train_hash,
            "normalization_sha256": data_info["normalization_sha256"],
            "code_version": CODE_VERSION,
        }
    epochs = int(config["epochs"])
    patience = int(config.get("early_stopping_patience", 20))
    early_stop_enabled = patience > 0
    early_stopped = False
    param_count = trainable_parameter_count(model)
    reset_peak_memory(device)
    clock.mark_train_start()
    for epoch in range(start_epoch, epochs):
        model.train()
        train_losses: list[float] = []
        for batch in loaders["loaders"]["train"]:
            batch = _move_batch(batch, device)
            prediction = model(batch["x"])
            if prediction.shape != batch["y"].shape:
                raise ReimplementationError("prediction/target shape mismatch")
            loss = criterion(prediction, batch["y_raw"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip = float(config.get("gradient_clip", 0.0))
            if clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        if not train_losses:
            raise ReimplementationError("training loader produced zero batches")
        learning_rate = float(optimizer.param_groups[0]["lr"])
        val_mae = _validation_mae(
            model,
            loaders["loaders"]["validation"],
            mean_y=loss_scale(data_info["mean_y_full"]),
            std_y=loss_scale(data_info["std_y_full"]),
            device=device,
        )
        scheduler.step(val_mae)
        improved = val_mae < best_mae
        if improved:
            best_mae = val_mae
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_mae": val_mae,
                "lr": learning_rate,
            }
        )
        save_checkpoint(rate_dir / "last_checkpoint.pt", checkpoint_payload(epoch, best_mae))
        if improved:
            save_checkpoint(rate_dir / "best_checkpoint.pt", checkpoint_payload(epoch, best_mae))
        logger.log(
            {
                "stage": "train_epoch",
                "model_name": MODEL_NAME,
                "penetration_rate": tag,
                "epoch": epoch,
                "train_loss": history[-1]["train_loss"],
                "validation_mae_raw_all": val_mae,
                "lr": learning_rate,
            }
        )
        if early_stop_enabled and wait >= patience:
            early_stopped = True
            break
    clock.mark_train_end()
    atomic_write_json(rate_dir / "training_history.json", history)
    if not history or best_epoch < 0:
        raise ReimplementationError("training produced no checkpoint")
    ckpt = reload_best_checkpoint(model, rate_dir / "best_checkpoint.pt", device)
    if int(ckpt["epoch"]) != int(best_epoch):
        raise ReimplementationError(f"best checkpoint epoch {ckpt['epoch']} != selected epoch {best_epoch}")
    mean_y = loss_scale(data_info["mean_y_full"])
    std_y = loss_scale(data_info["std_y_full"])
    n_his = int(config.get("n_his", 12))
    counts = data_info["sample_counts"]
    model.eval()

    def _predict(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return model(batch["x"])

    logger.log({"stage": "validating", "model_name": MODEL_NAME, "penetration_rate": tag, "status": "ok"})
    val_eval = evaluate_loader(
        loaders["loaders"]["validation"],
        _predict,
        mean_y=mean_y,
        std_y=std_y,
        device=device,
        output_dir=rate_dir,
        split="validation",
        expected_count=int(counts["validation"]),
        n_his=n_his,
    )
    logger.log({"stage": "testing", "model_name": MODEL_NAME, "penetration_rate": tag, "status": "ok"})
    test_eval = evaluate_loader(
        loaders["loaders"]["test"],
        _predict,
        mean_y=mean_y,
        std_y=std_y,
        device=device,
        output_dir=rate_dir,
        split="test",
        expected_count=int(counts["test"]),
        n_his=n_his,
    )
    clock.mark_experiment_end()
    epochs_executed = sum(1 for row in history if int(row["epoch"]) >= start_epoch)
    metadata = build_run_metadata(
        model=MODEL_NAME,
        penetration=tag,
        seed=seed,
        device=device,
        parameter_count=param_count,
        training_loss="unmasked_raw_mae",
        optimizer="Adam",
        initial_lr=float(config["learning_rate"]),
        scheduler="ReduceLROnPlateau_factor_0.5",
        max_epochs=epochs,
        epochs_executed=epochs_executed,
        best_epoch=best_epoch,
        best_metric=float(best_mae),
        early_stopping=early_stop_enabled,
        early_stopping_patience=patience if early_stop_enabled else None,
        selection_metric="validation_mae_raw_including_zero",
        clock=clock,
        extra={
            "model_label": MODEL_LABEL,
            "init_parameter_sha256": init_hash,
            "code_version": CODE_VERSION,
            "data_file_sha256": train_hash,
            "graph_sha256": graph_sha,
            "edge_weight": data_info["edge_weight"],
            "normalization_sha256": data_info["normalization_sha256"],
            "early_stopped": early_stopped,
            "output_activation": "linear",
            "decoder": "fixed_context_zero_state",
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
        "rate": tag,
        "best_epoch": best_epoch,
        "best_metric": float(best_mae),
        "early_stopped": early_stopped,
        "epochs": epochs,
        "epochs_executed": epochs_executed,
        "parameter_count": param_count,
        "validation_mae_raw": float(val_eval["metrics"]["mae"]),
        "validation_rmse_raw": float(val_eval["metrics"]["rmse"]),
        "validation_mape_nonzero": float(val_eval["metrics"]["mape_nonzero"]),
        "validation_wape": float(val_eval["metrics"]["wape"]),
        "test_mae_raw": float(test_eval["metrics"]["mae"]),
        "test_rmse_raw": float(test_eval["metrics"]["rmse"]),
        "test_mape_nonzero": float(test_eval["metrics"]["mape_nonzero"]),
        "test_wape": float(test_eval["metrics"]["wape"]),
        "did_not_use_test_for_selection": True,
        "checkpoint_monitor": "validation_mae_raw_including_zero",
        "target_mode": "last-observed-step",
        "train_wall_time_sec": metadata["train_wall_time_sec"],
        "average_epoch_time_sec": metadata["average_epoch_time_sec"],
        "output_check": output_check,
    }
    logger.log(
        {
            "stage": "completed",
            "model_name": MODEL_NAME,
            "penetration_rate": tag,
            "status": "ok",
            "best_epoch": best_epoch,
            "test_mae_raw": result["test_mae_raw"],
        }
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
) -> dict[str, Any]:
    rates = assert_official_rates(config["rates"])
    if resume_path is not None and len(rates) != 1:
        raise ReimplementationError("resume one penetration at a time; do not share checkpoints across rates")
    summaries = []
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
            )
        except Exception as exc:
            mark_rate_failed(output_root / rate_tag(rate) / f"seed_{int(config['seed'])}", reason=str(exc))
            raise
        summaries.append(summary)
        upsert_overall_summary(output_root / "overall_summary.json", summaries)
    return {"rates": summaries}
