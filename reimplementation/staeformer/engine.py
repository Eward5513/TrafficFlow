"""Training engine for R-only STAEformer.

Official 7-rate epoch training is implemented for a later run. This session
only uses ``run_p70_smoke`` (one training batch on CPU) and never writes
``experiments/r_only``.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from reimplementation.common.data.r_only_npz_dataset import (
    SPLITS,
    ROnlyNPZDataset,
    build_dataloader,
    expected_split_counts,
    invert_target,
    load_json,
    rate_tag,
    read_target_scaler,
)
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.utils.atomic_io import atomic_write_json
from reimplementation.common.utils.checkpoint import (
    assert_checkpoint_compatible,
    load_checkpoint,
    save_checkpoint,
)
from reimplementation.common.utils.hashing import sha256_file
from reimplementation.common.utils.reproducibility import seed_everything
from reimplementation.common.utils.structured_logging import JsonlLogger
from reimplementation.stgcn.engine import resolve_device, resolve_path
from reimplementation.staeformer.data import assemble_history_and_indices
from reimplementation.staeformer.losses import HuberRawLoss
from reimplementation.staeformer.model.staeformer import CODE_VERSION, STAEformer
from reimplementation.staeformer.temporal_features import ORIGINAL_PEMS_NPZ_DOW, load_weekday_mapping
from reimplementation.staeformer.validation import (
    load_node_ids,
    validate_model_shapes,
    validate_no_graph_config,
    validate_node_order,
    validate_prepared_data,
    validate_train_npz_only,
)


def parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, param in sorted(model.named_parameters(), key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        array = np.ascontiguousarray(param.detach().cpu().numpy())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _cpu_rss_bytes() -> float | None:
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if sys.platform != "darwin":
        rss *= 1024.0
    return rss


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _day_args(config: Mapping[str, Any], data_root: Path) -> dict[str, Any]:
    mapping_value = config.get("weekday_mapping")
    mapping = None
    if mapping_value:
        mapping = load_weekday_mapping(resolve_path(Path(config.get("project_root", ".")), mapping_value))
    manifest = load_json(data_root / "split_manifest.json")
    return {
        "day_of_week_source": str(config.get("day_of_week_source", ORIGINAL_PEMS_NPZ_DOW)),
        "weekday_mapping": mapping,
        "all_days": [str(item) for item in manifest.get("all_days", [])],
    }


def assemble_batch_payload(
    batch: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    *,
    day_of_week_source: str,
    weekday_mapping: Mapping[str, int] | None,
    all_days: list[str] | None,
) -> dict[str, torch.Tensor]:
    return assemble_history_and_indices(
        batch["x"],
        window_start_slot=batch["window_start_slot"],
        day_index=batch["day_index"],
        n_his=int(config.get("in_steps", config.get("n_his", 12))),
        input_dim=int(config.get("input_dim", 3)),
        tod_embedding_dim=int(config.get("tod_embedding_dim", 24)),
        dow_embedding_dim=int(config.get("dow_embedding_dim", 24)),
        day_of_week_source=day_of_week_source,
        weekday_mapping=weekday_mapping,
        all_days=all_days,
        target_slot=batch["target_slot"],
    )


def model_kwargs_from_payload(
    payload: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    kwargs: dict[str, torch.Tensor] = {
        "time_of_day_index": payload["time_of_day_index"].to(device),
    }
    if "day_of_week_index" in payload:
        kwargs["day_of_week_index"] = payload["day_of_week_index"].to(device)
    return kwargs


def build_optimizer(model: torch.nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    name = str(config.get("optimizer", "Adam")).lower()
    if name != "adam":
        raise ReimplementationError(f"original PeMS04 STAEformer optimizer is Adam, got {name}")
    return Adam(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.001)),
        weight_decay=float(config.get("weight_decay", 0.0005)),
        eps=float(config.get("adam_eps", 1e-8)),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
) -> MultiStepLR | None:
    name = str(config.get("lr_scheduler", "MultiStepLR")).lower()
    if name in {"none", "off", "false"}:
        return None
    if name not in {"multisteplr", "multi_step", "multistep"}:
        raise ReimplementationError("original PeMS04 STAEformer scheduler is MultiStepLR")
    milestones = [int(item) for item in config.get("lr_milestones", [15, 30, 50])]
    return MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=float(config.get("lr_gamma", 0.1)),
    )


def build_model(config: Mapping[str, Any]) -> STAEformer:
    validate_no_graph_config(config)
    return STAEformer.from_config(config)


def build_train_dataloader(
    data_root: Path,
    rate: str,
    config: Mapping[str, Any],
    split_counts: Mapping[str, int],
) -> dict[str, Any]:
    tag = rate_tag(rate)
    path = validate_train_npz_only(data_root / tag / "train.npz")
    dataset = ROnlyNPZDataset(
        path,
        n_his=int(config.get("in_steps", config.get("n_his", 12))),
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
    n_his = int(config.get("in_steps", config.get("n_his", 12)))
    for split in SPLITS:
        dataset = ROnlyNPZDataset(
            data_root / rate_tag(rate) / f"{split}.npz",
            n_his=n_his,
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


def collect_smoke_data_info(data_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    n_his = int(config.get("in_steps", config.get("n_his", 12)))
    manifest = load_json(data_root / "split_manifest.json")
    normalization = load_json(data_root / "normalization.json")
    mean_y, std_y = read_target_scaler(normalization)
    counts = expected_split_counts(manifest)
    node_ids = load_node_ids(data_root / "node_mapping.csv")
    train_path = validate_train_npz_only(data_root / "p70" / "train.npz")
    dataset = ROnlyNPZDataset(
        train_path,
        n_his=n_his,
        num_nodes=int(config["num_nodes"]),
        expected_count=int(counts["train"]),
    )
    recovered = invert_target(dataset.y[:8], mean_y, std_y)
    if float(np.max(np.abs(recovered - dataset.y_raw[:8]))) > 1e-3:
        raise ReimplementationError("mean_y_full/std_y_full does not recover y_raw")
    return {
        "sample_counts": counts,
        "mean_y_full": mean_y,
        "std_y_full": std_y,
        "node_ids": node_ids,
        "normalization_sha256": sha256_file(data_root / "normalization.json"),
        "split_manifest_target_mode": manifest.get("target_mode"),
        "all_days": [str(item) for item in manifest.get("all_days", [])],
        "file_hashes": {"p70/train.npz": sha256_file(train_path)},
        "did_not_read_validation_or_test": True,
        "did_not_read_adjacency": True,
    }


def collect_data_info(
    data_root: Path,
    config: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    n_his = int(config.get("in_steps", config.get("n_his", 12)))
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
    return data_info


def _gradient_flags(model: STAEformer) -> dict[str, bool]:
    flags = {
        "feature_projection": False,
        "tod_embedding": False if model.tod_embedding_dim > 0 else True,
        "dow_embedding": False if model.dow_embedding_dim > 0 else True,
        "spatial_embedding": False if model.spatial_embedding_dim > 0 else True,
        "adaptive_embedding": False if model.adaptive_embedding_dim > 0 else True,
        "temporal_attention": False,
        "spatial_attention": False,
        "ffn": False,
        "output_projection": False,
    }
    for name, tensor in model.named_parameters():
        if tensor.grad is None or not torch.isfinite(tensor.grad).all():
            raise ReimplementationError(f"smoke missing or non-finite grad: {name}")
        max_abs = float(tensor.grad.abs().max().cpu())
        if max_abs <= 0.0:
            continue
        if name.startswith("input_proj"):
            flags["feature_projection"] = True
        if name.startswith("tod_embedding"):
            flags["tod_embedding"] = True
        if name.startswith("dow_embedding"):
            flags["dow_embedding"] = True
        if name.startswith("node_emb"):
            flags["spatial_embedding"] = True
        if name.startswith("adaptive_embedding"):
            flags["adaptive_embedding"] = True
        if name.startswith("attn_layers_t"):
            flags["temporal_attention"] = True
            if "feed_forward" in name:
                flags["ffn"] = True
        if name.startswith("attn_layers_s"):
            flags["spatial_attention"] = True
            if "feed_forward" in name:
                flags["ffn"] = True
        if "output_proj" in name or name.startswith("temporal_proj"):
            flags["output_projection"] = True
    required = {
        "feature_projection": True,
        "output_projection": True,
        "temporal_attention": True,
        "spatial_attention": True,
        "ffn": True,
        "tod_embedding": model.tod_embedding_dim > 0,
        "dow_embedding": model.dow_embedding_dim > 0,
        "adaptive_embedding": model.adaptive_embedding_dim > 0,
        "spatial_embedding": model.spatial_embedding_dim > 0,
    }
    missing = [key for key, need in required.items() if need and not flags[key]]
    if missing:
        raise ReimplementationError(f"smoke missing gradients: {missing}")
    return flags


def run_p70_smoke(
    config: Mapping[str, Any],
    *,
    data_root: Path,
    data_info: Mapping[str, Any],
    logger: JsonlLogger | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    validate_no_graph_config(config)
    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device("cpu")
    if project_root is not None:
        validate_node_order(
            r_nodes_path=resolve_path(project_root, config["r_nodes"]),
            node_mapping=resolve_path(project_root, config["node_mapping"]),
            expected_nodes=int(config["num_nodes"]),
        )
    train_pack = build_train_dataloader(data_root, "p70", config, data_info["sample_counts"])
    model = build_model(config).to(device)
    init_hash = parameter_sha256(model)
    day = _day_args(config, data_root)
    probe = _move_batch(next(iter(train_pack["loader"])), device)
    payload_tensors = assemble_batch_payload(probe, config, **day)
    history = payload_tensors["history"].to(device)
    ids = model_kwargs_from_payload(payload_tensors, device)
    shape_info = validate_model_shapes(
        model,
        batch_size=int(history.size(0)),
        history=history,
        **ids,
    )
    optimizer = build_optimizer(model, config)
    criterion = HuberRawLoss(
        float(data_info["mean_y_full"]),
        float(data_info["std_y_full"]),
        delta=float(config.get("huber_delta", 1.0)),
    )
    model.train()
    pred, trace = model(history, return_trace=True, **ids)
    y = probe["y"]
    if pred.shape != y.shape:
        raise ReimplementationError(f"smoke prediction {tuple(pred.shape)} != target {tuple(y.shape)}")
    if not torch.isfinite(pred).all():
        raise ReimplementationError("smoke prediction is not finite")
    loss = criterion(pred, probe["y_raw"])
    if not torch.isfinite(loss):
        raise ReimplementationError(f"smoke loss is not finite: {loss}")
    before = {name: tensor.detach().cpu().clone() for name, tensor in model.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    flags = _gradient_flags(model)
    clip_grad = config.get("clip_grad", None)
    if clip_grad not in (None, False, 0, 0.0) or bool(config.get("clip_grad_norm", False)):
        max_norm = float(clip_grad if clip_grad not in (None, False, 0, 0.0) else config.get("max_grad_norm", 5.0))
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
    changed = 0
    for name, tensor in model.named_parameters():
        if not torch.equal(tensor.detach().cpu(), before[name]):
            changed += 1
    if changed == 0:
        raise ReimplementationError("optimizer step did not change parameters")
    recovered = invert_target(probe["y"], float(data_info["mean_y_full"]), float(data_info["std_y_full"]))
    y_raw_match = bool(torch.allclose(recovered, probe["y_raw"], atol=1e-4, rtol=1e-4))
    payload = {
        "status": "ok",
        "penetration_rate": "p70",
        "batch_size": int(probe["x"].size(0)),
        "device": str(device),
        "seed": seed,
        "input_shape": list(probe["x"].shape),
        "history_shape": list(history.shape),
        "tod_shape": list(ids["time_of_day_index"].shape),
        "dow_shape": list(ids["day_of_week_index"].shape) if "day_of_week_index" in ids else None,
        "target_shape": list(y.shape),
        "prediction_shape": list(pred.shape),
        "trace": trace,
        "training_loss": float(loss.detach().cpu()),
        "parameter_count": model.parameter_count(),
        "expected_parameter_count": model.expected_parameter_count(),
        "model_dim": model.model_dim,
        "init_parameter_sha256": init_hash,
        "parameters_changed": changed,
        "gradient_status": flags,
        "y_raw_roundtrip_ok": y_raw_match,
        "zero_targets_in_batch": bool((probe["y_raw"] == 0).any()),
        "used_adjacency": False,
        "spatial_embedding_dim": model.spatial_embedding_dim,
        "use_mixed_proj": model.use_mixed_proj,
        "attention_mask": model.attention_mask,
        "discarded_after_smoke": True,
        "did_not_write_official_checkpoint": True,
        "did_not_read_validation_or_test": True,
        "shape_info": shape_info,
        "cpu_memory_rss_bytes": _cpu_rss_bytes(),
        "code_version": CODE_VERSION,
        "day_of_week_source": day["day_of_week_source"],
        "task": "current-time full-flow reconstruction from a 12-step partially observed sequence",
    }
    if logger is not None:
        logger.log(
            {
                "stage": "smoke_testing",
                "model_name": "staeformer",
                "penetration_rate": "p70",
                "seed": seed,
                "status": "ok",
                "training_loss": payload["training_loss"],
                "device": str(device),
            }
        )
    print(
        f"[staeformer] stage=smoke_testing rate=p70 loss={payload['training_loss']:.6f} status=ok",
        flush=True,
    )
    del model
    del optimizer
    del train_pack
    return payload


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
    """Official one-rate training. Not called by this migration session."""
    tag = rate_tag(rate)
    seed = int(config["seed"])
    seed_everything(seed)
    device = resolve_device(str(config.get("device", "cpu")))
    rate_dir = output_root / tag / f"seed_{seed}"
    rate_dir.mkdir(parents=True, exist_ok=True)
    loaders_pack = build_dataloaders(data_root, tag, config, data_info["sample_counts"])
    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    criterion = HuberRawLoss(
        float(data_info["mean_y_full"]),
        float(data_info["std_y_full"]),
        delta=float(config.get("huber_delta", 1.0)),
    )
    start_epoch = 0
    best_mae = float("inf")
    wait = 0
    early_stop = int(config.get("early_stop", 20))
    train_hash = data_info["file_hashes"][f"{tag}/train.npz"]
    day = _day_args(config, data_root)
    if resume_path is not None:
        payload = load_checkpoint(resume_path, map_location="cpu")
        assert_checkpoint_compatible(
            payload,
            penetration_rate=tag,
            graph_sha256="",
            data_file_sha256=train_hash,
            num_nodes=int(config["num_nodes"]),
            n_his=int(config.get("in_steps", 12)),
            output_steps=1,
            input_channels=1,
            output_channels=1,
        )
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
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
            "normalization_sha256": data_info["normalization_sha256"],
            "code_version": CODE_VERSION,
        }

    history: list[dict[str, Any]] = []
    epochs = int(config.get("epochs", 300))
    stopped_epoch = start_epoch
    for epoch in range(start_epoch, epochs):
        stopped_epoch = epoch
        model.train()
        train_losses = []
        for batch in loaders_pack["loaders"]["train"]:
            batch = _move_batch(batch, device)
            packed = assemble_batch_payload(batch, config, **day)
            pred = model(packed["history"].to(device), **model_kwargs_from_payload(packed, device))
            if pred.shape != batch["y"].shape:
                raise ReimplementationError("prediction/target shape mismatch")
            loss = criterion(pred, batch["y_raw"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad = config.get("clip_grad", None)
            if clip_grad not in (None, False, 0, 0.0) or bool(config.get("clip_grad_norm", False)):
                max_norm = float(
                    clip_grad if clip_grad not in (None, False, 0, 0.0) else config.get("max_grad_norm", 5.0)
                )
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        if scheduler is not None:
            scheduler.step()
        model.eval()
        val_abs = []
        with torch.no_grad():
            for batch in loaders_pack["loaders"]["validation"]:
                batch = _move_batch(batch, device)
                packed = assemble_batch_payload(batch, config, **day)
                pred = model(packed["history"].to(device), **model_kwargs_from_payload(packed, device))
                pred_raw = invert_target(
                    pred, float(data_info["mean_y_full"]), float(data_info["std_y_full"])
                )
                val_abs.append(torch.abs(pred_raw - batch["y_raw"]).mean().cpu())
        val_mae = float(torch.stack(val_abs).mean()) if val_abs else float("inf")
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_mae": val_mae})
        save_checkpoint(rate_dir / "last_checkpoint.pt", checkpoint_payload(epoch, best_mae))
        if val_mae < best_mae:
            best_mae = val_mae
            wait = 0
            save_checkpoint(rate_dir / "best_checkpoint.pt", checkpoint_payload(epoch, best_mae))
        else:
            wait += 1
            if wait >= early_stop:
                break
        logger.log(
            {
                "stage": "train_epoch",
                "model_name": "staeformer",
                "penetration_rate": tag,
                "epoch": epoch,
                "train_loss": history[-1]["train_loss"],
                "validation_mae_raw_all": val_mae,
            }
        )
    atomic_write_json(rate_dir / "training_history.json", history)
    return {
        "rate": tag,
        "best_mae": best_mae,
        "epochs": stopped_epoch + 1,
        "early_stop": early_stop,
        "init_reset_seed": seed,
    }


def run_all_rates(
    config: Mapping[str, Any],
    *,
    data_root: Path,
    output_root: Path,
    data_info: Mapping[str, Any],
    logger: JsonlLogger,
    resume_path: Path | None = None,
) -> dict[str, Any]:
    summaries = []
    for rate in config["rates"]:
        summaries.append(
            run_single_rate(
                config,
                rate=rate,
                data_root=data_root,
                output_root=output_root,
                data_info=data_info,
                logger=logger,
                resume_path=resume_path,
            )
        )
    return {"rates": summaries}
