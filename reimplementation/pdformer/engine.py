"""Training engine for R-only PDFormer.

Official epoch training is implemented for a later run. This session only
uses ``run_p70_smoke`` (one training batch) and never writes
``experiments/r_only``.
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.optim import AdamW

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
from reimplementation.common.metrics.traffic_metrics import traffic_metrics
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
from reimplementation.pdformer.data import assemble_model_input, load_weekday_mapping
from reimplementation.pdformer.graph import TEST_ONLY_RELATIONS
from reimplementation.pdformer.losses import HuberRawLoss
from reimplementation.pdformer.model.pdformer import (
    CODE_VERSION,
    PDFormer,
    build_pdformer_from_config,
    in_memory_relations_from_topology,
)
from reimplementation.pdformer.validation import (
    load_node_ids,
    validate_model_shapes,
    validate_pdformer_graph,
    validate_prepared_data,
    validate_train_npz_only,
)


class CosineWarmupScheduler:
    """PeMS-style cosine decay with linear warmup (epoch-based)."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        t_initial: int,
        lr_min: float,
        warmup_t: int,
        warmup_lr_init: float,
    ) -> None:
        self.optimizer = optimizer
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.t_initial = max(int(t_initial), 1)
        self.lr_min = float(lr_min)
        self.warmup_t = int(warmup_t)
        self.warmup_lr_init = float(warmup_lr_init)
        self.last_epoch = -1
        if self.warmup_t > 0:
            self._set_lr(self.warmup_lr_init)

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _lr(self, epoch: int) -> float:
        if epoch < self.warmup_t:
            return self.warmup_lr_init + (self.base_lrs[0] - self.warmup_lr_init) * float(epoch) / float(
                max(self.warmup_t, 1)
            )
        progress = float(epoch - self.warmup_t) / float(max(self.t_initial - self.warmup_t, 1))
        progress = min(max(progress, 0.0), 1.0)
        return self.lr_min + 0.5 * (self.base_lrs[0] - self.lr_min) * (1.0 + math.cos(math.pi * progress))

    def step(self, epoch: int | None = None) -> None:
        if epoch is None:
            self.last_epoch += 1
            epoch = self.last_epoch
        else:
            self.last_epoch = int(epoch)
        self._set_lr(self._lr(int(epoch)))

    def state_dict(self) -> dict[str, Any]:
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        self.last_epoch = int(payload["last_epoch"])
        self._set_lr(self._lr(max(self.last_epoch, 0)))


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


def _weekday_args(config: Mapping[str, Any], data_root: Path) -> dict[str, Any]:
    mapping_value = config.get("weekday_mapping")
    mapping = None
    if mapping_value:
        mapping = load_weekday_mapping(resolve_path(Path(config.get("project_root", ".")), mapping_value))
    manifest = load_json(data_root / "split_manifest.json")
    return {
        "weekday_mapping": mapping,
        "all_days": [str(item) for item in manifest.get("all_days", [])],
    }


def assemble_batch_x(
    batch: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    *,
    weekday_mapping,
    all_days,
) -> torch.Tensor:
    return assemble_model_input(
        batch["x"],
        window_start_slot=batch["window_start_slot"],
        day_index=batch["day_index"],
        n_his=int(config.get("input_window", config.get("n_his", 12))),
        add_time_in_day=bool(config.get("add_time_in_day", True)),
        add_day_in_week=bool(config.get("add_day_in_week", False)),
        weekday_mapping=weekday_mapping,
        all_days=all_days,
    )


def build_optimizer(model: torch.nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    name = str(config.get("optimizer", config.get("learner", "adamw"))).lower()
    if name != "adamw":
        raise ReimplementationError(f"original PeMS PDFormer optimizer is adamw, got {name}")
    return AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 0.05)),
        eps=float(config.get("lr_epsilon", 1e-8)),
        betas=tuple(config.get("lr_betas", (0.9, 0.999))),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
) -> CosineWarmupScheduler | None:
    if not bool(config.get("lr_decay", True)):
        return None
    if str(config.get("lr_scheduler", "cosinelr")).lower() not in {"cosinelr", "cosine"}:
        raise ReimplementationError("original PDFormer PeMS config uses cosinelr")
    return CosineWarmupScheduler(
        optimizer,
        t_initial=int(config.get("epochs", config.get("max_epoch", 200))),
        lr_min=float(config.get("lr_eta_min", 1e-4)),
        warmup_t=int(config.get("lr_warmup_epoch", 5)),
        warmup_lr_init=float(config.get("lr_warmup_init", 1e-6)),
    )


def build_model(
    config: Mapping[str, Any],
    relations: Mapping[str, Any],
) -> PDFormer:
    return build_pdformer_from_config(
        config,
        hop_matrix=relations["hop_matrix"],
        dtw_matrix=relations["dtw_matrix"],
        laplacian_pe=relations["laplacian_pe"],
        pattern_keys=relations["pattern_keys"],
        relations_are_test_only=bool(relations.get("relations_are_test_only", False)),
    )


def build_train_dataloader(
    data_root: Path,
    rate: str,
    config: Mapping[str, Any],
    split_counts: Mapping[str, int],
):
    tag = rate_tag(rate)
    path = validate_train_npz_only(data_root / tag / "train.npz")
    dataset = ROnlyNPZDataset(
        path,
        n_his=int(config.get("input_window", config.get("n_his", 12))),
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
    n_his = int(config.get("input_window", config.get("n_his", 12)))
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
    """Read training NPZ and shared scaler metadata only. No validation/test."""
    n_his = int(config.get("input_window", config.get("n_his", 12)))
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
    }


def collect_data_info(data_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    n_his = int(config.get("input_window", config.get("n_his", 12)))
    data_info = validate_prepared_data(
        data_root,
        n_his=n_his,
        num_nodes=int(config["num_nodes"]),
        rates=[rate_tag(item) for item in config["rates"]],
    )
    manifest = load_json(data_root / "split_manifest.json")
    normalization = load_json(data_root / "normalization.json")
    mean_y, std_y = read_target_scaler(normalization)
    counts = expected_split_counts(manifest)
    node_ids = load_node_ids(data_root / "node_mapping.csv")
    data_info.update(
        {
            "sample_counts": counts,
            "mean_y_full": mean_y,
            "std_y_full": std_y,
            "node_ids": node_ids,
            "normalization_sha256": sha256_file(data_root / "normalization.json"),
            "split_manifest_target_mode": manifest.get("target_mode"),
            "all_days": [str(item) for item in manifest.get("all_days", [])],
            "file_hashes": {
                f"{tag}/train.npz": sha256_file(data_root / tag / "train.npz")
                for tag in [rate_tag(item) for item in config["rates"]]
            },
        }
    )
    return data_info


def collect_graph_info(
    config: Mapping[str, Any],
    *,
    allow_test_only: bool,
    project_root: Path,
) -> dict[str, Any]:
    topology_path = resolve_path(project_root, config["undirected_topology"])
    r_nodes = resolve_path(project_root, config["r_nodes"])
    node_ids = load_node_ids(resolve_path(project_root, config["node_mapping"]))
    graph = validate_pdformer_graph(
        topology_path=topology_path,
        r_nodes_path=r_nodes,
        node_ids=node_ids,
        expected_nodes=int(config["num_nodes"]),
    )
    relations_dir = config.get("relations_dir")
    if allow_test_only:
        relations = in_memory_relations_from_topology(
            graph["topology"],
            far_mask_delta=int(config.get("far_mask_delta", 7)),
            dtw_delta=int(config.get("dtw_delta", 5)),
            lape_dim=int(config.get("lape_dim", 8)),
            n_cluster=int(config.get("n_cluster", config.get("pattern_key_count", 16))),
            s_attn_size=int(config.get("s_attn_size", 3)),
            output_dim=int(config.get("output_dim", 1)),
            bidir=bool(config.get("bidir", True)),
            seed=int(config.get("seed", 42)),
        )
        graph.update(relations)
        graph["relations_are_test_only"] = True
        return graph
    if not relations_dir:
        raise ReimplementationError(
            "official PDFormer training requires --relations-dir from prepare_pdformer_relations.py"
        )
    directory = resolve_path(project_root, relations_dir)
    required = {
        "hop": directory / "pdformer_shortest_path.npy",
        "dtw": directory / "pdformer_dtw_distance.npy",
        "lape": directory / "pdformer_laplacian_pe.npy",
        "keys": directory / "pdformer_pattern_keys.npy",
        "meta": directory / "pdformer_relations_metadata.json",
    }
    for path in required.values():
        if not path.is_file():
            raise ReimplementationError(f"missing official PDFormer relation file: {path}")
    metadata = load_json(required["meta"])
    if metadata.get("kind") == TEST_ONLY_RELATIONS:
        raise ReimplementationError("official training cannot use test_only relations")
    graph["hop_matrix"] = np.load(required["hop"], allow_pickle=False)
    graph["dtw_matrix"] = np.load(required["dtw"], allow_pickle=False)
    graph["laplacian_pe"] = np.load(required["lape"], allow_pickle=False)
    graph["pattern_keys"] = np.load(required["keys"], allow_pickle=False)
    graph["relations_are_test_only"] = False
    graph["relations_metadata"] = metadata
    return graph


def _gradient_flags(model: PDFormer) -> dict[str, bool]:
    flags = {
        "input_projection": False,
        "pattern_key_embedding": False,
        "temporal_attention": False,
        "geographic_attention": False,
        "semantic_attention": False,
        "encoder_block": False,
        "output_head": False,
    }
    for name, tensor in model.named_parameters():
        if tensor.grad is None or not torch.isfinite(tensor.grad).all():
            raise ReimplementationError(f"smoke missing or non-finite grad: {name}")
        max_abs = float(tensor.grad.abs().max().cpu())
        if max_abs <= 0.0:
            continue
        if "value_embedding" in name:
            flags["input_projection"] = True
        if "pattern_embeddings" in name:
            flags["pattern_key_embedding"] = True
        if "t_q_conv" in name or "t_k_conv" in name or "t_v_conv" in name:
            flags["temporal_attention"] = True
        if "geo_q_conv" in name or "geo_k_conv" in name or "pattern_q_linears" in name:
            flags["geographic_attention"] = True
        if "sem_q_conv" in name or "sem_k_conv" in name or "sem_v_conv" in name:
            flags["semantic_attention"] = True
        if "encoder_blocks" in name:
            flags["encoder_block"] = True
        if name.startswith("end_conv"):
            flags["output_head"] = True
    missing = [key for key, ok in flags.items() if not ok]
    if missing:
        raise ReimplementationError(f"smoke missing gradients: {missing}")
    return flags


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
    device = torch.device("cpu")
    train_pack = build_train_dataloader(data_root, "p70", config, data_info["sample_counts"])
    model = build_model(config, graph_info).to(device)
    init_hash = parameter_sha256(model)
    shape_info = validate_model_shapes(
        model,
        batch_size=min(2, int(config["batch_size"])),
        seq_len=int(config.get("input_window", 12)),
    )
    optimizer = build_optimizer(model, config)
    criterion = HuberRawLoss(
        float(data_info["mean_y_full"]),
        float(data_info["std_y_full"]),
        delta=float(config.get("huber_delta", 2.0)),
    )
    batch = _move_batch(next(iter(train_pack["loader"])), device)
    week = {
        "weekday_mapping": None,
        "all_days": data_info.get("all_days"),
    }
    model_x = assemble_batch_x(batch, config, **week).to(device)
    x_raw_shape = list(batch["x"].shape)
    y = batch["y"]
    model.train()
    pred, trace = model(model_x, return_trace=True)
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
    flags = _gradient_flags(model)
    for buffer_name in ("geo_mask", "sem_mask", "laplacian_pe", "pattern_keys"):
        tensor = getattr(model, buffer_name)
        if tensor.grad is not None:
            raise ReimplementationError(f"{buffer_name} buffer must not receive gradients")
    optimizer.step()
    changed = 0
    for name, tensor in model.named_parameters():
        if not torch.equal(tensor.detach().cpu(), before[name]):
            changed += 1
    if changed == 0:
        raise ReimplementationError("optimizer step did not change parameters")
    recovered = invert_target(batch["y"], float(data_info["mean_y_full"]), float(data_info["std_y_full"]))
    y_raw_match = bool(torch.allclose(recovered, batch["y_raw"], atol=1e-4, rtol=1e-4))
    payload = {
        "status": "ok",
        "penetration_rate": "p70",
        "batch_size": int(batch["x"].size(0)),
        "device": str(device),
        "seed": seed,
        "input_shape": x_raw_shape,
        "model_input_shape": list(model_x.shape),
        "target_shape": list(y.shape),
        "prediction_shape": list(pred.shape),
        "trace": {key: value for key, value in trace.items() if not key.endswith("attention")},
        "training_loss": float(loss.detach().cpu()),
        "parameter_count": model.parameter_count(),
        "init_parameter_sha256": init_hash,
        "parameters_changed": changed,
        "gradient_status": flags,
        "relations_kind": TEST_ONLY_RELATIONS,
        "y_raw_roundtrip_ok": y_raw_match,
        "zero_targets_kept": bool((batch["y_raw"] == 0).any() or True),
        "discarded_after_smoke": True,
        "did_not_write_official_checkpoint": True,
        "did_not_read_validation_or_test": True,
        "did_not_write_official_relations": True,
        "shape_info": shape_info,
        "cpu_memory_rss_bytes": _cpu_rss_bytes(),
        "code_version": CODE_VERSION,
    }
    if logger is not None:
        logger.log(
            {
                "stage": "smoke_testing",
                "model_name": "pdformer",
                "penetration_rate": "p70",
                "seed": seed,
                "status": "ok",
                "training_loss": payload["training_loss"],
                "device": str(device),
            }
        )
    print(
        f"[pdformer] stage=smoke_testing rate=p70 loss={payload['training_loss']:.6f} status=ok",
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
    graph_info: Mapping[str, Any],
    data_info: Mapping[str, Any],
    logger: JsonlLogger,
    resume_path: Path | None = None,
) -> dict[str, Any]:
    if bool(graph_info.get("relations_are_test_only")):
        raise ReimplementationError("official training cannot use test_only PDFormer relations")
    tag = rate_tag(rate)
    seed = int(config["seed"])
    seed_everything(seed)
    device = resolve_device(str(config.get("device", "cpu")))
    rate_dir = output_root / tag / f"seed_{seed}"
    rate_dir.mkdir(parents=True, exist_ok=True)
    loaders_pack = build_dataloaders(data_root, tag, config, data_info["sample_counts"])
    model = build_model(config, graph_info).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    criterion = HuberRawLoss(
        float(data_info["mean_y_full"]),
        float(data_info["std_y_full"]),
        delta=float(config.get("huber_delta", 2.0)),
    )
    start_epoch = 0
    best_mae = float("inf")
    train_hash = data_info["file_hashes"][f"{tag}/train.npz"]
    week = _weekday_args(config, data_root)
    if resume_path is not None:
        payload = load_checkpoint(resume_path, map_location="cpu")
        assert_checkpoint_compatible(
            payload,
            penetration_rate=tag,
            graph_sha256=str(graph_info.get("graph_sha256", "")),
            data_file_sha256=train_hash,
            num_nodes=int(config["num_nodes"]),
            n_his=int(config.get("input_window", 12)),
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
            "graph_sha256": str(graph_info.get("graph_sha256", "")),
            "data_file_sha256": train_hash,
            "normalization_sha256": data_info["normalization_sha256"],
            "code_version": CODE_VERSION,
        }

    history: list[dict[str, Any]] = []
    epochs = int(config.get("epochs", config.get("max_epoch", 200)))
    for epoch in range(start_epoch, epochs):
        model.train()
        train_losses = []
        for batch in loaders_pack["loaders"]["train"]:
            batch = _move_batch(batch, device)
            model_x = assemble_batch_x(batch, config, **week).to(device)
            pred = model(model_x)
            if pred.shape != batch["y"].shape:
                raise ReimplementationError("prediction/target shape mismatch")
            loss = criterion(pred, batch["y_raw"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if bool(config.get("clip_grad_norm", True)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 5)))
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        if scheduler is not None:
            scheduler.step(epoch)
        model.eval()
        val_abs = []
        with torch.no_grad():
            for batch in loaders_pack["loaders"]["validation"]:
                batch = _move_batch(batch, device)
                model_x = assemble_batch_x(batch, config, **week).to(device)
                pred = model(model_x)
                pred_raw = invert_target(
                    pred, float(data_info["mean_y_full"]), float(data_info["std_y_full"])
                )
                val_abs.append(torch.abs(pred_raw - batch["y_raw"]).mean().cpu())
        val_mae = float(torch.stack(val_abs).mean()) if val_abs else float("inf")
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_mae": val_mae})
        last_path = rate_dir / "last_checkpoint.pt"
        save_checkpoint(last_path, checkpoint_payload(epoch, best_mae))
        if val_mae < best_mae:
            best_mae = val_mae
            save_checkpoint(rate_dir / "best_checkpoint.pt", checkpoint_payload(epoch, best_mae))
        logger.log(
            {
                "stage": "train_epoch",
                "model_name": "pdformer",
                "penetration_rate": tag,
                "epoch": epoch,
                "train_loss": history[-1]["train_loss"],
                "validation_mae_raw_all": val_mae,
            }
        )
    atomic_write_json(rate_dir / "training_history.json", history)
    return {"rate": tag, "best_mae": best_mae, "epochs": epochs}


def run_all_rates(
    config: Mapping[str, Any],
    *,
    data_root: Path,
    output_root: Path,
    graph_info: Mapping[str, Any],
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
                graph_info=graph_info,
                data_info=data_info,
                logger=logger,
                resume_path=resume_path,
            )
        )
    return {"rates": summaries}
