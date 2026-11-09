"""Train STJGCN on one Xuancheng26 penetration and the five sampling seeds.

The loss is the PeMSD7 raw-scale objective after one target-scaler inverse.
There is no learning-rate schedule. Checkpoint selection is the unweighted
mean of the five raw MAEs. Test runs only after the best checkpoint is restored.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.optim import Adam

from data_root import SAMPLING_SEEDS
from reimplementation.stjgcn.graph import load_predefined_graph
from reimplementation.stjgcn.losses import STJGCNRawLoss
from reimplementation.stjgcn.model.stjgcn import STJGCN
from reimplementation.stjgcn.provenance import CODE_VERSION, REFERENCE_SHA256, SOURCE_REPO
from reimplementation.stjgcn.temporal import time_index
from reimplementation.stjgcn.validation import validate_inputs, validate_node_order
from reimplementation.common.data.r_only_npz_dataset import SPLITS, build_dataloader, loss_scale, rate_tag
from reimplementation.common.data.sampling_seeds import iter_sampling_loaders, protect_historical_results, sampling_rate_dir
from reimplementation.common.data.state_windows import finalize_state_scaler, open_split_dataset
from reimplementation.common.evaluation import (
    collect_split_predictions,
    evaluate_loader,
    reload_best_checkpoint,
    save_split_outputs,
    upsert_overall_summary,
)
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.experiment_protocol import (
    ExperimentClock,
    assert_official_rates,
    best_epoch_from_history,
    build_run_metadata,
    finalize_completed_rate,
    mark_rate_failed,
    physical_gpu_id,
    reset_peak_memory,
    trainable_parameter_count,
)
from reimplementation.common.metrics.traffic_metrics import traffic_metrics
from reimplementation.common.sampling_evaluation import (
    aggregate_sampling_metrics,
    evaluate_sampling_predictor,
    sampling_history,
    save_sampling_outputs,
)
from reimplementation.common.utils.atomic_io import atomic_write_json
from reimplementation.common.utils.checkpoint import (
    assert_checkpoint_compatible,
    load_checkpoint,
    save_checkpoint,
)
from reimplementation.common.utils.reproducibility import seed_everything
from reimplementation.common.utils.structured_logging import JsonlLogger
from reimplementation.stgcn.engine import atomic_write_csv_rows, parameter_sha256, resolve_device, resolve_path

HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "val_loss",
    "mean_validation_mae",
    "std_validation_mae",
    "learning_rate",
    *[f"validation_seed_{seed}_mae" for seed in SAMPLING_SEEDS],
)
SELECTION_METRIC = "mean_validation_mae"


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _capture_rng() -> dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"]) else state["torch"])
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() if torch.is_tensor(item) else item for item in state["cuda"]])


def _move_optimizer(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for slot in optimizer.state.values():
        for key, value in slot.items():
            if torch.is_tensor(value):
                slot[key] = value.to(device)


def build_model(config: Mapping[str, Any]) -> STJGCN:
    return STJGCN.from_config(dict(config))


def build_optimizer(model: torch.nn.Module, config: Mapping[str, Any]) -> Adam:
    if str(config.get("optimizer", "Adam")).lower() != "adam":
        raise ReimplementationError("PeMSD7 STJGCN optimizer is Adam")
    if float(config.get("weight_decay", 0.0)) != 0.0:
        raise ReimplementationError("PeMSD7 STJGCN does not use weight decay")
    return Adam(model.parameters(), lr=float(config.get("learning_rate", 0.001)), weight_decay=0.0)


def predict_batch(
    model: STJGCN,
    batch: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    adjacency_fw: torch.Tensor,
    adjacency_bw: torch.Tensor,
) -> torch.Tensor:
    index = time_index(batch, int(config.get("n_his", 12)), slots_per_day=int(config.get("slots_per_day", 288)))
    return model(batch["x"], index, adjacency_fw, adjacency_bw)


def build_dataloaders(data_root: Path, rate: str, config: Mapping[str, Any], split_counts: Mapping[str, int]) -> dict[str, Any]:
    loaders = {}
    n_his = int(config.get("n_his", 12))
    for split in SPLITS:
        dataset = open_split_dataset(data_root, rate, split, config, n_his=n_his, expected_count=int(split_counts[split]))
        loaders[split] = build_dataloader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config.get("num_workers", 0)),
            seed=int(config["seed"]),
        )
    return {"loaders": loaders}


def collect_data_info(data_root: Path, config: Mapping[str, Any], *, project_root: Path) -> dict[str, Any]:
    info = validate_inputs(data_root, config)
    node_ids = validate_node_order(
        r_nodes_path=resolve_path(project_root, config["r_nodes"]),
        node_mapping=resolve_path(project_root, config["node_mapping"]),
        expected_nodes=int(config["num_nodes"]),
    )
    asset = load_predefined_graph(
        resolve_path(project_root, config["predefined_graph"]),
        edge_ids=node_ids,
        delta_pdf=float(config.get("delta_pdf", 0.9)),
        horizon=int(config.get("P", 12)),
    )
    info["node_ids"] = node_ids
    info["A_fw"] = asset["A_fw"]
    info["A_bw"] = asset["A_bw"]
    info["graph_sha256"] = asset["graph_sha256"]
    info["graph_metadata"] = {key: value for key, value in asset["metadata"].items() if key != "edge_ids"}
    return info


def _iter_train_batches(dataset, batch_size: int, limit: int | None):
    from reimplementation.common.data.r_only_npz_dataset import _collate_batch

    order = torch.randperm(len(dataset)).tolist()
    produced = 0
    for start in range(0, len(order), int(batch_size)):
        if limit is not None and produced >= int(limit):
            break
        chosen = order[start : start + int(batch_size)]
        yield _collate_batch([dataset[int(index)] for index in chosen])
        produced += 1
    if produced == 0:
        raise ReimplementationError("training epoch produced no batches")


class _LimitLoader:
    def __init__(self, loader, limit: int):
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.num_workers = loader.num_workers
        self.loader = loader
        self.limit = int(limit)

    def __iter__(self):
        for index, batch in enumerate(self.loader):
            if index >= self.limit:
                break
            yield batch


def _checked_mean_mae(report: Mapping[str, Any]) -> float:
    values = [float(report["per_seed"][str(seed)]["mae"]) for seed in SAMPLING_SEEDS]
    mean = float(report["mean"]["mae"])
    if abs(mean - float(np.mean(values))) > 1e-6:
        raise ReimplementationError(f"five-seed mean MAE {mean} != average of per-seed MAEs {float(np.mean(values))}")
    return mean


def _score_loader(loader, predict_fn, mean_y, std_y, device: torch.device, limit: int | None) -> dict[str, Any]:
    if limit is None:
        return evaluate_sampling_predictor(loader, predict_fn, mean_y=mean_y, std_y=std_y, device=device)
    per_seed = {}
    predictions = {}
    first = None
    for seed, single in iter_sampling_loaders(loader):
        packed = collect_split_predictions(_LimitLoader(single, limit), predict_fn, mean_y=mean_y, std_y=std_y, device=device)
        packed["metrics"] = traffic_metrics(packed["pred_raw"], packed["y_raw"])
        per_seed[str(seed)] = packed["metrics"]
        predictions[str(seed)] = packed
        if first is None:
            first = packed
    report = aggregate_sampling_metrics(per_seed)
    result = dict(first)
    result["metrics"] = {**first["metrics"], **report["mean"]}
    result["sampling_metrics"] = report
    result["sampling_predictions"] = predictions
    return result


def _loss_of_predictions(packed: Mapping[str, Any], criterion: torch.nn.Module, device: torch.device) -> float:
    losses = []
    for seed_pack in packed["sampling_predictions"].values():
        pred = torch.from_numpy(np.asarray(seed_pack["y_pred_normalized"])).to(device)
        label = torch.from_numpy(np.asarray(seed_pack["y_raw"])).to(device)
        losses.append(float(criterion(pred, label).detach().cpu()))
    return float(np.mean(losses))


def _rate_complete(rate_dir: Path) -> bool:
    meta = rate_dir / "run_metadata.json"
    if not meta.is_file():
        return False
    payload = json.loads(meta.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        return False
    required = ("best_checkpoint.pt", "last_checkpoint.pt", "test_metrics.json", "validation_metrics.json", "training_history.json")
    return all((rate_dir / name).is_file() and (rate_dir / name).stat().st_size > 0 for name in required)


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
    protocol_limit: int | None = None,
) -> dict[str, Any]:
    if protocol_limit is not None and "_smoke" not in Path(output_root).as_posix():
        raise ReimplementationError("a truncated protocol run must write under a _smoke directory")
    data_info = finalize_state_scaler(data_info, config, data_root)
    tag = rate_tag(rate)
    seed = int(config["seed"])
    rate_dir = sampling_rate_dir(output_root, config, tag)
    protect_historical_results(rate_dir)
    if protocol_limit is None and _rate_complete(rate_dir) and not overwrite and resume_path is None:
        print(f"[stjgcn] stage=skip rate={tag} dir={rate_dir}", flush=True)
        return {"status": "completed", "rate": tag, "skipped": True, "output_dir": rate_dir.as_posix()}
    if protocol_limit is None:
        device_probe = resolve_device(str(config.get("device", "cpu")))
        if device_probe.type == "cuda" and physical_gpu_id() is None:
            raise ReimplementationError("set CUDA_VISIBLE_DEVICES to the physical GPU before CUDA training")
    seed_everything(seed)
    device = resolve_device(str(config.get("device", "cpu")))
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    rate_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(rate_dir / "resolved_config.json", dict(config))
    loaders = build_dataloaders(data_root, tag, config, data_info["sample_counts"])
    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    adjacency_fw = torch.from_numpy(np.asarray(data_info["A_fw"], dtype=np.float32)).to(device)
    adjacency_bw = torch.from_numpy(np.asarray(data_info["A_bw"], dtype=np.float32)).to(device)
    mean_y = loss_scale(data_info["mean_y_full"])
    std_y = loss_scale(data_info["std_y_full"])
    criterion = STJGCNRawLoss(mean_y, std_y, beta=float(config.get("beta", 0.5)), epsilon=float(config.get("loss_epsilon", 1.0))).to(device)
    start_epoch = 0
    best_mae = float("inf")
    best_epoch = -1
    history: list[dict[str, Any]] = []
    loaded_epochs = 0
    train_hash = data_info["file_hashes"][f"{tag}/train.npz"]
    graph_sha = str(data_info["graph_sha256"])
    n_his = int(config.get("n_his", 12))
    auto_resume = rate_dir / "last_checkpoint.pt"
    if resume_path is None and not overwrite and protocol_limit is None and auto_resume.is_file():
        resume_path = auto_resume
    if resume_path is not None:
        payload = load_checkpoint(Path(resume_path), map_location="cpu")
        assert_checkpoint_compatible(
            payload,
            penetration_rate=tag,
            graph_sha256=graph_sha,
            data_file_sha256=train_hash,
            num_nodes=int(config["num_nodes"]),
            n_his=n_his,
            output_steps=1,
            input_channels=int(config["input_channels"]),
            output_channels=int(config["output_channels"]),
        )
        saved = payload.get("config") or {}
        if saved.get("input_type") != config.get("input_type") or saved.get("target_type") != config.get("target_type"):
            raise ReimplementationError("refusing to resume a checkpoint from a different input/target task")
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        _move_optimizer(optimizer, device)
        start_epoch = int(payload.get("epoch", -1)) + 1
        best_mae = float(payload.get("best_metric", best_mae))
        best_epoch = int(payload.get("best_epoch", -1))
        if payload.get("rng_state"):
            _restore_rng(payload["rng_state"])
        history_path = rate_dir / "training_history.json"
        if history_path.is_file():
            history = [row for row in json.loads(history_path.read_text(encoding="utf-8")) if int(row["epoch"]) < start_epoch]
            loaded_epochs = len(history)
    clock = ExperimentClock()
    clock.mark_experiment_start()
    reset_peak_memory(device)
    param_count = trainable_parameter_count(model)
    start_hash = parameter_sha256(model)
    epochs = int(config.get("epochs", 200))

    def checkpoint_payload(epoch: int, metric: float) -> dict[str, Any]:
        return {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,
            "rng_state": _capture_rng(),
            "epoch": int(epoch),
            "best_epoch": int(best_epoch),
            "best_metric": float(metric),
            "config": dict(config),
            "penetration_rate": tag,
            "seed": seed,
            "model_parameter_count": param_count,
            "graph_sha256": graph_sha,
            "data_file_sha256": train_hash,
            "normalization_sha256": data_info["normalization_sha256"],
            "code_version": CODE_VERSION,
        }

    def _predict(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return predict_batch(model, batch, config, adjacency_fw, adjacency_bw)

    clock.mark_train_start()
    for epoch in range(start_epoch, epochs):
        model.train()
        weighted = 0.0
        seen = 0
        for batch in _iter_train_batches(loaders["loaders"]["train"].dataset, int(config["batch_size"]), protocol_limit):
            batch = _move_batch(batch, device)
            pred = _predict(batch)
            if pred.shape != batch["y"].shape:
                raise ReimplementationError(f"prediction {tuple(pred.shape)} != target {tuple(batch['y'].shape)}")
            loss = criterion(pred, batch["y_raw"])
            if not torch.isfinite(loss):
                raise ReimplementationError(f"non-finite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = int(batch["y"].shape[0])
            weighted += float(loss.detach().cpu()) * count
            seen += count
        used_lr = float(optimizer.param_groups[0]["lr"])
        model.eval()
        with torch.no_grad():
            scored = _score_loader(loaders["loaders"]["validation"], _predict, mean_y, std_y, device, protocol_limit)
            val_mae = _checked_mean_mae(scored["sampling_metrics"])
            val_loss = _loss_of_predictions(scored, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": weighted / float(seen),
            "val_loss": val_loss,
            "learning_rate": used_lr,
            **sampling_history(scored["sampling_metrics"]),
        }
        history.append(row)
        if val_mae <= best_mae:
            best_mae = val_mae
            best_epoch = epoch
            save_checkpoint(rate_dir / "best_checkpoint.pt", checkpoint_payload(epoch, best_mae))
        save_checkpoint(rate_dir / "last_checkpoint.pt", checkpoint_payload(epoch, best_mae))
        atomic_write_json(rate_dir / "training_history.json", history)
        atomic_write_csv_rows(rate_dir / "training_history.csv", list(HISTORY_FIELDS), history)
        logger.log({"stage": "train_epoch", "model_name": "stjgcn", "penetration_rate": tag, "epoch": epoch, "train_loss": row["train_loss"], "val_mae": val_mae})
        print(f"[stjgcn] rate={tag} epoch={epoch} train_loss={row['train_loss']:.6f} val_mae={val_mae:.6f}", flush=True)
    clock.mark_train_end()
    epochs_this_run = len(history) - loaded_epochs
    if epochs_this_run <= 0:
        raise ReimplementationError("STJGCN produced no new training epochs")
    chosen = best_epoch_from_history(history, SELECTION_METRIC, rule="le")
    if int(chosen) != int(best_epoch):
        raise ReimplementationError(f"selected epoch {best_epoch} != latest minimum {SELECTION_METRIC} epoch {chosen}")
    ckpt = reload_best_checkpoint(model, rate_dir / "best_checkpoint.pt", device)
    if int(ckpt["epoch"]) != int(best_epoch):
        raise ReimplementationError("reloaded best checkpoint epoch does not match the selected epoch")
    model.eval()
    counts = data_info["sample_counts"]
    if protocol_limit is None:
        val_eval = evaluate_loader(
            loaders["loaders"]["validation"], _predict, mean_y=mean_y, std_y=std_y, device=device,
            output_dir=rate_dir, split="validation", expected_count=int(counts["validation"]), n_his=n_his,
        )
        test_eval = evaluate_loader(
            loaders["loaders"]["test"], _predict, mean_y=mean_y, std_y=std_y, device=device,
            output_dir=rate_dir, split="test", expected_count=int(counts["test"]), n_his=n_his,
        )
        val_mae_final = _checked_mean_mae(val_eval["sampling_metrics"])
        test_mae = _checked_mean_mae(test_eval["sampling_metrics"])
    else:
        with torch.no_grad():
            val_eval = _score_loader(loaders["loaders"]["validation"], _predict, mean_y, std_y, device, protocol_limit)
            test_eval = _score_loader(loaders["loaders"]["test"], _predict, mean_y, std_y, device, protocol_limit)
        save_sampling_outputs(rate_dir, "validation", val_eval)
        save_sampling_outputs(rate_dir, "test", test_eval)
        save_split_outputs(rate_dir, "validation", val_eval, expected_count=int(val_eval["pred_raw"].shape[0]), n_his=n_his)
        save_split_outputs(rate_dir, "test", test_eval, expected_count=int(test_eval["pred_raw"].shape[0]), n_his=n_his)
        atomic_write_json(rate_dir / "validation_metrics.json", val_eval["sampling_metrics"]["mean"])
        atomic_write_json(rate_dir / "test_metrics.json", test_eval["sampling_metrics"]["mean"])
        val_mae_final = _checked_mean_mae(val_eval["sampling_metrics"])
        test_mae = _checked_mean_mae(test_eval["sampling_metrics"])
    clock.mark_experiment_end()
    scope = "all_predicted_elements" if int(config["output_channels"]) > 1 else "target_channel"
    metadata = build_run_metadata(
        model="stjgcn",
        penetration=tag,
        seed=seed,
        device=device,
        parameter_count=param_count,
        training_loss="mae_plus_beta_100_mape_raw_mask_zero",
        optimizer="Adam",
        initial_lr=float(config.get("learning_rate", 0.001)),
        scheduler=None,
        max_epochs=epochs,
        epochs_executed=epochs_this_run,
        best_epoch=int(best_epoch),
        best_metric=float(best_mae),
        early_stopping=False,
        early_stopping_patience=None,
        selection_metric=SELECTION_METRIC,
        clock=clock,
        extra={
            "status": "smoke" if protocol_limit is not None else "completed",
            "effective_selection_metric": SELECTION_METRIC,
            "selection_tie": "latest_epoch",
            "mae_scope": scope,
            "input_type": config.get("input_type"),
            "target_type": config.get("target_type"),
            "input_channels": int(config["input_channels"]),
            "output_channels": int(config["output_channels"]),
            "code_version": CODE_VERSION,
            "source_repo": SOURCE_REPO,
            "reference_sha256": REFERENCE_SHA256,
            "graph_sha256": graph_sha,
            "graph_metadata": data_info.get("graph_metadata"),
            "data_file_sha256": train_hash,
            "normalization_sha256": data_info["normalization_sha256"],
            "parameter_sha256_at_train_start": start_hash,
            "batch_size": int(config["batch_size"]),
            "weight_decay": 0.0,
            "beta": float(config.get("beta", 0.5)),
            "loss_epsilon": float(config.get("loss_epsilon", 1.0)),
            "delta_pdf": float(config.get("delta_pdf", 0.9)),
            "delta_adt": float(config.get("delta_adt", 0.7)),
            "attention_node_chunk": int(config.get("attention_node_chunk", 64)),
            "protocol_limit": protocol_limit,
            "resumed_from": None if resume_path is None else Path(resume_path).as_posix(),
        },
    )
    if protocol_limit is None:
        finalize_completed_rate(
            rate_dir,
            metadata=metadata,
            history=history,
            expected_test_count=int(counts["test"]),
            expected_val_count=int(counts["validation"]),
            n_his=n_his,
            history_metric_key=SELECTION_METRIC,
            history_rule="le",
        )
    else:
        atomic_write_json(rate_dir / "run_metadata.json", metadata)
    result = {
        "status": metadata["status"],
        "rate": tag,
        "best_epoch": int(best_epoch),
        "best_metric": float(best_mae),
        "selection_metric": SELECTION_METRIC,
        "effective_selection_metric": SELECTION_METRIC,
        "validation_mae_raw": float(val_mae_final),
        "test_mae_raw": float(test_mae),
        "parameter_count": param_count,
        "output_dir": rate_dir.as_posix(),
        "did_not_use_test_for_selection": True,
        "train_wall_time_sec": metadata["train_wall_time_sec"],
        "peak_gpu_memory_allocated_mb": metadata.get("peak_gpu_memory_allocated_mb"),
    }
    print(f"[stjgcn] stage=rate_done rate={tag} best_epoch={best_epoch} val_mae={val_mae_final:.6f} test_mae={test_mae:.6f}", flush=True)
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
    for rate in assert_official_rates(config["rates"]):
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
        except Exception as exc:
            mark_rate_failed(sampling_rate_dir(output_root, config, rate), reason=str(exc))
            raise
        summaries.append(summary)
        upsert_overall_summary(output_root / "overall_summary.json", summaries)
    return {"rates": summaries}
