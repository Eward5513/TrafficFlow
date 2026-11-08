"""Fixed-backbone MSTT: train sampling42; select and evaluate on five samplings.

Each model keeps its own optimizer/loss. Validation selection uses the mean
physical-scale MAE, and the restored best checkpoint evaluates all five test inputs.
"""

from __future__ import annotations

import contextlib
import csv
import fcntl
import io
import itertools
import json
import random
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from data_root import sampling_run_results_dir
from reimplementation.common.data.sampling_seeds import configure_sampling
from reimplementation.common.sampling_evaluation import evaluate_sampling, sampling_history, save_sampling_outputs

from MSTT import CODE_VERSION, MODEL_NAME
from MSTT.config import (
    STATE_CHANNELS,
    TaskSpec,
    rate_tag,
    resolve_ffn_hidden_dim,
    task_spec,
    validate_config,
    with_legacy_defaults,
    validate_conservation_task,
    conservation_settings,
    conservation_output_root,
)
from MSTT.data.dataset import (
    build_loader,
    open_task_datasets,
    target_scaler,
    time_features,
    train_data_sha256,
    validate_state_package,
)
from MSTT.data.graph import load_r_graph
from MSTT.model import MSTT, MSTTConfig
from MSTT.utils.device import resolve_device
from MSTT.utils.losses import RawScaleLoss, conservation_objective, objective_description
from MSTT.utils.conservation_evaluation import collect_conservation_predictions, save_mstt_predictions
from MSTT.utils.metrics import CORE, flat_core, raw_metrics
from reimplementation.common.data.r_only_npz_dataset import SPLITS
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.evaluation import collect_split_predictions
from reimplementation.common.experiment_protocol import (
    ExperimentClock,
    LibCityCosineLRScheduler,
    build_run_metadata,
    peak_memory_mb,
    reset_peak_memory,
)
from reimplementation.common.utils.atomic_io import atomic_write_json, atomic_write_text
from reimplementation.common.utils.checkpoint import assert_checkpoint_compatible, load_checkpoint, save_checkpoint
from reimplementation.common.utils.reproducibility import seed_everything
from reimplementation.common.utils.structured_logging import JsonlLogger

BATCH_KEYS = ("x", "y", "y_raw", "day_index", "window_start_slot", "target_slot", "sample_index")
REQUIRED_OUTPUTS = ("best_checkpoint.pt", "last_checkpoint.pt", "training_history.json", "validation_sampling_metrics.json", "test_sampling_metrics.json")


def run_directory(output_root: Path, rate: str, task: str, seed: int) -> Path:
    spec = task_spec(task)
    return sampling_run_results_dir(output_root, spec.input_type, spec.target_type, rate_tag(rate), seed)


def is_smoke(config: Mapping[str, Any]) -> bool:
    return config.get("max_train_batches") is not None or config.get("max_eval_batches") is not None


def run_is_complete(run_dir: Path) -> bool:
    meta = Path(run_dir) / "run_metadata.json"
    if not meta.is_file():
        return False
    if json.loads(meta.read_text(encoding="utf-8")).get("status") != "completed":
        return False
    return all((Path(run_dir) / name).is_file() for name in REQUIRED_OUTPUTS)


def model_config(config: Mapping[str, Any], task: TaskSpec) -> MSTTConfig:
    validate_conservation_task(config, task)
    return MSTTConfig.from_mapping(
        config,
        num_nodes=int(config["num_nodes"]),
        in_channels=task.in_channels,
        out_channels=task.out_channels,
        ffn_hidden_dim=resolve_ffn_hidden_dim(int(config["d_model"]), config.get("ffn_hidden_dim")),
    )


_CONCAT_WIDTH_KEYS = ("value_embedding_dim", "node_embedding_dim", "tod_embedding_dim", "dow_embedding_dim")


def _effective_model_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """``add`` fusion ignores the concat widths, so they do not distinguish two ``add`` runs."""
    result = dict(payload)
    if result.get("conservation_mode", "none") != "projected":
        result.pop("projection_mlp_hidden_dim", None)
    if result.get("embedding_fusion") == "add":
        for key in _CONCAT_WIDTH_KEYS:
            result.pop(key, None)
    return result


def assert_run_dir_matches(run_dir: Path, config: Mapping[str, Any], task: TaskSpec) -> None:
    """Refuse to skip into or resume a run directory trained with another model config or batch size."""
    path = Path(run_dir) / "resolved_config.json"
    if not path.is_file():
        return
    stored = with_legacy_defaults(json.loads(path.read_text(encoding="utf-8")))
    current = _effective_model_config(model_config(config, task).to_dict())
    stored_model = _effective_model_config(stored.get("model_config") or {})
    diff = {key: (stored_model.get(key), value) for key, value in current.items() if stored_model.get(key) != value}
    if int(stored.get("batch_size", -1)) != int(config["batch_size"]):
        diff["batch_size"] = (stored.get("batch_size"), int(config["batch_size"]))
    if config.get("conservation_mode", "none") != "none":
        for key, value in conservation_settings(config).items():
            if conservation_settings(stored)[key] != value:
                diff[key] = (conservation_settings(stored)[key], value)
    if diff:
        raise ReimplementationError(
            f"{run_dir} holds a run with a different config (stored, requested): {diff}; "
            "use another --output-root for a new experiment, or --overwrite to replace it"
        )


def build_optimizer(model: torch.nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    """AdamW; matrices decay, while biases, norm scales, and edge-type logits do not."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.ndim >= 2 and "edge_type_bias" not in name else no_decay).append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(config["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(config["learning_rate"]),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: Mapping[str, Any]) -> LibCityCosineLRScheduler | None:
    if str(config["lr_scheduler"]) == "none":
        return None
    return LibCityCosineLRScheduler(
        optimizer,
        t_initial=int(config["epochs"]),
        lr_min=float(config["lr_min"]),
        warmup_t=int(config["lr_warmup_epochs"]),
        warmup_lr_init=float(config["lr_warmup_init"]),
    )


def autocast_context(device: torch.device, amp: str):
    if amp == "none" or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if amp == "bf16" else torch.float16)


def move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: batch[key].to(device, non_blocking=True) for key in BATCH_KEYS if key in batch}


def predict(model: MSTT, batch: Mapping[str, torch.Tensor], *, amp: str, device: torch.device) -> torch.Tensor:
    """Normalized prediction shaped like ``batch['y']``: ``[B, 1, N, C_out]``."""
    tod, dow = time_features(batch, model.config.input_window, steps_per_day=model.config.steps_per_day)
    with autocast_context(device, amp):
        output = model(batch["x"], tod, dow)
    return output.float().unsqueeze(1)


def predict_with_aux(model, batch, *, amp, device):
    tod, dow = time_features(batch, model.config.input_window, steps_per_day=model.config.steps_per_day)
    with autocast_context(device, amp):
        outputs = model.forward_with_aux(batch["x"], tod, dow)
    return {key: value.float().unsqueeze(1) for key, value in outputs.items()}


def assert_conservation_checkpoint(payload, config, *, normalization_sha256=None):
    saved_mode = payload.get("conservation_mode", payload.get("config", {}).get("conservation_mode", "none"))
    mode = config.get("conservation_mode", "none")
    if saved_mode != mode:
        raise ReimplementationError(f"checkpoint conservation_mode={saved_mode} != requested {mode}")
    if mode != "none":
        saved = conservation_settings(payload.get("config", {}))
        if saved != conservation_settings(config):
            raise ReimplementationError("checkpoint conservation coefficients/MLP config differ; start an independent run")
        if normalization_sha256 is not None and payload.get("normalization_sha256") != normalization_sha256:
            raise ReimplementationError("checkpoint target scaler SHA256 differs from state_normalization.json")


def _limited(loader, limit: int | None) -> Iterable:
    return loader if limit is None else itertools.islice(loader, int(limit))


class TaskRun:
    """Everything one (task, rate, seed) run needs, built once."""

    def __init__(self, config: Mapping[str, Any], *, task: str, rate: str, output_root: Path, splits=SPLITS) -> None:
        config = configure_sampling(config)
        validate_config(config)
        validate_conservation_task(config, task_spec(task))
        self.config = dict(config)
        self.config["output_root"] = Path(output_root).as_posix()
        self.task = task_spec(task)
        self.rate = rate_tag(rate)
        self.seed = int(config["seed"])
        self.output_root = Path(output_root)
        self.run_dir = run_directory(self.output_root, self.rate, self.task.name, self.seed)
        self.n_his = int(config["input_window"])
        self.num_nodes = int(config["num_nodes"])
        self.data_root = Path(config["data_root"])
        self.device, self.device_info = resolve_device(str(config["device"]))
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            torch.backends.cuda.matmul.allow_tf32 = bool(config["allow_tf32"])
            torch.backends.cudnn.allow_tf32 = bool(config["allow_tf32"])
        self.package = validate_state_package(self.data_root, num_nodes=self.num_nodes, n_his=self.n_his)
        self.graph, self.graph_info = load_r_graph(
            Path(config["graph_path"]),
            r_nodes_path=Path(config["r_nodes"]),
            window_node_mapping=Path(self.package["node_mapping"]),
            expected_nodes=self.num_nodes,
            direction=str(config["spatial_direction"]),
        )
        self.mean_y, self.std_y = target_scaler(self.data_root, self.task)
        self.datasets = open_task_datasets(
            self.data_root,
            self.rate,
            self.task,
            n_his=self.n_his,
            num_nodes=self.num_nodes,
            sample_counts=self.package["sample_counts"],
            splits=tuple(splits),
        )
        self.loaders = {
            split: build_loader(
                dataset,
                batch_size=int(config["batch_size"] if split == "train" else config["eval_batch_size"]),
                shuffle=split == "train",
                seed=self.seed,
                num_workers=int(config["num_workers"]),
            )
            for split, dataset in self.datasets.items()
        }
        self.criterion = RawScaleLoss(
            self.mean_y,
            self.std_y,
            kind=str(config["loss"]),
            huber_delta=float(config["huber_delta"]),
            channel_balance=str(config["channel_balance"]),
        ).to(self.device)
        self.model_config = model_config(config, self.task)

    @property
    def channel_names(self) -> tuple[str, ...]:
        return self.task.output_channels

    def scaler_args(self) -> tuple[Any, Any]:
        if self.mean_y.size == 1:
            return float(self.mean_y[0]), float(self.std_y[0])
        return self.mean_y, self.std_y

    def build_model(self) -> MSTT:
        return MSTT(self.model_config, self.graph, target_mean=self.mean_y, target_std=self.std_y).to(self.device)

    def evaluate(self, model: MSTT, split: str, limit: int | None = None):
        if split not in self.loaders:
            datasets = open_task_datasets(self.data_root, self.rate, self.task, n_his=self.n_his,
                                         num_nodes=self.num_nodes, sample_counts=self.package["sample_counts"], splits=(split,))
            self.loaders[split] = build_loader(datasets[split], batch_size=int(self.config["eval_batch_size"]),
                                               shuffle=False, seed=self.seed, num_workers=int(self.config["num_workers"]))
        def one(loader):
            metrics, packed = self._evaluate_single(model, split, loader, limit)
            packed["metrics"] = metrics
            return packed
        packed = evaluate_sampling(self.loaders[split], one)
        metrics = packed["metrics"]
        metrics.update(sampling_history(packed["sampling_metrics"]) if split == "validation" else {})
        return metrics, packed

    def _evaluate_single(self, model, split, loader, limit):
        model.eval()
        amp = str(self.config["amp"])
        mean, std = self.scaler_args()
        diagnostics = {}
        if self.task.name == "All2All":
            packed, diagnostics = collect_conservation_predictions(
                _limited(loader, limit),
                lambda batch: predict_with_aux(model, batch, amp=amp, device=self.device),
                criterion=self.criterion, device=self.device,
            )
        else:
            packed = collect_split_predictions(
                _limited(loader, limit),
                lambda batch: predict(model, batch, amp=amp, device=self.device),
                mean_y=mean, std_y=std, device=self.device,
            )
        metrics = raw_metrics(packed["pred_raw"], packed["y_raw"], self.channel_names, self.std_y)
        metrics.update(diagnostics)
        metrics["conservation_mode"] = self.config.get("conservation_mode", "none")
        with torch.no_grad():
            loss = self.criterion(
                torch.from_numpy(packed["y_pred_normalized"]).to(self.device),
                torch.from_numpy(packed["y_raw"]).to(self.device),
            )
        metrics["loss"] = float(loss)
        metrics["loss_name"] = self.criterion.describe()
        metrics["sample_count"] = int(packed["pred_raw"].shape[0])
        metrics["split"] = split
        return metrics, packed

    def selection_value(self, metrics: Mapping[str, Any]) -> float:
        """Checkpoint rule. The training loss stays the mean raw MAE.

        Four-channel inputs that predict inflow keep that loss, but the saved
        epoch is the one with the smallest validation inflow MAE. After the
        five-seed aggregation this is the mean of the five inflow MAEs.
        """
        if self.task.input_type == "all" and "inflow_mae" in metrics:
            return float(metrics["inflow_mae"])
        return float(metrics["mae"])


def _rng_state(loader) -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "loader": loader.generator.get_state() if loader.generator is not None else None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Mapping[str, Any], loader) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("loader") is not None and loader.generator is not None:
        loader.generator.set_state(state["loader"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def gradient_report(model: torch.nn.Module) -> dict[str, Any]:
    missing, nonfinite, zero = [], [], []
    for name, param in model.named_parameters():
        if param.grad is None:
            missing.append(name)
        elif not torch.isfinite(param.grad).all():
            nonfinite.append(name)
        elif float(param.grad.abs().max()) == 0.0:
            zero.append(name)
    return {"missing": missing, "nonfinite": nonfinite, "all_zero": zero}


def train_step(
    run: TaskRun,
    model: MSTT,
    batch: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    grad_scaler: torch.amp.GradScaler,
    *,
    amp: str,
    clip: float,
    epoch: int,
) -> tuple[float, float]:
    """One optimizer step; returns (loss, pre-clip grad norm or 0.0 without clipping)."""
    batch = move_batch(batch, run.device)
    if run.config.get("conservation_mode", "none") == "none":
        prediction = predict(model, batch, amp=amp, device=run.device)
        loss = run.criterion(prediction, batch["y_raw"])
        supervised_name = "supervised_mae" if run.criterion.describe() == "mae_raw" else "supervised_loss"
        parts = {supervised_name: loss, "total_loss": loss}
    else:
        outputs = predict_with_aux(model, batch, amp=amp, device=run.device)
        loss, parts = conservation_objective(run.criterion, outputs, batch["y_raw"], run.config)
    if not torch.isfinite(loss):
        raise ReimplementationError(f"non-finite training loss at epoch {epoch}: {float(loss)}")
    optimizer.zero_grad(set_to_none=True)
    grad_scaler.scale(loss).backward()
    norm = 0.0
    if clip > 0.0:
        grad_scaler.unscale_(optimizer)
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
    grad_scaler.step(optimizer)
    grad_scaler.update()
    run.last_train_components = {key: float(value.detach()) for key, value in parts.items()}
    return float(loss.detach()), norm


def _write_history(run_dir: Path, history: list[dict[str, Any]]) -> None:
    atomic_write_json(run_dir / "training_history.json", history)
    if not history:
        return
    fields = list(history[0].keys())
    for row in history[1:]:
        fields.extend(key for key in row if key not in fields)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(history)
    atomic_write_text(run_dir / "training_history.csv", buffer.getvalue())


def _save_split(run: TaskRun, split: str, metrics: dict[str, Any], packed: Mapping[str, np.ndarray], save_predictions: bool) -> None:
    save_sampling_outputs(run.run_dir, split, packed)
    atomic_write_json(run.run_dir / f"{split}_metrics.json", metrics)
    if save_predictions:
        save_mstt_predictions(run.run_dir / f"{split}_predictions.npz", packed)


def train_task(
    config: Mapping[str, Any],
    *,
    task: str,
    rate: str,
    output_root: Path,
    overwrite: bool = False,
    resume: Path | None = None,
) -> dict[str, Any]:
    validate_config(config)
    validate_conservation_task(config, task_spec(task))
    output_root = conservation_output_root(config, output_root)
    planned_dir = run_directory(output_root, rate, task_spec(task).name, int(config["seed"]))
    from reimplementation.common.data.sampling_seeds import protect_historical_results
    protect_historical_results(planned_dir)
    stored_config = planned_dir / "resolved_config.json"
    if stored_config.is_file():
        stored = json.loads(stored_config.read_text(encoding="utf-8"))
        if stored.get("conservation_mode", "none") != config.get("conservation_mode", "none"):
            raise ReimplementationError("cannot overwrite a different conservation mode; use an independent output root")
    if not overwrite:
        assert_run_dir_matches(planned_dir, config, task_spec(task))
    if run_is_complete(planned_dir) and not overwrite and resume is None:
        print(f"[MSTT] skip completed {planned_dir}", flush=True)
        return summary_from_disk(planned_dir)
    clock = ExperimentClock()
    clock.mark_experiment_start()
    process_start = time.perf_counter()
    previous_wall_seconds = 0.0
    seed_everything(int(config["seed"]))
    run = TaskRun(config, task=task, rate=rate, output_root=output_root, splits=("train", "validation"))
    run_dir = run.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = run.config
    smoke = is_smoke(cfg)
    data_sha = train_data_sha256(run.data_root, run.rate, run.task, run.package["days"]["train"])
    model = run.build_model()
    saved_config = {
        **cfg,
        "task": run.task.name,
        "penetration": run.rate,
        "input_type": run.task.input_type,
        "target_type": run.task.target_type,
        "input_channels": run.task.in_channels,
        "output_channels": run.task.out_channels,
        "n_his": run.n_his,
        "output_steps": 1,
        "model_config": run.model_config.to_dict(),
        "embedding": model.embedding_summary(),
        "spatial": model.spatial_summary(),
        "architecture": model.architecture_summary(),
        "smoke": smoke,
    }
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    amp = str(cfg["amp"])
    grad_scaler = torch.amp.GradScaler("cuda", enabled=amp == "fp16" and run.device.type == "cuda")
    train_loader = run.loaders["train"]
    history: list[dict[str, Any]] = []
    start_epoch, best_value, best_epoch, wait = 0, float("inf"), -1, 0
    resume_path = resume
    if resume_path is None and not overwrite and (run_dir / "last_checkpoint.pt").is_file():
        resume_path = run_dir / "last_checkpoint.pt"
    if resume_path is not None:
        payload = load_checkpoint(Path(resume_path), map_location="cpu")
        assert_conservation_checkpoint(payload, cfg, normalization_sha256=run.package["state_normalization_sha256"])
        assert_checkpoint_compatible(
            payload,
            penetration_rate=run.rate,
            graph_sha256=run.graph_info["graph_sha256"],
            data_file_sha256=data_sha,
            num_nodes=run.num_nodes,
            n_his=run.n_his,
            output_steps=1,
            input_channels=run.task.in_channels,
            output_channels=run.task.out_channels,
        )
        if payload.get("task") != run.task.name:
            raise ReimplementationError(f"checkpoint task {payload.get('task')} != {run.task.name}")
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and payload.get("scheduler_state_dict"):
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if payload.get("grad_scaler_state_dict"):
            grad_scaler.load_state_dict(payload["grad_scaler_state_dict"])
        start_epoch = int(payload["epoch"]) + 1
        best_value = float(payload["best_metric"])
        best_epoch = int(payload.get("best_epoch", -1))
        wait = int(payload.get("wait", 0))
        history = list(payload.get("history", []))
        previous_wall_seconds = float(payload.get("cumulative_wall_seconds", sum(row["epoch_total_seconds"] for row in history)))
        _restore_rng(payload["rng_state"], train_loader)
        print(f"[MSTT] resume {resume_path} at epoch {start_epoch}", flush=True)
    atomic_write_json(run_dir / "resolved_config.json", saved_config)
    param_count = model.parameter_count()
    epochs = int(cfg["epochs"])
    patience = cfg.get("early_stop_patience")
    clip = float(cfg["clip_grad_norm"]) if cfg.get("clip_grad_norm") else 0.0
    logger = JsonlLogger(run_dir / "training_log.jsonl")

    def checkpoint_payload(epoch: int) -> dict[str, Any]:
        return {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "grad_scaler_state_dict": grad_scaler.state_dict() if grad_scaler.is_enabled() else None,
            "epoch": epoch,
            "best_metric": best_value,
            "best_epoch": best_epoch,
            "wait": wait,
            "history": history,
            "cumulative_wall_seconds": previous_wall_seconds + time.perf_counter() - process_start,
            "rng_state": _rng_state(train_loader),
            "config": saved_config,
            "task": run.task.name,
            "conservation_mode": cfg.get("conservation_mode", "none"),
            "penetration_rate": run.rate,
            "seed": run.seed,
            "model_parameter_count": param_count,
            "graph_sha256": run.graph_info["graph_sha256"],
            "data_file_sha256": data_sha,
            "normalization_sha256": run.package["state_normalization_sha256"],
            "code_version": CODE_VERSION,
        }

    reset_peak_memory(run.device)
    clock.mark_train_start()
    early_stopped = False
    epochs_this_process = 0
    try:
        for epoch in range(start_epoch, epochs):
            epoch_start = time.perf_counter()
            model.train()
            loss_sum, loss_batches, grad_norm_max = 0.0, 0, 0.0
            component_sums, train_samples = {}, 0
            for batch in _limited(train_loader, cfg.get("max_train_batches")):
                loss_value, norm = train_step(run, model, batch, optimizer, grad_scaler, amp=amp, clip=clip, epoch=epoch)
                grad_norm_max = max(grad_norm_max, norm)
                loss_sum += loss_value
                loss_batches += 1
                batch_samples = int(batch["x"].shape[0])
                train_samples += batch_samples
                for key, value in run.last_train_components.items():
                    component_sums[key] = component_sums.get(key, 0.0) + value * batch_samples
            lr = float(optimizer.param_groups[0]["lr"])
            if scheduler is not None:
                scheduler.step(epoch + 1)
            train_seconds = time.perf_counter() - epoch_start
            val_metrics, _ = run.evaluate(model, "validation", cfg.get("max_eval_batches"))
            value = run.selection_value(val_metrics)
            if not np.isfinite(value):
                raise ReimplementationError(f"non-finite validation {cfg['selection_metric']} at epoch {epoch}")
            improved = value < best_value
            if improved:
                best_value, best_epoch, wait = value, epoch, 0
            else:
                wait += 1
            row = {
                "epoch": epoch,
                "learning_rate": lr,
                "train_loss": loss_sum / max(loss_batches, 1),
                "train_batches": loss_batches,
                **{f"train_{key}": value / max(train_samples, 1) for key, value in component_sums.items()},
                "grad_norm_max": grad_norm_max,
                "val_loss": val_metrics["loss"],
                **{k: v for k, v in val_metrics.items() if k.startswith(("mean_validation_", "std_validation_", "validation_seed_"))},
                **{f"val_{key}": val_metrics[key] for key in CORE + ("scaled_mae",)},
                **{f"val_{name}_mae": val_metrics[f"{name}_mae"] for name in run.channel_names},
                **{f"val_{key}": value for key, value in val_metrics.items()
                   if key.startswith(("conservation_", "raw_", "negative_", "correction_", "projection_"))},
                "selection_value": value,
                "improved": improved,
                "epoch_train_seconds": train_seconds,
                "epoch_total_seconds": time.perf_counter() - epoch_start,
                **peak_memory_mb(run.device),
            }
            history.append(row)
            epochs_this_process += 1
            if improved:
                save_checkpoint(run_dir / "best_checkpoint.pt", checkpoint_payload(epoch))
            save_checkpoint(run_dir / "last_checkpoint.pt", checkpoint_payload(epoch))
            _write_history(run_dir, history)
            logger.log({"stage": "epoch", "task": run.task.name, "rate": run.rate, **row})
            inflow_text = f" val_inflow={row['val_inflow_mae']:.5f}" if "val_inflow_mae" in row else ""
            print(
                f"[MSTT] {run.task.name} {run.rate} epoch={epoch} lr={lr:.2e} train={row['train_loss']:.5f} "
                f"val_loss={row['val_loss']:.5f} val_mae={row['val_mae']:.5f}{inflow_text} "
                f"{'*' if improved else ''} {row['epoch_total_seconds']:.1f}s",
                flush=True,
            )
            if patience is not None and wait >= int(patience):
                early_stopped = True
                break
        clock.mark_train_end()
        if not history:
            raise ReimplementationError("no epochs were run")
        if patience is None and [row["epoch"] for row in history] != list(range(epochs)):
            raise ReimplementationError("early stopping disabled but the complete epoch history is missing")
        expected_best = min(history, key=lambda item: (item["selection_value"], item["epoch"]))["epoch"]
        payload = load_checkpoint(run_dir / "best_checkpoint.pt", map_location=run.device)
        if int(payload["epoch"]) != int(expected_best):
            raise ReimplementationError(f"best checkpoint epoch {payload['epoch']} != history best {expected_best}")
        model.load_state_dict(payload["model_state_dict"])
        limit = cfg.get("max_eval_batches")
        val_metrics, val_packed = run.evaluate(model, "validation", limit)
        if abs(run.selection_value(val_metrics) - float(payload["best_metric"])) > 1e-4 * max(1.0, abs(best_value)):
            raise ReimplementationError("reloaded best checkpoint does not reproduce its validation metric")
        _save_split(run, "validation", val_metrics, val_packed, bool(cfg["save_validation_predictions"]))
        expected_validation = int(run.package["sample_counts"]["validation"])
        if not smoke and val_metrics["sample_count"] != expected_validation:
            raise ReimplementationError(
                f"validation evaluated {val_metrics['sample_count']} != {expected_validation}"
            )
        test_metrics, test_packed = run.evaluate(model, "test", limit)
        _save_split(run, "test", test_metrics, test_packed, bool(cfg["save_test_predictions"]))
        clock.mark_experiment_end()
        metadata = build_run_metadata(
            model=MODEL_NAME,
            penetration=run.rate,
            seed=run.seed,
            device=run.device,
            parameter_count=param_count,
            training_loss=objective_description(run.criterion, cfg),
            optimizer="AdamW",
            initial_lr=float(cfg["learning_rate"]),
            scheduler=None if scheduler is None else "libcity_cosine_warmup",
            max_epochs=epochs,
            epochs_executed=max(epochs_this_process, 1),
            best_epoch=best_epoch,
            best_metric=best_value,
            early_stopping=patience is not None,
            early_stopping_patience=None if patience is None else int(patience),
            selection_metric=str(cfg["selection_metric"]),
            clock=clock,
            extra={
                "status": "completed",
                "task": run.task.name,
                "input_type": run.task.input_type,
                "target_type": run.task.target_type,
                "smoke": smoke,
                "early_stopped": early_stopped,
                "epochs_in_history": len(history),
                "code_version": CODE_VERSION,
                "graph": {key: value for key, value in run.graph_info.items() if key != "node_ids"},
                "data_file_sha256": data_sha,
                "state_normalization_sha256": run.package["state_normalization_sha256"],
                "split_manifest_sha256": run.package["split_manifest_sha256"],
                "sample_counts": run.package["sample_counts"],
                "device_info": run.device_info,
                "amp": amp,
                "batch_size": int(cfg["batch_size"]),
                **conservation_settings(cfg),
                "selection_prediction": "final",
                "effective_selection_metric": (
                    "val_inflow_mae" if run.task.input_type == "all" and "inflow" in run.task.output_channels
                    else ("val_mae" if cfg.get("conservation_mode", "none") != "none" else cfg["selection_metric"])
                ),
                "embedding": model.embedding_summary(),
                "architecture": model.architecture_summary(),
                "spatial": model.spatial_summary(),
            },
        )
        metadata["total_epochs_executed"] = len(history)
        metadata["total_experiment_wall_time_sec"] += previous_wall_seconds
        for key in ("peak_gpu_memory_allocated_mb", "peak_gpu_memory_reserved_mb"):
            metadata[key] = max([metadata.get(key) or 0.0] + [row.get(key) or 0.0 for row in history])
        atomic_write_json(run_dir / "run_metadata.json", metadata)
    except Exception as exc:
        atomic_write_json(run_dir / "run_metadata.json", {"status": "failed", "reason": repr(exc), "task": run.task.name, "penetration": run.rate})
        raise
    finally:
        logger.close()
    summary = summary_from_disk(run_dir)
    merge_summary(output_root, [summary])
    print(
        f"[MSTT] done {run.task.name} {run.rate} best_epoch={best_epoch} "
        f"val_mae={val_metrics['mae']:.5f} val_wape={val_metrics['wape']:.3f}",
        flush=True,
    )
    return summary


SUMMARY_FIELDS = [
    "conservation_mode", "conservation_lambda", "projection_raw_loss_weight", "projection_consistency_weight",
    "task",
    "rate",
    "seed",
    "input_type",
    "target_type",
    "status",
    "smoke",
    "best_epoch",
    "epochs_in_history",
    "early_stopped",
    "parameter_count",
    "selection_metric",
    "best_metric",
    *[f"val_{key}" for key in CORE + ("scaled_mae",)],
    *[f"test_{key}" for key in CORE + ("scaled_mae",)],
    *[f"test_{name}_{key}" for name in STATE_CHANNELS for key in CORE],
    *[f"{split}_{key}" for split in ("val", "test") for key in (
        "conservation_mae", "conservation_max_abs", "raw_mae", "raw_conservation_mae",
        "raw_conservation_max_abs", "negative_fraction", "negative_mean_abs", "negative_max_abs",
    )],
    "train_wall_time_sec",
    "average_epoch_time_sec",
    "test_inference_sec",
    "peak_gpu_memory_allocated_mb",
    "run_dir",
]


def summary_from_disk(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    val = json.loads((run_dir / "validation_metrics.json").read_text(encoding="utf-8"))
    test_path = run_dir / "test_metrics.json"
    test = json.loads(test_path.read_text(encoding="utf-8")) if test_path.is_file() else None
    channels = [] if test is None else test.get("channels", [])
    row = {
        "task": meta.get("task"),
        "rate": meta.get("penetration"),
        "seed": meta.get("seed"),
        "input_type": meta.get("input_type"),
        "target_type": meta.get("target_type"),
        "status": meta.get("status"),
        "smoke": meta.get("smoke"),
        "best_epoch": meta.get("best_epoch"),
        "epochs_in_history": meta.get("epochs_in_history"),
        "early_stopped": meta.get("early_stopped"),
        "parameter_count": meta.get("parameter_count"),
        "selection_metric": meta.get("selection_metric"),
        "best_metric": meta.get("best_metric"),
        "train_wall_time_sec": meta.get("train_wall_time_sec"),
        "average_epoch_time_sec": meta.get("average_epoch_time_sec"),
        "test_inference_sec": meta.get("test_inference_sec"),
        "peak_gpu_memory_allocated_mb": meta.get("peak_gpu_memory_allocated_mb"),
        "run_dir": run_dir.as_posix(),
    }
    row.update(flat_core(val, [], "val"))
    if test is not None:
        row.update(flat_core(test, channels, "test"))
    row.update(conservation_settings(meta))
    present = [("val", val)] if test is None else [("val", val), ("test", test)]
    for prefix, metrics in present:
        for key, value in metrics.items():
            if key.startswith(("conservation_", "raw_", "negative_", "correction_", "projection_")):
                row[f"{prefix}_{key}"] = value
    return row


def merge_summary(output_root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Upsert rows keyed by (task, rate, seed) under a file lock; parallel runs keep each other's rows."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".summary.lock").open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            path = output_root / "overall_summary.json"
            existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
            keyed = {(row["task"], row["rate"], int(row["seed"])): row for row in existing}
            for row in rows:
                keyed[(row["task"], row["rate"], int(row["seed"]))] = row
            from MSTT.config import RATES, TASKS

            order = {name: index for index, name in enumerate(TASKS)}
            merged = sorted(keyed.values(), key=lambda row: (RATES.index(row["rate"]), order.get(row["task"], 99), row["seed"]))
            atomic_write_json(path, merged)
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=SUMMARY_FIELDS, lineterminator="\n", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(merged)
            atomic_write_text(output_root / "overall_summary.csv", buffer.getvalue())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return merged
