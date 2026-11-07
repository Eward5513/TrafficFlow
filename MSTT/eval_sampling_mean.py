"""Score one finished MSTT run on the five p05 sampling-seed test sets.

Training does not open ``window_data/p05/test``. That split was moved to
``training_data_3s/sampling_seeds/seed_<id>/window_data``. This script reads
those five test windows, keeps the graph, node order, and target scaler from
the run's ``resolved_config.json``, and reports the unweighted mean of the
raw metrics. It does not rewrite ``test_metrics.json`` or select a checkpoint.

    python MSTT/eval_sampling_mean.py --run-dir <experiment>/p05/All2I/seed_42
    python MSTT/eval_sampling_mean.py --run-dir <run> --check-data
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reimplementation.common.utils.atomic_io import atomic_write_json  # noqa: E402

SAMPLING_SEEDS = (42, 43, 44, 45, 46)
METRIC_KEYS = ("mae", "rmse", "mape_nonzero", "wape")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sampling-seeds", type=int, nargs="+", default=list(SAMPLING_SEEDS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--check-data", action="store_true")
    return parser.parse_args(argv)


def sampling_window_root(prepared_root: Path, sampling_seed: int) -> Path:
    return prepared_root.parent / "sampling_seeds" / f"seed_{int(sampling_seed)}" / "window_data"


def load_run_config(run_dir: Path) -> tuple[dict, dict]:
    from MSTT.config import load_config, with_legacy_defaults

    saved = with_legacy_defaults(json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8")))
    config = load_config(None)
    config.update({key: saved[key] for key in config if key in saved})
    mode = config.get("conservation_mode", "none")
    if mode not in ("none", "soft", "projected"):
        raise SystemExit(f"{run_dir} conservation_mode {mode!r} is not supported")
    if saved.get("penetration") != "p05":
        raise SystemExit(f"{run_dir} is {saved.get('penetration')}; sampling-seed tests exist for p05")
    return saved, config


def open_sampling_tests(saved: dict, config: dict, sampling_seeds: list[int]):
    from MSTT.config import task_spec
    from reimplementation.common.data.state_windows import open_split_dataset

    task = task_spec(saved["task"])
    prepared = Path(config["data_root"]).resolve()
    manifest = json.loads((prepared / "split_manifest.json").read_text(encoding="utf-8"))
    expected = int(manifest["actual_sample_counts"]["test"])
    scaler = (prepared / "state_normalization.json").resolve()
    selector = {
        "input_type": task.input_type,
        "target_type": task.target_type,
        "num_nodes": int(config["num_nodes"]),
    }
    datasets = []
    reference_y = None
    reference_x = None
    for sampling_seed in sampling_seeds:
        root = sampling_window_root(prepared, sampling_seed).resolve()
        linked_scaler = (root / "state_normalization.json").resolve()
        if linked_scaler != scaler:
            raise SystemExit(f"sampling seed {sampling_seed} scaler {linked_scaler} != {scaler}")
        dataset = open_split_dataset(
            root,
            "p05",
            "test",
            selector,
            n_his=int(config["input_window"]),
            num_nodes=int(config["num_nodes"]),
            expected_count=expected,
        )
        if reference_y is None:
            reference_y = np.array(dataset.y_raw, copy=True)
            reference_x = np.array(dataset.x, copy=True)
        else:
            if not np.array_equal(dataset.y_raw, reference_y):
                raise SystemExit(f"sampling seed {sampling_seed} labels differ from seed {sampling_seeds[0]}")
            if np.array_equal(dataset.x, reference_x):
                raise SystemExit(f"sampling seed {sampling_seed} observations match seed {sampling_seeds[0]}")
        datasets.append(dataset)
        print(
            f"[sampling] seed={sampling_seed} samples={len(dataset)} "
            f"x={tuple(dataset.x.shape)} y={tuple(dataset.y_raw.shape)}",
            flush=True,
        )
    return datasets


def aggregate_metrics(rows: list[dict]) -> tuple[dict[str, float], dict[str, float]]:
    if not rows:
        raise ValueError("no sampling rows to average")
    keys = [key for key in rows[0] if key not in ("sampling_seed", "sample_count") and isinstance(rows[0][key], float)]
    mean = {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}
    std = {key: float(np.std([float(row[key]) for row in rows], ddof=1)) for key in keys}
    return mean, std


def score_run(run_dir: Path, sampling_seeds: list[int], device_name: str) -> dict:
    from MSTT.config import task_spec
    from MSTT.data.dataset import build_loader, target_scaler
    from MSTT.data.graph import load_r_graph
    from MSTT.model import MSTT
    from MSTT.trainer import assert_conservation_checkpoint, model_config, predict
    from MSTT.utils.device import resolve_device
    from MSTT.utils.metrics import raw_metrics
    from reimplementation.common.evaluation import collect_split_predictions
    from reimplementation.common.utils.checkpoint import load_checkpoint
    from reimplementation.common.utils.hashing import sha256_file

    import torch

    saved, config = load_run_config(run_dir)
    task = task_spec(saved["task"])
    prepared = Path(config["data_root"])
    datasets = open_sampling_tests(saved, config, sampling_seeds)
    device, device_info = resolve_device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = bool(config["allow_tf32"])
        torch.backends.cudnn.allow_tf32 = bool(config["allow_tf32"])
    mean_y, std_y = target_scaler(prepared, task)
    graph, graph_info = load_r_graph(
        Path(config["graph_path"]),
        r_nodes_path=Path(config["r_nodes"]),
        window_node_mapping=prepared / "node_mapping.csv",
        expected_nodes=int(config["num_nodes"]),
        direction=str(config["spatial_direction"]),
    )
    checkpoint_path = run_dir / "best_checkpoint.pt"
    if not checkpoint_path.is_file():
        raise SystemExit(f"missing {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    assert_conservation_checkpoint(
        checkpoint,
        config,
        normalization_sha256=sha256_file(prepared / "state_normalization.json"),
    )
    if checkpoint.get("task") != task.name or checkpoint.get("penetration_rate") != saved.get("penetration"):
        raise SystemExit("checkpoint task/penetration does not match the run directory")
    if checkpoint.get("graph_sha256") != graph_info["graph_sha256"]:
        raise SystemExit("checkpoint graph SHA256 differs from r_graph.npz")
    model = MSTT(model_config(config, task), graph, target_mean=mean_y, target_std=std_y).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    mean_arg = float(mean_y[0]) if mean_y.size == 1 else mean_y
    std_arg = float(std_y[0]) if std_y.size == 1 else std_y
    rows = []
    for sampling_seed, dataset in zip(sampling_seeds, datasets):
        loader = build_loader(
            dataset,
            batch_size=int(config["eval_batch_size"]),
            shuffle=False,
            seed=int(config["seed"]),
            num_workers=0,
        )
        mode = config.get("conservation_mode", "none")
        if mode == "none":
            packed = collect_split_predictions(
                loader,
                lambda batch: predict(model, batch, amp=str(config["amp"]), device=device),
                mean_y=mean_arg,
                std_y=std_arg,
                device=device,
            )
            metrics = raw_metrics(packed["pred_raw"], packed["y_raw"], task.output_channels, std_y)
        else:
            from MSTT.trainer import predict_with_aux
            from MSTT.utils.conservation_evaluation import collect_conservation_predictions
            from MSTT.utils.losses import RawScaleLoss

            criterion = RawScaleLoss(
                mean_y, std_y, kind=str(config["loss"]), huber_delta=float(config["huber_delta"]),
                channel_balance=str(config["channel_balance"]),
            ).to(device)
            packed, diagnostics = collect_conservation_predictions(
                loader,
                lambda batch: predict_with_aux(model, batch, amp=str(config["amp"]), device=device),
                criterion=criterion,
                device=device,
            )
            metrics = raw_metrics(packed["pred_raw"], packed["y_raw"], task.output_channels, std_y)
            metrics.update({key: float(value) for key, value in diagnostics.items() if isinstance(value, float)})
        if int(packed["pred_raw"].shape[0]) != len(dataset):
            raise SystemExit(f"sampling seed {sampling_seed} scored {packed['pred_raw'].shape[0]} != {len(dataset)}")
        row = {"sampling_seed": int(sampling_seed), "sample_count": int(packed["pred_raw"].shape[0])}
        kept = METRIC_KEYS + (
            "inflow_mae", "outflow_mae", "accumulation_start_mae", "accumulation_end_mae",
            "conservation_mae", "conservation_max_abs", "negative_fraction",
            "raw_mae", "raw_conservation_mae",
        )
        row.update({key: float(metrics[key]) for key in kept if key in metrics})
        rows.append(row)
        print(
            f"[sampling] {task.name} seed={sampling_seed} "
            f"mae={row['mae']:.10f} rmse={row['rmse']:.10f} wape={row['wape']:.10f}",
            flush=True,
        )
        del loader, packed
        gc.collect()
    summary = {
        "task": task.name,
        "penetration": saved.get("penetration"),
        "run_dir": run_dir.resolve().as_posix(),
        "checkpoint": checkpoint_path.resolve().as_posix(),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "selection": "validation MAE; sampling tests were not used to select the checkpoint",
        "device": device_info,
        "per_seed": rows,
    }
    summary["mean"], summary["std"] = aggregate_metrics(rows)
    summary["std_ddof"] = 1
    del datasets, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main(argv=None) -> dict:
    args = parse_args(argv)
    seeds = [int(seed) for seed in args.sampling_seeds]
    if len(seeds) != len(set(seeds)):
        raise SystemExit("sampling seeds must be unique")
    run_dir = Path(args.run_dir).resolve()
    if not (run_dir / "resolved_config.json").is_file():
        raise SystemExit(f"missing resolved_config.json in {run_dir}")
    saved, config = load_run_config(run_dir)
    if args.check_data:
        open_sampling_tests(saved, config, seeds)
        print("[sampling] data check passed", flush=True)
        return {"status": "data-checked", "sampling_seeds": seeds, "task": saved.get("task")}
    summary = score_run(run_dir, seeds, args.device)
    output = args.output or (run_dir / "sampling_seed_means.json")
    atomic_write_json(output, summary)
    mean, std = summary["mean"], summary["std"]
    print(
        f"[sampling] {summary['task']} mean_mae={mean['mae']:.10f} std_mae={std['mae']:.10f} "
        f"mean_rmse={mean['rmse']:.10f} mean_wape={mean['wape']:.10f}",
        flush=True,
    )
    print(f"[sampling] wrote {output}", flush=True)
    return summary


if __name__ == "__main__":
    main()
