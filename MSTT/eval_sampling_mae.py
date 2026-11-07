"""Score finished MSTT checkpoints on the five p05 sampling-seed test sets.

The test windows come from
``training_data_3s/sampling_seeds/seed_<id>/window_data``. The graph, node
order, and target scaler stay on the original paths stored in
``resolved_config.json``. Sampling roots do not need ``node_mapping.csv`` or
``validation_summary.json``. Official ``test_metrics.json`` is not rewritten.

The reported value is the unweighted mean of the five per-seed raw metrics.
Each seed has the same number of windows, and the full-state labels match.

    python MSTT/eval_sampling_mae.py --check-data
    python MSTT/eval_sampling_mae.py --wait
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_root import assert_training_results_output, xuancheng26_model_results_3s_dir  # noqa: E402
from reimplementation.common.utils.atomic_io import atomic_write_json  # noqa: E402

ABLATION_ROOT = xuancheng26_model_results_3s_dir("MSTT") / "ablation_100ep"
DEFAULT_NAMES = ("single_tsf_adaptive128_100ep", "ttfssf_adaptive128_100ep")
SAMPLING_SEEDS = (42, 43, 44, 45, 46)
METRIC_KEYS = ("mae", "rmse", "mape_nonzero", "wape")


def default_run_dirs() -> list[Path]:
    return [ABLATION_ROOT / name / "p05" / "I2I" / "seed_42" for name in DEFAULT_NAMES]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", type=Path, default=None, help="Finished MSTT run directory. Repeat for more than one.")
    parser.add_argument("--sampling-seeds", type=int, nargs="+", default=list(SAMPLING_SEEDS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--wait", action="store_true", help="Poll until every run has finished its scheduled epochs and the worker has exited.")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--check-data", action="store_true", help="Load the sampling windows and stop before any checkpoint or GPU work.")
    return parser.parse_args(argv)


def experiment_dir(run_dir: Path) -> Path:
    """``<experiment>/p05/I2I/seed_42`` -> ``<experiment>``."""
    resolved = Path(run_dir).resolve()
    if resolved.parts[-3:] != ("p05", "I2I", "seed_42"):
        raise SystemExit(f"{resolved} is not a p05/I2I/seed_42 run directory")
    return resolved.parents[2]


def read_json(path: Path, fallback=None):
    if not path.is_file():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def process_start(pid: int | None):
    if pid is None:
        return None
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def process_alive(pid, start) -> bool:
    return pid is not None and start is not None and process_start(int(pid)) == start


def history_complete(run_dir: Path, epochs: int) -> bool:
    history = read_json(run_dir / "training_history.json", [])
    if not isinstance(history, list) or len(history) != int(epochs):
        return False
    return [int(row["epoch"]) for row in history] == list(range(int(epochs)))


def run_state(run_dir: Path) -> dict:
    return read_json(experiment_dir(run_dir) / "launcher_state.json", {}) or {}


def wait_until_finished(run_dirs: list[Path], *, wait: bool, poll_seconds: int) -> None:
    from MSTT.config import load_config, with_legacy_defaults

    pending = {path.resolve() for path in run_dirs}
    announced: dict[Path, int] = {}
    while pending:
        still = set()
        for run_dir in sorted(pending):
            saved = with_legacy_defaults(read_json(run_dir / "resolved_config.json", {}))
            config = load_config(None)
            config.update({key: saved[key] for key in config if key in saved})
            epochs = int(config["epochs"])
            done = history_complete(run_dir, epochs)
            state = run_state(run_dir)
            worker = process_alive(state.get("training_pid"), state.get("training_process_start"))
            supervisor = process_alive(state.get("waiting_pid"), state.get("waiting_process_start"))
            history = read_json(run_dir / "training_history.json", [])
            seen = len(history) if isinstance(history, list) else 0
            if done and not worker and (run_dir / "best_checkpoint.pt").is_file():
                print(f"[sampling] {experiment_dir(run_dir).name} finished {epochs} epochs", flush=True)
                continue
            if not worker and not supervisor:
                raise SystemExit(
                    f"{experiment_dir(run_dir).name} stopped before {epochs} epochs "
                    f"(history={seen}, status={state.get('status')})"
                )
            if not wait:
                raise SystemExit(
                    f"{experiment_dir(run_dir).name} is still training "
                    f"(history={seen}/{epochs}). Re-run with --wait."
                )
            if announced.get(run_dir) != seen:
                print(
                    f"[sampling] waiting for {experiment_dir(run_dir).name}: "
                    f"history={seen}/{epochs} status={state.get('status')}",
                    flush=True,
                )
                announced[run_dir] = seen
            still.add(run_dir)
        pending = still
        if pending:
            time.sleep(poll_seconds)


def sampling_window_root(prepared_root: Path, sampling_seed: int) -> Path:
    return prepared_root.parent / "sampling_seeds" / f"seed_{int(sampling_seed)}" / "window_data"


def load_run_config(run_dir: Path) -> tuple[dict, dict]:
    from MSTT.config import load_config, with_legacy_defaults

    saved = with_legacy_defaults(json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8")))
    config = load_config(None)
    config.update({key: saved[key] for key in config if key in saved})
    if config.get("conservation_mode", "none") != "none":
        raise SystemExit(f"{run_dir} conservation_mode is not none")
    return saved, config


def open_sampling_tests(config: dict, sampling_seeds: list[int]):
    from reimplementation.common.data.state_windows import open_split_dataset

    prepared = Path(config["data_root"]).resolve()
    manifest = json.loads((prepared / "split_manifest.json").read_text(encoding="utf-8"))
    expected = int(manifest["actual_sample_counts"]["test"])
    scaler = (prepared / "state_normalization.json").resolve()
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
            {"input_type": "inflow", "target_type": "inflow", "num_nodes": int(config["num_nodes"])},
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


def mean_metrics(rows: list[dict]) -> dict[str, float]:
    if not rows:
        raise ValueError("no sampling rows to average")
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in METRIC_KEYS}


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
    if task.name != "I2I" or saved.get("penetration") != "p05":
        raise SystemExit(f"{run_dir} is {saved.get('task')} {saved.get('penetration')}, expected I2I p05")
    prepared = Path(config["data_root"])
    datasets = open_sampling_tests(config, sampling_seeds)
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
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    assert_conservation_checkpoint(
        checkpoint,
        config,
        normalization_sha256=sha256_file(prepared / "state_normalization.json"),
    )
    if checkpoint.get("task") != task.name or checkpoint.get("penetration_rate") != "p05":
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
        packed = collect_split_predictions(
            loader,
            lambda batch: predict(model, batch, amp=str(config["amp"]), device=device),
            mean_y=mean_arg,
            std_y=std_arg,
            device=device,
        )
        metrics = raw_metrics(packed["pred_raw"], packed["y_raw"], task.output_channels, std_y)
        if int(packed["pred_raw"].shape[0]) != len(dataset):
            raise SystemExit(f"sampling seed {sampling_seed} scored {packed['pred_raw'].shape[0]} != {len(dataset)}")
        row = {"sampling_seed": int(sampling_seed), "sample_count": int(packed["pred_raw"].shape[0])}
        row.update({key: float(metrics[key]) for key in METRIC_KEYS})
        rows.append(row)
        print(
            f"[sampling] {experiment_dir(run_dir).name} seed={sampling_seed} "
            f"mae={row['mae']:.10f} rmse={row['rmse']:.10f} wape={row['wape']:.10f}",
            flush=True,
        )
        del loader, packed
        gc.collect()
    stored = read_json(run_dir / "test_metrics.json")
    summary = {
        "name": experiment_dir(run_dir).name,
        "run_dir": run_dir.resolve().as_posix(),
        "checkpoint": checkpoint_path.resolve().as_posix(),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "selection": "validation MAE; sampling tests were not used to select the checkpoint",
        "device": device_info,
        "per_seed": rows,
        "mean": mean_metrics(rows),
    }
    if stored and "mae" in stored:
        summary["training_test_mae"] = float(stored["mae"])
        summary["seed_42_abs_diff_vs_training_test"] = abs(rows[0]["mae"] - float(stored["mae"]))
    del datasets, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main(argv=None) -> dict:
    args = parse_args(argv)
    if args.poll_seconds < 1:
        raise SystemExit("--poll-seconds must be positive")
    seeds = [int(seed) for seed in args.sampling_seeds]
    if len(seeds) != len(set(seeds)):
        raise SystemExit("sampling seeds must be unique")
    run_dirs = [path.resolve() for path in (args.run_dir or default_run_dirs())]
    for run_dir in run_dirs:
        if not (run_dir / "resolved_config.json").is_file():
            raise SystemExit(f"missing resolved_config.json in {run_dir}")
    configs = []
    for run_dir in run_dirs:
        saved, config = load_run_config(run_dir)
        if saved.get("task") != "I2I" or saved.get("penetration") != "p05":
            raise SystemExit(f"{run_dir} is {saved.get('task')} {saved.get('penetration')}, expected I2I p05")
        configs.append(config)
    if any(config["data_root"] != configs[0]["data_root"] for config in configs):
        raise SystemExit("run directories do not share one prepared data root")
    if args.check_data:
        open_sampling_tests(configs[0], seeds)
        print("[sampling] data check passed", flush=True)
        return {"status": "data-checked", "sampling_seeds": seeds}
    wait_until_finished(run_dirs, wait=bool(args.wait), poll_seconds=int(args.poll_seconds))
    from MSTT.utils.device import resolve_device

    device, _device_info = resolve_device(args.device)
    output = args.output or (experiment_dir(run_dirs[0]).parent / "sampling_seed_means.json")
    output = assert_training_results_output(output, model_name="MSTT")
    models = [score_run(run_dir, seeds, str(device)) for run_dir in run_dirs]
    report = {
        "sampling_seeds": seeds,
        "aggregation": "unweighted mean of per-seed raw metrics",
        "prepared_data_root": configs[0]["data_root"],
        "models": models,
    }
    atomic_write_json(output, report)
    for model in models:
        mean = model["mean"]
        print(
            f"[sampling] {model['name']} mean_mae={mean['mae']:.10f} "
            f"mean_rmse={mean['rmse']:.10f} mean_wape={mean['wape']:.10f}",
            flush=True,
        )
    print(f"[sampling] wrote {output}", flush=True)
    return report


if __name__ == "__main__":
    main()
