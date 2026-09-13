"""Train R-only STID. Does not write into prepared data, graphs, or other models.

Do not pip install BasicTS, EasyTorch, or LibCity. Suggested (not executed).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from reimplementation.common.data.r_only_npz_dataset import rate_tag
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.utils.atomic_io import atomic_write_json
from reimplementation.common.utils.hashing import sha256_file
from reimplementation.common.utils.structured_logging import JsonlLogger
from reimplementation.stgcn.engine import resolve_path
from reimplementation.stid.engine import collect_data_info, collect_smoke_data_info, run_all_rates, run_p70_smoke
from reimplementation.stid.model.stid import CODE_VERSION
from reimplementation.stid.temporal_identity import ORIGINAL_DAY_SOURCE
from reimplementation.stid.validation import validate_no_graph_config

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = SCRIPT_PATH.parent / "configs" / "r_only_stid.json"

ORIGINAL_TO_MIGRATED = {
    "reference/STID/stid/arch/stid_arch.py::STID": "reimplementation/stid/model/stid.py::STID",
    "reference/STID/stid/arch/mlp.py::MultiLayerPerceptron": "reimplementation/stid/model/layers.py::MultiLayerPerceptron",
    "reference/STID/stid/arch/stid_arch.py::time-series flatten+Conv2d": "reimplementation/stid/model/layers.py::flatten_history_for_conv2d",
    "reference/STID/scripts/data_preparation/PEMS04/generate_training_data.py::time_of_day": "reimplementation/stid/temporal_identity.py::time_of_day_fraction",
    "reference/STID/scripts/data_preparation/PEMS04/generate_training_data.py::day_of_week": "reimplementation/stid/temporal_identity.py::sequential_day_of_week_index",
    "reference/STID/basicts/metrics/mae.py::masked_mae": "reimplementation/stid/losses.py::mae_all (zeros kept)",
    "output_len": "original 12 future steps -> R-only output_len=1 last-observed-step",
}


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train R-only STID (PyTorch port of original zezhishao/STID). No graph inputs."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", dest="data_root", type=Path, default=None)
    parser.add_argument("--r-nodes", dest="r_nodes", type=Path, default=None)
    parser.add_argument("--node-mapping", dest="node_mapping", type=Path, default=None)
    parser.add_argument("--normalization", type=Path, default=None)
    parser.add_argument("--split-manifest", dest="split_manifest", type=Path, default=None)
    parser.add_argument("--output-root", dest="output_root", type=Path, default=None)
    parser.add_argument("--rates", type=int, nargs="+", default=None)
    parser.add_argument("--n-his", dest="n_his", type=int, default=None)
    parser.add_argument("--output-window", dest="output_window", type=int, default=None)
    parser.add_argument("--target-mode", dest="target_mode", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test-only", action="store_true")
    parser.add_argument("--time-series-emb-dim", dest="embed_dim", type=int, default=None)
    parser.add_argument("--node-dim", dest="node_dim", type=int, default=None)
    parser.add_argument("--temp-dim-tid", dest="temp_dim_tid", type=int, default=None)
    parser.add_argument("--temp-dim-diw", dest="temp_dim_diw", type=int, default=None)
    parser.add_argument("--time-of-day-size", dest="time_of_day_size", type=int, default=None)
    parser.add_argument("--num-block", dest="num_layer", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--if-spatial", dest="if_spatial", type=int, choices=(0, 1), default=None)
    parser.add_argument("--if-time-in-day", dest="if_time_in_day", type=int, choices=(0, 1), default=None)
    parser.add_argument("--if-day-in-week", dest="if_day_in_week", type=int, choices=(0, 1), default=None)
    parser.add_argument("--input-dim", dest="input_dim", type=int, default=None)
    parser.add_argument(
        "--day-of-week-source",
        dest="day_of_week_source",
        type=str,
        default=None,
        help="original_sequential_index_mod_7 (default, matches PeMS STID) or calendar",
    )
    return parser.parse_args()


def overlay(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    mapping = {
        "data_root": args.data_root,
        "r_nodes": args.r_nodes,
        "node_mapping": args.node_mapping,
        "normalization": args.normalization,
        "split_manifest": args.split_manifest,
        "output_root": args.output_root,
        "target_mode": args.target_mode,
        "seed": args.seed,
        "device": args.device,
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "embed_dim": args.embed_dim,
        "node_dim": args.node_dim,
        "temp_dim_tid": args.temp_dim_tid,
        "temp_dim_diw": args.temp_dim_diw,
        "time_of_day_size": args.time_of_day_size,
        "num_layer": args.num_layer,
        "dropout": args.dropout,
        "input_dim": args.input_dim,
        "day_of_week_source": args.day_of_week_source,
    }
    for key, value in mapping.items():
        if value is not None:
            config[key] = value if not isinstance(value, Path) else value.as_posix()
    if args.n_his is not None:
        config["n_his"] = int(args.n_his)
        config["input_len"] = int(args.n_his)
    if args.output_window is not None:
        config["output_window"] = int(args.output_window)
        config["output_len"] = int(args.output_window)
    if args.if_spatial is not None:
        config["if_node"] = bool(args.if_spatial)
        config["if_spatial"] = bool(args.if_spatial)
    if args.if_time_in_day is not None:
        config["if_T_i_D"] = bool(args.if_time_in_day)
        config["if_time_in_day"] = bool(args.if_time_in_day)
    if args.if_day_in_week is not None:
        config["if_D_i_W"] = bool(args.if_day_in_week)
        config["if_day_in_week"] = bool(args.if_day_in_week)
    if args.rates is not None:
        config["rates"] = [rate_tag(item) for item in args.rates]
    return config


def _assert_task(config: dict[str, Any]) -> None:
    validate_no_graph_config(config)
    if str(config.get("target_mode")) != "last-observed-step":
        raise ReimplementationError("target_mode must be last-observed-step")
    if int(config.get("output_len", config.get("output_window", 0))) != 1:
        raise ReimplementationError("R-only STID output_len must be 1")
    if int(config.get("input_len", config.get("n_his", 0))) != 12:
        raise ReimplementationError("R-only STID input_len must be 12")
    if int(config.get("time_of_day_size", 288)) != 288:
        raise ReimplementationError("time_of_day_size must stay 288 for 5-minute slots")
    source = str(config.get("day_of_week_source", ORIGINAL_DAY_SOURCE))
    if source == "calendar" and not config.get("weekday_mapping"):
        raise ReimplementationError("calendar day-of-week requires weekday_mapping JSON")
    if bool(config.get("if_D_i_W", config.get("if_day_in_week", True))) and source not in {
        ORIGINAL_DAY_SOURCE,
        "calendar",
    }:
        raise ReimplementationError(f"unknown day_of_week_source {source}")


def source_manifest(root: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    skip_parts = {"prepared_data", "experiments", "__pycache__", "tests", "test_artifacts"}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in skip_parts for part in path.parts):
            continue
        if path.suffix not in {".py", ".json", ".md"}:
            continue
        records[path.relative_to(root).as_posix()] = sha256_file(path)
    return records


def original_hashes(project_root: Path) -> dict[str, str]:
    files = [
        "reference/STID/stid/arch/stid_arch.py",
        "reference/STID/stid/arch/mlp.py",
        "reference/STID/stid/PEMS04.py",
        "reference/STID/scripts/data_preparation/PEMS04/generate_training_data.py",
        "reference/STID/basicts/metrics/mae.py",
        "reference/STID/LICENSE",
        "reference/STID/readme.md",
    ]
    return {name: sha256_file(project_root / name) for name in files if (project_root / name).is_file()}


def main() -> None:
    args = parse_args()
    config = overlay(load_config(args.config), args)
    if "input_len" not in config:
        config["input_len"] = int(config.get("n_his", 12))
    if "output_len" not in config:
        config["output_len"] = int(config.get("output_window", 1))
    _assert_task(config)
    data_root = resolve_path(PROJECT_ROOT, config["data_root"])
    output_root = resolve_path(PROJECT_ROOT, config["output_root"])
    protected_dirs = [
        PROJECT_ROOT / "reimplementation" / "dcrnn",
        PROJECT_ROOT / "reimplementation" / "graph_wavenet",
        PROJECT_ROOT / "reimplementation" / "stsgcn",
        PROJECT_ROOT / "reimplementation" / "stfgnn",
        PROJECT_ROOT / "reimplementation" / "pdformer",
        PROJECT_ROOT / "reimplementation" / "stgcn" / "prepared_data",
        PROJECT_ROOT / "analysis" / "graph" / "r_graph",
    ]
    for path in protected_dirs:
        if output_root == path or path in output_root.parents:
            raise ReimplementationError(f"refusing to write STID outputs into {path}")
    if args.smoke_test_only:
        config["device"] = "cpu"
        data_info = collect_smoke_data_info(data_root, config)
        payload = run_p70_smoke(
            config,
            data_root=data_root,
            data_info=data_info,
            project_root=PROJECT_ROOT,
        )
        payload["original_to_migrated"] = ORIGINAL_TO_MIGRATED
        payload["python"] = sys.version
        payload["torch"] = torch.__version__
        payload["platform"] = platform.platform()
        payload["code_version"] = CODE_VERSION
        print(json.dumps({k: payload[k] for k in ("status", "training_loss", "prediction_shape")}, indent=2))
        return
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite and args.resume is None:
        raise ReimplementationError(f"{output_root} is not empty; pass --overwrite or --resume")
    data_info = collect_data_info(data_root, config, project_root=PROJECT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "resolved_config.json", config)
    atomic_write_json(output_root / "source_code_manifest.json", source_manifest(SCRIPT_PATH.parent))
    atomic_write_json(output_root / "original_hashes.json", original_hashes(PROJECT_ROOT))
    logger = JsonlLogger(output_root / "training_log.jsonl")
    try:
        run_all_rates(
            config,
            data_root=data_root,
            output_root=output_root,
            data_info=data_info,
            logger=logger,
            resume_path=args.resume,
        )
    finally:
        logger.close()


if __name__ == "__main__":
    try:
        main()
    except ReimplementationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
