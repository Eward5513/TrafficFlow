"""Train R-only STAEformer. Does not write into prepared data, graphs, or other models.

Do not pip install torchinfo or matplotlib for this entry point. Suggested (not executed).
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
from reimplementation.staeformer.engine import collect_data_info, collect_smoke_data_info, run_all_rates, run_p70_smoke
from reimplementation.staeformer.model.staeformer import CODE_VERSION
from reimplementation.staeformer.temporal_features import ORIGINAL_PEMS_NPZ_DOW
from reimplementation.staeformer.validation import validate_no_graph_config

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = SCRIPT_PATH.parent / "configs" / "r_only_staeformer.json"

ORIGINAL_TO_MIGRATED = {
    "reference/STAEformer/model/STAEformer.py::STAEformer": "reimplementation/staeformer/model/staeformer.py::STAEformer",
    "reference/STAEformer/model/STAEformer.py::AttentionLayer": "reimplementation/staeformer/model/attention.py::AttentionLayer",
    "reference/STAEformer/model/STAEformer.py::SelfAttentionLayer": "reimplementation/staeformer/model/layers.py::SelfAttentionLayer",
    "reference/STAEformer/model/STAEformer.py::mixed projection": "reimplementation/staeformer/model/layers.py::flatten_time_hidden",
    "reference/STAEformer/model/train.py::nn.HuberLoss PEMS04": "reimplementation/staeformer/losses.py::HuberRawLoss delta=1.0",
    "Torch-MTS generate_training_data.py tod/dow for PeMS npz": "reimplementation/staeformer/temporal_features.py",
    "out_steps": "original 12 future steps -> R-only out_steps=1 last-observed-step",
}


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train R-only STAEformer (PyTorch port of original XDZhelheim/STAEformer). No graph inputs."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", dest="data_root", type=Path, default=None)
    parser.add_argument("--r-nodes", dest="r_nodes", type=Path, default=None)
    parser.add_argument("--node-mapping", dest="node_mapping", type=Path, default=None)
    parser.add_argument("--normalization", type=Path, default=None)
    parser.add_argument("--split-manifest", dest="split_manifest", type=Path, default=None)
    parser.add_argument("--output-root", dest="output_root", type=Path, default=None)
    parser.add_argument("--rates", type=int, nargs="+", default=None)
    parser.add_argument("--in-steps", dest="in_steps", type=int, default=None)
    parser.add_argument("--out-steps", dest="out_steps", type=int, default=None)
    parser.add_argument("--n-his", dest="n_his", type=int, default=None)
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
    parser.add_argument("--input-embedding-dim", dest="input_embedding_dim", type=int, default=None)
    parser.add_argument("--tod-embedding-dim", dest="tod_embedding_dim", type=int, default=None)
    parser.add_argument("--dow-embedding-dim", dest="dow_embedding_dim", type=int, default=None)
    parser.add_argument("--spatial-embedding-dim", dest="spatial_embedding_dim", type=int, default=None)
    parser.add_argument("--adaptive-embedding-dim", dest="adaptive_embedding_dim", type=int, default=None)
    parser.add_argument("--feed-forward-dim", dest="feed_forward_dim", type=int, default=None)
    parser.add_argument("--num-heads", dest="num_heads", type=int, default=None)
    parser.add_argument("--num-layers", dest="num_layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--use-mixed-proj", dest="use_mixed_proj", type=int, choices=(0, 1), default=None)
    parser.add_argument("--input-dim", dest="input_dim", type=int, default=None)
    parser.add_argument(
        "--day-of-week-source",
        dest="day_of_week_source",
        type=str,
        default=None,
        help="original_pems_npz_sequential_index_mod_7 (default, matches PeMS04 npz) or calendar",
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
        "input_embedding_dim": args.input_embedding_dim,
        "tod_embedding_dim": args.tod_embedding_dim,
        "dow_embedding_dim": args.dow_embedding_dim,
        "spatial_embedding_dim": args.spatial_embedding_dim,
        "adaptive_embedding_dim": args.adaptive_embedding_dim,
        "feed_forward_dim": args.feed_forward_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "input_dim": args.input_dim,
        "day_of_week_source": args.day_of_week_source,
    }
    for key, value in mapping.items():
        if value is not None:
            config[key] = value if not isinstance(value, Path) else value.as_posix()
    if args.in_steps is not None:
        config["in_steps"] = int(args.in_steps)
        config["n_his"] = int(args.in_steps)
    if args.n_his is not None:
        config["n_his"] = int(args.n_his)
        config["in_steps"] = int(args.n_his)
    if args.out_steps is not None:
        config["out_steps"] = int(args.out_steps)
        config["output_window"] = int(args.out_steps)
    if args.use_mixed_proj is not None:
        config["use_mixed_proj"] = bool(args.use_mixed_proj)
    if args.rates is not None:
        config["rates"] = [rate_tag(item) for item in args.rates]
    return config


def _assert_task(config: dict[str, Any]) -> None:
    validate_no_graph_config(config)
    if str(config.get("target_mode")) != "last-observed-step":
        raise ReimplementationError("target_mode must be last-observed-step")
    if int(config.get("out_steps", config.get("output_window", 0))) != 1:
        raise ReimplementationError("R-only STAEformer out_steps must be 1")
    if int(config.get("in_steps", config.get("n_his", 0))) != 12:
        raise ReimplementationError("R-only STAEformer in_steps must be 12")
    if int(config.get("steps_per_day", 288)) != 288:
        raise ReimplementationError("steps_per_day must stay 288 for 5-minute slots")
    if int(config.get("spatial_embedding_dim", 0)) != 0:
        raise ReimplementationError("official PeMS04 spatial_embedding_dim is 0; refusing to enable node embedding")
    source = str(config.get("day_of_week_source", ORIGINAL_PEMS_NPZ_DOW))
    if source == "calendar" and not config.get("weekday_mapping"):
        raise ReimplementationError("calendar day-of-week requires weekday_mapping JSON")
    if int(config.get("dow_embedding_dim", 24)) > 0 and source not in {
        ORIGINAL_PEMS_NPZ_DOW,
        "calendar",
    }:
        raise ReimplementationError(f"unknown day_of_week_source {source}")
    if int(config.get("input_dim", 3)) != 3:
        raise ReimplementationError("official PeMS04 input_dim is 3; refusing to change feature projection width")


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
        "reference/STAEformer/model/STAEformer.py",
        "reference/STAEformer/model/STAEformer.yaml",
        "reference/STAEformer/model/train.py",
        "reference/STAEformer/lib/data_prepare.py",
        "reference/STAEformer/lib/metrics.py",
        "reference/STAEformer/lib/utils.py",
        "reference/STAEformer/README.md",
    ]
    return {name: sha256_file(project_root / name) for name in files if (project_root / name).is_file()}


def main() -> None:
    args = parse_args()
    config = overlay(load_config(args.config), args)
    if "in_steps" not in config:
        config["in_steps"] = int(config.get("n_his", 12))
    if "out_steps" not in config:
        config["out_steps"] = int(config.get("output_window", 1))
    _assert_task(config)
    data_root = resolve_path(PROJECT_ROOT, config["data_root"])
    output_root = resolve_path(PROJECT_ROOT, config["output_root"])
    protected_dirs = [
        PROJECT_ROOT / "reimplementation" / "dcrnn",
        PROJECT_ROOT / "reimplementation" / "graph_wavenet",
        PROJECT_ROOT / "reimplementation" / "stsgcn",
        PROJECT_ROOT / "reimplementation" / "stfgnn",
        PROJECT_ROOT / "reimplementation" / "pdformer",
        PROJECT_ROOT / "reimplementation" / "stid",
        PROJECT_ROOT / "reimplementation" / "stgcn",
        PROJECT_ROOT / "reimplementation" / "stgcn" / "prepared_data",
        PROJECT_ROOT / "analysis" / "graph" / "r_graph",
    ]
    for path in protected_dirs:
        if output_root == path or path in output_root.parents:
            raise ReimplementationError(f"refusing to write STAEformer outputs into {path}")
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
