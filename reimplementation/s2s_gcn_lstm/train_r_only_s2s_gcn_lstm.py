"""Train the 12-to-1 Seq2seq GCN-LSTM reimplementation.

This is not an official implementation. Full multi-rate training writes only
under the Xuancheng26 results directory for this model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from data_root import runtime_roots, xuancheng26_model_config
from reimplementation.common.data.r_only_npz_dataset import rate_tag
from reimplementation.common.data.state_windows import add_state_type_arguments, apply_state_types
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.utils.atomic_io import atomic_write_json
from reimplementation.common.utils.hashing import sha256_file
from reimplementation.common.utils.structured_logging import JsonlLogger
from reimplementation.s2s_gcn_lstm.engine import collect_data_info, collect_smoke_data_info, run_all_rates, run_p70_smoke
from reimplementation.s2s_gcn_lstm.model.s2s_gcn_lstm import CODE_VERSION, MODEL_LABEL
from reimplementation.s2s_gcn_lstm.validation import assert_task_config
from reimplementation.stgcn.engine import resolve_path, resolve_training_output

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = xuancheng26_model_config("s2s_gcn_lstm")


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReimplementationError(f"missing config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Train {MODEL_LABEL}. Not an official paper implementation."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", dest="data_root", type=Path, default=None)
    parser.add_argument("--spatial-adjacency", dest="spatial_adjacency", type=Path, default=None)
    parser.add_argument("--r-graph", dest="r_graph", type=Path, default=None)
    parser.add_argument("--r-nodes", dest="r_nodes", type=Path, default=None)
    parser.add_argument("--node-mapping", dest="node_mapping", type=Path, default=None)
    parser.add_argument("--normalization", type=Path, default=None)
    parser.add_argument("--split-manifest", dest="split_manifest", type=Path, default=None)
    parser.add_argument("--output-root", dest="output_root", type=Path, default=None)
    parser.add_argument("--rates", type=int, nargs="+", default=None)
    parser.add_argument("--n-his", dest="n_his", type=int, default=None)
    parser.add_argument("--output-steps", dest="output_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=None)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--gcn-hidden-dim", dest="gcn_hidden_dim", type=int, default=None)
    parser.add_argument("--lstm-hidden-dim", dest="lstm_hidden_dim", type=int, default=None)
    parser.add_argument("--encoder-gcn-layers", dest="encoder_gcn_layers", type=int, default=None)
    parser.add_argument("--decoder-gcn-layers", dest="decoder_gcn_layers", type=int, default=None)
    parser.add_argument("--lstm-layers", dest="lstm_layers", type=int, default=None)
    parser.add_argument("--early-stopping-patience", dest="early_stopping_patience", type=int, default=None)
    parser.add_argument("--lr-factor", dest="lr_factor", type=float, default=None)
    parser.add_argument("--lr-patience", dest="lr_patience", type=int, default=None)
    parser.add_argument("--gradient-clip", dest="gradient_clip", type=float, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test-only", action="store_true")
    add_state_type_arguments(parser)
    return parser.parse_args()


def overlay(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    mapping = {
        "data_root": args.data_root,
        "spatial_adjacency": args.spatial_adjacency,
        "r_graph": args.r_graph,
        "r_nodes": args.r_nodes,
        "node_mapping": args.node_mapping,
        "normalization": args.normalization,
        "split_manifest": args.split_manifest,
        "output_root": args.output_root,
        "seed": args.seed,
        "device": args.device,
        "num_workers": args.num_workers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "gcn_hidden_dim": args.gcn_hidden_dim,
        "lstm_hidden_dim": args.lstm_hidden_dim,
        "encoder_gcn_layers": args.encoder_gcn_layers,
        "decoder_gcn_layers": args.decoder_gcn_layers,
        "lstm_layers": args.lstm_layers,
        "early_stopping_patience": args.early_stopping_patience,
        "lr_factor": args.lr_factor,
        "lr_patience": args.lr_patience,
        "gradient_clip": args.gradient_clip,
    }
    for key, value in mapping.items():
        if value is not None:
            config[key] = value.as_posix() if isinstance(value, Path) else value
    if args.n_his is not None:
        config["n_his"] = int(args.n_his)
        config["input_window"] = int(args.n_his)
    if args.output_steps is not None:
        config["output_steps"] = int(args.output_steps)
        config["output_window"] = int(args.output_steps)
    if args.rates is not None:
        config["rates"] = [rate_tag(item) for item in args.rates]
    return config


def source_manifest(root: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    skip_parts = {"__pycache__", "tests"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in skip_parts for part in path.parts):
            continue
        if path.suffix not in {".py", ".json", ".md"}:
            continue
        records[path.relative_to(root).as_posix()] = sha256_file(path)
    return records


def main() -> None:
    args = parse_args()
    config = overlay(load_config(args.config), args)
    config = apply_state_types(config, args, model_name="s2s_gcn_lstm")
    config.setdefault("n_his", 12)
    config.setdefault("input_window", int(config["n_his"]))
    config.setdefault("output_steps", 1)
    config.setdefault("output_window", int(config["output_steps"]))
    assert_task_config(config)
    data_root = resolve_path(PROJECT_ROOT, config["data_root"])
    if args.smoke_test_only:
        config["device"] = "cpu"
        data_info = collect_smoke_data_info(data_root, config, project_root=PROJECT_ROOT)
        payload = run_p70_smoke(
            config,
            data_root=data_root,
            data_info=data_info,
            project_root=PROJECT_ROOT,
        )
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "model_label": payload["model_label"],
                    "training_loss": payload["training_loss"],
                    "prediction_shape": payload["prediction_shape"],
                    "trace": payload["trace"],
                    "gradient_status": payload["gradient_status"],
                    "edge_weight": payload["edge_weight"],
                },
                indent=2,
            )
        )
        return
    output_root = resolve_training_output(PROJECT_ROOT, config["output_root"], model_name="s2s_gcn_lstm")
    protected = [
        PROJECT_ROOT / "reimplementation" / name
        for name in (
            "stgcn",
            "dcrnn",
            "graph_wavenet",
            "stsgcn",
            "stfgnn",
            "pdformer",
            "stid",
            "staeformer",
        )
    ]
    for path in protected:
        if output_root == path or path in output_root.parents:
            raise ReimplementationError(f"refusing to write Seq2seq GCN-LSTM outputs into {path}")
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite and args.resume is None:
        raise ReimplementationError(f"{output_root} is not empty; pass --overwrite or --resume")
    data_info = collect_data_info(data_root, config, project_root=PROJECT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in config.items()}
    atomic_write_json(output_root / "resolved_config.json", serializable)
    atomic_write_json(
        output_root / "environment.json",
        {"code_version": CODE_VERSION, "model_label": MODEL_LABEL, "torch": torch.__version__, **runtime_roots()},
    )
    atomic_write_json(output_root / "source_code_manifest.json", source_manifest(SCRIPT_PATH.parent))
    atomic_write_json(
        output_root / "paper.json",
        {
            "official_code": False,
            "model_label": MODEL_LABEL,
            "doi": "10.1109/TITS.2022.3168865",
        },
    )
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
