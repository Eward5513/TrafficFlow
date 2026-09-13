"""Train R-only Graph WaveNet. Does not write into prepared data, graphs, DCRNN, or STGCN."""

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
from reimplementation.graph_wavenet.engine import atomic_write_csv_rows, resolve_path, run_all_rates
from reimplementation.graph_wavenet.model.graph_wavenet import CODE_VERSION

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = SCRIPT_PATH.parent / "configs" / "r_only_gwn.json"

ORIGINAL_TO_MIGRATED = {
    "reference/Graph-WaveNet/model.py::nconv": "reimplementation/graph_wavenet/model/graph_wavenet.py::NConv",
    "reference/Graph-WaveNet/model.py::linear": "reimplementation/graph_wavenet/model/graph_wavenet.py::LinearConv",
    "reference/Graph-WaveNet/model.py::gcn": "reimplementation/graph_wavenet/model/graph_wavenet.py::GCN",
    "reference/Graph-WaveNet/model.py::gwnet": "reimplementation/graph_wavenet/model/graph_wavenet.py::GraphWaveNet",
    "reference/Graph-WaveNet/model.py filter/gate": "tanh(filter_conv) * sigmoid(gate_conv)",
    "reference/Graph-WaveNet/model.py residual/skip/end": "GraphWaveNet residual add, skip add, end_conv_1/2",
    "reference/Graph-WaveNet/model.py adaptive adj": "softmax(relu(nodevec1 @ nodevec2), dim=1)",
    "reference/Graph-WaveNet/util.py::asym_adj": "reimplementation/graph_wavenet/graph.py::asym_adj",
    "reference/Graph-WaveNet/util.py::load_adj doubletransition": "build_gwn_supports",
    "reference/Graph-WaveNet/util.py::masked_mae": "reimplementation/graph_wavenet/losses.py::MaskedMAERawLoss",
    "reference/Graph-WaveNet/engine.py Adam": "torch.optim.Adam(weight_decay=1e-4)",
    "reference/Graph-WaveNet/engine.py clip=5": "torch.nn.utils.clip_grad_norm_",
    "reference/Graph-WaveNet/engine.py pad (1,0,0,0)": "GraphWaveNet.engine_left_pad=1",
}


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train R-only Graph WaveNet (port of original PyTorch GWN).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--adjacency", type=Path, default=None)
    parser.add_argument("--adjacency-pickle", type=Path, default=None)
    parser.add_argument("--graph-metadata", type=Path, default=None)
    parser.add_argument("--graph-validation", type=Path, default=None)
    parser.add_argument("--r-nodes", type=Path, default=None)
    parser.add_argument("--normalization", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--rates", type=int, nargs="+", default=None)
    parser.add_argument("--rate", type=int, default=None)
    parser.add_argument("--seq-length", dest="seq_length", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--target-mode", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=None)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--blocks", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--kernel-size", dest="kernel_size", type=int, default=None)
    parser.add_argument("--gcn-order", dest="gcn_order", type=int, default=None)
    parser.add_argument("--adjtype", type=str, default=None)
    parser.add_argument("--gcn-bool", dest="gcn_bool", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--addaptadj", dest="addaptadj", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--randomadj", dest="randomadj", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test-only", action="store_true")
    parser.add_argument("--clip-negative-predictions", action="store_true")
    return parser.parse_args()


def overlay(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    mapping = {
        "data_root": args.data_root,
        "adjacency": args.adjacency,
        "adjacency_pickle": args.adjacency_pickle,
        "graph_metadata": args.graph_metadata,
        "graph_validation": args.graph_validation,
        "r_nodes": args.r_nodes,
        "normalization": args.normalization,
        "split_manifest": args.split_manifest,
        "output_root": args.output_root,
        "horizon": args.horizon,
        "target_mode": args.target_mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "blocks": args.blocks,
        "layers": args.layers,
        "kernel_size": args.kernel_size,
        "gcn_order": args.gcn_order,
        "adjtype": args.adjtype,
        "seed": args.seed,
        "device": args.device,
        "num_workers": args.num_workers,
    }
    for key, value in mapping.items():
        if value is not None:
            config[key] = value if not isinstance(value, Path) else value.as_posix()
    if args.seq_length is not None:
        config["seq_len"] = int(args.seq_length)
        config["seq_length"] = int(args.seq_length)
        config["n_his"] = int(args.seq_length)
    for flag in ("gcn_bool", "addaptadj", "randomadj"):
        value = getattr(args, flag)
        if value is not None:
            config[flag] = bool(value)
    if args.clip_negative_predictions:
        config["clip_negative_predictions"] = True
    if args.rate is not None:
        config["rates"] = [rate_tag(args.rate)]
    elif args.rates is not None:
        config["rates"] = [rate_tag(item) for item in args.rates]
    return config


def source_manifest(root: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    skip_parts = {"prepared_data", "experiments", "__pycache__"}
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
        "reference/Graph-WaveNet/model.py",
        "reference/Graph-WaveNet/engine.py",
        "reference/Graph-WaveNet/util.py",
        "reference/Graph-WaveNet/train.py",
        "reference/Graph-WaveNet/README.md",
    ]
    return {name: sha256_file(project_root / name) for name in files if (project_root / name).is_file()}


def main() -> None:
    args = parse_args()
    config = overlay(load_config(args.config), args)
    if str(config.get("target_mode")) != "last-observed-step":
        raise ReimplementationError("target_mode must be last-observed-step")
    if int(config.get("horizon", 0)) != 1:
        raise ReimplementationError("R-only Graph WaveNet horizon must be 1")
    if int(config.get("seq_len", 0)) != 12:
        raise ReimplementationError("R-only Graph WaveNet seq_len must be 12")
    if not bool(config.get("gcn_bool", True)):
        raise ReimplementationError("official Graph WaveNet command enables --gcn_bool")
    if not bool(config.get("addaptadj", True)):
        raise ReimplementationError("official Graph WaveNet command enables --addaptadj")
    if str(config.get("adjtype")) != "doubletransition":
        raise ReimplementationError("official Graph WaveNet command uses --adjtype doubletransition")
    if bool(config.get("amp")):
        raise ReimplementationError("AMP is not part of the original Graph WaveNet trainer")
    data_root = resolve_path(PROJECT_ROOT, config["data_root"])
    adjacency = resolve_path(PROJECT_ROOT, config["adjacency"])
    pickle_path = resolve_path(PROJECT_ROOT, config["adjacency_pickle"])
    metadata_path = resolve_path(PROJECT_ROOT, config["graph_metadata"])
    graph_validation = resolve_path(PROJECT_ROOT, config["graph_validation"])
    r_nodes = resolve_path(PROJECT_ROOT, config["r_nodes"])
    output_root = resolve_path(PROJECT_ROOT, config["output_root"])
    dcrnn_output = PROJECT_ROOT / "reimplementation" / "dcrnn" / "experiments" / "r_only"
    if output_root.resolve() == dcrnn_output.resolve():
        raise ReimplementationError("GWN output_root must not be the DCRNN experiment directory")
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite and args.resume is None:
        raise ReimplementationError(f"{output_root} is not empty; pass --overwrite or --resume")
    output_root.mkdir(parents=True, exist_ok=True)
    protected = [
        adjacency,
        pickle_path,
        metadata_path,
        graph_validation,
        r_nodes,
        data_root / "normalization.json",
        PROJECT_ROOT / "reference" / "Graph-WaveNet" / "model.py",
        PROJECT_ROOT / "reference" / "Graph-WaveNet" / "util.py",
        PROJECT_ROOT / "reference" / "dcrnn" / "model" / "dcrnn_cell.py",
    ]
    before = {str(path): sha256_file(path) for path in protected if path.is_file()}
    atomic_write_json(
        output_root / "experiment_manifest.json",
        {
            "model_name": "graph_wavenet",
            "framework": "pytorch",
            "code_version": CODE_VERSION,
            "target_mode": config.get("target_mode"),
            "seq_len": config.get("seq_len"),
            "horizon": config.get("horizon"),
            "num_nodes": config.get("num_nodes"),
            "seed": config.get("seed"),
            "rates": list(config["rates"]),
            "data_root": data_root.as_posix(),
            "adjacency": adjacency.as_posix(),
            "adjacency_pickle": pickle_path.as_posix(),
            "r_nodes": r_nodes.as_posix(),
            "output_root": output_root.as_posix(),
            "smoke_test_only": bool(args.smoke_test_only),
            "resumed": args.resume is not None,
            "did_not_write_dcrnn_outputs": True,
        },
    )
    atomic_write_json(output_root / "resolved_config.json", config)
    atomic_write_json(output_root / "original_to_migrated_mapping.json", ORIGINAL_TO_MIGRATED)
    atomic_write_json(
        output_root / "environment.json",
        {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "numpy": __import__("numpy").__version__,
            "scipy": __import__("scipy").__version__,
            "platform": platform.platform(),
            "code_version": CODE_VERSION,
            "cuda_available": torch.cuda.is_available(),
            "device": str(config.get("device")),
            "cpu_thread_count": int(torch.get_num_threads()),
            "command": sys.argv,
            "original_gwn_sha256": original_hashes(PROJECT_ROOT),
        },
    )
    atomic_write_json(output_root / "source_code_manifest.json", source_manifest(PROJECT_ROOT / "reimplementation"))
    logger = JsonlLogger(output_root / "training_log.jsonl")
    print(
        f"[gwn] stage=scanning output_root={output_root.as_posix()} rates={config['rates']} device={config.get('device')}",
        flush=True,
    )
    try:
        result = run_all_rates(
            config,
            project_root=PROJECT_ROOT,
            rates=list(config["rates"]),
            data_root=data_root,
            output_root=output_root,
            adjacency_path=adjacency,
            pickle_path=pickle_path,
            metadata_path=metadata_path,
            graph_validation_path=graph_validation,
            r_nodes_path=r_nodes,
            logger=logger,
            smoke_only=args.smoke_test_only,
            resume=args.resume,
        )
        after = {str(path): sha256_file(path) for path in protected if path.is_file()}
        if before != after:
            raise ReimplementationError("protected input files changed during Graph WaveNet run")
        atomic_write_json(output_root / "data_runtime_validation.json", result["data_info"])
        atomic_write_json(output_root / "graph_runtime_validation.json", result["graph_info"])
        if result.get("smoke") is not None:
            atomic_write_json(output_root / "smoke_test.json", result["smoke"])
            atomic_write_json(output_root / "model_runtime_validation.json", result["smoke"].get("shape_info") or {})
        atomic_write_json(output_root / "overall_summary.json", result["rates"])
        summary_rows = []
        for row in result["rates"]:
            summary_rows.append(
                {
                    "rate": row.get("rate"),
                    "status": row.get("status", "trained"),
                    "best_epoch": row.get("best_epoch"),
                    "best_metric": row.get("best_metric"),
                    "parameter_count": row.get("parameter_count"),
                    "init_parameter_sha256": row.get("init_parameter_sha256"),
                    "validation_mae_raw": row.get("validation_mae_raw"),
                    "validation_rmse_raw": row.get("validation_rmse_raw"),
                    "validation_mape_nonzero": row.get("validation_mape_nonzero"),
                    "validation_wape": row.get("validation_wape"),
                    "test_mae_raw": row.get("test_mae_raw"),
                    "test_rmse_raw": row.get("test_rmse_raw"),
                    "test_mape_nonzero": row.get("test_mape_nonzero"),
                    "test_wape": row.get("test_wape"),
                    "test_negative_prediction_count": row.get("test_negative_prediction_count"),
                    "test_negative_prediction_fraction": row.get("test_negative_prediction_fraction"),
                }
            )
        atomic_write_csv_rows(
            output_root / "overall_summary.csv",
            [
                "rate",
                "status",
                "best_epoch",
                "best_metric",
                "parameter_count",
                "init_parameter_sha256",
                "validation_mae_raw",
                "validation_rmse_raw",
                "validation_mape_nonzero",
                "validation_wape",
                "test_mae_raw",
                "test_rmse_raw",
                "test_mape_nonzero",
                "test_wape",
                "test_negative_prediction_count",
                "test_negative_prediction_fraction",
            ],
            summary_rows,
        )
        logger.log({"stage": "completed", "model_name": "graph_wavenet", "status": "ok"})
        print("[gwn] stage=completed status=ok", flush=True)
    except Exception as exc:
        logger.log({"stage": "failed", "model_name": "graph_wavenet", "status": "failed", "latest_error": str(exc)})
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    try:
        main()
    except ReimplementationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
