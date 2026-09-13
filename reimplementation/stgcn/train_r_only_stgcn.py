"""Train R-only STGCN. Does not write into prepared data or the R graph."""

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
from reimplementation.stgcn.engine import atomic_write_csv_rows, resolve_path, run_all_rates
from reimplementation.stgcn.model import CODE_VERSION

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = SCRIPT_PATH.parent / "configs" / "r_only_stgcn.json"


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train R-only STGCN (PyTorch port of original TF STGCN).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--weighted-adjacency", type=Path, default=None)
    parser.add_argument("--r-nodes", type=Path, default=None)
    parser.add_argument("--normalization", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--rates", type=int, nargs="+", default=None)
    parser.add_argument("--rate", type=int, default=None)
    parser.add_argument("--n-his", type=int, default=None)
    parser.add_argument("--target-mode", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--ks", dest="Ks", type=int, default=None)
    parser.add_argument("--kt", dest="Kt", type=int, default=None)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    parser.add_argument("--smoke-test-only", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def overlay(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    mapping = {
        "data_root": args.data_root,
        "weighted_adjacency": args.weighted_adjacency,
        "r_nodes": args.r_nodes,
        "normalization": args.normalization,
        "split_manifest": args.split_manifest,
        "output_root": args.output_root,
        "n_his": args.n_his,
        "target_mode": args.target_mode,
        "seed": args.seed,
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "Ks": args.Ks,
        "Kt": args.Kt,
        "num_workers": args.num_workers,
    }
    for key, value in mapping.items():
        if value is not None:
            config[key] = value if not isinstance(value, Path) else value.as_posix()
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


def main() -> None:
    args = parse_args()
    config = overlay(load_config(args.config), args)
    if str(config.get("target_mode")) != "last-observed-step":
        raise ReimplementationError("target_mode must be last-observed-step")
    data_root = resolve_path(PROJECT_ROOT, config["data_root"])
    adjacency = resolve_path(PROJECT_ROOT, config["weighted_adjacency"])
    r_nodes = resolve_path(PROJECT_ROOT, config["r_nodes"])
    output_root = resolve_path(PROJECT_ROOT, config["output_root"])
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite and args.resume is None:
        raise ReimplementationError(f"{output_root} is not empty; pass --overwrite or --resume")
    output_root.mkdir(parents=True, exist_ok=True)
    protected = [adjacency, r_nodes, data_root / "normalization.json"]
    before = {str(path): sha256_file(path) for path in protected if path.is_file()}
    atomic_write_json(
        output_root / "experiment_manifest.json",
        {
            "model_name": config.get("model_name", "stgcn"),
            "framework": "pytorch",
            "code_version": CODE_VERSION,
            "target_mode": config.get("target_mode"),
            "n_his": config.get("n_his"),
            "output_steps": config.get("output_steps"),
            "num_nodes": config.get("num_nodes"),
            "seed": config.get("seed"),
            "rates": list(config["rates"]),
            "data_root": data_root.as_posix(),
            "weighted_adjacency": adjacency.as_posix(),
            "r_nodes": r_nodes.as_posix(),
            "output_root": output_root.as_posix(),
            "smoke_test_only": bool(args.smoke_test_only),
        },
    )
    atomic_write_json(output_root / "resolved_config.json", config)
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
        },
    )
    atomic_write_json(output_root / "source_code_manifest.json", source_manifest(PROJECT_ROOT / "reimplementation"))
    logger = JsonlLogger(output_root / "training_log.jsonl")
    print(
        f"[stgcn] stage=scanning output_root={output_root.as_posix()} rates={config['rates']} device={config.get('device')}",
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
            r_nodes_path=r_nodes,
            logger=logger,
            smoke_only=args.smoke_test_only,
            resume=args.resume,
        )
        after = {str(path): sha256_file(path) for path in protected if path.is_file()}
        if before != after:
            raise ReimplementationError("protected input files changed during training")
        atomic_write_json(output_root / "data_runtime_validation.json", result["data_info"])
        atomic_write_json(output_root / "graph_runtime_validation.json", result["graph_info"])
        if result.get("smoke") is not None:
            atomic_write_json(output_root / "smoke_test.json", result["smoke"])
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
            ],
            summary_rows,
        )
        logger.log({"stage": "completed", "model_name": "stgcn", "status": "ok"})
    except Exception as exc:
        logger.log({"stage": "failed", "model_name": "stgcn", "status": "failed", "latest_error": str(exc)})
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    try:
        main()
    except ReimplementationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
