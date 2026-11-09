"""Train Xuancheng26 TLAST. Does not write into prepared data or other models."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_root import runtime_roots
from reimplementation.common.data.r_only_npz_dataset import rate_tag
from reimplementation.common.data.state_windows import add_state_type_arguments, apply_state_types
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.utils.atomic_io import atomic_write_json
from reimplementation.common.utils.hashing import sha256_file
from reimplementation.common.utils.structured_logging import JsonlLogger
from reimplementation.stgcn.engine import resolve_path, resolve_training_output
from reimplementation.tlast.engine import check_real_batch, collect_data_info, run_all_rates
from reimplementation.tlast.model.tlast import CODE_VERSION
from reimplementation.tlast.validation import assert_task_config

DEFAULT_CONFIG = SCRIPT_PATH.parent / "configs" / "r_only_tlast.json"

def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Xuancheng26 TLAST on four traffic states.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", dest="data_root", type=Path, default=None)
    parser.add_argument("--output-root", dest="output_root", type=Path, default=None)
    parser.add_argument("--rates", nargs="+", default=None, help="fixed pXX/integer, or a dataset observation key such as p10_p20")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=None)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test-only", action="store_true")
    parser.add_argument(
        "--smoke-pipeline",
        action="store_true",
        help="One real epoch each of S2S inflow, A2A, and A2S outflow under output_root/_smoke.",
    )
    parser.add_argument("--use-input-residual", dest="use_input_residual", type=int, choices=(0, 1), default=None)
    add_state_type_arguments(parser)
    return parser.parse_args()


def overlay(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    mapping = {
        "data_root": args.data_root,
        "output_root": args.output_root,
        "seed": args.seed,
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_workers": args.num_workers,
    }
    for key, value in mapping.items():
        if value is not None:
            config[key] = value.as_posix() if isinstance(value, Path) else value
    if args.use_input_residual is not None:
        config["use_input_residual"] = bool(args.use_input_residual)
    if args.rates is not None:
        config["rates"] = [rate_tag(item) for item in args.rates]
    return config


def _prepare(args: argparse.Namespace, *, input_type: str | None = None, target_type: str | None = None) -> tuple[dict[str, Any], Path, Path]:
    config = overlay(load_config(args.config), args)
    if input_type is not None:
        args.input_type = input_type
    if target_type is not None:
        args.target_type = target_type
    config = apply_state_types(config, args, model_name="tlast")
    config["input_len"] = int(config.get("input_len", config.get("n_his", 12)))
    config["n_his"] = int(config["input_len"])
    config["output_len"] = int(config.get("output_len", config.get("output_window", 1)))
    config["output_window"] = int(config["output_len"])
    assert_task_config(config)
    data_root = resolve_path(PROJECT_ROOT, config["data_root"])
    output_root = resolve_training_output(PROJECT_ROOT, config["output_root"], model_name="tlast")
    return config, data_root, output_root


def _run(config: dict[str, Any], data_root: Path, output_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    from reimplementation.common.data.sampling_seeds import task_metadata_dir
    metadata_root = task_metadata_dir(output_root, config)
    metadata_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(metadata_root / "resolved_config.json", config)
    atomic_write_json(metadata_root / "environment.json", {"code_version": CODE_VERSION, **runtime_roots(), "python": sys.version, "torch": torch.__version__, "platform": platform.platform()})
    data_info = collect_data_info(data_root, config, project_root=PROJECT_ROOT)
    logger = JsonlLogger(metadata_root / "training_log.jsonl")
    try:
        result = run_all_rates(
            config,
            data_root=data_root,
            output_root=output_root,
            data_info=data_info,
            logger=logger,
            resume_path=args.resume,
            overwrite=bool(args.overwrite),
        )
    finally:
        logger.close()
    return result


def _smoke_one(args: argparse.Namespace, input_type: str, target_type: str) -> dict[str, Any]:
    config, data_root, output_root = _prepare(args, input_type=input_type, target_type=target_type)
    if args.smoke_test_only:
        config["device"] = "cpu"
    config["rates"] = ["p70"]
    data_info = collect_data_info(data_root, config, project_root=PROJECT_ROOT)
    report = check_real_batch(config, data_root=data_root, data_info=data_info, rate="p70")
    if args.smoke_test_only:
        return report
    config["epochs"] = 1
    config["output_root"] = (output_root.parent / "_smoke" / output_root.name).as_posix()
    smoke_root = resolve_training_output(PROJECT_ROOT, config["output_root"], model_name="tlast")
    smoke_args = argparse.Namespace(resume=None, overwrite=True)
    result = _run(config, data_root, smoke_root, smoke_args)
    report["one_epoch"] = {
        "status": result["rates"][0]["status"],
        "test_mae_raw": result["rates"][0]["test_mae_raw"],
        "prediction_checked": True,
        "output_root": smoke_root.as_posix(),
    }
    return report


def main() -> None:
    args = parse_args()
    if args.smoke_test_only or args.smoke_pipeline:
        specs = [("inflow", "inflow")] if args.smoke_test_only else [("inflow", "inflow"), ("all", "all"), ("all", "outflow")]
        reports = []
        for input_type, target_type in specs:
            reports.append(_smoke_one(args, input_type, target_type))
        print(json.dumps(reports, indent=2, default=str))
        return
    config, data_root, output_root = _prepare(args)
    _run(config, data_root, output_root, args)


if __name__ == "__main__":
    try:
        main()
    except ReimplementationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
