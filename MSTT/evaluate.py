"""Re-evaluate a finished MSTT run from its checkpoint, independent of training.

    python MSTT/evaluate.py --task All2All --penetration p30
    python MSTT/evaluate.py --run-dir <.../MSTT/p30/All2All/seed_42> --splits validation test

Rebuilds data and model from ``resolved_config.json``, loads
``best_checkpoint.pt`` (or ``--checkpoint``), and writes
``reeval/<split>_metrics.json`` inside the run directory. Stored official
metrics are never overwritten; the report lists the largest difference
against them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MSTT.config import TASKS, load_config, rate_tag  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an MSTT checkpoint.")
    parser.add_argument("--run-dir", "--run_dir", dest="run_dir", type=Path, default=None)
    parser.add_argument("--task", choices=list(TASKS), default=None)
    parser.add_argument("--penetration", "--rate", dest="penetration", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", "--output_root", dest="output_root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["test"], choices=["validation", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-predictions", "--save_predictions", dest="save_predictions", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    from MSTT.trainer import TaskRun, run_directory
    from reimplementation.common.evaluation import save_predictions_npz
    from reimplementation.common.utils.atomic_io import atomic_write_json
    from reimplementation.common.utils.checkpoint import load_checkpoint

    run_dir = args.run_dir
    if run_dir is None:
        if args.task is None or args.penetration is None:
            raise SystemExit("pass --run-dir or both --task and --penetration")
        root = args.output_root or Path(load_config(None)["output_root"])
        run_dir = run_directory(root, rate_tag(args.penetration), args.task, args.seed)
    run_dir = Path(run_dir).resolve()
    saved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    config = load_config(None)
    config.update({key: saved[key] for key in config if key in saved})
    if args.device is not None:
        config["device"] = args.device
    output_root = run_dir.parents[2]
    run = TaskRun(config, task=saved["task"], rate=saved["penetration"], output_root=output_root, splits=tuple(args.splits))
    if run.run_dir != run_dir:
        raise SystemExit(f"resolved run directory {run.run_dir} != {run_dir}")
    checkpoint = load_checkpoint(args.checkpoint or run_dir / "best_checkpoint.pt", map_location=run.device)
    if checkpoint.get("task") != run.task.name or checkpoint.get("penetration_rate") != run.rate:
        raise SystemExit("checkpoint task/penetration does not match the run directory")
    if checkpoint.get("graph_sha256") != run.graph_info["graph_sha256"]:
        raise SystemExit("checkpoint graph SHA256 differs from the current r_graph.npz")
    model = run.build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    report = {"run_dir": run_dir.as_posix(), "checkpoint_epoch": int(checkpoint["epoch"]), "splits": {}}
    for split in args.splits:
        metrics, packed = run.evaluate(model, split, config.get("max_eval_batches"))
        atomic_write_json(run_dir / "reeval" / f"{split}_metrics.json", metrics)
        if args.save_predictions:
            save_predictions_npz(run_dir / "reeval" / f"{split}_predictions.npz", packed)
        stored_path = run_dir / f"{split}_metrics.json"
        entry = {key: metrics[key] for key in ("mae", "rmse", "mape_nonzero", "wape", "sample_count")}
        if stored_path.is_file():
            stored = json.loads(stored_path.read_text(encoding="utf-8"))
            entry["max_abs_diff_vs_stored"] = max(abs(float(metrics[key]) - float(stored[key])) for key in ("mae", "rmse", "mape_nonzero", "wape"))
        report["splits"][split] = entry
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
