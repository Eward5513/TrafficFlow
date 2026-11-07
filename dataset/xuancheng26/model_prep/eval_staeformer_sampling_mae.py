#!/usr/bin/env python3
"""Score the published STAEformer p05 inflow checkpoint on five test samplings.

Loads the existing best checkpoint and does not train. Each sampling seed is
evaluated separately; the reported value is the mean of those five MAEs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reimplementation.common.data.r_only_npz_dataset import build_dataloader, loss_scale
from reimplementation.common.data.state_windows import open_split_dataset, read_state_target_scaler
from reimplementation.common.evaluation import collect_split_predictions, compute_raw_metrics
from reimplementation.staeformer.engine import (
    assemble_batch_payload,
    build_model,
    model_kwargs_from_payload,
)

TRAINING_ROOT = Path("/data5/tzhang174/TrafficFlow/xuancheng26/training_data_3s")
RESULTS_ROOT = Path("/data5/tzhang174/TrafficFlow/xuancheng26/training_results_3s/staeformer")
SEEDS = (42, 43, 44, 45, 46)
EXPECTED_COUNT = 1385


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="in_inflow__out_inflow")
    args = parser.parse_args()
    run_dir = RESULTS_ROOT / args.run / "p05" / "seed_42"
    checkpoint = run_dir / "best_checkpoint.pt"
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    published = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
    published_mae = float(published["mae"])
    target_type = str(config["target_type"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    reference_y = None
    rows = []
    for sampling_seed in SEEDS:
        data_root = TRAINING_ROOT / "sampling_seeds" / f"seed_{sampling_seed}" / "window_data"
        dataset = open_split_dataset(
            data_root,
            "p05",
            "test",
            config,
            n_his=int(config["in_steps"]),
            num_nodes=int(config["num_nodes"]),
            expected_count=EXPECTED_COUNT,
        )
        if reference_y is None:
            reference_y = np.array(dataset.y_raw, copy=True)
        elif not np.array_equal(dataset.y_raw, reference_y):
            raise RuntimeError(f"sampling seed {sampling_seed} test labels differ from seed 42")
        mean_y, std_y = read_state_target_scaler(data_root, target_type)
        loader = build_dataloader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=0,
            seed=int(config["seed"]),
        )
        manifest = json.loads((data_root / "split_manifest.json").read_text(encoding="utf-8"))

        def predict(batch, _manifest=manifest):
            packed = assemble_batch_payload(
                batch,
                config,
                day_of_week_source=str(config["day_of_week_source"]),
                weekday_mapping=None,
                all_days=[str(item) for item in _manifest.get("all_days", [])],
            )
            return model(
                packed["history"].to(device),
                **model_kwargs_from_payload(packed, device),
            )

        packed = collect_split_predictions(
            loader,
            predict,
            mean_y=loss_scale(mean_y),
            std_y=loss_scale(std_y),
            device=device,
        )
        metrics = compute_raw_metrics(packed["pred_raw"], packed["y_raw"])
        row = {
            "sampling_seed": sampling_seed,
            "mae": float(metrics["mae"]),
            "rmse": float(metrics["rmse"]),
            "wape": float(metrics["wape"]),
            "mape_nonzero": float(metrics["mape_nonzero"]),
            "sample_count": int(packed["pred_raw"].shape[0]),
        }
        rows.append(row)
        print(
            f"sampling_seed={sampling_seed} mae={row['mae']:.10f} "
            f"rmse={row['rmse']:.10f} wape={row['wape']:.10f}",
            flush=True,
        )
        del dataset, loader, packed
    maes = [row["mae"] for row in rows]
    summary = {
        "model": "staeformer",
        "task": args.run,
        "input_type": config.get("input_type"),
        "target_type": target_type,
        "penetration": "p05",
        "training_seed": 42,
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "selection": "original validation huber; test samplings were not used to select",
        "sampling_seeds": list(SEEDS),
        "per_seed": rows,
        "mean_mae": float(np.mean(maes)),
        "published_seed_42_mae": published_mae,
        "seed_42_abs_diff_vs_published": abs(rows[0]["mae"] - published_mae),
    }
    slug = f"{config.get('input_type')}_to_{target_type}"
    out = TRAINING_ROOT / "sampling_seeds" / f"staeformer_p05_{slug}_sampling_mae.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"mean_mae={summary['mean_mae']:.10f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
