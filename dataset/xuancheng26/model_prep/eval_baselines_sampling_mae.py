#!/usr/bin/env python3
"""Score published p05 inflow-to-inflow checkpoints on five test samplings.

Does not train and does not rewrite the official result directories.
"""

from __future__ import annotations

import argparse
import gc
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
from reimplementation.dcrnn.engine import build_model as build_dcrnn
from reimplementation.dcrnn.validation import validate_dcrnn_graph
from reimplementation.graph_wavenet.engine import build_model as build_gwn
from reimplementation.graph_wavenet.validation import validate_gwn_graph
from reimplementation.s2s_gcn_lstm.engine import build_model as build_s2s
from reimplementation.s2s_gcn_lstm.validation import validate_s2s_graph
from reimplementation.stfgnn.engine import build_model as build_stfgnn
from reimplementation.stfgnn.validation import load_node_ids, validate_stfgnn_graph
from reimplementation.stgcn.engine import build_model as build_stgcn
from reimplementation.stgcn.validation import validate_graph
from reimplementation.stid.engine import assemble_batch_history, build_model as build_stid, identity_kwargs
from reimplementation.stsgcn.engine import build_model as build_stsgcn
from reimplementation.stsgcn.validation import validate_stsgcn_graph
from reimplementation.tlast.engine import build_model as build_tlast, predict_batch

TRAINING_ROOT = Path("/data5/tzhang174/TrafficFlow/xuancheng26/training_data_3s")
RESULTS_ROOT = Path("/data5/tzhang174/TrafficFlow/xuancheng26/training_results_3s")
SEEDS = (42, 43, 44, 45, 46)
EXPECTED_COUNT = 1385
MODELS = (
    "stgcn",
    "dcrnn",
    "graph_wavenet",
    "stsgcn",
    "stfgnn",
    "s2s_gcn_lstm",
    "stid",
    "tlast",
)


def _node_ids(config: dict) -> list[str]:
    return load_node_ids(Path(config["node_mapping"]))


def build_predictor(name: str, config: dict, device: torch.device):
    nodes = _node_ids(config)
    expected = int(config["num_nodes"])
    if name == "stgcn":
        graph = validate_graph(
            Path(config["weighted_adjacency"]),
            int(config["Ks"]),
            expected,
            nodes,
            Path(config["r_nodes"]),
        )
        model = build_stgcn(config, graph["cheb_kernel"]).to(device)

        def predict(batch, _model=model):
            return _model(batch["x"])

    elif name == "dcrnn":
        graph = validate_dcrnn_graph(
            adjacency_path=Path(config["adjacency"]),
            pickle_path=Path(config["adjacency_pickle"]),
            r_nodes_path=Path(config["r_nodes"]),
            node_ids=nodes,
            expected_nodes=expected,
            filter_type=str(config.get("filter_type", "dual_random_walk")),
        )
        model = build_dcrnn(config, graph["supports"]).to(device)

        def predict(batch, _model=model):
            return _model(batch["x"], global_step=0)

    elif name == "graph_wavenet":
        graph = validate_gwn_graph(
            adjacency_path=Path(config["adjacency"]),
            pickle_path=Path(config["adjacency_pickle"]),
            r_nodes_path=Path(config["r_nodes"]),
            node_ids=nodes,
            expected_nodes=expected,
            adjtype=str(config.get("adjtype", "doubletransition")),
        )
        model = build_gwn(config, graph["supports"]).to(device)

        def predict(batch, _model=model):
            return _model(batch["x"])

    elif name == "stsgcn":
        graph = validate_stsgcn_graph(
            topology_path=Path(config["spatial_adjacency"]),
            r_nodes_path=Path(config["r_nodes"]),
            node_ids=nodes,
            expected_nodes=expected,
        )
        model = build_stsgcn(config, graph["spatial_topology"]).to(device)

        def predict(batch, _model=model):
            return _model(batch["x"])

    elif name == "stfgnn":
        graph = validate_stfgnn_graph(
            topology_path=Path(config["spatial_adjacency"]),
            r_nodes_path=Path(config["r_nodes"]),
            node_ids=nodes,
            expected_nodes=expected,
            temporal_path=Path(config["temporal_adjacency"]),
        )
        model = build_stfgnn(
            config,
            graph["spatial_topology"],
            graph["temporal_adjacency"],
            temporal_graph_is_test_only=False,
        ).to(device)

        def predict(batch, _model=model):
            return _model(batch["x"])

    elif name == "s2s_gcn_lstm":
        graph = validate_s2s_graph(config, project_root=PROJECT_ROOT)
        model = build_s2s(config, graph["adjacency"]).to(device)

        def predict(batch, _model=model):
            return _model(batch["x"])

    elif name == "stid":
        model = build_stid(config).to(device)

        def predict(batch, _model=model, _config=config):
            manifest_days = batch.pop("_all_days", None)
            day = {
                "day_of_week_source": str(_config.get("day_of_week_source")),
                "weekday_mapping": None,
                "all_days": manifest_days,
            }
            history = assemble_batch_history(batch, _config, **day).to(device)
            return _model(history, **identity_kwargs(batch, _config, day))

    elif name == "tlast":
        model = build_tlast(config).to(device)

        def predict(batch, _model=model, _config=config, _mean=None, _std=None):
            return predict_batch(_model, batch, _config, _mean, _std)

    else:
        raise RuntimeError(f"unknown model {name}")
    return model, predict


def score_model(name: str, device: torch.device, reference_y: np.ndarray | None) -> tuple[dict, np.ndarray]:
    run_dir = RESULTS_ROOT / name / "in_inflow__out_inflow" / "p05" / "seed_42"
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    published = float(json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))["mae"])
    checkpoint = run_dir / "best_checkpoint.pt"
    model, predict = build_predictor(name, config, device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    rows = []
    n_his = int(config.get("in_steps", config.get("input_len", config.get("n_his", config.get("seq_len", 12)))))
    for sampling_seed in SEEDS:
        data_root = TRAINING_ROOT / "sampling_seeds" / f"seed_{sampling_seed}" / "window_data"
        dataset = open_split_dataset(
            data_root, "p05", "test", config,
            n_his=n_his, num_nodes=int(config["num_nodes"]), expected_count=EXPECTED_COUNT,
        )
        labels = np.array(dataset.y_raw, copy=False)
        if reference_y is None:
            reference_y = np.array(labels, copy=True)
        elif not np.array_equal(labels, reference_y):
            raise RuntimeError(f"{name} seed {sampling_seed} labels differ from the shared full inflow target")
        mean_y, std_y = read_state_target_scaler(data_root, "inflow")
        mean_s, std_s = loss_scale(mean_y), loss_scale(std_y)
        loader = build_dataloader(
            dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0, seed=42,
        )
        manifest = json.loads((data_root / "split_manifest.json").read_text(encoding="utf-8"))
        all_days = [str(item) for item in manifest.get("all_days", [])]

        def call(batch, _predict=predict, _days=all_days, _mean=mean_s, _std=std_s, _name=name):
            if _name == "stid":
                batch = dict(batch)
                batch["_all_days"] = _days
                return _predict(batch)
            if _name == "tlast":
                return _predict(batch, _mean=_mean, _std=_std)
            return _predict(batch)

        packed = collect_split_predictions(loader, call, mean_y=mean_s, std_y=std_s, device=device)
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
        print(f"{name} sampling_seed={sampling_seed} mae={row['mae']:.10f}", flush=True)
        del dataset, loader, packed
    maes = [row["mae"] for row in rows]
    summary = {
        "model": name,
        "task": "in_inflow__out_inflow",
        "penetration": "p05",
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "per_seed": rows,
        "mean_mae": float(np.mean(maes)),
        "published_seed_42_mae": published,
        "seed_42_abs_diff_vs_published": abs(rows[0]["mae"] - published),
    }
    del model, payload
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, reference_y


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_y = None
    results = []
    for name in args.models:
        print(f"=== {name} ===", flush=True)
        summary, reference_y = score_model(name, device, reference_y)
        results.append(summary)
        print(
            f"{name} mean_mae={summary['mean_mae']:.10f} "
            f"seed42_diff={summary['seed_42_abs_diff_vs_published']:.3e}",
            flush=True,
        )
    stae_path = TRAINING_ROOT / "sampling_seeds" / "staeformer_p05_inflow_to_inflow_sampling_mae.json"
    if not stae_path.is_file():
        stae_path = TRAINING_ROOT / "sampling_seeds" / "staeformer_p05_inflow_sampling_mae.json"
    if stae_path.is_file():
        results.append(json.loads(stae_path.read_text(encoding="utf-8")))
    results.sort(key=lambda item: item["mean_mae"])
    out = TRAINING_ROOT / "sampling_seeds" / "baselines_p05_inflow_sampling_mae.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
