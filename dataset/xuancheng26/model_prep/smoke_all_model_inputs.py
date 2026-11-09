#!/usr/bin/env python3
"""Real-input smoke for the 14 Xuancheng26 models. No official checkpoint is written."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_root import data_root, resolve_xuancheng_dataset  # noqa: E402
from reimplementation.common.data.state_windows import AlignedStateWindowDataset, apply_state_types  # noqa: E402
import generate_r_level_windows as windows  # noqa: E402

TRAINING = data_root() / "xuancheng26/training_data_3s"
GRAPH = TRAINING / "graph"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATASETS = (
    ("fixed_uniform", ("p05", "p10", "p20", "p30", "p40", "p50", "p70")),
    ("variable_30min_p10_p20", ("p10_p20",)),
    ("variable_1h_p10_p20", ("p10_p20",)),
    ("variable_2h_p10_p20", ("p10_p20",)),
    ("variable_day_p10_p20", ("p10_p20",)),
)
FORWARD_CASES = (
    ("fixed_uniform", "p05", 42, "train"),
    ("fixed_uniform", "p70", 43, "validation"),
    ("variable_30min_p10_p20", "p10_p20", 46, "test"),
    ("variable_1h_p10_p20", "p10_p20", 42, "train"),
    ("variable_2h_p10_p20", "p10_p20", 43, "validation"),
    ("variable_day_p10_p20", "p10_p20", 46, "test"),
)
MODELS = (
    "stgcn", "dcrnn", "graph_wavenet", "stsgcn", "stfgnn", "pdformer",
    "stid", "staeformer", "s2s_gcn_lstm", "tlast", "asttn", "stjgcn", "RF", "MSTT",
)


def window_root(dataset_id: str, seed: int) -> Path:
    return TRAINING / "datasets" / dataset_id / "sampling_seeds" / f"seed_{seed}" / "flow" / "window_data"


def one_batch(dataset_id: str, key: str, seed: int, split: str, input_type: str, target_type: str):
    counts = {"train": 5540, "validation": 1108, "test": 1385}
    data = AlignedStateWindowDataset(
        window_root(dataset_id, seed), key, split,
        input_type=input_type, target_type=target_type, expected_count=counts[split],
    )
    item = data[0]
    batched = {}
    for name, value in item.items():
        batched[name] = value.reshape(1) if value.ndim == 0 else value.unsqueeze(0)
    return batched


def check_raw_against_csv(dataset_id: str, key: str, seed: int, split: str) -> str:
    edges = windows.load_edge_ids()
    day = json.loads((window_root(dataset_id, seed) / "split_manifest.json").read_text())[split][0]
    path = window_root(dataset_id, seed) / key / split / day / "inflow.npz"
    with np.load(path) as payload:
        x_raw = payload["x_raw"]
        y_raw = payload["y_raw"]
        start = int(payload["window_start_slot"][0])
        target = int(payload["target_slot"][0])
    if target != start + 11:
        raise RuntimeError(f"{path} target is not the last observed step")
    partial = windows.load_aligned_matrix(
        TRAINING / "datasets" / dataset_id / "sampling_seeds" / f"seed_{seed}" / "flow" / "partial_flow" / key / day / "inflow.csv",
        edges,
    )
    full = windows.load_aligned_matrix(TRAINING / "full_flow" / "r-level" / day / "inflow.csv", edges)
    if not np.array_equal(x_raw[0, :, :, 0], partial[start:start + 12]):
        raise RuntimeError(f"{dataset_id} {key} seed {seed} x_raw is not the partial slice")
    if not np.array_equal(y_raw[0, 0, :, 0], full[target]):
        raise RuntimeError(f"{dataset_id} {key} seed {seed} y_raw is not full[target]")
    return f"{day}:{start}"


def _cfg(path: Path, input_type: str, target_type: str, dataset_id: str) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["input_type"] = input_type
    config["target_type"] = target_type
    config["input_channels"] = 4 if input_type == "all" else 1
    config["output_channels"] = 4 if target_type == "all" else 1
    config["num_nodes"] = 1406
    config["dataset_id"] = dataset_id
    return config


def forward_stid(batch, input_type, target_type, dataset_id):
    from reimplementation.stid.engine import assemble_batch_history, build_model, identity_kwargs
    config = _cfg(TRAINING / "config/stid/r_only_stid.json", input_type, target_type, dataset_id)
    config["input_dim"] = config["input_channels"] + 2
    config["output_dim"] = config["output_channels"]
    day = {"day_of_week_source": config.get("day_of_week_source", "original_sequential_index_mod_7"), "weekday_mapping": None, "all_days": None}
    model = build_model(config).to(DEVICE).eval()
    history = assemble_batch_history(batch, config, **day).to(DEVICE)
    with torch.no_grad():
        return model(history, **identity_kwargs(batch, config, day))


def forward_staeformer(batch, input_type, target_type, dataset_id):
    from reimplementation.staeformer.engine import assemble_batch_payload, build_model, model_kwargs_from_payload
    config = _cfg(TRAINING / "config/staeformer/r_only_staeformer.json", input_type, target_type, dataset_id)
    config["input_dim"] = config["input_channels"] + 2
    config["output_dim"] = config["output_channels"]
    day = {"day_of_week_source": config.get("day_of_week_source", "original_pems_npz_sequential_index_mod_7"), "weekday_mapping": None, "all_days": None}
    model = build_model(config).to(DEVICE).eval()
    payload = assemble_batch_payload(batch, config, **day)
    with torch.no_grad():
        return model(payload["history"].to(DEVICE), **model_kwargs_from_payload(payload, DEVICE))


def forward_tlast(batch, input_type, target_type, dataset_id):
    from reimplementation.tlast.engine import build_model, predict_batch
    from reimplementation.tlast.residual import latest_residual
    config = _cfg(CODE / "reimplementation/tlast/configs/r_only_tlast.json", input_type, target_type, dataset_id)
    config["input_dim"] = config["input_channels"]
    config["output_dim"] = config["output_channels"]
    config["use_input_residual"] = target_type != "inflow"
    norm = json.loads((window_root(dataset_id, 42) / "state_normalization.json").read_text())
    names = ("inflow", "outflow", "accumulation_start", "accumulation_end") if target_type == "all" else (target_type,)
    mean = [norm["target"][name]["mean"] for name in names]
    std = [norm["target"][name]["std"] for name in names]
    if target_type not in {"inflow", "all"}:
        picked = latest_residual(batch, config, mean, std)
        channel = ("inflow", "outflow", "accumulation_start", "accumulation_end").index(target_type)
        if not torch.allclose(picked, batch["x_raw"][:, -1:, :, channel:channel + 1]):
            raise RuntimeError("TLAST residual did not take the target channel")
    model = build_model(config).to(DEVICE).eval()
    moved = {key: value.to(DEVICE) if torch.is_tensor(value) else value for key, value in batch.items()}
    with torch.no_grad():
        return predict_batch(model, moved, config, mean, std)


def forward_stgcn(batch, input_type, target_type, dataset_id):
    from reimplementation.stgcn.engine import build_model
    from reimplementation.stgcn.graph import build_stgcn_chebyshev_kernel
    config = _cfg(TRAINING / "config/stgcn/r_only_stgcn.json", input_type, target_type, dataset_id)
    config["blocks"][0][0] = config["input_channels"]
    weights = np.load(GRAPH / "stgcn_weighted_adjacency.npy")
    kernel = build_stgcn_chebyshev_kernel(weights, int(config["Ks"]))
    model = build_model(config, kernel).to(DEVICE).eval()
    with torch.no_grad():
        return model(batch["x"].to(DEVICE))


def forward_dcrnn(batch, input_type, target_type, dataset_id):
    from reimplementation.dcrnn.engine import build_model
    from reimplementation.dcrnn.graph import load_dcrnn_graph
    config = _cfg(TRAINING / "config/dcrnn/r_only_dcrnn.json", input_type, target_type, dataset_id)
    config["input_dim"] = config["input_channels"]
    config["output_dim"] = config["output_channels"]
    ids = [str(x) for x in np.load(GRAPH / "r_graph.npz", allow_pickle=False)["edge_ids"].tolist()]
    graph = load_dcrnn_graph(GRAPH / "dcrnn_weighted_adjacency.npy", pickle_path=GRAPH / "dcrnn_adj_mx.pkl", expected_nodes=1406, node_ids=ids, filter_type=str(config.get("filter_type", "dual_random_walk")))
    model = build_model(config, graph["supports"]).to(DEVICE).eval()
    with torch.no_grad():
        return model(batch["x"].to(DEVICE), batch["y"].to(DEVICE))


def forward_gwn(batch, input_type, target_type, dataset_id):
    from reimplementation.graph_wavenet.engine import build_model
    from reimplementation.graph_wavenet.graph import load_gwn_graph
    config = _cfg(TRAINING / "config/gwn/r_only_gwn.json", input_type, target_type, dataset_id)
    config["in_dim"] = config["input_channels"]
    config["out_dim"] = config["output_channels"]
    ids = [str(x) for x in np.load(GRAPH / "r_graph.npz", allow_pickle=False)["edge_ids"].tolist()]
    graph = load_gwn_graph(GRAPH / "dcrnn_weighted_adjacency.npy", pickle_path=GRAPH / "dcrnn_adj_mx.pkl", expected_nodes=1406, node_ids=ids, adjtype=str(config.get("adjtype", "doubletransition")))
    model = build_model(config, graph["supports"]).to(DEVICE).eval()
    with torch.no_grad():
        return model(batch["x"].to(DEVICE))


def forward_stsgcn(batch, input_type, target_type, dataset_id):
    from reimplementation.stsgcn.engine import build_model
    from reimplementation.stsgcn.graph import load_stsgcn_graph
    config = _cfg(TRAINING / "config/stsgcn/r_only_stsgcn.json", input_type, target_type, dataset_id)
    ids = [str(x) for x in np.load(GRAPH / "r_graph.npz", allow_pickle=False)["edge_ids"].tolist()]
    graph = load_stsgcn_graph(topology_path=GRAPH / "stsgcn_undirected_topology.npy", r_nodes_path=GRAPH / "r_nodes.csv", node_ids=ids, expected_nodes=1406)
    model = build_model(config, graph["spatial_topology"]).to(DEVICE).eval()
    with torch.no_grad():
        return model(batch["x"].to(DEVICE))


def forward_stfgnn(batch, input_type, target_type, dataset_id):
    from reimplementation.stfgnn.engine import build_model
    from reimplementation.stfgnn.graph import load_stfgnn_graph
    config = _cfg(TRAINING / "config/stfgnn/r_only_stfgnn.json", input_type, target_type, dataset_id)
    state = input_type if input_type != "all" else "inflow"
    temporal = TRAINING / "config/stfgnn" / f"temporal_adjacency_{state}.npy"
    ids = [str(x) for x in np.load(GRAPH / "r_graph.npz", allow_pickle=False)["edge_ids"].tolist()]
    graph = load_stfgnn_graph(topology_path=GRAPH / "stsgcn_undirected_topology.npy", r_nodes_path=GRAPH / "r_nodes.csv", node_ids=ids, temporal_path=temporal)
    model = build_model(config, graph["spatial_topology"], graph["temporal_adjacency"], temporal_graph_is_test_only=False).to(DEVICE).eval()
    with torch.no_grad():
        return model(batch["x"].to(DEVICE))


def forward_pdformer(batch, input_type, target_type, dataset_id):
    if input_type == "all" or target_type == "all":
        raise RuntimeError("unsupported_by_design: PDFormer keeps a one-channel stream")
    from reimplementation.pdformer.engine import assemble_batch_x, build_model
    directory = TRAINING / "config/pdformer"
    relations = {
        "hop_matrix": np.load(directory / "pdformer_shortest_path.npy"),
        "dtw_matrix": np.load(directory / "pdformer_dtw_distance.npy"),
        "laplacian_pe": np.load(directory / "pdformer_laplacian_pe.npy"),
        "pattern_keys": np.load(directory / "pdformer_pattern_keys.npy"),
        "relations_are_test_only": False,
    }
    config = _cfg(directory / "r_only_pdformer.json", input_type, target_type, dataset_id)
    assembled = assemble_batch_x(batch, config, weekday_mapping=None, all_days=None)
    config["input_channels"] = int(assembled.shape[-1])
    config["input_dim"] = int(assembled.shape[-1])
    model = build_model(config, relations).to(DEVICE).eval()
    with torch.no_grad():
        return model(assembled.to(DEVICE))


def forward_s2s(batch, input_type, target_type, dataset_id):
    from reimplementation.s2s_gcn_lstm.engine import build_model
    from reimplementation.s2s_gcn_lstm.graph import load_s2s_graph
    config = _cfg(TRAINING / "config/s2s_gcn_lstm/r_only_s2s_gcn_lstm.json", input_type, target_type, dataset_id)
    graph = load_s2s_graph(
        topology_path=GRAPH / "stsgcn_undirected_topology.npy",
        r_graph_path=GRAPH / "r_graph.npz",
        r_nodes_path=GRAPH / "r_nodes.csv",
        node_mapping_path=window_root(dataset_id, 42) / "node_mapping.csv",
        expected_nodes=1406,
    )
    model = build_model(config, graph["adjacency"]).to(DEVICE).eval()
    with torch.no_grad():
        return model(batch["x"].to(DEVICE))


def forward_asttn(batch, input_type, target_type, dataset_id):
    from reimplementation.asttn.engine import build_model, predict_batch
    config = _cfg(CODE / "reimplementation/asttn/configs/r_only_asttn.json", input_type, target_type, dataset_id)
    embedding = np.load(GRAPH / "asttn/node2vec_r128.npy")
    model = build_model(config, embedding).to(DEVICE).eval()
    moved = {key: value.to(DEVICE) if torch.is_tensor(value) else value for key, value in batch.items()}
    with torch.no_grad():
        return predict_batch(model, moved, config)


def forward_stjgcn(batch, input_type, target_type, dataset_id):
    from reimplementation.stjgcn.engine import build_model, predict_batch
    from reimplementation.stjgcn.graph import load_predefined_graph
    config = _cfg(CODE / "reimplementation/stjgcn/configs/r_only_stjgcn.json", input_type, target_type, dataset_id)
    ids = [str(item) for item in np.load(GRAPH / "r_graph.npz", allow_pickle=False)["edge_ids"].tolist()]
    graph = load_predefined_graph(GRAPH / "stjgcn/predefined_stjg.npz", edge_ids=ids, delta_pdf=float(config.get("delta_pdf", 0.01)), horizon=int(config.get("P", config.get("horizon", 1))))
    model = build_model(config).to(DEVICE).eval()
    moved = {key: value.to(DEVICE) if torch.is_tensor(value) else value for key, value in batch.items()}
    with torch.no_grad():
        return predict_batch(model, moved, config, torch.as_tensor(graph["A_fw"], device=DEVICE), torch.as_tensor(graph["A_bw"], device=DEVICE))


def forward_mstt(batch, input_type, target_type, dataset_id):
    from MSTT.config import default_config
    from MSTT.data.dataset import time_features
    from MSTT.data.graph import load_r_graph
    from MSTT.model.mstt import MSTT, MSTTConfig
    config = default_config()
    channels_in = 4 if input_type == "all" else 1
    channels_out = 4 if target_type == "all" else 1
    model_cfg = MSTTConfig.from_mapping(config, in_channels=channels_in, out_channels=channels_out)
    graph, _info = load_r_graph(GRAPH / "r_graph.npz", r_nodes_path=GRAPH / "r_nodes.csv", window_node_mapping=window_root(dataset_id, 42) / "node_mapping.csv", expected_nodes=1406, direction="forward")
    norm = json.loads((window_root(dataset_id, 42) / "state_normalization.json").read_text())
    names = ("inflow", "outflow", "accumulation_start", "accumulation_end") if target_type == "all" else (target_type,)
    mean = torch.tensor([norm["target"][name]["mean"] for name in names], dtype=torch.float32, device=DEVICE)
    std = torch.tensor([norm["target"][name]["std"] for name in names], dtype=torch.float32, device=DEVICE)
    model = MSTT(model_cfg, graph, target_mean=mean, target_std=std).to(DEVICE).eval()
    tod, dow = time_features(batch, 12)
    with torch.no_grad():
        return model(batch["x"].to(DEVICE), tod.to(DEVICE), dow.to(DEVICE))


def forward_rf(batch, input_type, target_type, dataset_id, key, seed, split):
    from reimplementation.RF.data_loader import XuanchengRFData
    spec = resolve_xuancheng_dataset(dataset_id=dataset_id, training_root=TRAINING)
    data = XuanchengRFData(seed, dataset_id=dataset_id, training_root=TRAINING)
    day = json.loads((window_root(dataset_id, seed) / "split_manifest.json").read_text())[split][0]
    cube, target = data.day_cube(key, day, input_type if input_type != "all" else "inflow", target_type if target_type != "all" else "inflow")
    if not np.isfinite(cube).all() or not np.isfinite(target).all():
        raise RuntimeError("RF cube is not finite")
    if cube.shape[-1] != 7:
        raise RuntimeError(f"RF width {cube.shape[-1]}")
    return cube[:1], target[:1]


FORWARDS = {
    "stgcn": forward_stgcn,
    "dcrnn": forward_dcrnn,
    "graph_wavenet": forward_gwn,
    "stsgcn": forward_stsgcn,
    "stfgnn": forward_stfgnn,
    "pdformer": forward_pdformer,
    "stid": forward_stid,
    "staeformer": forward_staeformer,
    "s2s_gcn_lstm": forward_s2s,
    "tlast": forward_tlast,
    "asttn": forward_asttn,
    "stjgcn": forward_stjgcn,
    "MSTT": forward_mstt,
}


def main() -> int:
    rows = []
    for dataset_id, keys in DATASETS:
        for key in keys:
            for seed in (42, 43, 44, 45, 46):
              for split in (("train", "validation", "test") if seed == 42 else ("validation", "test")):
                try:
                    identity = check_raw_against_csv(dataset_id, key, seed, split)
                    rows.append({"model": "data", "dataset_id": dataset_id, "observation_key": key, "sampling_seed": seed, "split": split, "status": "passed", "sample_identity": identity})
                except Exception as exc:
                    rows.append({"model": "data", "dataset_id": dataset_id, "observation_key": key, "sampling_seed": seed, "split": split, "status": "failed", "error": str(exc)})
    for model, function in FORWARDS.items():
        for dataset_id, key, seed, split in FORWARD_CASES:
            try:
                batch = one_batch(dataset_id, key, seed, split, "inflow", "inflow")
                output = function(batch, "inflow", "inflow", dataset_id)
                tensor = output[0] if isinstance(output, tuple) else output
                if not torch.isfinite(tensor).all():
                    raise RuntimeError("non-finite output")
                rows.append({"model": model, "dataset_id": dataset_id, "observation_key": key, "task": "inflow", "sampling_seed": seed, "split": split, "status": "passed", "output_shape": list(tensor.shape), "device": str(DEVICE)})
            except Exception as exc:
                status = "unsupported_by_design" if "unsupported_by_design" in str(exc) else "failed"
                rows.append({"model": model, "dataset_id": dataset_id, "observation_key": key, "task": "inflow", "sampling_seed": seed, "split": split, "status": status, "error": traceback.format_exc(limit=4)[-500:]})
        for dataset_id, key in (("fixed_uniform", "p10"), ("variable_30min_p10_p20", "p10_p20")):
            for input_type, target_type in (("outflow", "outflow"), ("all", "inflow"), ("all", "all")):
                try:
                    if model == "pdformer" and "all" in (input_type, target_type):
                        apply_state_types({"output_root": "/tmp/pdformer", "dataset_id": dataset_id, "num_nodes": 1406, "explicit_path_keys": ["dataset_id"]}, type("A", (), {"dataset_id": dataset_id, "sampling_root": None, "model_seed": None, "train_sampling_seed": None, "evaluation_sampling_seeds": None, "input_type": input_type, "target_type": target_type, "seed": None, "data_root": None})(), model_name="pdformer")
                    batch = one_batch(dataset_id, key, 42, "train", input_type, target_type)
                    output = function(batch, input_type, target_type, dataset_id)
                    tensor = output[0] if isinstance(output, tuple) else output
                    rows.append({"model": model, "dataset_id": dataset_id, "observation_key": key, "task": f"{input_type}->{target_type}", "sampling_seed": 42, "split": "train", "status": "passed", "output_shape": list(tensor.shape)})
                except Exception as exc:
                    text = str(exc)
                    status = "unsupported_by_design" if "unsupported" in text or "one-channel" in text or "not, because" in text else "failed"
                    rows.append({"model": model, "dataset_id": dataset_id, "observation_key": key, "task": f"{input_type}->{target_type}", "sampling_seed": 42, "split": "train", "status": status, "error": text[:400]})
    from reimplementation.RF.data_loader import XuanchengRFData
    for dataset_id, key, seed, split in FORWARD_CASES:
        try:
            data = XuanchengRFData(seed, dataset_id=dataset_id, training_root=TRAINING)
            day = json.loads((window_root(dataset_id, seed) / "split_manifest.json").read_text())[split][0]
            cube, target = data.day_cube(key, day, "inflow", "inflow")
            if cube.shape[-1] != 7 or not np.isfinite(cube).all():
                raise RuntimeError(f"bad RF cube {cube.shape}")
            rows.append({"model": "RF", "dataset_id": dataset_id, "observation_key": key, "task": "inflow", "sampling_seed": seed, "split": split, "status": "passed", "output_shape": list(cube.shape)})
        except Exception as exc:
            rows.append({"model": "RF", "dataset_id": dataset_id, "observation_key": key, "task": "inflow", "sampling_seed": seed, "split": split, "status": "failed", "error": str(exc)[:400]})
    out = TRAINING / "architecture_audit/model_data_compatibility/smoke_inputs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failed = [row for row in rows if row["status"] == "failed"]
    print(f"rows={len(rows)} failed={len(failed)} report={out}")
    for row in failed[:12]:
        print(row["model"], row.get("dataset_id"), row.get("task"), (row.get("error") or "").strip().splitlines()[-1][:220])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
