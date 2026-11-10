"""Shared baseline resources and per-dataset RF speeds for one penetration range."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[3]

TASKS = ("I2I", "A2A", "A2I")
PD_TASKS = ("I2I",)
N_NODES = 1406

STATIC = {
    "stgcn": {
        "kind": "static_graph",
        "tasks": TASKS,
        "files": {"weighted_adjacency": "graph/stgcn_weighted_adjacency.npy"},
        "shapes": {"weighted_adjacency": (N_NODES, N_NODES)},
    },
    "dcrnn": {
        "kind": "static_graph",
        "tasks": TASKS,
        "files": {
            "adjacency": "graph/dcrnn_weighted_adjacency.npy",
            "adjacency_pickle": "graph/dcrnn_adj_mx.pkl",
        },
        "shapes": {"adjacency": (N_NODES, N_NODES)},
    },
    "graph_wavenet": {
        "kind": "static_graph",
        "tasks": TASKS,
        "files": {
            "adjacency": "graph/dcrnn_weighted_adjacency.npy",
            "adjacency_pickle": "graph/dcrnn_adj_mx.pkl",
        },
        "shapes": {"adjacency": (N_NODES, N_NODES)},
    },
    "stsgcn": {
        "kind": "static_graph",
        "tasks": TASKS,
        "files": {"spatial_adjacency": "graph/stsgcn_undirected_topology.npy"},
        "shapes": {"spatial_adjacency": (N_NODES, N_NODES)},
    },
    "stfgnn": {
        "kind": "train_full_statistic",
        "tasks": TASKS,
        "files": {
            "spatial_adjacency": "graph/stsgcn_undirected_topology.npy",
            "temporal_adjacency": "config/stfgnn/temporal_adjacency.npy",
        },
        "shapes": {
            "spatial_adjacency": (N_NODES, N_NODES),
            "temporal_adjacency": (N_NODES, N_NODES),
        },
        "note": "Temporal adjacency is the published training-day full-flow graph, shared across datasets and sampling seeds.",
    },
    "pdformer": {
        "kind": "static_topology_and_train_full",
        "tasks": PD_TASKS,
        "unsupported_tasks": ["A2A", "A2I"],
        "files": {
            "shortest_path": "config/pdformer/pdformer_shortest_path.npy",
            "dtw": "config/pdformer/pdformer_dtw_distance.npy",
            "laplacian_pe": "config/pdformer/pdformer_laplacian_pe.npy",
            "pattern_keys": "config/pdformer/pdformer_pattern_keys.npy",
            "undirected_topology": "graph/stsgcn_undirected_topology.npy",
        },
        "shapes": {
            "shortest_path": (N_NODES, N_NODES),
            "dtw": (N_NODES, N_NODES),
            "laplacian_pe": (N_NODES, 8),
            "pattern_keys": (16, 3, 1),
            "undirected_topology": (N_NODES, N_NODES),
        },
        "note": "DTW and pattern keys stay the published training-day full-flow priors. PDFormer accepts one state channel only.",
    },
    "stid": {"kind": "none", "tasks": TASKS, "files": {}, "shapes": {}},
    "staeformer": {"kind": "none", "tasks": TASKS, "files": {}, "shapes": {}},
    "s2s_gcn_lstm": {
        "kind": "static_graph",
        "tasks": TASKS,
        "files": {
            "spatial_adjacency": "graph/stsgcn_undirected_topology.npy",
            "distance_matrix": "graph/stgcn_weighted_adjacency.npy",
        },
        "shapes": {
            "spatial_adjacency": (N_NODES, N_NODES),
            "distance_matrix": (N_NODES, N_NODES),
        },
    },
    "tlast": {"kind": "none", "tasks": TASKS, "files": {}, "shapes": {}},
    "asttn": {
        "kind": "static_graph",
        "tasks": TASKS,
        "files": {"spatial_embedding": "graph/asttn/node2vec_r128.npy"},
        "shapes": {"spatial_embedding": (N_NODES, 128)},
        "note": "Node2vec is built from the road graph. Its training seed is not a vehicle sampling seed.",
    },
    "stjgcn": {
        "kind": "static_graph",
        "tasks": TASKS,
        "files": {"predefined_graph": "graph/stjgcn/predefined_stjg.npz"},
        "shapes": {},
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _config_for(model: str) -> Path:
    disk_name = "gwn" if model == "graph_wavenet" else model
    disk = Path("/data5/tzhang174/TrafficFlow/xuancheng26/training_data_3s/config") / disk_name / f"r_only_{disk_name}.json"
    repo = CODE_ROOT / "reimplementation" / model / "configs" / f"r_only_{model}.json"
    if disk.is_file():
        return disk
    if repo.is_file():
        return repo
    raise FileNotFoundError(model)


def _check_array(path: Path, shape: tuple[int, ...]) -> dict:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            arrays = {name: tuple(int(item) for item in payload[name].shape) for name in payload.files}
        expected = {"A_fw": (12, N_NODES, N_NODES), "A_bw": (12, N_NODES, N_NODES), "distance": (N_NODES, N_NODES)}
        if arrays != expected:
            raise RuntimeError(f"{path} shapes {arrays} != {expected}")
        return {"path": str(path), "shape": arrays, "sha256": _sha256(path), "bytes": path.stat().st_size}
    array = np.load(path, mmap_mode="r")
    got = tuple(int(item) for item in array.shape)
    if got != tuple(shape):
        raise RuntimeError(f"{path} shape {got} != {shape}")
    return {"path": str(path), "shape": list(got), "dtype": str(array.dtype), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _road_attributes(training_root: Path, edges: list[str]) -> dict:
    import csv

    path = training_root / "config" / "RF" / "road_attributes.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        found = [row["edge_id"] for row in csv.DictReader(handle)]
    if found != edges:
        raise RuntimeError("config/RF/road_attributes.csv edge order differs from node_mapping.csv")
    return {"path": str(path), "rows": len(found), "sha256": _sha256(path), "kind": "shared_static"}


def _rf_counts(dataset_root: Path, key: str, seeds: list[int], dates: list[str]) -> tuple[int, list[str]]:
    found = 0
    missing = []
    for seed in seeds:
        for date in dates:
            path = dataset_root / "sampling_seeds" / f"seed_{seed}" / "model_prepared" / "RF" / key / f"{date}.npz"
            if path.is_file() and path.stat().st_size > 0:
                found += 1
            else:
                missing.append(str(path))
    return found, missing


def _commands(dataset_id: str, key: str, results_root: Path) -> list[dict]:
    commands = []
    for task, input_type, target_type in (("I2I", "inflow", "inflow"), ("A2A", "all", "all"), ("A2I", "all", "inflow")):
        commands.append({
            "model": "RF",
            "task": task,
            "argv": [
                "python3", "reimplementation/RF/run_rf.py",
                "--dataset-id", dataset_id,
                "--penetrations", key,
                "--input-type", input_type,
                "--target-type", target_type,
                "--results-root", str(results_root / "RF"),
                "--experiment-id", f"variable_{key}_train42_eval5",
            ],
        })
    for model, spec in STATIC.items():
        script = "reimplementation/RF/run_rf.py" if model == "RF" else f"reimplementation/{model}/train_r_only_{'gwn' if model == 'graph_wavenet' else model}.py"
        for task in spec["tasks"]:
            if task == "I2I":
                input_type, target_type = "inflow", "inflow"
            elif task == "A2A":
                input_type, target_type = "all", "all"
            else:
                input_type, target_type = "all", "inflow"
            if model == "RF":
                argv = [
                    "python3", script,
                    "--dataset-id", dataset_id,
                    "--penetrations", key,
                    "--input-type", input_type,
                    "--target-type", target_type,
                    "--results-root", str(results_root / "RF"),
                    "--experiment-id", f"variable_{key}_train42_eval5",
                ]
            else:
                argv = [
                    "python3", script,
                    "--config", str(_config_for(model)),
                    "--dataset-id", dataset_id,
                    "--rates", key,
                    "--input-type", input_type,
                    "--target-type", target_type,
                    "--output-root", str(results_root / model),
                    "--experiment-id", f"variable_{key}_train42_eval5",
                ]
            commands.append({"model": model, "task": task, "argv": argv})
    return commands


def write_model_manifest(training_root: Path, dataset_id: str) -> dict:
    root = training_root / "datasets" / dataset_id
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    split = json.loads((training_root / "shared" / "split_manifest.json").read_text(encoding="utf-8"))
    dates = [day for name in ("train", "validation", "test") for day in split[name]]
    import csv
    with (training_root / "shared" / "node_mapping.csv").open(encoding="utf-8", newline="") as handle:
        edges = [row["edge_id"] for row in csv.DictReader(handle)]
    key = manifest["observation_keys"][0]
    seeds = [int(seed) for seed in manifest["sampling_seeds"]]
    failures = []
    models = []
    try:
        roads = _road_attributes(training_root, edges)
    except Exception as exc:
        roads = None
        failures.append(f"{dataset_id} RF road attributes: {exc}")
    rf_found, rf_missing = _rf_counts(root, key, seeds, dates)
    expected_rf = len(seeds) * len(dates)
    rf_status = "ready" if rf_found == expected_rf and roads is not None else "failed"
    if rf_found != expected_rf:
        failures.append(f"{dataset_id} RF npz {rf_found} != {expected_rf}")
    models.append({
        "model": "RF",
        "kind": "observed_sampling",
        "tasks": list(TASKS),
        "status": rf_status,
        "generated": {
            "path": f"sampling_seeds/seed_<seed>/model_prepared/RF/{key}/<date>.npz",
            "files": rf_found,
            "expected": expected_rf,
        },
        "shared": roads,
    })
    for model, spec in STATIC.items():
        resources = {}
        status = "ready"
        for name, relative in spec["files"].items():
            path = training_root / relative
            try:
                if path.suffix == ".npz" or name in spec["shapes"]:
                    resources[name] = _check_array(path, spec["shapes"].get(name, ()))
                elif path.suffix == ".pkl":
                    if not path.is_file() or path.stat().st_size <= 0:
                        raise RuntimeError(f"missing {path}")
                    resources[name] = {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
                else:
                    raise RuntimeError(f"no shape check for {path}")
            except Exception as exc:
                status = "failed"
                failures.append(f"{dataset_id} {model} {name}: {exc}")
        if not spec["files"]:
            resources = {"required": "window NPZ, node ids, and the existing time-feature interface"}
        models.append({
            "model": model,
            "kind": spec["kind"],
            "tasks": list(spec["tasks"]),
            "unsupported_tasks": list(spec.get("unsupported_tasks") or []),
            "status": status,
            "reuse": "shared" if spec["files"] else "none",
            "resources": resources,
            "note": spec.get("note"),
        })
    from data_root import xuancheng26_training_results_3s_root

    payload = {
        "dataset_id": dataset_id,
        "observation_key": key,
        "sampling_seeds": seeds,
        "models_ready": not failures and all(item["status"] == "ready" for item in models),
        "models": models,
        "train_commands": _commands(dataset_id, key, xuancheng26_training_results_3s_root()) if not failures else [],
        "failures": failures,
    }
    _write_json(root / "model_preparation_manifest.json", payload)
    return payload


def prepare_models(*, training_root: Path, fcd_root: Path, net_file: Path, dataset_ids: list[str], workers: int) -> dict:
    from prepare_published_rf_features import main as prepare_rf

    code = prepare_rf([
        "--training-root", str(training_root),
        "--fcd-root", str(fcd_root),
        "--net-file", str(net_file),
        "--workers", str(workers),
        "--dataset-ids", *[str(item) for item in dataset_ids],
    ])
    failures = []
    if code:
        failures.append("RF feature preparation returned a failure")
    reports = []
    for dataset_id in dataset_ids:
        report = write_model_manifest(Path(training_root), dataset_id)
        reports.append({"dataset_id": dataset_id, "models_ready": report["models_ready"], "failures": report["failures"]})
        failures.extend(report["failures"])
    return {"failures": failures, "datasets": reports}
