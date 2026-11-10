#!/usr/bin/env python3
"""Build RF speed features for explicitly selected datasets.

One FCD pass per day updates that day's selected observation groups. Fixed
rates are nested only inside the same sampling seed. Variable groups are not
nested with them.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

CODE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare_rf_features as rf  # noqa: E402
from data_root import data_root  # noqa: E402
from independent_states import PHYSICAL_VERSION  # noqa: E402

SEEDS = (42, 43, 44, 45, 46)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_indices(path: Path) -> tuple[list[int], dict[int, str]]:
    indices = []
    owners: dict[int, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        import csv
        rows = csv.reader(handle)
        header = next(rows)
        if header != ["vehicle_id"]:
            raise rf.FeatureError(f"bad vehicle header {path}")
        for row in rows:
            if not row:
                continue
            vehicle_id = row[0]
            index = rf.parse_vehicle_index(vehicle_id)
            if index is None:
                raise rf.FeatureError(f"unparseable id {vehicle_id} in {path}")
            previous = owners.get(index)
            if previous not in (None, vehicle_id):
                raise rf.FeatureError(f"vehicle suffix collision {index}: {previous} vs {vehicle_id}")
            owners[index] = vehicle_id
            indices.append(index)
    if len(indices) != len(set(indices)):
        raise rf.FeatureError(f"duplicate ids in {path}")
    return indices, owners


def specs_from_ids(training: Path, dataset_ids: list[str]) -> list[dict]:
    specs = []
    for dataset_id in dataset_ids:
        manifest_path = training / "datasets" / dataset_id / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_id") != dataset_id:
            raise rf.FeatureError(f"{manifest_path} dataset_id is {manifest.get('dataset_id')!r}")
        keys = tuple(str(item) for item in manifest.get("observation_keys") or ())
        seeds = tuple(int(item) for item in manifest.get("sampling_seeds") or ())
        if not keys or not seeds:
            raise rf.FeatureError(f"{dataset_id} manifest has no observation keys or sampling seeds")
        specs.append({"dataset_id": dataset_id, "keys": keys, "seeds": seeds})
    return specs


def groups_for(training: Path, date: str, specs: list[dict]) -> list[dict]:
    groups = []
    for spec in specs:
        dataset_id = spec["dataset_id"]
        for seed in spec["seeds"]:
            previous: set[int] | None = None
            for key in spec["keys"]:
                members = training / "datasets" / dataset_id / "sampling_seeds" / f"seed_{seed}" / "sampled_vehicle" / key / f"{date}.csv"
                inflow = training / "datasets" / dataset_id / "sampling_seeds" / f"seed_{seed}" / "flow" / "partial_flow" / key / date / "inflow.csv"
                out = training / "datasets" / dataset_id / "sampling_seeds" / f"seed_{seed}" / "model_prepared" / "RF" / key / f"{date}.npz"
                indices, _owners = load_indices(members)
                if previous is not None and not set(indices) >= previous:
                    raise rf.FeatureError(f"{dataset_id} seed {seed} {date} fixed rates are not nested")
                if dataset_id == "fixed_uniform":
                    previous = set(indices)
                groups.append({
                    "dataset_id": dataset_id,
                    "sampling_seed": seed,
                    "key": key,
                    "members": str(members),
                    "members_sha256": sha256_file(members),
                    "inflow": str(inflow),
                    "out": str(out),
                    "indices": indices,
                })
    return groups


def complete(path: Path, group: dict, *, edges: list[str], source_fcd: Path, node_sha: str) -> bool:
    if not path.is_file():
        return False
    try:
        import numpy as np
        stat = source_fcd.stat()
        with np.load(path, allow_pickle=False) as payload:
            stored_edges = [str(item) for item in payload["edge_ids"].tolist()]
            return (
                str(payload["dataset_id"]) == group["dataset_id"]
                and int(payload["sampling_seed"]) == int(group["sampling_seed"])
                and str(payload["penetration"]) == group["key"]
                and str(payload["vehicle_list_sha256"]) == group["members_sha256"]
                and str(payload["physical_definition_version"]) == PHYSICAL_VERSION
                and int(payload["complete_day"]) == 1
                and stored_edges == edges
                and str(payload["source_fcd"]) == str(source_fcd)
                and int(payload["fcd_size"]) == int(stat.st_size)
                and int(payload["fcd_mtime_ns"]) == int(stat.st_mtime_ns)
                and str(payload["node_order_sha256"]) == node_sha
            )
    except Exception:
        return False


def _init_worker(payload: dict) -> None:
    rf.FCD_ROOT = Path(payload["fcd_root"])
    rf.TRAIN_ROOT = Path(payload["training_root"])
    rf.NET_FILE = Path(payload["net_file"])
    rf.LANE_TO_INDEX = payload["lane_to_index"]
    rf.INTERNAL_LANES = payload["internal_lanes"]
    rf.BASE_MS = payload["base_ms"]
    rf.BASE_KMH = payload["base_kmh"]
    rf.EDGES = payload["edges"]


def process_day(task: dict) -> dict:
    date = task["date"]
    try:
        groups = list(task["groups"])
        source_expected = Path(task["source_fcd"])
        pending = [
            group for group in groups
            if task["force"] or not complete(
                Path(group["out"]), group, edges=list(task["edges"]),
                source_fcd=source_expected, node_sha=task["node_order_sha256"],
            )
        ]
        if not pending:
            return {"date": date, "ok": True, "action": "skipped", "groups": len(groups)}
        mask = rf.build_mask([group["indices"] for group in pending])
        source = Path(task["source_fcd"])
        if not source.is_file():
            raise rf.FeatureError(f"missing FCD {source}")
        rf.log(f"[{date}] rf groups={len(pending)} fcd={source.name}")
        sums, counts, stats = rf.parse_fcd_stream(
            rf.iter_file_blocks(source),
            mask=mask,
            lane_to_index=rf.LANE_TO_INDEX,
            internal_lanes=rf.INTERNAL_LANES,
            base_ms=rf.BASE_MS,
            n_rates=len(pending),
            n_edges=len(rf.EDGES),
            until_time=None,
            progress_label=date,
        )
        if int(stats["unknown_lane"]) > 0:
            raise rf.FeatureError(f"unknown lanes {stats['unknown_examples']}")
        weekday = rf.weekday_index(date)
        for index, group in enumerate(pending):
            average, normalized, _capped = rf.finalize_speeds(sums[index], counts[index], rf.BASE_KMH)
            count32 = counts[index].astype("int32", copy=False)
            inflow = rf.load_inflow(Path(group["inflow"]), rf.EDGES)
            bad, examples = rf.flow_without_speed(count32, inflow)
            if bad:
                raise rf.FeatureError(f"{date} {group['dataset_id']} {group['key']} seed {group['sampling_seed']} inflow without speed {bad} {examples}")
            rf.write_rate_npz(
                Path(group["out"]),
                average=average,
                normalized=normalized,
                count=count32,
                edges=rf.EDGES,
                base_kmh=rf.BASE_KMH,
                day_name=date,
                rate=group["key"],
                weekday=weekday,
                source_fcd=source,
                complete=True,
                parse_stats=stats,
                rate_index=index,
                dataset_id=group["dataset_id"],
                sampling_seed=group["sampling_seed"],
                vehicle_list_sha256=group["members_sha256"],
                physical_definition_version=PHYSICAL_VERSION,
                node_order_sha256=task["node_order_sha256"],
            )
        rf.log(f"[{date}] rf written={len(pending)}")
        return {"date": date, "ok": True, "action": "written", "groups": len(pending)}
    except Exception:
        return {"date": date, "ok": False, "error": traceback.format_exc(limit=8)}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import multiprocessing as mp
    parser = argparse.ArgumentParser(description="RF speeds for selected Xuancheng26 datasets")
    parser.add_argument("--training-root", type=Path, default=data_root() / "xuancheng26/training_data_3s")
    parser.add_argument("--fcd-root", type=Path, default=data_root() / "xuancheng26/simulation_data/fcd_data_3s")
    parser.add_argument("--net-file", type=Path, default=CODE / "dataset/xuancheng26/xuancheng_tls_3s.net.xml")
    parser.add_argument("--dataset-ids", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=29)
    parser.add_argument("--dates", nargs="*")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    rf.FCD_ROOT = args.fcd_root.expanduser().resolve()
    rf.TRAIN_ROOT = args.training_root.expanduser().resolve()
    rf.NET_FILE = args.net_file.expanduser().resolve()
    rf.NODES_PATH = rf.TRAIN_ROOT / "graph" / "r_nodes.csv"
    rf.GRAPH_PATH = rf.TRAIN_ROOT / "graph" / "r_graph.npz"
    edges, node_lanes, speed_text = rf.load_canonical_edges()
    lane_to_index, internal_lanes, speed_ms, _lane_count = rf.load_network_attributes(edges, node_lanes, speed_text)
    rf.LANE_TO_INDEX = lane_to_index
    rf.INTERNAL_LANES = internal_lanes
    rf.BASE_MS = speed_ms
    rf.BASE_KMH = speed_ms * rf.MS_TO_KMH
    rf.EDGES = edges
    import csv

    mapped = []
    with (rf.TRAIN_ROOT / "shared" / "node_mapping.csv").open(encoding="utf-8", newline="") as handle:
        mapped = [row["edge_id"] for row in csv.DictReader(handle)]
    if mapped != edges:
        raise rf.FeatureError("r_nodes.csv order differs from shared/node_mapping.csv")
    node_sha = sha256_file(rf.TRAIN_ROOT / "shared" / "node_mapping.csv")
    specs = specs_from_ids(rf.TRAIN_ROOT, [str(item) for item in args.dataset_ids])
    manifest = json.loads((rf.TRAIN_ROOT / "shared" / "split_manifest.json").read_text(encoding="utf-8"))
    dates = list(args.dates) if args.dates else [day for name in ("train", "validation", "test") for day in manifest[name]]
    fcd_format = rf.benchmark_fcd(dates[0])
    tasks = []
    for date in dates:
        source = rf.resolve_fcd(date, fcd_format)
        tasks.append({
            "date": date,
            "training_root": str(rf.TRAIN_ROOT),
            "fcd_root": str(rf.FCD_ROOT),
            "source_fcd": str(source),
            "force": args.force,
            "edges": edges,
            "node_order_sha256": node_sha,
            "groups": groups_for(rf.TRAIN_ROOT, date, specs),
        })
    rf.log(f"rf selected datasets={args.dataset_ids} days={len(dates)} workers={args.workers} fcd_format={fcd_format}")
    failed = []
    workers = max(1, min(args.workers, len(tasks)))
    payload = {
        "fcd_root": str(rf.FCD_ROOT),
        "training_root": str(rf.TRAIN_ROOT),
        "net_file": str(rf.NET_FILE),
        "lane_to_index": lane_to_index,
        "internal_lanes": internal_lanes,
        "base_ms": speed_ms,
        "base_kmh": rf.BASE_KMH,
        "edges": edges,
    }
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork"), initializer=_init_worker, initargs=(payload,)) as pool:
        futures = {pool.submit(process_day, task): task["date"] for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            if not row.get("ok"):
                failed.append(row["date"])
                rf.log(f"[{row['date']}] FAIL {(row.get('error') or '')[-400:]}")
            else:
                rf.log(f"[{row['date']}] {row.get('action')} groups={row.get('groups')}")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
