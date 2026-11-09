#!/usr/bin/env python3
"""Build RF speed features for every published dataset, key, and sampling seed.

One FCD pass per day updates all 55 observation groups. Fixed rates are nested
only inside the same sampling seed. Variable groups are not nested with them.
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
DATASETS = (
    ("fixed_uniform", ("p05", "p10", "p20", "p30", "p40", "p50", "p70")),
    ("variable_30min_p10_p20", ("p10_p20",)),
    ("variable_1h_p10_p20", ("p10_p20",)),
    ("variable_2h_p10_p20", ("p10_p20",)),
    ("variable_day_p10_p20", ("p10_p20",)),
)


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


def groups_for(training: Path, date: str) -> list[dict]:
    groups = []
    for dataset_id, keys in DATASETS:
        for seed in SEEDS:
            previous: set[int] | None = None
            for key in keys:
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


def complete(path: Path, dataset_id: str, seed: int, key: str, members_sha: str) -> bool:
    if not path.is_file():
        return False
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as payload:
            return (
                str(payload["dataset_id"]) == dataset_id
                and int(payload["sampling_seed"]) == seed
                and str(payload["penetration"]) == key
                and str(payload["vehicle_list_sha256"]) == members_sha
                and str(payload["physical_definition_version"]) == PHYSICAL_VERSION
                and int(payload["complete_day"]) == 1
            )
    except Exception:
        return False


def process_day(task: dict) -> dict:
    date = task["date"]
    try:
        groups = groups_for(Path(task["training_root"]), date)
        pending = [group for group in groups if task["force"] or not complete(Path(group["out"]), group["dataset_id"], group["sampling_seed"], group["key"], group["members_sha256"])]
        if not pending:
            return {"date": date, "ok": True, "action": "skipped", "groups": len(groups)}
        mask = rf.build_mask([group["indices"] for group in pending])
        source = rf.resolve_fcd(date, task["fcd_format"])
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
            )
        rf.log(f"[{date}] rf written={len(pending)}")
        return {"date": date, "ok": True, "action": "written", "groups": len(pending)}
    except Exception:
        return {"date": date, "ok": False, "error": traceback.format_exc(limit=8)}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="RF speeds for published Xuancheng26 datasets")
    parser.add_argument("--training-root", type=Path, default=data_root() / "xuancheng26/training_data_3s")
    parser.add_argument("--fcd-root", type=Path, default=data_root() / "xuancheng26/simulation_data/fcd_data_3s")
    parser.add_argument("--net-file", type=Path, default=CODE / "dataset/xuancheng26/xuancheng_tls_3s.net.xml")
    parser.add_argument("--workers", type=int, default=29)
    parser.add_argument("--dates", nargs="*")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
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
    manifest = json.loads((rf.TRAIN_ROOT / "shared" / "split_manifest.json").read_text(encoding="utf-8"))
    dates = list(args.dates) if args.dates else [day for name in ("train", "validation", "test") for day in manifest[name]]
    fcd_format = rf.benchmark_fcd(dates[0])
    rf.log(f"rf published days={len(dates)} workers={args.workers} fcd_format={fcd_format}")
    tasks = [{"date": date, "training_root": str(rf.TRAIN_ROOT), "fcd_format": fcd_format, "force": args.force} for date in dates]
    failed = []
    workers = max(1, min(args.workers, len(tasks)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
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
