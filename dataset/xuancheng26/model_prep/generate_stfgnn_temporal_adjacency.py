#!/usr/bin/env python3
"""Build the Xuancheng26 STFGNN temporal adjacency from training-day full inflow.

DTW compares 24-hour daily profiles (288 five-minute slots), not a concatenated
multi-day series. A road-day enters that profile only when its daily entry
count reaches ``MIN_DAILY_FLOW``. A road participates only when it has at
least ``MIN_VALID_DAYS`` such days. Pairwise cost is the mean absolute
difference over the two roads' common valid days, and only when that overlap
is at least ``MIN_COMMON_VALID_DAYS``.

Only ``split_manifest.json`` training days are read. Validation and test days
are not used for thresholds, eligibility, DTW, or the adjacency.

The saved matrix is the 0/1 sparsity graph, not the raw distance matrix.
Invalid roads stay in the 1406-by-1406 matrix with a self-loop and no temporal
neighbors. ``k`` other roads are selected; the self-loop is added afterwards.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

CODE_ROOT = Path("/home/tzhang174/TrafficFlow")
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from reimplementation.stfgnn.model.dtw import constrained_dtw_distance

TRAINING_ROOT = Path("/data5/tzhang174/TrafficFlow/xuancheng26/training_data")
WINDOW_ROOT = TRAINING_ROOT / "window_data"
FULL_ROOT = TRAINING_ROOT / "full_flow" / "r-level"
R_GRAPH = TRAINING_ROOT / "graph" / "r_graph.npz"
OUTPUT = TRAINING_ROOT / "config" / "stfgnn" / "temporal_adjacency.npy"
BACKUP = OUTPUT.with_name("temporal_adjacency.before_activity_filter.npy")
DISTRIBUTION_JSON = OUTPUT.with_name("temporal_activity_distribution.json")
ROAD_CSV = OUTPUT.with_name("temporal_activity_per_road.csv")
REPORT_JSON = OUTPUT.with_name("temporal_activity_generation_report.json")

STEPS_PER_DAY = 288
BIN_SECONDS = 300
DTW_WINDOW = 12
DTW_ORDER = 1
SPARSITY = 0.01
EXPECTED_NODES = 1406

# Reliability of a 288-bin daily shape, chosen from the training-day entry
# distribution before any model is trained. These are not accuracy tunings.
MIN_DAILY_FLOW = 200
MIN_VALID_DAYS = 10
MIN_COMMON_VALID_DAYS = 10

_STATE: dict[str, object] = {}


class GraphError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_edge_ids() -> list[str]:
    with np.load(R_GRAPH, allow_pickle=False) as payload:
        edge_ids = [str(item) for item in payload["edge_ids"].tolist()]
        num_nodes = int(np.asarray(payload["num_nodes"]).reshape(-1)[0])
    if len(edge_ids) != num_nodes:
        raise GraphError(f"r_graph has {len(edge_ids)} ids and num_nodes={num_nodes}")
    if num_nodes != EXPECTED_NODES:
        raise GraphError(f"node count {num_nodes} != sanity check {EXPECTED_NODES}")
    return edge_ids


def training_days() -> list[str]:
    manifest = json.loads((WINDOW_ROOT / "split_manifest.json").read_text(encoding="utf-8"))
    days = [str(item) for item in manifest["training_days"]]
    validation = [str(item) for item in manifest.get("validation_days", [])]
    test = [str(item) for item in manifest.get("test_days", [])]
    heldout = set(validation) | set(test)
    if set(days) & heldout:
        raise GraphError("training days overlap validation or test")
    if not days:
        raise GraphError("training_days is empty")
    return days


def load_training_flow(edge_ids: list[str], days: list[str]) -> np.ndarray:
    cube = np.empty((len(days), STEPS_PER_DAY, len(edge_ids)), dtype=np.float64)
    for day_index, day_name in enumerate(days):
        path = FULL_ROOT / day_name / "entry_flow.csv"
        if not path.is_file():
            raise GraphError(f"missing training full inflow {path}")
        with path.open(encoding="utf-8") as handle:
            header = handle.readline().rstrip("\n").split(",")
        if header[:2] != ["time_bin", "start_time"]:
            raise GraphError(f"{path} header is not time_bin,start_time")
        columns = header[2:]
        if set(columns) != set(edge_ids):
            raise GraphError(f"{path} edge set does not match r_graph")
        order = [columns.index(edge_id) for edge_id in edge_ids]
        table = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
        if table.shape != (STEPS_PER_DAY, len(edge_ids) + 2):
            raise GraphError(f"{path} shape {table.shape}")
        if not np.array_equal(table[:, 0], np.arange(STEPS_PER_DAY)):
            raise GraphError(f"{path} time_bin is not 0..{STEPS_PER_DAY - 1}")
        if not np.array_equal(table[:, 1], np.arange(STEPS_PER_DAY) * BIN_SECONDS):
            raise GraphError(f"{path} start_time is not {BIN_SECONDS}-second bins")
        cube[day_index] = table[:, 2:][:, order]
    if not np.isfinite(cube).all():
        raise GraphError("training entry flow contains NaN or inf")
    return cube


def daily_total_flow(cube: np.ndarray) -> np.ndarray:
    """``[days, roads]`` total entries in each 24-hour profile."""
    return cube.sum(axis=1)


def activity_masks(
    daily: np.ndarray,
    *,
    min_daily_flow: float,
    min_valid_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    day_valid = daily >= float(min_daily_flow)
    road_valid = day_valid.sum(axis=0) >= int(min_valid_days)
    return day_valid, road_valid


def zscore_valid_days(cube: np.ndarray, day_valid: np.ndarray) -> np.ndarray:
    """Per-day z-score on valid road-days only. Invalid days stay 0 and are masked later."""
    centered = cube - cube.mean(axis=1, keepdims=True)
    std = cube.std(axis=1, keepdims=True)
    std_safe = np.where(std == 0.0, 1.0, std)
    scaled = centered / std_safe
    return np.where(day_valid[:, None, :], scaled, 0.0)


def distribution_payload(daily: np.ndarray) -> dict[str, object]:
    flat = daily.reshape(-1)
    percentiles = {
        "min": float(np.min(flat)),
        "p1": float(np.percentile(flat, 1)),
        "p5": float(np.percentile(flat, 5)),
        "p10": float(np.percentile(flat, 10)),
        "p25": float(np.percentile(flat, 25)),
        "median": float(np.percentile(flat, 50)),
        "p75": float(np.percentile(flat, 75)),
        "p90": float(np.percentile(flat, 90)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "max": float(np.max(flat)),
    }
    bins = (
        ("=0", (flat == 0)),
        ("1-4", (flat >= 1) & (flat <= 4)),
        ("5-9", (flat >= 5) & (flat <= 9)),
        ("10-19", (flat >= 10) & (flat <= 19)),
        ("20-49", (flat >= 20) & (flat <= 49)),
        ("50-99", (flat >= 50) & (flat <= 99)),
        ("100-199", (flat >= 100) & (flat <= 199)),
        (">=200", flat >= 200),
    )
    counts = {
        name: {"count": int(mask.sum()), "fraction": float(mask.mean())}
        for name, mask in bins
    }
    return {
        "n_road_days": int(flat.size),
        "n_days": int(daily.shape[0]),
        "n_roads": int(daily.shape[1]),
        "percentiles": percentiles,
        "bins": counts,
    }


def print_distribution(payload: dict[str, object]) -> None:
    print("daily total flow percentiles", flush=True)
    for key, value in payload["percentiles"].items():
        print(f"  {key}={value:.4f}", flush=True)
    print("daily total flow bins", flush=True)
    for name, item in payload["bins"].items():
        print(f"  {name:8s} count={item['count']:6d} fraction={item['fraction']:.4f}", flush=True)


def select_directed(dist: np.ndarray, road_valid: np.ndarray, k: int) -> np.ndarray:
    """Directed kNN over finite pairs of other valid roads. Self is not a candidate."""
    if k < 0:
        raise GraphError(f"k must be >= 0, got {k}")
    directed = np.zeros(dist.shape, dtype=np.float32)
    valid = np.flatnonzero(np.asarray(road_valid, dtype=bool))
    for i in valid:
        candidate_index: list[int] = []
        candidate_distance: list[float] = []
        for j in valid:
            if int(j) == int(i):
                continue
            value = float(dist[int(i), int(j)])
            if np.isfinite(value):
                candidate_index.append(int(j))
                candidate_distance.append(value)
        if not candidate_index:
            continue
        order = np.argsort(np.asarray(candidate_distance, dtype=np.float64), kind="stable")
        for position in order[: min(int(k), len(order))]:
            directed[int(i), candidate_index[int(position)]] = 1.0
    return directed


def symmetrize(directed: np.ndarray) -> np.ndarray:
    """OR-symmetrize a 0/1 matrix and then set the diagonal to 1."""
    adjacency = np.maximum(directed, directed.T)
    np.fill_diagonal(adjacency, 1.0)
    return np.ascontiguousarray(adjacency, dtype=np.float32)


def _self_check() -> None:
    dist = np.full((4, 4), np.inf)
    dist[0, 1] = dist[1, 0] = 0.2
    dist[0, 2] = dist[2, 0] = np.inf
    dist[0, 3] = dist[3, 0] = 5.0
    valid = np.array([True, True, True, False])
    directed = select_directed(dist, valid, k=14)
    if directed[0, 1] != 1 or directed[0, 2] != 0 or directed[0, 3] != 0 or directed[0, 0] != 0:
        raise GraphError("top-k accepted self, inf, or an invalid road")
    if directed[3].sum() != 0:
        raise GraphError("invalid road selected neighbors")
    ties = np.zeros((3, 3))
    directed_ties = select_directed(ties, np.array([True, True, True]), k=1)
    if directed_ties[2, 0] != 1 or directed_ties[2, 1] != 0:
        raise GraphError("equal distances did not keep the lower index")
    identical = np.zeros((2, 4))
    identical[0, :] = np.arange(4)
    identical[1, :] = np.arange(4)
    cost = np.abs(identical[:, :, None] - identical[:, None, :]).mean(axis=0)
    distance = constrained_dtw_distance(cost, order=DTW_ORDER, window=DTW_WINDOW)
    if distance != 0.0:
        raise GraphError(f"identical profiles produced DTW {distance}")
    if abs(
        constrained_dtw_distance(cost, order=DTW_ORDER, window=DTW_WINDOW)
        - constrained_dtw_distance(cost.T, order=DTW_ORDER, window=DTW_WINDOW)
    ) > 1e-9:
        raise GraphError("Sakoe-Chiba distance changed under transpose")


def _init_worker(z_roads: np.ndarray, day_valid: np.ndarray, road_valid: np.ndarray, min_common: int) -> None:
    _STATE["z"] = z_roads
    _STATE["day_valid"] = day_valid
    _STATE["road_valid"] = road_valid
    _STATE["min_common"] = int(min_common)


def _row_distances(node: int) -> tuple[int, list[tuple[int, float]], int]:
    z_roads = _STATE["z"]
    day_valid = _STATE["day_valid"]
    road_valid = _STATE["road_valid"]
    min_common = int(_STATE["min_common"])
    n_nodes = int(road_valid.shape[0])
    left_days = day_valid[:, node]
    left = z_roads[node]
    pairs: list[tuple[int, float]] = []
    skipped_common = 0
    for other in range(node + 1, n_nodes):
        if not bool(road_valid[other]):
            continue
        common = left_days & day_valid[:, other]
        n_common = int(common.sum())
        if n_common < min_common:
            skipped_common += 1
            continue
        series_i = left[common]
        series_j = z_roads[other][common]
        cost = np.abs(series_i[:, :, None] - series_j[:, None, :]).mean(axis=0)
        distance = constrained_dtw_distance(cost, order=DTW_ORDER, window=DTW_WINDOW)
        if not np.isfinite(distance):
            raise GraphError(f"non-finite DTW for roads {node} and {other}")
        pairs.append((other, float(distance)))
    return node, pairs, skipped_common


def pairwise_distances(
    z_cube: np.ndarray,
    day_valid: np.ndarray,
    road_valid: np.ndarray,
    *,
    min_common: int,
    workers: int,
) -> tuple[np.ndarray, int]:
    """Upper-triangle DTW among valid roads. Invalid pairs stay +inf."""
    n_nodes = int(road_valid.shape[0])
    distances = np.full((n_nodes, n_nodes), np.inf, dtype=np.float64)
    np.fill_diagonal(distances, 0.0)
    z_roads = np.ascontiguousarray(np.transpose(z_cube, (2, 0, 1)))
    sources = [int(node) for node in np.flatnonzero(road_valid) if int(node) < n_nodes - 1]
    skipped_common = 0
    done = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(z_roads, day_valid, road_valid, min_common),
    ) as pool:
        for node, pairs, skipped in pool.map(_row_distances, sources, chunksize=1):
            for other, distance in pairs:
                distances[node, other] = distance
                distances[other, node] = distance
            skipped_common += int(skipped)
            done += 1
            if done % 50 == 0 or done == len(sources):
                print(f"dtw source rows {done}/{len(sources)}", flush=True)
    return distances, skipped_common


def write_road_csv(
    path: Path,
    edge_ids: list[str],
    daily: np.ndarray,
    road_valid: np.ndarray,
    outgoing: np.ndarray,
    incoming: np.ndarray,
    degree: np.ndarray,
    *,
    min_daily_flow: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "road_index",
                "edge_id",
                "total_flow_20d",
                "mean_daily_flow",
                "median_daily_flow",
                "nonzero_days",
                "days_ge_5",
                "days_ge_10",
                "days_ge_20",
                "days_ge_50",
                "days_ge_min_daily_flow",
                "road_valid",
                "outgoing_knn",
                "incoming_knn",
                "degree",
            ]
        )
        for index, edge_id in enumerate(edge_ids):
            column = daily[:, index]
            writer.writerow(
                [
                    index,
                    edge_id,
                    float(column.sum()),
                    float(column.mean()),
                    float(np.median(column)),
                    int((column > 0).sum()),
                    int((column >= 5).sum()),
                    int((column >= 10).sum()),
                    int((column >= 20).sum()),
                    int((column >= 50).sum()),
                    int((column >= float(min_daily_flow)).sum()),
                    int(bool(road_valid[index])),
                    int(outgoing[index]),
                    int(incoming[index]),
                    int(degree[index]),
                ]
            )


def backup_existing(path: Path) -> None:
    if path.is_file() and not BACKUP.is_file():
        shutil.copy2(path, BACKUP)
        print(f"backed up {path} -> {BACKUP}", flush=True)
    elif BACKUP.is_file():
        print(f"backup already exists: {BACKUP}", flush=True)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npy")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Xuancheng26 STFGNN temporal adjacency.")
    parser.add_argument("--min-daily-flow", type=float, default=MIN_DAILY_FLOW)
    parser.add_argument("--min-valid-days", type=int, default=MIN_VALID_DAYS)
    parser.add_argument("--min-common-valid-days", type=int, default=MIN_COMMON_VALID_DAYS)
    parser.add_argument("--sparsity", type=float, default=SPARSITY)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    _self_check()
    args = parse_args()
    started = time.perf_counter()
    started_at = utc_now()
    print(f"start_time={started_at}", flush=True)
    backup_existing(OUTPUT)
    edge_ids = load_edge_ids()
    days = training_days()
    manifest = json.loads((WINDOW_ROOT / "split_manifest.json").read_text(encoding="utf-8"))
    print(
        f"training_days={len(days)} validation_days_used=0 test_days_used=0 "
        f"validation_days_in_manifest={len(manifest.get('validation_days', []))} "
        f"test_days_in_manifest={len(manifest.get('test_days', []))}",
        flush=True,
    )
    cube = load_training_flow(edge_ids, days)
    daily = daily_total_flow(cube)
    distribution = distribution_payload(daily)
    print_distribution(distribution)
    DISTRIBUTION_JSON.parent.mkdir(parents=True, exist_ok=True)
    DISTRIBUTION_JSON.write_text(json.dumps(distribution, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {DISTRIBUTION_JSON}", flush=True)
    day_valid, road_valid = activity_masks(
        daily,
        min_daily_flow=float(args.min_daily_flow),
        min_valid_days=int(args.min_valid_days),
    )
    n_nodes = len(edge_ids)
    n_valid = int(road_valid.sum())
    n_invalid = n_nodes - n_valid
    all_zero = daily.sum(axis=0) == 0
    print(
        f"MIN_DAILY_FLOW={args.min_daily_flow} MIN_VALID_DAYS={args.min_valid_days} "
        f"MIN_COMMON_VALID_DAYS={args.min_common_valid_days}",
        flush=True,
    )
    print(
        f"valid_roads={n_valid} invalid_roads={n_invalid} all_zero_roads={int(all_zero.sum())} "
        f"all_zero_marked_invalid={int((all_zero & ~road_valid).sum())}",
        flush=True,
    )
    if int((all_zero & road_valid).sum()) != 0:
        raise GraphError("an all-zero road was marked temporal-valid")
    z_cube = zscore_valid_days(cube, day_valid)
    total_pairs = n_nodes * (n_nodes - 1) // 2
    valid_pairs = n_valid * (n_valid - 1) // 2
    skipped_invalid = total_pairs - valid_pairs
    workers = int(args.workers)
    if workers <= 0:
        affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        workers = max(1, min(affinity, 32))
    k = int(n_nodes * float(args.sparsity))
    print(
        f"workers={workers} k={k} total_pairs={total_pairs} "
        f"skipped_invalid_road_pairs={skipped_invalid} valid_candidate_pairs={valid_pairs}",
        flush=True,
    )
    distances, skipped_common = pairwise_distances(
        z_cube,
        day_valid,
        road_valid,
        min_common=int(args.min_common_valid_days),
        workers=workers,
    )
    upper = distances[np.triu_indices(n_nodes, k=1)]
    computed = int(np.isfinite(upper).sum())
    if computed != valid_pairs - skipped_common:
        raise GraphError(
            f"DTW pair count {computed} != valid pairs {valid_pairs} - skipped common {skipped_common}"
        )
    if not np.isfinite(distances[np.isfinite(distances)]).all():
        raise GraphError("distance matrix has NaN")
    directed = select_directed(distances, road_valid, k)
    adjacency = symmetrize(directed)
    if adjacency.shape != (n_nodes, n_nodes) or adjacency.dtype != np.float32:
        raise GraphError(f"adjacency {adjacency.dtype} {adjacency.shape}")
    if not np.array_equal(adjacency, adjacency.T):
        raise GraphError("temporal adjacency is not symmetric")
    if not np.array_equal(np.diag(adjacency), np.ones(n_nodes, dtype=np.float32)):
        raise GraphError("temporal adjacency diagonal is not 1")
    unique = [float(item) for item in np.unique(adjacency)]
    if unique != [0.0, 1.0]:
        raise GraphError(f"temporal adjacency values {unique}")
    degree = adjacency.sum(axis=1) - 1.0
    if np.any(degree[~road_valid] != 0):
        raise GraphError("a temporal-invalid road has an off-diagonal neighbor")
    outgoing = directed.sum(axis=1)
    incoming = directed.sum(axis=0)
    chosen = directed > 0
    if np.any(~np.isfinite(distances[chosen])):
        raise GraphError("a non-finite distance was selected")
    if np.any(directed[np.arange(n_nodes), np.arange(n_nodes)] != 0):
        raise GraphError("self occupied a directed top-k slot")
    atomic_save_npy(OUTPUT, adjacency)
    write_road_csv(
        ROAD_CSV,
        edge_ids,
        daily,
        road_valid,
        outgoing,
        incoming,
        degree,
        min_daily_flow=float(args.min_daily_flow),
    )
    ended_at = utc_now()
    elapsed = time.perf_counter() - started
    report = {
        "start_time": started_at,
        "end_time": ended_at,
        "runtime_sec": elapsed,
        "training_days": days,
        "validation_days_used": 0,
        "test_days_used": 0,
        "min_daily_flow": float(args.min_daily_flow),
        "min_valid_days": int(args.min_valid_days),
        "min_common_valid_days": int(args.min_common_valid_days),
        "k": k,
        "sparsity": float(args.sparsity),
        "dtw_order": DTW_ORDER,
        "dtw_window": DTW_WINDOW,
        "valid_roads": n_valid,
        "invalid_roads": n_invalid,
        "all_zero_roads": int(all_zero.sum()),
        "total_possible_pairs": total_pairs,
        "pairs_skipped_invalid_road": skipped_invalid,
        "pairs_skipped_insufficient_common_days": skipped_common,
        "dtw_pairs_computed": computed,
        "degree_min": float(degree.min()),
        "degree_mean": float(degree.mean()),
        "degree_median": float(np.median(degree)),
        "degree_p90": float(np.quantile(degree, 0.90)),
        "degree_p95": float(np.quantile(degree, 0.95)),
        "degree_p99": float(np.quantile(degree, 0.99)),
        "degree_max": float(degree.max()),
        "distribution": distribution,
        "output": OUTPUT.as_posix(),
        "backup": BACKUP.as_posix(),
        "road_csv": ROAD_CSV.as_posix(),
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"end_time={ended_at} runtime_sec={elapsed:.1f} "
        f"dtw_pairs={computed} skipped_invalid={skipped_invalid} skipped_common={skipped_common}",
        flush=True,
    )
    print(
        "degree "
        f"min={degree.min():.0f} mean={degree.mean():.4f} median={np.median(degree):.0f} "
        f"p90={np.quantile(degree, 0.90):.4f} p95={np.quantile(degree, 0.95):.4f} "
        f"p99={np.quantile(degree, 0.99):.4f} max={degree.max():.0f}",
        flush=True,
    )
    print(f"wrote {OUTPUT}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GraphError as exc:
        print(f"error: {exc}")
        raise SystemExit(1)
