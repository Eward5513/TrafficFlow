"""
Count subgraph edge vehicle entries by time window and plot one line chart per edge.

Reads a day's extracted trajectories.csv. An event is counted when a vehicle's
edge_id changes (or on its first row) and the new edge is in subgraph.txt.
Stay-on-edge seconds are not counted. Subgraph membership is exact string
match against subgraph.txt, not is_in_subgraph.

The aggregation window defaults to 5 minutes and can be changed.

Usage:

    python3 analysis/simulation/count_edge_flow.py \\
        --trajectories .../day_01/trajectories.csv \\
        --subgraph analysis/simulation/data/subgraph.txt \\
        --output-dir .../day_01 \\
        --day-label day_01
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

SECONDS_PER_DAY = 24 * 3600
DEFAULT_WINDOW_MINUTES = 5
INDEX_CSV_PREFIX = "edge_plot_index"
FLOW_CSV_PREFIX = "edge_flow"
PLOT_DIR_PREFIX = "edge_flow_plots"
PROGRESS_EVERY = 500_000
MISMATCH_EXAMPLES = 8


@dataclass
class SubgraphSpec:
    raw_line_count: int
    nonempty_line_count: int
    unique_ids: list[str]
    unique_set: set[str]
    duplicate_ids: list[str]


@dataclass
class EntryStats:
    entry_count: int = 0
    rows: int = 0
    timestamp_min: float | None = None
    timestamp_max: float | None = None
    subgraph_edges_with_entry: set[str] = field(default_factory=set)
    flag_mismatches: int = 0
    flag_mismatch_examples: list[dict[str, str]] = field(default_factory=list)
    skipped_outside_day: int = 0


class FlowError(Exception):
    pass


def window_label(window_seconds: int) -> str:
    if window_seconds % 60 == 0:
        return f"{window_seconds // 60}min"
    return f"{window_seconds}s"


def flow_csv_name(window_seconds: int) -> str:
    return f"{FLOW_CSV_PREFIX}_{window_label(window_seconds)}.csv"


def index_csv_name(window_seconds: int) -> str:
    return f"{INDEX_CSV_PREFIX}_{window_label(window_seconds)}.csv"


def plot_dir_name(window_seconds: int) -> str:
    return f"{PLOT_DIR_PREFIX}_{window_label(window_seconds)}"


def resolve_window_seconds(window_minutes: int, window_seconds: int | None) -> int:
    seconds = window_seconds if window_seconds is not None else window_minutes * 60
    if seconds <= 0 or SECONDS_PER_DAY % seconds != 0:
        raise SystemExit("window must be a positive divisor of 86400 seconds")
    return seconds


def load_subgraph(path: Path) -> SubgraphSpec:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    nonempty = [line.strip() for line in raw_lines if line.strip()]
    unique: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for edge_id in nonempty:
        if edge_id in seen:
            duplicates.append(edge_id)
            continue
        seen.add(edge_id)
        unique.append(edge_id)
    if not unique:
        raise FlowError(f"no edge IDs in {path}")
    return SubgraphSpec(
        raw_line_count=len(raw_lines),
        nonempty_line_count=len(nonempty),
        unique_ids=unique,
        unique_set=seen,
        duplicate_ids=duplicates,
    )


def flag_is_true(raw: str) -> bool:
    return raw.strip().lower() in {"true", "1", "yes"}


def window_index_for(timestamp: float, window_seconds: int) -> int | None:
    if timestamp < 0:
        return None
    index = math.floor(timestamp / window_seconds)
    n_windows = SECONDS_PER_DAY // window_seconds
    if index < 0 or index >= n_windows:
        return None
    return index


def time_label(window_index: int, window_seconds: int) -> str:
    seconds = window_index * window_seconds
    hours, rem = divmod(seconds, 3600)
    minutes = rem // 60
    return f"{hours:02d}:{minutes:02d}"


def safe_plot_filename(edge_id: str, used: set[str]) -> str:
    chars: list[str] = []
    for ch in edge_id:
        if ch.isalnum() or ch in ".-":
            chars.append(ch)
        else:
            chars.append("_")
    body = "".join(chars).strip("._") or "edge"
    digest = hashlib.sha1(edge_id.encode("utf-8")).hexdigest()
    length = 8
    while True:
        name = f"{body}_{digest[:length]}.png"
        if name not in used:
            used.add(name)
            return name
        length += 2
        if length > len(digest):
            raise FlowError(f"could not build unique plot name for {edge_id!r}")


def count_entries(
    trajectories: Path,
    subgraph: set[str],
    *,
    window_seconds: int,
    n_windows: int,
) -> tuple[dict[str, list[int]], EntryStats, dict[str, str]]:
    counts = {edge_id: [0] * n_windows for edge_id in subgraph}
    last_edge: dict[str, str] = {}
    stats = EntryStats()
    with trajectories.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = ("timestamp", "vehicle_id", "edge_id", "is_in_subgraph")
        if reader.fieldnames is None:
            raise FlowError(f"{trajectories} has no header")
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise FlowError(f"{trajectories} missing fields {missing}: {reader.fieldnames}")
        for row in reader:
            stats.rows += 1
            if stats.rows % PROGRESS_EVERY == 0:
                print(
                    f"  scanned {stats.rows:,} rows entries={stats.entry_count:,}",
                    flush=True,
                )
            timestamp = float(row["timestamp"])
            if stats.timestamp_min is None or timestamp < stats.timestamp_min:
                stats.timestamp_min = timestamp
            if stats.timestamp_max is None or timestamp > stats.timestamp_max:
                stats.timestamp_max = timestamp
            vehicle_id = row["vehicle_id"]
            edge_id = row["edge_id"]
            in_subgraph = edge_id in subgraph
            flagged = flag_is_true(row["is_in_subgraph"])
            if in_subgraph != flagged:
                stats.flag_mismatches += 1
                if len(stats.flag_mismatch_examples) < MISMATCH_EXAMPLES:
                    stats.flag_mismatch_examples.append(
                        {
                            "timestamp": row["timestamp"],
                            "vehicle_id": vehicle_id,
                            "edge_id": edge_id,
                            "is_in_subgraph": row["is_in_subgraph"],
                            "expected": "true" if in_subgraph else "false",
                        }
                    )
            previous = last_edge.get(vehicle_id)
            entered = previous is None or previous != edge_id
            last_edge[vehicle_id] = edge_id
            if not entered or not in_subgraph:
                continue
            index = window_index_for(timestamp, window_seconds)
            if index is None:
                stats.skipped_outside_day += 1
                continue
            counts[edge_id][index] += 1
            stats.entry_count += 1
            stats.subgraph_edges_with_entry.add(edge_id)
    return counts, stats, last_edge


def write_flow_csv(
    path: Path,
    unique_ids: list[str],
    counts: dict[str, list[int]],
    *,
    window_seconds: int,
    n_windows: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["window_index", "window_start", "time_label", "edge_id", "vehicle_count"]
        )
        for edge_id in unique_ids:
            series = counts[edge_id]
            for index, value in enumerate(series):
                writer.writerow(
                    [
                        index,
                        index * window_seconds,
                        time_label(index, window_seconds),
                        edge_id,
                        value,
                    ]
                )


def write_index_csv(
    path: Path,
    unique_ids: list[str],
    counts: dict[str, list[int]],
    filenames: dict[str, str],
    *,
    window_seconds: int,
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "edge_id",
                "plot_filename",
                "total_vehicle_entries",
                "peak_vehicle_entries",
                "peak_time",
            ]
        )
        for edge_id in unique_ids:
            series = counts[edge_id]
            total = sum(series)
            peak = max(series) if series else 0
            peak_index = series.index(peak) if series else 0
            writer.writerow(
                [
                    edge_id,
                    filenames[edge_id],
                    total,
                    peak,
                    time_label(peak_index, window_seconds),
                ]
            )


def plot_edges(
    plot_dir: Path,
    unique_ids: list[str],
    counts: dict[str, list[int]],
    filenames: dict[str, str],
    *,
    day_label: str,
    window_seconds: int,
    n_windows: int,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if plot_dir.exists():
        shutil.rmtree(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    hours = [index * window_seconds / 3600.0 for index in range(n_windows)]
    xticks = list(range(0, 25, 2))
    xticklabels = [f"{hour:02d}:00" for hour in xticks]
    ylabel = f"Vehicle Entries per {window_seconds // 60} min" if window_seconds % 60 == 0 else (
        f"Vehicle Entries per {window_seconds} s"
    )
    for edge_id in unique_ids:
        series = counts[edge_id]
        fig, ax = plt.subplots(figsize=(10.5, 4.2))
        ax.plot(hours, series, color="#1f4e79", linewidth=1.2, marker="none", solid_capstyle="butt")
        ax.set_xlim(0, 24)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ymax = max(series) if series else 0
        ax.set_ylim(0, 1.0 if ymax <= 0 else ymax * 1.08)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Time of Day")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{day_label}  {edge_id}")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        fig.tight_layout()
        fig.savefig(plot_dir / filenames[edge_id], dpi=dpi)
        plt.close(fig)


def read_flow_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_outputs(
    *,
    subgraph: SubgraphSpec,
    stats: EntryStats,
    counts: dict[str, list[int]],
    flow_path: Path,
    index_path: Path,
    plot_dir: Path,
    filenames: dict[str, str],
    n_windows: int,
    window_seconds: int,
    trajectories: Path,
    traj_stat_before: tuple[int, int],
    seed: int,
) -> list[str]:
    failures: list[str] = []
    unique_set = subgraph.unique_set
    n_edges = len(subgraph.unique_ids)

    if not flow_path.is_file():
        return [f"missing {flow_path}"]
    rows = read_flow_csv(flow_path)
    expected_rows = n_edges * n_windows
    if len(rows) != expected_rows:
        failures.append(f"flow csv rows {len(rows)} != {n_edges} * {n_windows}")

    flow_edges = {row["edge_id"] for row in rows}
    if flow_edges != unique_set:
        failures.append("flow csv edge set != subgraph unique IDs")
    extra = sorted(flow_edges - unique_set)
    if extra:
        failures.append(f"flow csv has non-subgraph edges: {extra[:8]}")

    per_edge_windows: dict[str, list[int]] = {edge_id: [] for edge_id in subgraph.unique_ids}
    total = 0
    non_int = 0
    negative = 0
    for row in rows:
        try:
            value = int(row["vehicle_count"])
        except ValueError:
            non_int += 1
            continue
        if str(value) != row["vehicle_count"].strip():
            non_int += 1
        if value < 0:
            negative += 1
        total += value
        edge_id = row["edge_id"]
        if edge_id in per_edge_windows:
            per_edge_windows[edge_id].append(int(row["window_index"]))
    if non_int:
        failures.append(f"{non_int} vehicle_count values are not integers")
    if negative:
        failures.append(f"{negative} negative vehicle_count values")
    if total != stats.entry_count:
        failures.append(f"flow csv sum {total} != entry events {stats.entry_count}")

    for edge_id in subgraph.unique_ids:
        windows = per_edge_windows[edge_id]
        if windows != list(range(n_windows)):
            failures.append(f"{edge_id} does not have complete 0..{n_windows - 1} windows")
            break

    rebuilt: dict[str, list[int]] = {edge_id: [0] * n_windows for edge_id in subgraph.unique_ids}
    for row in rows:
        rebuilt[row["edge_id"]][int(row["window_index"])] = int(row["vehicle_count"])
    for edge_id in subgraph.unique_ids:
        if rebuilt[edge_id] != counts[edge_id]:
            failures.append(f"flow csv series mismatch for {edge_id}")
            break

    if not index_path.is_file():
        failures.append(f"missing {index_path}")
    else:
        index_rows = read_flow_csv(index_path)
        index_edges = {row["edge_id"] for row in index_rows}
        if index_edges != unique_set:
            failures.append("plot index edge set != subgraph unique IDs")
        if len(index_rows) != n_edges:
            failures.append(f"plot index rows {len(index_rows)} != {n_edges}")

    pngs = sorted(plot_dir.glob("*.png")) if plot_dir.is_dir() else []
    if len(pngs) != n_edges:
        failures.append(f"png count {len(pngs)} != {n_edges}")
    png_names = [p.name for p in pngs]
    if len(png_names) != len(set(png_names)):
        failures.append("duplicate plot filenames")
    expected_names = {filenames[edge_id] for edge_id in subgraph.unique_ids}
    actual_names = set(png_names)
    if expected_names != actual_names:
        failures.append("plot filenames do not match index mapping")
    for png in pngs:
        if png.stat().st_size <= 0:
            failures.append(f"empty png {png.name}")
            break

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.image as mpimg

        for png in pngs[:3]:
            image = mpimg.imread(png)
            if image.size == 0:
                failures.append(f"unreadable png {png.name}")
    except Exception as exc:
        failures.append(f"png read failed: {exc}")

    traj_stat_after = trajectories.stat()
    if (traj_stat_after.st_size, traj_stat_after.st_mtime_ns) != traj_stat_before:
        failures.append("trajectories.csv size or mtime changed")

    rng = random.Random(seed)
    sample_edges = rng.sample(subgraph.unique_ids, k=min(3, n_edges))
    independent = recount_selected_edges(
        trajectories,
        unique_set,
        selected=set(sample_edges),
        window_seconds=window_seconds,
        n_windows=n_windows,
    )
    print("Sampled edges (seed %s): %s" % (seed, sample_edges), flush=True)
    for edge_id in sample_edges:
        if independent[edge_id] != counts[edge_id]:
            failures.append(f"independent recount mismatch for {edge_id}")
        else:
            total_e = sum(counts[edge_id])
            peak = max(counts[edge_id])
            peak_i = counts[edge_id].index(peak)
            print(
                f"  {edge_id}: total={total_e} peak={peak} at {time_label(peak_i, window_seconds)}",
                flush=True,
            )
    return failures


def recount_selected_edges(
    trajectories: Path,
    subgraph: set[str],
    *,
    selected: set[str],
    window_seconds: int,
    n_windows: int,
) -> dict[str, list[int]]:
    counts = {edge_id: [0] * n_windows for edge_id in selected}
    last_edge: dict[str, str] = {}
    with trajectories.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            vehicle_id = row["vehicle_id"]
            edge_id = row["edge_id"]
            previous = last_edge.get(vehicle_id)
            entered = previous is None or previous != edge_id
            last_edge[vehicle_id] = edge_id
            if not entered or edge_id not in selected:
                continue
            index = window_index_for(float(row["timestamp"]), window_seconds)
            if index is None:
                continue
            counts[edge_id][index] += 1
    return counts


def _bin_entries(
    rows: list[tuple[float, str, str]],
    subgraph: set[str],
    window_seconds: int,
) -> dict[str, list[int]]:
    n_windows = SECONDS_PER_DAY // window_seconds
    counts = {edge_id: [0] * n_windows for edge_id in subgraph}
    last_edge: dict[str, str] = {}
    for timestamp, vehicle_id, edge_id in rows:
        previous = last_edge.get(vehicle_id)
        entered = previous is None or previous != edge_id
        last_edge[vehicle_id] = edge_id
        if entered and edge_id in subgraph:
            counts[edge_id][math.floor(timestamp / window_seconds)] += 1
    return counts


def run_synthetic_checks() -> None:
    subgraph = {"edge_A", "edge_B"}
    rows = [
        (100.0, "v1", "edge_A"),
        (101.0, "v1", "edge_A"),
        (102.0, "v1", "edge_A"),
        (103.0, "v1", ":internal_1"),
        (104.0, "v1", "edge_B"),
        (200.0, "v1", "edge_A"),
        (100.0, "v2", "outside"),
        (110.0, "v2", "edge_A"),
        (400.0, "v3", "edge_A"),
    ]
    counts_1min = _bin_entries(rows, subgraph, 60)
    if sum(counts_1min["edge_A"]) != 4:
        raise FlowError(f"synthetic 1min edge_A expected 4 entries, got {sum(counts_1min['edge_A'])}")
    if counts_1min["edge_A"][1] != 2 or counts_1min["edge_A"][3] != 1 or counts_1min["edge_A"][6] != 1:
        raise FlowError(f"synthetic 1min edge_A bins wrong: {counts_1min['edge_A'][1:7]}")
    if counts_1min["edge_B"][1] != 1 or sum(counts_1min["edge_B"]) != 1:
        raise FlowError("synthetic 1min edge_B bins wrong")

    counts_5min = _bin_entries(rows, subgraph, 300)
    if sum(counts_5min["edge_A"]) != 4:
        raise FlowError(f"synthetic 5min edge_A expected 4 entries, got {sum(counts_5min['edge_A'])}")
    if counts_5min["edge_A"][0] != 3 or counts_5min["edge_A"][1] != 1:
        raise FlowError(f"synthetic 5min edge_A bins wrong: {counts_5min['edge_A'][:3]}")
    if counts_5min["edge_B"][0] != 1 or sum(counts_5min["edge_B"]) != 1:
        raise FlowError("synthetic 5min edge_B bins wrong")
    print("Synthetic stay / change / re-entry checks passed.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count subgraph edge vehicle entries by time window from a day's "
            "trajectories.csv and write one line plot per edge."
        )
    )
    parser.add_argument("--trajectories", type=Path, required=True, help="Day trajectories.csv")
    parser.add_argument("--subgraph", type=Path, required=True, help="Subgraph edge-id text file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Day output directory.")
    parser.add_argument("--day-label", required=True, help="Label used in plot titles.")
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=DEFAULT_WINDOW_MINUTES,
        help="Aggregation window in minutes (default 5). Ignored if --window-seconds is set.",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=None,
        help="Aggregation window in seconds. Overrides --window-minutes.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="PNG directory (default: <output-dir>/edge_flow_plots_<window>).",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Plot DPI (default 150).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for edge sampling checks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    window_seconds = resolve_window_seconds(args.window_minutes, args.window_seconds)
    n_windows = SECONDS_PER_DAY // window_seconds
    trajectories = args.trajectories.expanduser().resolve()
    subgraph_path = args.subgraph.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    plot_dir = (
        args.plot_dir.expanduser().resolve()
        if args.plot_dir is not None
        else output_dir / plot_dir_name(window_seconds)
    )
    for required in (trajectories, subgraph_path):
        if not required.is_file():
            raise SystemExit(f"not found: {required}")

    run_synthetic_checks()
    subgraph = load_subgraph(subgraph_path)
    print(
        f"subgraph {subgraph_path}: raw={subgraph.raw_line_count} "
        f"nonempty={subgraph.nonempty_line_count} unique={len(subgraph.unique_ids)} "
        f"duplicates={subgraph.duplicate_ids or 'none'}",
        flush=True,
    )
    print(
        f"window={window_seconds}s ({window_label(window_seconds)}) n_windows={n_windows}",
        flush=True,
    )
    traj_stat_before = (trajectories.stat().st_size, trajectories.stat().st_mtime_ns)
    print(f"Scanning {trajectories}", flush=True)
    counts, stats, _last_edge = count_entries(
        trajectories,
        subgraph.unique_set,
        window_seconds=window_seconds,
        n_windows=n_windows,
    )
    print(
        f"  rows={stats.rows:,} subgraph_entries={stats.entry_count:,} "
        f"edges_with_entry={len(stats.subgraph_edges_with_entry)} "
        f"timestamp={stats.timestamp_min}..{stats.timestamp_max}",
        flush=True,
    )
    if stats.flag_mismatches:
        print(
            f"WARNING is_in_subgraph mismatches vs subgraph.txt: {stats.flag_mismatches}",
            file=sys.stderr,
            flush=True,
        )
        for example in stats.flag_mismatch_examples:
            print(f"  {example}", file=sys.stderr, flush=True)

    used_names: set[str] = set()
    filenames = {
        edge_id: safe_plot_filename(edge_id, used_names) for edge_id in subgraph.unique_ids
    }
    flow_path = output_dir / flow_csv_name(window_seconds)
    index_path = output_dir / index_csv_name(window_seconds)
    write_flow_csv(
        flow_path,
        subgraph.unique_ids,
        counts,
        window_seconds=window_seconds,
        n_windows=n_windows,
    )
    write_index_csv(
        index_path,
        subgraph.unique_ids,
        counts,
        filenames,
        window_seconds=window_seconds,
    )
    print(f"Wrote {flow_path}", flush=True)
    print(f"Plotting {len(subgraph.unique_ids)} edges -> {plot_dir}", flush=True)
    plot_edges(
        plot_dir,
        subgraph.unique_ids,
        counts,
        filenames,
        day_label=args.day_label,
        window_seconds=window_seconds,
        n_windows=n_windows,
        dpi=args.dpi,
    )
    print(f"Wrote {index_path}", flush=True)

    zero_edges = [edge_id for edge_id in subgraph.unique_ids if sum(counts[edge_id]) == 0]
    ranked = sorted(
        ((sum(counts[edge_id]), edge_id) for edge_id in subgraph.unique_ids),
        reverse=True,
    )
    print(f"zero-flow edges ({len(zero_edges)}): {zero_edges or 'none'}", flush=True)
    print("top 10 edges by entries:", flush=True)
    for total, edge_id in ranked[:10]:
        print(f"  {edge_id} {total}", flush=True)

    failures = verify_outputs(
        subgraph=subgraph,
        stats=stats,
        counts=counts,
        flow_path=flow_path,
        index_path=index_path,
        plot_dir=plot_dir,
        filenames=filenames,
        n_windows=n_windows,
        window_seconds=window_seconds,
        trajectories=trajectories,
        traj_stat_before=traj_stat_before,
        seed=args.seed,
    )
    if failures:
        raise SystemExit("verification failed:\n" + "\n".join(failures))
    print("All flow/plot checks passed.", flush=True)


if __name__ == "__main__":
    main()
