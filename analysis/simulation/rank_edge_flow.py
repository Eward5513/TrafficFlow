"""
Rank subgraph edges by daily vehicle entries and plot intra-day differences.

Reads already written edge_flow_<window>.csv. Does not open trajectories or FCD.

Usage:

    python3 analysis/simulation/rank_edge_flow.py \\
        --root analysis/simulation/data/processed/subgraph_trajectories \\
        --window-minutes 5
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

SECONDS_PER_DAY = 24 * 3600
DEFAULT_WINDOW_MINUTES = 5


@dataclass
class EdgeRank:
    rank: int
    edge_id: str
    total: int
    share_pct: float
    peak: int
    peak_time: str
    trough: int
    intra_day_range: int
    n_windows: int
    mean_per_window: float
    series: list[int]


class RankError(Exception):
    pass


def window_label(window_seconds: int) -> str:
    if window_seconds % 60 == 0:
        return f"{window_seconds // 60}min"
    return f"{window_seconds}s"


def flow_csv_name(window_seconds: int) -> str:
    return f"edge_flow_{window_label(window_seconds)}.csv"


def rank_csv_name(window_seconds: int) -> str:
    return f"edge_flow_rank_{window_label(window_seconds)}.csv"


def compare_plot_name(window_seconds: int) -> str:
    return f"edge_flow_compare_{window_label(window_seconds)}.png"


def mean_plot_name(window_seconds: int) -> str:
    return f"edge_flow_mean_{window_label(window_seconds)}.png"


def time_label(window_index: int, window_seconds: int) -> str:
    seconds = window_index * window_seconds
    hours, rem = divmod(seconds, 3600)
    minutes = rem // 60
    return f"{hours:02d}:{minutes:02d}"


def load_flow_csv(path: Path) -> tuple[dict[str, list[int]], int]:
    by_index: dict[str, dict[int, int]] = {}
    starts: dict[int, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = ("window_index", "window_start", "edge_id", "vehicle_count")
        if reader.fieldnames is None:
            raise RankError(f"{path} has no header")
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise RankError(f"{path} missing fields {missing}")
        for row in reader:
            edge_id = row["edge_id"]
            index = int(row["window_index"])
            by_index.setdefault(edge_id, {})[index] = int(row["vehicle_count"])
            starts[index] = int(row["window_start"])
    if not by_index:
        raise RankError(f"no rows in {path}")
    n_windows = max(max(indexes) for indexes in by_index.values()) + 1
    counts = {
        edge_id: [values.get(index, 0) for index in range(n_windows)]
        for edge_id, values in by_index.items()
    }
    window_seconds = starts.get(1, SECONDS_PER_DAY // n_windows)
    if window_seconds <= 0:
        window_seconds = SECONDS_PER_DAY // n_windows
    return counts, window_seconds


def rank_edges(counts: dict[str, list[int]], window_seconds: int) -> list[EdgeRank]:
    grand_total = sum(sum(series) for series in counts.values())
    items: list[tuple[int, str, list[int]]] = []
    for edge_id, series in counts.items():
        items.append((sum(series), edge_id, series))
    items.sort(key=lambda item: (-item[0], item[1]))
    ranked: list[EdgeRank] = []
    for rank, (total, edge_id, series) in enumerate(items, start=1):
        peak = max(series) if series else 0
        trough = min(series) if series else 0
        peak_index = series.index(peak) if series else 0
        share = (100.0 * total / grand_total) if grand_total else 0.0
        n_windows = len(series)
        mean_per_window = (total / n_windows) if n_windows else 0.0
        ranked.append(
            EdgeRank(
                rank=rank,
                edge_id=edge_id,
                total=total,
                share_pct=share,
                peak=peak,
                peak_time=time_label(peak_index, window_seconds),
                trough=trough,
                intra_day_range=peak - trough,
                n_windows=n_windows,
                mean_per_window=mean_per_window,
                series=series,
            )
        )
    return ranked


def write_rank_csv(path: Path, ranked: list[EdgeRank]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "rank",
                "edge_id",
                "total_vehicle_entries",
                "n_windows",
                "mean_entries_per_window",
                "share_pct",
                "peak_vehicle_entries",
                "peak_time",
                "trough_vehicle_entries",
                "intra_day_range",
            ]
        )
        for row in ranked:
            writer.writerow(
                [
                    row.rank,
                    row.edge_id,
                    row.total,
                    row.n_windows,
                    f"{row.mean_per_window:.4f}",
                    f"{row.share_pct:.4f}",
                    row.peak,
                    row.peak_time,
                    row.trough,
                    row.intra_day_range,
                ]
            )


def plot_compare(
    path: Path,
    ranked: list[EdgeRank],
    *,
    day_label: str,
    window_seconds: int,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    n_edges = len(ranked)
    n_windows = len(ranked[0].series) if ranked else 0
    ids = [row.edge_id for row in ranked]
    totals = [row.total for row in ranked]
    matrix = [row.series for row in ranked]
    y = list(range(n_edges))
    fig, (ax_bar, ax_heat) = plt.subplots(
        1,
        2,
        figsize=(16.5, max(9.0, 0.22 * n_edges + 2.2)),
        gridspec_kw={"width_ratios": [1.05, 2.35], "wspace": 0.12},
        sharey=True,
        layout="constrained",
    )
    ax_bar.barh(y, totals, color="#1f4e79", height=0.78)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(ids, fontsize=7)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Daily Vehicle Entries")
    ax_bar.set_title("Ranked daily total")
    ax_bar.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax_bar.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    ax_bar.set_axisbelow(True)

    image = ax_heat.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="Blues",
        origin="upper",
    )
    hour_ticks = list(range(0, 25, 2))
    col_ticks = [hour * 3600 / window_seconds for hour in hour_ticks]
    ax_heat.set_xlim(-0.5, n_windows - 0.5)
    ax_heat.set_xticks(col_ticks)
    ax_heat.set_xticklabels([f"{hour:02d}:00" for hour in hour_ticks], fontsize=8)
    ax_heat.set_xlabel("Time of Day")
    ax_heat.set_title("Entries per window")
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.035, pad=0.02)
    window_text = (
        f"{window_seconds // 60} min" if window_seconds % 60 == 0 else f"{window_seconds} s"
    )
    colorbar.set_label(f"Vehicle Entries per {window_text}")
    fig.suptitle(
        f"{day_label}  subgraph edges ranked by daily flow",
        fontsize=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_mean_flow(
    path: Path,
    ranked: list[EdgeRank],
    *,
    day_label: str,
    window_seconds: int,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_edges = len(ranked)
    n_windows = ranked[0].n_windows if ranked else 0
    ids = [row.edge_id for row in ranked]
    means = [row.mean_per_window for row in ranked]
    subgraph_mean = (sum(means) / n_edges) if n_edges else 0.0
    y = list(range(n_edges))
    window_text = (
        f"{window_seconds // 60} min" if window_seconds % 60 == 0 else f"{window_seconds} s"
    )
    fig, ax = plt.subplots(figsize=(8.8, max(9.0, 0.22 * n_edges + 2.2)), layout="constrained")
    ax.barh(y, means, color="#1f4e79", height=0.78)
    ax.axvline(
        subgraph_mean,
        color="#b35c1e",
        linestyle="--",
        linewidth=1.1,
        label=f"subgraph mean  {subgraph_mean:.2f}",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(ids, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"Mean Vehicle Entries per {window_text}")
    ax.set_title(f"{day_label}  daily total / {n_windows} windows")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def process_day(
    day_dir: Path,
    *,
    window_seconds: int,
    day_label: str,
    dpi: int,
) -> list[EdgeRank]:
    flow_path = day_dir / flow_csv_name(window_seconds)
    if not flow_path.is_file():
        raise RankError(f"missing {flow_path}")
    counts, detected = load_flow_csv(flow_path)
    if detected != window_seconds:
        print(
            f"WARNING {day_label}: flow csv window {detected}s != requested {window_seconds}s",
            flush=True,
        )
    ranked = rank_edges(counts, window_seconds)
    write_rank_csv(day_dir / rank_csv_name(window_seconds), ranked)
    plot_compare(
        day_dir / compare_plot_name(window_seconds),
        ranked,
        day_label=day_label,
        window_seconds=window_seconds,
        dpi=dpi,
    )
    plot_mean_flow(
        day_dir / mean_plot_name(window_seconds),
        ranked,
        day_label=day_label,
        window_seconds=window_seconds,
        dpi=dpi,
    )
    return ranked


def write_summary(
    root: Path,
    by_day: dict[str, list[EdgeRank]],
    *,
    window_seconds: int,
    dpi: int,
) -> None:
    if not by_day:
        return
    edge_ids: list[str] = []
    seen: set[str] = set()
    for ranked in by_day.values():
        for row in ranked:
            if row.edge_id not in seen:
                seen.add(row.edge_id)
                edge_ids.append(row.edge_id)
    n_windows = len(next(iter(by_day.values()))[0].series)
    n_days = len(by_day)
    totals: dict[str, list[int]] = {edge_id: [] for edge_id in edge_ids}
    ranks: dict[str, list[int]] = {edge_id: [] for edge_id in edge_ids}
    mean_series: dict[str, list[float]] = {
        edge_id: [0.0] * n_windows for edge_id in edge_ids
    }
    for ranked in by_day.values():
        for row in ranked:
            totals[row.edge_id].append(row.total)
            ranks[row.edge_id].append(row.rank)
            for index, value in enumerate(row.series):
                mean_series[row.edge_id][index] += value / n_days
    summary_rows: list[EdgeRank] = []
    items = []
    for edge_id in edge_ids:
        mean_total = sum(totals[edge_id]) / n_days
        series = [int(round(value)) for value in mean_series[edge_id]]
        items.append((mean_total, edge_id, series, totals[edge_id], ranks[edge_id]))
    items.sort(key=lambda item: (-item[0], item[1]))
    grand = sum(item[0] for item in items)
    path = root / f"edge_flow_rank_{window_label(window_seconds)}_summary.csv"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "rank",
                "edge_id",
                "mean_daily_entries",
                "n_windows",
                "mean_entries_per_window",
                "min_daily_entries",
                "max_daily_entries",
                "mean_rank",
                "share_pct",
                "peak_of_mean_profile",
                "peak_time",
            ]
        )
        for rank, (mean_total, edge_id, series, day_totals, day_ranks) in enumerate(
            items, start=1
        ):
            peak = max(series) if series else 0
            peak_index = series.index(peak) if series else 0
            share = (100.0 * mean_total / grand) if grand else 0.0
            mean_per_window = (mean_total / n_windows) if n_windows else 0.0
            writer.writerow(
                [
                    rank,
                    edge_id,
                    f"{mean_total:.2f}",
                    n_windows,
                    f"{mean_per_window:.4f}",
                    min(day_totals),
                    max(day_totals),
                    f"{sum(day_ranks) / len(day_ranks):.2f}",
                    f"{share:.4f}",
                    peak,
                    time_label(peak_index, window_seconds),
                ]
            )
            summary_rows.append(
                EdgeRank(
                    rank=rank,
                    edge_id=edge_id,
                    total=int(round(mean_total)),
                    share_pct=share,
                    peak=peak,
                    peak_time=time_label(peak_index, window_seconds),
                    trough=min(series) if series else 0,
                    intra_day_range=peak - (min(series) if series else 0),
                    n_windows=n_windows,
                    mean_per_window=mean_per_window,
                    series=series,
                )
            )
    plot_compare(
        root / f"edge_flow_compare_{window_label(window_seconds)}_summary.png",
        summary_rows,
        day_label=f"mean of {n_days} days",
        window_seconds=window_seconds,
        dpi=dpi,
    )
    plot_mean_flow(
        root / f"edge_flow_mean_{window_label(window_seconds)}_summary.png",
        summary_rows,
        day_label=f"mean of {n_days} days",
        window_seconds=window_seconds,
        dpi=dpi,
    )
    print(f"Wrote {path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank subgraph edges by daily vehicle entries from edge_flow CSV "
            "and plot a ranked bar chart plus intra-day heatmap."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Directory of day_* folders. Processes every day that has a flow CSV.",
    )
    parser.add_argument("--flow-csv", type=Path, default=None, help="One day's edge_flow CSV.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for one day.")
    parser.add_argument("--day-label", default=None, help="Label used in the comparison plot title.")
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
    parser.add_argument("--dpi", type=int, default=150, help="Plot DPI (default 150).")
    return parser.parse_args()


def resolve_window_seconds(window_minutes: int, window_seconds: int | None) -> int:
    seconds = window_seconds if window_seconds is not None else window_minutes * 60
    if seconds <= 0 or SECONDS_PER_DAY % seconds != 0:
        raise SystemExit("window must be a positive divisor of 86400 seconds")
    return seconds


def main() -> None:
    args = parse_args()
    window_seconds = resolve_window_seconds(args.window_minutes, args.window_seconds)
    if args.root is not None:
        root = args.root.expanduser().resolve()
        day_dirs = sorted(path for path in root.glob("day_*") if path.is_dir())
        if not day_dirs:
            raise SystemExit(f"no day_* directories in {root}")
        by_day: dict[str, list[EdgeRank]] = {}
        for day_dir in day_dirs:
            flow_path = day_dir / flow_csv_name(window_seconds)
            if not flow_path.is_file():
                print(f"skip {day_dir.name}: missing {flow_path.name}", flush=True)
                continue
            ranked = process_day(
                day_dir,
                window_seconds=window_seconds,
                day_label=day_dir.name,
                dpi=args.dpi,
            )
            by_day[day_dir.name] = ranked
            print(
                f"{day_dir.name}: {len(ranked)} edges "
                f"top={ranked[0].edge_id} mean={ranked[0].mean_per_window:.2f} "
                f"bottom={ranked[-1].edge_id} mean={ranked[-1].mean_per_window:.2f}",
                flush=True,
            )
        write_summary(root, by_day, window_seconds=window_seconds, dpi=args.dpi)
        print(f"ranked {len(by_day)} days", flush=True)
        return

    if args.flow_csv is None or args.output_dir is None or args.day_label is None:
        raise SystemExit("single-day mode needs --flow-csv --output-dir --day-label")
    output_dir = args.output_dir.expanduser().resolve()
    counts, _detected = load_flow_csv(args.flow_csv.expanduser().resolve())
    ranked = rank_edges(counts, window_seconds)
    write_rank_csv(output_dir / rank_csv_name(window_seconds), ranked)
    plot_compare(
        output_dir / compare_plot_name(window_seconds),
        ranked,
        day_label=args.day_label,
        window_seconds=window_seconds,
        dpi=args.dpi,
    )
    plot_mean_flow(
        output_dir / mean_plot_name(window_seconds),
        ranked,
        day_label=args.day_label,
        window_seconds=window_seconds,
        dpi=args.dpi,
    )
    print(
        f"Wrote rank CSV, compare plot, and mean-flow plot for {args.day_label} "
        f"({len(ranked)} edges)",
        flush=True,
    )


if __name__ == "__main__":
    main()
