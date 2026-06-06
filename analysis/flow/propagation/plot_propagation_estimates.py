"""Plot incoming-edge propagation estimates against downstream actuals."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


PROPAGATION_DIR = Path(__file__).resolve().parent
FLOW_DIR = PROPAGATION_DIR.parent
PROPAGATION_METRICS_DIR = PROPAGATION_DIR / "metrics"
PLOT_DIR = PROPAGATION_DIR / "plot"

DEFAULT_SEED = 42
TARGET_NMAE_FIELD = "zero_filtered_scale_smoothed_average_estimate"
PLOT_ESTIMATE_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("upstream_estimate", "upstream estimate", "-"),
    ("smoothed_upstream_estimate", "upstream estimate smoothed", "--"),
    (
        "zero_filtered_smoothed_upstream_estimate",
        "upstream estimate zero-filtered smoothed",
        "-.",
    ),
    (
        TARGET_NMAE_FIELD,
        "zero-filtered upstream + scale smoothed weighted average",
        ":",
    ),
)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def penetration_sort_key(penetration: str) -> tuple[int, float | str]:
    try:
        return (0, float(penetration.replace("_", ".")))
    except ValueError:
        return (1, penetration)


def overall_metrics_by_target(
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Read generated overall propagation metrics grouped by downstream junction."""
    grouped_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for metrics_file in sorted(
        PROPAGATION_METRICS_DIR.glob("*/*_overall_propagation_metrics.json"),
        key=lambda path: (penetration_sort_key(path.parent.name), path.name),
    ):
        metrics_data = load_json(metrics_file)
        if int(metrics_data.get("seed", seed)) != seed:
            continue

        downstream_junction_id = str(metrics_data["downstreamJunctionId"])
        penetration = str(metrics_data.get("penetration") or metrics_file.parent.name)
        grouped_metrics.setdefault(downstream_junction_id, {})[
            penetration
        ] = metrics_data

    if not grouped_metrics:
        raise FileNotFoundError(
            f"no overall propagation metrics found under: {PROPAGATION_METRICS_DIR}"
        )

    return {
        downstream_junction_id: dict(
            sorted(
                metrics_by_penetration.items(),
                key=lambda item: penetration_sort_key(item[0]),
            )
        )
        for downstream_junction_id, metrics_by_penetration in sorted(
            grouped_metrics.items()
        )
    }


def safe_filename_part(value: str) -> str:
    """Return a filesystem-safe filename component."""
    return "".join(
        character if character.isalnum() or character in ("-", "_", "#") else "_"
        for character in value
    )


def incoming_edge_plot_file(
    downstream_junction_id: str,
    penetration: str,
    seed: int,
    incoming_edge: str,
) -> Path:
    """Return output path for a per-incoming-edge propagation plot."""
    return (
        PLOT_DIR
        / penetration
        / (
            f"{downstream_junction_id}_"
            f"{penetration}_{seed}_edge_{safe_filename_part(incoming_edge)}_"
            "scale_vs_propagation.png"
        )
    )


def nmae_by_penetration_file(
    downstream_junction_id: str,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Return output path for the propagation NMAE trend plot."""
    return (
        PLOT_DIR
        / f"{downstream_junction_id}_{seed}_"
        "zero_filtered_scale_smoothed_average_nmae_by_penetration.png"
    )


def clear_plot_dir() -> None:
    """Remove existing plot outputs before creating new ones."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for path in PLOT_DIR.iterdir():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def incoming_edges(metrics_data: dict[str, Any]) -> list[str]:
    """Return downstream incoming edge ids present in an overall metrics payload."""
    return sorted(
        {
            str(record["downstreamIncomingEdge"])
            for record in metrics_data.get("records", [])
            if "downstreamIncomingEdge" in record
        }
    )


def records_for_incoming_edge(
    metrics_data: dict[str, Any],
    incoming_edge: str | None,
) -> list[dict[str, Any]]:
    """Return all records, or records for one downstream incoming edge."""
    records = metrics_data.get("records", [])
    if incoming_edge is None:
        return list(records)
    return [
        record
        for record in records
        if str(record.get("downstreamIncomingEdge")) == incoming_edge
    ]


def nmae_for_records(
    records: list[dict[str, Any]],
    field_name: str = TARGET_NMAE_FIELD,
) -> float | None:
    """Calculate record-level NMAE without cycle or phase aggregation."""
    valid_records = [
        record
        for record in records
        if field_name in record and record.get(field_name) is not None
    ]
    actual_total = sum(
        float(record.get("actualCount", 0.0) or 0.0)
        for record in valid_records
    )
    if not valid_records or actual_total == 0:
        return None
    absolute_error = sum(
        abs(float(record[field_name]) - float(record.get("actualCount", 0.0) or 0.0))
        for record in valid_records
    )
    return absolute_error / actual_total


def plot_nmae_by_penetration(
    nmae_by_series: dict[str, list[tuple[str, float]]],
    plot_file: Path,
    downstream_junction_id: str,
) -> None:
    """Plot per-link and overall propagation NMAE across penetration rates."""
    if not any(nmae_by_series.values()):
        print("skipped propagation NMAE plot with no data")
        return

    plt.figure(figsize=(10, 6))
    for series_name, nmae_points in nmae_by_series.items():
        if not nmae_points:
            continue

        ordered_points = sorted(
            nmae_points,
            key=lambda item: penetration_sort_key(item[0]),
        )
        penetrations = [penetration for penetration, _nmae in ordered_points]
        x_values = [float(penetration.replace("_", ".")) for penetration in penetrations]
        y_values = [nmae for _penetration, nmae in ordered_points]
        plt.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.0,
            markersize=5,
            label=series_name,
        )

    ordered_penetrations = sorted(
        {
            penetration
            for nmae_points in nmae_by_series.values()
            for penetration, _nmae in nmae_points
        },
        key=penetration_sort_key,
    )
    x_ticks = [
        float(penetration.replace("_", "."))
        for penetration in ordered_penetrations
    ]
    plt.title(
        "Record-level propagation zero-filtered scale smoothed weighted estimate "
        f"NMAE by Penetration - {downstream_junction_id}"
    )
    plt.xlabel("Penetration Rate (%)")
    plt.ylabel("NMAE")
    plt.xticks(x_ticks, ordered_penetrations)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plot_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_file, dpi=200)
    plt.close()


def plot_link_records(
    metrics_data: dict[str, Any],
    records: list[dict[str, Any]],
    plot_file: Path,
    penetration: str,
    downstream_junction_id: str,
    series_name: str,
) -> None:
    """Plot actual downstream counts against propagation estimates for one link."""
    if not records:
        print(f"skipped {penetration} {series_name}: no records")
        return

    _ = metrics_data
    ordered_records = sorted(records, key=lambda record: int(record["cycleIndex"]))
    plotted_cycle_indices = [int(record["cycleIndex"]) for record in ordered_records]

    plt.figure(figsize=(12, 6))
    plt.plot(
        plotted_cycle_indices,
        [float(record.get("actualCount", 0.0) or 0.0) for record in ordered_records],
        marker="o",
        linewidth=1.8,
        markersize=3,
        label="actual",
    )
    for field_name, label, linestyle in PLOT_ESTIMATE_FIELDS:
        if not any(field_name in record for record in records):
            continue
        plt.plot(
            plotted_cycle_indices,
            [
                float(record.get(field_name, 0.0) or 0.0)
                for record in ordered_records
            ],
            marker="o",
            linestyle=linestyle,
            linewidth=2.2,
            markersize=4,
            label=label,
        )

    plt.title(
        f"{series_name} Actual Count vs Upstream Estimates - "
        f"{downstream_junction_id} ({penetration}%)"
    )
    plt.xlabel("Downstream cycle index")
    plt.ylabel("Vehicle Count")
    plt.xticks(plotted_cycle_indices)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plot_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_file, dpi=200)
    plt.close()


if __name__ == "__main__":
    clear_plot_dir()
    metrics_by_target = overall_metrics_by_target(DEFAULT_SEED)

    for downstream_junction_id, metrics_by_penetration in metrics_by_target.items():
        target_incoming_edges = sorted(
            {
                incoming_edge
                for metrics_data in metrics_by_penetration.values()
                for incoming_edge in incoming_edges(metrics_data)
            }
        )
        nmae_by_series: dict[str, list[tuple[str, float]]] = {
            incoming_edge: []
            for incoming_edge in target_incoming_edges
        }
        nmae_by_series["overall"] = []

        for penetration, metrics_data in metrics_by_penetration.items():
            for incoming_edge in target_incoming_edges:
                link_records = records_for_incoming_edge(metrics_data, incoming_edge)
                nmae = nmae_for_records(link_records)
                if nmae is not None:
                    nmae_by_series[incoming_edge].append(
                        (str(penetration), nmae)
                    )

                plot_file = incoming_edge_plot_file(
                    downstream_junction_id,
                    str(penetration),
                    DEFAULT_SEED,
                    incoming_edge,
                )
                plot_link_records(
                    metrics_data,
                    link_records,
                    plot_file,
                    str(penetration),
                    downstream_junction_id,
                    f"incoming edge {incoming_edge}",
                )
                print(f"saved {penetration} {incoming_edge}: {plot_file.resolve()}")

            overall_records = records_for_incoming_edge(metrics_data, None)
            nmae = nmae_for_records(overall_records)
            if nmae is not None:
                nmae_by_series["overall"].append((str(penetration), nmae))

        plot_file = nmae_by_penetration_file(downstream_junction_id, DEFAULT_SEED)
        plot_nmae_by_penetration(nmae_by_series, plot_file, downstream_junction_id)
        print(f"saved propagation NMAE by penetration: {plot_file.resolve()}")
