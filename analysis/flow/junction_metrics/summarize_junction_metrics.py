"""Write overall propagation and scale metric summaries for selected junctions."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


JUNCTION_METRICS_DIR = Path(__file__).resolve().parent
FLOW_DIR = JUNCTION_METRICS_DIR.parent
SCALE_DIR = FLOW_DIR / "scale"
PROPAGATION_DIR = FLOW_DIR / "propagation"
PROPAGATION_METRICS_DIR = FLOW_DIR / "propagation" / "metrics"
SCALE_METRICS_DIR = FLOW_DIR / "scale" / "metrics"
DEFAULT_SEED = 42
PREPROCESSING_SCRIPTS: tuple[Path, ...] = (
    SCALE_DIR / "calculate_scale_estimates.py",
    SCALE_DIR / "plot_scale_estimates.py",
    PROPAGATION_DIR / "calculate_propagation_estimates.py",
    PROPAGATION_DIR / "plot_propagation_estimates.py",
)
JUNCTION_IDS: tuple[str, ...] = (
    "cluster_1262396675_2350807772",
    "cluster_1746667327_1746667337",
    "1746667341",
)

ESTIMATE_FIELDS: tuple[tuple[str, str], ...] = (
    ("upstreamEstimate", "upstream_estimate"),
    ("smoothedUpstreamEstimate", "smoothed_upstream_estimate"),
    (
        "zeroFilteredSmoothedUpstreamEstimate",
        "zero_filtered_smoothed_upstream_estimate",
    ),
    (
        "zeroFilteredScaleSmoothedAverageEstimate",
        "zero_filtered_scale_smoothed_average_estimate",
    ),
)


def penetration_sort_key(penetration: str) -> tuple[int, float | str]:
    """Sort numeric penetration labels before non-numeric labels."""
    try:
        return (0, float(penetration.replace("_", ".")))
    except ValueError:
        return (1, penetration)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON object to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_script(script_path: Path) -> None:
    """Run one preprocessing script with the current Python interpreter."""
    if not script_path.exists():
        raise FileNotFoundError(f"script not found: {script_path}")
    print(f"running {script_path.relative_to(FLOW_DIR)}")
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(script_path.parent),
        check=True,
    )


def run_preprocessing_scripts() -> None:
    """Regenerate scale and propagation metrics/plots before summarizing."""
    for script_path in PREPROCESSING_SCRIPTS:
        run_script(script_path)


def overall_metrics_files(
    downstream_junction_id: str,
    seed: int,
    metrics_dir: Path,
) -> list[Path]:
    """Return matching overall propagation metrics files."""
    pattern = f"{downstream_junction_id}_*_{seed}_overall_propagation_metrics.json"
    return sorted(
        metrics_dir.glob(f"*/{pattern}"),
        key=lambda path: (penetration_sort_key(path.parent.name), path.name),
    )


def scale_metrics_file(
    junction_id: str,
    penetration: str,
    seed: int,
    metrics_dir: Path,
) -> Path:
    """Return the scale metrics file for one junction and penetration."""
    return metrics_dir / penetration / f"{junction_id}_{seed}_scale_metrics.json"


def summarize_records(
    records: list[dict[str, Any]],
    estimate_field: str,
) -> dict[str, float | int | None]:
    """Summarize one estimate field against actual counts."""
    valid_pairs = [
        (float(record[estimate_field]), float(record["actualCount"]))
        for record in records
        if estimate_field in record and record.get(estimate_field) is not None
    ]
    if not valid_pairs:
        return {
            "periodCount": 0,
            "mae": None,
            "rmse": None,
            "estimatedTotal": 0.0,
            "actualTotal": 0.0,
            "nmae": None,
        }

    errors = [estimate - actual for estimate, actual in valid_pairs]
    absolute_error_sum = sum(abs(error) for error in errors)
    squared_error_sum = sum(error**2 for error in errors)
    estimated_total = sum(estimate for estimate, _actual in valid_pairs)
    actual_total = sum(actual for _estimate, actual in valid_pairs)
    period_count = len(valid_pairs)
    return {
        "periodCount": period_count,
        "mae": absolute_error_sum / period_count,
        "rmse": math.sqrt(squared_error_sum / period_count),
        "estimatedTotal": estimated_total,
        "actualTotal": actual_total,
        "nmae": absolute_error_sum / actual_total if actual_total else None,
    }


def summarize_propagation_overall(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize all configured propagation estimate fields for overall records."""
    return {
        metric_name: summarize_records(records, field_name)
        for metric_name, field_name in ESTIMATE_FIELDS
    }


def summarize_propagation_by_incoming_edge(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize propagation metrics for each downstream incoming edge."""
    records_by_edge: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        incoming_edge = str(record["downstreamIncomingEdge"])
        records_by_edge.setdefault(incoming_edge, []).append(record)

    return {
        incoming_edge: {
            "upstreamJunctionId": str(edge_records[0]["upstreamJunctionId"]),
            "phaseIndex": int(edge_records[0]["phaseIndex"]),
            "recordCount": len(edge_records),
            "metrics": summarize_propagation_overall(edge_records),
        }
        for incoming_edge, edge_records in sorted(records_by_edge.items())
    }


def empty_scale_summary() -> dict[str, Any]:
    """Return an empty scale metric summary."""
    return {
        "periodCount": 0,
        "mae": None,
        "mse": None,
        "rmse": None,
        "estimatedTotal": 0.0,
        "actualTotal": 0.0,
        "nmae": None,
    }


def summarize_scale_period_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize scale period count rows."""
    if not rows:
        return empty_scale_summary()

    period_count = len(rows)
    absolute_error_total = sum(float(row["absoluteError"]) for row in rows)
    squared_error_total = sum(float(row["squaredError"]) for row in rows)
    estimated_total = sum(float(row["observedCount"]) for row in rows)
    actual_total = sum(float(row["actualCount"]) for row in rows)
    mse = squared_error_total / period_count
    return {
        "periodCount": period_count,
        "mae": absolute_error_total / period_count,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "estimatedTotal": estimated_total,
        "actualTotal": actual_total,
        "nmae": absolute_error_total / actual_total if actual_total else None,
    }


def scale_overall_metrics_for_cycle_range(
    metrics_data: dict[str, Any],
    cycle_index_range: dict[str, Any],
) -> dict[str, Any]:
    """Summarize scale overall metrics over the propagation cycle range."""
    min_cycle_index = int(cycle_index_range["min"])
    max_cycle_index = int(cycle_index_range["max"])
    summaries = {}
    for method_name, method_data in metrics_data.get("methods", {}).items():
        rows = []
        for entry in method_data.get("metricTimeline", []):
            cycle_index = int(entry["cycleIndex"])
            if not min_cycle_index <= cycle_index <= max_cycle_index:
                continue
            rows.extend(entry.get("periodCounts", {}).values())
        summaries[method_name] = summarize_scale_period_counts(rows)
    return summaries


def build_junction_summary(
    downstream_junction_id: str,
    seed: int,
    propagation_metrics_dir: Path,
    scale_metrics_dir: Path,
) -> dict[str, Any]:
    """Build overall propagation and scale metric summaries for all penetrations."""
    propagation_files = overall_metrics_files(
        downstream_junction_id,
        seed,
        propagation_metrics_dir,
    )
    if not propagation_files:
        raise FileNotFoundError(
            "no overall propagation metrics found for "
            f"{downstream_junction_id}, seed={seed}, under {propagation_metrics_dir}"
        )

    penetration_summaries = []
    for propagation_file in propagation_files:
        propagation_data = load_json(propagation_file)
        penetration = str(
            propagation_data.get("penetration") or propagation_file.parent.name
        )
        scale_file = scale_metrics_file(
            downstream_junction_id,
            penetration,
            seed,
            scale_metrics_dir,
        )
        scale_data = load_json(scale_file)
        propagation_cycle_index_range = propagation_data.get("cycleIndexRange")
        propagation_records = propagation_data.get("records", [])

        penetration_summaries.append(
            {
                "penetration": penetration,
                "propagation": {
                    "sourceFile": str(propagation_file),
                    "cycleIndexRange": propagation_data.get("cycleIndexRange"),
                    "metricEndTimeSeconds": propagation_data.get(
                        "metricEndTimeSeconds"
                    ),
                    "recordCount": propagation_data.get("recordCount"),
                    "overall": summarize_propagation_overall(propagation_records),
                    "byIncomingEdge": summarize_propagation_by_incoming_edge(
                        propagation_records
                    ),
                },
                "scale": {
                    "sourceFile": str(scale_file),
                    "cycleRange": scale_data.get("cycleRange"),
                    "metricCycleRange": scale_data.get("metricCycleRange"),
                    "summarizedCycleIndexRange": propagation_cycle_index_range,
                    "penetrationRate": scale_data.get("penetrationRate"),
                    "overall": scale_overall_metrics_for_cycle_range(
                        scale_data,
                        propagation_cycle_index_range,
                    ),
                },
            }
        )

    return {
        "downstreamJunctionId": downstream_junction_id,
        "seed": seed,
        "propagationMetricsDir": str(propagation_metrics_dir),
        "scaleMetricsDir": str(scale_metrics_dir),
        "propagationEstimateFields": [
            {"metricName": metric_name, "recordField": field_name}
            for metric_name, field_name in ESTIMATE_FIELDS
        ],
        "penetrations": penetration_summaries,
    }


def output_file(
    output_dir: Path,
    downstream_junction_id: str,
    seed: int,
) -> Path:
    """Return the summary JSON output path for one junction."""
    return output_dir / f"{downstream_junction_id}_{seed}_metrics_summary.json"


def main() -> None:
    """Write one summary JSON file per configured junction."""
    run_preprocessing_scripts()
    for downstream_junction_id in JUNCTION_IDS:
        summary = build_junction_summary(
            downstream_junction_id=downstream_junction_id,
            seed=DEFAULT_SEED,
            propagation_metrics_dir=PROPAGATION_METRICS_DIR,
            scale_metrics_dir=SCALE_METRICS_DIR,
        )
        write_json(
            output_file(JUNCTION_METRICS_DIR, downstream_junction_id, DEFAULT_SEED),
            summary,
        )


if __name__ == "__main__":
    main()
