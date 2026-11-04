#!/usr/bin/env python3
"""Join seven-rate partial R-level and M-level states into MR-level CSVs.

No vehicle is resampled and no FCD is read. One worker handles one day and
walks the rates p05..p70. For each rate and state the R-level partial matrix
and the M-level partial matrix are concatenated in ``mr_graph.npz`` node order,
exactly as ``generate_mr_level_full_flow.py`` does for the full data.

Every cell is checked: R/M blocks equal their inputs, no negative values,
per-node stock recursion, partial <= full MR, and p05 <= p10 <= ... <= p70 for
each of the four states (adjacent rates; the order is transitive).

R-level partial days live at ``partial_flow/<rate>/<date>/`` (no ``r-level``
directory); M-level ones at ``partial_flow/m-level/<rate>/<date>/``. Outputs go
to ``partial_flow/mr-level/<rate>/<date>/`` with ``day_stats/<date>.json``,
``generation_summary.csv`` and ``mr_level_metadata.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import generate_m_level_full_flow as mfull
import generate_mr_level_full_flow as mrfull
import generate_r_level_partial_flow as rpartial
import prep_paths

RATES = rpartial.RATES
STATE_FILES = mrfull.STATE_FILES
log = mrfull.log

SUMMARY_FIELDS = [
    "date",
    "penetration_rate",
    "status",
    "skipped",
    "r_input_files",
    "m_input_files",
    "mr_output_files",
    "rows",
    "num_nodes",
    "num_r_nodes",
    "num_m_nodes",
    "header_time_bin_check",
    "nonnegative_integer_check",
    "r_submatrix_check",
    "m_submatrix_check",
    "r_input_conservation_check",
    "m_input_conservation_check",
    "mr_conservation_check",
    "initial_stock_zero_check",
    "partial_le_full_check",
    "partial_gt_full_cells",
    "nested_lower_rate",
    "nested_check",
    "nested_violation_cells",
    "r_total_inflow",
    "r_total_outflow",
    "m_total_inflow",
    "m_total_outflow",
    "r_inflow_ratio",
    "m_inflow_ratio",
    "elapsed_s",
    "graph_signature",
    "error",
]


def r_partial_root(root: Path) -> Path:
    """R-level partial rates sit directly under partial_flow/ in this project."""
    nested = root / "partial_flow" / "r-level"
    return nested if nested.is_dir() else root / "partial_flow"


def violations(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, int]:
    """Cells where left > right, per state."""
    return {name: int((left[name] > right[name]).sum()) for name in STATE_FILES}


def load_full(full_root: Path, date_label: str, layout: mrfull.Layout) -> dict[str, np.ndarray]:
    stats_path = full_root / "day_stats" / f"{date_label}.json"
    if not stats_path.is_file():
        raise RuntimeError(f"full MR stats missing for {date_label}; run generate_mr_level_full_flow.py first")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if stats.get("status") != "PASS" or stats.get("graph_signature") != layout.signature:
        raise RuntimeError(f"full MR {date_label} is not PASS for the current mr_graph.npz")
    values = mrfull.read_level(full_root / date_label, layout.node_ids)
    mrfull.conserves(values, layout.node_ids, f"{date_label} full MR")
    return values


def failed_row(date_label: str, rate: str, dirs: tuple[Path, Path, Path], detail: str) -> dict[str, object]:
    r_dir, m_dir, out_dir = dirs
    return {
        "date": date_label,
        "penetration_rate": rate,
        "status": "FAIL",
        "skipped": "",
        "r_input_files": f"{mrfull.file_count(r_dir)}/{len(STATE_FILES)}",
        "m_input_files": f"{mrfull.file_count(m_dir)}/{len(STATE_FILES)}",
        "mr_output_files": f"{mrfull.file_count(out_dir)}/{len(STATE_FILES)}",
        "partial_le_full_check": "FAIL",
        "nested_check": "FAIL",
        "error": detail,
    }


def rate_dirs(task: dict, rate: str) -> tuple[Path, Path, Path]:
    date_label = task["date"]
    return (
        Path(task["r_root"]) / rate / date_label,
        Path(task["m_root"]) / rate / date_label,
        Path(task["out_root"]) / rate / date_label,
    )


def process_day(task: dict) -> list[dict[str, object]]:
    date_label = task["date"]
    layout: mrfull.Layout = task["layout"]
    out_root = Path(task["out_root"])
    stats_path = out_root / "day_stats" / f"{date_label}.json"
    began = time.time()
    try:
        full = load_full(Path(task["full_root"]), date_label, layout)
    except Exception:
        detail = traceback.format_exc(limit=6).strip().replace("\n", " | ")
        log(f"[{date_label}] FAIL {detail}")
        return [failed_row(date_label, rate, rate_dirs(task, rate), detail) for rate in RATES]
    num_r = layout.num_r
    full_r_in = int(full["inflow.csv"][:, :num_r].sum())
    full_m_in = int(full["inflow.csv"][:, num_r:].sum())
    previous = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
    previous_rows = {row["penetration_rate"]: row for row in previous.get("rows", [])}
    # Per-rate PASS markers live in this file; drop it until the day is rewritten.
    if stats_path.is_file():
        stats_path.unlink()
    rows: list[dict[str, object]] = []
    lower: dict[str, np.ndarray] | None = None
    for index, rate in enumerate(RATES):
        r_dir, m_dir, out_dir = rate_dirs(task, rate)
        label = f"{date_label} {rate}"
        rate_began = time.time()
        try:
            may_skip = task["skip_complete"] and mrfull.previously_passed(previous_rows.get(rate), layout)
            merged, row = mrfull.merge_day(r_dir, m_dir, out_dir, layout, label, may_skip)
            above_full = violations(merged, full)
            if any(above_full.values()):
                raise RuntimeError(f"{label} partial exceeds full MR: {above_full}")
            nested = "PASS"
            nested_cells: dict[str, int] | str = ""
            if index:
                if lower is None:
                    nested = "FAIL"
                    nested_cells = f"{RATES[index - 1]} unavailable"
                else:
                    nested_cells = violations(lower, merged)
                    if any(nested_cells.values()):
                        raise RuntimeError(f"{label} below {RATES[index - 1]}: {nested_cells}")
            lower = merged
            row = {
                "date": date_label,
                "penetration_rate": rate,
                **row,
                "partial_le_full_check": "PASS",
                "partial_gt_full_cells": json.dumps(above_full, separators=(",", ":")),
                "nested_lower_rate": RATES[index - 1] if index else "-",
                "nested_check": nested,
                "nested_violation_cells": json.dumps(nested_cells, separators=(",", ":")) if nested_cells else "",
                "r_inflow_ratio": f"{row['r_total_inflow'] / full_r_in if full_r_in else 0:.8f}",
                "m_inflow_ratio": f"{row['m_total_inflow'] / full_m_in if full_m_in else 0:.8f}",
                "elapsed_s": round(time.time() - rate_began, 1),
            }
            if nested != "PASS":
                row["status"] = "FAIL"
                row["error"] = f"nesting against {RATES[index - 1]} could not be checked"
        except Exception:
            detail = traceback.format_exc(limit=6).strip().replace("\n", " | ")
            log(f"[{label}] FAIL {detail}")
            row = failed_row(date_label, rate, (r_dir, m_dir, out_dir), detail)
            lower = None
        rows.append(row)
    status = "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL"
    mfull.write_json(stats_path, {"date": date_label, "status": status, "graph_signature": layout.signature, "rows": rows})
    log(
        f"[{date_label}] {status} skipped={','.join(str(r.get('skipped', '')) for r in rows)} "
        + " ".join(f"{r['penetration_rate']}={r.get('r_inflow_ratio', '-')}/{r.get('m_inflow_ratio', '-')}" for r in rows)
        + f" elapsed={time.time() - began:.1f}s"
    )
    return rows


def collect_summary(rows: list[dict[str, object]], stats_dir: Path, dates: list[str]) -> list[dict[str, object]]:
    by_date: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_date.setdefault(str(row["date"]), []).append(row)
    for date_label in dates:
        path = stats_dir / f"{date_label}.json"
        if date_label not in by_date and path.is_file():
            previous = json.loads(path.read_text(encoding="utf-8"))
            by_date[date_label] = [{**row, "skipped": "not_run"} for row in previous["rows"]]
    ordered = []
    for date_label in sorted(by_date):
        ordered.extend(sorted(by_date[date_label], key=lambda r: RATES.index(str(r["penetration_rate"]))))
    return ordered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Join seven-rate partial R-level and M-level states into MR-level CSVs")
    prep_paths.add_3s_path_arguments(parser)
    parser.add_argument("--workers", type=int, default=29)
    parser.add_argument("--dates", nargs="*", default=None)
    parser.add_argument("--skip-complete", action="store_true")
    args = parser.parse_args(argv)
    root = prep_paths.refuse_code_root(args.training_root)
    r_root = r_partial_root(root)
    m_root = root / "partial_flow" / "m-level"
    full_root = root / "full_flow" / "mr-level"
    out_root = root / "partial_flow" / "mr-level"
    layout = mrfull.load_layout(root / "graph")
    log(f"graph={layout.graph_path} signature={layout.signature[:12]} nodes={len(layout.node_ids)}")
    log(f"r_partial_root={r_root} m_partial_root={m_root}")
    log(f"full_mr_root={full_root} output_root={out_root}")
    date_sets = {f"R {rate}": mrfull.day_dirs(r_root / rate) for rate in RATES}
    date_sets.update({f"M {rate}": mrfull.day_dirs(m_root / rate) for rate in RATES})
    date_sets["full MR"] = mrfull.day_dirs(full_root)
    all_dates = date_sets["full MR"]
    differing = {name: len(found) for name, found in date_sets.items() if found != all_dates}
    if not all_dates or differing:
        log(f"date directories differ from full MR ({len(all_dates)} days): {differing}")
        return 1
    dates = args.dates or all_dates
    missing = [d for d in dates if d not in all_dates]
    if missing:
        log(f"unknown dates {missing}")
        return 1
    workers = max(1, min(args.workers, len(dates)))
    log(f"dates={len(dates)} rates={','.join(RATES)} workers={workers} skip_complete={args.skip_complete}")
    tasks = [
        {"date": d, "layout": layout, "r_root": str(r_root), "m_root": str(m_root), "full_root": str(full_root),
         "out_root": str(out_root), "skip_complete": args.skip_complete}
        for d in dates
    ]
    rows = mrfull.run_pool(
        tasks, workers, process_day,
        lambda d, detail: [failed_row(d, rate, rate_dirs({**tasks[0], "date": d}, rate), detail) for rate in RATES],
    )
    summary_rows = collect_summary(rows, out_root / "day_stats", all_dates)
    mfull.write_summary(out_root / "generation_summary.csv", SUMMARY_FIELDS, summary_rows)
    mrfull.write_metadata(
        out_root / "mr_level_metadata.json", layout,
        {
            "r-level": f"{r_root}/<rate>/<date>",
            "m-level": f"{m_root}/<rate>/<date>",
            "full_mr_level": f"{full_root}/<date>",
        },
        all_dates,
        {
            "penetration_rates": list(RATES),
            "penetration_definition": (
                "pXX keeps XX% of the vehicles that actually departed that day (sampled_vehicle/, seed 42); "
                "lower rates are prefixes of one shuffled list, so samples are nested. "
                "R and M partial states count only sampled vehicles; MR concatenates them unchanged."
            ),
            "rate_layout": "partial_flow/mr-level/<rate>/<date>/<state>.csv",
        },
    )
    failed = sorted({f"{r['date']} {r['penetration_rate']}" for r in rows if r.get("status") != "PASS"})
    passed = [r for r in summary_rows if r.get("status") == "PASS"]
    expected_rows = len(all_dates) * len(RATES)
    files = sum(mrfull.file_count(out_root / rate / d) for rate in RATES for d in all_dates)
    log(
        f"summary={out_root / 'generation_summary.csv'} run_rows={len(rows)} run_fail={len(failed)} "
        f"failed={','.join(failed) or 'none'} summary_pass={len(passed)}/{expected_rows} "
        f"state_files={files}/{expected_rows * len(STATE_FILES)}"
    )
    for rate in RATES:
        selected = [r for r in summary_rows if r["penetration_rate"] == rate and r.get("status") == "PASS"]
        if selected:
            log(
                f"{rate} days={len(selected)} R inflow={sum(int(r['r_total_inflow']) for r in selected)} "
                f"M inflow={sum(int(r['m_total_inflow']) for r in selected)}"
            )
    ok = not failed and len(passed) == expected_rows and files == expected_rows * len(STATE_FILES)
    if ok:
        log("MR-level partial-flow generation completed successfully.")
    return 0 if ok else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
