#!/usr/bin/env python3
"""Build 5-minute partial inflow and outflow from sampled vehicle IDs.

One worker handles one day, scans that day's FCD once, and updates every
penetration rate from the same edge events used by the full-flow generator.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

import generate_r_level_full_flow as full

FCD_ROOT = full.FCD_ROOT
NET_FILE = full.NET_FILE
FULL_ROOT = full.OUT_ROOT
SAMPLED_ROOT = Path(
    "/data5/tzhang174/TrafficFlow/xuancheng26/training_data/sampled_vehicle"
)
OUT_ROOT = Path(
    "/data5/tzhang174/TrafficFlow/xuancheng26/training_data/partial_flow"
)
SUMMARY_PATH = OUT_ROOT / "generation_summary.csv"
RATES = ("p05", "p10", "p20", "p30", "p40", "p50", "p70")
SUMMARY_FIELDS = [
    "date",
    "penetration_rate",
    "sampled_vehicle_count",
    "total_inflow",
    "total_outflow",
    "full_total_inflow",
    "full_total_outflow",
    "inflow_ratio",
    "outflow_ratio",
    "nested_check",
    "partial_le_full_check",
    "rows",
    "edge_count",
    "status",
]


def log(message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"{stamp} {message}", flush=True)


def discover_dates() -> list[str]:
    return full.discover_dates()


def reference_edges() -> list[str]:
    sample = next(FULL_ROOT.glob("2023_04_*/entry_flow.csv"))
    with sample.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split(",")
    if header[:2] != ["time_bin", "start_time"]:
        raise RuntimeError(f"unexpected full-flow header in {sample}")
    return header[2:]


def load_sampled(date_label: str) -> tuple[dict[str, int], dict[str, int]]:
    """Return vehicle -> lowest rate index, and sampled counts per rate."""
    level_of: dict[str, int] = {}
    counts: dict[str, int] = {}
    sets: list[set[str]] = []
    for index, name in enumerate(RATES):
        path = SAMPLED_ROOT / name / f"{date_label}.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if header != ["vehicle_id"]:
                raise RuntimeError(f"unexpected sampled header in {path}")
            chosen = [row[0] for row in reader if row]
        if len(chosen) != len(set(chosen)):
            raise RuntimeError(f"duplicate sampled ids in {path}")
        chosen_set = set(chosen)
        sets.append(chosen_set)
        counts[name] = len(chosen)
        for vehicle_id in chosen:
            level_of.setdefault(vehicle_id, index)
    for low, high in zip(sets, sets[1:]):
        if not low.issubset(high):
            raise RuntimeError("sampled vehicle files are not nested")
    return level_of, counts


def load_full_matrix(path: Path, edges: list[str]) -> np.ndarray:
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split(",")
    if header != ["time_bin", "start_time", *edges]:
        raise RuntimeError(f"full-flow columns differ from the reference schema: {path}")
    matrix = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        dtype=np.int64,
        usecols=range(2, 2 + len(edges)),
    )
    if matrix.shape != (full.N_BINS, len(edges)):
        raise RuntimeError(f"full-flow shape {matrix.shape} in {path}")
    return matrix


def blank_rows(date_label: str, edges: list[str], status: str = "FAIL") -> list[dict[str, object]]:
    rows = []
    for name in RATES:
        rows.append(
            {
                "date": date_label,
                "penetration_rate": name,
                "sampled_vehicle_count": "",
                "total_inflow": "",
                "total_outflow": "",
                "full_total_inflow": "",
                "full_total_outflow": "",
                "inflow_ratio": "",
                "outflow_ratio": "",
                "nested_check": "FAIL",
                "partial_le_full_check": "FAIL",
                "rows": "",
                "edge_count": len(edges),
                "status": status,
            }
        )
    return rows


def process_day(task: tuple) -> list[dict[str, object]]:
    date_label, edges, lane_to_edge, internal_lanes = task
    try:
        log(f"[{date_label}] start")
        level_of, sampled_counts = load_sampled(date_label)
        p70 = {vehicle_id for vehicle_id, level in level_of.items() if level <= RATES.index("p70")}
        inflows = [np.zeros((full.N_BINS, len(edges)), dtype=np.int64) for _ in RATES[:-1]]
        outflows = [np.zeros((full.N_BINS, len(edges)), dtype=np.int64) for _ in RATES[:-1]]

        def on_event(kind: str, slot: int, column: int, vehicle_id: str) -> None:
            # stream_fcd writes the p70 matrix itself. Mirror the same event
            # into every lower nested rate that contains this vehicle.
            start = level_of[vehicle_id]
            target = inflows if kind == "entry" else outflows
            for index in range(start, len(RATES) - 1):
                target[index][slot, column] += 1

        fcd_path = full.prefer_xml(FCD_ROOT / date_label, f"fcd_{date_label}")
        trip_path = full.prefer_xml(FCD_ROOT / date_label, f"tripinfo_{date_label}")
        arrivals = full.load_arrivals(trip_path)
        p70_in, p70_out, _counts, _seen = full.stream_fcd(
            fcd_path,
            edges,
            lane_to_edge,
            internal_lanes,
            arrivals,
            date_label,
            include=p70,
            on_event=on_event,
        )
        inflows.append(p70_in)
        outflows.append(p70_out)
        full_in = load_full_matrix(FULL_ROOT / date_label / "entry_flow.csv", edges)
        full_out = load_full_matrix(FULL_ROOT / date_label / "exit_flow.csv", edges)
        full_in_total = int(full_in.sum())
        full_out_total = int(full_out.sum())
        reasons: list[str] = []
        nested_ok = True
        bounded_ok = True
        inflow_totals = []
        outflow_totals = []
        for index, name in enumerate(RATES):
            if int(inflows[index].min()) < 0 or int(outflows[index].min()) < 0:
                reasons.append(f"{name} has a negative count")
            if np.any(inflows[index] > full_in) or np.any(outflows[index] > full_out):
                bounded_ok = False
                reasons.append(f"{name} exceeds full flow")
            if index:
                if np.any(inflows[index - 1] > inflows[index]) or np.any(outflows[index - 1] > outflows[index]):
                    nested_ok = False
                    reasons.append(f"{RATES[index - 1]} exceeds {name}")
            inflow_totals.append(int(inflows[index].sum()))
            outflow_totals.append(int(outflows[index].sum()))
        if any(left >= right for left, right in zip(inflow_totals, inflow_totals[1:])):
            nested_ok = False
            reasons.append("inflow totals are not strictly increasing")
        if any(left >= right for left, right in zip(outflow_totals, outflow_totals[1:])):
            nested_ok = False
            reasons.append("outflow totals are not strictly increasing")
        if reasons:
            raise RuntimeError("; ".join(reasons))
        for index, name in enumerate(RATES):
            out_dir = OUT_ROOT / name / date_label
            full.write_matrix(out_dir / "inflow.csv", inflows[index], edges)
            full.write_matrix(out_dir / "out_flow.csv", outflows[index], edges)
            full.validate_csv(out_dir / "inflow.csv", edges)
            full.validate_csv(out_dir / "out_flow.csv", edges)
            for path in (out_dir / "inflow.csv", out_dir / "out_flow.csv"):
                if path.stat().st_size <= 0:
                    raise RuntimeError(f"empty output {path}")
        rows = []
        for index, name in enumerate(RATES):
            inflow_total = inflow_totals[index]
            outflow_total = outflow_totals[index]
            rows.append(
                {
                    "date": date_label,
                    "penetration_rate": name,
                    "sampled_vehicle_count": sampled_counts[name],
                    "total_inflow": inflow_total,
                    "total_outflow": outflow_total,
                    "full_total_inflow": full_in_total,
                    "full_total_outflow": full_out_total,
                    "inflow_ratio": f"{(inflow_total / full_in_total) if full_in_total else 0:.8f}",
                    "outflow_ratio": f"{(outflow_total / full_out_total) if full_out_total else 0:.8f}",
                    "nested_check": "PASS" if nested_ok else "FAIL",
                    "partial_le_full_check": "PASS" if bounded_ok else "FAIL",
                    "rows": full.N_BINS,
                    "edge_count": len(edges),
                    "status": "PASS",
                }
            )
        log(
            f"[{date_label}] PASS "
            + " ".join(f"{name}={inflow_totals[i]}/{outflow_totals[i]}" for i, name in enumerate(RATES))
        )
        return rows
    except Exception:
        detail = traceback.format_exc(limit=8).strip().replace("\n", " | ")
        log(f"[{date_label}] FAIL {detail}")
        return blank_rows(date_label, edges)


def write_summary(rows: list[dict[str, object]]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = SUMMARY_PATH.with_suffix(SUMMARY_PATH.suffix + ".tmp")
    ordered = sorted(rows, key=lambda item: (str(item["date"]), RATES.index(str(item["penetration_rate"]))))
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in ordered:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})
    os.replace(temporary, SUMMARY_PATH)


def print_overall(rows: list[dict[str, object]]) -> None:
    by_date: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_date.setdefault(str(row["date"]), []).append(row)
    failed_dates = [date_label for date_label, items in by_date.items() if any(item.get("status") != "PASS" for item in items)]
    passed_dates = [date_label for date_label in by_date if date_label not in failed_dates]
    files = list(OUT_ROOT.glob("p*/2023_04_*/*.csv"))
    log(f"dates={len(by_date)} pass={len(passed_dates)} fail={len(failed_dates)} files={len(files)}")
    nested_bad = [row for row in rows if row.get("nested_check") != "PASS"]
    bound_bad = [row for row in rows if row.get("partial_le_full_check") != "PASS"]
    log(f"nested_check_failures={len(nested_bad)} partial_le_full_failures={len(bound_bad)}")
    for name in RATES:
        selected = [row for row in rows if row.get("penetration_rate") == name and row.get("status") == "PASS"]
        if not selected:
            log(f"{name} no passing rows")
            continue
        vehicles = sum(int(row["sampled_vehicle_count"]) for row in selected)
        inflow = sum(int(row["total_inflow"]) for row in selected)
        outflow = sum(int(row["total_outflow"]) for row in selected)
        full_in = sum(int(row["full_total_inflow"]) for row in selected)
        full_out = sum(int(row["full_total_outflow"]) for row in selected)
        log(
            f"{name} vehicles={vehicles} inflow={inflow} outflow={outflow} "
            f"inflow_ratio={inflow / full_in:.6f} outflow_ratio={outflow / full_out:.6f}"
        )
    log(f"failed={','.join(sorted(failed_dates)) if failed_dates else 'none'}")
    if (
        len(by_date) == 29
        and not failed_dates
        and not nested_bad
        and not bound_bad
        and len(files) == 406
    ):
        log("R-level partial-flow generation completed successfully.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate partial R-level inflow and outflow")
    parser.add_argument("--workers", type=int, default=29)
    parser.add_argument("--dates", nargs="*", default=None)
    args = parser.parse_args(argv)
    full.self_test()
    dates = args.dates or discover_dates()
    network_edges, lane_to_edge, internal_lanes = full.load_network(NET_FILE)
    edges = reference_edges()
    if edges != network_edges:
        raise RuntimeError("full-flow edge order does not match the network edge order")
    workers = max(1, min(args.workers, len(dates)))
    log(f"edges={len(edges)} dates={len(dates)} workers={workers} read_buffer_GiB={full.READ_BYTES / 1024**3:.0f}")
    tasks = [(date_label, edges, lane_to_edge, internal_lanes) for date_label in dates]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
        futures = {pool.submit(process_day, task): task[0] for task in tasks}
        for future in as_completed(futures):
            date_label = futures[future]
            try:
                rows.extend(future.result())
            except Exception:
                detail = traceback.format_exc(limit=6).strip().replace("\n", " | ")
                log(f"[{date_label}] FAIL {detail}")
                rows.extend(blank_rows(date_label, edges))
    write_summary(rows)
    print_overall(rows)
    return 0 if all(row.get("status") == "PASS" for row in rows) else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
