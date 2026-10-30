#!/usr/bin/env python3
"""Build four 5-minute partial states from sampled vehicle IDs.

One worker handles one day. ``--migrate-legacy`` reuses the audited
``inflow.csv`` and ``out_flow.csv`` instead of scanning FCD. Outputs use the
same names as the full states: ``inflow.csv``, ``outflow.csv``,
``accumulation_start.csv`` and ``accumulation_end.csv``.
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
import prep_paths

FCD_ROOT = full.FCD_ROOT
NET_FILE = full.NET_FILE
FULL_ROOT = full.OUT_ROOT
SAMPLED_ROOT = prep_paths.DEFAULT_TRAINING_ROOT / "sampled_vehicle"
OUT_ROOT = prep_paths.DEFAULT_TRAINING_ROOT / "partial_flow"
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
    matches = sorted(FULL_ROOT.glob("2023_04_*/inflow.csv"))
    if not matches:
        matches = sorted(FULL_ROOT.glob("2023_04_*/entry_flow.csv"))
    sample = matches[0]
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


def full_matrix(date_label: str, kind: str, edges: list[str]) -> np.ndarray:
    directory = FULL_ROOT / date_label
    path = directory / f"{kind}.csv"
    if not path.is_file():
        legacy = {"inflow": "entry_flow.csv", "outflow": "exit_flow.csv"}[kind]
        path = directory / legacy
    return full.read_flow_csv(path, edges)


def process_day(task: tuple) -> list[dict[str, object]]:
    date_label, edges, lane_to_edge, internal_lanes, migrate_legacy, skip_complete, remove_legacy = task
    try:
        log(f"[{date_label}] start out={OUT_ROOT} migrate={migrate_legacy}")
        level_of, sampled_counts = load_sampled(date_label)
        if migrate_legacy:
            inflows = []
            outflows = []
            for name in RATES:
                day_dir = OUT_ROOT / name / date_label
                inflows.append(full.read_flow_csv(day_dir / "inflow.csv", edges))
                legacy_out = day_dir / "out_flow.csv"
                source_out = legacy_out if legacy_out.is_file() else day_dir / "outflow.csv"
                outflows.append(full.read_flow_csv(source_out, edges))
                if legacy_out.is_file():
                    audited = full.read_flow_csv(legacy_out, edges)
                    if not np.array_equal(outflows[-1], audited):
                        raise RuntimeError(f"{date_label} {name} outflow differs from audited out_flow.csv")
        else:
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
        full_in = full_matrix(date_label, "inflow", edges)
        full_out = full_matrix(date_label, "outflow", edges)
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
            label = f"{date_label} {name}"
            if skip_complete and full.states_match(out_dir, inflows[index], outflows[index], edges, label):
                log(f"[{label}] SKIP verified four-state files")
            else:
                preserve_inflow = migrate_legacy and (out_dir / "inflow.csv").is_file()
                full.write_state_matrices(
                    out_dir,
                    inflows[index],
                    outflows[index],
                    edges,
                    label,
                    rewrite_inflow=not preserve_inflow,
                )
            if remove_legacy:
                legacy = out_dir / "out_flow.csv"
                if legacy.is_file():
                    legacy.unlink()
            for state_name in full.STATE_FILES:
                path = out_dir / state_name
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
    state_files = [
        path
        for rate in RATES
        for day_dir in (OUT_ROOT / rate).glob("2023_04_*")
        for name in full.STATE_FILES
        if (path := day_dir / name).is_file()
    ]
    log(f"dates={len(by_date)} pass={len(passed_dates)} fail={len(failed_dates)} state_files={len(state_files)}")
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
        and len(state_files) == 29 * len(RATES) * len(full.STATE_FILES)
    ):
        log("R-level partial-flow generation completed successfully.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate partial R-level inflow and outflow")
    prep_paths.add_path_arguments(parser, fcd=True, net=True)
    parser.add_argument("--workers", type=int, default=29)
    parser.add_argument("--dates", nargs="*", default=None)
    parser.add_argument("--migrate-legacy", action="store_true")
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--remove-legacy", action="store_true")
    args = parser.parse_args(argv)
    root = prep_paths.resolved(args.training_root)
    fcd_root = prep_paths.resolved(args.fcd_root)
    net_file = prep_paths.resolved(args.net_file)
    full_root = root / "full_flow" / "r-level"
    out_root = root / "partial_flow"
    prep_paths.bind(
        sys.modules[__name__],
        FCD_ROOT=fcd_root,
        NET_FILE=net_file,
        FULL_ROOT=full_root,
        SAMPLED_ROOT=root / "sampled_vehicle",
        OUT_ROOT=out_root,
        SUMMARY_PATH=(
            out_root / "state_migration_summary.csv" if args.migrate_legacy else out_root / "generation_summary.csv"
        ),
    )
    prep_paths.bind(full, FCD_ROOT=fcd_root, NET_FILE=net_file, OUT_ROOT=full_root)
    log(f"fcd_root={FCD_ROOT}")
    log(f"net_file={NET_FILE}")
    log(f"full_root={FULL_ROOT}")
    log(f"output_root={OUT_ROOT}")
    prep_paths.publish_runtime(fcd_root=FCD_ROOT, net_file=NET_FILE, training_root=root)
    full.self_test()
    dates = args.dates or discover_dates()
    if not dates:
        log("no date directories")
        return 1
    full.confirm_fcd_starts_at_zero(fcd_root, dates)
    network_edges, lane_to_edge, internal_lanes = full.load_network(NET_FILE)
    edges = reference_edges()
    if edges != network_edges:
        raise RuntimeError("full-flow edge order does not match the network edge order")
    workers = max(1, min(args.workers, len(dates)))
    log(f"edges={len(edges)} dates={len(dates)} workers={workers} read_buffer_GiB={full.READ_BYTES / 1024**3:.0f}")
    tasks = [
        (date_label, edges, lane_to_edge, internal_lanes, args.migrate_legacy, args.skip_complete, args.remove_legacy)
        for date_label in dates
    ]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        max_tasks_per_child=1,
        initializer=prep_paths.init_partial_flow_worker,
    ) as pool:
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
