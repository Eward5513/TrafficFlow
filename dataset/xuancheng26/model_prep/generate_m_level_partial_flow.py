#!/usr/bin/env python3
"""Build four 5-minute Movement-level partial states for all seven rates.

One worker handles one day and scans its FCD once. Only p70 vehicles are
tracked; each Movement event is added to the vehicle's lowest nested rate and
every higher rate. Event rules, Movement order and bins are the ones in
``generate_m_level_full_flow.py``.

Every cell is checked against the full M-level states: partial inflow,
outflow and accumulation never exceed full, lower rates never exceed higher
rates, and the stock recursion holds without negative values.

Outputs: ``<training-root>/partial_flow/m-level/<rate>/<date>/`` with the four
state files, and ``partial_flow/m-level/generation_summary.csv``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import generate_m_level_full_flow as mfull
import generate_r_level_full_flow as rfull
import generate_r_level_partial_flow as rpartial
import prep_paths

RATES = rpartial.RATES
STATE_FILES = rfull.STATE_FILES
log = mfull.log

SUMMARY_FIELDS = [
    "date",
    "penetration_rate",
    "status",
    "skipped",
    "sampled_vehicle_count",
    "tracked_vehicles_seen",
    "total_inflow",
    "total_outflow",
    "full_total_inflow",
    "full_total_outflow",
    "inflow_ratio",
    "outflow_ratio",
    "active_movements",
    "full_active_movements",
    "via_entries",
    "via_exits",
    "chained_via_exits",
    "direct_transitions",
    "final_exits",
    "mismatched_entries",
    "mismatched_exits",
    "unmapped_transitions",
    "unresolved_vehicles",
    "end_of_day_stock",
    "nested_check",
    "partial_le_full_check",
    "accumulation_le_full_check",
    "conservation_check",
    "rows",
    "movement_count",
    "elapsed_s",
    "graph_signature",
    "error",
]


def blank_rows(date_label: str, detail: str) -> list[dict[str, object]]:
    return [
        {
            "date": date_label,
            "penetration_rate": name,
            "status": "FAIL",
            "nested_check": "FAIL",
            "partial_le_full_check": "FAIL",
            "accumulation_le_full_check": "FAIL",
            "conservation_check": "FAIL",
            "error": detail,
        }
        for name in RATES
    ]


def compare(
    states: list[dict[str, np.ndarray]],
    full_states: dict[str, np.ndarray],
) -> tuple[bool, bool, bool, list[str]]:
    """Cell-wise nested and partial <= full checks for all rates."""
    reasons: list[str] = []
    nested = bounded = stock_bounded = True
    for index, name in enumerate(RATES):
        current = states[index]
        for key in ("inflow.csv", "outflow.csv"):
            if np.any(current[key] > full_states[key]):
                bounded = False
                reasons.append(f"{name} {key} exceeds full")
        for key in ("accumulation_start.csv", "accumulation_end.csv"):
            if np.any(current[key] > full_states[key]):
                stock_bounded = False
                reasons.append(f"{name} {key} exceeds full")
        if index:
            lower = states[index - 1]
            for key in STATE_FILES:
                if np.any(lower[key] > current[key]):
                    nested = False
                    reasons.append(f"{RATES[index - 1]} {key} exceeds {name}")
    for key in ("inflow.csv", "outflow.csv"):
        totals = [int(item[key].sum()) for item in states]
        if any(left >= right for left, right in zip(totals, totals[1:])):
            nested = False
            reasons.append(f"{key} totals are not strictly increasing")
    return nested, bounded, stock_bounded, reasons


def build_rows(
    date_label: str,
    states: list[dict[str, np.ndarray]],
    full_states: dict[str, np.ndarray],
    sampled_counts: dict[str, int],
    counters: list[dict[str, int]],
    checks: tuple[bool, bool, bool],
    signature: str,
    elapsed: float,
    skipped: str,
) -> list[dict[str, object]]:
    full_in = int(full_states["inflow.csv"].sum())
    full_out = int(full_states["outflow.csv"].sum())
    full_active = int((full_states["inflow.csv"].sum(axis=0) > 0).sum())
    nested, bounded, stock_bounded = checks
    rows = []
    for index, name in enumerate(RATES):
        inflow = states[index]["inflow.csv"]
        outflow = states[index]["outflow.csv"]
        total_in, total_out = int(inflow.sum()), int(outflow.sum())
        c = counters[index]
        rows.append({
            "date": date_label,
            "penetration_rate": name,
            "status": "PASS",
            "skipped": skipped,
            "sampled_vehicle_count": sampled_counts[name],
            "tracked_vehicles_seen": c.get("vehicles_seen", ""),
            "total_inflow": total_in,
            "total_outflow": total_out,
            "full_total_inflow": full_in,
            "full_total_outflow": full_out,
            "inflow_ratio": f"{(total_in / full_in) if full_in else 0:.8f}",
            "outflow_ratio": f"{(total_out / full_out) if full_out else 0:.8f}",
            "active_movements": int((inflow.sum(axis=0) > 0).sum()),
            "full_active_movements": full_active,
            "via_entries": c.get("via_entries", ""),
            "via_exits": c.get("via_exits", ""),
            "chained_via_exits": c.get("chained_via_exits", ""),
            "direct_transitions": c.get("direct_transitions", ""),
            "final_exits": c.get("final_exits", ""),
            "mismatched_entries": c.get("mismatched_entries", ""),
            "mismatched_exits": c.get("mismatched_exits", ""),
            "unmapped_transitions": c.get("unmapped_transitions", ""),
            "unresolved_vehicles": c.get("unresolved_vehicles", ""),
            "end_of_day_stock": int(states[index]["accumulation_end.csv"][-1].sum()),
            "nested_check": "PASS" if nested else "FAIL",
            "partial_le_full_check": "PASS" if bounded else "FAIL",
            "accumulation_le_full_check": "PASS" if stock_bounded else "FAIL",
            "conservation_check": "PASS",
            "rows": rfull.N_BINS,
            "movement_count": inflow.shape[1],
            "elapsed_s": round(elapsed, 1),
            "graph_signature": signature,
            "error": "",
        })
    return rows


def load_full_states(full_root: Path, date_label: str, mnet: mfull.MovementNet) -> dict[str, np.ndarray]:
    stats_path = full_root / "day_stats" / f"{date_label}.json"
    if not stats_path.is_file():
        raise RuntimeError(f"full M-level stats missing for {date_label}; run generate_m_level_full_flow.py first")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if stats.get("status") != "PASS" or stats.get("graph_signature") != mnet.signature:
        raise RuntimeError(f"full M-level {date_label} is not PASS for the current m_graph.npz")
    values = mfull.read_states(full_root / date_label, mnet.movement_ids)
    if values is None:
        raise RuntimeError(f"full M-level files for {date_label} are missing or fail conservation")
    return values


def process_day(task: dict) -> list[dict[str, object]]:
    date_label = task["date"]
    mnet: mfull.MovementNet = task["mnet"]
    ids = mnet.movement_ids
    out_root = Path(task["out_root"])
    stats_path = out_root / "day_stats" / f"{date_label}.json"
    rpartial.SAMPLED_ROOT = Path(task["sampled_root"])
    rfull.READ_BYTES = int(task["read_bytes"])
    began = time.time()
    try:
        level_of, sampled_counts = rpartial.load_sampled(date_label)
        full_states = load_full_states(Path(task["full_root"]), date_label, mnet)
        if task["skip_complete"] and stats_path.is_file():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            if stats.get("status") == "PASS" and stats.get("graph_signature") == mnet.signature:
                states = [mfull.read_states(out_root / name / date_label, ids) for name in RATES]
                if all(item is not None for item in states):
                    nested, bounded, stock_bounded, reasons = compare(states, full_states)
                    if not reasons:
                        log(f"[{date_label}] SKIP verified seven-rate files")
                        return build_rows(
                            date_label, states, full_states, sampled_counts, stats["counters"],
                            (nested, bounded, stock_bounded), mnet.signature, time.time() - began, "yes",
                        )
        if stats_path.is_file():
            stats_path.unlink()
        day_dir = Path(task["fcd_root"]) / date_label
        fcd_path = rfull.prefer_xml(day_dir, f"fcd_{date_label}")
        trip_path = rfull.prefer_xml(day_dir, f"tripinfo_{date_label}")
        log(f"[{date_label}] start fcd={fcd_path} tracked={len(level_of)} out={out_root}")
        arrivals, running = mfull.load_arrival_lanes(trip_path, mnet.lane_to_edge_index)
        tracker = mfull.stream_movements(
            fcd_path, mnet, arrivals, running, date_label,
            level_of=level_of, n_levels=len(RATES), keep_anomalies=False,
        )
        problems = []
        states = []
        for index, name in enumerate(RATES):
            inflow, outflow = tracker.inflow[index], tracker.outflow[index]
            if int(inflow.min()) < 0 or int(outflow.min()) < 0:
                problems.append(f"{name} negative flow")
            problems.extend(f"{name} {item}" for item in tracker.identities(index))
            start, end = rfull.accumulate_states(inflow, outflow, ids, f"{date_label} {name}")
            states.append({
                "inflow.csv": inflow,
                "outflow.csv": outflow,
                "accumulation_start.csv": start,
                "accumulation_end.csv": end,
            })
        seen = tracker.counts["vehicles_seen"]
        for index, name in enumerate(RATES):
            if int(seen[index]) > sampled_counts[name]:
                problems.append(f"{name} tracked {int(seen[index])} vehicles but sampled {sampled_counts[name]}")
        nested, bounded, stock_bounded, reasons = compare(states, full_states)
        problems.extend(reasons)
        if problems:
            raise RuntimeError("; ".join(problems))
        for index, name in enumerate(RATES):
            rfull.write_state_matrices(
                out_root / name / date_label, tracker.inflow[index], tracker.outflow[index], ids, f"{date_label} {name}"
            )
        counters = [
            {key: int(value[index]) for key, value in tracker.counts.items()} for index in range(len(RATES))
        ]
        rows = build_rows(
            date_label, states, full_states, sampled_counts, counters,
            (nested, bounded, stock_bounded), mnet.signature, time.time() - began, "no",
        )
        mfull.write_json(stats_path, {
            "date": date_label,
            "status": "PASS",
            "graph_signature": mnet.signature,
            "counters": counters,
            "rows": rows,
        })
        log(
            f"[{date_label}] PASS "
            + " ".join(f"{row['penetration_rate']}={row['total_inflow']}/{row['total_outflow']}" for row in rows)
            + f" elapsed={rows[0]['elapsed_s']}s"
        )
        return rows
    except Exception:
        detail = traceback.format_exc(limit=8).strip().replace("\n", " | ")
        log(f"[{date_label}] FAIL {detail}")
        return blank_rows(date_label, detail)


def collect_summary(rows: list[dict[str, object]], stats_dir: Path, dates: list[str]) -> list[dict[str, object]]:
    by_date: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_date.setdefault(str(row["date"]), []).append(row)
    for date_label in dates:
        if date_label in by_date:
            continue
        path = stats_dir / f"{date_label}.json"
        if path.is_file():
            previous = json.loads(path.read_text(encoding="utf-8"))
            by_date[date_label] = [{**row, "skipped": "not_run"} for row in previous["rows"]]
    ordered = []
    for date_label in sorted(by_date):
        ordered.extend(sorted(by_date[date_label], key=lambda r: RATES.index(str(r["penetration_rate"]))))
    return ordered


def print_overall(summary_rows: list[dict[str, object]], out_root: Path, dates: list[str]) -> bool:
    failed_dates = sorted({str(r["date"]) for r in summary_rows if r.get("status") != "PASS"})
    covered = sorted({str(r["date"]) for r in summary_rows})
    state_files = [
        path
        for rate in RATES
        for date_label in dates
        for name in STATE_FILES
        if (path := out_root / rate / date_label / name).is_file()
    ]
    nested_bad = [r for r in summary_rows if r.get("nested_check") != "PASS"]
    bound_bad = [r for r in summary_rows if r.get("partial_le_full_check") != "PASS"]
    stock_bad = [r for r in summary_rows if r.get("accumulation_le_full_check") != "PASS"]
    log(
        f"dates_in_summary={len(covered)}/{len(dates)} failed_dates={','.join(failed_dates) if failed_dates else 'none'} "
        f"state_files={len(state_files)}/{len(dates) * len(RATES) * len(STATE_FILES)}"
    )
    log(f"nested_failures={len(nested_bad)} partial_le_full_failures={len(bound_bad)} accumulation_le_full_failures={len(stock_bad)}")
    for name in RATES:
        selected = [r for r in summary_rows if r.get("penetration_rate") == name and r.get("status") == "PASS"]
        if not selected:
            log(f"{name} no passing rows")
            continue
        vehicles = sum(int(r["sampled_vehicle_count"]) for r in selected)
        inflow = sum(int(r["total_inflow"]) for r in selected)
        outflow = sum(int(r["total_outflow"]) for r in selected)
        full_in = sum(int(r["full_total_inflow"]) for r in selected)
        full_out = sum(int(r["full_total_outflow"]) for r in selected)
        log(
            f"{name} days={len(selected)} vehicles={vehicles} inflow={inflow} outflow={outflow} "
            f"inflow_ratio={inflow / full_in:.6f} outflow_ratio={outflow / full_out:.6f}"
        )
    ok = (
        len(covered) == len(dates)
        and not failed_dates
        and not nested_bad
        and not bound_bad
        and not stock_bad
        and len(state_files) == len(dates) * len(RATES) * len(STATE_FILES)
    )
    if ok:
        log("M-level partial-flow generation completed successfully.")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate seven-rate partial M-level four-state flow from FCD")
    prep_paths.add_3s_path_arguments(parser, fcd=True, net=True)
    parser.add_argument("--workers", type=int, default=29)
    parser.add_argument("--dates", nargs="*", default=None)
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--read-buffer-gib", type=float, default=mfull.DEFAULT_READ_GIB)
    args = parser.parse_args(argv)
    root = prep_paths.refuse_code_root(args.training_root)
    fcd_root = prep_paths.resolved(args.fcd_root)
    net_file = prep_paths.resolved(args.net_file)
    graph_path = root / "graph" / "m_graph.npz"
    full_root = root / "full_flow" / "m-level"
    out_root = root / "partial_flow" / "m-level"
    sampled_root = root / "sampled_vehicle"
    summary_path = out_root / "generation_summary.csv"
    log(f"fcd_root={fcd_root}")
    log(f"net_file={net_file}")
    log(f"graph={graph_path}")
    log(f"full_root={full_root}")
    log(f"sampled_root={sampled_root}")
    log(f"output_root={out_root}")
    mfull.self_test()
    all_dates = mfull.discover_dates(fcd_root)
    dates = args.dates or all_dates
    missing = [d for d in dates if d not in all_dates]
    if not dates or missing:
        log(f"no usable dates; missing={missing}")
        return 1
    rfull.confirm_fcd_starts_at_zero(fcd_root, dates)
    mnet = mfull.load_movement_net(net_file, graph_path)
    workers = max(1, min(args.workers, len(dates)))
    read_bytes = int(args.read_buffer_gib * (1 << 30))
    log(
        f"movements={len(mnet.movement_ids)} rates={','.join(RATES)} dates={len(dates)} workers={workers} "
        f"read_buffer_GiB={args.read_buffer_gib:g} signature={mnet.signature[:12]}"
    )
    tasks = [
        {
            "date": date_label,
            "mnet": mnet,
            "fcd_root": str(fcd_root),
            "full_root": str(full_root),
            "out_root": str(out_root),
            "sampled_root": str(sampled_root),
            "skip_complete": args.skip_complete,
            "read_bytes": read_bytes,
        }
        for date_label in dates
    ]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
        futures = {pool.submit(process_day, task): task["date"] for task in tasks}
        for future in as_completed(futures):
            date_label = futures[future]
            try:
                rows.extend(future.result())
            except Exception:
                detail = traceback.format_exc(limit=6).strip().replace("\n", " | ")
                log(f"[{date_label}] FAIL {detail}")
                rows.extend(blank_rows(date_label, detail))
    summary_rows = collect_summary(rows, out_root / "day_stats", all_dates)
    mfull.write_summary(summary_path, SUMMARY_FIELDS, summary_rows)
    log(f"summary={summary_path}")
    print_overall(summary_rows, out_root, all_dates)
    return 0 if all(row.get("status") == "PASS" for row in rows) else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
