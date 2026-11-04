#!/usr/bin/env python3
"""Join full R-level and M-level four-state CSVs into MR-level CSVs.

No FCD is read and no value is recounted. For every day and state the R-level
matrix (1406 road columns) and the M-level matrix (3877 Movement columns) are
placed side by side in ``mr_graph.npz`` node order:

    X_MR[t] = [X_R[t], X_M[t]]

Columns are named by the MR ``node_ids`` (``R::<edge_id>`` then the Movement
ids). R columns keep road stock, M columns keep connector stock; nothing is
summed across node types. Conservation is checked per node only:

    N_end[t] = N_start[t] + inflow[t] - outflow[t],  N_start[t+1] = N_end[t],
    N_start[0] = 0

Outputs: ``<training-root>/full_flow/mr-level/<date>/{inflow,outflow,
accumulation_start,accumulation_end}.csv``, ``day_stats/<date>.json``,
``generation_summary.csv`` and ``mr_level_metadata.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

import generate_m_level_full_flow as mfull
import generate_mr_graph as mrgraph
import generate_r_level_full_flow as rfull
import prep_paths

N_BINS = rfull.N_BINS
BIN_SECONDS = rfull.BIN_SECONDS
STATE_FILES = rfull.STATE_FILES
log = mfull.log

SUMMARY_FIELDS = [
    "date",
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
    "r_total_inflow",
    "r_total_outflow",
    "m_total_inflow",
    "m_total_outflow",
    "r_end_of_day_stock",
    "m_end_of_day_stock",
    "elapsed_s",
    "graph_signature",
    "error",
]

STATE_MEANINGS = {
    "R": {
        "inflow": "vehicles entering the road (real SUMO edge) in the 5-minute bin",
        "outflow": "vehicles leaving the road in the bin",
        "accumulation_start": "vehicles on the road at the start of the bin",
        "accumulation_end": "vehicles on the road at the end of the bin",
    },
    "M": {
        "inflow": "vehicles entering the Movement connector (its via lanes) in the bin",
        "outflow": "vehicles leaving the connector onto the outgoing road in the bin",
        "accumulation_start": "vehicles inside the connector at the start of the bin",
        "accumulation_end": "vehicles inside the connector at the end of the bin",
    },
}


@dataclass
class Layout:
    node_ids: list[str]
    r_ids: list[str]
    m_ids: list[str]
    signature: str
    graph_path: str

    @property
    def num_r(self) -> int:
        return len(self.r_ids)

    @property
    def num_m(self) -> int:
        return len(self.m_ids)


def load_layout(graph_dir: Path) -> Layout:
    """MR column order from mr_graph.npz, checked against the current R and M graphs."""
    path = graph_dir / "mr_graph.npz"
    audit_path = graph_dir / "mr_graph_audit.json"
    if not path.is_file() or not audit_path.is_file():
        raise FileNotFoundError(f"missing {path} or its audit; run generate_mr_graph.py first")
    if json.loads(audit_path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError(f"{audit_path} is not PASS")
    with np.load(path, allow_pickle=False) as payload:
        node_ids = [str(v) for v in payload["node_ids"].tolist()]
        types = payload["node_types"]
        num_r, num_m = int(payload["num_r_nodes"]), int(payload["num_m_nodes"])
        r_ids = [str(v) for v in payload["r_edge_ids"].tolist()]
        m_ids = [str(v) for v in payload["movement_ids"].tolist()]
        signature = str(payload["graph_signature"])
        hashes = {"r_graph.npz": str(payload["r_graph_sha256"]), "m_graph.npz": str(payload["m_graph_sha256"])}
    for name, digest in hashes.items():
        if mrgraph.sha256_file(graph_dir / name) != digest:
            raise RuntimeError(f"{name} changed after mr_graph.npz was built; rerun generate_mr_graph.py --force")
    with np.load(graph_dir / "r_graph.npz", allow_pickle=False) as payload:
        r_graph_ids = [str(v) for v in payload["edge_ids"].tolist()]
    with np.load(graph_dir / "m_graph.npz", allow_pickle=False) as payload:
        m_graph_ids = [str(v) for v in payload["movement_ids"].tolist()]
    if r_ids != r_graph_ids or m_ids != m_graph_ids:
        raise RuntimeError("mr_graph.npz R/M ids differ from r_graph.npz / m_graph.npz")
    if len(node_ids) != num_r + num_m or node_ids != [f"R::{e}" for e in r_ids] + m_ids:
        raise RuntimeError("mr_graph.npz node_ids are not R::<edge_id> followed by movement ids")
    if not (np.array_equal(types[:num_r], np.zeros(num_r)) and np.array_equal(types[num_r:], np.ones(num_m))):
        raise RuntimeError("mr_graph.npz node_types are not R then M")
    return Layout(node_ids, r_ids, m_ids, signature, str(path))


def file_count(directory: Path) -> int:
    return sum(1 for name in STATE_FILES if (directory / name).is_file())


def read_level(directory: Path, ids: list[str]) -> dict[str, np.ndarray]:
    """Four states with exact header, 0..287 bins, non-negative integers."""
    missing = [name for name in STATE_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{directory} missing {', '.join(missing)}")
    return {name: rfull.read_flow_csv(directory / name, ids) for name in STATE_FILES}


def conserves(values: dict[str, np.ndarray], ids: list[str], label: str) -> None:
    """Per-node stock recursion, continuity, zero initial stock, no negatives."""
    start, end = rfull.accumulate_states(values["inflow.csv"], values["outflow.csv"], ids, label)
    if not np.array_equal(start, values["accumulation_start.csv"]):
        raise RuntimeError(f"{label} accumulation_start differs from the inflow/outflow recursion")
    if not np.array_equal(end, values["accumulation_end.csv"]):
        raise RuntimeError(f"{label} accumulation_end differs from the inflow/outflow recursion")


def merge(r_values: dict[str, np.ndarray], m_values: dict[str, np.ndarray], layout: Layout) -> dict[str, np.ndarray]:
    merged = {}
    for name in STATE_FILES:
        r_part, m_part = r_values[name], m_values[name]
        if r_part.shape != (N_BINS, layout.num_r) or m_part.shape != (N_BINS, layout.num_m):
            raise RuntimeError(f"{name} shapes R={r_part.shape} M={m_part.shape}")
        merged[name] = np.concatenate([r_part, m_part], axis=1)
    return merged


def submatrices_match(
    values: dict[str, np.ndarray], r_values: dict[str, np.ndarray], m_values: dict[str, np.ndarray], layout: Layout,
) -> tuple[bool, bool]:
    r_ok = all(np.array_equal(values[n][:, :layout.num_r], r_values[n]) for n in STATE_FILES)
    m_ok = all(np.array_equal(values[n][:, layout.num_r:], m_values[n]) for n in STATE_FILES)
    return r_ok, m_ok


def existing_matches(
    out_dir: Path, r_values: dict[str, np.ndarray], m_values: dict[str, np.ndarray], layout: Layout,
) -> bool:
    try:
        values = read_level(out_dir, layout.node_ids)
    except Exception:
        return False
    r_ok, m_ok = submatrices_match(values, r_values, m_values, layout)
    return r_ok and m_ok and all(values[n].shape == (N_BINS, len(layout.node_ids)) for n in STATE_FILES)


def write_level(
    out_dir: Path, merged: dict[str, np.ndarray], r_values: dict[str, np.ndarray], m_values: dict[str, np.ndarray],
    layout: Layout, label: str,
) -> None:
    """Atomic write per file, then strict reread and cell-wise R/M comparison."""
    for name in STATE_FILES:
        rfull.write_matrix(out_dir / name, merged[name], layout.node_ids)
    for name in STATE_FILES:
        rfull.validate_csv(out_dir / name, layout.node_ids)
    reread = read_level(out_dir, layout.node_ids)
    r_ok, m_ok = submatrices_match(reread, r_values, m_values, layout)
    if not (r_ok and m_ok):
        raise RuntimeError(f"{label} reread MR files differ from the R ({r_ok}) / M ({m_ok}) inputs")
    conserves(reread, layout.node_ids, f"{label} reread")


def previously_passed(row: dict | None, layout: Layout) -> bool:
    return bool(row) and row.get("status") == "PASS" and row.get("graph_signature") == layout.signature


def merge_day(
    r_dir: Path, m_dir: Path, out_dir: Path, layout: Layout, label: str, may_skip: bool,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Read, check, merge and write one day/rate. ``may_skip`` keeps output that rereads identical."""
    r_values = read_level(r_dir, layout.r_ids)
    m_values = read_level(m_dir, layout.m_ids)
    conserves(r_values, layout.r_ids, f"{label} R input")
    conserves(m_values, layout.m_ids, f"{label} M input")
    merged = merge(r_values, m_values, layout)
    conserves(merged, layout.node_ids, f"{label} MR")
    if may_skip and existing_matches(out_dir, r_values, m_values, layout):
        skipped = "yes"
    else:
        skipped = "no"
        write_level(out_dir, merged, r_values, m_values, layout, label)
    num_r = layout.num_r
    row = {
        "status": "PASS",
        "skipped": skipped,
        "r_input_files": f"{file_count(r_dir)}/{len(STATE_FILES)}",
        "m_input_files": f"{file_count(m_dir)}/{len(STATE_FILES)}",
        "mr_output_files": f"{file_count(out_dir)}/{len(STATE_FILES)}",
        "rows": N_BINS,
        "num_nodes": len(layout.node_ids),
        "num_r_nodes": num_r,
        "num_m_nodes": layout.num_m,
        "header_time_bin_check": "PASS",
        "nonnegative_integer_check": "PASS",
        "r_submatrix_check": "PASS",
        "m_submatrix_check": "PASS",
        "r_input_conservation_check": "PASS",
        "m_input_conservation_check": "PASS",
        "mr_conservation_check": "PASS",
        "initial_stock_zero_check": "PASS",
        "r_total_inflow": int(merged["inflow.csv"][:, :num_r].sum()),
        "r_total_outflow": int(merged["outflow.csv"][:, :num_r].sum()),
        "m_total_inflow": int(merged["inflow.csv"][:, num_r:].sum()),
        "m_total_outflow": int(merged["outflow.csv"][:, num_r:].sum()),
        "r_end_of_day_stock": int(merged["accumulation_end.csv"][-1, :num_r].sum()),
        "m_end_of_day_stock": int(merged["accumulation_end.csv"][-1, num_r:].sum()),
        "graph_signature": layout.signature,
        "error": "",
    }
    return merged, row


def failed_row(date_label: str, r_dir: Path, m_dir: Path, out_dir: Path, detail: str) -> dict[str, object]:
    return {
        "date": date_label,
        "status": "FAIL",
        "skipped": "",
        "r_input_files": f"{file_count(r_dir)}/{len(STATE_FILES)}",
        "m_input_files": f"{file_count(m_dir)}/{len(STATE_FILES)}",
        "mr_output_files": f"{file_count(out_dir)}/{len(STATE_FILES)}",
        "error": detail,
    }


def process_day(task: dict) -> dict[str, object]:
    date_label = task["date"]
    layout: Layout = task["layout"]
    r_dir = Path(task["r_root"]) / date_label
    m_dir = Path(task["m_root"]) / date_label
    out_dir = Path(task["out_root"]) / date_label
    stats_path = Path(task["out_root"]) / "day_stats" / f"{date_label}.json"
    began = time.time()
    try:
        previous = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else None
        if stats_path.is_file():
            stats_path.unlink()
        may_skip = task["skip_complete"] and previously_passed(previous, layout)
        _merged, row = merge_day(r_dir, m_dir, out_dir, layout, date_label, may_skip)
        row = {"date": date_label, **row, "elapsed_s": round(time.time() - began, 1)}
        mfull.write_json(stats_path, row)
        log(f"[{date_label}] PASS skipped={row['skipped']} R in/out={row['r_total_inflow']}/{row['r_total_outflow']} "
            f"M in/out={row['m_total_inflow']}/{row['m_total_outflow']} elapsed={row['elapsed_s']}s")
        return row
    except Exception:
        detail = traceback.format_exc(limit=6).strip().replace("\n", " | ")
        log(f"[{date_label}] FAIL {detail}")
        return failed_row(date_label, r_dir, m_dir, out_dir, detail)


def day_dirs(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and mrgraph.DAY_PATTERN.fullmatch(p.name))


def column_digest(node_ids: list[str]) -> str:
    return hashlib.sha256(",".join(["time_bin", "start_time", *node_ids]).encode("utf-8")).hexdigest()


def write_metadata(path: Path, layout: Layout, sources: dict[str, str], dates: list[str], extra: dict | None = None) -> None:
    num_r, num_m = layout.num_r, layout.num_m
    payload = {
        "level": "mr-level",
        "graph": layout.graph_path,
        "graph_signature": layout.signature,
        "bin_seconds": BIN_SECONDS,
        "bins_per_day": N_BINS,
        "dates": dates,
        "state_files": list(STATE_FILES),
        "csv_layout": "time_bin,start_time,<node_ids of mr_graph.npz>",
        "column_header_sha256": column_digest(layout.node_ids),
        "num_nodes": num_r + num_m,
        "num_r_nodes": num_r,
        "num_m_nodes": num_m,
        "r_nodes": {
            "global_index_range": [0, num_r - 1],
            "csv_value_column_range": [0, num_r - 1],
            "node_id_format": "R::<edge_id>",
            "order": "r_graph.npz edge_ids",
            "first": layout.node_ids[0],
            "last": layout.node_ids[num_r - 1],
            "state_meaning": STATE_MEANINGS["R"],
        },
        "m_nodes": {
            "global_index_range": [num_r, num_r + num_m - 1],
            "csv_value_column_range": [num_r, num_r + num_m - 1],
            "node_id_format": "M::<junction_id>::<incoming_edge_id>::<outgoing_edge_id>",
            "order": "m_graph.npz movement_ids",
            "first": layout.node_ids[num_r],
            "last": layout.node_ids[-1],
            "state_meaning": STATE_MEANINGS["M"],
        },
        "construction": "column concatenation [R-level | M-level] per state and bin; no value is recounted, summed or normalized",
        "conservation": "checked per node: end = start + inflow - outflow, start[t+1] = end[t], start[0] = 0",
        "not_comparable": "R accumulation is road stock and M accumulation is connector stock; never add or equate them",
        "sources": sources,
        **(extra or {}),
    }
    mfull.write_json(path, payload)


def run_pool(tasks: list[dict], workers: int, worker, on_error) -> list:
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, task): task["date"] for task in tasks}
        for future in as_completed(futures):
            date_label = futures[future]
            try:
                result = future.result()
            except Exception:
                detail = traceback.format_exc(limit=6).strip().replace("\n", " | ")
                log(f"[{date_label}] FAIL {detail}")
                result = on_error(date_label, detail)
            results.extend(result if isinstance(result, list) else [result])
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Join full R-level and M-level four-state CSVs into MR-level CSVs")
    prep_paths.add_3s_path_arguments(parser)
    parser.add_argument("--workers", type=int, default=29)
    parser.add_argument("--dates", nargs="*", default=None)
    parser.add_argument("--skip-complete", action="store_true")
    args = parser.parse_args(argv)
    root = prep_paths.refuse_code_root(args.training_root)
    r_root = root / "full_flow" / "r-level"
    m_root = root / "full_flow" / "m-level"
    out_root = root / "full_flow" / "mr-level"
    layout = load_layout(root / "graph")
    log(f"graph={layout.graph_path} signature={layout.signature[:12]} nodes={len(layout.node_ids)} R={layout.num_r} M={layout.num_m}")
    log(f"r_root={r_root}")
    log(f"m_root={m_root}")
    log(f"output_root={out_root}")
    r_dates, m_dates = day_dirs(r_root), day_dirs(m_root)
    if r_dates != m_dates or not r_dates:
        log(f"R and M full-flow dates differ: only R={sorted(set(r_dates) - set(m_dates))} only M={sorted(set(m_dates) - set(r_dates))}")
        return 1
    all_dates = r_dates
    dates = args.dates or all_dates
    missing = [d for d in dates if d not in all_dates]
    if missing:
        log(f"unknown dates {missing}")
        return 1
    workers = max(1, min(args.workers, len(dates)))
    log(f"dates={len(dates)} workers={workers} skip_complete={args.skip_complete}")
    tasks = [
        {"date": d, "layout": layout, "r_root": str(r_root), "m_root": str(m_root),
         "out_root": str(out_root), "skip_complete": args.skip_complete}
        for d in dates
    ]
    rows = run_pool(
        tasks, workers, process_day,
        lambda d, detail: failed_row(d, r_root / d, m_root / d, out_root / d, detail),
    )
    summary_rows = mfull.collect_summary(rows, out_root / "day_stats", all_dates)
    mfull.write_summary(out_root / "generation_summary.csv", SUMMARY_FIELDS, summary_rows)
    write_metadata(
        out_root / "mr_level_metadata.json", layout,
        {"r-level": f"{r_root}/<date>", "m-level": f"{m_root}/<date>"}, all_dates,
    )
    failed = sorted(str(r["date"]) for r in rows if r.get("status") != "PASS")
    passed = [r for r in summary_rows if r.get("status") == "PASS"]
    files = sum(file_count(out_root / d) for d in all_dates)
    log(
        f"summary={out_root / 'generation_summary.csv'} run_pass={len(rows) - len(failed)} run_fail={len(failed)} "
        f"failed={','.join(failed) or 'none'} summary_pass={len(passed)}/{len(all_dates)} "
        f"state_files={files}/{len(all_dates) * len(STATE_FILES)}"
    )
    ok = not failed and len(passed) == len(all_dates) and files == len(all_dates) * len(STATE_FILES)
    if ok:
        log("MR-level full-flow generation completed successfully.")
    return 0 if ok else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
