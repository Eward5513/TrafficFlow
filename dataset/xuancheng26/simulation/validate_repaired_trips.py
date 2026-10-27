#!/usr/bin/env python3
"""Independent validation of repaired Xuancheng26 trips.

Re-reads official outputs. Does not trust repair-script in-memory counters.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from repair_lib import (
    CODE_ROOT,
    SOURCE_ROOT,
    scan_code_files,
    scan_large_data_in_code,
    data_paths,
    date_from_trip_name,
    depart_time_invalid,
    ensure_data_dirs,
    fields_except_route_equal,
    iter_json_array,
    iter_jsonl_gz,
    load_config,
    load_pickle,
    parse_sumo_net,
    sha256_file,
    source_trip_files,
    trip_path_for_date,
    unique_active_dates,
    write_json_atomic,
)


def load_graph(paths: dict):
    pkl = paths["road_topology"] / "sumo_graph.pkl"
    if pkl.is_file():
        return load_pickle(pkl)
    return parse_sumo_net(paths["base_net"])


def output_paths(paths: dict, date: str) -> dict:
    tag = date.replace("-", "_")
    return {
        "repaired": paths["repaired"] / f"data_{tag}_repaired.json",
        "rejected": paths["rejected"] / f"data_{tag}_rejected.jsonl.gz",
        "metadata": paths["repair_metadata"] / f"data_{tag}_metadata.jsonl.gz",
        "manifest": paths["manifests"] / f"manifest_{tag.replace('-', '_')}.json"
        if False
        else paths["manifests"] / f"manifest_{date.replace('-', '_')}.json",
    }


def count_json_array(path: Path) -> int:
    n = 0
    for _ in iter_json_array(path):
        n += 1
    return n


def count_jsonl_gz(path: Path) -> int:
    n = 0
    for _ in iter_jsonl_gz(path):
        n += 1
    return n


def validate_date(
    date: str,
    cfg: dict,
    paths: dict,
    graph,
    *,
    sample_insert: int = 100,
    sample_delete: int = 100,
    sample_rejected: int = 500,
    seed: int = 42,
) -> dict:
    outs = output_paths(paths, date)
    source = trip_path_for_date(cfg, date)
    issues: list[str] = []
    n_illegal_pairs = 0
    n_internal = 0
    n_unknown = 0
    n_short = 0
    n_k_gt_3 = 0
    n_repaired = 0
    n_rejected = 0
    n_source = 0
    n_field_mismatch = 0
    n_invalid_time_in_repaired = 0
    n_meta = 0
    insert_candidates: list[int] = []
    delete_candidates: list[int] = []
    rejected_rows: list[dict] = []

    meta_by_idx: dict[int, dict] = {}
    if not outs["metadata"].is_file():
        issues.append(f"missing metadata {outs['metadata']}")
    else:
        for rec in iter_jsonl_gz(outs["metadata"]):
            meta_by_idx[int(rec["source_record_index"])] = rec
            n_meta += 1

    if not outs["repaired"].is_file():
        issues.append(f"missing repaired {outs['repaired']}")
        repaired_iter = iter(())
    else:
        repaired_iter = iter_json_array(outs["repaired"])
    repaired_buf = []
    for rec in repaired_iter:
        repaired_buf.append(rec)
        n_repaired += 1
        route = rec.get("route") or []
        if depart_time_invalid(rec, limit=int(cfg.get("depart_time_limit", 86400))):
            n_invalid_time_in_repaired += 1
            issues.append("invalid_depart_time leaked into repaired")
        if len(route) < 2:
            n_short += 1
        for edge in route:
            if str(edge).startswith(":") or edge not in graph.normal_set:
                if str(edge).startswith(":"):
                    n_internal += 1
                if edge not in graph.normal_set:
                    n_unknown += 1
        for a, b in zip(route, route[1:]):
            if not graph.has_connection(str(a), str(b)):
                n_illegal_pairs += 1

    if not outs["rejected"].is_file():
        issues.append(f"missing rejected {outs['rejected']}")
    else:
        for rec in iter_jsonl_gz(outs["rejected"]):
            n_rejected += 1
            if len(rejected_rows) < sample_rejected:
                rejected_rows.append(rec)

    repaired_ptr = 0
    for rec in iter_json_array(source):
        idx = n_source
        n_source += 1
        meta = meta_by_idx.get(idx)
        if meta is None:
            issues.append(f"{date} missing metadata for index {idx}")
            continue
        if meta.get("status") == "repaired":
            if repaired_ptr >= len(repaired_buf):
                issues.append(f"{date} repaired file shorter than metadata")
                continue
            out_rec = repaired_buf[repaired_ptr]
            repaired_ptr += 1
            if not fields_except_route_equal(rec, out_rec):
                n_field_mismatch += 1
            if int(meta.get("max_inserted_edges_single_bridge") or 0) > int(cfg["max_inserted_edges"]):
                n_k_gt_3 += 1
            action = meta.get("action")
            if action == "insertion_only":
                insert_candidates.append(idx)
            if action in ("deletion_only", "deletion_insertion"):
                delete_candidates.append(idx)
        elif meta.get("status") != "rejected":
            issues.append(f"{date} index {idx} has unknown status {meta.get('status')}")
    if repaired_ptr != n_repaired:
        issues.append(f"{date} repaired pointer {repaired_ptr} != n_repaired {n_repaired}")

    if n_source != n_repaired + n_rejected:
        issues.append(
            f"{date} conservation failed source={n_source} repaired={n_repaired} rejected={n_rejected}"
        )
    if n_meta != n_source:
        issues.append(f"{date} metadata count {n_meta} != source {n_source}")

    rng = random.Random(seed)
    sampled_insert = (
        rng.sample(insert_candidates, min(sample_insert, len(insert_candidates)))
        if insert_candidates
        else []
    )
    sampled_delete = (
        rng.sample(delete_candidates, min(sample_delete, len(delete_candidates)))
        if delete_candidates
        else []
    )
    for idx in sampled_insert + sampled_delete:
        meta = meta_by_idx[idx]
        if int(meta.get("max_inserted_edges_single_bridge") or 0) > int(cfg["max_inserted_edges"]):
            n_k_gt_3 += 1

    ok = (
        not issues
        and n_illegal_pairs == 0
        and n_internal == 0
        and n_unknown == 0
        and n_short == 0
        and n_k_gt_3 == 0
        and n_field_mismatch == 0
        and n_invalid_time_in_repaired == 0
    )
    return {
        "date": date,
        "ok": ok,
        "n_source": n_source,
        "n_repaired": n_repaired,
        "n_rejected": n_rejected,
        "n_illegal_pairs": n_illegal_pairs,
        "n_internal": n_internal,
        "n_unknown": n_unknown,
        "n_short": n_short,
        "n_k_gt_3": n_k_gt_3,
        "n_field_mismatch": n_field_mismatch,
        "n_invalid_time_in_repaired": n_invalid_time_in_repaired,
        "n_sampled_insert": len(sampled_insert),
        "n_sampled_delete": len(sampled_delete),
        "n_sampled_rejected": len(rejected_rows),
        "issues": issues[:50],
        "n_issues": len(issues),
    }


def _worker(payload: dict) -> dict:
    cfg = payload["cfg"]
    date = payload["date"]
    seed = payload["seed"]
    paths = ensure_data_dirs(cfg)
    graph = load_graph(paths)
    t0 = time.time()
    rec = validate_date(date, cfg, paths, graph, seed=seed)
    rec["elapsed_s"] = round(time.time() - t0, 3)
    return rec


def hash_protected_inputs(cfg: dict) -> dict:
    files = []
    files.extend(source_trip_files(SOURCE_ROOT))
    files.append(Path(cfg["sumo_net"]))
    files.append(Path(cfg["cityflow_roadnet"]))
    meta_dir = SOURCE_ROOT / "metadata"
    if meta_dir.is_dir():
        files.extend(sorted(p for p in meta_dir.iterdir() if p.is_file()))
    out = {}
    for path in files:
        rel = str(path.relative_to(SOURCE_ROOT)) if str(path).startswith(str(SOURCE_ROOT)) else str(path)
        out[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate repaired Xuancheng26 trips")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dates", type=str, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    paths = ensure_data_dirs(cfg)

    if args.dates:
        dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    else:
        dates = unique_active_dates(cfg)

    max_workers = int(cfg.get("max_workers", 10))
    workers = args.workers
    if workers is None:
        workers = int(cfg.get("default_workers", 10))
    workers = max(1, min(workers, max_workers, len(dates) or 1))
    print(f"[validate] dates={len(dates)} workers={workers}", flush=True)

    apr9_repaired = paths["repaired"] / "data_2023_04_09_repaired.json"
    daily = []
    all_ok = True
    total_illegal = 0
    by_date: dict[str, dict] = {}
    if workers == 1 or len(dates) <= 1:
        graph = load_graph(paths)
        for date in dates:
            t0 = time.time()
            rec = validate_date(date, cfg, paths, graph, seed=args.seed)
            rec["elapsed_s"] = round(time.time() - t0, 3)
            by_date[date] = rec
    else:
        payloads = [{"cfg": cfg, "date": date, "seed": args.seed} for date in dates]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_worker, payload): payload["date"] for payload in payloads}
            for fut in as_completed(futs):
                date = futs[fut]
                rec = fut.result()
                by_date[date] = rec
    for date in dates:
        rec = by_date[date]
        daily.append(rec)
        total_illegal += rec["n_illegal_pairs"]
        if not rec["ok"]:
            all_ok = False
            print(f"[validate] FAIL {date} issues={rec['issues'][:5]}", flush=True)
        else:
            print(
                f"[validate] OK {date} source={rec['n_source']} repaired={rec['n_repaired']} "
                f"rejected={rec['n_rejected']} illegal_pairs=0 elapsed={rec.get('elapsed_s')}s",
                flush=True,
            )

    hashes_now = hash_protected_inputs(cfg)
    before_path = paths["manifests"] / "input_hashes_before.json"
    source_unchanged = True
    hash_mismatches = []
    if before_path.is_file():
        before = json.loads(before_path.read_text(encoding="utf-8"))
        before_map = {row["relative_path"]: row["sha256"] for row in before.get("files") or []}
        for rel, rec in hashes_now.items():
            if rel in before_map and before_map[rel] != rec["sha256"]:
                source_unchanged = False
                hash_mismatches.append(rel)
    else:
        source_unchanged = False
        hash_mismatches.append("missing input_hashes_before.json")

    code_hits = scan_code_files(paths["data_root"])
    data_in_code = scan_large_data_in_code(CODE_ROOT)
    exclude_ok = not apr9_repaired.exists()
    n_dates = len(dates)
    unique_ok = n_dates == 29 if args.dates is None else True

    summary = {
        "ok": bool(
            all_ok
            and source_unchanged
            and not code_hits
            and not data_in_code
            and exclude_ok
            and total_illegal == 0
            and unique_ok
        ),
        "n_dates": n_dates,
        "workers": workers,
        "all_dates_ok": all_ok,
        "total_illegal_pairs": total_illegal,
        "source_unchanged": source_unchanged,
        "hash_mismatches": hash_mismatches,
        "apr9_repaired_absent": exclude_ok,
        "code_files_in_data_dir": code_hits,
        "large_data_in_code_dir": data_in_code,
        "daily": daily,
    }
    write_json_atomic(summary, paths["reports"] / "validation_summary.json", paths["tmp"])
    write_json_atomic(hashes_now, paths["manifests"] / "input_hashes_after.json", paths["tmp"])
    print(
        f"[validate] DONE ok={summary['ok']} illegal_pairs={total_illegal} "
        f"source_unchanged={source_unchanged} code_in_data={len(code_hits)}",
        flush=True,
    )
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
