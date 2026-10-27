#!/usr/bin/env python3
"""Summarize Xuancheng26 trip-repair outputs into reports/."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from repair_lib import (
    SOURCE_ROOT,
    data_paths,
    date_from_trip_name,
    ensure_data_dirs,
    iter_json_array,
    iter_jsonl_gz,
    load_config,
    sha256_file,
    source_trip_files,
    trip_path_for_date,
    unique_active_dates,
    write_json_atomic,
)


def percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    seq = sorted(values)
    if len(seq) == 1:
        return float(seq[0])
    k = (p / 100.0) * (len(seq) - 1)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return float(seq[lo])
    w = k - lo
    return float(seq[lo] * (1.0 - w) + seq[hi] * w)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize trip repair outputs")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--wall-clock-s", type=float, default=None)
    parser.add_argument("--peak-rss-mb", type=float, default=None)
    parser.add_argument("--ready", type=str, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    paths = ensure_data_dirs(cfg)
    dates = unique_active_dates(cfg)
    exclude = cfg.get("exclude_dates") or {}

    daily_rows = []
    action_counter: Counter[str] = Counter()
    reason_counter: Counter[str] = Counter()
    deleted_edge_counter: Counter[str] = Counter()
    inserted_path_counter: Counter[str] = Counter()
    per_trip_deleted: list[float] = []
    per_trip_inserted: list[float] = []
    route_len_delta: list[float] = []
    n_trips = 0
    n_repaired = 0
    n_rejected = 0
    n_invalid_time = 0
    n_direct = 0
    n_ins_only = 0
    n_del_only = 0
    n_del_ins = 0
    n_endpoint = 0
    n_cycle = 0
    n_amb = 0
    n_orig_edges = 0
    n_kept_edges = 0
    n_deleted_edges = 0
    n_inserted_edges = 0
    hourly_before = Counter()
    hourly_after = Counter()
    source_hashes = {}
    output_hashes = {}

    validation_path = paths["reports"] / "validation_summary.json"
    validation = {}
    if validation_path.is_file():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))

    for date in dates:
        tag = date.replace("-", "_")
        manifest_path = paths["manifests"] / f"manifest_{tag}.json"
        meta_path = paths["repair_metadata"] / f"data_{tag}_metadata.jsonl.gz"
        repaired_path = paths["repaired"] / f"data_{tag}_repaired.json"
        rejected_path = paths["rejected"] / f"data_{tag}_rejected.jsonl.gz"
        source = trip_path_for_date(cfg, date)
        source_hashes[str(source)] = sha256_file(source)
        if repaired_path.is_file():
            output_hashes[str(repaired_path)] = sha256_file(repaired_path)
        if rejected_path.is_file():
            output_hashes[str(rejected_path)] = sha256_file(rejected_path)
        if meta_path.is_file():
            output_hashes[str(meta_path)] = sha256_file(meta_path)

        day_repaired = 0
        day_rejected = 0
        day_invalid = 0
        day_source = 0
        day_direct = 0
        day_ins = 0
        day_del = 0
        day_both = 0
        for rec in iter_jsonl_gz(meta_path):
            day_source += 1
            n_trips += 1
            action = rec.get("action") or ""
            action_counter[action] += 1
            n_orig_edges += int(rec.get("original_route_length") or 0)
            n_kept_edges += int(rec.get("original_edge_count_retained") or 0)
            n_deleted_edges += int(rec.get("deleted_observed_edge_count") or 0)
            n_inserted_edges += int(rec.get("inserted_edge_count") or 0)
            per_trip_deleted.append(float(rec.get("deleted_observed_edge_count") or 0))
            per_trip_inserted.append(float(rec.get("inserted_edge_count") or 0))
            orig_n = float(rec.get("original_route_length") or 0)
            rep_n = float(rec.get("repaired_route_length") or 0)
            route_len_delta.append(rep_n - orig_n)
            if rec.get("endpoint_deleted"):
                n_endpoint += 1
            if rec.get("introduced_cycle"):
                n_cycle += 1
            if rec.get("ambiguous_bridge"):
                n_amb += 1
            for edge in rec.get("deleted_observed_edge_ids") or []:
                deleted_edge_counter[str(edge)] += 1
            inserted = rec.get("inserted_edge_ids") or []
            if inserted:
                inserted_path_counter[",".join(str(x) for x in inserted)] += 1
            if rec.get("status") == "repaired":
                day_repaired += 1
                n_repaired += 1
                act = rec.get("action")
                if act == "direct_legal":
                    day_direct += 1
                    n_direct += 1
                elif act == "insertion_only":
                    day_ins += 1
                    n_ins_only += 1
                elif act == "deletion_only":
                    day_del += 1
                    n_del_only += 1
                elif act == "deletion_insertion":
                    day_both += 1
                    n_del_ins += 1
            else:
                day_rejected += 1
                n_rejected += 1
                reason = rec.get("reject_reason") or "unknown"
                reason_counter[reason] += 1
                if reason == "invalid_depart_time":
                    day_invalid += 1
                    n_invalid_time += 1

        for rec in iter_json_array(source):
            try:
                hour = int(float(rec.get("startTime") or 0) // 3600)
            except (TypeError, ValueError):
                hour = -1
            hourly_before[hour] += 1
        if repaired_path.is_file():
            for rec in iter_json_array(repaired_path):
                try:
                    hour = int(float(rec.get("startTime") or 0) // 3600)
                except (TypeError, ValueError):
                    hour = -1
                hourly_after[hour] += 1

        retain_rate = (day_repaired / day_source) if day_source else 0.0
        daily_rows.append(
            {
                "date": date,
                "n_source": day_source,
                "n_repaired": day_repaired,
                "n_rejected": day_rejected,
                "n_invalid_depart_time": day_invalid,
                "n_direct_legal": day_direct,
                "n_insertion_only": day_ins,
                "n_deletion_only": day_del,
                "n_deletion_insertion": day_both,
                "vehicle_retain_rate": f"{retain_rate:.6f}",
            }
        )

    # April 9 original count, not processed.
    apr9 = SOURCE_ROOT / "trips" / "data_2023_04_09_type_filtered.json"
    apr8 = SOURCE_ROOT / "trips" / "data_2023_04_08_type_filtered.json"
    n_apr9 = 0
    if apr9.is_file():
        for _ in iter_json_array(apr9):
            n_apr9 += 1
        source_hashes[str(apr9)] = sha256_file(apr9)
    n_apr8 = 0
    if apr8.is_file():
        for _ in iter_json_array(apr8):
            n_apr8 += 1

    illegal = 0
    for rec in validation.get("daily") or []:
        illegal += int(rec.get("n_illegal_pairs") or 0)
    legal_rate = 1.0 if n_repaired and illegal == 0 else (0.0 if n_repaired else None)

    retained_rate = (n_kept_edges / n_orig_edges) if n_orig_edges else 0.0
    ready = args.ready
    if ready is None:
        ready = "YES" if validation.get("ok") else "NO"

    report = []
    report.append("# Xuancheng26 trip repair report")
    report.append("")
    report.append("## Totals")
    report.append("")
    report.append(f"- input date files: 30")
    report.append(f"- unique dates processed: {len(dates)}")
    report.append(f"- total trips (unique dates): {n_trips}")
    report.append(f"- repaired: {n_repaired}")
    report.append(f"- rejected: {n_rejected}")
    report.append(f"- invalid departure time: {n_invalid_time}")
    report.append(f"- original directly legal: {n_direct}")
    report.append(f"- insertion-only: {n_ins_only}")
    report.append(f"- deletion-only: {n_del_only}")
    report.append(f"- deletion+insertion: {n_del_ins}")
    report.append(f"- endpoint deleted: {n_endpoint}")
    report.append(f"- introduced cycle flags: {n_cycle}")
    report.append(f"- ambiguous bridges: {n_amb}")
    report.append(f"- original observed edges: {n_orig_edges}")
    report.append(f"- retained original edges: {n_kept_edges}")
    report.append(f"- deleted original edges: {n_deleted_edges}")
    report.append(f"- inserted edges: {n_inserted_edges}")
    report.append(f"- original edge retention rate: {retained_rate:.6f}")
    report.append(
        "- deleted edges/trip P50/P90/P95/P99/max: "
        f"{percentile(per_trip_deleted,50)}/{percentile(per_trip_deleted,90)}/"
        f"{percentile(per_trip_deleted,95)}/{percentile(per_trip_deleted,99)}/"
        f"{max(per_trip_deleted) if per_trip_deleted else None}"
    )
    report.append(
        "- inserted edges/trip P50/P90/P95/P99/max: "
        f"{percentile(per_trip_inserted,50)}/{percentile(per_trip_inserted,90)}/"
        f"{percentile(per_trip_inserted,95)}/{percentile(per_trip_inserted,99)}/"
        f"{max(per_trip_inserted) if per_trip_inserted else None}"
    )
    report.append(
        "- route length delta P50/P90/P95/P99/max: "
        f"{percentile(route_len_delta,50)}/{percentile(route_len_delta,90)}/"
        f"{percentile(route_len_delta,95)}/{percentile(route_len_delta,99)}/"
        f"{max(route_len_delta) if route_len_delta else None}"
    )
    report.append(f"- repaired SUMO legal pair rate: {legal_rate}")
    report.append(f"- illegal adjacent pairs in repaired outputs: {illegal}")
    report.append(f"- wall-clock seconds: {args.wall_clock_s}")
    report.append(f"- peak RSS MB: {args.peak_rss_mb}")
    report.append("")
    report.append("## April 8 / 9 duplicate")
    report.append("")
    report.append(
        f"2023-04-09: excluded_duplicate_of=2023-04-08. "
        f"April 8 original records={n_apr8}. April 9 original records={n_apr9}. "
        "The original April 9 file was not deleted. No repaired file was written for April 9."
    )
    report.append("")
    report.append("## Invalid departure times")
    report.append("")
    report.append(
        "Records with startTime>=86400 were still passed through route repair, "
        "but were written to rejected/ with reject_reason=invalid_depart_time. "
        "Times were not modulo-reduced or shifted by 86400. "
        f"Unique-date invalid_depart_time count={n_invalid_time}. "
        "The original 30-file audit total of 53 includes 2 records from the excluded April 9 duplicate."
    )
    report.append("")
    report.append("## Daily vehicle retain rates")
    report.append("")
    report.append("| date | source | repaired | rejected | retain_rate |")
    report.append("| --- | ---: | ---: | ---: | ---: |")
    for row in daily_rows:
        report.append(
            f"| {row['date']} | {row['n_source']} | {row['n_repaired']} | "
            f"{row['n_rejected']} | {row['vehicle_retain_rate']} |"
        )
    report.append("")
    report.append("## Hourly departures, original vs repaired")
    report.append("")
    report.append("| hour | original | repaired |")
    report.append("| ---: | ---: | ---: |")
    hours = sorted(set(hourly_before) | set(hourly_after))
    for hour in hours:
        report.append(f"| {hour} | {hourly_before[hour]} | {hourly_after[hour]} |")
    report.append("")
    report.append("## Top deleted edges")
    report.append("")
    for edge, cnt in deleted_edge_counter.most_common(20):
        report.append(f"- {edge}: {cnt}")
    report.append("")
    report.append("## Top inserted paths")
    report.append("")
    for path, cnt in inserted_path_counter.most_common(20):
        report.append(f"- `{path}`: {cnt}")
    report.append("")
    report.append("## Reject reasons")
    report.append("")
    for reason, cnt in reason_counter.most_common():
        report.append(f"- {reason}: {cnt}")
    report.append("")
    report.append("## Source and output SHA256")
    report.append("")
    report.append("See `source_and_output_manifest.json`.")
    report.append("")
    report.append("## Acceptance")
    report.append("")
    report.append(f"READY_FOR_SUMO_ROUTE_CONVERSION = {ready}")
    report.append("")

    (paths["reports"] / "trip_repair_report.md").write_text("\n".join(report), encoding="utf-8")

    write_csv(
        paths["reports"] / "daily_repair_statistics.csv",
        daily_rows,
        [
            "date",
            "n_source",
            "n_repaired",
            "n_rejected",
            "n_invalid_depart_time",
            "n_direct_legal",
            "n_insertion_only",
            "n_deletion_only",
            "n_deletion_insertion",
            "vehicle_retain_rate",
        ],
    )
    write_csv(
        paths["reports"] / "repair_action_distribution.csv",
        [{"action": k, "count": v} for k, v in action_counter.most_common()],
        ["action", "count"],
    )
    write_csv(
        paths["reports"] / "top_deleted_edges.csv",
        [{"edge_id": k, "count": v} for k, v in deleted_edge_counter.most_common(200)],
        ["edge_id", "count"],
    )
    write_csv(
        paths["reports"] / "top_inserted_paths.csv",
        [{"inserted_path": k, "count": v} for k, v in inserted_path_counter.most_common(200)],
        ["inserted_path", "count"],
    )
    write_csv(
        paths["reports"] / "rejected_reason_counts.csv",
        [{"reject_reason": k, "count": v} for k, v in reason_counter.most_common()],
        ["reject_reason", "count"],
    )
    write_json_atomic(
        {
            "unique_dates": dates,
            "excluded": exclude,
            "n_trips": n_trips,
            "n_repaired": n_repaired,
            "n_rejected": n_rejected,
            "n_invalid_depart_time": n_invalid_time,
            "n_apr9_original_records": n_apr9,
            "n_apr8_original_records": n_apr8,
            "source_sha256": source_hashes,
            "output_sha256": output_hashes,
            "illegal_pairs": illegal,
        },
        paths["reports"] / "source_and_output_manifest.json",
        paths["tmp"],
    )
    print(f"[summarize] wrote {paths['reports'] / 'trip_repair_report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
