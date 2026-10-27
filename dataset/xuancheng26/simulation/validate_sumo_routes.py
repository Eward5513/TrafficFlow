#!/usr/bin/env python3
"""Full validation of prepared Xuancheng26 SUMO route files, including SUMO load."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from repair_lib import iter_json_array, sha256_file
from sumo_route_common import (
    CODE_ROOT,
    DATA_ROOT,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    DEPART_LIMIT,
    EXCLUDED_DATES,
    EXPECTED_DATES,
    MANIFEST_FIELDS,
    MIN_ROUTE_EDGES,
    N_DATES_EXPECTED,
    ORIGINAL_NET_FILE,
    SOURCE_ROOT,
    classify_vehicle,
    coerce_route,
    collect_env,
    compare_hash_rows,
    date_to_tag,
    discover_dates,
    file_meta,
    format_depart,
    hash_file_row,
    load_pickle,
    load_topology,
    log_print,
    max_overlap,
    parse_route_date,
    parse_start_time,
    prepared_route_files,
    resolve_net_file,
    route_name_for_date,
    run_sumo_load,
    save_pickle_atomic,
    scan_code_in_data,
    scan_large_or_day_data_in_code,
    snapshot_original_inputs,
    source_name_for_date,
    strip_ns,
    vehicle_id_for,
    write_json_atomic,
    write_text_atomic,
    write_tsv,
)

WORKER_STARTED_AT: Optional[float] = None


def _init_worker(topo_path: str) -> None:
    global WORKER_STARTED_AT
    WORKER_STARTED_AT = time.time()
    _init_worker.topology = load_pickle(Path(topo_path))  # type: ignore[attr-defined]


def _topology() -> Any:
    topo = getattr(_init_worker, "topology", None)
    if topo is None:
        raise RuntimeError("worker topology was not initialized")
    return topo


def _as_float(text: str) -> Optional[float]:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def validate_one(payload: Dict[str, Any]) -> Dict[str, Any]:
    date = payload["date"]
    source = Path(payload["source"])
    output = Path(payload["output"])
    net_file = Path(payload["net_file"])
    log_path = Path(payload["log_path"])
    stdout_path = Path(payload["sumo_stdout_path"])
    stderr_path = Path(payload["sumo_stderr_path"])
    pid = os.getpid()
    started = WORKER_STARTED_AT if WORKER_STARTED_AT is not None else time.time()
    t0 = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    issues: List[str] = []
    warnings: List[str] = []

    def log(msg: str) -> None:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} pid={pid} {msg}\n")

    log_path.write_text(
        f"{time.strftime('%Y-%m-%dT%H:%M:%S')} pid={pid} start date={date}\n",
        encoding="utf-8",
    )

    result: Dict[str, Any] = {
        "date": date,
        "ok": False,
        "source_file": str(source),
        "output_file": str(output),
        "validation_pid": pid,
        "worker_started_ts": started,
        "xml_validation_ok": False,
        "topology_validation_ok": False,
        "sumo_load_ok": False,
        "sumo_exit_code": None,
        "issues": issues,
        "warnings": warnings,
        "log_path": str(log_path),
    }

    try:
        topo = _topology()
        if not output.is_file():
            issues.append(f"missing route file: {output}")
            return result
        if not source.is_file():
            issues.append(f"missing source file: {source}")
            return result

        source_records = 0
        filtered = 0
        # Stream source and XML together without storing all routes.
        src_iter = iter_json_array(source)

        def next_kept_expected() -> Optional[Dict[str, Any]]:
            nonlocal source_records, filtered
            for rec in src_iter:
                idx = source_records
                source_records += 1
                try:
                    depart = parse_start_time(rec.get("startTime"))
                except Exception as exc:
                    issues.append(f"source[{idx}] invalid startTime: {exc}")
                    continue
                if depart < 0:
                    issues.append(f"source[{idx}] negative startTime={rec.get('startTime')!r}")
                    continue
                vtype = classify_vehicle(rec.get("vehicle"))
                if vtype == "other":
                    issues.append(f"source[{idx}] unknown vehicle fingerprint")
                    continue
                if depart >= DEPART_LIMIT:
                    filtered += 1
                    continue
                try:
                    edges = coerce_route(rec.get("route"))
                except Exception as exc:
                    issues.append(f"source[{idx}] {exc}")
                    continue
                return {
                    "index": idx,
                    "id": vehicle_id_for(date, idx),
                    "type": vtype,
                    "depart": depart,
                    "depart_text": format_depart(rec.get("startTime"), depart),
                    "edges": edges,
                }
            return None

        n_vtype = 0
        vtype_ids: List[str] = []
        n_trip = 0
        n_flow = 0
        n_top_route = 0
        n_vehicle = 0
        n_passenger = 0
        n_freight = 0
        n_other_type = 0
        duplicate_ids = 0
        invalid_depart = 0
        unsorted_depart = 0
        short_routes = 0
        unknown_edges = 0
        internal_edges = 0
        illegal_pairs = 0
        id_mismatch = 0
        type_mismatch = 0
        depart_mismatch = 0
        route_mismatch = 0
        extra_xml_vehicles = 0
        missing_xml_vehicles = 0
        seen_ids = set()
        last_depart: Optional[float] = None
        last_index: Optional[int] = None
        root_ok = False
        in_vehicle = False
        min_depart = None
        max_depart = None
        route_len_min = None
        route_len_max = None
        route_len_sum = 0
        xml_ok = True
        topology_ok = True

        try:
            context = ET.iterparse(str(output), events=("start", "end"))
            for event, elem in context:
                tag = strip_ns(elem.tag)
                if event == "start":
                    if not root_ok:
                        if tag != "routes":
                            issues.append(f"root element is <{tag}>, expected <routes>")
                            xml_ok = False
                        root_ok = True
                    if tag == "vehicle":
                        in_vehicle = True
                    elif tag == "trip":
                        n_trip += 1
                    elif tag == "flow":
                        n_flow += 1
                    elif tag == "route" and not in_vehicle:
                        n_top_route += 1
                    continue

                if tag == "vType":
                    n_vtype += 1
                    vtype_ids.append(str(elem.attrib.get("id") or ""))
                    elem.clear()
                    continue
                if tag == "route":
                    # Keep nested <route> until the parent <vehicle> is processed.
                    continue
                if tag != "vehicle":
                    if tag in {"trip", "flow"}:
                        xml_ok = False
                    if tag not in {"routes"}:
                        elem.clear()
                    continue

                in_vehicle = False
                veh_id = str(elem.attrib.get("id") or "")
                veh_type = str(elem.attrib.get("type") or "")
                depart_text = elem.attrib.get("depart")
                routes = [child for child in list(elem) if strip_ns(child.tag) == "route"]
                n_vehicle += 1
                if veh_id in seen_ids:
                    duplicate_ids += 1
                    xml_ok = False
                else:
                    seen_ids.add(veh_id)
                depart_val = _as_float(depart_text if depart_text is not None else "")
                if depart_val is None or not (0.0 <= depart_val < DEPART_LIMIT):
                    invalid_depart += 1
                    xml_ok = False
                else:
                    if last_depart is not None and depart_val < last_depart:
                        unsorted_depart += 1
                        xml_ok = False
                    last_depart = depart_val
                    if min_depart is None or depart_val < min_depart:
                        min_depart = depart_val
                    if max_depart is None or depart_val > max_depart:
                        max_depart = depart_val
                if veh_type == "passenger":
                    n_passenger += 1
                elif veh_type == "freight":
                    n_freight += 1
                else:
                    n_other_type += 1
                    xml_ok = False
                if len(routes) != 1:
                    issues.append(f"{veh_id} has {len(routes)} <route> children")
                    xml_ok = False
                    edges: List[str] = []
                else:
                    edges = [item for item in (routes[0].attrib.get("edges") or "").split() if item]
                if len(edges) < MIN_ROUTE_EDGES:
                    short_routes += 1
                    xml_ok = False
                    topology_ok = False
                for edge in edges:
                    kind = topo.classify_edge(edge)
                    if kind == "internal":
                        internal_edges += 1
                        topology_ok = False
                    elif kind == "unknown":
                        unknown_edges += 1
                        topology_ok = False
                for frm, to in zip(edges, edges[1:]):
                    if not topo.has_connection(frm, to):
                        illegal_pairs += 1
                        topology_ok = False

                expected = next_kept_expected()
                if expected is None:
                    extra_xml_vehicles += 1
                    xml_ok = False
                else:
                    if last_index is not None and expected["index"] < last_index:
                        unsorted_depart += 1
                        xml_ok = False
                    last_index = expected["index"]
                    if veh_id != expected["id"]:
                        id_mismatch += 1
                        xml_ok = False
                    if veh_type != expected["type"]:
                        type_mismatch += 1
                        xml_ok = False
                    if depart_val is None or depart_val != expected["depart"]:
                        depart_mismatch += 1
                        xml_ok = False
                    if edges != expected["edges"]:
                        route_mismatch += 1
                        xml_ok = False
                rlen = len(edges)
                if rlen:
                    if route_len_min is None or rlen < route_len_min:
                        route_len_min = rlen
                    if route_len_max is None or rlen > route_len_max:
                        route_len_max = rlen
                    route_len_sum += rlen
                elem.clear()
        except ET.ParseError as exc:
            xml_ok = False
            issues.append(f"XML parse error: {exc}")

        leftover = next_kept_expected()
        while leftover is not None:
            missing_xml_vehicles += 1
            leftover = next_kept_expected()
        # Drain remaining source records (should only be filtered / already counted).
        for rec in src_iter:
            idx = source_records
            source_records += 1
            try:
                depart = parse_start_time(rec.get("startTime"))
            except Exception as exc:
                issues.append(f"source[{idx}] invalid startTime: {exc}")
                continue
            if depart < 0:
                issues.append(f"source[{idx}] negative startTime={rec.get('startTime')!r}")
                continue
            if classify_vehicle(rec.get("vehicle")) == "other":
                issues.append(f"source[{idx}] unknown vehicle fingerprint")
                continue
            if depart >= DEPART_LIMIT:
                filtered += 1
            else:
                missing_xml_vehicles += 1

        if not root_ok:
            xml_ok = False
            issues.append("file has no root element")
        if n_trip:
            xml_ok = False
            issues.append(f"contains {n_trip} <trip> elements")
        if n_flow:
            xml_ok = False
            issues.append(f"contains {n_flow} <flow> elements")
        if n_top_route:
            xml_ok = False
            issues.append(f"contains {n_top_route} top-level <route> elements")
        if n_vtype != 2 or set(vtype_ids) != {"passenger", "freight"}:
            xml_ok = False
            issues.append(f"vType ids={vtype_ids}, expected passenger and freight")
        expected_vehicles = source_records - filtered
        count_ok = n_vehicle == expected_vehicles and extra_xml_vehicles == 0 and missing_xml_vehicles == 0
        if not count_ok:
            xml_ok = False
            issues.append(
                f"count mismatch xml={n_vehicle} source={source_records} filtered={filtered} "
                f"extra={extra_xml_vehicles} missing={missing_xml_vehicles}"
            )
        if duplicate_ids:
            xml_ok = False
            issues.append(f"duplicate vehicle ids={duplicate_ids}")
        if invalid_depart:
            xml_ok = False
            issues.append(f"invalid depart={invalid_depart}")
        if unsorted_depart:
            xml_ok = False
            issues.append(f"unsorted depart/original-order errors={unsorted_depart}")
        if short_routes:
            topology_ok = False
            issues.append(f"short routes={short_routes}")
        if unknown_edges:
            topology_ok = False
            issues.append(f"unknown edges={unknown_edges}")
        if internal_edges:
            topology_ok = False
            issues.append(f"internal edges={internal_edges}")
        if illegal_pairs:
            topology_ok = False
            issues.append(f"illegal pairs={illegal_pairs}")
        if id_mismatch or type_mismatch or depart_mismatch or route_mismatch or n_other_type:
            xml_ok = False
            issues.append(
                f"source alignment id={id_mismatch} type={type_mismatch} "
                f"depart={depart_mismatch} route={route_mismatch} other_type={n_other_type}"
            )

        result.update(
            {
                "source_records": source_records,
                "filtered_ge_86400": filtered,
                "output_vehicles": n_vehicle,
                "passenger_count": n_passenger,
                "freight_count": n_freight,
                "other_fingerprint_count": n_other_type,
                "min_depart": min_depart,
                "max_depart": max_depart,
                "min_route_length": route_len_min,
                "max_route_length": route_len_max,
                "mean_route_length": None
                if not n_vehicle
                else round(route_len_sum / n_vehicle, 6),
                "unknown_edges": unknown_edges,
                "internal_edges": internal_edges,
                "illegal_pairs": illegal_pairs,
                "duplicate_vehicle_ids": duplicate_ids,
                "invalid_depart": invalid_depart,
                "unsorted_depart": unsorted_depart,
                "xml_validation_ok": xml_ok and count_ok,
                "topology_validation_ok": topology_ok and unknown_edges == 0 and internal_edges == 0 and illegal_pairs == 0 and short_routes == 0,
                "vtype_ids": vtype_ids,
            }
        )
        log(
            f"xml/topology xml_ok={result['xml_validation_ok']} topo_ok={result['topology_validation_ok']} "
            f"vehicles={n_vehicle} issues={len(issues)}"
        )

        log("starting SUMO load")
        sumo = run_sumo_load(net_file, output, stdout_path, stderr_path)
        result["sumo"] = {
            "command": sumo["command"],
            "pid": sumo["pid"],
            "exit_code": sumo["exit_code"],
            "elapsed_s": sumo["elapsed_s"],
            "stdout_path": sumo["stdout_path"],
            "stderr_path": sumo["stderr_path"],
            "n_errors": sumo["n_errors"],
            "n_network_warnings": sumo["n_network_warnings"],
            "n_route_warnings": sumo["n_route_warnings"],
            "n_other_warnings": sumo["n_other_warnings"],
            "errors": sumo["errors"][:50],
            "route_warnings": sumo["route_warnings"][:50],
        }
        result["sumo_exit_code"] = sumo["exit_code"]
        result["sumo_pid"] = sumo["pid"]
        result["sumo_load_ok"] = bool(sumo["ok"])
        if not sumo["ok"]:
            issues.append(
                f"SUMO load failed exit={sumo['exit_code']} errors={sumo['n_errors']} "
                f"route_warnings={sumo['n_route_warnings']}"
            )
            if sumo["errors"]:
                issues.append("SUMO error: " + sumo["errors"][0])
            elif sumo["route_warnings"]:
                issues.append("SUMO route warning: " + sumo["route_warnings"][0])
        log(
            f"SUMO load ok={sumo['ok']} exit={sumo['exit_code']} pid={sumo['pid']} "
            f"elapsed={sumo['elapsed_s']:.3f}s network_warnings={sumo['n_network_warnings']}"
        )

        result["ok"] = (
            result["xml_validation_ok"]
            and result["topology_validation_ok"]
            and result["sumo_load_ok"]
            and not issues
        )
        result["issues"] = issues
        return result
    except Exception as exc:
        issues.append(f"{type(exc).__name__}: {exc}")
        result["ok"] = False
        result["exception"] = repr(exc)
        log(f"exception {exc!r}")
        return result
    finally:
        result["worker_ended_ts"] = time.time()
        result["validation_elapsed_s"] = round(time.time() - t0, 6)
        log(f"done ok={result.get('ok')} elapsed={result['validation_elapsed_s']:.3f}s")


def load_generation_map(prep_path: Path) -> Dict[str, Dict[str, Any]]:
    if not prep_path.is_file():
        return {}
    data = json_load(prep_path)
    out = {}
    for row in data.get("daily") or []:
        if row.get("date"):
            out[row["date"]] = row
    return out


def json_load(path: Path) -> Dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def load_hash_tsv(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        rows = []
        for row in reader:
            if "bytes" in row and row["bytes"] != "":
                row["bytes"] = int(row["bytes"])
            if "mtime_ns" in row and row["mtime_ns"] != "":
                row["mtime_ns"] = int(row["mtime_ns"])
            rows.append(row)
        return rows


def bool_tsv(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return ""
    return str(value)


def build_report(summary: Dict[str, Any]) -> str:
    lines = [
        "# Xuancheng26 SUMO route validation report",
        "",
        f"- generated_at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- ok: `{summary.get('ok')}`",
        f"- n_dates_expected: {summary.get('n_dates_expected')}",
        f"- n_dates_generated: {summary.get('n_dates_generated')}",
        f"- generation_workers: {summary.get('generation_workers')}",
        f"- validation_workers: {summary.get('validation_workers')}",
        f"- generation_overlap: {summary.get('generation_overlap')}",
        f"- validation_overlap: {summary.get('validation_overlap')}",
        f"- source_unchanged: `{summary.get('source_unchanged')}`",
        f"- net_unchanged: `{summary.get('net_unchanged')}`",
        f"- original_net_unchanged: `{summary.get('original_net_unchanged')}`",
        "",
        "## Totals",
        "",
        f"- total_source_records: {summary.get('total_source_records')}",
        f"- total_filtered_ge_86400: {summary.get('total_filtered_ge_86400')}",
        f"- total_output_vehicles: {summary.get('total_output_vehicles')}",
        f"- total_passenger: {summary.get('total_passenger')}",
        f"- total_freight: {summary.get('total_freight')}",
        f"- total_other_fingerprints: {summary.get('total_other_fingerprints')}",
        f"- total_unknown_edges: {summary.get('total_unknown_edges')}",
        f"- total_internal_edges: {summary.get('total_internal_edges')}",
        f"- total_illegal_pairs: {summary.get('total_illegal_pairs')}",
        f"- total_duplicate_vehicle_ids: {summary.get('total_duplicate_vehicle_ids')}",
        f"- total_invalid_depart: {summary.get('total_invalid_depart')}",
        f"- total_unsorted_depart: {summary.get('total_unsorted_depart')}",
        f"- sumo_load_failures: {summary.get('sumo_load_failures')}",
        "",
        "## Environment",
        "",
    ]
    env = summary.get("env") or {}
    lines.extend(
        [
            f"- python: `{env.get('python_executable')}` {env.get('python_version')}",
            f"- sumo: `{env.get('sumo_bin')}` {env.get('sumo_version')}",
            f"- SUMO_HOME: `{env.get('SUMO_HOME')}`",
            f"- net_file: `{summary.get('net_file')}`",
            f"- SUMO load uses `--route-steps 0 --begin 0 --end 1 --ignore-route-errors false`.",
            f"- Installed SUMO is 1.19.0, not 1.25.0; verified the requested option names exist.",
            "",
            "## Code / data separation",
            "",
            f"- code files in prepared dir: {summary.get('code_files_in_data_dir')}",
            f"- large/day data in code dir: {summary.get('large_data_in_code_dir')}",
            f"- april 9 output absent: `{summary.get('apr9_output_absent')}`",
            "",
            "## Per-day results",
            "",
            "| date | gen_ok | xml | topo | sumo | exit | vehicles | filtered | pid_gen | pid_val | issues |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary.get("daily") or []:
        issues = row.get("issues") or []
        issue_text = "; ".join(str(x) for x in issues[:3]) if issues else ""
        lines.append(
            "| {date} | {gen} | {xml} | {topo} | {sumo} | {exit} | {veh} | {filt} | {pg} | {pv} | {iss} |".format(
                date=row.get("date"),
                gen=row.get("generation_ok"),
                xml=row.get("xml_validation_ok"),
                topo=row.get("topology_validation_ok"),
                sumo=row.get("sumo_load_ok"),
                exit=row.get("sumo_exit_code"),
                veh=row.get("output_vehicles"),
                filt=row.get("filtered_ge_86400"),
                pg=row.get("generation_pid"),
                pv=row.get("validation_pid"),
                iss=issue_text.replace("|", "/"),
            )
        )
    if summary.get("hash_mismatches"):
        lines.extend(["", "## Hash mismatches", ""])
        for item in summary["hash_mismatches"]:
            lines.append(f"- {item}")
    failed = [row for row in (summary.get("daily") or []) if not row.get("ok")]
    lines.extend(["", "## Failures", ""])
    if not failed and summary.get("ok"):
        lines.append("All 29 dates passed generation, XML/topology checks, and SUMO load.")
    else:
        if not failed:
            lines.append("No per-day XML/SUMO failures, but a global check failed. See top-level flags.")
        for row in failed:
            lines.append(
                f"- {row.get('date')}: {(row.get('issues') or ['unknown failure'])[:5]} "
                f"log={row.get('log_path')}"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate prepared SUMO route files")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--net-file", type=Path, default=None)
    parser.add_argument("--workers", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    source_dir = args.source_dir.resolve()
    prepared_dir = args.prepared_dir.resolve()
    workers = int(args.workers)
    net_file = resolve_net_file(args.net_file).resolve()
    validation_dir = prepared_dir / "validation"
    log_dir = validation_dir / "logs"
    tmp_dir = prepared_dir / ".tmp"
    validation_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    main_log_path = log_dir / "validate_main.log"
    main_log = main_log_path.open("w", encoding="utf-8")

    def log(msg: str) -> None:
        log_print(main_log, msg)

    t_all = time.time()
    env = collect_env()
    log(f"validate start source={source_dir} prepared={prepared_dir} net={net_file} workers={workers}")
    log(f"env python={env.get('python_executable')} {env.get('python_version')}")
    log(f"env sumo={env.get('sumo_bin')} {env.get('sumo_version')} SUMO_HOME={env.get('SUMO_HOME')}")

    prep_path = validation_dir / "preparation_summary.json"
    gen_map = load_generation_map(prep_path)
    generation_workers = (json_load(prep_path).get("generation_workers") if prep_path.is_file() else None)
    generation_overlap = (json_load(prep_path).get("max_observed_overlap") if prep_path.is_file() else None)

    dates, missing, extras = discover_dates(source_dir)
    global_issues: List[str] = []
    if len(dates) != N_DATES_EXPECTED or missing or extras:
        global_issues.append(f"source dates invalid missing={missing} extras={extras}")
    route_files = prepared_route_files(prepared_dir)
    route_dates = [parse_route_date(path.name) for path in route_files]
    if "2023-04-09" in route_dates:
        global_issues.append("April 9 route file exists")
    unexpected_routes = [d for d in route_dates if d not in EXPECTED_DATES]
    missing_routes = [d for d in EXPECTED_DATES if d not in route_dates]
    if unexpected_routes:
        global_issues.append(f"unexpected route dates: {unexpected_routes}")
    if missing_routes:
        global_issues.append(f"missing route dates: {missing_routes}")
    if workers != N_DATES_EXPECTED:
        global_issues.append(f"--workers is {workers}, official run requires {N_DATES_EXPECTED}")

    log("hashing repaired sources after generation")
    after_rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(hash_file_row, source_dir / source_name_for_date(date), date): date
            for date in EXPECTED_DATES
        }
        for fut in as_completed(futs):
            after_rows.append(fut.result())
    after_rows.sort(key=lambda row: row["date"])
    write_tsv(
        validation_dir / "source_hashes_after.tsv",
        ["date", "path", "bytes", "mtime_iso", "mtime_ns", "sha256"],
        after_rows,
    )
    before_rows = load_hash_tsv(validation_dir / "source_hashes_before.tsv")
    hash_mismatches = compare_hash_rows(before_rows, after_rows)
    source_unchanged = not hash_mismatches and len(before_rows) == len(after_rows) == N_DATES_EXPECTED
    if not source_unchanged:
        global_issues.append("repaired source files changed or hash TSV mismatch")

    working_net_sha = sha256_file(net_file)
    original_now = snapshot_original_inputs()
    prep = json_load(prep_path) if prep_path.is_file() else {}
    net_before = (prep.get("working_net") or {}).get("sha256")
    orig_before = (prep.get("original_snapshot") or {}).get("original_net") or {}
    net_unchanged = net_before == working_net_sha if net_before else True
    original_net_unchanged = True
    if orig_before.get("sha256") and original_now.get("original_net"):
        original_net_unchanged = orig_before["sha256"] == original_now["original_net"]["sha256"]
    if not net_unchanged:
        global_issues.append("working SUMO net hash changed")
    if not original_net_unchanged:
        global_issues.append("original download net hash changed")

    log("loading topology")
    topo = load_topology(net_file)
    topo_path = tmp_dir / "topology.pkl"
    save_pickle_atomic(topo, topo_path, tmp_dir)

    payloads = []
    for date in EXPECTED_DATES:
        payloads.append(
            {
                "date": date,
                "source": str(source_dir / source_name_for_date(date)),
                "output": str(prepared_dir / route_name_for_date(date)),
                "net_file": str(net_file),
                "log_path": str(log_dir / f"validate_{date_to_tag(date)}.log"),
                "sumo_stdout_path": str(log_dir / f"sumo_{date_to_tag(date)}.stdout.log"),
                "sumo_stderr_path": str(log_dir / f"sumo_{date_to_tag(date)}.stderr.log"),
            }
        )

    log(f"submitting {len(payloads)} validation tasks with max_workers={workers}")
    submit_ts = time.time()
    daily: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(topo_path),),
    ) as pool:
        futs = {pool.submit(validate_one, payload): payload["date"] for payload in payloads}
        log(f"all {len(futs)} tasks submitted in {time.time() - submit_ts:.3f}s")
        for fut in as_completed(futs):
            date = futs[fut]
            try:
                row = fut.result()
            except Exception as exc:
                row = {
                    "date": date,
                    "ok": False,
                    "issues": [f"worker crashed: {type(exc).__name__}: {exc}"],
                    "xml_validation_ok": False,
                    "topology_validation_ok": False,
                    "sumo_load_ok": False,
                    "sumo_exit_code": None,
                    "validation_pid": None,
                    "validation_elapsed_s": None,
                }
            gen = gen_map.get(date) or {}
            row["generation_ok"] = bool(gen.get("ok"))
            row["generation_pid"] = gen.get("generation_pid")
            row["generation_elapsed_s"] = gen.get("generation_elapsed_s")
            row["source_depart_sorted"] = gen.get("source_depart_sorted")
            row["output_bytes"] = gen.get("output_bytes")
            row["output_sha256"] = gen.get("output_sha256")
            if not row.get("output_bytes") and Path(row["output_file"]).is_file():
                outp = Path(row["output_file"])
                row["output_bytes"] = outp.stat().st_size
                row["output_sha256"] = sha256_file(outp)
            daily.append(row)
            log(
                f"validation {'OK' if row.get('ok') else 'FAIL'} date={date} "
                f"pid={row.get('validation_pid')} xml={row.get('xml_validation_ok')} "
                f"topo={row.get('topology_validation_ok')} sumo={row.get('sumo_load_ok')} "
                f"exit={row.get('sumo_exit_code')} issues={row.get('issues')}"
            )

    daily.sort(key=lambda row: row["date"])
    intervals = [
        (row["worker_started_ts"], row["worker_ended_ts"])
        for row in daily
        if row.get("worker_started_ts") is not None and row.get("worker_ended_ts") is not None
    ]
    overlap = max_overlap(intervals) if intervals else 0

    code_in_data = scan_code_in_data(prepared_dir)
    data_in_code = scan_large_or_day_data_in_code(CODE_ROOT)
    if code_in_data:
        global_issues.append(f"code files in prepared dir: {code_in_data}")
    if data_in_code:
        global_issues.append(f"large/day data in code dir: {data_in_code}")
    apr9_absent = not (prepared_dir / route_name_for_date("2023-04-09")).exists()
    if not apr9_absent:
        global_issues.append("April 9 output present")

    totals = Counter()
    for row in daily:
        for key, dest in (
            ("source_records", "total_source_records"),
            ("filtered_ge_86400", "total_filtered_ge_86400"),
            ("output_vehicles", "total_output_vehicles"),
            ("passenger_count", "total_passenger"),
            ("freight_count", "total_freight"),
            ("other_fingerprint_count", "total_other_fingerprints"),
            ("unknown_edges", "total_unknown_edges"),
            ("internal_edges", "total_internal_edges"),
            ("illegal_pairs", "total_illegal_pairs"),
            ("duplicate_vehicle_ids", "total_duplicate_vehicle_ids"),
            ("invalid_depart", "total_invalid_depart"),
            ("unsorted_depart", "total_unsorted_depart"),
        ):
            val = row.get(key)
            if isinstance(val, (int, float)) and val is not None:
                totals[dest] += int(val)
    sumo_failures = sum(1 for row in daily if not row.get("sumo_load_ok"))
    all_dates_ok = all(row.get("ok") and row.get("generation_ok") for row in daily) and len(daily) == N_DATES_EXPECTED
    n_generated = len([d for d in EXPECTED_DATES if (prepared_dir / route_name_for_date(d)).is_file()])
    count_formula_ok = totals["total_output_vehicles"] == (
        totals["total_source_records"] - totals["total_filtered_ge_86400"]
    )
    if not count_formula_ok:
        global_issues.append("global count formula failed")
    if totals["total_other_fingerprints"]:
        global_issues.append("other fingerprints are not zero")
    if sumo_failures:
        global_issues.append(f"SUMO load failures={sumo_failures}")
    if overlap != workers:
        # Overlap can theoretically be <29 if tasks finish staggeringly fast, but
        # 29 submitted workers should overlap for this workload. Record, do not
        # silently claim 29-way concurrency if it did not happen.
        if overlap < workers:
            global_issues.append(
                f"validation max overlap was {overlap}, expected {workers} concurrent workers"
            )

    ok = (
        all_dates_ok
        and source_unchanged
        and net_unchanged
        and original_net_unchanged
        and n_generated == N_DATES_EXPECTED
        and not code_in_data
        and not data_in_code
        and apr9_absent
        and not hash_mismatches
        and workers == N_DATES_EXPECTED
        and generation_workers == N_DATES_EXPECTED
        and not global_issues
        and totals["total_other_fingerprints"] == 0
        and totals["total_unknown_edges"] == 0
        and totals["total_internal_edges"] == 0
        and totals["total_illegal_pairs"] == 0
        and totals["total_duplicate_vehicle_ids"] == 0
        and totals["total_invalid_depart"] == 0
        and totals["total_unsorted_depart"] == 0
        and sumo_failures == 0
        and count_formula_ok
    )

    summary = {
        "ok": ok,
        "n_dates_expected": N_DATES_EXPECTED,
        "n_dates_generated": n_generated,
        "generation_workers": generation_workers,
        "validation_workers": workers,
        "generation_overlap": generation_overlap,
        "validation_overlap": overlap,
        "all_dates_ok": all_dates_ok,
        "source_unchanged": source_unchanged,
        "net_unchanged": net_unchanged,
        "original_net_unchanged": original_net_unchanged,
        "total_source_records": totals["total_source_records"],
        "total_filtered_ge_86400": totals["total_filtered_ge_86400"],
        "total_output_vehicles": totals["total_output_vehicles"],
        "total_passenger": totals["total_passenger"],
        "total_freight": totals["total_freight"],
        "total_other_fingerprints": totals["total_other_fingerprints"],
        "total_unknown_edges": totals["total_unknown_edges"],
        "total_internal_edges": totals["total_internal_edges"],
        "total_illegal_pairs": totals["total_illegal_pairs"],
        "total_duplicate_vehicle_ids": totals["total_duplicate_vehicle_ids"],
        "total_invalid_depart": totals["total_invalid_depart"],
        "total_unsorted_depart": totals["total_unsorted_depart"],
        "sumo_load_failures": sumo_failures,
        "hash_mismatches": hash_mismatches,
        "code_files_in_data_dir": code_in_data,
        "large_data_in_code_dir": data_in_code,
        "apr9_output_absent": apr9_absent,
        "global_issues": global_issues,
        "env": env,
        "net_file": str(net_file),
        "net_sha256": working_net_sha,
        "original_net_sha256": (original_now.get("original_net") or {}).get("sha256"),
        "elapsed_s": round(time.time() - t_all, 6),
        "daily": daily,
        "code_root": str(CODE_ROOT),
        "source_root": str(SOURCE_ROOT),
        "data_root": str(DATA_ROOT),
        "original_net_file": str(ORIGINAL_NET_FILE),
    }
    write_json_atomic(summary, validation_dir / "validation_summary.json", tmp_dir)

    manifest_rows = []
    for row in daily:
        gen = gen_map.get(row["date"]) or {}
        manifest_rows.append(
            {
                "date": row.get("date"),
                "source_file": row.get("source_file") or gen.get("source_file"),
                "output_file": row.get("output_file") or gen.get("output_file"),
                "source_records": row.get("source_records"),
                "filtered_ge_86400": row.get("filtered_ge_86400"),
                "output_vehicles": row.get("output_vehicles"),
                "passenger_count": row.get("passenger_count"),
                "freight_count": row.get("freight_count"),
                "other_fingerprint_count": row.get("other_fingerprint_count"),
                "min_depart": row.get("min_depart"),
                "max_depart": row.get("max_depart"),
                "min_route_length": row.get("min_route_length"),
                "max_route_length": row.get("max_route_length"),
                "mean_route_length": row.get("mean_route_length"),
                "source_depart_sorted": bool_tsv(row.get("source_depart_sorted")),
                "output_bytes": row.get("output_bytes") or gen.get("output_bytes"),
                "output_sha256": row.get("output_sha256") or gen.get("output_sha256"),
                "generation_pid": row.get("generation_pid"),
                "generation_elapsed_s": row.get("generation_elapsed_s"),
                "xml_validation_ok": bool_tsv(row.get("xml_validation_ok")),
                "topology_validation_ok": bool_tsv(row.get("topology_validation_ok")),
                "sumo_load_ok": bool_tsv(row.get("sumo_load_ok")),
                "sumo_exit_code": row.get("sumo_exit_code"),
                "validation_pid": row.get("validation_pid"),
                "validation_elapsed_s": row.get("validation_elapsed_s"),
            }
        )
    write_tsv(validation_dir / "prepared_manifest.tsv", MANIFEST_FIELDS, manifest_rows)
    write_text_atomic(build_report(summary), validation_dir / "validation_report.md", tmp_dir)
    log(
        f"validation done ok={ok} overlap={overlap} vehicles={summary['total_output_vehicles']} "
        f"sumo_fail={sumo_failures} source_unchanged={source_unchanged} issues={global_issues}"
    )
    main_log.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
