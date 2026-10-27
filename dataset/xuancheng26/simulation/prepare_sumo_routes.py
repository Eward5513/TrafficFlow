#!/usr/bin/env python3
"""Convert repaired Xuancheng26 day-level JSON trips into explicit SUMO route files."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from repair_lib import iter_json_array, sha256_file
from sumo_route_common import (
    CODE_ROOT,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    DEPART_LIMIT,
    EXCLUDED_DATES,
    EXPECTED_DATES,
    MIN_ROUTE_EDGES,
    N_DATES_EXPECTED,
    ORIGINAL_NET_FILE,
    PrepareError,
    ROUTES_HEADER,
    classify_vehicle,
    coerce_route,
    collect_env,
    date_to_tag,
    discover_dates,
    fingerprint_as_dict,
    format_depart,
    hash_file_row,
    load_pickle,
    load_topology,
    log_print,
    max_overlap,
    parse_start_time,
    resolve_net_file,
    route_name_for_date,
    save_pickle_atomic,
    snapshot_original_inputs,
    source_name_for_date,
    vehicle_fingerprint,
    vehicle_id_for,
    write_json_atomic,
    write_tsv,
    xml_attr,
)

WORKER_STARTED_AT: Optional[float] = None


def _init_worker(topo_path: str) -> None:
    global WORKER_STARTED_AT
    WORKER_STARTED_AT = time.time()
    # Load topology once per process so 29 workers do not re-parse the net.
    _init_worker.topology = load_pickle(Path(topo_path))  # type: ignore[attr-defined]


def _topology() -> Any:
    topo = getattr(_init_worker, "topology", None)
    if topo is None:
        raise PrepareError("worker topology was not initialized")
    return topo


def check_route_topology(route: List[str], topo: Any) -> Tuple[int, int, int, Optional[str]]:
    unknown = 0
    internal = 0
    illegal = 0
    first_error = None
    if len(route) < MIN_ROUTE_EDGES:
        return 0, 0, 0, f"route_length={len(route)} < {MIN_ROUTE_EDGES}"
    for edge in route:
        kind = topo.classify_edge(edge)
        if kind == "internal":
            internal += 1
            if first_error is None:
                first_error = f"internal_edge={edge}"
        elif kind == "unknown":
            unknown += 1
            if first_error is None:
                first_error = f"unknown_edge={edge}"
    for frm, to in zip(route, route[1:]):
        if not topo.has_connection(frm, to):
            illegal += 1
            if first_error is None:
                first_error = f"illegal_pair={frm}->{to}"
    return unknown, internal, illegal, first_error


def write_vehicle(fh, veh_id: str, vtype: str, depart_text: str, edges: List[str]) -> None:
    fh.write('    <vehicle id="')
    fh.write(xml_attr(veh_id))
    fh.write('"\n             type="')
    fh.write(xml_attr(vtype))
    fh.write('"\n             depart="')
    fh.write(xml_attr(depart_text))
    fh.write('">\n        <route edges="')
    fh.write(xml_attr(" ".join(edges)))
    fh.write('"/>\n    </vehicle>\n')


def generate_one(payload: Dict[str, Any]) -> Dict[str, Any]:
    date = payload["date"]
    source = Path(payload["source"])
    output = Path(payload["output"])
    tmp_dir = Path(payload["tmp_dir"])
    log_path = Path(payload["log_path"])
    pid = os.getpid()
    started = WORKER_STARTED_AT if WORKER_STARTED_AT is not None else time.time()
    t0 = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{output.name}.{pid}"
    errors: List[str] = []
    warnings: List[str] = []
    other_fps: Dict[str, Dict[str, Any]] = {}

    def finish(ok: bool, extra: Dict[str, Any]) -> Dict[str, Any]:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        if (not ok) and output.exists():
            # Do not leave a previous file looking like a successful new result
            # when this run failed; only delete if we created a tmp this run.
            pass
        elapsed = time.time() - t0
        row = {
            "date": date,
            "ok": ok,
            "source_file": str(source),
            "output_file": str(output),
            "generation_pid": pid,
            "worker_started_ts": started,
            "worker_ended_ts": time.time(),
            "generation_elapsed_s": round(elapsed, 6),
            "errors": errors,
            "warnings": warnings,
            "other_fingerprints": list(other_fps.values()),
            "log_path": str(log_path),
        }
        row.update(extra)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(
                f"{time.strftime('%Y-%m-%dT%H:%M:%S')} pid={pid} "
                f"ok={ok} elapsed={elapsed:.3f}s errors={errors}\n"
            )
        return row

    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(
            f"{time.strftime('%Y-%m-%dT%H:%M:%S')} pid={pid} start date={date} "
            f"source={source}\n"
        )

    try:
        topo = _topology()
        source_records = 0
        filtered = 0
        passenger = 0
        freight = 0
        other = 0
        unknown_edges = 0
        internal_edges = 0
        illegal_pairs = 0
        route_len_min = None
        route_len_max = None
        route_len_sum = 0
        min_depart = None
        max_depart = None
        prev_depart = None
        source_sorted = True
        invalid_start = 0
        output_vehicles = 0

        tmp_dir.mkdir(parents=True, exist_ok=True)
        if tmp_path.exists():
            tmp_path.unlink()
        fh = tmp_path.open("w", encoding="utf-8")
        try:
            fh.write(ROUTES_HEADER)
            for rec in iter_json_array(source):
                idx = source_records
                source_records += 1
                try:
                    start_raw = rec.get("startTime")
                    depart = parse_start_time(start_raw)
                except PrepareError as exc:
                    invalid_start += 1
                    errors.append(f"record[{idx}] {exc}")
                    continue
                if depart < 0:
                    invalid_start += 1
                    errors.append(f"record[{idx}] negative startTime={start_raw!r}")
                    continue
                if prev_depart is not None and depart < prev_depart:
                    source_sorted = False
                prev_depart = depart
                interval = rec.get("interval")
                if interval != 1:
                    warnings.append(f"record[{idx}] interval={interval!r} (not expanded)")
                if rec.get("endTime") != rec.get("startTime"):
                    warnings.append(
                        f"record[{idx}] endTime={rec.get('endTime')!r} != startTime={start_raw!r}"
                    )
                vtype = classify_vehicle(rec.get("vehicle"))
                if vtype == "other":
                    other += 1
                    veh = rec.get("vehicle") if isinstance(rec.get("vehicle"), dict) else {"_raw": rec.get("vehicle")}
                    fp_key = str(sorted(veh.items()) if isinstance(veh, dict) else veh)
                    if fp_key not in other_fps:
                        other_fps[fp_key] = {
                            "date": date,
                            "count": 0,
                            "first_index": idx,
                            "vehicle": veh,
                            "fingerprint": fingerprint_as_dict(vehicle_fingerprint(veh))
                            if isinstance(veh, dict)
                            else {"_raw": veh},
                        }
                    other_fps[fp_key]["count"] += 1
                    if len(errors) < 20:
                        errors.append(f"record[{idx}] unknown vehicle fingerprint")
                    continue
                if depart >= DEPART_LIMIT:
                    filtered += 1
                    continue
                try:
                    edges = coerce_route(rec.get("route"))
                except PrepareError as exc:
                    errors.append(f"record[{idx}] {exc}")
                    continue
                n_unknown, n_internal, n_illegal, route_err = check_route_topology(edges, topo)
                unknown_edges += n_unknown
                internal_edges += n_internal
                illegal_pairs += n_illegal
                if route_err is not None:
                    if len(errors) < 50:
                        errors.append(f"record[{idx}] {route_err}")
                    continue
                rlen = len(edges)
                if route_len_min is None or rlen < route_len_min:
                    route_len_min = rlen
                if route_len_max is None or rlen > route_len_max:
                    route_len_max = rlen
                route_len_sum += rlen
                if min_depart is None or depart < min_depart:
                    min_depart = depart
                if max_depart is None or depart > max_depart:
                    max_depart = depart
                if vtype == "passenger":
                    passenger += 1
                else:
                    freight += 1
                write_vehicle(
                    fh,
                    vehicle_id_for(date, idx),
                    vtype,
                    format_depart(start_raw, depart),
                    edges,
                )
                output_vehicles += 1
            fh.write("</routes>\n")
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fh.close()

        if errors:
            return finish(
                False,
                {
                    "source_records": source_records,
                    "filtered_ge_86400": filtered,
                    "output_vehicles": 0,
                    "passenger_count": passenger,
                    "freight_count": freight,
                    "other_fingerprint_count": other,
                    "unknown_edges": unknown_edges,
                    "internal_edges": internal_edges,
                    "illegal_pairs": illegal_pairs,
                    "invalid_startTime_errors": invalid_start,
                    "source_depart_sorted": source_sorted,
                    "output_bytes": None,
                    "output_sha256": None,
                    "min_depart": min_depart,
                    "max_depart": max_depart,
                    "min_route_length": route_len_min,
                    "max_route_length": route_len_max,
                    "mean_route_length": None,
                },
            )
        if other > 0:
            errors.append(f"unknown vehicle fingerprints: {other}")
            return finish(False, {"source_records": source_records, "other_fingerprint_count": other})
        if unknown_edges or internal_edges or illegal_pairs:
            errors.append(
                f"topology failures unknown={unknown_edges} internal={internal_edges} illegal={illegal_pairs}"
            )
            return finish(
                False,
                {
                    "source_records": source_records,
                    "unknown_edges": unknown_edges,
                    "internal_edges": internal_edges,
                    "illegal_pairs": illegal_pairs,
                },
            )
        expected_out = source_records - filtered
        if output_vehicles != expected_out:
            errors.append(
                f"count mismatch output={output_vehicles} != source={source_records} - filtered={filtered}"
            )
            return finish(False, {"source_records": source_records, "output_vehicles": output_vehicles})
        if not source_sorted:
            warnings.append("source was not depart-sorted; output still follows source order which is required to be sorted")
            errors.append("source startTime is not non-decreasing; refusing to emit an unsorted SUMO route file")
            return finish(
                False,
                {
                    "source_records": source_records,
                    "source_depart_sorted": False,
                    "filtered_ge_86400": filtered,
                    "output_vehicles": output_vehicles,
                },
            )

        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp_path, output)
        out_bytes = output.stat().st_size
        out_sha = sha256_file(output)
        mean_len = (route_len_sum / output_vehicles) if output_vehicles else None
        extra = {
            "source_records": source_records,
            "filtered_ge_86400": filtered,
            "output_vehicles": output_vehicles,
            "passenger_count": passenger,
            "freight_count": freight,
            "other_fingerprint_count": other,
            "unknown_edges": unknown_edges,
            "internal_edges": internal_edges,
            "illegal_pairs": illegal_pairs,
            "invalid_startTime_errors": invalid_start,
            "min_depart": min_depart,
            "max_depart": max_depart,
            "min_route_length": route_len_min,
            "max_route_length": route_len_max,
            "mean_route_length": None if mean_len is None else round(mean_len, 6),
            "source_depart_sorted": source_sorted,
            "output_bytes": out_bytes,
            "output_sha256": out_sha,
            "count_formula_ok": output_vehicles == source_records - filtered,
        }
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(
                f"{time.strftime('%Y-%m-%dT%H:%M:%S')} pid={pid} wrote {output} "
                f"vehicles={output_vehicles} filtered={filtered} bytes={out_bytes}\n"
            )
        return finish(True, extra)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        return finish(
            False,
            {
                "source_records": None,
                "filtered_ge_86400": None,
                "output_vehicles": None,
                "exception": repr(exc),
            },
        )


def precheck(input_dir: Path, output_dir: Path, net_file: Path, workers: int) -> Dict[str, Any]:
    dates, missing, extras = discover_dates(input_dir)
    issues: List[str] = []
    if len(dates) != N_DATES_EXPECTED:
        issues.append(f"repaired date count is {len(dates)}, expected {N_DATES_EXPECTED}")
    if missing:
        issues.append(f"missing dates: {missing}")
    if extras:
        issues.append(f"unexpected repaired files/dates: {extras}")
    for excluded in EXCLUDED_DATES:
        excluded_path = input_dir / source_name_for_date(excluded)
        if excluded_path.exists():
            issues.append(f"excluded date is present: {excluded_path}")
    if not net_file.is_file():
        issues.append(f"net file missing: {net_file}")
    if workers != N_DATES_EXPECTED:
        issues.append(f"--workers is {workers}, official run requires {N_DATES_EXPECTED}")
    sample_path = input_dir / source_name_for_date(EXPECTED_DATES[0])
    sample_keys = None
    sample_vehicle_keys = None
    sample_ok = False
    if sample_path.is_file():
        first = next(iter_json_array(sample_path), None)
        if not isinstance(first, dict):
            issues.append(f"{sample_path.name} first record is not an object")
        else:
            sample_keys = sorted(first.keys())
            veh = first.get("vehicle")
            sample_vehicle_keys = sorted(veh.keys()) if isinstance(veh, dict) else None
            required = {"vehicle", "interval", "startTime", "endTime", "route"}
            if not required.issubset(first.keys()):
                issues.append(f"sample record missing keys {sorted(required - set(first.keys()))}")
            elif not isinstance(first.get("route"), list) or not first["route"]:
                issues.append("sample record route is not a non-empty list")
            else:
                sample_ok = True
    else:
        issues.append(f"sample file missing: {sample_path}")
    env = collect_env()
    if not env.get("sumo_bin"):
        issues.append("sumo binary not found on PATH")
    return {
        "ok": not issues,
        "issues": issues,
        "dates": dates,
        "missing": missing,
        "extras": extras,
        "net_file": str(net_file),
        "sample_file": str(sample_path),
        "sample_ok": sample_ok,
        "sample_keys": sample_keys,
        "sample_vehicle_keys": sample_vehicle_keys,
        "env": env,
        "workers": workers,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SUMO route files from repaired JSON")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--net-file", type=Path, default=None)
    parser.add_argument("--workers", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    workers = int(args.workers)
    net_file = resolve_net_file(args.net_file).resolve()

    validation_dir = output_dir / "validation"
    log_dir = validation_dir / "logs"
    tmp_dir = output_dir / ".tmp"
    validation_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    main_log_path = log_dir / "prepare_main.log"
    main_log = main_log_path.open("w", encoding="utf-8")

    def log(msg: str) -> None:
        log_print(main_log, msg)

    t_all = time.time()
    log(f"prepare start input={input_dir} output={output_dir} net={net_file} workers={workers}")
    pre = precheck(input_dir, output_dir, net_file, workers)
    log(f"precheck ok={pre['ok']} dates={len(pre['dates'])} issues={pre['issues']}")
    log(f"env python={pre['env'].get('python_executable')} {pre['env'].get('python_version')}")
    log(f"env sumo={pre['env'].get('sumo_bin')} {pre['env'].get('sumo_version')}")
    log(f"env SUMO_HOME={pre['env'].get('SUMO_HOME')}")
    log(f"sample_keys={pre.get('sample_keys')} vehicle_keys={pre.get('sample_vehicle_keys')}")
    if not pre["ok"]:
        summary = {
            "ok": False,
            "stage": "precheck",
            "precheck": pre,
            "n_dates_expected": N_DATES_EXPECTED,
            "n_dates_generated": 0,
            "generation_workers": workers,
        }
        write_json_atomic(summary, validation_dir / "preparation_summary.json", tmp_dir)
        log("precheck failed; not generating routes")
        main_log.close()
        return 1

    log("hashing 29 repaired source files")
    source_rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(hash_file_row, input_dir / source_name_for_date(date), date): date
            for date in EXPECTED_DATES
        }
        for fut in as_completed(futs):
            source_rows.append(fut.result())
    source_rows.sort(key=lambda row: row["date"])
    write_tsv(
        validation_dir / "source_hashes_before.tsv",
        ["date", "path", "bytes", "mtime_iso", "mtime_ns", "sha256"],
        source_rows,
    )
    log("source hashes written")

    working_net_meta = {
        "path": str(net_file),
        "sha256": sha256_file(net_file),
        "bytes": net_file.stat().st_size,
        "mtime_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(net_file.stat().st_mtime)),
    }
    original_snapshot = snapshot_original_inputs()
    log(f"working_net sha256={working_net_meta['sha256']}")
    if original_snapshot.get("original_net"):
        log(f"original_net sha256={original_snapshot['original_net']['sha256']}")

    log("parsing SUMO net for topology")
    topo = load_topology(net_file)
    if topo.net_sha256 != working_net_meta["sha256"]:
        log("ERROR topology hash mismatch against working net")
        main_log.close()
        return 1
    topo_path = tmp_dir / "topology.pkl"
    save_pickle_atomic(topo, topo_path, tmp_dir)
    log(
        f"topology normal={len(topo.normal_set)} internal={len(topo.internal_set)} "
        f"adj_sources={len(topo.adj_set)}"
    )

    payloads = []
    for date in EXPECTED_DATES:
        payloads.append(
            {
                "date": date,
                "source": str(input_dir / source_name_for_date(date)),
                "output": str(output_dir / route_name_for_date(date)),
                "tmp_dir": str(tmp_dir),
                "log_path": str(log_dir / f"generate_{date_to_tag(date)}.log"),
            }
        )

    log(f"submitting {len(payloads)} generation tasks with max_workers={workers}")
    submit_ts = time.time()
    daily: List[Dict[str, Any]] = []
    n_fail = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(topo_path),),
    ) as pool:
        futs = {pool.submit(generate_one, payload): payload["date"] for payload in payloads}
        submitted = time.time()
        log(f"all {len(futs)} tasks submitted in {submitted - submit_ts:.3f}s")
        for fut in as_completed(futs):
            date = futs[fut]
            try:
                result = fut.result()
            except Exception as exc:
                result = {
                    "date": date,
                    "ok": False,
                    "errors": [f"worker crashed: {type(exc).__name__}: {exc}"],
                    "generation_pid": None,
                    "generation_elapsed_s": None,
                }
            daily.append(result)
            status = "OK" if result.get("ok") else "FAIL"
            log(
                f"generation {status} date={date} pid={result.get('generation_pid')} "
                f"elapsed={result.get('generation_elapsed_s')} "
                f"vehicles={result.get('output_vehicles')} filtered={result.get('filtered_ge_86400')} "
                f"errors={result.get('errors')}"
            )
            if not result.get("ok"):
                n_fail += 1

    daily.sort(key=lambda row: row["date"])
    intervals = [
        (row["worker_started_ts"], row["worker_ended_ts"])
        for row in daily
        if row.get("worker_started_ts") is not None and row.get("worker_ended_ts") is not None
    ]
    overlap = max_overlap(intervals) if intervals else 0
    n_ok = sum(1 for row in daily if row.get("ok"))
    n_generated_files = sum(
        1 for date in EXPECTED_DATES if (output_dir / route_name_for_date(date)).is_file()
    )
    all_ok = n_fail == 0 and n_ok == N_DATES_EXPECTED and n_generated_files == N_DATES_EXPECTED
    totals = Counter()
    for row in daily:
        for key in (
            "source_records",
            "filtered_ge_86400",
            "output_vehicles",
            "passenger_count",
            "freight_count",
            "other_fingerprint_count",
            "unknown_edges",
            "internal_edges",
            "illegal_pairs",
        ):
            val = row.get(key)
            if isinstance(val, (int, float)) and val is not None:
                totals[key] += int(val)

    summary = {
        "ok": all_ok,
        "stage": "generation",
        "n_dates_expected": N_DATES_EXPECTED,
        "n_dates_generated": n_generated_files,
        "n_dates_ok": n_ok,
        "generation_workers": workers,
        "max_observed_overlap": overlap,
        "workers_were_concurrent_29": overlap == workers == N_DATES_EXPECTED,
        "all_dates_ok": all_ok,
        "excluded_dates": list(EXCLUDED_DATES),
        "net_file": str(net_file),
        "net_sha256": working_net_meta["sha256"],
        "original_net_sha256": (original_snapshot.get("original_net") or {}).get("sha256"),
        "precheck": {
            "ok": pre["ok"],
            "sample_keys": pre.get("sample_keys"),
            "sample_vehicle_keys": pre.get("sample_vehicle_keys"),
            "issues": pre.get("issues"),
            "env": pre.get("env"),
        },
        "original_snapshot": original_snapshot,
        "working_net": working_net_meta,
        "source_hashes_before": str(validation_dir / "source_hashes_before.tsv"),
        "totals": dict(totals),
        "daily": daily,
        "elapsed_s": round(time.time() - t_all, 6),
        "code_root": str(CODE_ROOT),
        "original_net_file": str(ORIGINAL_NET_FILE),
    }
    write_json_atomic(summary, validation_dir / "preparation_summary.json", tmp_dir)
    log(
        f"generation done ok={all_ok} n_ok={n_ok}/{N_DATES_EXPECTED} overlap={overlap} "
        f"vehicles={totals.get('output_vehicles')} elapsed={summary['elapsed_s']:.3f}s"
    )
    main_log.close()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
