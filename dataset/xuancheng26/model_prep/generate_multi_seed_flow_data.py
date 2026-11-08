#!/usr/bin/env python3
"""Stage 2: stream each missing date once, then audit all four states and windows.

Consumes the passed stage-1 manifest. Never chooses or resamples vehicle IDs.
Legacy files are read-only; only verified file symlinks enter the new flow tree.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import multiprocessing as mp
import os
from pathlib import Path
import sys
import traceback

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

import generate_r_level_full_flow as full
import generate_r_level_windows as windows
import multi_seed_common as common

CONTEXT = {}


def read_ids(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader) != ["vehicle_id"]:
            raise RuntimeError(f"invalid ID schema: {path}")
        rows = list(reader)
    if any(len(row) != 1 or not row[0] for row in rows):
        raise RuntimeError(f"invalid ID row: {path}")
    return [row[0] for row in rows]


def load_states(directory, edges):
    states = {}
    for name in common.CHANNELS:
        path = directory / f"{name}.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            if next(csv.reader(handle)) != ["time_bin", "start_time", *edges]:
                raise RuntimeError(f"road order/header mismatch: {path}")
        states[name] = windows.load_aligned_matrix(path, edges)
    check_states(states, str(directory))
    return states


def check_states(states, label, complete=None, previous=None):
    start, end = full.accumulate_states(states["inflow"], states["outflow"], CONTEXT["edges"], label)
    for name, expected in (("accumulation_start", start), ("accumulation_end", end)):
        if not np.array_equal(states[name], expected):
            raise RuntimeError(f"{label}/{name}: conservation/continuity/initial stock failed")
    for name, values in states.items():
        if values.shape != (288, 1406) or not np.isfinite(values).all() or np.any(values < 0):
            raise RuntimeError(f"{label}/{name}: invalid count matrix")
        for bound, relation, upper in ((complete, "partial exceeds full", True), (previous, "rate nesting", False)):
            if bound is not None:
                bad = values > bound[name] if upper else values < bound[name]
                if np.any(bad):
                    row, col = np.argwhere(bad)[0]
                    raise RuntimeError(f"{label}/{name}: {relation} slot={row}, edge={CONTEXT['edges'][col]}")


def expected_arrays(partial, complete, rate, day, names):
    norm = CONTEXT["normalization"]
    xs, ys = [], []
    for name in names:
        x, y, starts, targets = windows.make_windows(partial[name], complete[name])
        xs.append(x)
        ys.append(y)
    x_raw = np.concatenate(xs, axis=-1) if len(xs) > 1 else xs[0]
    y_raw = np.concatenate(ys, axis=-1) if len(ys) > 1 else ys[0]
    mx, sx, my, sy = [np.array(values, dtype=np.float32).reshape(1, 1, 1, -1) for values in (
        [norm["input"][rate][n]["mean"] for n in names], [norm["input"][rate][n]["std"] for n in names],
        [norm["target"][n]["mean"] for n in names], [norm["target"][n]["std"] for n in names])]
    return {"x": windows._normalize(x_raw, mx, sx), "y": windows._normalize(y_raw, my, sy),
            "x_raw": np.ascontiguousarray(x_raw), "y_raw": np.ascontiguousarray(y_raw),
            "day_index": np.full(277, windows.weekday_index(day), dtype=np.int32),
            "window_start_slot": starts, "target_slot": targets}


def verify_npz(path, arrays):
    with np.load(path, allow_pickle=False) as got:
        if tuple(got.files) != windows.NPZ_KEYS:
            raise RuntimeError(f"{path}: wrong NPZ keys/order")
        for name, expected in arrays.items():
            actual = got[name]
            if actual.dtype != expected.dtype or actual.shape != expected.shape:
                raise RuntimeError(f"{path}/{name}: dtype/shape mismatch")
            if not np.isfinite(actual).all() or not np.array_equal(actual, expected):
                raise RuntimeError(f"{path}/{name}: arrays differ from CSV/window indices/shared scaler")


def prepare_windows(root, seed, rate, split, day, states, complete, protected):
    output = common.window_path(root, seed, rate, split, day)
    old = common.historical(root, seed, rate, split, day, "window")
    records, actions = [], Counter()
    for name in windows.STATE_FILES:
        names = common.CHANNELS if name == "all" else (name,)
        arrays = expected_arrays(states, complete, rate, day, names)
        path = output / f"{name}.npz"
        source = old / path.name if old else None
        if source:
            before = common.signature(source)
            verify_npz(source, arrays)
            protected.append(before)
        if path.exists() or path.is_symlink():
            # Same inode/target was just checked; otherwise check every array here.
            if source is None or path.resolve() != source.resolve():
                verify_npz(path, arrays)
            action = "skipped"
        elif CONTEXT["verify_only"]:
            raise FileNotFoundError(path)
        elif source:
            common.link_existing(path, source)
            action = "reused"
        else:
            with common.new_file(path, root) as temp:
                with temp.open("wb") as handle:
                    np.savez(handle, **arrays)
                verify_npz(temp, arrays)
            action = "generated"
        actions[action] += 1
        records.append({**common.signature(path), "state": name, "action": action,
                        "source": str(source) if source else None})
        del arrays
    return records, actions


def count_pending(day, ids, pending):
    """Count disjoint minimum-rate tiers, then cumulative-sum along rate.

    Each event costs at most one increment per seed, instead of seven increments.
    Existing groups never need re-counting. A gap in pending rates is harmless:
    the first pending prefix owns all of its vehicles.
    """
    owners = {}
    for si, seed in enumerate(common.SEEDS):
        minimum = {}
        for ri, rate in enumerate(common.RATES):
            if (seed, rate) in pending:
                for vehicle in ids[seed, rate]:
                    minimum.setdefault(vehicle, ri)
        for vehicle, ri in minimum.items():
            owners.setdefault(vehicle, []).append((si, ri))
    counts = np.zeros((2, 5, 7, 288, 1406), dtype=np.int64)

    def on_event(kind, slot, column, vehicle):
        target = counts[0 if kind == "entry" else 1]
        for si, ri in owners[vehicle]:
            target[si, ri, slot, column] += 1

    fcd_root = CONTEXT["fcd"]
    full.READ_BYTES = 16 << 20
    arrivals = full.load_arrivals(full.prefer_xml(fcd_root / day, f"tripinfo_{day}"))
    path = full.prefer_xml(fcd_root / day, f"fcd_{day}")
    full.log(f"[{day}] single FCD scan; missing groups={len(pending)} bytes={path.stat().st_size}")
    _, _, stats, vehicles = full.stream_fcd(path, CONTEXT["edges"], CONTEXT["lanes"], CONTEXT["internal"],
                                           arrivals, day, include=set(owners), on_event=on_event)
    np.cumsum(counts, axis=2, out=counts)
    return counts, {"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
                    "scans": 1, "event_counts": stats, "included_vehicles": vehicles}


def process_day(day):
    root = CONTEXT["root"]
    try:
        split = next(s for s, days in CONTEXT["splits"].items() if day in days)
        edges = CONTEXT["edges"]
        ids, protected = {}, []
        for seed in common.SEEDS:
            previous = set()
            for rate in common.RATES:
                row = CONTEXT["sampling_records"][seed, rate, day]
                path = common.vehicle_path(root, seed, rate, day)
                if str(path) != row["vehicle_file"] or common.sha256(path) != row["sha256"]:
                    raise RuntimeError(f"stage-1 file changed: {path}")
                chosen = read_ids(path)
                chosen_set = set(chosen)
                if len(chosen) != row["sample_count"] or len(chosen) != len(chosen_set) or not previous <= chosen_set:
                    raise RuntimeError(f"stage-1 count/uniqueness/nesting invalid: {path}")
                previous = chosen_set
                ids[seed, rate] = chosen
        complete_dir = root / "full_flow/r-level" / day
        protected.extend(common.signature(complete_dir / f"{n}.csv") for n in common.CHANNELS)
        complete = load_states(complete_dir, edges)
        states_by_group, pending, flow_records = {}, set(), {}
        # Audit and reuse existing CSVs first, before deciding whether FCD is needed.
        for seed in common.SEEDS:
            for rate in common.RATES:
                out = common.flow_path(root, seed, rate, day)
                old = common.historical(root, seed, rate, split, day, "flow")
                if old:
                    protected.extend(common.signature(old / f"{n}.csv") for n in common.CHANNELS)
                    original = load_states(old, edges)
                    check_states(original, f"{day}/{seed}/{rate}", complete)
                    for name in common.CHANNELS:
                        dest = out / f"{name}.csv"
                        if not dest.exists():
                            if CONTEXT["verify_only"]:
                                raise FileNotFoundError(dest)
                            common.link_existing(dest, old / dest.name)
                if all((out / f"{n}.csv").is_file() for n in common.CHANNELS):
                    states = load_states(out, edges)
                    check_states(states, f"{day}/{seed}/{rate}", complete)
                    if old and any(not np.array_equal(states[n], original[n]) for n in common.CHANNELS):
                        raise RuntimeError(f"historical states differ: {out}")
                    states_by_group[seed, rate] = states
                    flow_records[seed, rate] = "reused" if old else "skipped"
                else:
                    if CONTEXT["verify_only"]:
                        raise FileNotFoundError(out)
                    pending.add((seed, rate))
        scan = {"scans": 0}
        if pending:
            counts, scan = count_pending(day, ids, pending)
            for si, seed in enumerate(common.SEEDS):
                for ri, rate in enumerate(common.RATES):
                    if (seed, rate) not in pending:
                        continue
                    inflow, outflow = counts[:, si, ri]
                    start, end = full.accumulate_states(inflow, outflow, edges, f"{day}/{seed}/{rate}")
                    states = dict(zip(common.CHANNELS, (inflow.copy(), outflow.copy(), start, end)))
                    check_states(states, f"{day}/{seed}/{rate}", complete)
                    out = common.flow_path(root, seed, rate, day)
                    for name, matrix in states.items():
                        path = out / f"{name}.csv"
                        if path.exists() or path.is_symlink():
                            if not np.array_equal(windows.load_aligned_matrix(path, edges), matrix):
                                raise RuntimeError(f"partial existing CSV differs: {path}")
                        else:
                            with common.new_file(path, root) as temp:
                                full.write_matrix(temp, matrix, edges)
                                if not np.array_equal(windows.load_aligned_matrix(temp, edges), matrix):
                                    raise RuntimeError(f"CSV roundtrip failed: {path}")
                    states_by_group[seed, rate] = load_states(out, edges)
                    flow_records[seed, rate] = "generated"
            del counts
        records, window_counts = [], Counter()
        for seed in common.SEEDS:
            previous = None
            for rate in common.RATES:
                states = states_by_group[seed, rate]
                check_states(states, f"{day}/{seed}/{rate}", complete, previous)
                previous = states
                npzs, actions = prepare_windows(root, seed, rate, split, day, states, complete, protected)
                window_counts.update(actions)
                records.append({"sampling_seed": seed, "rate": rate, "date": day, "split": split,
                                "flow_action": flow_records[seed, rate], "windows": npzs,
                                "states": [common.signature(common.flow_path(root, seed, rate, day) / f"{n}.csv")
                                           for n in common.CHANNELS]})
            full.log(f"[{day}] seed={seed} all 7 rates: CSV conservation/bounds/nesting and all NPZ arrays PASS")
        common.unchanged(protected)
        # Recheck sampled lists after the full computation as well.
        for seed, rate in ids:
            if common.sha256(common.vehicle_path(root, seed, rate, day)) != CONTEXT["sampling_records"][seed, rate, day]["sha256"]:
                raise RuntimeError(f"stage-1 IDs changed during generation: {day}/{seed}/{rate}")
        result = {"date": day, "passed": True, "groups": 35, "records": records, "fcd": scan,
                  "flow_counts": dict(Counter(flow_records.values())), "window_file_counts": dict(window_counts),
                  "protected_sources": protected, "historical_sources_unchanged": True,
                  "checks": {name: True for name in ("integer_conservation", "zero_initial_stock", "stock_continuity",
                    "nonnegative_finite", "partial_le_full", "rate_nesting", "road_order", "288_slots", "277_windows",
                    "all_npz_keys_dtypes_arrays", "shared_target_all_seeds_rates", "shared_scaler", "historical_arrays_equal")}}
        full.log(f"[{day}] PASS groups=35 flow={result['flow_counts']} windows={dict(window_counts)}")
    except Exception:
        result = {"date": day, "passed": False, "error": traceback.format_exc()}
        full.log(f"[{day}] FAIL {result['error']}")
    common.write_report(root / "sampling_seeds/audit_v2" / f"{day}.json", result, root)
    return result


def validate_inputs(root, fcd, splits):
    manifest_path = root / "sampling_seeds/vehicle_sampling_manifest_v2.json"
    manifest = common.load_json(manifest_path)
    report = common.load_json(root / "sampling_seeds/vehicle_sampling_validation_v2.json")
    if not report.get("passed") or not manifest.get("passed") or report["verified_groups"] != 1015:
        raise RuntimeError("stage 1 must pass all 1015 groups before Flow generation")
    if common.sha256(manifest_path) != report["manifest_sha256"]:
        raise RuntimeError("stage-1 manifest hash mismatch")
    if manifest["splits"] != splits or manifest["fcd_root"] != str(fcd) or manifest["training_root"] != str(root):
        raise RuntimeError("stage-1 roots/splits mismatch")
    records = {(row["sampling_seed"], row["rate"], row["date"]): row for row in manifest["records"]}
    if len(records) != 1015:
        raise RuntimeError("duplicate or missing stage-1 groups")
    common.unchanged([manifest["split_manifest"], *manifest["tripinfo_sources"]])
    normalization = common.load_json(root / "shared/state_normalization.json")
    if normalization.get("fit_split") != "train" or normalization.get("training_days") != splits["train"]:
        raise RuntimeError("historical normalization does not use exactly the train split; refusing to change it")
    if normalization.get("channels") != list(common.CHANNELS) or normalization.get("formula") != "(value - mean) / std":
        raise RuntimeError("historical normalization schema/formula mismatch")
    for table in [normalization["target"], *(normalization["input"][r] for r in common.RATES)]:
        for name in common.CHANNELS:
            if not np.isfinite([table[name]["mean"], table[name]["std"]]).all() or table[name]["std"] <= 0:
                raise RuntimeError("invalid historical scaler")
    return records, normalization


def publish_metadata(root, results, resources, protected, dates, all_dates):
    failed = [r for r in results if not r["passed"]]
    passed = not failed and sorted(dates) == all_dates and sum(r.get("groups", 0) for r in results) == 1015
    records = [row for day in results for row in day.get("records", [])]
    counts = Counter(row["flow_action"] for row in records)
    window_counts = Counter()
    for result in results:
        window_counts.update(result.get("window_file_counts", {}))
    payload = {"data_version": common.VERSION, "feature_family": "flow", "passed": passed,
               "training_root": str(root), "fcd_root": str(CONTEXT["fcd"]),
               "selected_dates_passed": not failed, "dates": dates, "seeds": list(common.SEEDS), "rates": list(common.RATES),
               "splits": CONTEXT["splits"], "resources": resources, "verified_groups": len(records),
               "flow_group_counts": dict(counts), "window_file_counts": dict(window_counts),
               "csv_files": len(records) * 4, "npz_files": len(records) * 5,
               "fcd_scans_this_run": sum(r.get("fcd", {}).get("scans", 0) for r in results),
               "failures": failed, "protected_inputs": protected,
               "scaler_source": common.signature(root / "shared/state_normalization.json"),
               "shared_full_labels": str(root / "full_flow/r-level"), "shared_graph": str(root / "graph"),
               "label_storage": "Authoritative full CSVs and graph are shared; y/y_raw embedded only as required by legacy NPZ schema.",
               "layout": "seed_{sampling_seed}/flow/window_data/{rate}/{split}/{date}/{state}.npz",
               "channels": list(common.CHANNELS), "records": records}
    base = root / "sampling_seeds"
    if sorted(dates) != all_dates:
        # A smoke/retry subset must never replace the authoritative complete report.
        common.write_report(base / "flow_generation_subset_v2.json", payload, root)
        return not failed
    common.write_report(base / "flow_generation_manifest_v2.json", payload, root)
    validation = {k: v for k, v in payload.items() if k != "records"}
    validation["manifest_sha256"] = common.sha256(base / "flow_generation_manifest_v2.json")
    validation["daily_reports"] = [str(base / "audit_v2" / f"{day}.json") for day in dates]
    common.write_report(base / "flow_generation_validation_v2.json", validation, root)
    for seed in common.SEEDS:
        dest = common.window_root(root, seed)
        for name in ("split_manifest.json", "state_normalization.json"):
            common.link_existing(dest / name, root / "shared" / name)
        for name in ("node_mapping.csv",):
            source = root / "shared" / name
            if source.is_file():
                common.link_existing(dest / name, source)
        for name in ("state_window_manifest.json", "validation_summary.json"):
            common.link_existing(dest / name, root / "shared" / name)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_paths(parser, flow=True)
    args = parser.parse_args()
    root, fcd = common.roots(args)
    splits, all_dates = common.split_manifest(root)
    dates = sorted(set(args.dates or all_dates))
    if not set(dates) <= set(all_dates):
        raise RuntimeError("requested date outside authoritative split manifest")
    resources = common.resource_check(root, args.workers, flow=True)
    resources["workers"] = min(resources["workers"], len(dates))
    with common.stage_lock(root):
        records, normalization = validate_inputs(root, fcd, splits)
        windows.GRAPH_PATH = root / "graph/r_graph.npz"
        edges = windows.load_edge_ids()
        network, lanes, internal = full.load_network(args.net_file)
        if edges != network:
            raise RuntimeError("graph and network node order differ")
        full.self_test()
        full.confirm_fcd_starts_at_zero(fcd, dates)
        protected = [common.signature(p) for p in (
            root / "shared/split_manifest.json", root / "shared/state_normalization.json",
            root / "graph/r_graph.npz", args.net_file,
            root / "sampling_seeds/vehicle_sampling_manifest_v2.json",
            root / "sampling_seeds/vehicle_sampling_validation_v2.json")]
        CONTEXT.update(root=root, fcd=fcd, splits=splits, edges=edges, lanes=lanes, internal=internal,
                       normalization=normalization, sampling_records=records, verify_only=args.verify_only)
        full.log(f"stage=flow dates={len(dates)} resources={resources} verify_only={args.verify_only}")
        results = []
        with ProcessPoolExecutor(max_workers=resources["workers"], mp_context=mp.get_context("fork")) as pool:
            futures = {pool.submit(process_day, day): day for day in dates}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception:
                    results.append({"date": futures[future], "passed": False, "error": traceback.format_exc()})
        results.sort(key=lambda item: item["date"])
        common.unchanged(protected)
        ok = publish_metadata(root, results, resources, protected, dates, all_dates)
        full.log(f"Flow {'PASS' if ok else 'FAIL'} dates={len(dates)} groups={sum(r.get('groups',0) for r in results)}")
        return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
