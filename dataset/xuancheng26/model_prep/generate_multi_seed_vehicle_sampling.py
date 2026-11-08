#!/usr/bin/env python3
"""Stage 1: independently generate and audit all 1015 nested vehicle lists."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import sys
import traceback

import vehicle_sampling as sampling
from vehicle_sampling import read_ids
import generate_r_level_full_flow as full
import multi_seed_common as common

CONTEXT = {}


def process_day(day):
    root, fcd, splits = (CONTEXT[k] for k in ("root", "fcd", "splits"))
    split = next(s for s, days in splits.items() if day in days)
    try:
        trip = full.prefer_xml(fcd / day, f"tripinfo_{day}")
        eligible = sampling.load_eligible_ids(trip)
        if not eligible:
            raise RuntimeError(f"{day}: empty eligible population")
        planned, seen_sets = [], {rate: set() for rate in common.RATES}
        # Validate every historical list for this day before publishing anything.
        for seed in common.SEEDS:
            derived = sampling.daily_seed(day, seed)
            permutation = sampling.permute(eligible, derived)
            if permutation != sampling.permute(eligible, derived):
                raise RuntimeError(f"{day}/{seed}: non-deterministic shuffle")
            samples = sampling.sample_prefixes(permutation, len(eligible))
            sampling.validate_samples(eligible, samples)
            for rate, chosen in samples.items():
                chosen = sorted(chosen)
                identity = frozenset(chosen)
                if identity in seen_sets[rate]:
                    raise RuntimeError(f"{day}/{seed}/{rate}: identical sets across seeds")
                seen_sets[rate].add(identity)
                path = common.vehicle_path(root, seed, rate, day)
                old = common.historical(root, seed, rate, split, day, "vehicle")
                for candidate in {old, path} - {None}:
                    if candidate == old or candidate.exists() or candidate.is_symlink():
                        stored = read_ids(candidate)
                        if stored != chosen:
                            raise RuntimeError(f"{day}/{seed}/{rate}: historical/existing IDs mismatch at {candidate}; "
                                               f"only_expected={len(set(chosen)-set(stored))}, only_stored={len(set(stored)-set(chosen))}")
                planned.append((seed, rate, derived, chosen, path, old))
        records = []
        for seed, rate, derived, chosen, path, old in planned:
            protected = common.signature(old) if old else None
            if path.exists():
                action = "skipped"
            elif old:
                common.link_existing(path, old)
                action = "reused"
            else:
                with common.new_file(path, root) as temp:
                    # Existing writer is used only on a fresh private temporary path.
                    sampling.write_ids(temp, chosen)
                    if read_ids(temp) != chosen:
                        raise RuntimeError(f"{path}: temporary IDs mismatch")
                action = "generated"
            if read_ids(path) != chosen:
                raise RuntimeError(f"{path}: persisted IDs mismatch")
            if protected:
                common.unchanged([protected])
            records.append({"sampling_seed": seed, "date": day, "split": split, "rate": rate,
                            "eligible_vehicle_count": len(eligible), "sample_count": len(chosen),
                            "actual_ratio": len(chosen) / len(eligible), "daily_seed": str(derived),
                            "vehicle_file": str(path), "sha256": common.sha256(path), "action": action,
                            "historical_source": protected})
        sampling.log(f"[{day}] PASS 35 vehicle groups; {dict(Counter(r['action'] for r in records))}")
        return {"date": day, "passed": True, "tripinfo": common.signature(trip), "records": records}
    except Exception:
        error = traceback.format_exc()
        sampling.log(f"[{day}] FAIL {error}")
        return {"date": day, "passed": False, "error": error, "records": []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_paths(parser)
    args = parser.parse_args()
    root, fcd = common.roots(args)
    splits, dates = common.split_manifest(root)
    resources = common.resource_check(root, args.workers)
    with common.stage_lock(root):
        protected = common.signature(root / "shared/split_manifest.json")
        CONTEXT.update(root=root, fcd=fcd, splits=splits)
        sampling.log(f"stage=vehicle_sampling resources={resources}")
        results = []
        with ProcessPoolExecutor(max_workers=resources["workers"], mp_context=mp.get_context("fork")) as pool:
            futures = {pool.submit(process_day, day): day for day in dates}
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: item["date"])
        records = [row for result in results for row in result["records"]]
        failed = [item for item in results if not item["passed"]]
        common.unchanged([protected])
        passed = not failed and len(records) == 1015
        manifest = {"data_version": common.VERSION, "algorithm_version": "sha256_sampling_seed_date_v1",
                    "algorithm": "sort string IDs; Random(int.from_bytes(SHA256(seed_date)[:8], 'big')).shuffle; sorted floor(N*rate) prefixes",
                    "eligibility": "tripinfo depart >= 0", "fcd_root": str(fcd), "training_root": str(root),
                    "seeds": list(common.SEEDS), "rates": list(common.RATES), "splits": splits,
                    "split_manifest": protected, "resources": resources, "passed": passed,
                    "tripinfo_sources": [r["tripinfo"] for r in results if r["passed"]], "records": records}
        dest = root / "sampling_seeds/vehicle_sampling_manifest_v2.json"
        common.write_report(dest, manifest, root)
        report = {"data_version": common.VERSION, "passed": passed, "expected_groups": 1015,
                  "verified_groups": len(records), "date_count": len(results), "seed_count": 5, "rate_count": 7,
                  "counts": dict(Counter(row["action"] for row in records)), "failures": failed,
                  "manifest_sha256": common.sha256(dest), "resources": resources,
                  "checks": {key: passed for key in ("counts", "unique_ids", "eligible_membership", "nested_rates",
                    "deterministic_repeat", "different_seed_sets", "seed42_history_equal", "p05_test_history_equal",
                    "history_unchanged", "all_files_reread")}}
        common.write_report(root / "sampling_seeds/vehicle_sampling_validation_v2.json", report, root)
        sampling.log(f"vehicle sampling {'PASS' if passed else 'FAIL'} groups={len(records)} counts={report['counts']}")
        return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
