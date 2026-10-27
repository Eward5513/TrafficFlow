#!/usr/bin/env python3
"""Sample nested vehicle-ID lists at fixed penetration rates.

Each day is shuffled once with a deterministic seed. Higher penetration rates
are prefixes of that same permutation. Original simulation files are only read.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import os
import random
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from xml.parsers import expat

FCD_ROOT = Path("/data5/tzhang174/TrafficFlow/xuancheng26/simulation_data/fcd_data")
OUT_ROOT = Path(
    "/data5/tzhang174/TrafficFlow/xuancheng26/training_data/sampled_vehicle"
)
SUMMARY_PATH = OUT_ROOT / "sampling_summary.csv"
GLOBAL_SEED = 42
RATES = (
    ("p05", 0.05),
    ("p10", 0.10),
    ("p20", 0.20),
    ("p30", 0.30),
    ("p40", 0.40),
    ("p50", 0.50),
    ("p70", 0.70),
)
SUMMARY_FIELDS = [
    "date",
    "eligible_vehicle_count",
    "p05_count",
    "p10_count",
    "p20_count",
    "p30_count",
    "p40_count",
    "p50_count",
    "p70_count",
    "nested_check",
    "status",
]


def log(message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"{stamp} {message}", flush=True)


def discover_dates() -> list[str]:
    dates = []
    for path in FCD_ROOT.iterdir():
        if path.is_dir() and len(path.name) == 10 and path.name[4] == "_" and path.name[7] == "_":
            dates.append(path.name)
    return sorted(dates)


def tripinfo_path(date_label: str) -> Path:
    day_dir = FCD_ROOT / date_label
    plain = day_dir / f"tripinfo_{date_label}.xml"
    compressed = day_dir / f"tripinfo_{date_label}.xml.gz"
    if plain.is_file() and plain.stat().st_size > 0:
        return plain
    if compressed.is_file() and compressed.stat().st_size > 0:
        return compressed
    raise FileNotFoundError(f"missing tripinfo for {date_label}")


def open_input(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rb")
    return path.open("rb")


def load_eligible_ids(path: Path) -> list[str]:
    """Vehicles with depart >= 0. Undeparted vehicles are excluded."""
    found: set[str] = set()

    def start(name: str, attrs) -> None:
        if name != "tripinfo":
            return
        vehicle_id = attrs.get("id", "")
        if not vehicle_id or "depart" not in attrs:
            return
        try:
            depart = float(attrs["depart"])
        except ValueError:
            return
        if depart >= 0:
            found.add(vehicle_id)

    parser = expat.ParserCreate()
    parser.StartElementHandler = start
    with open_input(path) as handle:
        parser.ParseFile(handle)
    return sorted(found)


def daily_seed(date_label: str) -> int:
    digest = hashlib.sha256(f"{GLOBAL_SEED}_{date_label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def permute(vehicle_ids: list[str], seed: int) -> list[str]:
    ordered = list(vehicle_ids)
    random.Random(seed).shuffle(ordered)
    return ordered


def sample_prefixes(permutation: list[str], population: int) -> dict[str, list[str]]:
    return {name: permutation[: int(population * rate)] for name, rate in RATES}


def nested_ok(samples: dict[str, list[str]]) -> bool:
    names = [name for name, _rate in RATES]
    for low, high in zip(names, names[1:]):
        if not set(samples[low]).issubset(samples[high]):
            return False
    return True


def write_ids(path: Path, vehicle_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["vehicle_id"])
        for vehicle_id in sorted(vehicle_ids):
            writer.writerow([vehicle_id])
    os.replace(temporary, path)


def validate_samples(eligible: list[str], samples: dict[str, list[str]]) -> None:
    population = set(eligible)
    count = len(eligible)
    for name, rate in RATES:
        chosen = samples[name]
        expected = int(count * rate)
        if len(chosen) != expected:
            raise RuntimeError(f"{name} count {len(chosen)} != {expected}")
        if len(set(chosen)) != len(chosen):
            raise RuntimeError(f"{name} contains duplicate ids")
        if not set(chosen).issubset(population):
            raise RuntimeError(f"{name} contains an id outside the eligible population")
    if not nested_ok(samples):
        raise RuntimeError("penetration rates are not nested")


def process_day(date_label: str) -> dict[str, object]:
    try:
        eligible = load_eligible_ids(tripinfo_path(date_label))
        if not eligible:
            raise RuntimeError("no eligible vehicles")
        seed = daily_seed(date_label)
        permutation = permute(eligible, seed)
        if permute(eligible, seed) != permutation:
            raise RuntimeError("sampling is not deterministic")
        samples = sample_prefixes(permutation, len(eligible))
        validate_samples(eligible, samples)
        for name, _rate in RATES:
            write_ids(OUT_ROOT / name / f"{date_label}.csv", samples[name])
        row: dict[str, object] = {
            "date": date_label,
            "eligible_vehicle_count": len(eligible),
            "nested_check": "PASS",
            "status": "PASS",
        }
        for name, _rate in RATES:
            row[f"{name}_count"] = len(samples[name])
        log(
            f"[{date_label}] PASS eligible={len(eligible)} "
            + " ".join(f"{name}={row[f'{name}_count']}" for name, _rate in RATES)
        )
        return row
    except Exception:
        detail = traceback.format_exc(limit=6).strip().replace("\n", " | ")
        log(f"[{date_label}] FAIL {detail}")
        row = {field: "" for field in SUMMARY_FIELDS}
        row["date"] = date_label
        row["nested_check"] = "FAIL"
        row["status"] = "FAIL"
        return row


def write_summary(rows: list[dict[str, object]]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = SUMMARY_PATH.with_suffix(SUMMARY_PATH.suffix + ".tmp")
    ordered = sorted(rows, key=lambda item: str(item["date"]))
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in ordered:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})
    os.replace(temporary, SUMMARY_PATH)


def print_overall(rows: list[dict[str, object]]) -> None:
    passed = [row for row in rows if row.get("status") == "PASS"]
    failed = [str(row["date"]) for row in rows if row.get("status") != "PASS"]
    counts = [int(row["eligible_vehicle_count"]) for row in passed]
    log(f"dates={len(rows)} pass={len(passed)} fail={len(failed)}")
    if counts:
        log(
            f"eligible min={min(counts)} max={max(counts)} "
            f"mean={sum(counts) / len(counts):.1f}"
        )
        for name, _rate in RATES:
            total = sum(int(row[f"{name}_count"]) for row in passed)
            log(f"{name}_total={total}")
    nested_fail = [str(row["date"]) for row in rows if row.get("nested_check") != "PASS"]
    file_counts = {
        name: len(list((OUT_ROOT / name).glob("2023_04_*.csv"))) for name, _rate in RATES
    }
    log("files " + " ".join(f"{name}={count}" for name, count in file_counts.items()))
    log(f"summary={SUMMARY_PATH}")
    log(f"failed={','.join(failed) if failed else 'none'}")
    log(f"nested_failed={','.join(nested_fail) if nested_fail else 'none'}")
    complete = (
        not failed
        and not nested_fail
        and len(rows) == 29
        and all(count == 29 for count in file_counts.values())
    )
    if complete:
        log("Vehicle-level penetration sampling completed successfully.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample nested vehicle IDs by penetration rate")
    parser.add_argument("--workers", type=int, default=29)
    parser.add_argument("--dates", nargs="*", default=None)
    args = parser.parse_args(argv)
    dates = args.dates or discover_dates()
    if not dates:
        log("no date directories")
        return 1
    workers = max(1, min(args.workers, len(dates)))
    log(f"dates={len(dates)} workers={workers} seed={GLOBAL_SEED}")
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_day, date_label): date_label for date_label in dates}
        for future in as_completed(futures):
            date_label = futures[future]
            try:
                rows.append(future.result())
            except Exception:
                detail = traceback.format_exc(limit=6).strip().replace("\n", " | ")
                log(f"[{date_label}] FAIL {detail}")
                row = {field: "" for field in SUMMARY_FIELDS}
                row["date"] = date_label
                row["nested_check"] = "FAIL"
                row["status"] = "FAIL"
                rows.append(row)
    write_summary(rows)
    print_overall(rows)
    return 0 if all(row.get("status") == "PASS" for row in rows) else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
