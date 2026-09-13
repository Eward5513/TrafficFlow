"""
Sample nested vehicle-ID lists at observation penetrations.

Reads each day's vehicle_ids.txt. Does not open trajectories, the SUMO
network, subgraph.txt, or FCD. One SHA256 ordering per day produces all
penetration files, so samples are nested prefixes.

Usage:

    python3 analysis/simulation/sample_vehicle_ids.py
    python3 analysis/simulation/sample_vehicle_ids.py --day day_01
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = BASE_DIR / "data" / "processed" / "subgraph_trajectories"
DEFAULT_IDS_NAME = "vehicle_ids.txt"
DEFAULT_OUTPUT_SUBDIR = "sampled_vehicle_ids"
DEFAULT_SEED = 42
DEFAULT_RATES = (5, 10, 20, 30, 40, 50, 70)
DEFAULT_EXPECTED_DAYS = 20
HASH_SEPARATOR = "\0"
SORT_METHOD = (
    "sha256(seed + '\\0' + day_identifier + '\\0' + vehicle_id) ascending; "
    "ties broken by vehicle_id"
)


class SampleError(Exception):
    pass


@dataclass
class SourceIds:
    path: Path
    nonempty_lines: list[str]
    unique_ids: list[str]
    duplicate_count: int
    empty_line_count: int
    sha256: str
    size_bytes: int
    mtime_ns: int


@dataclass
class PenetrationResult:
    percent: int
    expected_size: int
    ids: list[str]
    filename: str
    sha256: str
    rewritten: bool


def parse_rates(raw: list[str]) -> list[int]:
    rates: list[int] = []
    seen: set[int] = set()
    for item in raw:
        for part in item.split(","):
            part = part.strip().rstrip("%")
            if not part:
                continue
            value = int(part)
            if value <= 0 or value > 100:
                raise SampleError(f"penetration rate out of range: {part}")
            if value not in seen:
                seen.add(value)
                rates.append(value)
    if not rates:
        raise SampleError("no penetration rates")
    return rates


def sample_filename(percent: int) -> str:
    return f"vehicles_p{percent:02d}.txt"


def sample_size(n_day: int, percent: int) -> int:
    return n_day * percent // 100


def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render_id_file(ids: list[str]) -> bytes:
    return "".join(f"{vehicle_id}\n" for vehicle_id in ids).encode("utf-8")


def load_vehicle_ids(path: Path) -> SourceIds:
    stat = path.stat()
    text = path.read_text(encoding="utf-8")
    digest = file_sha256(text.encode("utf-8"))
    unique: list[str] = []
    seen: set[str] = set()
    duplicate_count = 0
    empty_line_count = 0
    nonempty: list[str] = []
    for line in text.splitlines():
        vehicle_id = line.strip()
        if not vehicle_id:
            empty_line_count += 1
            continue
        nonempty.append(vehicle_id)
        if vehicle_id in seen:
            duplicate_count += 1
            continue
        seen.add(vehicle_id)
        unique.append(vehicle_id)
    return SourceIds(
        path=path,
        nonempty_lines=nonempty,
        unique_ids=unique,
        duplicate_count=duplicate_count,
        empty_line_count=empty_line_count,
        sha256=digest,
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def order_vehicles(seed: str, day_identifier: str, vehicle_ids: list[str]) -> list[str]:
    keyed: list[tuple[str, str]] = []
    for vehicle_id in vehicle_ids:
        payload = f"{seed}{HASH_SEPARATOR}{day_identifier}{HASH_SEPARATOR}{vehicle_id}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        keyed.append((digest, vehicle_id))
    keyed.sort()
    return [vehicle_id for _digest, vehicle_id in keyed]


def discover_day_dirs(root: Path, ids_name: str) -> list[Path]:
    days: list[Path] = []
    if not root.is_dir():
        raise SampleError(f"root is not a directory: {root}")
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        if child.is_dir() and (child / ids_name).is_file():
            days.append(child)
    return days


def relpath(path: Path, start: Path) -> str:
    try:
        return path.resolve().relative_to(start.resolve()).as_posix()
    except ValueError:
        return str(path)


def write_bytes_if_needed(path: Path, expected: bytes) -> bool:
    if path.is_file() and path.read_bytes() == expected:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(expected)
    tmp.replace(path)
    return True


def nested_prefix_ok(samples: list[list[str]]) -> bool:
    previous: list[str] | None = None
    for sample in samples:
        if previous is not None:
            if sample[: len(previous)] != previous:
                return False
            if not set(previous).issubset(set(sample)):
                return False
        previous = sample
    return True


def validate_samples(
    source: SourceIds,
    ordered: list[str],
    results: list[PenetrationResult],
) -> dict[str, bool]:
    universe = set(source.unique_ids)
    membership = True
    sizes = True
    unique_ok = True
    for result in results:
        if len(result.ids) != result.expected_size:
            sizes = False
        if len(result.ids) != len(set(result.ids)):
            unique_ok = False
        if any(vehicle_id not in universe for vehicle_id in result.ids):
            membership = False
        expected_ids = ordered[: result.expected_size]
        if result.ids != expected_ids:
            sizes = False
    nested = nested_prefix_ok([result.ids for result in results])
    return {
        "nested_validation_passed": nested,
        "membership_validation_passed": membership,
        "cross_day_mix_passed": membership,
        "size_validation_passed": sizes,
        "unique_in_sample_passed": unique_ok,
    }


def process_day(
    day_dir: Path,
    *,
    root: Path,
    ids_name: str,
    output_subdir: str,
    seed: int,
    rates: list[int],
) -> dict[str, object]:
    ids_path = day_dir / ids_name
    if not ids_path.is_file():
        raise SampleError(f"missing {ids_path}")
    source_stat_before = (ids_path.stat().st_size, ids_path.stat().st_mtime_ns)
    source = load_vehicle_ids(ids_path)
    day_identifier = day_dir.name
    seed_text = str(seed)
    ordered = order_vehicles(seed_text, day_identifier, source.unique_ids)
    ordered_again = order_vehicles(seed_text, day_identifier, list(reversed(source.unique_ids)))
    reproducibility = ordered == ordered_again
    n_day = len(source.unique_ids)
    out_dir = day_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[PenetrationResult] = []
    for percent in rates:
        expected = sample_size(n_day, percent)
        sample = ordered[:expected]
        filename = sample_filename(percent)
        payload = render_id_file(sample)
        path = out_dir / filename
        rewritten = write_bytes_if_needed(path, payload)
        results.append(
            PenetrationResult(
                percent=percent,
                expected_size=expected,
                ids=sample,
                filename=filename,
                sha256=file_sha256(payload),
                rewritten=rewritten,
            )
        )
        on_disk = path.read_bytes()
        if file_sha256(on_disk) != file_sha256(payload):
            reproducibility = False

    checks = validate_samples(source, ordered, results)
    source_stat_after = (ids_path.stat().st_size, ids_path.stat().st_mtime_ns)
    source_unchanged = source_stat_after == source_stat_before
    extra_traj = (out_dir / "trajectories.csv").exists()
    overall = (
        all(checks.values())
        and reproducibility
        and source_unchanged
        and not extra_traj
        and n_day == len(set(source.unique_ids))
    )
    penetrations = []
    for result in results:
        actual_rate = (result.expected_size / n_day) if n_day else 0.0
        penetrations.append(
            {
                "penetration_percent": result.percent,
                "penetration_rate": result.percent / 100.0,
                "expected_sample_size": result.expected_size,
                "actual_sample_size": len(result.ids),
                "actual_rate": actual_rate,
                "output_file": relpath(out_dir / result.filename, root),
                "sha256": result.sha256,
                "rewritten": result.rewritten,
            }
        )
    summary = {
        "day": day_identifier,
        "vehicle_ids_path": relpath(ids_path, root),
        "seed": seed,
        "sort_method": SORT_METHOD,
        "day_identifier": day_identifier,
        "source_nonempty_line_count": len(source.nonempty_lines),
        "unique_vehicle_count": n_day,
        "duplicate_id_count": source.duplicate_count,
        "empty_line_count": source.empty_line_count,
        "source_sha256": source.sha256,
        "source_size_bytes": source.size_bytes,
        "output_subdirectory": relpath(out_dir, root),
        "penetrations": penetrations,
        "nested_validation_passed": checks["nested_validation_passed"],
        "membership_validation_passed": checks["membership_validation_passed"],
        "cross_day_mix_passed": checks["cross_day_mix_passed"],
        "size_validation_passed": checks["size_validation_passed"],
        "unique_in_sample_passed": checks["unique_in_sample_passed"],
        "reproducibility_passed": reproducibility,
        "source_unchanged": source_unchanged,
        "no_trajectory_copy": not extra_traj,
        "overall_validation_passed": overall,
    }
    summary_path = out_dir / "sampling_summary.json"
    summary_bytes = (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    write_bytes_if_needed(summary_path, summary_bytes)
    if not overall:
        raise SampleError(f"{day_identifier} sampling validation failed")
    return summary


def write_all_days_csv(path: Path, summaries: list[dict[str, object]], seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "day",
                "seed",
                "penetration_rate",
                "total_unique_vehicles",
                "expected_sample_size",
                "actual_sample_size",
                "actual_rate",
                "output_file",
                "sha256",
                "subset_validation_passed",
                "membership_validation_passed",
                "duplicate_count_in_source",
                "overall_validation_passed",
            ]
        )
        for summary in summaries:
            for item in summary["penetrations"]:
                writer.writerow(
                    [
                        summary["day"],
                        seed,
                        f"{item['penetration_rate']:.2f}",
                        summary["unique_vehicle_count"],
                        item["expected_sample_size"],
                        item["actual_sample_size"],
                        f"{item['actual_rate']:.10f}",
                        item["output_file"],
                        item["sha256"],
                        str(summary["nested_validation_passed"]).lower(),
                        str(summary["membership_validation_passed"]).lower(),
                        summary["duplicate_id_count"],
                        str(summary["overall_validation_passed"]).lower(),
                    ]
                )


def run_synthetic_checks() -> None:
    seed = "42"
    vehicles = ["b", "a", "10", "10.0", "59135.dup1"]
    first = order_vehicles(seed, "day_x", vehicles)
    second = order_vehicles(seed, "day_x", list(reversed(vehicles)))
    if first != second:
        raise SampleError("synthetic order is not independent of input order")
    other_day = order_vehicles(seed, "day_y", vehicles)
    if first == other_day:
        raise SampleError("different day identifiers produced the same order")
    n_day = len(vehicles)
    empty = first[: sample_size(n_day, 5)]
    if empty:
        raise SampleError("synthetic 5% of 5 vehicles should be empty")
    p50 = first[: sample_size(n_day, 50)]
    p70 = first[: sample_size(n_day, 70)]
    if p50 != p70[: len(p50)]:
        raise SampleError("synthetic nested prefix failed")
    if sample_size(20677, 5) != math.floor(20677 * 0.05):
        raise SampleError("sample_size does not match floor(N * p)")
    payload = render_id_file(["a", "b"])
    if payload != b"a\nb\n":
        raise SampleError("id file rendering is wrong")
    print("Synthetic nested / hash-order checks passed.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample nested vehicle-ID lists at observation penetrations "
            "from each day's vehicle_ids.txt."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Directory that contains one subdirectory per day.",
    )
    parser.add_argument(
        "--ids-name",
        default=DEFAULT_IDS_NAME,
        help="Vehicle ID filename inside each day directory.",
    )
    parser.add_argument(
        "--output-subdir",
        default=DEFAULT_OUTPUT_SUBDIR,
        help="Sampling output subdirectory name inside each day directory.",
    )
    parser.add_argument(
        "--rates",
        nargs="+",
        default=[str(rate) for rate in DEFAULT_RATES],
        help="Penetration percents (default: 5 10 20 30 40 50 70).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Fixed sampling seed.")
    parser.add_argument(
        "--day",
        action="append",
        default=None,
        help="Day directory name to process. Repeatable. Default: all discovered days.",
    )
    parser.add_argument(
        "--expected-days",
        type=int,
        default=DEFAULT_EXPECTED_DAYS,
        help="When processing all days, require this many vehicle_ids.txt files.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="All-days summary CSV (default: <root>/sampling_summary_all_days.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_synthetic_checks()
    root = args.root.expanduser().resolve()
    rates = parse_rates(list(args.rates))
    discovered = discover_day_dirs(root, args.ids_name)
    if args.day:
        wanted = []
        missing = []
        by_name = {path.name: path for path in discovered}
        for name in args.day:
            if name in by_name:
                wanted.append(by_name[name])
            else:
                missing.append(name)
        if missing:
            raise SystemExit(
                "day directory with vehicle_ids.txt not found: "
                + ", ".join(missing)
                + f"; discovered: {', '.join(path.name for path in discovered) or 'none'}"
            )
        day_dirs = wanted
    else:
        if len(discovered) != args.expected_days:
            names = ", ".join(path.name for path in discovered) or "none"
            raise SystemExit(
                f"expected {args.expected_days} day directories with {args.ids_name}, "
                f"found {len(discovered)}: {names}"
            )
        day_dirs = discovered

    summaries: list[dict[str, object]] = []
    for day_dir in day_dirs:
        summary = process_day(
            day_dir,
            root=root,
            ids_name=args.ids_name,
            output_subdir=args.output_subdir,
            seed=args.seed,
            rates=rates,
        )
        summaries.append(summary)
        n_day = summary["unique_vehicle_count"]
        sizes = ", ".join(
            f"p{item['penetration_percent']:02d}={item['actual_sample_size']}"
            for item in summary["penetrations"]
        )
        print(
            f"{summary['day']}: N={n_day} dups={summary['duplicate_id_count']} {sizes}",
            flush=True,
        )

    summary_csv = (
        args.summary_csv.expanduser().resolve()
        if args.summary_csv is not None
        else root / "sampling_summary_all_days.csv"
    )
    write_all_days_csv(summary_csv, summaries, args.seed)
    expected_rows = len(day_dirs) * len(rates)
    print(f"Wrote {summary_csv} ({expected_rows} rows)", flush=True)
    if not all(summary["overall_validation_passed"] for summary in summaries):
        raise SystemExit("one or more days failed sampling validation")
    print(f"All {len(day_dirs)} day(s) passed sampling validation.", flush=True)


if __name__ == "__main__":
    main()
