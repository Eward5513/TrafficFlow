"""Deterministic vehicle sampling primitives; no standalone legacy generator."""
from __future__ import annotations
import csv
import gzip
import hashlib
import random
from pathlib import Path
from xml.parsers import expat
GLOBAL_SEED = 42
RATES = (("p05", .05), ("p10", .10), ("p20", .20), ("p30", .30), ("p40", .40), ("p50", .50), ("p70", .70))

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


def daily_seed(date_label: str, global_seed: int = GLOBAL_SEED) -> int:
    """Derive one day's shuffle seed from a sampling seed and the date label.

    ``global_seed`` is the sampling seed, not a model-training seed. The
    default remains 42, so existing callers keep the original permutation.
    """
    digest = hashlib.sha256(f"{int(global_seed)}_{date_label}".encode("utf-8")).digest()
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


def read_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != ["vehicle_id"]:
            raise RuntimeError(f"unexpected vehicle header in {path}: {header}")
        values = [row[0] for row in reader if row]
    if any(not isinstance(item, str) or item == "" for item in values):
        raise RuntimeError(f"blank vehicle id in {path}")
    return values

