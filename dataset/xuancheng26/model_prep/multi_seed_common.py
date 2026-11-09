"""Shared paths and non-destructive I/O for the two 3s sampling stages."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import prep_paths

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from data_root import data_root, code_root

SEEDS = (42, 43, 44, 45, 46)
RATES = ("p05", "p10", "p20", "p30", "p40", "p50", "p70")
CHANNELS = ("inflow", "outflow", "accumulation_start", "accumulation_end")
VERSION = "xuancheng26_3s_multi_seed_v2"


def add_paths(parser, *, flow=False):
    parser.add_argument("--training-root", type=Path, default=data_root() / "xuancheng26/training_data_3s")
    parser.add_argument("--fcd-root", type=Path, default=data_root() / "xuancheng26/simulation_data/fcd_data_3s")
    parser.add_argument("--workers", type=int, default=24 if flow else 29)
    if flow:
        parser.add_argument("--net-file", type=Path, default=code_root() / "dataset/xuancheng26/xuancheng_tls_3s.net.xml")
        parser.add_argument("--dates", nargs="+", help="Only these manifest dates; never reports full completion for a subset")
        parser.add_argument("--verify-only", action="store_true")


def roots(args):
    root = prep_paths.refuse_code_root(args.training_root)
    fcd = args.fcd_root.expanduser().resolve()
    if root.name != "training_data_3s" or fcd.name != "fcd_data_3s":
        raise RuntimeError("this version requires training_data_3s and fcd_data_3s")
    return root, fcd


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(path):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path), "sha256": sha256(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def unchanged(records):
    for item in records:
        if signature(item["path"]) != item:
            raise RuntimeError(f"protected input changed: {item['path']}")


def split_manifest(root):
    path = root / "shared/split_manifest.json"
    payload = load_json(path)
    splits = {key: payload[key] for key in ("train", "validation", "test")}
    dates = [day for days in splits.values() for day in days]
    if [len(v) for v in splits.values()] != [20, 4, 5] or len(set(dates)) != 29:
        raise RuntimeError("expected disjoint 20/4/5 dates in authoritative split manifest")
    if payload["n_his"] != 12 or payload["target_mode"] != "last-observed-step":
        raise RuntimeError("unexpected reconstruction contract")
    return splits, sorted(dates)


def sampling_base(root):
    """Fixed-uniform seed tree. A dataset directory keeps its own ``sampling_seeds``."""
    root = Path(root)
    if root.name == "sampling_seeds":
        return root
    if (root / "dataset_manifest.json").is_file() and (root / "sampling_seeds").is_dir():
        return root / "sampling_seeds"
    return root / "datasets" / "fixed_uniform" / "sampling_seeds"


def seed_root(root, seed):
    return sampling_base(root) / f"seed_{seed}"


def vehicle_path(root, seed, rate, day):
    return seed_root(root, seed) / "sampled_vehicle" / rate / f"{day}.csv"


def flow_path(root, seed, rate, day):
    return seed_root(root, seed) / "flow/partial_flow" / rate / day


def window_root(root, seed):
    return seed_root(root, seed) / "flow/window_data"


def window_path(root, seed, rate, split, day):
    return window_root(root, seed) / rate / split / day


def historical(root, seed, rate, split, day, kind):
    """Canonical data has no historical fallback after the layout migration."""
    return None


def assert_output(path, root):
    """Never follow a legacy alias to write an original file."""
    parent = sampling_base(root).resolve()
    if not Path(path).resolve().is_relative_to(parent):
        raise RuntimeError(f"output escapes sampling_seeds: {path}")


def link_existing(path, source):
    path, source = Path(path), Path(source)
    if not source.is_file():
        raise FileNotFoundError(source)
    if path.exists() or path.is_symlink():
        if path.resolve() != source.resolve() and sha256(path) != sha256(source):
            raise RuntimeError(f"existing {path} differs from {source}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(os.path.relpath(source, path.parent))


@contextlib.contextmanager
def new_file(path, root):
    """Write a temporary file, then atomically publish without replacing a name."""
    path = Path(path)
    assert_output(path, root)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp = Path(name)
    try:
        yield temp
        with temp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temp, path)  # atomic no-clobber publication, even across concurrent attempts
    finally:
        temp.unlink(missing_ok=True)


def write_report(path, payload, root):
    """Only our v2 metadata is replaceable; data files never are."""
    path = Path(path)
    assert_output(path, root)
    if path.is_symlink():
        raise RuntimeError(f"refusing metadata symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


@contextlib.contextmanager
def stage_lock(root):
    path = root / "sampling_seeds/.multi_seed_v2.lock"
    assert_output(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def resource_check(root, requested, *, flow=False):
    """Leave half the CPUs and at least 32 GiB available to training."""
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        fields = line.split()
        mem[fields[0].rstrip(":")] = int(fields[1]) * 1024
    cpus = len(os.sched_getaffinity(0))
    memory_workers = max(1, int((mem["MemAvailable"] - (32 << 30)) // (3 << 30)))
    workers = max(1, min(requested, 29, cpus // 2, memory_workers))
    free = shutil.disk_usage(root).free
    # Bound from uncompressed float32 x/y/raw arrays, five NPZs, 812 new groups.
    estimate = 300 << 30 if flow else 5 << 30
    if free < estimate + (20 << 30):
        raise RuntimeError(f"insufficient disk: free={free}, conservative required={estimate + (20 << 30)}")
    return {"workers": workers, "cpu_affinity_count": cpus, "available_memory_bytes": mem["MemAvailable"],
            "disk_free_bytes": free, "estimated_additional_bytes_upper_bound": estimate,
            "fcd_read_buffer_bytes": 16 << 20, "gpu_used": False}
