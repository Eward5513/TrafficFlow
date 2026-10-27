"""Shared helpers for Xuancheng26 repaired-JSON to SUMO route conversion."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple
from xml.sax.saxutils import escape

from repair_lib import (
    CODE_ROOT,
    SOURCE_ROOT,
    is_internal_edge_id,
    iter_json_array,
    parse_sumo_net,
    sha256_file,
    strip_ns,
)

DATA_ROOT = Path("/data5/tzhang174/TrafficFlow/xuancheng26")
DEFAULT_INPUT_DIR = DATA_ROOT / "simulation_data" / "trip_data" / "repaired"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "simulation_data" / "trip_data" / "prepared"
DEFAULT_NET_FILE_EXPECTED = (
    DATA_ROOT / "simulation_data" / "road_network" / "xuancheng.net.xml"
)
DEFAULT_NET_FILE_ACTUAL = (
    DATA_ROOT / "simulation_data" / "road_network" / "base" / "xuancheng.net.xml"
)
ORIGINAL_NET_FILE = SOURCE_ROOT / "network" / "xuancheng.net.xml"
ORIGINAL_TRIPS_DIR = SOURCE_ROOT / "trips"

EXPECTED_DATES: Tuple[str, ...] = tuple(
    f"2023-04-{day:02d}" for day in range(1, 31) if day != 9
)
EXCLUDED_DATES = ("2023-04-09",)
N_DATES_EXPECTED = 29
DEPART_LIMIT = 86400.0
MIN_ROUTE_EDGES = 2

SOURCE_NAME_RE = re.compile(r"^data_(\d{4})_(\d{2})_(\d{2})_repaired\.json$")
ROUTE_NAME_RE = re.compile(r"^routes_(\d{4})_(\d{2})_(\d{2})\.rou\.xml$")

CODE_SUFFIXES = {".py", ".sh", ".ipynb"}
CODE_NAMES = {"requirements.txt", "setup.py", "pyproject.toml"}
PREPARED_FORBIDDEN_CODE_SUFFIXES = {".py", ".sh", ".ipynb"}
DAY_JSON_RE = re.compile(r"data_\d{4}_\d{2}_\d{2}")

PASSENGER_VEHICLE = {
    "length": 5,
    "width": 1.8,
    "maxPosAcc": 2.6,
    "maxNegAcc": 4.5,
    "usualPosAcc": 2.6,
    "usualNegAcc": 4.5,
    "minGap": 2.5,
    "maxSpeed": 27.78,
    "headwayTime": 2,
}
FREIGHT_VEHICLE = {
    "length": 7.1,
    "width": 2.4,
    "maxPosAcc": 1.3,
    "maxNegAcc": 4,
    "usualPosAcc": 1.3,
    "usualNegAcc": 4,
    "minGap": 2.5,
    "maxSpeed": 27.78,
    "headwayTime": 2,
}

ROUTES_HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">

    <vType id="passenger"
           vClass="passenger"
           length="5"
           width="1.8"
           accel="2.6"
           decel="4.5"
           minGap="2.5"
           maxSpeed="27.78"
           tau="2"/>

    <vType id="freight"
           vClass="truck"
           length="7.1"
           width="2.4"
           accel="1.3"
           decel="4"
           minGap="2.5"
           maxSpeed="27.78"
           tau="2"/>

"""

MANIFEST_FIELDS = [
    "date",
    "source_file",
    "output_file",
    "source_records",
    "filtered_ge_86400",
    "output_vehicles",
    "passenger_count",
    "freight_count",
    "other_fingerprint_count",
    "min_depart",
    "max_depart",
    "min_route_length",
    "max_route_length",
    "mean_route_length",
    "source_depart_sorted",
    "output_bytes",
    "output_sha256",
    "generation_pid",
    "generation_elapsed_s",
    "xml_validation_ok",
    "topology_validation_ok",
    "sumo_load_ok",
    "sumo_exit_code",
    "validation_pid",
    "validation_elapsed_s",
]


class PrepareError(RuntimeError):
    """Fatal per-day or precheck error."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def local_iso(ts: Optional[float] = None) -> str:
    value = time.localtime(time.time() if ts is None else ts)
    return time.strftime("%Y-%m-%dT%H:%M:%S", value)


def date_to_tag(date: str) -> str:
    return date.replace("-", "_")


def date_to_compact(date: str) -> str:
    return date.replace("-", "")


def source_name_for_date(date: str) -> str:
    return f"data_{date_to_tag(date)}_repaired.json"


def route_name_for_date(date: str) -> str:
    return f"routes_{date_to_tag(date)}.rou.xml"


def vehicle_id_for(date: str, record_index: int) -> str:
    return f"veh_{date_to_compact(date)}_{record_index:09d}"


def parse_source_date(name: str) -> Optional[str]:
    match = SOURCE_NAME_RE.match(Path(name).name)
    if not match:
        return None
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"


def parse_route_date(name: str) -> Optional[str]:
    match = ROUTE_NAME_RE.match(Path(name).name)
    if not match:
        return None
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"


def list_repaired_files(input_dir: Path) -> List[Path]:
    return sorted(path for path in Path(input_dir).glob("*.json") if path.is_file())


def discover_dates(input_dir: Path) -> Tuple[List[str], List[str], List[str]]:
    files = list_repaired_files(input_dir)
    dates: List[str] = []
    extras: List[str] = []
    for path in files:
        date = parse_source_date(path.name)
        if date is None:
            extras.append(path.name)
            continue
        dates.append(date)
    missing = [date for date in EXPECTED_DATES if date not in dates]
    unexpected = [date for date in dates if date not in EXPECTED_DATES]
    extra_names = extras + [source_name_for_date(date) for date in unexpected]
    return dates, missing, extra_names


def xml_attr(value: str) -> str:
    return escape(str(value), {'"': "&quot;", "'": "&apos;"})


def format_depart(raw: Any, parsed: float) -> str:
    if isinstance(raw, bool):
        raise PrepareError("startTime is boolean")
    if isinstance(raw, int):
        return str(raw)
    if isinstance(raw, float) and parsed.is_integer():
        return str(int(parsed))
    if isinstance(raw, str) and raw.strip().lstrip("+-").isdigit():
        return str(int(parsed))
    if isinstance(raw, str):
        return raw.strip()
    if parsed.is_integer():
        return str(int(parsed))
    return repr(parsed)


def canon_number(value: Any) -> float:
    return float(value)


def vehicle_fingerprint(vehicle: Dict[str, Any]) -> Tuple[Tuple[str, Tuple[str, Any]], ...]:
    items: List[Tuple[str, Tuple[str, Any]]] = []
    for key in sorted(vehicle):
        val = vehicle[key]
        if isinstance(val, bool):
            items.append((key, ("bool", val)))
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            items.append((key, ("num", float(val))))
        else:
            items.append((key, ("other", val)))
    return tuple(items)


PASSENGER_FP = vehicle_fingerprint(PASSENGER_VEHICLE)
FREIGHT_FP = vehicle_fingerprint(FREIGHT_VEHICLE)


def fingerprint_as_dict(fp: Tuple[Tuple[str, Tuple[str, Any]], ...]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, (kind, val) in fp:
        if kind == "num" and float(val).is_integer():
            out[key] = int(val)
        else:
            out[key] = val
    return out


def classify_vehicle(vehicle: Any) -> str:
    if not isinstance(vehicle, dict):
        return "other"
    fp = vehicle_fingerprint(vehicle)
    if fp == PASSENGER_FP:
        return "passenger"
    if fp == FREIGHT_FP:
        return "freight"
    return "other"


def parse_start_time(raw: Any) -> float:
    if raw is None:
        raise PrepareError("startTime is missing")
    if isinstance(raw, bool):
        raise PrepareError("startTime is boolean")
    if isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, str):
        text = raw.strip().lower()
        if text in {"", "nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
            raise PrepareError(f"startTime is not a finite number: {raw!r}")
        try:
            value = float(raw)
        except ValueError as exc:
            raise PrepareError(f"startTime is unparseable: {raw!r}") from exc
    else:
        raise PrepareError(f"startTime has unsupported type {type(raw).__name__}")
    if math.isnan(value) or math.isinf(value):
        raise PrepareError(f"startTime is not finite: {raw!r}")
    return value


def coerce_route(route: Any) -> List[str]:
    if not isinstance(route, list):
        raise PrepareError(f"route is not a list: {type(route).__name__}")
    edges: List[str] = []
    for item in route:
        if not isinstance(item, str):
            raise PrepareError(
                f"route edge is {type(item).__name__}, IDs must remain strings"
            )
        edges.append(item)
    return edges


@dataclass
class Topology:
    net_path: str
    net_sha256: str
    normal_set: Set[str]
    internal_set: Set[str]
    adj_set: Dict[str, Set[str]]

    def classify_edge(self, edge_id: str) -> str:
        if is_internal_edge_id(edge_id) or edge_id in self.internal_set:
            return "internal"
        if edge_id not in self.normal_set:
            return "unknown"
        return "normal"

    def has_connection(self, frm: str, to: str) -> bool:
        return to in self.adj_set.get(frm, ())


def topology_from_graph(graph: Any, net_path: Path) -> Topology:
    return Topology(
        net_path=str(net_path),
        net_sha256=str(graph.net_sha256),
        normal_set=set(graph.normal_set),
        internal_set=set(graph.internal_ids),
        adj_set={key: set(vals) for key, vals in graph.adj_set.items()},
    )


def load_topology(net_path: Path) -> Topology:
    graph = parse_sumo_net(Path(net_path))
    return topology_from_graph(graph, Path(net_path))


def save_pickle_atomic(obj: Any, final_path: Path, tmp_dir: Path) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=final_path.name + ".",
        suffix=".partial",
        dir=str(tmp_dir),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        import pickle

        with tmp_path.open("wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, final_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def load_pickle(path: Path) -> Any:
    import pickle

    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def write_json_atomic(obj: Any, final_path: Path, tmp_dir: Path) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=final_path.name + ".",
        suffix=".partial",
        dir=str(tmp_dir),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, final_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def write_text_atomic(text: str, final_path: Path, tmp_dir: Path) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=final_path.name + ".",
        suffix=".partial",
        dir=str(tmp_dir),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            fh.write(text)
            if not text.endswith("\n"):
                fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, final_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def write_tsv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(fieldnames),
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def file_meta(path: Path, *, with_hash: bool = True) -> Dict[str, Any]:
    st = path.stat()
    meta = {
        "path": str(path),
        "bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "mtime_iso": local_iso(st.st_mtime),
        "mode": stat.S_IMODE(st.st_mode),
    }
    if with_hash:
        meta["sha256"] = sha256_file(path)
    return meta


def hash_file_row(path: Path, date: Optional[str] = None) -> Dict[str, Any]:
    meta = file_meta(path, with_hash=True)
    meta["date"] = date or parse_source_date(path.name) or parse_route_date(path.name) or ""
    meta["source_file"] = str(path)
    return meta


def resolve_net_file(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise PrepareError(f"--net-file does not exist: {path}")
        return path
    expected = DEFAULT_NET_FILE_EXPECTED
    if expected.is_file():
        return expected
    root = DATA_ROOT / "simulation_data" / "road_network"
    candidates = sorted(root.rglob("*.net.xml"))
    if not candidates:
        raise PrepareError(
            f"SUMO net not found at {expected} and no *.net.xml under {root}"
        )
    if len(candidates) > 1:
        listed = "\n".join(str(item) for item in candidates)
        raise PrepareError(
            "Expected net file is missing and multiple *.net.xml candidates exist; "
            f"refusing to guess:\n{listed}"
        )
    return candidates[0]


def collect_env() -> Dict[str, Any]:
    python_version = sys.version.replace("\n", " ")
    sumo_home = os.environ.get("SUMO_HOME", "")
    sumo_bin = shutil_which("sumo")
    sumo_version = None
    if sumo_bin:
        proc = subprocess.run(
            [sumo_bin, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        sumo_version = (proc.stdout or proc.stderr).strip().splitlines()[0] if (proc.stdout or proc.stderr) else ""
    return {
        "python_version": f"Python {sys.version.split()[0]}",
        "python_version_full": python_version,
        "python_executable": sys.executable,
        "sumo_bin": sumo_bin,
        "sumo_version": sumo_version,
        "SUMO_HOME": sumo_home,
        "hostname": os.uname().nodename,
        "pid": os.getpid(),
        "cwd": os.getcwd(),
    }


def shutil_which(name: str) -> Optional[str]:
    from shutil import which

    return which(name)


def max_overlap(intervals: Sequence[Tuple[float, float]]) -> int:
    events: List[Tuple[float, int]] = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    events.sort(key=lambda item: (item[0], -item[1]))
    current = 0
    peak = 0
    for _, delta in events:
        current += delta
        if current > peak:
            peak = current
    return peak


def log_print(log_fh: Any, message: str) -> None:
    line = f"{local_iso()} pid={os.getpid()} {message}"
    print(line, flush=True)
    if log_fh is not None:
        log_fh.write(line + "\n")
        log_fh.flush()


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{local_iso()} pid={os.getpid()} {message}\n")


def snapshot_original_inputs() -> Dict[str, Any]:
    trips = []
    if ORIGINAL_TRIPS_DIR.is_dir():
        for path in sorted(ORIGINAL_TRIPS_DIR.glob("data_2023_04_*_type_filtered.json")):
            trips.append(file_meta(path, with_hash=False))
    net_meta = file_meta(ORIGINAL_NET_FILE, with_hash=True) if ORIGINAL_NET_FILE.is_file() else None
    return {"original_net": net_meta, "original_trips": trips}


def scan_code_in_data(root: Path) -> List[str]:
    found: List[str] = []
    if not root.exists():
        return found
    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in PREPARED_FORBIDDEN_CODE_SUFFIXES or path.name in CODE_NAMES:
            found.append(str(path))
    return found


def scan_large_or_day_data_in_code(code_root: Path, min_bytes: int = 10 * 1024 * 1024) -> List[str]:
    found: List[str] = []
    for path in Path(code_root).rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        name = path.name
        suffix = path.suffix.lower()
        size = path.stat().st_size
        if name.endswith(".rou.xml") or suffix == ".rou":
            found.append(f"{path} ({size} bytes)")
            continue
        if DAY_JSON_RE.search(name) and suffix == ".json":
            found.append(f"{path} ({size} bytes)")
            continue
        if name.endswith(".net.xml"):
            found.append(f"{path} ({size} bytes)")
            continue
        if size >= min_bytes:
            found.append(f"{path} ({size} bytes)")
    return found


def prepared_route_files(prepared_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in Path(prepared_dir).glob("routes_*.rou.xml")
        if path.is_file()
    )


ROUTE_ERROR_HINTS = (
    "route",
    "edge",
    "connection",
    "vehicle type",
    "vtype",
    "disconnected",
    "unknown",
    "not known",
    "invalid",
    "not allowed",
    "depart",
)


def classify_sumo_output(stdout: str, stderr: str) -> Dict[str, Any]:
    errors: List[str] = []
    network_warnings: List[str] = []
    route_warnings: List[str] = []
    other_warnings: List[str] = []
    for raw_line in (stdout + "\n" + stderr).splitlines():
        line = raw_line.strip("\n")
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if stripped.startswith("Error:") or lower.startswith("error:"):
            errors.append(stripped)
            continue
        if "missing yellow phase" in lower or "tllogic" in lower:
            network_warnings.append(stripped)
            continue
        if stripped.startswith("Warning:") or lower.startswith("warning:"):
            if any(hint in lower for hint in ROUTE_ERROR_HINTS):
                route_warnings.append(stripped)
            else:
                other_warnings.append(stripped)
    return {
        "errors": errors,
        "network_warnings": network_warnings,
        "route_warnings": route_warnings,
        "other_warnings": other_warnings,
        "n_errors": len(errors),
        "n_network_warnings": len(network_warnings),
        "n_route_warnings": len(route_warnings),
        "n_other_warnings": len(other_warnings),
    }


def sumo_load_command(net_file: Path, route_file: Path) -> List[str]:
    return [
        "sumo",
        "--net-file",
        str(net_file),
        "--route-files",
        str(route_file),
        "--route-steps",
        "0",
        "--begin",
        "0",
        "--end",
        "1",
        "--ignore-route-errors",
        "false",
        "--no-step-log",
        "true",
        "--duration-log.disable",
        "true",
    ]


def run_sumo_load(
    net_file: Path,
    route_file: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> Dict[str, Any]:
    cmd = sumo_load_command(net_file, route_file)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    pid = proc.pid
    stdout, stderr = proc.communicate()
    elapsed = time.time() - started
    stdout = stdout or ""
    stderr = stderr or ""
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    classified = classify_sumo_output(stdout, stderr)
    ok = (
        proc.returncode == 0
        and classified["n_errors"] == 0
        and classified["n_route_warnings"] == 0
    )
    return {
        "command": cmd,
        "pid": pid,
        "exit_code": proc.returncode,
        "elapsed_s": round(elapsed, 6),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "ok": ok,
        **classified,
    }


def compare_hash_rows(
    before_rows: Sequence[Dict[str, Any]],
    after_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    before_map = {row["path"]: row for row in before_rows}
    after_map = {row["path"]: row for row in after_rows}
    mismatches: List[Dict[str, Any]] = []
    for path in sorted(set(before_map) | set(after_map)):
        left = before_map.get(path)
        right = after_map.get(path)
        if left is None or right is None:
            mismatches.append(
                {
                    "path": path,
                    "before": None if left is None else left.get("sha256"),
                    "after": None if right is None else right.get("sha256"),
                    "reason": "missing_before" if left is None else "missing_after",
                }
            )
            continue
        if left.get("sha256") != right.get("sha256") or left.get("bytes") != right.get("bytes"):
            mismatches.append(
                {
                    "path": path,
                    "before": left.get("sha256"),
                    "after": right.get("sha256"),
                    "reason": "sha256_or_size_changed",
                    "bytes_before": left.get("bytes"),
                    "bytes_after": right.get("bytes"),
                }
            )
    return mismatches
