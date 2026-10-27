#!/usr/bin/env python3
"""Launch one single-thread SUMO process per refined Xuancheng26 route day.

Dates come from route filenames. This script does not modify the network,
route files, night_peak.add.xml, or vehicle / signal behavior.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from xml.sax.saxutils import escape

CODE_ROOT = Path("/home/tzhang174/TrafficFlow/dataset/xuancheng26")
NET_FILE = CODE_ROOT / "xuancheng_tls_rebuilt.net.xml"
ADDITIONAL_FILE = CODE_ROOT / "night_peak.add.xml"
CONFIG_DIR = CODE_ROOT / "fcd_configs"
ROUTE_DIR = Path(
    "/data5/tzhang174/TrafficFlow/xuancheng26/simulation_data/trip_data/refined"
)
FCD_ROOT = Path("/data5/tzhang174/TrafficFlow/xuancheng26/simulation_data/fcd_data")
DATA_DISK = Path("/data5")

ROUTE_NAME_RE = re.compile(r"^refined_routes_(\d{4})_(\d{2})_(\d{2})\.rou\.xml$")
# Gate for this batch. The date list itself is discovered from filenames.
REQUIRED_ROUTE_COUNT = 29
MIN_FREE_BYTES = 100 * 1024**3
SEED = 42

PROTECTED_INPUTS = (NET_FILE, ADDITIONAL_FILE, ROUTE_DIR)


@dataclass(frozen=True)
class Day:
    year: str
    month: str
    day: str
    route_path: Path

    @property
    def tag(self) -> str:
        return f"{self.year}_{self.month}_{self.day}"

    @property
    def iso(self) -> str:
        return f"{self.year}-{self.month}-{self.day}"


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"{utc_now()} {message}", flush=True)


def sumo_binary() -> str:
    found = shutil.which("sumo")
    if not found:
        raise SystemExit("sumo binary not found on PATH")
    return found


def sumo_version(binary: str) -> str:
    proc = subprocess.run(
        [binary, "--version"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    first = ""
    for line in proc.stdout.splitlines():
        if "SUMO" in line or "sumo" in line:
            first = line.strip()
            break
    if proc.returncode != 0 or not first:
        raise SystemExit(f"sumo --version failed:\n{proc.stdout}")
    return first


def disk_free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return int(usage.free)


def discover_routes(route_dir: Path) -> list[Day]:
    if not route_dir.is_dir():
        raise SystemExit(f"route directory missing: {route_dir}")
    days: list[Day] = []
    seen: dict[str, Path] = {}
    for path in sorted(route_dir.glob("refined_routes_2023_04_*.rou.xml")):
        match = ROUTE_NAME_RE.match(path.name)
        if match is None:
            raise SystemExit(f"route filename does not match expected pattern: {path.name}")
        year, month, day = match.groups()
        tag = f"{year}_{month}_{day}"
        if tag in seen:
            raise SystemExit(f"duplicate route date {tag}: {seen[tag]} and {path}")
        seen[tag] = path
        days.append(Day(year, month, day, path.resolve()))
    days.sort(key=lambda item: item.tag)
    return days


def require_inputs(days: list[Day]) -> None:
    missing = [str(path) for path in (NET_FILE, ADDITIONAL_FILE) if not path.is_file()]
    if missing:
        raise SystemExit("missing input files:\n" + "\n".join(missing))
    if len(days) != REQUIRED_ROUTE_COUNT:
        found = ", ".join(day.iso for day in days) or "(none)"
        raise SystemExit(
            f"expected exactly {REQUIRED_ROUTE_COUNT} route files, found {len(days)}: {found}"
        )
    for day in days:
        if not day.route_path.is_file() or day.route_path.stat().st_size <= 0:
            raise SystemExit(f"route file missing or empty: {day.route_path}")


def require_disk_space() -> int:
    free = disk_free_bytes(DATA_DISK)
    log(f"disk_free_bytes path={DATA_DISK} free={free} ({free / 1024**3:.1f} GiB)")
    if free < MIN_FREE_BYTES:
        raise SystemExit(
            f"/data5 free space {free / 1024**3:.1f} GiB is below {MIN_FREE_BYTES / 1024**3:.0f} GiB; not starting"
        )
    return free


def output_paths(day: Day, out_dir: Path) -> dict[str, Path]:
    tag = day.tag
    return {
        "fcd": out_dir / f"fcd_{tag}.xml.gz",
        "summary": out_dir / f"summary_{tag}.xml.gz",
        "tripinfo": out_dir / f"tripinfo_{tag}.xml.gz",
        "statistics": out_dir / f"statistics_{tag}.xml.gz",
        "sumo_log": out_dir / "sumo.log",
        "error_log": out_dir / "error.log",
        "console_log": out_dir / "console.log",
    }


def config_path_for(day: Day) -> Path:
    return CONFIG_DIR / f"{day.tag}.sumocfg"


def xml_value(value: str) -> str:
    return escape(value, {'"': "&quot;"})


def write_sumocfg(
    day: Day,
    cfg_path: Path,
    out_dir: Path,
    *,
    begin: str = "0",
    end: str = "86400",
) -> Path:
    """Write a SUMO 1.19 configuration.

    device.fcd.period belongs in fcd_device, and seed belongs in
    random_number. Those are the schema sections for this SUMO version.
    Rerouting is left unset, so vehicles keep the routes in the route file.
    """
    outputs = output_paths(day, out_dir)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{xml_value(str(NET_FILE))}"/>
        <route-files value="{xml_value(str(day.route_path))}"/>
        <additional-files value="{xml_value(str(ADDITIONAL_FILE))}"/>
    </input>
    <time>
        <begin value="{xml_value(begin)}"/>
        <end value="{xml_value(end)}"/>
        <step-length value="1"/>
    </time>
    <processing>
        <time-to-teleport value="600"/>
        <ignore-route-errors value="false"/>
        <max-depart-delay value="-1"/>
        <route-steps value="200"/>
        <threads value="1"/>
    </processing>
    <output>
        <fcd-output value="{xml_value(str(outputs["fcd"]))}"/>
        <summary-output value="{xml_value(str(outputs["summary"]))}"/>
        <summary-output.period value="60"/>
        <tripinfo-output value="{xml_value(str(outputs["tripinfo"]))}"/>
        <tripinfo-output.write-unfinished value="true"/>
        <tripinfo-output.write-undeparted value="true"/>
        <statistic-output value="{xml_value(str(outputs["statistics"]))}"/>
    </output>
    <fcd_device>
        <device.fcd.period value="1"/>
    </fcd_device>
    <random_number>
        <seed value="{SEED}"/>
    </random_number>
    <report>
        <no-step-log value="true"/>
        <duration-log.statistics value="true"/>
        <log value="{xml_value(str(outputs["sumo_log"]))}"/>
        <error-log value="{xml_value(str(outputs["error_log"]))}"/>
        <log.timestamps value="true"/>
        <log.processid value="true"/>
    </report>
</configuration>
"""
    tmp_path = cfg_path.with_suffix(cfg_path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, cfg_path)
    return cfg_path


def iter_process_cmdlines() -> Iterable[tuple[int, str]]:
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        cmdline_path = entry / "cmdline"
        try:
            raw = cmdline_path.read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        command = raw.replace(b"\0", b" ").decode("utf-8", "replace")
        yield int(entry.name), command


def sumo_pids_for_day(day: Day) -> list[int]:
    needles = (
        str(config_path_for(day)),
        str(FCD_ROOT / day.tag / f"fcd_{day.tag}.xml.gz"),
        f"{day.tag}.sumocfg",
    )
    pids: list[int] = []
    for pid, command in iter_process_cmdlines():
        binary = command.split(" ", 1)[0]
        if Path(binary).name not in {"sumo", "sumoD"}:
            continue
        if any(needle in command for needle in needles):
            pids.append(pid)
    return pids


def assert_safe_day_dir(day_dir: Path) -> Path:
    resolved = day_dir.resolve()
    root = FCD_ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f"refusing to clean outside FCD root: {resolved}")
    if resolved.name != day_dir.name:
        raise RuntimeError(f"refusing to clean unexpected day directory: {resolved}")
    for protected in PROTECTED_INPUTS:
        protected_resolved = protected.resolve()
        if resolved == protected_resolved or protected_resolved in resolved.parents:
            raise RuntimeError(f"refusing to clean protected input: {resolved}")
        if resolved in protected_resolved.parents:
            raise RuntimeError(f"refusing to clean a parent of protected input: {resolved}")
    return resolved


def clean_failed_outputs(day: Day, reason: str) -> list[str]:
    day_dir = assert_safe_day_dir(FCD_ROOT / day.tag)
    if not day_dir.exists():
        return []
    removed: list[str] = []
    for path in sorted(day_dir.rglob("*")):
        if not path.is_file() and not path.is_symlink():
            continue
        resolved = path.resolve()
        if day_dir not in resolved.parents and resolved != day_dir:
            raise RuntimeError(f"refusing to remove {resolved}")
        resolved.unlink()
        removed.append(str(resolved))
    for path in sorted(day_dir.rglob("*"), reverse=True):
        if path.is_dir():
            path.rmdir()
    log(f"[cleanup] date={day.iso} reason={reason} removed={len(removed)}")
    for name in removed:
        log(f"[cleanup] removed {name}")
    cleanup_log = FCD_ROOT / "cleanup_actions.log"
    FCD_ROOT.mkdir(parents=True, exist_ok=True)
    with cleanup_log.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"time": utc_now(), "date": day.iso, "reason": reason, "removed": removed},
                ensure_ascii=False,
            )
            + "\n"
        )
    return removed


def classify_day(day: Day) -> str:
    running = sumo_pids_for_day(day)
    fcd_path = FCD_ROOT / day.tag / f"fcd_{day.tag}.xml.gz"
    if running:
        return "running"
    if fcd_path.is_file() and fcd_path.stat().st_size > 0:
        return "complete"
    day_dir = FCD_ROOT / day.tag
    has_remnant = False
    if fcd_path.is_file() and fcd_path.stat().st_size == 0:
        has_remnant = True
    if day_dir.is_dir():
        for path in day_dir.iterdir():
            if path.is_file():
                has_remnant = True
                break
    if has_remnant:
        return "failed_remnant"
    return "fresh"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def gzip_text_head(path: Path, limit: int = 200000) -> str:
    chunks: list[str] = []
    size = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        while size < limit:
            block = handle.read(65536)
            if not block:
                break
            chunks.append(block)
            size += len(block)
    return "".join(chunks)


def check_fcd_sample(fcd_path: Path) -> dict:
    text = gzip_text_head(fcd_path)
    if "<timestep" not in text:
        raise RuntimeError(f"FCD gzip has no timestep: {fcd_path}")
    times: list[float] = []
    for match in re.finditer(r"<timestep\b[^>]*\btime=\"([0-9.]+)\"", text):
        times.append(float(match.group(1)))
        if len(times) >= 5:
            break
    if len(times) < 2:
        raise RuntimeError(f"FCD sample has fewer than 2 timesteps: {times}")
    diffs = [round(b - a, 3) for a, b in zip(times, times[1:])]
    if any(diff != 1.0 for diff in diffs):
        raise RuntimeError(f"FCD timestep interval is not 1.00 second: times={times} diffs={diffs}")
    vehicle = re.search(r"<vehicle\b([^>]*)/>", text)
    if vehicle is None:
        vehicle = re.search(r"<vehicle\b([^>]*)>", text)
    if vehicle is None:
        raise RuntimeError("FCD sample has timesteps but no vehicle element")
    attrs = vehicle.group(1)
    required = ("id=", "x=", "y=", "angle=", "type=", "speed=", "pos=", "lane=")
    missing = [name for name in required if name not in attrs]
    if missing:
        raise RuntimeError(f"FCD vehicle is missing attributes {missing}: {attrs[:300]}")
    return {"times": times, "diffs": diffs, "vehicle_attrs": " ".join(attrs.split())[:300]}


def smoke_test(binary: str, days: list[Day]) -> None:
    preferred = next((day for day in days if day.iso == "2023-04-01"), days[0])
    tmp_root = Path(tempfile.mkdtemp(prefix="xuancheng26_fcd_smoke_"))
    log(f"[smoke] temp={tmp_root} date={preferred.iso}")
    try:
        cfg_path = tmp_root / f"{preferred.tag}.sumocfg"
        out_dir = tmp_root / preferred.tag
        write_sumocfg(preferred, cfg_path, out_dir, begin="0", end="2")
        cfg_text = cfg_path.read_text(encoding="utf-8")
        if str(ADDITIONAL_FILE) not in cfg_text:
            raise RuntimeError("smoke config does not reference night_peak.add.xml")
        if 'value="86400"' in cfg_text:
            raise RuntimeError("smoke config unexpectedly uses the full-day end time")
        if "<routing>" in cfg_text or "device.rerouting" in cfg_text:
            raise RuntimeError("smoke config still enables device rerouting")
        proc = subprocess.run(
            [binary, "-c", str(cfg_path)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=600,
        )
        console_path = tmp_root / "smoke_console.txt"
        console_path.write_text(proc.stdout, encoding="utf-8")
        outputs = output_paths(preferred, out_dir)
        if proc.returncode != 0:
            error_tail = ""
            if outputs["error_log"].is_file():
                error_tail = outputs["error_log"].read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(
                f"smoke SUMO exit={proc.returncode}\n{proc.stdout[-4000:]}\n{error_tail}"
            )
        for key in ("fcd", "summary", "tripinfo", "statistics"):
            path = outputs[key]
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"smoke output missing or empty: {path}")
            with gzip.open(path, "rb") as handle:
                magic = handle.read(64)
            if not magic:
                raise RuntimeError(f"smoke gzip could not be read: {path}")
        sumo_log = outputs["sumo_log"].read_text(encoding="utf-8", errors="replace")
        if str(ADDITIONAL_FILE) not in sumo_log and "night_peak.add.xml" not in sumo_log:
            raise RuntimeError("SUMO log does not show night_peak.add.xml being loaded")
        error_text = outputs["error_log"].read_text(encoding="utf-8", errors="replace")
        fatal_lines = [
            line
            for line in error_text.splitlines()
            if "Error:" in line or "Quitting (on error)" in line
        ]
        if fatal_lines:
            raise RuntimeError("smoke error.log contains fatal lines:\n" + "\n".join(fatal_lines[:20]))
        sample = check_fcd_sample(outputs["fcd"])
        log(f"[smoke] ok times={sample['times']} diffs={sample['diffs']}")
        log(f"[smoke] vehicle {sample['vehicle_attrs']}")
        log("[smoke] night_peak.add.xml loaded; temporary outputs will be deleted")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        if tmp_root.exists():
            raise SystemExit(f"failed to delete smoke output: {tmp_root}")
        log(f"[smoke] deleted {tmp_root}")


def launch(binary: str, days: list[Day], workers: int, *, overwrite: bool = False) -> int:
    if workers < 1:
        raise SystemExit("--workers must be >= 1")
    FCD_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    decisions: dict[str, str] = {}
    to_run: list[Day] = []
    for day in days:
        state = classify_day(day)
        decisions[day.iso] = state
        if state == "running":
            log(f"[skip] date={day.iso} reason=sumo_already_running pids={sumo_pids_for_day(day)}")
            continue
        if state == "complete" and not overwrite:
            fcd_path = FCD_ROOT / day.tag / f"fcd_{day.tag}.xml.gz"
            log(
                f"[skip] date={day.iso} reason=existing_nonempty_fcd "
                f"bytes={fcd_path.stat().st_size} path={fcd_path}"
            )
            continue
        if state == "failed_remnant" or (state == "complete" and overwrite):
            reason = (
                "overwrite_requested"
                if state == "complete"
                else "zero_byte_or_failed_remnant_without_sumo_process"
            )
            clean_failed_outputs(day, reason)
        to_run.append(day)

    if any(state == "complete" for state in decisions.values()) and not overwrite:
        log("[protect] at least one day already has a non-empty FCD; those days were not overwritten")
    if overwrite:
        log("[overwrite] existing day outputs will be replaced before SUMO starts")
    if not to_run:
        log("[launch] nothing to start")
        return 0

    jobs: dict[str, dict] = {}
    running: list[tuple[Day, subprocess.Popen]] = []
    pending = list(to_run)
    failed: list[str] = []

    def start_one(day: Day) -> None:
        if sumo_pids_for_day(day):
            log(f"[skip] date={day.iso} reason=sumo_appeared_before_start")
            decisions[day.iso] = "running"
            return
        out_dir = FCD_ROOT / day.tag
        cfg_path = write_sumocfg(day, config_path_for(day), out_dir, begin="0", end="86400")
        cfg_text = cfg_path.read_text(encoding="utf-8")
        if str(ADDITIONAL_FILE) not in cfg_text:
            raise SystemExit(f"config missing night_peak.add.xml: {cfg_path}")
        console_path = out_dir / "console.log"
        console_handle = console_path.open("w", encoding="utf-8")
        started = time.time()
        proc = subprocess.Popen(
            [binary, "-c", str(cfg_path)],
            stdout=console_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        console_handle.close()
        jobs[day.iso] = {
            "tag": day.tag,
            "pid": proc.pid,
            "config": str(cfg_path),
            "route": str(day.route_path),
            "output_dir": str(out_dir),
            "additional_files": str(ADDITIONAL_FILE),
            "start_time": utc_now(),
            "start_unix": started,
            "status": "RUNNING",
        }
        running.append((day, proc))
        decisions[day.iso] = "started"
        log(f"[launch] date={day.iso} pid={proc.pid} config={cfg_path}")

    status_path = FCD_ROOT / "batch_status.json"

    def flush_status() -> None:
        payload = {
            "updated_at": utc_now(),
            "sumo": binary,
            "workers": workers,
            "seed": SEED,
            "threads_per_process": 1,
            "route_count": len(days),
            "dates": [day.iso for day in days],
            "decisions": decisions,
            "jobs": jobs,
        }
        write_json(status_path, payload)

    while pending or running:
        while pending and len(running) < workers:
            start_one(pending.pop(0))
            flush_status()
        if not running:
            break
        time.sleep(2)
        still: list[tuple[Day, subprocess.Popen]] = []
        for day, proc in running:
            code = proc.poll()
            if code is None:
                still.append((day, proc))
                continue
            finished = time.time()
            jobs[day.iso]["end_time"] = utc_now()
            jobs[day.iso]["exit_code"] = code
            jobs[day.iso]["elapsed_s"] = round(finished - float(jobs[day.iso]["start_unix"]), 3)
            if code == 0:
                jobs[day.iso]["status"] = "SUCCESS"
                log(f"[done] date={day.iso} pid={proc.pid} exit=0")
            else:
                jobs[day.iso]["status"] = "FAILED"
                failed.append(day.iso)
                log(f"[fail] date={day.iso} pid={proc.pid} exit={code}")
        running = still
        flush_status()

    success = sum(1 for item in jobs.values() if item.get("status") == "SUCCESS")
    log(
        f"[summary] success={success} failed={len(failed)} "
        f"skipped_complete={sum(1 for state in decisions.values() if state == 'complete')} "
        f"skipped_running={sum(1 for state in decisions.values() if state == 'running')}"
    )
    if failed:
        log("[summary] FAILED " + ",".join(failed))
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Xuancheng26 full-day SUMO FCD simulations")
    parser.add_argument("--workers", type=int, default=29)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing outputs for every discovered date, then run those days again",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Load one day from 0 to 2 seconds in a temporary directory, then delete it",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    binary = sumo_binary()
    version = sumo_version(binary)
    log(f"sumo_binary={binary}")
    log(f"sumo_version={version}")
    days = discover_routes(ROUTE_DIR)
    require_inputs(days)
    require_disk_space()
    log("dates=" + ",".join(day.iso for day in days))
    log(f"route_count={len(days)} net={NET_FILE} additional={ADDITIONAL_FILE}")
    if args.smoke_test:
        smoke_test(binary, days)
        log("[smoke] passed")
        return 0
    return launch(binary, days, args.workers, overwrite=args.overwrite)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log(f"[fatal] {exc}")
        raise
