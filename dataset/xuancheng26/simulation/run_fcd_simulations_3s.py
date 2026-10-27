#!/usr/bin/env python3
"""Run the 29-day Xuancheng26 FCD batch on the 3-second TLS network.

This is a separate runner from run_fcd_simulations.py. It does not write the
original fcd_data directory, the original sumocfg files, or the rebuilt net.

Simulation options match the 2026-10-13 batch that finished all 29 days:
begin 0, end 86400, step-length 1, time-to-teleport 600, one thread, seed 42,
device.fcd.period 1, and no device.rerouting section. SUMO 1.19 compresses
FCD because the output path ends in .xml.gz. That version has no
--fcd-output.compression option.
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
NET_FILE = CODE_ROOT / "xuancheng_tls_3s.net.xml"
ADDITIONAL_FILE = CODE_ROOT / "night_peak_3s.add.xml"
CONFIG_DIR = CODE_ROOT / "fcd_configs_3s"
ROUTE_DIR = Path(
    "/data5/tzhang174/TrafficFlow/xuancheng26/simulation_data/trip_data/refined"
)
FCD_ROOT = Path("/data5/tzhang174/TrafficFlow/xuancheng26/simulation_data/fcd_data_3s")
OLD_FCD_ROOT = Path("/data5/tzhang174/TrafficFlow/xuancheng26/simulation_data/fcd_data")
OLD_NET = CODE_ROOT / "xuancheng_tls_rebuilt.net.xml"
OLD_ADDITIONAL = CODE_ROOT / "night_peak.add.xml"
DATA_DISK = Path("/data5")

ROUTE_NAME_RE = re.compile(r"^refined_routes_(\d{4})_(\d{2})_(\d{2})\.rou\.xml$")
REQUIRED_ROUTE_COUNT = 29
MIN_FREE_BYTES = 200 * 1024**3
SEED = 42
LOW_DISK_BYTES = 80 * 1024**3

PROTECTED_INPUTS = (
    NET_FILE,
    ADDITIONAL_FILE,
    OLD_NET,
    OLD_ADDITIONAL,
    ROUTE_DIR,
    OLD_FCD_ROOT,
    CODE_ROOT / "fcd_configs",
)


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
    return int(shutil.disk_usage(path).free)


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
    if OLD_NET.resolve() == NET_FILE.resolve():
        raise SystemExit("3s net path aliases the original rebuilt net")
    if FCD_ROOT.resolve() == OLD_FCD_ROOT.resolve():
        raise SystemExit("3s output directory aliases the original FCD directory")


def require_disk_space() -> int:
    free = disk_free_bytes(DATA_DISK)
    log(f"disk_free_bytes path={DATA_DISK} free={free} ({free / 1024**3:.1f} GiB)")
    if free < MIN_FREE_BYTES:
        raise SystemExit(
            f"/data5 free space {free / 1024**3:.1f} GiB is below "
            f"{MIN_FREE_BYTES / 1024**3:.0f} GiB; not starting"
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
        "result": out_dir / "day_result.json",
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
    if "device.rerouting" in text or "<routing>" in text:
        raise SystemExit(f"refusing to write a config that enables rerouting: {cfg_path}")
    if str(OLD_NET) in text or str(OLD_ADDITIONAL) in text:
        raise SystemExit(f"config still points at the original TLS inputs: {cfg_path}")
    if not str(outputs["fcd"]).endswith(".xml.gz"):
        raise SystemExit(f"FCD output is not a gzip path: {outputs['fcd']}")
    tmp_path = cfg_path.with_suffix(cfg_path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, cfg_path)
    return cfg_path


def iter_process_cmdlines() -> Iterable[tuple[int, str]]:
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        yield int(entry.name), raw.replace(b"\0", b" ").decode("utf-8", "replace")


def sumo_pids_for_day(day: Day) -> list[int]:
    needles = (
        str(config_path_for(day)),
        str(FCD_ROOT / day.tag / f"fcd_{day.tag}.xml.gz"),
        f"fcd_configs_3s/{day.tag}.sumocfg",
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
    old_root = OLD_FCD_ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f"refusing to clean outside 3s FCD root: {resolved}")
    if resolved == old_root or old_root in resolved.parents or resolved in old_root.parents:
        raise RuntimeError(f"refusing to clean the original FCD tree: {resolved}")
    if resolved.parent != root or not re.fullmatch(r"2023_04_\d{2}", resolved.name):
        raise RuntimeError(f"refusing to clean unexpected day directory: {resolved}")
    for protected in PROTECTED_INPUTS:
        protected_resolved = protected.resolve()
        if resolved == protected_resolved or protected_resolved in resolved.parents:
            raise RuntimeError(f"refusing to clean protected input: {resolved}")
        if resolved in protected_resolved.parents:
            raise RuntimeError(f"refusing to clean a parent of protected input: {resolved}")
    return resolved


def read_result(day: Day) -> Optional[dict]:
    path = FCD_ROOT / day.tag / "day_result.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def classify_day(day: Day) -> str:
    if sumo_pids_for_day(day):
        return "running"
    result = read_result(day)
    fcd_path = FCD_ROOT / day.tag / f"fcd_{day.tag}.xml.gz"
    if (
        result
        and result.get("status") == "SUCCESS"
        and result.get("exit_code") == 0
        and fcd_path.is_file()
        and fcd_path.stat().st_size > 0
        and str(result.get("statistics", {}).get("end")) in {"86400.00", "86400"}
    ):
        return "complete"
    day_dir = FCD_ROOT / day.tag
    if day_dir.is_dir() and any(path.is_file() for path in day_dir.iterdir()):
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
    for match in re.finditer(r'<timestep\b[^>]*\btime="([0-9.]+)"', text):
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


def tag_attrs(text: str, tag: str) -> dict[str, str]:
    match = re.search(fr"<{tag}\b([^>]*)/?>", text)
    if match is None:
        return {}
    return dict(re.findall(r'(\w+)="([^"]*)"', match.group(1)))


def parse_statistics(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size <= 0:
        return {}
    text = gzip.open(path, "rt", encoding="utf-8", errors="replace").read()
    body = text.split("</configuration>", 1)[-1]
    performance = tag_attrs(body, "performance")
    vehicles = tag_attrs(body, "vehicles")
    teleports = tag_attrs(body, "teleports")
    safety = tag_attrs(body, "safety")
    config = text.split("</configuration>", 1)[0]
    return {
        "end": performance.get("end"),
        "duration": performance.get("duration"),
        "clock_duration": performance.get("clockDuration"),
        "loaded": vehicles.get("loaded"),
        "inserted": vehicles.get("inserted"),
        "running": vehicles.get("running"),
        "waiting": vehicles.get("waiting"),
        "teleports": teleports.get("total"),
        "teleports_jam": teleports.get("jam"),
        "teleports_yield": teleports.get("yield"),
        "teleports_wrong_lane": teleports.get("wrongLane"),
        "collisions": safety.get("collisions"),
        "emergency_stops": safety.get("emergencyStops"),
        "emergency_braking": safety.get("emergencyBraking"),
        "loaded_net": NET_FILE.name in config or str(NET_FILE) in config,
        "loaded_additional": ADDITIONAL_FILE.name in config or str(ADDITIONAL_FILE) in config,
        "loaded_rerouting": "device.rerouting" in config,
    }


def fatal_error_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return [
        line
        for line in text.splitlines()
        if "Error:" in line or "Quitting (on error)" in line
    ]


def archive_failure(day: Day, reason: str) -> None:
    day_dir = FCD_ROOT / day.tag
    if not day_dir.is_dir():
        return
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    archive = FCD_ROOT / "failed_attempts" / day.tag / stamp
    archive.mkdir(parents=True, exist_ok=True)
    for name in ("console.log", "error.log", "sumo.log", "day_result.json"):
        source = day_dir / name
        if source.is_file():
            shutil.copy2(source, archive / name)
    (archive / "reason.txt").write_text(reason + "\n", encoding="utf-8")
    log(f"[archive] date={day.iso} path={archive} reason={reason}")


def clean_failed_outputs(day: Day, reason: str) -> list[str]:
    if sumo_pids_for_day(day):
        raise RuntimeError(f"refusing to clean a day that still has SUMO running: {day.iso}")
    archive_failure(day, reason)
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
        if path.is_dir() and path != day_dir:
            path.rmdir()
    try:
        day_dir.rmdir()
    except OSError:
        pass
    log(f"[cleanup] date={day.iso} reason={reason} removed={len(removed)}")
    return removed


def smoke_test(binary: str, days: list[Day]) -> None:
    preferred = next((day for day in days if day.iso == "2023-04-01"), days[0])
    tmp_root = Path(tempfile.mkdtemp(prefix="xuancheng26_fcd_3s_smoke_"))
    log(f"[smoke] temp={tmp_root} date={preferred.iso}")
    try:
        cfg_path = tmp_root / f"{preferred.tag}.sumocfg"
        out_dir = tmp_root / preferred.tag
        write_sumocfg(preferred, cfg_path, out_dir, begin="0", end="2")
        cfg_text = cfg_path.read_text(encoding="utf-8")
        for required in (str(NET_FILE), str(ADDITIONAL_FILE), str(preferred.route_path)):
            if required not in cfg_text:
                raise RuntimeError(f"smoke config missing {required}")
        if str(OLD_NET) in cfg_text or "xuancheng_tls_rebuilt.net.xml" in cfg_text:
            raise RuntimeError("smoke config references the original rebuilt net")
        if 'value="86400"' in cfg_text:
            raise RuntimeError("smoke config unexpectedly uses the full-day end time")
        if "<routing>" in cfg_text or "device.rerouting" in cfg_text:
            raise RuntimeError("smoke config enables device rerouting")
        proc = subprocess.run(
            [binary, "-c", str(cfg_path)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=600,
        )
        (tmp_root / "smoke_console.txt").write_text(proc.stdout, encoding="utf-8")
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
                if not handle.read(64):
                    raise RuntimeError(f"smoke gzip could not be read: {path}")
        sumo_log = outputs["sumo_log"].read_text(encoding="utf-8", errors="replace")
        if str(NET_FILE) not in sumo_log:
            raise RuntimeError("SUMO log does not show xuancheng_tls_3s.net.xml being loaded")
        if str(ADDITIONAL_FILE) not in sumo_log:
            raise RuntimeError("SUMO log does not show night_peak_3s.add.xml being loaded")
        # SUMO 1.19 does not echo the route path in sumo.log. The statistics
        # header repeats the configuration, including route-files.
        stats_head = gzip_text_head(outputs["statistics"], limit=20000)
        if str(preferred.route_path) not in stats_head:
            raise RuntimeError("statistics header does not include the original refined route file")
        if str(NET_FILE) not in stats_head or str(ADDITIONAL_FILE) not in stats_head:
            raise RuntimeError("statistics header does not include the 3-second TLS inputs")
        if "device.rerouting" in stats_head:
            raise RuntimeError("statistics header shows device.rerouting")
        fatal = fatal_error_lines(outputs["error_log"])
        if fatal:
            raise RuntimeError("smoke error.log contains fatal lines:\n" + "\n".join(fatal[:20]))
        sample = check_fcd_sample(outputs["fcd"])
        log(f"[smoke] ok times={sample['times']} diffs={sample['diffs']}")
        log(f"[smoke] vehicle {sample['vehicle_attrs']}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        if tmp_root.exists():
            raise SystemExit(f"failed to delete smoke output: {tmp_root}")
        log(f"[smoke] deleted {tmp_root}")


def verify_config_text(day: Day, cfg_path: Path) -> None:
    cfg_text = cfg_path.read_text(encoding="utf-8")
    required = {
        "net": str(NET_FILE),
        "additional": str(ADDITIONAL_FILE),
        "route": str(day.route_path),
        "begin": '<begin value="0"/>',
        "end": '<end value="86400"/>',
        "teleport": '<time-to-teleport value="600"/>',
        "fcd_period": '<device.fcd.period value="1"/>',
        "seed": f'<seed value="{SEED}"/>',
        "threads": '<threads value="1"/>',
    }
    missing = [name for name, snippet in required.items() if snippet not in cfg_text]
    if missing:
        raise SystemExit(f"{cfg_path} missing {missing}")
    if "device.rerouting" in cfg_text or str(OLD_NET) in cfg_text or str(OLD_ADDITIONAL) in cfg_text:
        raise SystemExit(f"{cfg_path} does not match the 3-second batch inputs")


def finalize_day(day: Day, proc: subprocess.Popen, job: dict) -> None:
    code = proc.returncode
    outputs = output_paths(day, FCD_ROOT / day.tag)
    stats = parse_statistics(outputs["statistics"]) if outputs["statistics"].is_file() else {}
    fatal = fatal_error_lines(outputs["error_log"])
    fcd_ok = outputs["fcd"].is_file() and outputs["fcd"].stat().st_size > 0
    reached_end = str(stats.get("end")) in {"86400.00", "86400"}
    success = code == 0 and fcd_ok and reached_end and not fatal and not stats.get("loaded_rerouting")
    reason = ""
    if not success:
        parts = [f"exit={code}"]
        if not fcd_ok:
            parts.append("fcd_missing")
        if not reached_end:
            parts.append(f"end={stats.get('end')}")
        if fatal:
            parts.append("fatal_error")
        if stats.get("loaded_rerouting"):
            parts.append("rerouting_enabled")
        reason = ",".join(parts)
    result = {
        "status": "SUCCESS" if success else "FAILED",
        "exit_code": code,
        "reason": reason,
        "date": day.iso,
        "pid": job.get("pid"),
        "elapsed_s": job.get("elapsed_s"),
        "start_time": job.get("start_time"),
        "end_time": job.get("end_time"),
        "net": str(NET_FILE),
        "additional": str(ADDITIONAL_FILE),
        "route": str(day.route_path),
        "fcd": str(outputs["fcd"]),
        "fcd_bytes": outputs["fcd"].stat().st_size if outputs["fcd"].is_file() else 0,
        "statistics": stats,
        "fatal_error_lines": fatal[:20],
    }
    write_json(outputs["result"], result)
    job["status"] = result["status"]
    job["reason"] = reason
    job["statistics"] = stats
    job["fcd_bytes"] = result["fcd_bytes"]


def confirm_processes(running: list[tuple[Day, subprocess.Popen]]) -> None:
    deadline = time.time() + 90
    pending = {day.iso: day for day, proc in running if proc.poll() is None}
    while pending and time.time() < deadline:
        still = {}
        for iso, day in pending.items():
            # --log is block-buffered, so the loading lines often appear in
            # console.log first. Either file is enough to confirm the inputs.
            chunks = []
            for name in ("sumo.log", "console.log"):
                log_path = FCD_ROOT / day.tag / name
                if log_path.is_file() and log_path.stat().st_size > 0:
                    chunks.append(log_path.read_text(encoding="utf-8", errors="replace"))
            if not chunks:
                still[iso] = day
                continue
            text = "\n".join(chunks)
            # SUMO 1.19 logs the net and additional files, not the route file.
            if str(NET_FILE) not in text or str(ADDITIONAL_FILE) not in text:
                still[iso] = day
                continue
            if str(OLD_NET) in text or str(OLD_ADDITIONAL) in text:
                raise SystemExit(f"{day.iso} loaded the original TLS inputs")
            cfg_text = config_path_for(day).read_text(encoding="utf-8")
            if str(day.route_path) not in cfg_text:
                raise SystemExit(f"{day.iso} config does not reference {day.route_path}")
            log(f"[check] date={day.iso} loaded 3s net, night_peak_3s, and original route config")
        pending = still
        if pending:
            time.sleep(2)
    if pending:
        log("[check] logs not ready yet for " + ",".join(sorted(pending)))


def launch(binary: str, days: list[Day], workers: int) -> int:
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
        if state == "complete":
            fcd_path = FCD_ROOT / day.tag / f"fcd_{day.tag}.xml.gz"
            log(
                f"[skip] date={day.iso} reason=existing_success "
                f"bytes={fcd_path.stat().st_size} path={fcd_path}"
            )
            continue
        if state == "failed_remnant":
            clean_failed_outputs(day, "failed_or_partial_output_without_success_marker")
        to_run.append(day)

    if not to_run:
        log("[launch] nothing to start")
        write_summary(days)
        return 0

    jobs: dict[str, dict] = {}
    failed: list[str] = []
    status_path = FCD_ROOT / "batch_status.json"

    def flush_status(running_count: int) -> None:
        payload = {
            "updated_at": utc_now(),
            "sumo": binary,
            "workers": workers,
            "seed": SEED,
            "threads_per_process": 1,
            "net": str(NET_FILE),
            "additional": str(ADDITIONAL_FILE),
            "route_dir": str(ROUTE_DIR),
            "fcd_root": str(FCD_ROOT),
            "rerouting": "unset",
            "fcd_compression": "gzip path .xml.gz (SUMO 1.19 has no fcd-output.compression option)",
            "disk_free_bytes": disk_free_bytes(DATA_DISK),
            "route_count": len(days),
            "dates": [day.iso for day in days],
            "decisions": decisions,
            "running_count": running_count,
            "jobs": jobs,
        }
        write_json(status_path, payload)

    def run_queue(queue: list[Day], attempt: str) -> list[str]:
        running: list[tuple[Day, subprocess.Popen]] = []
        pending = list(queue)
        attempt_failed: list[str] = []
        checked = False

        def start_one(day: Day) -> None:
            if sumo_pids_for_day(day):
                log(f"[skip] date={day.iso} reason=sumo_appeared_before_start")
                decisions[day.iso] = "running"
                return
            free = disk_free_bytes(DATA_DISK)
            if free < LOW_DISK_BYTES:
                raise SystemExit(
                    f"/data5 free space fell to {free / 1024**3:.1f} GiB; not starting {day.iso}"
                )
            out_dir = FCD_ROOT / day.tag
            cfg_path = write_sumocfg(day, config_path_for(day), out_dir, begin="0", end="86400")
            verify_config_text(day, cfg_path)
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
                "attempt": attempt,
                "config": str(cfg_path),
                "route": str(day.route_path),
                "net": str(NET_FILE),
                "output_dir": str(out_dir),
                "additional_files": str(ADDITIONAL_FILE),
                "start_time": utc_now(),
                "start_unix": started,
                "status": "RUNNING",
            }
            running.append((day, proc))
            decisions[day.iso] = "started"
            log(f"[launch] attempt={attempt} date={day.iso} pid={proc.pid} config={cfg_path}")

        while pending or running:
            while pending and len(running) < workers:
                start_one(pending.pop(0))
                flush_status(len(running))
            if not checked and not pending and running:
                confirm_processes(running)
                checked = True
                flush_status(len(running))
            if not running:
                break
            time.sleep(5)
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
                finalize_day(day, proc, jobs[day.iso])
                if jobs[day.iso]["status"] == "SUCCESS":
                    log(
                        f"[done] date={day.iso} pid={proc.pid} exit=0 "
                        f"elapsed_s={jobs[day.iso]['elapsed_s']} "
                        f"teleports={jobs[day.iso]['statistics'].get('teleports')} "
                        f"running={jobs[day.iso]['statistics'].get('running')}"
                    )
                else:
                    attempt_failed.append(day.iso)
                    log(
                        f"[fail] date={day.iso} pid={proc.pid} exit={code} "
                        f"reason={jobs[day.iso].get('reason')}"
                    )
            running = still
            flush_status(len(running))
        return attempt_failed

    failed.extend(run_queue(to_run, "1"))
    if failed:
        log("[retry] failed dates " + ",".join(failed))
        retry_days = [day for day in days if day.iso in failed]
        for day in retry_days:
            if sumo_pids_for_day(day):
                log(f"[retry] date={day.iso} still running; not cleaning")
                continue
            clean_failed_outputs(day, "retry_after_failure")
        failed = run_queue(retry_days, "2")

    success = sum(1 for item in jobs.values() if item.get("status") == "SUCCESS")
    log(f"[summary] launched_success={success} failed={len(failed)}")
    if failed:
        log("[summary] FAILED " + ",".join(failed))
    write_summary(days)
    return 1 if failed else 0


def load_baseline(day: Day) -> dict:
    path = OLD_FCD_ROOT / day.tag / f"statistics_{day.tag}.xml.gz"
    stats = parse_statistics(path)
    stats["source"] = str(path) if path.is_file() else ""
    return stats


def anomaly_notes(new_stats: dict, old_stats: dict) -> list[str]:
    notes: list[str] = []
    if not new_stats:
        return ["statistics_missing"]
    if str(new_stats.get("end")) not in {"86400.00", "86400"}:
        notes.append(f"simulation_end={new_stats.get('end')}")
    if new_stats.get("loaded_rerouting"):
        notes.append("rerouting_was_enabled")
    if new_stats.get("loaded_net") is False:
        notes.append("statistics_config_missing_3s_net")

    def as_int(value: Optional[str]) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(float(value))
        except ValueError:
            return None

    new_tp = as_int(new_stats.get("teleports"))
    old_tp = as_int(old_stats.get("teleports"))
    new_running = as_int(new_stats.get("running"))
    old_running = as_int(old_stats.get("running"))
    waiting = as_int(new_stats.get("waiting"))
    if new_tp is not None and (new_tp >= 100 or (old_tp is not None and new_tp >= max(30, old_tp * 5))):
        notes.append(f"teleports={new_tp} baseline={old_tp}")
    if new_running is not None and (
        new_running >= 2000 or (old_running is not None and new_running >= max(1000, old_running * 5))
    ):
        notes.append(f"running_at_end={new_running} baseline={old_running}")
    if waiting is not None and waiting > 0:
        notes.append(f"waiting_at_end={waiting}")
    return notes


def write_summary(days: list[Day]) -> None:
    rows = []
    for day in days:
        result = read_result(day) or {}
        outputs = output_paths(day, FCD_ROOT / day.tag)
        stats = result.get("statistics") or parse_statistics(outputs["statistics"])
        old = load_baseline(day)
        state = classify_day(day)
        notes = anomaly_notes(stats, old) if state == "complete" else [state]
        fcd_bytes = outputs["fcd"].stat().st_size if outputs["fcd"].is_file() else 0
        rows.append(
            {
                "date": day.iso,
                "status": result.get("status", state),
                "exit_code": result.get("exit_code"),
                "elapsed_s": result.get("elapsed_s"),
                "clock_duration": stats.get("clock_duration"),
                "fcd": str(outputs["fcd"]),
                "fcd_bytes": fcd_bytes,
                "loaded": stats.get("loaded"),
                "inserted": stats.get("inserted"),
                "running": stats.get("running"),
                "waiting": stats.get("waiting"),
                "teleports": stats.get("teleports"),
                "teleports_jam": stats.get("teleports_jam"),
                "collisions": stats.get("collisions"),
                "emergency_stops": stats.get("emergency_stops"),
                "emergency_braking": stats.get("emergency_braking"),
                "baseline_teleports": old.get("teleports"),
                "baseline_running": old.get("running"),
                "baseline_inserted": old.get("inserted"),
                "baseline_collisions": old.get("collisions"),
                "notes": notes,
            }
        )
    payload = {
        "updated_at": utc_now(),
        "net": str(NET_FILE),
        "additional": str(ADDITIONAL_FILE),
        "route_dir": str(ROUTE_DIR),
        "fcd_root": str(FCD_ROOT),
        "complete": sum(1 for row in rows if row["status"] == "SUCCESS"),
        "failed": sum(1 for row in rows if row["status"] != "SUCCESS"),
        "days": rows,
    }
    write_json(FCD_ROOT / "batch_summary.json", payload)
    lines = [
        "# Xuancheng26 3-second TLS FCD batch",
        "",
        f"- Updated: {payload['updated_at']}",
        f"- Net: `{NET_FILE}`",
        f"- Additional: `{ADDITIONAL_FILE}`",
        f"- Routes: `{ROUTE_DIR}`",
        f"- Output: `{FCD_ROOT}`",
        f"- Complete: {payload['complete']} / {len(rows)}",
        "",
        "| date | status | elapsed_s | fcd_GiB | inserted | running | waiting | teleports | baseline_tp | collisions | baseline_col | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        gib = row["fcd_bytes"] / 1024**3
        lines.append(
            f"| {row['date']} | {row['status']} | {row['elapsed_s']} | {gib:.2f} | "
            f"{row['inserted']} | {row['running']} | {row['waiting']} | {row['teleports']} | "
            f"{row['baseline_teleports']} | {row['collisions']} | {row['baseline_collisions']} | "
            f"{'; '.join(row['notes'])} |"
        )
    lines.append("")
    (FCD_ROOT / "batch_summary.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"[summary] wrote {FCD_ROOT / 'batch_summary.md'}")


def select_days(days: list[Day], requested: Optional[str]) -> list[Day]:
    if not requested:
        return days
    wanted = {item.strip() for item in requested.split(",") if item.strip()}
    selected = [day for day in days if day.iso in wanted or day.tag in wanted]
    missing = wanted - {day.iso for day in selected} - {day.tag for day in selected}
    if missing:
        raise SystemExit("unknown dates: " + ",".join(sorted(missing)))
    if not selected:
        raise SystemExit("no dates selected")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Xuancheng26 3-second TLS FCD simulations")
    parser.add_argument("--workers", type=int, default=29)
    parser.add_argument("--dates", default="", help="Comma-separated ISO dates to run; default is all 29")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    binary = sumo_binary()
    version = sumo_version(binary)
    log(f"sumo_binary={binary}")
    log(f"sumo_version={version}")
    days = discover_routes(ROUTE_DIR)
    require_inputs(days)
    selected = select_days(days, args.dates)
    if args.summarize_only:
        write_summary(days)
        return 0
    require_disk_space()
    log("dates=" + ",".join(day.iso for day in selected))
    log(f"route_count={len(selected)} net={NET_FILE} additional={ADDITIONAL_FILE}")
    if args.smoke_test:
        smoke_test(binary, selected)
        log("[smoke] passed")
        return 0
    return launch(binary, selected, args.workers)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log(f"[fatal] {exc}")
        raise
