#!/usr/bin/env python3
"""Orchestrate Xuancheng26 trip repair: graph, tests, dry-run, full run, validate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from repair_lib import (
    CODE_ROOT,
    SOURCE_ROOT,
    code_file_hash,
    config_hash,
    data_paths,
    ensure_data_dirs,
    load_config,
    peak_rss_mb,
    scan_code_files,
    scan_large_data_in_code,
    sha256_file,
    source_trip_files,
    unique_active_dates,
    write_json_atomic,
)


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Xuancheng26 trip repair pipeline")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dates", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    paths = ensure_data_dirs(cfg)
    t0 = time.time()

    py = sys.executable
    code = CODE_ROOT
    cfg_path = Path(cfg["_config_path"])

    if args.validate_only:
        val_only = [py, "validate_repaired_trips.py", "--config", str(cfg_path)]
        if args.workers is not None:
            val_only.extend(["--workers", str(args.workers)])
        run(val_only, code)
        run([py, "summarize_repairs.py", "--config", str(cfg_path)], code)
        return 0

    # Stage 1 already created dirs; refresh inventory pointer.
    print("[pipeline] stage1 read-only check", flush=True)
    print(f"[pipeline] SOURCE_ROOT={SOURCE_ROOT}", flush=True)
    print(f"[pipeline] trip files={len(source_trip_files(SOURCE_ROOT))}", flush=True)
    print(f"[pipeline] unique dates={unique_active_dates(cfg)}", flush=True)
    print(f"[pipeline] code_sha256={code_file_hash(CODE_ROOT)}", flush=True)
    print(f"[pipeline] config_sha256={config_hash(cfg_path)}", flush=True)

    print("[pipeline] stage2 graph", flush=True)
    graph_cmd = [py, "build_sumo_edge_graph.py", "--config", str(cfg_path)]
    if args.force:
        graph_cmd.append("--force")
    run(graph_cmd, code)

    if not args.skip_tests:
        print("[pipeline] stage3 unit tests", flush=True)
        run([py, "-m", "pytest", "-q", "tests/test_trip_repair.py"], code)

    repair_cmd = [py, "repair_trips.py", "--config", str(cfg_path)]
    if args.dry_run:
        repair_cmd.extend(["--dry-run", "--dates", args.dates or "2023-04-05"])
    elif args.dates:
        repair_cmd.extend(["--dates", args.dates])
    if args.workers is not None:
        repair_cmd.extend(["--workers", str(args.workers)])
    if args.force:
        repair_cmd.append("--force")
    if args.resume:
        repair_cmd.append("--resume")
    else:
        repair_cmd.append("--no-resume")
    print("[pipeline] stage4/5 repair", flush=True)
    run(repair_cmd, code)

    val_cmd = [py, "validate_repaired_trips.py", "--config", str(cfg_path)]
    if args.dry_run:
        val_cmd.extend(["--dates", args.dates or "2023-04-05"])
    elif args.dates:
        val_cmd.extend(["--dates", args.dates])
    if args.workers is not None:
        val_cmd.extend(["--workers", str(args.workers)])
    print("[pipeline] stage6 validate", flush=True)
    run(val_cmd, code)

    elapsed = time.time() - t0
    rss = peak_rss_mb()
    val_path = paths["reports"] / "validation_summary.json"
    ready = "NO"
    if val_path.is_file():
        summary = json.loads(val_path.read_text(encoding="utf-8"))
        if summary.get("ok") and (args.dry_run or len(unique_active_dates(cfg)) == 29):
            if not args.dry_run:
                ready = "YES"
            else:
                ready = "NO"
    run(
        [
            py,
            "summarize_repairs.py",
            "--config",
            str(cfg_path),
            "--wall-clock-s",
            f"{elapsed:.1f}",
            "--peak-rss-mb",
            f"{rss:.1f}",
            "--ready",
            ready,
        ],
        code,
    )
    write_json_atomic(
        {
            "elapsed_s": elapsed,
            "peak_rss_mb": rss,
            "code_files_in_data": scan_code_files(paths["data_root"]),
            "large_data_in_code": scan_large_data_in_code(CODE_ROOT),
        },
        paths["reports"] / "pipeline_runtime.json",
        paths["tmp"],
    )
    print(f"[pipeline] elapsed_s={elapsed:.1f} rss_mb={rss:.1f} ready={ready}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
