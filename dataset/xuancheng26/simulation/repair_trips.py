#!/usr/bin/env python3
"""Repair Xuancheng26 CityFlow trips against the SUMO ordinary-edge graph."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from repair_lib import (
    BridgeCache,
    CODE_ROOT,
    code_file_hash,
    config_hash,
    copy_record_with_route,
    data_paths,
    date_from_trip_name,
    depart_time_invalid,
    ensure_data_dirs,
    iter_json_array,
    load_config,
    load_pickle,
    parse_sumo_net,
    peak_rss_mb,
    repair_route,
    result_to_metadata,
    sha256_file,
    trip_id_for,
    trip_path_for_date,
    unique_active_dates,
    write_json_atomic,
    AtomicGzipJsonlWriter,
    AtomicJsonArrayWriter,
    SumoGraph,
)

LOG = logging.getLogger("repair_trips")


def setup_logging(report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = report_dir / "repair_trips.log"
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    class FlushHandler(logging.StreamHandler):
        def emit(self, record: logging.LogRecord) -> None:
            super().emit(record)
            self.flush()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = FlushHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOG.addHandler(fh)
    LOG.addHandler(sh)
    LOG.propagate = False


def load_graph_and_cache(cfg: dict, paths: dict) -> tuple[SumoGraph, BridgeCache]:
    topo_pkl = paths["road_topology"] / "sumo_graph.pkl"
    cache_pkl = paths["road_cache"] / "bridge_cache.pkl"
    if topo_pkl.is_file():
        graph = load_pickle(topo_pkl)
    else:
        graph = parse_sumo_net(paths["base_net"])
    cache = BridgeCache(graph, max_inserted_edges=int(cfg["max_inserted_edges"]))
    if cache_pkl.is_file():
        payload = load_pickle(cache_pkl)
        meta = payload.get("meta") or {}
        if meta.get("net_sha256") != graph.net_sha256:
            raise RuntimeError("bridge cache net hash does not match loaded graph")
        cache._store = payload["store"]
        cache._precomputed_sources = set(payload.get("precomputed_sources") or [])
    return graph, cache


def manifest_path(paths: dict, date: str) -> Path:
    return paths["manifests"] / f"manifest_{date.replace('-', '_')}.json"


def output_paths(paths: dict, date: str) -> dict:
    tag = date.replace("-", "_")
    return {
        "repaired": paths["repaired"] / f"data_{tag}_repaired.json",
        "rejected": paths["rejected"] / f"data_{tag}_rejected.jsonl.gz",
        "metadata": paths["repair_metadata"] / f"data_{tag}_metadata.jsonl.gz",
        "manifest": manifest_path(paths, date),
    }


def load_manifest(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def can_skip(manifest: dict, expected: dict) -> bool:
    if not manifest or manifest.get("status") != "complete":
        return False
    for key in (
        "source_sha256",
        "code_sha256",
        "config_sha256",
        "net_sha256",
        "max_inserted_edges",
    ):
        if manifest.get(key) != expected.get(key):
            return False
    for file_key in ("repaired", "rejected", "metadata"):
        rec = (manifest.get("outputs") or {}).get(file_key) or {}
        path = Path(rec.get("path") or "")
        if not path.is_file():
            return False
        if sha256_file(path) != rec.get("sha256"):
            return False
    return True


def process_date(
    date: str,
    cfg: dict,
    paths: dict,
    graph: SumoGraph,
    cache: BridgeCache,
    *,
    force: bool = False,
    resume: bool = True,
) -> dict:
    outs = output_paths(paths, date)
    source = trip_path_for_date(cfg, date)
    if not source.is_file():
        raise FileNotFoundError(source)
    source_sha = sha256_file(source)
    expected = {
        "source_sha256": source_sha,
        "code_sha256": code_file_hash(CODE_ROOT),
        "config_sha256": config_hash(Path(cfg["_config_path"])),
        "net_sha256": graph.net_sha256,
        "max_inserted_edges": int(cfg["max_inserted_edges"]),
    }
    existing = load_manifest(outs["manifest"])
    if resume and (not force) and can_skip(existing, expected):
        LOG.info("%s skip resume-complete records=%s", date, existing.get("n_source"))
        return existing

    for key in ("repaired", "rejected", "metadata"):
        path = outs[key]
        if path.exists():
            path.unlink()

    t0 = time.time()
    repaired_w = AtomicJsonArrayWriter(outs["repaired"], paths["tmp"])
    rejected_w = AtomicGzipJsonlWriter(outs["rejected"], paths["tmp"])
    meta_w = AtomicGzipJsonlWriter(outs["metadata"], paths["tmp"])
    n_source = 0
    n_repaired = 0
    n_rejected = 0
    n_invalid_time = 0
    n_insertion_only = 0
    n_deletion_insertion = 0
    n_direct = 0
    n_deletion_only = 0
    try:
        for rec in iter_json_array(source):
            idx = n_source
            n_source += 1
            trip_id = trip_id_for(date, idx)
            route = rec.get("route") or []
            if not isinstance(route, list):
                route = []
            result = repair_route(
                route,
                graph,
                cache,
                min_route_edges=int(cfg.get("min_route_edges", 2)),
            )
            invalid_time = depart_time_invalid(
                rec, limit=int(cfg.get("depart_time_limit", 86400))
            )
            status = "repaired"
            reject_reason = None
            if invalid_time:
                status = "rejected"
                reject_reason = "invalid_depart_time"
                n_invalid_time += 1
            elif result.status != "repaired" or not result.repaired_route:
                status = "rejected"
                reject_reason = result.reject_reason or "no_repairable_subsequence"

            meta = result_to_metadata(
                trip_id=trip_id,
                date=date,
                source_file=str(source),
                source_record_index=idx,
                result=result,
                reject_reason=reject_reason,
                status=status,
            )
            meta_w.write(meta)

            if status == "repaired":
                repaired_w.write(copy_record_with_route(rec, result.repaired_route or []))
                n_repaired += 1
                if result.action == "insertion_only":
                    n_insertion_only += 1
                elif result.action == "deletion_insertion":
                    n_deletion_insertion += 1
                elif result.action == "direct_legal":
                    n_direct += 1
                elif result.action == "deletion_only":
                    n_deletion_only += 1
            else:
                rejected_w.write(
                    {
                        "trip_id": trip_id,
                        "date": date,
                        "source_file": str(source),
                        "source_record_index": idx,
                        "reject_reason": reject_reason,
                        "original_record": rec,
                        "original_route": list(route),
                        "repaired_route_attempt": result.repaired_route,
                        "kept_indices": result.kept_indices,
                        "unsolved_pair": result.unsolved_pair,
                        "algorithm_status": result.algorithm_status,
                        "invalid_depart_time": invalid_time,
                    }
                )
                n_rejected += 1
        repaired_w.commit()
        rejected_w.commit()
        meta_w.commit()
    except Exception:
        repaired_w.abort()
        rejected_w.abort()
        meta_w.abort()
        raise

    elapsed = time.time() - t0
    manifest = {
        "date": date,
        "status": "complete",
        "source_file": str(source),
        "source_sha256": source_sha,
        "code_sha256": expected["code_sha256"],
        "config_sha256": expected["config_sha256"],
        "net_sha256": expected["net_sha256"],
        "max_inserted_edges": expected["max_inserted_edges"],
        "n_source": n_source,
        "n_repaired": n_repaired,
        "n_rejected": n_rejected,
        "n_invalid_depart_time": n_invalid_time,
        "n_direct_legal": n_direct,
        "n_insertion_only": n_insertion_only,
        "n_deletion_only": n_deletion_only,
        "n_deletion_insertion": n_deletion_insertion,
        "elapsed_s": round(elapsed, 3),
        "peak_rss_mb": round(peak_rss_mb(), 2),
        "conservation_ok": n_source == n_repaired + n_rejected,
        "outputs": {
            "repaired": {
                "path": str(outs["repaired"]),
                "bytes": outs["repaired"].stat().st_size,
                "sha256": sha256_file(outs["repaired"]),
            },
            "rejected": {
                "path": str(outs["rejected"]),
                "bytes": outs["rejected"].stat().st_size,
                "sha256": sha256_file(outs["rejected"]),
            },
            "metadata": {
                "path": str(outs["metadata"]),
                "bytes": outs["metadata"].stat().st_size,
                "sha256": sha256_file(outs["metadata"]),
            },
        },
    }
    write_json_atomic(manifest, outs["manifest"], paths["tmp"])
    LOG.info(
        "%s input=%d repaired=%d rejected=%d insertion_only=%d deletion+insertion=%d "
        "direct=%d deletion_only=%d invalid_time=%d elapsed=%.1fs rss_mb=%.1f",
        date,
        n_source,
        n_repaired,
        n_rejected,
        n_insertion_only,
        n_deletion_insertion,
        n_direct,
        n_deletion_only,
        n_invalid_time,
        elapsed,
        peak_rss_mb(),
    )
    return manifest


def _worker(payload: dict) -> dict:
    cfg = payload["cfg"]
    date = payload["date"]
    force = payload["force"]
    resume = payload["resume"]
    paths = ensure_data_dirs(cfg)
    graph, cache = load_graph_and_cache(cfg, paths)
    setup_logging(paths["reports"])
    return process_date(date, cfg, paths, graph, cache, force=force, resume=resume)


def parse_dates(raw: Optional[str], cfg: dict) -> list[str]:
    active = unique_active_dates(cfg)
    if not raw:
        return active
    wanted = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "_" in item and item.count("_") == 2:
            y, m, d = item.split("_")
            item = f"{y}-{m}-{d}"
        wanted.append(item)
    exclude = set((cfg.get("exclude_dates") or {}).keys())
    out = []
    for date in wanted:
        if date in exclude:
            continue
        if date not in active:
            raise ValueError(f"date not in active unique dates: {date}")
        out.append(date)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repair Xuancheng26 trips")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dates", type=str, default=None, help="comma-separated YYYY-MM-DD")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    paths = ensure_data_dirs(cfg)
    setup_logging(paths["reports"])

    dates = parse_dates(args.dates, cfg)
    if args.dry_run and not args.dates:
        dates = ["2023-04-05"]
        LOG.info("dry-run default date 2023-04-05")
    max_workers = int(cfg.get("max_workers", 8))
    workers = args.workers
    if workers is None:
        workers = int(cfg.get("default_workers", 4))
    workers = max(1, min(workers, max_workers, len(dates) or 1))

    LOG.info("dates=%s workers=%s force=%s resume=%s", dates, workers, args.force, args.resume)
    graph, cache = load_graph_and_cache(cfg, paths)
    LOG.info(
        "graph normal_edges=%d cached_pairs=%d precomputed_sources=%d",
        len(graph.normal_ids),
        len(cache._store),
        len(cache._precomputed_sources),
    )

    if workers == 1 or len(dates) <= 1:
        for date in dates:
            process_date(date, cfg, paths, graph, cache, force=args.force, resume=args.resume)
        return 0

    payloads = [
        {"cfg": cfg, "date": date, "force": args.force, "resume": args.resume}
        for date in dates
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_worker, payload): payload["date"] for payload in payloads}
        for fut in as_completed(futs):
            date = futs[fut]
            try:
                fut.result()
            except Exception:
                LOG.exception("date %s failed", date)
                raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
