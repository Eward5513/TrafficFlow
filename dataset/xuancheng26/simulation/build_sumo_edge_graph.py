#!/usr/bin/env python3
"""Build the ordinary-edge SUMO transfer graph and pair-bridge cache.

Writes only under /data5/tzhang174/TrafficFlow/xuancheng26. The original
Xuancheng26 download tree is read-only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from repair_lib import (
    BridgeCache,
    CODE_ROOT,
    DATA_ROOT,
    SOURCE_ROOT,
    config_hash,
    data_paths,
    ensure_data_dirs,
    load_config,
    parse_sumo_net,
    peak_rss_mb,
    save_pickle_atomic,
    sha256_file,
    write_json_atomic,
)


def copy_base_net(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    src_hash = sha256_file(src)
    dst_hash = sha256_file(dst)
    if src_hash != dst_hash:
        raise RuntimeError(f"copied net hash mismatch: {src_hash} vs {dst_hash}")
    return src_hash


def write_topology(graph, paths: dict, tmp_dir: Path) -> dict:
    topo = paths["road_topology"]
    stats = {
        "net_sha256": graph.net_sha256,
        "n_normal_edges": len(graph.normal_ids),
        "n_internal_edges": len(graph.internal_ids),
        "n_conn_raw": graph.n_conn_raw,
        "n_edge_level_pairs": graph.n_conn_edge_pairs,
        "expected_normal_edges": 1744,
    }
    write_json_atomic(stats, topo / "graph_stats.json", tmp_dir)
    write_json_atomic(
        {
            "edge_ids": graph.normal_ids,
            "length_of": graph.length_of,
            "from": graph.edge_from,
            "to": graph.edge_to,
            "nlanes": graph.nlanes_of,
        },
        topo / "normal_edges.json",
        tmp_dir,
    )
    write_json_atomic(graph.adj, topo / "adjacency.json", tmp_dir)
    write_json_atomic(
        [{"from": a, "to": b, "dir": d} for (a, b), d in sorted(graph.conn_dir.items())],
        topo / "edge_connections.json",
        tmp_dir,
    )
    lane_path = topo / "lane_connections.jsonl"
    with lane_path.open("w", encoding="utf-8") as fh:
        for conn in graph.lane_conns:
            fh.write(
                json.dumps(
                    {
                        "from": conn.frm,
                        "to": conn.to,
                        "fromLane": conn.from_lane,
                        "toLane": conn.to_lane,
                        "dir": conn.direction,
                        "via": conn.via,
                        "tl": conn.tl,
                        "linkIndex": conn.link_index,
                        "state": conn.state,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    save_pickle_atomic(graph, topo / "sumo_graph.pkl", tmp_dir)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Xuancheng26 SUMO edge graph")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    paths = ensure_data_dirs(cfg)
    tmp_dir = paths["tmp"]

    src_net = Path(cfg["sumo_net"])
    if not src_net.is_file():
        raise FileNotFoundError(src_net)
    src_hash = sha256_file(src_net)
    dst_net = paths["base_net"]
    copied_hash = copy_base_net(src_net, dst_net)
    print(f"[graph] copied base net sha256={copied_hash}", flush=True)
    if copied_hash != src_hash:
        raise RuntimeError("base net copy hash mismatch")

    graph = parse_sumo_net(dst_net)
    expected = int(cfg.get("expected_normal_edges", 1744))
    print(
        f"[graph] normal_edges={len(graph.normal_ids)} internal={len(graph.internal_ids)} "
        f"edge_pairs={graph.n_conn_edge_pairs} raw_conn={graph.n_conn_raw}",
        flush=True,
    )
    if len(graph.normal_ids) != expected:
        raise RuntimeError(
            f"expected {expected} ordinary edges, found {len(graph.normal_ids)}"
        )
    stats = write_topology(graph, paths, tmp_dir)

    cache = BridgeCache(graph, max_inserted_edges=int(cfg["max_inserted_edges"]))
    n_ok = cache.precompute_all()
    cache_path = paths["road_cache"] / "bridge_cache.pkl"
    save_pickle_atomic(
        {
            "meta": cache.cache_key_meta(),
            "store": cache._store,
            "precomputed_sources": cache._precomputed_sources,
        },
        cache_path,
        tmp_dir,
    )
    write_json_atomic(
        {
            "meta": cache.cache_key_meta(),
            "n_cached_pairs": len(cache._store),
            "n_reachable_pairs": n_ok,
            "config_sha256": config_hash(Path(cfg["_config_path"])),
            "base_net_sha256": copied_hash,
            "source_net_sha256": src_hash,
        },
        paths["road_cache"] / "bridge_cache_meta.json",
        tmp_dir,
    )
    print(
        f"[graph] cached reachable pairs={n_ok} store={len(cache._store)} "
        f"rss_mb={peak_rss_mb():.1f}",
        flush=True,
    )
    print(f"[graph] SOURCE_ROOT={SOURCE_ROOT} (read-only)", flush=True)
    print(f"[graph] DATA_ROOT={DATA_ROOT}", flush=True)
    print(f"[graph] CODE_ROOT={CODE_ROOT}", flush=True)
    print("[graph] OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
