#!/usr/bin/env python3
"""Unit tests for Xuancheng26 trip repair. Synthetic graphs only."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from repair_lib import (
    AtomicJsonArrayWriter,
    BridgeCache,
    graph_from_synthetic,
    iter_json_array,
    parse_sumo_net,
    repair_route,
)
from repair_trips import can_skip


def run(route, graph, min_route_edges=2):
    cache = BridgeCache(graph, max_inserted_edges=3)
    return repair_route(route, graph, cache, min_route_edges=min_route_edges)


def test_k0_direct_legal():
    g = graph_from_synthetic(["A", "B"], [("A", "B")])
    r = run(["A", "B"], g)
    assert r.status == "repaired"
    assert r.repaired_route == ["A", "B"]
    assert r.inserted_edge_count == 0
    assert r.deleted_observed_edge_count == 0
    assert r.action == "direct_legal"
    assert r.max_inserted_single == 0


def test_insert_one_edge():
    g = graph_from_synthetic(["A", "X", "B"], [("A", "X"), ("X", "B")], lengths={"X": 4.0})
    r = run(["A", "B"], g)
    assert r.status == "repaired"
    assert r.repaired_route == ["A", "X", "B"]
    assert r.inserted_ids == ["X"]
    assert r.deleted_observed_edge_count == 0
    assert r.action == "insertion_only"
    assert r.max_inserted_single == 1


def test_insert_two_edges():
    g = graph_from_synthetic(
        ["A", "X", "Y", "B"],
        [("A", "X"), ("X", "Y"), ("Y", "B")],
    )
    r = run(["A", "B"], g)
    assert r.repaired_route == ["A", "X", "Y", "B"]
    assert r.inserted_ids == ["X", "Y"]
    assert r.max_inserted_single == 2


def test_insert_three_edges():
    g = graph_from_synthetic(
        ["A", "X", "Y", "Z", "B"],
        [("A", "X"), ("X", "Y"), ("Y", "Z"), ("Z", "B")],
    )
    r = run(["A", "B"], g)
    assert r.repaired_route == ["A", "X", "Y", "Z", "B"]
    assert r.inserted_ids == ["X", "Y", "Z"]
    assert r.max_inserted_single == 3


def test_cannot_bridge_four_inserts():
    g = graph_from_synthetic(
        ["A", "X1", "X2", "X3", "X4", "B"],
        [("A", "X1"), ("X1", "X2"), ("X2", "X3"), ("X3", "X4"), ("X4", "B")],
    )
    r = run(["A", "B"], g)
    assert r.status == "rejected"
    assert r.reject_reason in {"no_repairable_subsequence", "fewer_than_two_edges_after_repair"}
    cache = BridgeCache(g, max_inserted_edges=3)
    br = cache.get("A", "B")
    assert not br.ok


def test_delete_right_anomaly_then_recover():
    # A-B cannot, B-C cannot, A-C direct. Delete B (right of the A-B jump).
    g = graph_from_synthetic(["A", "B", "C"], [("A", "C")])
    r = run(["A", "B", "C"], g)
    assert r.status == "repaired"
    assert r.repaired_route == ["A", "C"]
    assert r.deleted_ids == ["B"]
    assert r.deleted_indices == [1]


def test_delete_left_anomaly_then_recover():
    # A-B cannot, A-C cannot, B-C direct. Delete A (left of the jump).
    g = graph_from_synthetic(["A", "B", "C"], [("B", "C")])
    r = run(["A", "B", "C"], g)
    assert r.status == "repaired"
    assert r.repaired_route == ["B", "C"]
    assert r.deleted_ids == ["A"]
    assert r.deleted_indices == [0]
    assert r.endpoint_deleted is True


def test_two_consecutive_anomalies():
    g = graph_from_synthetic(["A", "X", "Y", "C"], [("A", "C")])
    r = run(["A", "X", "Y", "C"], g)
    assert r.status == "repaired"
    assert r.repaired_route == ["A", "C"]
    assert r.deleted_ids == ["X", "Y"]


def test_unreachable_pair_delete_anomaly():
    g = graph_from_synthetic(
        ["A", "B", "C", "Z"],
        [("A", "C")],
    )
    r = run(["A", "B", "C"], g)
    assert r.status == "repaired"
    assert r.repaired_route == ["A", "C"]
    assert "B" in r.deleted_ids


def test_same_hop_prefers_shorter_length():
    g = graph_from_synthetic(
        ["A", "P", "Q", "B"],
        [("A", "P"), ("P", "B"), ("A", "Q"), ("Q", "B")],
        lengths={"P": 10.0, "Q": 1.0, "A": 1.0, "B": 1.0},
    )
    r = run(["A", "B"], g)
    assert r.repaired_route == ["A", "Q", "B"]
    assert r.inserted_ids == ["Q"]


def test_same_length_is_deterministic_lex():
    g = graph_from_synthetic(
        ["A", "M", "N", "B"],
        [("A", "M"), ("M", "B"), ("A", "N"), ("N", "B")],
        lengths={"M": 5.0, "N": 5.0},
    )
    r1 = run(["A", "B"], g)
    r2 = run(["A", "B"], g)
    assert r1.repaired_route == r2.repaired_route
    assert r1.repaired_route == ["A", "M", "B"]


def test_first_edge_anomaly():
    g = graph_from_synthetic(["X", "A", "B"], [("A", "B")])
    r = run(["X", "A", "B"], g)
    assert r.repaired_route == ["A", "B"]
    assert r.deleted_ids == ["X"]
    assert r.endpoint_deleted is True
    assert "endpoint_deleted" in r.quality_flags


def test_last_edge_anomaly():
    g = graph_from_synthetic(["A", "B", "X"], [("A", "B")])
    r = run(["A", "B", "X"], g)
    assert r.repaired_route == ["A", "B"]
    assert r.deleted_ids == ["X"]
    assert r.endpoint_deleted is True


def test_fewer_than_two_after_delete():
    g = graph_from_synthetic(["A", "X"], [])
    r = run(["A", "X"], g)
    assert r.status == "rejected"
    assert r.reject_reason in {"no_repairable_subsequence", "fewer_than_two_edges_after_repair"}


def test_unknown_edge_rejected():
    g = graph_from_synthetic(["A", "B"], [("A", "B")])
    r = run(["A", "UNK", "B"], g)
    assert r.status == "rejected"
    assert r.reject_reason == "unknown_edge_id"


def test_internal_edges_not_written(tmp_path: Path):
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<net>
  <edge id="A" from="1" to="2">
    <lane id="A_0" index="0" length="10" speed="13"/>
  </edge>
  <edge id=":J_0" function="internal">
    <lane id=":J_0_0" index="0" length="3" speed="8"/>
  </edge>
  <edge id="B" from="2" to="3">
    <lane id="B_0" index="0" length="20" speed="13"/>
  </edge>
  <connection from="A" to="B" fromLane="0" toLane="0" via=":J_0_0" dir="s"/>
  <connection from=":J_0" to="B" fromLane="0" toLane="0" dir="s"/>
</net>
"""
    net_path = tmp_path / "tiny.net.xml"
    net_path.write_text(xml, encoding="utf-8")
    graph = parse_sumo_net(net_path)
    assert "A" in graph.normal_set
    assert "B" in graph.normal_set
    assert ":J_0" not in graph.normal_set
    assert graph.has_connection("A", "B")
    r = run(["A", "B"], graph)
    assert r.repaired_route == ["A", "B"]
    assert all(not e.startswith(":") for e in r.repaired_route)


def test_introduced_cycle_flag():
    g = graph_from_synthetic(
        ["A", "C", "D"],
        [("A", "D"), ("D", "C"), ("C", "D")],
    )
    r = run(["A", "C", "D"], g)
    assert r.status == "repaired"
    assert r.repaired_route == ["A", "D", "C", "D"]
    assert r.introduced_cycle is True
    assert "introduced_cycle" in r.quality_flags


def test_repeat_run_identical():
    g = graph_from_synthetic(
        ["A", "X", "Y", "B", "C"],
        [("A", "X"), ("X", "B"), ("B", "C"), ("A", "Y"), ("Y", "B")],
        lengths={"X": 2.0, "Y": 2.0},
    )
    route = ["A", "Zfake"] if False else ["A", "B", "C"]
    r1 = run(route, g)
    r2 = run(route, g)
    assert r1.repaired_route == r2.repaired_route
    assert r1.deleted_indices == r2.deleted_indices
    assert r1.inserted_ids == r2.inserted_ids


def test_resume_does_not_redo_complete(tmp_path: Path):
    final = tmp_path / "out.json"
    tmp_dir = tmp_path / "tmp"
    writer = AtomicJsonArrayWriter(final, tmp_dir)
    writer.write({"route": ["A", "B"]})
    writer.commit()
    digest = __import__("hashlib").sha256(final.read_bytes()).hexdigest()
    manifest = {
        "status": "complete",
        "source_sha256": "s",
        "code_sha256": "c",
        "config_sha256": "k",
        "net_sha256": "n",
        "max_inserted_edges": 3,
        "outputs": {
            "repaired": {"path": str(final), "sha256": digest},
            "rejected": {"path": str(final), "sha256": digest},
            "metadata": {"path": str(final), "sha256": digest},
        },
    }
    expected = {
        "source_sha256": "s",
        "code_sha256": "c",
        "config_sha256": "k",
        "net_sha256": "n",
        "max_inserted_edges": 3,
    }
    assert can_skip(manifest, expected) is True
    expected2 = dict(expected)
    expected2["code_sha256"] = "changed"
    assert can_skip(manifest, expected2) is False


def test_partial_tmp_not_treated_as_complete(tmp_path: Path):
    final = tmp_path / "official.json"
    tmp_dir = tmp_path / "tmp"
    writer = AtomicJsonArrayWriter(final, tmp_dir)
    writer.write({"route": ["A", "B"]})
    writer.abort()
    assert not final.exists()
    leftovers = list(tmp_dir.glob("*.partial"))
    assert leftovers == []
    manifest = {"status": "complete"}
    assert can_skip(manifest, {}) is False


def test_json_array_roundtrip(tmp_path: Path):
    final = tmp_path / "arr.json"
    tmp_dir = tmp_path / "tmp"
    writer = AtomicJsonArrayWriter(final, tmp_dir)
    writer.write({"startTime": 0, "route": ["A", "B"]})
    writer.write({"startTime": 10, "route": ["C", "D"]})
    writer.commit()
    recs = list(iter_json_array(final))
    assert len(recs) == 2
    assert recs[0]["startTime"] == 0
    assert recs[1]["route"] == ["C", "D"]


def test_empty_route_rejected():
    g = graph_from_synthetic(["A", "B"], [("A", "B")])
    r = run([], g)
    assert r.status == "rejected"
    assert r.reject_reason == "empty_route"


def test_quality_flags_high_delete_ratio():
    edges = ["S", "T"] + [f"Z{i}" for i in range(8)]
    g = graph_from_synthetic(edges, [("S", "T")])
    route = ["S"] + [f"Z{i}" for i in range(8)] + ["T"]
    r = run(route, g)
    assert r.status == "repaired"
    assert r.repaired_route == ["S", "T"]
    assert "deleted_ratio_gt_0.5" in r.quality_flags
    assert "deleted_ratio_gt_0.2" in r.quality_flags
    assert r.status != "rejected"
