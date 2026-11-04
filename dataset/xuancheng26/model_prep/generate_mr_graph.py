#!/usr/bin/env python3
"""Build the directed R-M heterogeneous (MR) graph for Xuancheng26.

Reuses ``graph/r_graph.npz`` and ``graph/m_graph.npz``; the net and FCD are not
read. Global nodes are every R node in ``r_graph`` order (0..R-1), then every M
node in ``m_graph`` order (R..R+M-1). Four directed relations keep their type:

    0 R->R  r_graph adjacency, row-major
    1 M->M  m_graph edge_index, same order (aligns with its transition arrays)
    2 R->M  incoming road of a Movement -> the Movement
    3 M->R  Movement -> its outgoing road

A node's local index is its column in the R-level or M-level flow CSVs. No
edge is symmetrized and nothing is added beyond these four relations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

import generate_m_graph as mgraph
import generate_r_graph as rgraph
import prep_paths

NODE_TYPE_NAMES = ("R", "M")
NODE_R, NODE_M = 0, 1
RELATION_NAMES = ("R->R", "M->M", "R->M", "M->R")
RELATION_ENDPOINTS = ((NODE_R, NODE_R), (NODE_M, NODE_M), (NODE_R, NODE_M), (NODE_M, NODE_R))
RELATION_KEYS = ("r_r", "m_m", "r_m", "m_r")
REL_RR, REL_MM, REL_RM, REL_MR = 0, 1, 2, 3
OUTPUT_NAMES = ("mr_graph.npz", "mr_nodes.csv", "mr_edges.csv", "mr_graph_audit.json", "mr_graph_audit.md")
FLOW_FILES = ("inflow.csv", "outflow.csv", "accumulation_start.csv", "accumulation_end.csv")
DAY_PATTERN = re.compile(r"\d{4}_\d{2}_\d{2}")
RATE_PATTERN = re.compile(r"p\d{2}")
EXAMPLES = 8

log = mgraph.log


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Inputs:
    r_ids: list[str]
    r_adj: np.ndarray
    m_ids: list[str]
    m_in: list[str]
    m_out: list[str]
    m_junction: list[str]
    m_turn: list[str]
    m_in_r_index: np.ndarray
    m_out_r_index: np.ndarray
    m_adj: np.ndarray
    m_edge_index: np.ndarray
    m_transition_edges: list[str]
    m_signature: str
    hashes: dict[str, str]


def strings(values: np.ndarray) -> list[str]:
    if values.dtype.kind != "U":
        raise RuntimeError(f"expected a string array, got {values.dtype}")
    return [str(value) for value in values.tolist()]


def load_inputs(graph_dir: Path) -> Inputs:
    r_path, m_path = graph_dir / "r_graph.npz", graph_dir / "m_graph.npz"
    for path in (r_path, m_path, graph_dir / "m_nodes.csv", graph_dir / "m_transitions.csv"):
        if not path.is_file():
            raise FileNotFoundError(f"missing input {path}")
    hashes = {name: sha256_file(graph_dir / name) for name in ("r_graph.npz", "m_graph.npz", "m_nodes.csv", "m_transitions.csv")}
    with np.load(r_path, allow_pickle=False) as payload:
        r_ids = strings(payload["edge_ids"])
        r_adj = payload["adjacency"].copy()
        r_edges = int(payload["num_graph_edges"])
    with np.load(m_path, allow_pickle=False) as payload:
        inputs = Inputs(
            r_ids=r_ids,
            r_adj=r_adj,
            m_ids=strings(payload["movement_ids"]),
            m_in=strings(payload["incoming_edge_ids"]),
            m_out=strings(payload["outgoing_edge_ids"]),
            m_junction=strings(payload["junction_ids"]),
            m_turn=strings(payload["turn_directions"]),
            m_in_r_index=payload["incoming_r_index"].astype(np.int64),
            m_out_r_index=payload["outgoing_r_index"].astype(np.int64),
            m_adj=payload["adjacency"].copy(),
            m_edge_index=payload["edge_index"].astype(np.int64),
            m_transition_edges=strings(payload["transition_edge_ids"]),
            m_signature=str(payload["graph_signature"]),
            hashes=hashes,
        )
        m_r_edge_ids = strings(payload["r_edge_ids"])
        m_edges = int(payload["num_graph_edges"])
    check_inputs(inputs, r_edges, m_edges, m_r_edge_ids, graph_dir)
    return inputs


def check_inputs(inp: Inputs, r_edges: int, m_edges: int, m_r_edge_ids: list[str], graph_dir: Path) -> None:
    """The R and M graphs were audited before; fail fast if they drifted apart."""
    num_r, num_m = len(inp.r_ids), len(inp.m_ids)
    if len(set(inp.r_ids)) != num_r or any(e.startswith(":") for e in inp.r_ids):
        raise RuntimeError("r_graph edge_ids are not unique real edges")
    if inp.r_adj.shape != (num_r, num_r) or set(np.unique(inp.r_adj).tolist()) - {0, 1}:
        raise RuntimeError("r_graph adjacency is not a 0/1 square matrix")
    if int(inp.r_adj.sum()) != r_edges:
        raise RuntimeError("r_graph adjacency sum differs from num_graph_edges")
    if m_r_edge_ids != inp.r_ids:
        raise RuntimeError("m_graph r_edge_ids differ from r_graph edge_ids")
    if len(set(inp.m_ids)) != num_m:
        raise RuntimeError("m_graph movement_ids are not unique")
    for name in ("m_in", "m_out", "m_junction", "m_turn"):
        if len(getattr(inp, name)) != num_m:
            raise RuntimeError(f"m_graph {name} length differs from movement count")
    r_lookup = {edge_id: index for index, edge_id in enumerate(inp.r_ids)}
    for j, movement in enumerate(inp.m_ids):
        if movement != mgraph.movement_id(inp.m_junction[j], inp.m_in[j], inp.m_out[j]):
            raise RuntimeError(f"movement id {movement} does not match its junction/incoming/outgoing")
        if r_lookup.get(inp.m_in[j]) != int(inp.m_in_r_index[j]) or r_lookup.get(inp.m_out[j]) != int(inp.m_out_r_index[j]):
            raise RuntimeError(f"movement {movement} has inconsistent R indices")
    if inp.m_adj.shape != (num_m, num_m) or int(inp.m_adj.sum()) != m_edges:
        raise RuntimeError("m_graph adjacency shape or sum is inconsistent")
    if inp.m_edge_index.shape != (2, m_edges) or len(inp.m_transition_edges) != m_edges:
        raise RuntimeError("m_graph edge_index or transition_edge_ids length is inconsistent")
    rebuilt = np.zeros_like(inp.m_adj)
    rebuilt[inp.m_edge_index[0], inp.m_edge_index[1]] = 1
    if not np.array_equal(rebuilt, inp.m_adj):
        raise RuntimeError("m_graph edge_index does not reproduce its adjacency")
    with (graph_dir / "m_nodes.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [row["movement_id"] for row in rows] != inp.m_ids:
        raise RuntimeError("m_nodes.csv order differs from m_graph.npz")
    for j, row in enumerate(rows):
        if (
            int(row["movement_index"]) != j
            or row["incoming_edge_id"] != inp.m_in[j]
            or row["outgoing_edge_id"] != inp.m_out[j]
            or row["junction_id"] != inp.m_junction[j]
            or int(row["incoming_r_index"]) != int(inp.m_in_r_index[j])
            or int(row["outgoing_r_index"]) != int(inp.m_out_r_index[j])
        ):
            raise RuntimeError(f"m_nodes.csv row {j} differs from m_graph.npz")
    with (graph_dir / "m_transitions.csv").open(encoding="utf-8", newline="") as handle:
        trans = list(csv.DictReader(handle))
    if len(trans) != m_edges:
        raise RuntimeError("m_transitions.csv row count differs from m_graph.npz")
    for k, row in enumerate(trans):
        if (
            int(row["source_movement_index"]) != int(inp.m_edge_index[0, k])
            or int(row["target_movement_index"]) != int(inp.m_edge_index[1, k])
            or row["intermediate_edge_id"] != inp.m_transition_edges[k]
        ):
            raise RuntimeError(f"m_transitions.csv row {k} differs from m_graph.npz")
    r_nodes = graph_dir / "r_nodes.csv"
    if r_nodes.is_file():
        with r_nodes.open(encoding="utf-8", newline="") as handle:
            ids = [(row["node_id"], row["edge_id"]) for row in csv.DictReader(handle)]
        if ids != [(f"R::{e}", e) for e in inp.r_ids]:
            raise RuntimeError("r_nodes.csv order or R:: naming differs from r_graph.npz")


def build_graph(inp: Inputs) -> dict[str, object]:
    num_r, num_m = len(inp.r_ids), len(inp.m_ids)
    num_nodes = num_r + num_m
    r_lookup = {edge_id: index for index, edge_id in enumerate(inp.r_ids)}
    node_ids = [f"R::{e}" for e in inp.r_ids] + list(inp.m_ids)
    node_types = np.concatenate([np.full(num_r, NODE_R), np.full(num_m, NODE_M)]).astype(np.int64)
    node_local = np.concatenate([np.arange(num_r), np.arange(num_m)]).astype(np.int64)
    r_global = np.arange(num_r, dtype=np.int64)
    m_global = num_r + np.arange(num_m, dtype=np.int64)
    m_in_g = np.array([r_lookup[e] for e in inp.m_in], dtype=np.int64)
    m_out_g = np.array([r_lookup[e] for e in inp.m_out], dtype=np.int64)
    pair_movement: dict[tuple[int, int], int] = {}
    for j in range(num_m):
        key = (int(m_in_g[j]), int(m_out_g[j]))
        if key in pair_movement:
            raise RuntimeError(f"two movements share the road pair {inp.m_in[j]} -> {inp.m_out[j]}")
        pair_movement[key] = int(m_global[j])

    rr = np.argwhere(inp.r_adj > 0).T.astype(np.int64).reshape(2, -1)
    mm = (inp.m_edge_index + num_r).astype(np.int64)
    rm = np.vstack([m_in_g, m_global])
    mr = np.vstack([m_global, m_out_g])
    parts = (rr, mm, rm, mr)
    edge_index = np.concatenate(parts, axis=1)
    edge_type = np.concatenate([np.full(part.shape[1], code, dtype=np.int64) for code, part in enumerate(parts)])
    offsets = np.cumsum([0] + [part.shape[1] for part in parts]).astype(np.int64)

    road = np.full(edge_index.shape[1], -1, dtype=np.int64)
    movement = np.full(edge_index.shape[1], -1, dtype=np.int64)
    rr_slice = slice(offsets[REL_RR], offsets[REL_RR + 1])
    movement[rr_slice] = [pair_movement.get((int(s), int(t)), -1) for s, t in rr.T]
    road[offsets[REL_MM]:offsets[REL_MM + 1]] = [r_lookup[e] for e in inp.m_transition_edges]
    road[offsets[REL_RM]:offsets[REL_RM + 1]] = rm[0]
    movement[offsets[REL_RM]:offsets[REL_RM + 1]] = rm[1]
    road[offsets[REL_MR]:offsets[REL_MR + 1]] = mr[1]
    movement[offsets[REL_MR]:offsets[REL_MR + 1]] = mr[0]

    down_ptr = np.concatenate([[0], np.cumsum(np.bincount(m_in_g, minlength=num_r))]).astype(np.int64)
    up_ptr = np.concatenate([[0], np.cumsum(np.bincount(m_out_g, minlength=num_r))]).astype(np.int64)
    down = (num_r + np.argsort(m_in_g, kind="stable")).astype(np.int64)
    up = (num_r + np.argsort(m_out_g, kind="stable")).astype(np.int64)
    return {
        "num_nodes": num_nodes,
        "num_r": num_r,
        "num_m": num_m,
        "node_ids": node_ids,
        "node_types": node_types,
        "node_local": node_local,
        "node_original": list(inp.r_ids) + list(inp.m_ids),
        "r_global": r_global,
        "m_global": m_global,
        "m_in_g": m_in_g,
        "m_out_g": m_out_g,
        "parts": parts,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "offsets": offsets,
        "edge_road": road,
        "edge_movement": movement,
        "down_ptr": down_ptr,
        "down": down,
        "up_ptr": up_ptr,
        "up": up,
    }


def graph_signature(node_ids: list[str], edge_index: np.ndarray, edge_type: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update("\x1f".join(node_ids).encode("utf-8"))
    digest.update(b"\x1e")
    digest.update(np.ascontiguousarray(edge_index, dtype="<i8").tobytes())
    digest.update(b"\x1e")
    digest.update(np.ascontiguousarray(edge_type, dtype="<i8").tobytes())
    return digest.hexdigest()


def pair_set(src: np.ndarray, dst: np.ndarray) -> set[tuple[int, int]]:
    return set(zip(src.tolist(), dst.tolist()))


def pair_examples(pairs, ids: list[str]) -> list[list[str]]:
    return [[ids[s], ids[t]] for s, t in sorted(pairs)[:EXAMPLES]]


def verify_structure(inp: Inputs, g: dict) -> tuple[dict[str, dict], list[str]]:
    """Check every node and edge against the raw R/M inputs, not against build_graph."""
    checks: dict[str, dict] = {}
    failures: list[str] = []

    def check(name: str, ok: bool, **detail) -> None:
        checks[name] = {"status": "PASS" if ok else "FAIL", **detail}
        if not ok:
            failures.append(name)

    num_r, num_m = len(inp.r_ids), len(inp.m_ids)
    n = g["num_nodes"]
    ids, types, local, original = g["node_ids"], g["node_types"], g["node_local"], g["node_original"]
    r_lookup = {edge_id: index for index, edge_id in enumerate(inp.r_ids)}

    # Nodes.
    check("node_count", n == num_r + num_m == len(ids) == len(types) == len(local) == len(original),
          num_nodes=n, num_r_nodes=num_r, num_m_nodes=num_m)
    duplicates = sorted(node for node, count in Counter(ids).items() if count > 1)
    check("node_ids_unique", not duplicates, duplicates=duplicates[:EXAMPLES])
    check("r_node_order", ids[:num_r] == [f"R::{e}" for e in inp.r_ids] and original[:num_r] == inp.r_ids)
    check("m_node_order", ids[num_r:] == inp.m_ids and original[num_r:] == inp.m_ids)
    check("node_types", bool((types[:num_r] == NODE_R).all() and (types[num_r:] == NODE_M).all()))
    check("id_prefix_matches_type",
          all(i.startswith("R::") for i in ids[:num_r]) and all(i.startswith("M::") for i in ids[num_r:]))
    check(
        "local_global_mapping",
        bool(
            np.array_equal(local[:num_r], np.arange(num_r))
            and np.array_equal(local[num_r:], np.arange(num_m))
            and np.array_equal(g["r_global"], np.arange(num_r))
            and np.array_equal(g["m_global"], num_r + np.arange(num_m))
            and all(ids[int(g["m_global"][j])] == inp.m_ids[j] for j in range(num_m))
            and all(original[int(g["r_global"][i])] == inp.r_ids[i] for i in range(num_r))
        ),
    )
    bad_endpoint = [
        inp.m_ids[j]
        for j in range(num_m)
        if not (
            int(g["m_in_g"][j]) == r_lookup[inp.m_in[j]] == int(inp.m_in_r_index[j])
            and int(g["m_out_g"][j]) == r_lookup[inp.m_out[j]] == int(inp.m_out_r_index[j])
            and ids[int(g["m_in_g"][j])] == f"R::{inp.m_in[j]}"
            and ids[int(g["m_out_g"][j])] == f"R::{inp.m_out[j]}"
        )
    ]
    check("movement_endpoint_mapping", not bad_endpoint, mismatches=bad_endpoint[:EXAMPLES])
    downstream: dict[str, list[int]] = defaultdict(list)
    upstream: dict[str, list[int]] = defaultdict(list)
    for j in range(num_m):
        downstream[inp.m_in[j]].append(num_r + j)
        upstream[inp.m_out[j]].append(num_r + j)
    bad_csr = [
        e for i, e in enumerate(inp.r_ids)
        if g["down"][g["down_ptr"][i]:g["down_ptr"][i + 1]].tolist() != downstream.get(e, [])
        or g["up"][g["up_ptr"][i]:g["up_ptr"][i + 1]].tolist() != upstream.get(e, [])
    ]
    check("road_movement_lists", not bad_csr and len(g["down"]) == len(g["up"]) == num_m, mismatches=bad_csr[:EXAMPLES])

    # Edges.
    edge_index, edge_type, offsets = g["edge_index"], g["edge_type"], g["offsets"]
    num_edges = edge_index.shape[1] if edge_index.ndim == 2 else -1
    check(
        "edge_array_shapes",
        edge_index.shape == (2, num_edges) and edge_type.shape == (num_edges,)
        and int(offsets[-1]) == num_edges
        and all(np.array_equal(edge_index[:, offsets[c]:offsets[c + 1]], g["parts"][c]) for c in range(4))
        and all(bool((edge_type[offsets[c]:offsets[c + 1]] == c).all()) for c in range(4)),
        num_graph_edges=num_edges,
    )
    out_of_range = int(((edge_index < 0) | (edge_index >= n)).any(axis=0).sum())
    check("index_range", out_of_range == 0, out_of_range_edges=out_of_range)
    undefined = sorted(set(np.unique(edge_type).tolist()) - set(range(len(RELATION_NAMES))))
    check("relation_types_defined", not undefined, undefined_types=undefined)
    wrong_types = {}
    for code, (source_type, target_type) in enumerate(RELATION_ENDPOINTS):
        part = edge_index[:, edge_type == code]
        wrong_types[RELATION_NAMES[code]] = int(((types[part[0]] != source_type) | (types[part[1]] != target_type)).sum())
    check("endpoint_node_types", not any(wrong_types.values()), wrong_endpoint_edges=wrong_types)
    duplicate_counts = {}
    for code in range(4):
        part = edge_index[:, edge_type == code]
        duplicate_counts[RELATION_NAMES[code]] = part.shape[1] - len(pair_set(part[0], part[1]))
    all_pairs = pair_set(edge_index[0], edge_index[1])
    check("no_duplicate_edges", not any(duplicate_counts.values()) and len(all_pairs) == num_edges,
          duplicates_per_relation=duplicate_counts, duplicate_pairs_across_relations=num_edges - len(all_pairs))
    self_loops = int((edge_index[0] == edge_index[1]).sum())
    check("no_self_loops", self_loops == 0, self_loops=self_loops)

    rr = g["parts"][REL_RR]
    r_source = pair_set(*np.nonzero(inp.r_adj))
    r_built = pair_set(rr[0], rr[1])
    check("r_r_equals_r_graph", r_built == r_source and rr.shape[1] == len(r_source),
          r_graph_edges=len(r_source), mr_edges=rr.shape[1],
          missing=pair_examples(r_source - r_built, inp.r_ids), extra=pair_examples(r_built - r_source, inp.r_ids))

    mm = g["parts"][REL_MM] - num_r
    m_from_adj = pair_set(*np.nonzero(inp.m_adj))
    m_from_index = pair_set(inp.m_edge_index[0], inp.m_edge_index[1])
    m_built = pair_set(mm[0], mm[1])
    by_incoming: dict[str, list[int]] = defaultdict(list)
    for j, edge_id in enumerate(inp.m_in):
        by_incoming[edge_id].append(j)
    m_rederived = {(i, j) for i in range(num_m) for j in by_incoming.get(inp.m_out[i], []) if i != j}
    check(
        "m_m_equals_m_graph",
        m_built == m_from_adj == m_from_index and np.array_equal(mm, inp.m_edge_index),
        m_graph_edges=len(m_from_adj), mr_edges=mm.shape[1], same_order_as_m_graph=bool(np.array_equal(mm, inp.m_edge_index)),
        missing=pair_examples(m_from_adj - m_built, inp.m_ids), extra=pair_examples(m_built - m_from_adj, inp.m_ids),
    )
    check("m_m_rederived_from_shared_road", m_built == m_rederived,
          rederived_edges=len(m_rederived), missing=pair_examples(m_rederived - m_built, inp.m_ids),
          extra=pair_examples(m_built - m_rederived, inp.m_ids))

    for code, key, side, names in ((REL_RM, "r_m", 0, inp.m_in), (REL_MR, "m_r", 1, inp.m_out)):
        part = g["parts"][code]
        movement_side = part[1 - side] - num_r
        road_side = part[side]
        valid = (movement_side >= 0) & (movement_side < num_m)
        per_movement = np.bincount(movement_side[valid], minlength=num_m) if valid.any() else np.zeros(num_m, int)
        wrong = [inp.m_ids[int(m)] for m, r in zip(movement_side[valid], road_side[valid])
                 if not (0 <= int(r) < num_r and inp.r_ids[int(r)] == names[int(m)])]
        check(
            f"{key}_one_per_movement",
            bool(valid.all() and (per_movement == 1).all()) and not wrong and part.shape[1] == num_m,
            edges=part.shape[1], movements_without_edge=int((per_movement == 0).sum()),
            movements_with_several_edges=int((per_movement > 1).sum()), wrong_road=wrong[:EXAMPLES],
        )

    two_hop = {(r_lookup[inp.m_in[j]], r_lookup[inp.m_out[j]]) for j in range(num_m)}
    check("r_m_r_paths_equal_r_r", two_hop == r_source and len(two_hop) == num_m,
          r_m_r_pairs=len(two_hop), r_r_edges=len(r_source))

    road, movement = g["edge_road"], g["edge_movement"]
    evidence_issues: dict[str, int] = {}
    seg = lambda code: slice(int(offsets[code]), int(offsets[code + 1]))  # noqa: E731
    rr_movement = movement[seg(REL_RR)]
    evidence_issues["R->R without its Movement"] = int(sum(
        1 for (s, t), m in zip(rr.T.tolist(), rr_movement.tolist())
        if m < num_r or inp.m_in[m - num_r] != inp.r_ids[s] or inp.m_out[m - num_r] != inp.r_ids[t]
    ))
    mm_road = road[seg(REL_MM)]
    evidence_issues["M->M shared road mismatch"] = int(sum(
        1 for (s, t), r, via in zip(mm.T.tolist(), mm_road.tolist(), inp.m_transition_edges)
        if not (inp.r_ids[r] == via == inp.m_out[s] == inp.m_in[t])
    ))
    for code in (REL_RM, REL_MR):
        part = g["parts"][code]
        road_side, movement_side = (part[0], part[1]) if code == REL_RM else (part[1], part[0])
        evidence_issues[f"{RELATION_NAMES[code]} link mismatch"] = int(
            (~((road[seg(code)] == road_side) & (movement[seg(code)] == movement_side))).sum()
        )
    check("edge_evidence", not any(evidence_issues.values()), issues=evidence_issues)

    rm_pairs = pair_set(*g["parts"][REL_RM])
    reverse_pairs = sum(1 for s, t in pair_set(*g["parts"][REL_MR]) if (t, s) in rm_pairs)
    checks["r_m_reverse_pairs_info"] = {
        "status": "INFO",
        "count": reverse_pairs,
        "note": "R->M and M->R on the same pair would need a Movement whose incoming and outgoing road are equal",
    }
    return checks, failures


def check_flow_columns(training_root: Path, inp: Inputs) -> tuple[dict, list[str]]:
    """Header of every flow CSV must list the R or M ids in local-index order."""
    trees: list[tuple[str, str, Path, list[str]]] = [
        ("full", "r-level", training_root / "full_flow" / "r-level", inp.r_ids),
        ("full", "m-level", training_root / "full_flow" / "m-level", inp.m_ids),
    ]
    partial = training_root / "partial_flow"
    for level, base, expected in (("r-level", partial, inp.r_ids), ("m-level", partial / "m-level", inp.m_ids)):
        rates = sorted(p.name for p in base.iterdir() if p.is_dir() and RATE_PATTERN.fullmatch(p.name)) if base.is_dir() else []
        trees.extend((f"partial {rate}", level, base / rate, expected) for rate in rates)
    results = []
    failures = []
    for label, level, root, expected in trees:
        header_prefix = ["time_bin", "start_time"]
        days = sorted(p.name for p in root.iterdir() if p.is_dir() and DAY_PATTERN.fullmatch(p.name)) if root.is_dir() else []
        missing: list[str] = []
        mismatched: list[str] = []
        checked = 0
        for day in days:
            for name in FLOW_FILES:
                path = root / day / name
                if not path.is_file():
                    missing.append(f"{day}/{name}")
                    continue
                with path.open(encoding="utf-8") as handle:
                    header = handle.readline().rstrip("\r\n").split(",")
                checked += 1
                if header[:2] != header_prefix or header[2:] != expected:
                    mismatched.append(f"{day}/{name}")
        status = "PASS" if days and not missing and not mismatched else ("MISSING" if not days else "FAIL")
        if status == "FAIL":
            failures.append(f"flow columns {label} {level}")
        results.append({
            "set": label, "level": level, "root": str(root), "days": len(days), "files_checked": checked,
            "missing_files": missing[:EXAMPLES], "column_mismatch_files": mismatched[:EXAMPLES], "status": status,
        })
    return {"trees": results, "status": "PASS" if not failures else "FAIL"}, failures


def csr(num_nodes: int, src: np.ndarray, dst: np.ndarray) -> tuple[list[int], list[int]]:
    order = np.lexsort((dst, src))
    ptr = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=num_nodes))])
    return ptr.tolist(), dst[order].tolist()


def weak_labels(num_nodes: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    parent = list(range(num_nodes))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for s, t in zip(src.tolist(), dst.tolist()):
        a, b = find(s), find(t)
        if a != b:
            parent[max(a, b)] = min(a, b)
    roots = [find(node) for node in range(num_nodes)]
    relabel: dict[int, int] = {}
    return np.array([relabel.setdefault(root, len(relabel)) for root in roots], dtype=np.int64)


def strong_labels(num_nodes: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Iterative Kosaraju on sparse adjacency lists."""
    out_ptr, out_nb = csr(num_nodes, src, dst)
    in_ptr, in_nb = csr(num_nodes, dst, src)
    visited = [False] * num_nodes
    order: list[int] = []
    for start in range(num_nodes):
        if visited[start]:
            continue
        visited[start] = True
        stack = [(start, out_ptr[start])]
        while stack:
            node, position = stack[-1]
            if position < out_ptr[node + 1]:
                stack[-1] = (node, position + 1)
                nxt = out_nb[position]
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append((nxt, out_ptr[nxt]))
            else:
                stack.pop()
                order.append(node)
    labels = [-1] * num_nodes
    count = 0
    for start in reversed(order):
        if labels[start] != -1:
            continue
        labels[start] = count
        stack = [start]
        while stack:
            node = stack.pop()
            for position in range(in_ptr[node], in_ptr[node + 1]):
                nxt = in_nb[position]
                if labels[nxt] == -1:
                    labels[nxt] = count
                    stack.append(nxt)
        count += 1
    return np.array(labels, dtype=np.int64)


def component_summary(labels: np.ndarray, types: np.ndarray, ids: list[str]) -> dict[str, object]:
    sizes = np.bincount(labels)
    order = sorted(range(len(sizes)), key=lambda c: (-int(sizes[c]), c))
    largest = order[0]
    members = labels == largest
    small = []
    for component in order[1:]:
        nodes = np.flatnonzero(labels == component)
        small.append({
            "size": int(sizes[component]),
            "r_nodes": int((types[nodes] == NODE_R).sum()),
            "m_nodes": int((types[nodes] == NODE_M).sum()),
            "examples": [ids[int(i)] for i in nodes[:4]],
        })
    return {
        "count": int(len(sizes)),
        "largest_size": int(sizes[largest]),
        "largest_r_nodes": int((types[members] == NODE_R).sum()),
        "largest_m_nodes": int((types[members] == NODE_M).sum()),
        "singleton_count": int((sizes == 1).sum()),
        "size_distribution": {str(k): int(v) for k, v in sorted(Counter(sizes.tolist()).items())},
        "other_components_examples": small[:EXAMPLES],
    }


def connectivity(inp: Inputs, g: dict) -> tuple[dict, list[str]]:
    num_r, num_m, n = g["num_nodes"] - len(inp.m_ids), len(inp.m_ids), g["num_nodes"]
    ids, types = g["node_ids"], g["node_types"]
    parts = g["parts"]
    views = {
        "R->R only (R nodes)": (num_r, parts[REL_RR], ids[:num_r], types[:num_r]),
        "M->M only (M nodes)": (num_m, parts[REL_MM] - num_r, ids[num_r:], types[num_r:]),
        "R->M + M->R (all nodes)": (n, np.concatenate([parts[REL_RM], parts[REL_MR]], axis=1), ids, types),
        "MR all relations": (n, g["edge_index"], ids, types),
    }
    result = {}
    for name, (size, edges, view_ids, view_types) in views.items():
        result[name] = {
            "num_nodes": size,
            "num_edges": int(edges.shape[1]),
            "weak": component_summary(weak_labels(size, edges[0], edges[1]), view_types, view_ids),
            "strong": component_summary(strong_labels(size, edges[0], edges[1]), view_types, view_ids),
        }
    reference = {
        "R->R only (R nodes)": (rgraph.weakly_connected(inp.r_adj)[0], rgraph.strongly_connected_count(inp.r_adj)),
        "M->M only (M nodes)": (rgraph.weakly_connected(inp.m_adj)[0], rgraph.strongly_connected_count(inp.m_adj)),
    }
    failures = []
    for name, (weak, strong) in reference.items():
        result[name]["dense_reference"] = {"weak": int(weak), "strong": int(strong)}
        if (result[name]["weak"]["count"], result[name]["strong"]["count"]) != (weak, strong):
            failures.append(f"component count differs from dense reference for {name}")
    return result, failures


def degree_tables(g: dict) -> dict[str, np.ndarray]:
    n = g["num_nodes"]
    table = {}
    for code, key in enumerate(RELATION_KEYS):
        part = g["parts"][code]
        table[f"in_{key}"] = np.bincount(part[1], minlength=n).astype(np.int64)
        table[f"out_{key}"] = np.bincount(part[0], minlength=n).astype(np.int64)
    table["in"] = np.bincount(g["edge_index"][1], minlength=n).astype(np.int64)
    table["out"] = np.bincount(g["edge_index"][0], minlength=n).astype(np.int64)
    return table


def degree_audit(g: dict, degrees: dict[str, np.ndarray], inp: Inputs) -> dict[str, object]:
    types, ids = g["node_types"], g["node_ids"]
    num_r = g["num_r"]
    by_type = {}
    for code, type_name in enumerate(NODE_TYPE_NAMES):
        mask = types == code
        entry = {"total_in": mgraph.degree_stats(degrees["in"][mask]), "total_out": mgraph.degree_stats(degrees["out"][mask])}
        for rel_code, key in enumerate(RELATION_KEYS):
            source_type, target_type = RELATION_ENDPOINTS[rel_code]
            if target_type == code:
                entry[f"in {RELATION_NAMES[rel_code]}"] = mgraph.degree_stats(degrees[f"in_{key}"][mask])
            if source_type == code:
                entry[f"out {RELATION_NAMES[rel_code]}"] = mgraph.degree_stats(degrees[f"out_{key}"][mask])
        entry["zero_in_examples"] = [ids[int(i)] for i in np.flatnonzero(mask & (degrees["in"] == 0))[:EXAMPLES]]
        entry["zero_out_examples"] = [ids[int(i)] for i in np.flatnonzero(mask & (degrees["out"] == 0))[:EXAMPLES]]
        by_type[type_name] = entry
    isolated = np.flatnonzero((degrees["in"] == 0) & (degrees["out"] == 0))
    r_iso = np.flatnonzero((inp.r_adj.sum(axis=0) == 0) & (inp.r_adj.sum(axis=1) == 0))
    m_iso = np.flatnonzero((inp.m_adj.sum(axis=0) == 0) & (inp.m_adj.sum(axis=1) == 0))
    source_isolated = [int(i) for i in r_iso] + [num_r + int(j) for j in m_iso]
    return {
        "by_node_type": by_type,
        "isolated_nodes_in_mr": [ids[int(i)] for i in isolated],
        "isolated_in_r_graph": [ids[int(i)] for i in r_iso],
        "isolated_in_m_graph": [ids[num_r + int(j)] for j in m_iso],
        "source_isolated_now_connected": [
            {"node_id": ids[i], "in": int(degrees["in"][i]), "out": int(degrees["out"][i])}
            for i in source_isolated if degrees["in"][i] + degrees["out"][i] > 0
        ],
    }


def joined(values) -> str:
    return "|".join(str(v) for v in values)


def write_npz(path: Path, inp: Inputs, g: dict, signature: str) -> dict[str, np.ndarray]:
    arrays = {
        "node_ids": np.array(g["node_ids"], dtype=np.str_),
        "node_types": g["node_types"],
        "node_type_names": np.array(NODE_TYPE_NAMES, dtype=np.str_),
        "node_local_indices": g["node_local"],
        "node_original_ids": np.array(g["node_original"], dtype=np.str_),
        "edge_index": g["edge_index"],
        "edge_type": g["edge_type"],
        "edge_type_names": np.array(RELATION_NAMES, dtype=np.str_),
        "edge_type_source_node_type": np.array([s for s, _t in RELATION_ENDPOINTS], dtype=np.int64),
        "edge_type_target_node_type": np.array([t for _s, t in RELATION_ENDPOINTS], dtype=np.int64),
        "relation_edge_offsets": g["offsets"],
        "edge_road_global_index": g["edge_road"],
        "edge_movement_global_index": g["edge_movement"],
        "r_r_edge_index": g["parts"][REL_RR],
        "m_m_edge_index": g["parts"][REL_MM],
        "r_m_edge_index": g["parts"][REL_RM],
        "m_r_edge_index": g["parts"][REL_MR],
        "num_nodes": np.int64(g["num_nodes"]),
        "num_r_nodes": np.int64(g["num_r"]),
        "num_m_nodes": np.int64(g["num_m"]),
        "num_graph_edges": np.int64(g["edge_index"].shape[1]),
        "num_edges_per_relation": np.diff(g["offsets"]).astype(np.int64),
        "r_global_index": g["r_global"],
        "m_global_index": g["m_global"],
        "r_edge_ids": np.array(inp.r_ids, dtype=np.str_),
        "movement_ids": np.array(inp.m_ids, dtype=np.str_),
        "m_junction_ids": np.array(inp.m_junction, dtype=np.str_),
        "m_incoming_edge_ids": np.array(inp.m_in, dtype=np.str_),
        "m_outgoing_edge_ids": np.array(inp.m_out, dtype=np.str_),
        "m_incoming_r_global_index": g["m_in_g"],
        "m_outgoing_r_global_index": g["m_out_g"],
        "r_downstream_m_ptr": g["down_ptr"],
        "r_downstream_m_global_index": g["down"],
        "r_upstream_m_ptr": g["up_ptr"],
        "r_upstream_m_global_index": g["up"],
        "r_graph_sha256": np.array(inp.hashes["r_graph.npz"]),
        "m_graph_sha256": np.array(inp.hashes["m_graph.npz"]),
        "m_graph_signature": np.array(inp.m_signature),
        "graph_signature": np.array(signature),
    }
    np.savez_compressed(path, **arrays)
    return arrays


def node_rows(inp: Inputs, g: dict, degrees: dict[str, np.ndarray]) -> tuple[list[str], list[list[object]]]:
    header = [
        "global_index", "node_id", "node_type", "node_type_code", "local_index", "original_id",
        "flow_level", "flow_column_index", "flow_column_name",
        "junction_id", "incoming_edge_id", "outgoing_edge_id",
        "incoming_r_global_index", "outgoing_r_global_index", "turn_direction",
        "upstream_movement_global_indices", "downstream_movement_global_indices",
        "num_upstream_movements", "num_downstream_movements",
        "in_degree", "out_degree",
        *[f"{d}_degree_{key}" for key in RELATION_KEYS for d in ("in", "out")],
    ]
    num_r = g["num_r"]
    rows = []
    for gi in range(g["num_nodes"]):
        local = int(g["node_local"][gi])
        relation_degrees = [int(degrees[f"{d}_{key}"][gi]) for key in RELATION_KEYS for d in ("in", "out")]
        if gi < num_r:
            up = g["up"][g["up_ptr"][local]:g["up_ptr"][local + 1]]
            down = g["down"][g["down_ptr"][local]:g["down_ptr"][local + 1]]
            specific = ["", "", "", "", "", "", joined(up.tolist()), joined(down.tolist()), len(up), len(down)]
            level = "r-level"
        else:
            specific = [
                inp.m_junction[local], inp.m_in[local], inp.m_out[local],
                int(g["m_in_g"][local]), int(g["m_out_g"][local]), inp.m_turn[local], "", "", "", "",
            ]
            level = "m-level"
        rows.append([
            gi, g["node_ids"][gi], NODE_TYPE_NAMES[int(g["node_types"][gi])], int(g["node_types"][gi]), local,
            g["node_original"][gi], level, local, g["node_original"][gi], *specific,
            int(degrees["in"][gi]), int(degrees["out"][gi]), *relation_degrees,
        ])
    return header, rows


def edge_rows(g: dict) -> tuple[list[str], list[list[object]]]:
    header = [
        "edge_position", "source_global_index", "target_global_index", "source_node_id", "target_node_id",
        "source_node_type", "target_node_type", "source_local_index", "target_local_index",
        "relation_type", "relation_name", "road_edge_id", "road_global_index", "movement_id", "movement_global_index",
    ]
    ids, types, local, original = g["node_ids"], g["node_types"], g["node_local"], g["node_original"]
    rows = []
    for k, (s, t, code, road, movement) in enumerate(zip(
        g["edge_index"][0].tolist(), g["edge_index"][1].tolist(), g["edge_type"].tolist(),
        g["edge_road"].tolist(), g["edge_movement"].tolist(),
    )):
        rows.append([
            k, s, t, ids[s], ids[t], NODE_TYPE_NAMES[int(types[s])], NODE_TYPE_NAMES[int(types[t])],
            int(local[s]), int(local[t]), code, RELATION_NAMES[code],
            original[road] if road >= 0 else "", road if road >= 0 else "",
            ids[movement] if movement >= 0 else "", movement if movement >= 0 else "",
        ])
    return header, rows


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def reload_and_check(npz_path: Path, nodes_path: Path, edges_path: Path, arrays: dict[str, np.ndarray], g: dict) -> None:
    with np.load(npz_path, allow_pickle=False) as payload:
        if sorted(payload.files) != sorted(arrays):
            raise RuntimeError("reloaded mr_graph keys differ")
        for key, value in arrays.items():
            loaded = payload[key]
            if loaded.dtype.kind == "O":
                raise RuntimeError(f"{key} needs pickle")
            if loaded.shape != value.shape or not np.array_equal(loaded, value):
                raise RuntimeError(f"reloaded {key} differs")
        rebuilt = np.concatenate([payload[f"{key}_edge_index"] for key in RELATION_KEYS], axis=1)
        if not np.array_equal(rebuilt, payload["edge_index"]):
            raise RuntimeError("per-relation edge_index arrays do not concatenate to edge_index")
        types = payload["edge_type"]
        offsets = payload["relation_edge_offsets"]
        for code in range(len(RELATION_NAMES)):
            if not np.array_equal(np.flatnonzero(types == code), np.arange(offsets[code], offsets[code + 1])):
                raise RuntimeError(f"edge_type {code} is not the contiguous block given by relation_edge_offsets")
    with nodes_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [row["node_id"] for row in rows] != g["node_ids"]:
        raise RuntimeError("mr_nodes.csv node order differs")
    if [int(row["global_index"]) for row in rows] != list(range(g["num_nodes"])):
        raise RuntimeError("mr_nodes.csv global_index is not 0..N-1")
    if [int(row["local_index"]) for row in rows] != g["node_local"].tolist():
        raise RuntimeError("mr_nodes.csv local_index differs")
    with edges_path.open(encoding="utf-8", newline="") as handle:
        edges = list(csv.DictReader(handle))
    if len(edges) != g["edge_index"].shape[1]:
        raise RuntimeError("mr_edges.csv row count differs")
    for k, row in enumerate(edges):
        if (
            int(row["source_global_index"]) != int(g["edge_index"][0, k])
            or int(row["target_global_index"]) != int(g["edge_index"][1, k])
            or int(row["relation_type"]) != int(g["edge_type"][k])
            or row["source_node_id"] != g["node_ids"][int(g["edge_index"][0, k])]
            or row["target_node_id"] != g["node_ids"][int(g["edge_index"][1, k])]
        ):
            raise RuntimeError(f"mr_edges.csv row {k} differs from edge_index")


def audit_markdown(audit: dict) -> str:
    counts = audit["edges_per_relation"]
    conn = audit["connectivity"]
    deg = audit["degrees"]
    flow = audit["flow_column_mapping"]
    lines = [
        "# Xuancheng26 MR 异构图审计",
        "",
        f"- 状态：**{audit['status']}**" + (f"（{'; '.join(audit['failures'])}）" if audit["failures"] else ""),
        f"- 生成时间：{audit['generated_at']}",
        f"- 输入：`{audit['inputs']['r_graph']}`、`{audit['inputs']['m_graph']}`（运行前后哈希不变：{audit['inputs']['unchanged_after_run']}）",
        f"- MR 图签名：`{audit['graph_signature']}`；M 图签名：`{audit['inputs']['m_graph_signature']}`",
        "",
        "## 节点",
        "",
        "| 类型 | 编码 | 数量 | global_index | local_index 含义 |",
        "| --- | --- | --- | --- | --- |",
        f"| R（真实道路，`R::<edge_id>`） | 0 | {audit['num_r_nodes']} | 0–{audit['num_r_nodes'] - 1} | `r_graph.npz` 的 `edge_ids` 序号 = R-level 流量 CSV 数据列序号 |",
        f"| M（Movement，`M::<junction>::<in>::<out>`） | 1 | {audit['num_m_nodes']} | {audit['num_r_nodes']}–{audit['num_nodes'] - 1} | `m_graph.npz` 的 `movement_ids` 序号 = M-level 流量 CSV 数据列序号 |",
        f"| 合计 | | {audit['num_nodes']} | | |",
        "",
        "## 关系",
        "",
        "| 编码 | 关系 | 边数 | 依据 |",
        "| --- | --- | --- | --- |",
        f"| 0 | R->R | {counts['R->R']} | `r_graph.npz` 邻接矩阵，逐边相同 |",
        f"| 1 | M->M | {counts['M->M']} | `m_graph.npz` edge_index，逐边相同且顺序一致 |",
        f"| 2 | R->M | {counts['R->M']} | Movement 的 incoming_edge -> Movement |",
        f"| 3 | M->R | {counts['M->R']} | Movement -> outgoing_edge |",
        f"| | 合计 | {audit['num_graph_edges']} | 密度 {audit['graph_density']:.8f}，自环 {audit['self_loops']} |",
        "",
        "所有边保持交通传播方向，没有无向化、自环或额外远距离连接。R->M->R 两跳得到的道路对与 R->R 完全相同；"
        "R->R 与 M->M 作为独立关系保留，供后续消融。",
        "",
        "## 结构校验",
        "",
        "| 检查 | 结果 |",
        "| --- | --- |",
        *[f"| {name} | {item['status']} |" for name, item in audit["checks"].items()],
        f"| flow_column_mapping | {flow['status']} |",
        f"| reload_check | {audit['reload_check']} |",
        "",
        "## 度",
        "",
        "| 节点类型 | 入度 min/mean/max | 出度 min/mean/max | 入度为 0 | 出度为 0 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for type_name, entry in deg["by_node_type"].items():
        i, o = entry["total_in"], entry["total_out"]
        lines.append(f"| {type_name} | {i['min']}/{i['mean']}/{i['max']} | {o['min']}/{o['mean']}/{o['max']} | {i['zero_count']} | {o['zero_count']} |")
    lines += [
        "",
        "按关系拆分的度见 `mr_graph_audit.json` 的 `degrees.by_node_type`。",
        "",
        "## 连通性",
        "",
        "| 视图 | 节点 | 边 | 弱连通分量 | 最大弱分量（R/M） | 强连通分量 | 最大强分量（R/M） | 强分量单点数 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, item in conn.items():
        w, s = item["weak"], item["strong"]
        lines.append(
            f"| {name} | {item['num_nodes']} | {item['num_edges']} | {w['count']} | {w['largest_size']} ({w['largest_r_nodes']}/{w['largest_m_nodes']}) "
            f"| {s['count']} | {s['largest_size']} ({s['largest_r_nodes']}/{s['largest_m_nodes']}) | {s['singleton_count']} |"
        )
    lines += [
        "",
        f"- MR 图孤立节点：{len(deg['isolated_nodes_in_mr'])}",
        f"- 原 R 图孤立节点：{len(deg['isolated_in_r_graph'])}；原 M 图孤立节点：{len(deg['isolated_in_m_graph'])}"
        + (f"（{', '.join('`' + x + '`' for x in deg['isolated_in_m_graph'])}）" if deg["isolated_in_m_graph"] else ""),
        f"- 原图孤立、在 MR 中经跨类型关系接通的节点：{len(deg['source_isolated_now_connected'])}。这是跨类型关系的正常结果，不是异常。",
        "",
        "## 流量列映射",
        "",
        "联合图节点 `g` 对应的流量列：`node_types[g]` 选 R-level 或 M-level，`node_local_indices[g]` 是 CSV 中 `time_bin,start_time` 之后的列序号，"
        "列名等于 `node_original_ids[g]`。R 与 M 的四个状态物理含义不同（道路存量 vs. 连接区存量），本图不合并它们。",
        "",
        "| 数据 | 层级 | 天数 | 检查文件 | 结果 |",
        "| --- | --- | --- | --- | --- |",
        *[f"| {t['set']} | {t['level']} | {t['days']} | {t['files_checked']} | {t['status']} |" for t in flow["trees"]],
        "",
    ]
    return "\n".join(lines)


def build_all(graph_dir: Path, training_root: Path) -> int:
    inp = load_inputs(graph_dir)
    log(f"inputs: R nodes={len(inp.r_ids)} R edges={int(inp.r_adj.sum())} M nodes={len(inp.m_ids)} M edges={inp.m_edge_index.shape[1]}")
    g = build_graph(inp)
    signature = graph_signature(g["node_ids"], g["edge_index"], g["edge_type"])
    checks, failures = verify_structure(inp, g)
    flow, flow_failures = check_flow_columns(training_root, inp)
    failures += flow_failures
    conn, conn_failures = connectivity(inp, g)
    failures += conn_failures
    degrees = degree_tables(g)
    num_edges = int(g["edge_index"].shape[1])
    counts = {name: int(c) for name, c in zip(RELATION_NAMES, np.diff(g["offsets"]))}
    audit = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "graph_dir": str(graph_dir),
        "graph_signature": signature,
        "inputs": {
            "r_graph": str(graph_dir / "r_graph.npz"),
            "m_graph": str(graph_dir / "m_graph.npz"),
            "sha256": inp.hashes,
            "m_graph_signature": inp.m_signature,
        },
        "node_type_codes": {name: code for code, name in enumerate(NODE_TYPE_NAMES)},
        "relation_type_codes": {
            name: {"code": code, "source_type": NODE_TYPE_NAMES[s], "target_type": NODE_TYPE_NAMES[t]}
            for code, (name, (s, t)) in enumerate(zip(RELATION_NAMES, RELATION_ENDPOINTS))
        },
        "num_nodes": g["num_nodes"],
        "num_r_nodes": g["num_r"],
        "num_m_nodes": g["num_m"],
        "num_graph_edges": num_edges,
        "edges_per_relation": counts,
        "graph_density": num_edges / float(g["num_nodes"] ** 2),
        "self_loops": int((g["edge_index"][0] == g["edge_index"][1]).sum()),
        "checks": checks,
        "flow_column_mapping": flow,
        "degrees": degree_audit(g, degrees, inp),
        "connectivity": conn,
    }
    log(f"structure checks: {sum(c['status'] == 'PASS' for c in checks.values())} pass, failures={failures or 'none'}")
    log(f"flow column mapping {flow['status']}: " + ", ".join(f"{t['set']} {t['level']}={t['status']}" for t in flow["trees"]))

    outputs = {name: graph_dir / name for name in OUTPUT_NAMES}
    temps = {
        "mr_graph.npz": graph_dir / "mr_graph.tmp.npz",
        **{name: graph_dir / f"{name}.tmp" for name in OUTPUT_NAMES if name != "mr_graph.npz"},
    }
    try:
        arrays = write_npz(temps["mr_graph.npz"], inp, g, signature)
        write_csv(temps["mr_nodes.csv"], *node_rows(inp, g, degrees))
        write_csv(temps["mr_edges.csv"], *edge_rows(g))
        reload_and_check(temps["mr_graph.npz"], temps["mr_nodes.csv"], temps["mr_edges.csv"], arrays, g)
        audit["reload_check"] = "PASS"
        after = {name: sha256_file(graph_dir / name) for name in inp.hashes}
        audit["inputs"]["unchanged_after_run"] = after == inp.hashes
        if after != inp.hashes:
            failures.append("input graph files changed during the run")
        audit["status"] = "PASS" if not failures else "FAIL"
        audit["failures"] = failures
        temps["mr_graph_audit.json"].write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temps["mr_graph_audit.md"].write_text(audit_markdown(audit), encoding="utf-8")
        publish = OUTPUT_NAMES if not failures else ("mr_graph_audit.json", "mr_graph_audit.md")
        for name in publish:
            os.replace(temps[name], outputs[name])
    finally:
        for path in temps.values():
            if path.exists():
                path.unlink()

    full = conn["MR all relations"]
    log(f"nodes={g['num_nodes']} (R={g['num_r']}, M={g['num_m']}) edges={num_edges} {counts}")
    log(f"density={audit['graph_density']:.8f} self_loops={audit['self_loops']} isolated={len(audit['degrees']['isolated_nodes_in_mr'])}")
    log(f"weak components={full['weak']['count']} strong components={full['strong']['count']} "
        f"largest strong={full['strong']['largest_size']}")
    if failures:
        log(f"audit status=FAIL failures={failures}; only the audit files were written to {graph_dir}")
        return 2
    log(f"audit status=PASS output={graph_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Xuancheng26 directed R-M heterogeneous graph")
    prep_paths.add_3s_path_arguments(parser)
    parser.add_argument("--force", action="store_true", help="overwrite existing mr_* outputs")
    args = parser.parse_args(argv)
    root = prep_paths.refuse_code_root(args.training_root)
    graph_dir = root / "graph"
    log(f"training_root={root}")
    log(f"graph_dir={graph_dir}")
    existing = [name for name in OUTPUT_NAMES if (graph_dir / name).exists()]
    if existing and not args.force:
        log(f"refusing to overwrite {', '.join(existing)}; pass --force")
        return 1
    return build_all(graph_dir, root)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
