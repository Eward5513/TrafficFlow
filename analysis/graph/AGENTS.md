# Static graph construction instructions

## Scope

This directory builds static graphs for later movement-flow models. The current
stage is **R-only**: a directed road-level subgraph. Do not read trajectories,
vehicle IDs, penetration samples, or traffic counts here. Do not build M-only
or MR graphs in this directory until a later task asks for them.

Follow the repository-root instructions as well.

## Inputs

- Subgraph edge list: `../simulation/data/subgraph.txt`. One SUMO edge ID per
  line. This file is the only authority for the R-node set. IDs are strings;
  keep `-`, `#`, `_`, and `:` unchanged. First occurrence wins; later duplicates
  are reported, not reordered.
- Canonical SUMO network: `../road_network/net_tls.net.xml`. R-graph edges come
  only from `<connection from="..." to="...">` records whose both endpoints are
  in the subgraph set. Do not infer edges from geometry, shared junctions,
  names, shortest paths, or trajectories.

`build_r_graph.py` defaults to those two paths via `__file__`. CLI flags
override them. Do not hard-code node counts or connection lists.

## Outputs

Generated artifacts go to `r_graph/` and must not overwrite simulation,
trajectory, or sampling files.

| File | Role |
|---|---|
| `r_nodes.csv` | R nodes, `node_index` 0..N-1 in subgraph first-seen order |
| `r_edges.csv` | Unique directed road-level edges plus raw lane-connection counts |
| `r_sumo_connections.csv` | Every kept lane-level SUMO connection |
| `r_adjacency.npy` | Dense `uint8` binary directed adjacency, shape `[N_R, N_R]` |
| `r_adjacency_sparse.npz` | SciPy CSR copy of the same matrix |
| `r_edge_index.npy` | `[2, E_R]` source/target indices, sorted by `(source, target)` |
| `r_graph_metadata.json` | Construction contract and input/output hashes |
| `r_graph_validation.json` | Machine-readable checks |
| `r_graph_preview.png` | Geographic directed preview labeled by `node_index` |
| `r_graph_preview_unlabeled.png` | Same layout without labels |

## Adjacency contract

```text
A[i, j] = 1  iff  node i (source/from/upstream) -> node j (target/to/downstream)
```

Rows are sources, columns are targets. The saved matrix is raw, binary, and
directed. Do not add `I`, symmetrize, normalize, or store lane-count weights.
If a later model needs `A + I` or a normalized Laplacian, build that at load
time from this raw matrix.

Multiple lane-level connections between the same edge pair become one R-graph
edge. `raw_connection_count` and `r_sumo_connections.csv` keep the original
lane records for a later M-graph.

## Safety

- Do not edit `subgraph.txt` or `net_tls.net.xml`.
- Do not delete `../simulation/data/` or processed trajectory directories.
- `--overwrite` replaces only known artifacts in `--output-dir`.
- Weak connectivity is expected but must be measured, not enforced. Do not add
  edges to make the graph weakly or strongly connected.
- Internal SUMO edges (`:` prefix) are R nodes only if they appear in the
  subgraph file. Do not add them automatically.

## Validation

`python3 analysis/graph/build_r_graph.py --help` is a safe interface check.
A default run parses the real net file and writes `r_graph/`; it does not
launch SUMO. The script first runs a tiny in-memory fixture (`--skip-synthetic`
disables it). After a real run, `r_graph_validation.json` must report
`status=ok` before downstream model work treats the R graph as authoritative.
