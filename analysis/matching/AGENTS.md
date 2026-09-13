# Trajectory matching instructions

## Scope

This directory converts OCR-reconstructed OSM edge sequences into continuous
SUMO edge routes. Follow the repository-root instructions as well.

## Inputs, algorithm, and outputs

- The main trajectory input is
  `../../ocr/ocr_output_all/route_by_edge_split.txt`, with records shaped as
  `vin HH:MM:SS osm_edge_id ...`.
- `build_sumo_edge_graph.py` reads `../road_network/net_tls.net.xml`, excludes
  internal or passenger-inaccessible edges, and builds an edge-to-edge directed
  graph from SUMO `<connection>` elements.
- `match_sumo_edges.py` maps each OSM ID to candidate SUMO edges, connects
  candidates with cached nearby paths/BFS, and selects a route with layered
  dynamic programming. Fuzzy fallback may drop unmappable observations but must
  record what was dropped and which bridge edges were inserted.
- `parse_trajectories.py` orchestrates matching and writes `matched_routes.txt`
  plus failure/fuzzy logs under `log/`.
- `export_sumo_routes.py` converts successful text records into
  `matched_routes.rou.xml` for SUMO.

Route text, route XML, and log files are generated outputs. Do not hand-edit
them to fix matching behavior.

## Matching invariants

- Treat OSM and SUMO IDs as strings. SUMO IDs may be signed and split into
  segments such as `123#1` or `-123#1`.
- Candidate lookup must consider both directions without losing the selected
  direction or segment suffix.
- Every emitted adjacent SUMO edge pair must be connected in the passenger
  graph. Never fabricate a direct join merely because two edges share an OSM
  base ID.
- Keep strict matching behavior deterministic. Candidate ordering, BFS
  traversal, DP tie-breaking, and cache keys affect reproducibility.
- Fuzzy fallback is an auditable fallback, not permission to hide failures.
  Preserve failure reasons, dropped OSM edges, chosen candidates, and inserted
  bridge-edge logging.
- Preserve each record's VIN and first timestamp when exporting. Vehicle IDs in
  SUMO XML must remain unique.
- Loop removal must not break route connectivity. Keep matching-time loop
  cleanup distinct from the reverse-edge cleanup performed later by
  `simulation/merge_basic_routes.py`.
- Maintain streaming input behavior; OCR trajectory collections and SUMO
  networks can be large.

## Validation

- Prefer small synthetic directed graphs for algorithm changes. Cover exact
  match, split OSM ways, reverse direction, a one/two-hop bridge, unreachable
  candidates, fuzzy dropping, deterministic ties, and loop cleanup.
- Verify every edge in a produced fixture exists in the graph and every
  adjacent pair is connected.
- `python analysis/matching/parse_trajectories.py --help` and
  `python analysis/matching/export_sumo_routes.py --help` are safe interface
  checks. Running them without `--help` may overwrite route and log outputs.
- After a deliberate real-data regeneration, review counts of successful,
  fuzzy, failed, and loop-pruned trajectories before accepting the XML output.
