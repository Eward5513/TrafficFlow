# Flow analysis instructions

## Scope

This directory turns SUMO FCD trajectories and traffic-light definitions into
cycle/phase/movement flow counts, then evaluates estimators for partially
observed traffic. Follow the repository-root instructions as well.

## Inputs and stage order

The normal order is:

1. `print_junction_tls.py` extracts controlled connections and green phases
   from `../road_network/net_tls.net.xml` into `junction_tls.json`.
2. `count_edge_departures_by_cycle.py` reads
   `../simulation/output/<penetration>/fcd.csv` and writes per-junction ground
   truth under `full_flow/<penetration>/`.
3. `scale/calculate_scale_estimates.py` joins full-flow events with sampled VIN
   lists from `../simulation/penetration/` and writes scale-method metrics.
4. `propagation/calculate_propagation_estimates.py` propagates configured
   upstream estimates into downstream service windows.
5. Plotting scripts and `junction_metrics/summarize_junction_metrics.py` render
   and aggregate those results.

`junction_tls.json`, `full_flow/`, `scale/metrics/`, `propagation/metrics/`, and
plot files are derived outputs unless a task explicitly treats a checked-in
file as an experiment fixture.

## Data semantics and invariants

- “Full flow” is computed from all vehicles in FCD and serves as ground truth.
  “Observed” flow includes only IDs in the penetration sample file.
- Preserve the hierarchy and keys for junction, cycle, phase, direction,
  movement, lane, and time slice. Plotting and aggregation rely on them.
- Keep time-window boundary semantics consistent across counting, scaling, and
  propagation. Do not silently shift events between cycles or phases.
- SUMO edge and lane IDs may contain `-`, `#`, `_`, and `:`. Only strip a lane
  suffix when it is known to be a numeric lane index.
- Green states are represented by both `G` and `g`. Preserve direction codes
  and connection ordering from the TLS data.
- Scaling must handle zero observations and zero/invalid penetration safely.
  Capacity capping is lane based; current code assumes approximately one
  vehicle per two seconds of effective green. Changing that is an experiment
  definition change, not a refactor.
- Propagation links are intentionally configured for selected corridors. Do not
  infer that the current implementation performs general network assignment or
  downstream turning-split estimation.
- Preserve raw counts alongside estimated values and metrics so results remain
  auditable.

## Implementation and validation

- Put reusable counting/metric logic in pure functions; keep file discovery,
  multiprocessing, and writing at the entry-point boundary.
- Do not duplicate estimator math in plotting scripts. Plots should consume the
  JSON/row outputs produced by calculation scripts.
- The counting entry point processes every available penetration directory in
  parallel and overwrites derived JSON. Use small fixtures for routine checks.
- When changing counting, test at least: a vehicle remaining on one edge, one
  valid crossing, an internal edge, a phase boundary, and a repeated FCD row.
- When changing estimators, test zero samples, low penetration, capacity
  clipping, smoothing at the first cycle, and missing upstream data.
- For a deliberate full refresh, run the five stages in order and compare both
  aggregate metrics and representative per-cycle records.
