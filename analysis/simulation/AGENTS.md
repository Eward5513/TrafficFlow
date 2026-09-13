# SUMO simulation instructions

## Scope

This directory builds the daily traffic demand, runs SUMO on it, and stores the
large route and simulation artifacts under `data/`. Follow the repository-root
instructions as well.

## Demand and experiment model

The demand model is 20 consecutive days, one route file per day.

- `generate_random_routes.py` calls `$SUMO_HOME/tools/randomTrips.py` once per
  day and writes `data/random_trip/<index>_<seed>.trips.xml`,
  `data/random_trip/<index>_<seed>.rou.xml`,
  `data/random_trip/<index>_<seed>.vtype.xml`, and
  `data/random_trip/logs/<index>_<seed>.log`. It needs a working SUMO
  environment because `--validate` runs `duarouter`. `--vtype` must be that
  per-day file: duarouter also uses it as `vtype-output`, so a shared path
  races across parallel jobs and produces `unterminated comment`.
- `merge_basic_routes.py` doubles the matched trajectories from
  `../matching/matched_routes.rou.xml`, jitters every depart independently,
  merges that day's random background flow, and writes
  `data/route_demand/<index>_<seed>.rou.xml` sorted by ascending depart.
- Indices are `1..20` without zero padding. Seeds are drawn randomly by
  `generate_random_routes.py` and recorded permanently in the filenames.
- `merge_basic_routes.py` parses the index and seed out of the
  `data/random_trip/` filenames. It never draws seeds itself. The same seed drives both that day's
  background flow and that day's depart jitter, so the merge step can be re-run
  at any time with identical output.
- `run_simulations.py` is the entry point for the whole chain. It generates any
  missing random background flow and route demand by calling the other two
  scripts as subprocesses, then runs one SUMO simulation per day from
  `data/route_demand/<index>_<seed>.rou.xml` plus
  `../road_network/net_tls.net.xml` into `data/simulation/<index>_<seed>/`.
- The three scripts communicate only through the `<index>_<seed>` filename stem.
  Changing that naming convention breaks the handoff in all directions.

Per-day composition and cleanup:

- Matched trajectories are duplicated: the original keeps its id, the copy gets
  a `.dup1` suffix, and both draw their own jitter from `[-300, +300)` seconds
  before being clamped at `0`.
- The background flow is neither duplicated nor jittered; its departs are
  preserved exactly.
- Reverse-edge cleanup uses `../road_network/reverse_edge_pairs.txt` and checks
  connectivity against `../road_network/net_tls.net.xml`. Routes shorter than
  three edges are then dropped entirely.
- Loop pruning and short-route filtering depend only on the edge sequence, not
  on depart, so the matched side is cleaned once and reused across all 20 days.
  Any change must preserve that equivalence.

Reference numbers from the current inputs: 161,374 matched trajectories, loop
pruning touches 31,001 vehicles and removes 81,038 edges, 5,360 short routes are
dropped, leaving 156,014. Each day therefore holds about
`156,014 * 2 + 30,000` vehicles in roughly 205 MB, and the full run is a few
minutes of CPU plus about 4 GB on disk.

## Simulation stage

`run_simulations.py` defaults to all 20 days and resolves its own dependencies,
so a full run is a single command. Its order of work matters and must be kept:

1. Days whose five output files already exist and are non-empty are dropped
   first, by scanning the `data/simulation/<index>_<seed>/` folder names. This
   happens *before* any upstream check so a finished day never triggers a
   200 MB route-demand rebuild. `--force` and `--generate-only` skip this
   filter.
2. Missing random background flow is generated with
   `generate_random_routes.py --days <max index> --jobs <jobs>`. That script
   only generates `1..N` contiguously, so running a later day (e.g. `--days 5`)
   still fills in days 1..5 of `data/random_trip/`, and `data/random_trip/`
   stays the single source of the index-to-seed mapping.
3. Missing route demand is built with one `merge_basic_routes.py --only ...`
   call covering every missing day at once. Do not call it per day: it reparses
   and cleans 161k matched trajectories on every invocation.
4. Only then are the configs written and SUMO started.

Both upstream scripts already skip work that exists, and this script only calls
them when a file is actually missing, so re-running is cheap. A route file that
does not end in `</routes>` counts as missing and gets rebuilt. `--no-upstream`
turns every missing input into an error instead of a rebuild.

`run_simulations.py` generates each day's `sumocfg` itself; there is no
checked-in template. Every path inside the config is relative to the day folder
that holds it, so the config, the log, and all outputs stay together:

```text
data/simulation/<index>_<seed>/
    simulation.sumocfg  fcd.csv.gz     tripinfo.xml
    vehroute.xml        summary.xml    statistics.xml    simulation.log
```

- SUMO's `--seed` is the day's seed from the filename stem, so a day is
  reproducible and the days are independent of each other.
- `../road_network/tls_schedule.add.xml` is loaded as an additional file. It
  supplies the `offpeak` / `evening_peak` programs and the WAUT that switches at
  60000 s and back at 75000 s; without it every junction stays on the network's
  `programID="0"` for the whole day. `--no-tls-schedule` disables it.
- The simulation window is `begin=0`, `end=86400` (24 hours). Only data within
  this interval is recorded. `--end` can shorten the window for smoke tests.
- FCD is written straight to `.gz` because a full day of per-second records is
  tens of GB uncompressed. Downstream readers must open it with `gzip`.
- `--force` re-runs finished days and overwrites them. The script never deletes
  a directory.

`data/` is entirely generated. The Python scripts are the place to make
behavioral changes.

## Subgraph FCD extraction

`extract_subgraph_trajectories.py` reads `data/subgraph.txt` (one SUMO edge
id per line) and the day folders under `data/simulation/<index>_<seed>/fcd.csv.gz`.
With no day arguments it processes every discovered day (currently 1..20).
Days run as independent process-pool jobs (`--jobs`, default 4): each job
decompresses that day's gzip into `--temp-dir/day_<NN>/`, truncates vehicles
that touch the subgraph, writes `data/processed/subgraph_trajectories/day_<NN>/`,
then deletes only that day's temporary CSV. Vehicle state is never shared
across days.

A vehicle is kept if any FCD row has `edge_id` exactly in the subgraph set.
`edge_id` is parsed from the FCD `lane` field by stripping the trailing
`_<lane_index>`. Internal edges (`:` prefix) stay in the truncated trajectory
but are not subgraph matches. Each kept vehicle is written as one contiguous
span from its first subgraph appearance through its last, including any
off-subgraph or internal-edge rows in between.

Per-day outputs:

- `trajectories.csv`
- `vehicle_ids.txt` (first subgraph-entry order)
- `validation_summary.json`

Default batch: no extra flags. Restrict with `--day 1` or `--from-day 1 --to-day 10`.
`--jobs 1` is sequential. Extraction of all selected days finishes first; a unified
validation pass then checks each day's outputs and writes
`validation_summary_all_days.csv`. A failed day is recorded and does not delete
other days; rerun that day alone with `--day N`. `--skip-existing` skips a day
whose `validation_summary.json` already reports `status=ok`.

## Subgraph edge flow

`count_edge_flow.py` reads an already extracted day's `trajectories.csv` plus
`data/subgraph.txt`. It does not open FCD. A vehicle entry is counted when
`edge_id` changes (or on the vehicle's first row) and the new id is in the
unique `subgraph.txt` set. Seconds spent on the same edge are not counted.
Internal and out-of-subgraph edges are only used to detect the change. The
aggregation window is `--window-minutes` (default 5) or `--window-seconds`.
The script writes `edge_flow_<N>min.csv` (unique subgraph edges × one window
per bin, zeros filled), `edge_plot_index_<N>min.csv`, and one line plot per
unique subgraph edge under `edge_flow_plots_<N>min/`. Paths, day label, window
length, and DPI are CLI arguments; do not hard-code a day. Run one day at a
time until the counts and plots are accepted.

`rank_edge_flow.py` reads those already written `edge_flow_<N>min.csv` files.
It does not open trajectories or FCD. It writes `edge_flow_rank_<N>min.csv`
(edges sorted by daily vehicle entries, including `mean_entries_per_window` =
daily total / window count), `edge_flow_mean_<N>min.png` (ranked day-scale
average flow), and `edge_flow_compare_<N>min.png` (daily-total bar plus
intra-day heatmap). `--root` processes every `day_*` folder that has a flow
CSV and also writes 20-day mean ranking, mean-flow, and comparison plots in
the root.

## Subgraph observed edge flow

`count_observed_edge_flow.py` reads an already extracted day's
`trajectories.csv`, the day's `edge_flow_5min.csv`, `vehicle_ids.txt`, and
`sampled_vehicle_ids/vehicles_pXX.txt`. It does not open FCD and does not
resample vehicles. Road-entry events use the same rule as
`count_edge_flow.py`: an event is a change of `edge_id` (or the vehicle's
first row) onto a subgraph / R-node id. Internal `:` edges are kept in the
pointer chain but are not counted. One trajectory pass builds the event
table; the seven nested ID files only filter that table. There is no `1/p`
scaling.

Per-day outputs go under `observed_edge_flow_5min/`:
`edge_flow_5min_pXX.csv` plus `validation_summary.json`. CSV key columns match
the full `edge_flow_5min.csv` (`window_index`, `window_start`, `time_label`,
`edge_id`, `vehicle_count`) and add `day`, `node_index`, `penetration_rate`.
Node order follows `analysis/graph/r_graph/r_nodes.csv`. A root summary
`observed_edge_flow_5min_summary_all_days.csv` has one row per day × rate
(140 rows for 20 days and 7 rates). `--day day_01` reruns one failed day;
`--overwrite` replaces that day's observed CSVs only.

## Subgraph vehicle-ID penetration sampling

`sample_vehicle_ids.py` reads each day's `vehicle_ids.txt` only. It does not
open trajectories, the network, `subgraph.txt`, or FCD. One SHA256 ordering
per day (`sha256(seed + '\\0' + day_directory_name + '\\0' + vehicle_id)`,
seed default 42) is truncated to `floor(N * p)` for p in 5, 10, 20, 30, 40,
50, 70 percent, so the samples are nested prefixes. Outputs go under each
day's `sampled_vehicle_ids/` plus `sampling_summary_all_days.csv` in the
trajectory root. Discover day folders by the presence of `vehicle_ids.txt`;
do not hard-code day names. Default all-day runs require exactly 20 such
folders.

## Safety and reproducibility

- `merge_basic_routes.py` overwrites `data/route_demand/<index>_<seed>.rou.xml`
  in place. It does not delete directories, but re-running it does replace
  existing day files. Use `--only <index>` when validating.
- `generate_random_routes.py` skips days whose route file already exists and
  looks complete, and only draws new seeds for the missing indices. Do not make
  it renumber or re-seed existing days.
- `run_simulations.py` with no arguments builds and simulates all 20 days,
  which can mean hours of SUMO plus hundreds of GB. Use `--days 1` when
  validating, and `--no-upstream` when the upstream data must not be rebuilt.
- `run_simulations.py --force` overwrites a day's whole output folder contents.
  A full day is expensive and large; get explicit intent before replacing
  existing simulation results.
- Never fabricate files in `data/random_trip/`. They must come from an actual
  `randomTrips.py` run, because downstream code treats them as the background
  demand of record. Point a test at a temporary directory instead.
- Preserve unique vehicle IDs, numeric `depart` values, ascending depart order,
  valid vType references, and connected edge sequences.
- Keep the background flow's `random` id prefix and `random_passenger` vType id.
  The prefix distinguishes background vehicles from matched-trajectory VINs, and
  a differing vType definition makes the merge step fail its conflict check.
  SUMO configs assign the rerouting device to all vehicles with
  `device.rerouting.probability=1`.
- Do not confuse traffic demand duplication with observation penetration.
  Duplication changes how many vehicles SUMO simulates; penetration changes
  which vehicles are observed.
- SUMO config relative paths are resolved from the config/run location. Verify
  them after moving a config or output directory.
- Do not edit FCD, tripinfo, vehroute, summary, statistics, or state files by
  hand. Regenerate them from route/config sources when replacement is intended.
- Full SUMO runs are expensive and write large FCD/state files. Use a short time
  range and temporary output paths for routine validation when possible.

## Validation

- `--help` on any of the three scripts is a safe interface check.
  `run_simulations.py --days 1 --generate-only --no-upstream` prepares
  nothing and only writes that day's `sumocfg`, so it is safe to run for an
  already-built day.
- `run_simulations.py`'s dependency resolution is testable without SUMO: point
  `SIMULATION_DIR` at a temporary directory, stub `run_upstream_script`, and
  check that a finished day short-circuits before any upstream call, that a
  missing day reaches the right script with the right arguments, and that
  `--no-upstream` turns both into errors.
- For route-building changes, use a tiny XML fixture covering conflicting
  vTypes, depart sorting, reverse-pair deletion, connectivity protection, short
  routes, duplicate IDs, jitter bounds, and the clamp at 0.
- Check jitter determinism by building the same day twice and comparing bytes,
  and confirm a different seed produces a different file.
- A real-scale check can point `DATA_DIR` / `RANDOM_DIR` / `DEMAND_DIR` at a
  temporary directory while still reading the real matched routes and network.
  Assert
  156,014 matched trajectories survive cleanup, ids are unique, departs are
  non-negative and non-decreasing, adjacent edges are connected, every matched
  id appears exactly twice, and the background departs are untouched.
- For an intentional SUMO run, check the process exit code and log, then confirm
  all expected FCD/tripinfo/vehroute/summary/statistics files are present before
  downstream flow processing.
