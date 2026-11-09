# TrafficFlow repository instructions

## Scope and purpose

These instructions apply to the whole repository. More specific `AGENTS.md`
files under `analysis/` and `reimplementation/` override or extend them for
their own directory trees.

TrafficFlow is a research/data-analysis pipeline rather than an application or
service. It reconstructs vehicle trajectories from OCR data, maps OSM road IDs
to a SUMO network, runs traffic simulations at different observation
penetrations, and evaluates intersection-flow estimators.

The main data flow is:

```text
ocr -> analysis/road_network + analysis/matching
    -> analysis/simulation -> analysis/flow
                        \-> analysis/graph
 \-> reimplementation/stgcn
 \-> reimplementation/dcrnn
 \-> reimplementation/graph_wavenet
 \-> reimplementation/stsgcn
 \-> reimplementation/stfgnn
 \-> reimplementation/pdformer
 \-> reimplementation/stid
 \-> reimplementation/staeformer
 \-> reimplementation/s2s_gcn_lstm
 \-> reimplementation/tlast
 \-> reimplementation/asttn
 \-> reimplementation/stjgcn
 \-> MSTT (new backbone, not a baseline port; see MSTT/AGENTS.md)
 ```

Read `README.md` for the conceptual workflow, but treat the current scripts and
their constants as authoritative when documentation and code disagree.

Keep the git checkout and the data disk separate:

- `CODE_ROOT` = `/home/tzhang174/TrafficFlow` — scripts, JSON configs, tests, docs
- `DATA_ROOT` = `/data5/tzhang174/TrafficFlow` — `ocr_data/`, `simulation_data/`,
  `training_data/`, `training_results/` (`data_root.py`, override with
  `TRAFFICFLOW_DATA_ROOT`)

Do not copy scripts onto the data disk. Do not write checkpoints, NPZ windows,
or training logs into the git tree. SUMO `sumocfg` paths stay relative to each
config file.

## Repository-wide working rules

- Preserve existing user changes. The repository commonly contains modified
  configs, generated networks, simulation results, and experiment outputs.
- Resolve data paths through `data_root.py`. The earlier Xuancheng package
  remains at `xuancheng26/training_data/`. The 3-second-phase package is
  `xuancheng26/training_data_3s/`. Fixed penetrations live in
  `training_data_3s/datasets/fixed_uniform/sampling_seeds/`; `sampling_seeds`
  remains a compatibility symlink to that directory. Shared split, node order,
  and the shared target scaler stay in `training_data_3s/shared/`. The current
  four road states are `independent_fcd_v1`: events for inflow/outflow and FCD
  boundary snapshots for accumulation. The retired cumsum stock is
  `archive/cumsum_stock_v1/`. Variable-penetration datasets are
  `training_data_3s/datasets/<dataset_id>/` and carry their own input scaler.
  Train on sampling seed 42. Validation and test score seeds 42–46 and select
  checkpoints by the mean raw MAE. `sampling_seed`, `schedule_seed`, and
  `model_seed` are different. New 3-second results go to
  `training_results_3s/<model>/physical_independent_fcd_v1/datasets/<dataset_id>/in_<input>__out_<target>/<observation_key>/seed_<model_seed>/`.
  Older result directories stay where they are.
  `reimplementation/` does not generate datasets. Never write those artifacts
  under `CODE_ROOT`, and do not overwrite an existing result directory.
- Keep ID-like values such as VINs, OSM IDs, SUMO edge IDs, junction IDs, and
  lane IDs as strings. Do not coerce them to integers.
- Preserve deterministic seeds and stable output ordering. The default seed is
  usually `42`; changing it changes the experiment and must be explicit.
- Prefer streaming or incremental parsing for large XML/CSV files. Do not load
  FCD files, SUMO state collections, or the full network into memory without a
  concrete reason.
- Keep text, JSON, and XML output UTF-8. Preserve existing field names and XML
  structures because downstream scripts consume them directly.
- Keep SUMO config paths relative to the config file. Python/JSON defaults for
  generated data use `data_root.py` (`CODE_ROOT` vs `DATA_ROOT`); do not hard-code
  extra machine paths in model code.
- Update the relevant README or `AGENTS.md` when changing pipeline stages,
  default inputs, output locations, or experiment semantics.

## Source files versus generated artifacts

Python scripts, Markdown files, hand-maintained SUMO templates, and experiment
configuration are source. Large `.rou.xml`, `.trips.xml`, FCD CSV, result XML,
plots, logs, and `state_*.xml.gz` files are usually data or generated artifacts.

- Do not reformat, regenerate, delete, or replace large artifacts unless the
  task explicitly requires it.
- Never use a generated output as the only place to implement a behavioral
  change; update the generator or source configuration.
- Before running a script, inspect whether it overwrites outputs or recreates a
  directory. Full simulations are expensive and some runners are destructive.

## Validation

There is no repository-wide automated test suite. Validate proportionally:

1. Parse changed Python files or run their `--help` entry point when available.
2. Exercise changed pure functions with a small synthetic fixture.
3. Parse changed JSON/XML files with the standard library.
4. Run SUMO, Tesseract, network downloads, or full FCD processing only when the
   task requires those external or expensive operations.
5. Inspect `git diff --check` and ensure generated-data churn is intentional.

Do not report an end-to-end simulation as verified unless SUMO actually ran and
its exit code and log were checked.
