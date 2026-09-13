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
```

Read `README.md` for the conceptual workflow, but treat the current scripts and
their constants as authoritative when documentation and code disagree.

## Repository-wide working rules

- Preserve existing user changes. The repository commonly contains modified
  configs, generated networks, simulation results, and experiment outputs.
- Resolve data paths with `pathlib.Path` relative to `__file__` unless an
  existing script deliberately uses another convention.
- Keep ID-like values such as VINs, OSM IDs, SUMO edge IDs, junction IDs, and
  lane IDs as strings. Do not coerce them to integers.
- Preserve deterministic seeds and stable output ordering. The default seed is
  usually `42`; changing it changes the experiment and must be explicit.
- Prefer streaming or incremental parsing for large XML/CSV files. Do not load
  FCD files, SUMO state collections, or the full network into memory without a
  concrete reason.
- Keep text, JSON, and XML output UTF-8. Preserve existing field names and XML
  structures because downstream scripts consume them directly.
- Do not introduce machine-specific absolute paths. SUMO config paths are
  intentionally relative to the config file or the script working directory.
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
