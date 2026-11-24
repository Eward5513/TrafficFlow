# RF baseline

Othman et al., ITSC 2022, adapted to Xuancheng26. This is not a neural model
and does not use a GPU or the road adjacency.

- Read speed features from
  `datasets/<dataset_id>/sampling_seeds/seed_<s>/model_prepared/RF/<key>/<date>.npz`.
  Static road attributes stay in `<training-root>/config/RF/road_attributes.csv`.
  Do not reuse one penetration's speed cube for another sampling seed.
- Do not rewrite partial flow, full flow, FCD, or other models' results.
- Train on `--train-sampling-seed` (default 42, one of 42–46). That seed
  selects partial flow, the prepared RF speed cube, and, for `all_sp_c`,
  the daily raw speed and snapshot count. `--seed` remains the model seed.
  Validate and test seeds 42–46 separately, then average the raw metrics.
  Fit the feature scaler on that training seed's rows only. Daily reports
  labeled sampling seed 42 are the evaluation-seed-42 tables.
- Write runs under `<results-root>/in_<input>__out_<target>/<rate>/seed_<model_seed>/`.
  Pass `--run-id` only when nesting a named experiment beside that layout.
- After predictions and metrics are saved, delete the fitted scaler, K-Means,
  forest, and LAD coefficients. Keep `model_config.json` and write
  `model_discarded.json` with file sizes and those parameters. Keep
  `predictions/` and `metrics/`. Do not delete a forest that has no test
  predictions yet.
- A single state predicts itself from seven features. Column 0 is that
  state's partial count. `all` input uses ten features: the four partial
  counts, then speed, normalized speed, lanes, speed limit, weekday, and hour.
  `all_sp_c` keeps those ten columns and appends `average_speed_raw` and
  `speed_observation_count` from `traffic_observations`. `fcd_average_speed`
  stays the existing RF speed.
- The LAD line always uses the partial count of the target state. `all`
  output fits four independent estimators. Do not make one joint 4-D head.
- Keep the 20/4/5 manifest and supervised bins 11–287.
- Fit the scaler, K-Means, LAD coefficients, and forest on training days only.
- Soft vote with `classes_` aligned to cluster ids. Do not replace LAD with OLS.
