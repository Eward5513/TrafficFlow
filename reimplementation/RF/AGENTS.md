# RF baseline

Othman et al., ITSC 2022, adapted to Xuancheng26. This is not a neural model
and does not use a GPU or the road adjacency.

- Read auxiliary features from `<training-root>/config/RF/`. The 3s root is
  `xuancheng26/training_data_3s`.
- Do not rewrite partial flow, full flow, FCD, or other models' results.
- Write runs under `<results-root>/in_<input>__out_<target>/<rate>/`.
  Pass `--run-id` to nest a named experiment under that pair directory.
- After predictions and metrics are saved, delete the fitted scaler, K-Means,
  forest, and LAD coefficients. Keep `model_config.json` and write
  `model_discarded.json` with file sizes and those parameters. Keep
  `predictions/` and `metrics/`. Do not delete a forest that has no test
  predictions yet.
- A single state predicts itself from seven features. Column 0 is that
  state's partial count. `all` input uses ten features: the four partial
  counts, then speed, normalized speed, lanes, speed limit, weekday, and hour.
- The LAD line always uses the partial count of the target state. `all`
  output fits four independent estimators. Do not make one joint 4-D head.
- Keep the 20/4/5 manifest and supervised bins 11–287.
- Fit the scaler, K-Means, LAD coefficients, and forest on training days only.
- Soft vote with `classes_` aligned to cluster ids. Do not replace LAD with OLS.
