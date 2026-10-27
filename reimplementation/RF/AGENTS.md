# RF baseline

Othman et al., ITSC 2022, adapted to Xuancheng26. This is not a neural model
and does not use a GPU or the road adjacency.

- Read auxiliary features from `xuancheng26/training_data/config/RF/`.
- Do not rewrite partial flow, full flow, FCD, or other models' results.
- Write runs only under `xuancheng26/training_results/RF/YYYY_MM_DD/<run_id>/`.
- Keep the 20/4/5 manifest and supervised bins 11–287.
- K-Means and the forest use the seven features. The line uses raw FCD flow only.
- Fit the scaler, K-Means, LAD coefficients, and forest on training days only.
- Soft vote with `classes_` aligned to cluster ids. Do not replace LAD with OLS.
