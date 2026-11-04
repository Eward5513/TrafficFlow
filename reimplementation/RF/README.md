# RF baseline

CPU reproduction of Othman et al., IEEE ITSC 2022, adapted to Xuancheng26.
K-Means (50) and a 50-tree random forest use standardised features.
Each cluster has a univariate least-absolute-deviations line on the partial
count of the target state. The prediction is a soft vote. Number of lanes
replaces functional class, and the SUMO speed limit replaces capacity.

A single state keeps the original seven features and must predict itself.
`all` input adds the other three partial counts. `all` output trains four
independent estimators. Results go to
`<results-root>/in_<input>__out_<target>/<rate>/`.

Once a penetration has written its test predictions and metrics, the run
deletes the fitted scaler, K-Means, random forest, and LAD coefficients.
`model_discarded.json` in that penetration directory records the file sizes,
model settings, cluster centers, and LAD coefficients. Predictions and metrics
stay. Starting the same task again skips training and removes a forest left
by an older run. `--evaluate-only` cannot reload a forest that was removed.

```bash
python3 reimplementation/RF/run_rf.py --self-test
python3 reimplementation/RF/run_rf.py --audit-only \
  --training-root /data5/tzhang174/TrafficFlow/xuancheng26/training_data_3s \
  --input-type outflow --target-type outflow
python3 reimplementation/RF/run_rf.py \
  --training-root /data5/tzhang174/TrafficFlow/xuancheng26/training_data_3s \
  --results-root /data5/tzhang174/TrafficFlow/xuancheng26/training_results_3s/RF \
  --input-type all --target-type inflow --penetrations p05
```

Supervised bins are 11–287. The target is the current full 5-minute count.
