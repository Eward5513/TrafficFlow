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
