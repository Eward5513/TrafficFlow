# RF baseline

CPU reproduction of Othman et al., IEEE ITSC 2022, adapted to Xuancheng26.
K-Means (50) and a 50-tree random forest use seven standardised features.
Each cluster has a univariate least-absolute-deviations line on raw FCD flow.
The prediction is a soft vote. Number of lanes replaces functional class, and
the SUMO speed limit replaces capacity.

Code stays in this directory. Results go to
`/data5/tzhang174/TrafficFlow/xuancheng26/training_results/RF/YYYY_MM_DD/<run_id>/`.

```bash
python3 reimplementation/RF/run_rf.py --self-test
python3 reimplementation/RF/run_rf.py --audit-only
python3 reimplementation/RF/run_rf.py --parallel-models 7 --n-jobs-per-model 8
python3 reimplementation/RF/run_rf.py --penetrations p05
python3 reimplementation/RF/run_rf.py --evaluate-only --run-id <id>
python3 reimplementation/RF/run_rf.py --force --run-id <id>
```

Supervised bins are 11–287. The target is the current full 5-minute entry count.
