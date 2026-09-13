# R-only PDFormer

PyTorch port of [PDFormer](https://github.com/aptx1231/PDFormer) (AAAI 2023)
for the R-only reconstruction task:

```text
12 steps of observed R-edge flow -> full R-edge flow at the last input slot
```

Original code lives in `reference/PDFormer` (LibCity, MIT). This directory
does not vendor LibCity.

Default hyperparameters follow `PeMS04.json` overlaying
`libcity/config/model/traffic_state_pred/PDFormer.json`, except:

- `output_window=1`
- `seed=42`
- `add_day_in_week=false` until a real calendar mapping exists

```bash
python3 reimplementation/pdformer/tests/run_and_report.py
python3 reimplementation/pdformer/train_r_only_pdformer.py --smoke-test-only
```

Do not generate official relation matrices or start 7-rate training unless
explicitly requested.
