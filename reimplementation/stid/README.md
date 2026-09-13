# R-only STID

PyTorch port of [STID](https://github.com/zezhishao/STID) (Shao et al., CIKM 2022)
for the TrafficFlow R-only reconstruction task:

12 five-minute observed inflow windows → full inflow at the last observed slot.

The original model is spatial/temporal identity embeddings plus residual 1×1
MLPs. It does **not** use a road graph. This port keeps that boundary.

Original traffic config: `reference/STID/stid/PEMS04.py`.

```bash
python3 reimplementation/stid/tests/run_and_report.py
python3 reimplementation/stid/train_r_only_stid.py --smoke-test-only
# later, official 7-rate training (do not run during this migration):
python3 reimplementation/stid/train_r_only_stid.py --config reimplementation/stid/configs/r_only_stid.json
```
