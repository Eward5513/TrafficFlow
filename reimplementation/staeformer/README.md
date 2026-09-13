# R-only STAEformer

PyTorch port of [STAEformer](https://github.com/XDZhelheim/STAEformer)
(Liu et al., CIKM 2023) for the TrafficFlow R-only reconstruction task:

12 five-minute observed inflow windows → full inflow at the last observed slot.

The original model concatenates feature, periodicity, and spatio-temporal
adaptive embeddings, then runs all temporal Transformer layers before all
spatial Transformer layers. It does **not** use a road graph.

Original traffic config: `reference/STAEformer/model/STAEformer.yaml` PEMS04.

```bash
python3 reimplementation/staeformer/tests/run_and_report.py
python3 reimplementation/staeformer/train_r_only_staeformer.py --smoke-test-only
# later, official 7-rate training (do not run during this migration):
python3 reimplementation/staeformer/train_r_only_staeformer.py --config reimplementation/staeformer/configs/r_only_staeformer.json
```
