# Xuancheng26 STJGCN instructions

This directory ports `reference/STJGCN/PeMSD7` (IEEE TKDE 2024,
https://github.com/zhengchuanpan/STJGCN) to PyTorch. Do not edit
`reference/STJGCN`. The snapshot has no LICENSE file; keep the URL and the
file hashes in `provenance.py`. Only the PeMSD7 width and defaults are used.

The task is last-observed-step reconstruction on the 3-second Xuancheng26
windows. Train on sampling seed 42. Select by the mean raw MAE of seeds
42–46. The training loss still masks zero labels; the reported MAE does not.

```bash
CUDA_VISIBLE_DEVICES=<gpu> python3 reimplementation/stjgcn/train_r_only_stjgcn.py \
  --input-type inflow --target-type inflow --rates 5
```

Results:
`training_results_3s/stjgcn/in_<input>__out_<target>/<rate>/seed_<model_seed>/`.

Predefined graphs are
`training_data_3s/graph/stjgcn/predefined_stjg.npz`.
