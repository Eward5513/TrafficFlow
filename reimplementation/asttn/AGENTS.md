# Xuancheng26 ASTTN instructions

This directory ports `reference/ASTTN_pytorch` `Model_Adp` (CIKM 2022,
https://github.com/yokifly/ASTTN_pytorch). Do not edit `reference/ASTTN_pytorch`.
The snapshot has no LICENSE file; keep the URL and the file hashes in
`provenance.py`.

The task is last-observed-step reconstruction on the 3-second Xuancheng26
windows. Train on sampling seed 42. Select by the mean raw MAE of seeds
42–46. Test uses that checkpoint and does not select it.

```bash
CUDA_VISIBLE_DEVICES=<gpu> python3 reimplementation/asttn/train_r_only_asttn.py \
  --input-type inflow --target-type inflow --rates 5
```

Results:
`training_results_3s/asttn/in_<input>__out_<target>/<rate>/seed_<model_seed>/`.

The frozen node2vec table is
`training_data_3s/graph/asttn/node2vec_r128.npy`.
