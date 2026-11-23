# MSTT

MSTT is the project backbone: one temporal attention, one spatial attention,
and one SwiGLU. Finished experiments that are no longer trained from this
tree are recorded in [TRAINING_RESULTS.md](TRAINING_RESULTS.md).

## Layout

```text
MSTT/
├── train.py
├── evaluate.py
├── run_experiments.py
├── run_all_sp_c_state_aux_gate.sh
├── config.py
├── trainer.py
├── configs/default.json
├── configs/experiment_A_separate_value_mlp_projected.json
├── configs/all_sp_c_state_aux_gate_d128_projected_all.json
├── model/
│   ├── embedding.py
│   ├── adaptive_embedding.py
│   ├── attention.py
│   ├── transformer_block.py
│   ├── ffn.py
│   ├── head.py
│   ├── conservation.py
│   ├── auxiliary_gate.py
│   └── mstt.py
├── data/
└── utils/
```

`model/` does not open datasets or branch on task names. The graph is passed
in. A new spatial attention registers in `SPATIAL_ATTENTION`. A new traffic
input is added beside `InputEmbedding`, not inside `mstt.py`.

## Three paths

| Path | Config | Task |
| --- | --- | --- |
| Default single TSF | `configs/default.json` | `I2I` or `All2All` |
| Four value MLPs, shared TSF, projection | `configs/experiment_A_separate_value_mlp_projected.json` | `All2All` |
| Speed and count gate on those four MLPs | `configs/all_sp_c_state_aux_gate_d128_projected_all.json` | `AllSpC2All` |

The default copies the frozen run `single_tsf_adaptive128_100ep`, including
`ffn_hidden_dim=176`. Value is 32, road 16, TOD 8, DOW 8, then a 64-d
adaptive embedding. One T→S→FFN block uses global spatial attention, QK-Norm,
and temporal RoPE. The head is flatten.

The four-state control keeps that block and head. Each state has its own
`Linear(1, 8) → SiLU → Linear(8, 8)`. Conservation projection uses raw-loss
weight 0.5.

The gate uses the same four MLPs and the same block. Standardized
`average_speed_raw` and `speed_observation_count` only build

```text
g = 2 * sigmoid(Linear(32, 32)(SiLU(Linear(2, 32)(aux))))
```

`g` starts at 1 and multiplies the 32-d state embedding before the road,
TOD, and DOW tables. The same JSON is reused for another frequency or
penetration by passing that dataset id and its observation key.
`bash MSTT/run_all_sp_c_state_aux_gate.sh GPU [GPU ...]` launches the four
`p10_p20` datasets.

## Data and selection

Train on sampling seed 42. Validation and test score seeds 42, 43, 44, 45,
and 46. The checkpoint is the lowest mean raw validation MAE. All2All and
AllSpC2All select on the mean validation inflow MAE. Test does not select
the checkpoint. Channel order is inflow, outflow, accumulation_start,
accumulation_end.

Windows live under
`training_data_3s/datasets/<dataset_id>/sampling_seeds/seed_<id>/flow/`.
The graph is `training_data_3s/graph/r_graph.npz`. Results go to
`training_results_3s/datasets/<dataset_id>/physical_<version>/MSTT/in_<input>__out_<target>/<observation_key>/<experiment_id>/seed_<model_seed>/<run_id>/`.
A new run gets its own `run_id`.

```bash
python MSTT/train.py --task I2I --penetration p05
python MSTT/train.py --config MSTT/configs/experiment_A_separate_value_mlp_projected.json \
  --task All2All --dataset-id variable_30min_p5_p15 --penetration p5_p15 --gpu 0
python MSTT/train.py --config MSTT/configs/all_sp_c_state_aux_gate_d128_projected_all.json \
  --task AllSpC2All --dataset-id variable_30min_p5_p15 --penetration p5_p15 --gpu 0
python MSTT/evaluate.py --run-dir <run directory> --splits validation test
```

`--max-train-batches` and `--max-eval-batches` write a smoke run under a
`_smoke` segment. Do not point that at an official experiment directory.

## Unfinished four-TSF run

`backbone_mode=state_independent` is still in the tree because
`state_independent_full_context_independent_tsf_mlp_projected` stopped on
2026-11-16 without test metrics. The finished four-TSF configs stay with
that module. Their numbers, and every removed experiment, are in
[TRAINING_RESULTS.md](TRAINING_RESULTS.md).
