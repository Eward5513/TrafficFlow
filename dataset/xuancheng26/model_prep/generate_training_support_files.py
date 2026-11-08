#!/usr/bin/env python3
"""Validate prepared support assets and refresh model configs for the sampling protocol.

Graph generation remains in the dedicated graph/relation generators. This tool
never merges windows, refits scalers, or rewrites labels. Existing paper-specific
optimizer and architecture settings are authoritative.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from data_root import xuancheng26_training_3s_root, xuancheng26_model_results_3s_dir, sampling_window_root
from reimplementation.common.data.sampling_seeds import configure_sampling, validate_sampling_package
from reimplementation.common.utils.atomic_io import atomic_write_json


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-root',type=Path,default=xuancheng26_training_3s_root())
    p.add_argument('--write-configs',action='store_true',help='Update only existing default model configs; never historical run configs.')
    p.add_argument('--rates',nargs='+',default=['p05','p10','p20','p30','p40','p50','p70'])
    args=p.parse_args(argv);root=args.training_root
    win=sampling_window_root(root/'sampling_seeds')
    info=validate_sampling_package(win,n_his=12,num_nodes=1406,rates=args.rates)
    models=[]
    for path in sorted((root/'config').glob('*/r_only_*.json')):
        cfg=json.loads(path.read_text());name=cfg['model_name']
        cfg=configure_sampling({**cfg,'sampling_root':str(root/'sampling_seeds')})
        cfg.update(input_type=cfg.get('input_type','inflow'),target_type=cfg.get('target_type','inflow'),
                   output_root=str(xuancheng26_model_results_3s_dir(name)),checkpoint_monitor='mean_validation_mae',
                   dataset_metadata=str(root/'shared/state_window_manifest.json'))
        for key in ('adjacency','adjacency_pickle','weighted_adjacency','spatial_adjacency','temporal_adjacency','undirected_topology','relations_dir'):
            if cfg.get(key) and not Path(cfg[key]).exists():raise FileNotFoundError(f'{name}: {key}={cfg[key]}')
        if args.write_configs:atomic_write_json(path,cfg)
        models.append(name)
    print(json.dumps(dict(passed=True,models=models,sample_counts=info['sample_counts'],configs_written=args.write_configs),indent=2))


if __name__=='__main__':main()
