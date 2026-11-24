"""Evaluate a baseline checkpoint on all five validation/test samplings; never train."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from data_root import SAMPLING_SEEDS
from reimplementation.common.data.sampling_seeds import configure_sampling
from reimplementation.common.data.state_windows import open_split_dataset,read_state_target_scaler
from reimplementation.common.data.r_only_npz_dataset import build_dataloader, load_json
from reimplementation.common.model_adapters import build_predictor
from reimplementation.common.evaluation import evaluate_loader


def _checkpoint_train_seed(payload: object) -> int | None:
    if not isinstance(payload, dict):
        return None
    if payload.get('train_sampling_seed') is not None:
        return int(payload['train_sampling_seed'])
    nested = payload.get('config')
    if isinstance(nested, dict) and nested.get('train_sampling_seed') is not None:
        return int(nested['train_sampling_seed'])
    return None


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--model',required=True)
    p.add_argument('--rate',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--splits',nargs='+',choices=['validation','test'],default=['validation','test'])
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise RuntimeError('evaluation output must be empty')
    from data_root import assert_training_results_output
    assert_training_results_output(args.output_dir)
    stored=json.loads((args.run_dir/'resolved_config.json').read_text())
    cfg=configure_sampling(stored, model_name=args.model, allowed_train_sampling_seeds=SAMPLING_SEEDS)
    root=Path(cfg['data_root'])
    train_seed=int(cfg['train_sampling_seed'])
    if root.parent.parent.name!=f'seed_{train_seed}':
        raise RuntimeError(f'resolved training window is {root.parent.parent.name}, expected seed_{train_seed}')
    device=torch.device(args.device)
    model,predict=build_predictor(args.model,cfg,device)
    ckpt=torch.load(args.run_dir/'best_checkpoint.pt',map_location=device,weights_only=False)
    checkpoint_seed=_checkpoint_train_seed(ckpt)
    if checkpoint_seed is not None and checkpoint_seed!=train_seed:
        raise RuntimeError(f'checkpoint train_sampling_seed={checkpoint_seed} != resolved {train_seed}')
    model.load_state_dict(ckpt['model_state_dict']);model.eval()
    mean,std=read_state_target_scaler(root,cfg.get('target_type','inflow'))
    manifest=load_json(root/'split_manifest.json')
    for split in args.splits:
        count=int(manifest[f'{split}_samples_per_rate'])
        ds=open_split_dataset(root,args.rate,split,cfg,expected_count=count)
        loader=build_dataloader(ds,batch_size=int(cfg.get('eval_batch_size',cfg['batch_size'])),shuffle=False)
        evaluate_loader(loader,predict,mean_y=mean,std_y=std,device=device,output_dir=args.output_dir,split=split,expected_count=count)


if __name__=='__main__':main()
