"""Five-sampling selection, metric aggregation, path isolation and CPU model contracts."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data_root import SAMPLING_SEEDS, sampling_window_root, sampling_run_results_dir
from reimplementation.common.data.state_windows import CHANNELS, AlignedStateWindowDataset, apply_state_types
from reimplementation.common.data.r_only_npz_dataset import build_dataloader
from reimplementation.common.sampling_evaluation import evaluate_sampling_predictor, aggregate_sampling_metrics, sampling_history
from reimplementation.common.data.sampling_seeds import configure_sampling, protect_historical_results


def fixture_root(tmp_path):
    root=tmp_path/'sampling_seeds'
    for seed in SAMPLING_SEEDS:
        w=sampling_window_root(root,seed);w.mkdir(parents=True)
        (w/'split_manifest.json').write_text(json.dumps({s:[s] for s in ('train','validation','test')}))
        norm={'fit_split':'train','target':{c:{'mean':0.,'std':1.} for c in CHANNELS}}
        (w/'state_normalization.json').write_text(json.dumps(norm))
        for split in ('train','validation','test'):
            d=w/'p05'/split/split;d.mkdir(parents=True)
            # Deliberately uneven final batch catches mean-of-batch-means bugs.
            x=np.ones((3,12,16,1),dtype=np.float32)*(seed-41)
            x[-1]*=3
            y=np.ones((3,1,16,1),dtype=np.float32)
            np.savez(d/'inflow.npz',x=x,x_raw=x,y=y,y_raw=y,day_index=np.zeros(3,dtype=np.int64),window_start_slot=np.arange(3),target_slot=np.arange(3)+11)
    return root


def test_five_sampling_protocol_and_selection(tmp_path):
    root=fixture_root(tmp_path)
    cfg=configure_sampling({'sampling_root':str(root),'model_seed':123})
    assert cfg['seed']==123 and cfg['train_sampling_seed']==42
    assert cfg['data_root']==str(sampling_window_root(root,42))
    ds=AlignedStateWindowDataset(Path(cfg['data_root']),'p05','validation',input_type='inflow',target_type='inflow',num_nodes=16)
    loader=build_dataloader(ds,batch_size=2,shuffle=False)
    result=evaluate_sampling_predictor(loader,lambda b:b['x'][:,-1:],mean_y=0.,std_y=1.,device=torch.device('cpu'))
    report=result['sampling_metrics']
    expected=[float(np.abs(ds.y_raw-(np.ones_like(ds.y_raw)*(s-41)*np.array([1,1,3])[:,None,None,None])).mean()) for s in SAMPLING_SEEDS]
    np.testing.assert_allclose([report['per_seed'][str(s)]['mae'] for s in SAMPLING_SEEDS],expected)
    assert report['mean']['mae']==pytest.approx(np.mean(expected))
    assert report['std']['mae']==pytest.approx(np.std(expected,ddof=1))
    history=[{'epoch':0,**sampling_history(report)},{'epoch':1,'mean_validation_mae':100.,'test_mae':0.}]
    assert min(history,key=lambda r:r['mean_validation_mae'])['epoch']==0
    assert len(ds)==3
    assert not any('test' in k for k in sampling_history(report))


def test_rmse_is_mean_of_seed_rmse():
    per={str(s):{'mae':float(s-41),'rmse':float(s-41)} for s in SAMPLING_SEEDS}
    report=aggregate_sampling_metrics(per)
    assert report['mean']['rmse']==3.
    assert report['mean']['rmse']!=np.sqrt(np.mean(np.arange(1,6)**2))


def test_paths_and_historical_protection(tmp_path):
    path=sampling_run_results_dir(tmp_path,'all','all','p05',123)
    assert path==tmp_path/'in_all__out_all/p05/seed_123'
    path.mkdir(parents=True);(path/'resolved_config.json').write_text('{"seed":42}')
    with pytest.raises(RuntimeError,match='historical'):protect_historical_results(path)
    with pytest.raises(ValueError):configure_sampling({'train_sampling_seed':43})


MODELS=('MSTT','dcrnn','graph_wavenet','stgcn','stsgcn','stfgnn','pdformer','stid','staeformer','s2s_gcn_lstm','tlast')


def model_case(name,kind='inflow'):
    from data_root import xuancheng26_model_config
    from MSTT.config import default_config
    n=16
    if name=='MSTT': cfg=default_config()
    else:
        file=Path('reimplementation/tlast/configs/r_only_tlast.json') if name=='tlast' else xuancheng26_model_config(name)
        cfg=json.loads(file.read_text())
        cfg=apply_state_types(cfg,SimpleNamespace(input_type=kind,target_type=kind),model_name=name)
    cfg.update(num_nodes=n,device='cpu',batch_size=2)
    c=4 if kind=='all' else 1
    x=torch.randn(2,12,n,c)
    batch=dict(x=x,x_raw=x,y=torch.ones(2,1,n,c),y_raw=torch.ones(2,1,n,c),
               window_start_slot=torch.tensor([0,1]),target_slot=torch.tensor([11,12]),day_index=torch.tensor([0,1]))
    adj=np.zeros((n,n),np.float32)
    for i in range(n-1):adj[i,i+1]=1
    sym=adj+adj.T
    if name=='MSTT':
        from MSTT.model.graph import build_spatial_graph
        from MSTT.model import MSTT,MSTTConfig
        from MSTT.data.dataset import time_features
        model=MSTT(MSTTConfig.from_mapping(cfg,in_channels=c,out_channels=c),build_spatial_graph(adj))
        def predict():return model(x,*time_features(batch,12)).unsqueeze(1)
    else:
        e=importlib.import_module(f'reimplementation.{name}.engine')
        if name in ('dcrnn','graph_wavenet'):model=e.build_model(cfg,[adj,adj.T])
        elif name=='stgcn':model=e.build_model(cfg,np.tile(np.eye(n,dtype=np.float32),(1,int(cfg['Ks']))))
        elif name=='stsgcn':model=e.build_model(cfg,sym)
        elif name=='stfgnn':model=e.build_model(cfg,sym,sym,temporal_graph_is_test_only=True)
        elif name=='s2s_gcn_lstm':model=e.build_model(cfg,sym)
        elif name=='pdformer':
            from reimplementation.pdformer.model.pdformer import in_memory_relations_from_topology
            rel=in_memory_relations_from_topology(sym,far_mask_delta=7,dtw_delta=5,lape_dim=cfg.get('lape_dim',8),n_cluster=16,s_attn_size=3,output_dim=1)
            model=e.build_model(cfg,rel)
        else:model=e.build_model(cfg)
        def predict():
            if name=='stid':
                day=dict(day_of_week_source=cfg.get('day_of_week_source','original_sequential_index_mod_7'),weekday_mapping=None,all_days=None)
                return model(e.assemble_batch_history(batch,cfg,**day),**e.identity_kwargs(batch,cfg,day))
            if name=='staeformer':
                day=dict(day_of_week_source=cfg.get('day_of_week_source','original_pems_npz_dow'),weekday_mapping=None,all_days=None)
                p=e.assemble_batch_payload(batch,cfg,**day)
                return model(p['history'],**e.model_kwargs_from_payload(p,torch.device('cpu')))
            if name=='pdformer':return model(e.assemble_batch_x(batch,cfg,weekday_mapping=None,all_days=None))
            if name=='tlast':return e.predict_batch(model,batch,cfg,0.,1.)
            return model(x)
    return model,predict


@pytest.mark.parametrize('name',MODELS)
@pytest.mark.parametrize('kind',['inflow','all'])
def test_model_cpu_forward(name,kind):
    if name=='pdformer' and kind=='all':
        with pytest.raises(Exception,match='PDFormer'):
            model_case(name,kind)
        return
    model,predict=model_case(name,kind);model.eval()
    with torch.no_grad(): y=predict()
    assert y.shape==(2,1,16,4 if kind=='all' else 1)
    assert torch.isfinite(y).all() and y.device.type=='cpu'


def test_mstt_conservation_modes():
    from MSTT.config import default_config
    from MSTT.model import MSTT,MSTTConfig
    from MSTT.model.graph import build_spatial_graph
    from MSTT.model.conservation import conservation_residual
    cfg=default_config();cfg['num_nodes']=4
    for mode in ('soft','projected'):
        cfg['conservation_mode']=mode
        model=MSTT(MSTTConfig.from_mapping(cfg,in_channels=4,out_channels=4),build_spatial_graph(np.eye(4)),target_mean=np.zeros(4),target_std=np.ones(4))
        out=model(torch.randn(1,12,4,4),torch.arange(12)[None],torch.zeros(1,12,dtype=torch.long))
        assert out.shape==(1,4,4) and torch.isfinite(out).all()
        out.square().mean().backward()
        if mode=='projected':assert torch.max(torch.abs(conservation_residual(out)))<1e-5


@pytest.mark.parametrize('name',[n for n in MODELS if n!='MSTT'])
def test_real_engine_one_synthetic_epoch(name,tmp_path,monkeypatch):
    """Exercise each ORIGINAL engine with tiny fixtures, including restore/save; no formal training."""
    import inspect
    from data_root import xuancheng26_model_config
    from reimplementation.common.utils.structured_logging import JsonlLogger
    root=fixture_root(tmp_path)
    path=Path('reimplementation/tlast/configs/r_only_tlast.json') if name=='tlast' else xuancheng26_model_config(name)
    cfg=json.loads(path.read_text());cfg.update(sampling_root=str(root),model_seed=42)
    cfg=apply_state_types(cfg,SimpleNamespace(input_type='inflow',target_type='inflow'),model_name=name)
    cfg.update(num_nodes=16,device='cpu',batch_size=2,epochs=1,rates=['p05'],num_workers=0)
    # The underlying optimizer/loss/scheduler remain the original model choices.
    e=importlib.import_module(f'reimplementation.{name}.engine')
    n=16;adj=np.zeros((n,n),np.float32)
    for i in range(n-1):adj[i,i+1]=1
    sym=adj+adj.T
    data=dict(mean_y_full=0.,std_y_full=1.,sample_counts={s:3 for s in ('train','validation','test')},
              node_ids=[str(i) for i in range(n)],file_hashes={'p05/train.npz':'synthetic'},normalization_sha256='synthetic',
              adjacency=sym,graph_sha256='synthetic',edge_weight='distance')
    graph=dict(graph_sha256='synthetic',supports=[adj,adj.T],spatial_topology=sym,temporal_adjacency=sym,
               temporal_sha256='synthetic',cheb_kernel=np.tile(np.eye(n,dtype=np.float32),(1,int(cfg.get('Ks',3)))))
    if name=='pdformer':
        from reimplementation.pdformer.model.pdformer import in_memory_relations_from_topology
        graph.update(in_memory_relations_from_topology(sym,far_mask_delta=7,dtw_delta=5,lape_dim=cfg.get('lape_dim',8),n_cluster=16,s_attn_size=3,output_dim=1))
        # Test fixture stays exclusively in tmp_path; production checks remain active.
        graph['relations_are_test_only']=False
    with JsonlLogger(tmp_path/'log.jsonl') as logger:
        kw=dict(rate='p05',data_root=sampling_window_root(root,42),output_root=tmp_path/'results',data_info=data,logger=logger)
        parameters=inspect.signature(e.run_single_rate).parameters
        if 'graph_info' in parameters:kw['graph_info']=graph
        if 'project_root' in parameters:kw['project_root']=Path.cwd()
        e.run_single_rate(cfg,**kw)
    output=sampling_run_results_dir(tmp_path/'results','inflow','inflow','p05',42)
    for split in ('validation','test'):
        report=json.loads((output/f'{split}_sampling_metrics.json').read_text())
        assert report['sampling_seeds']==list(SAMPLING_SEEDS)
        assert report['mean']['mae']==pytest.approx(np.mean([m['mae'] for m in report['per_seed'].values()]))
        for seed in SAMPLING_SEEDS:assert (output/'predictions'/f'seed_{seed}'/f'{split}.npz').is_file()
    history=json.loads((output/'training_history.json').read_text())
    assert len(history)==1 and 'mean_validation_mae' in history[0]
    assert (output/'best_checkpoint.pt').is_file() and (output/'last_checkpoint.pt').is_file()


def test_mstt_one_synthetic_epoch(tmp_path):
    from MSTT.config import default_config
    from MSTT.trainer import train_task
    root=fixture_root(tmp_path)
    nodes='node_index,edge_id\n'+''.join(f'{i},{i}\n' for i in range(16))
    for seed in SAMPLING_SEEDS:
        w=sampling_window_root(root,seed)
        manifest={s:[s] for s in ('train','validation','test')}
        manifest.update(n_his=12,target_mode='last-observed-step',actual_sample_counts={s:3 for s in ('train','validation','test')})
        (w/'split_manifest.json').write_text(json.dumps(manifest))
        norm=json.loads((w/'state_normalization.json').read_text())
        norm.update(channels=list(CHANNELS),training_days=['train'],input={'p05':{c:{'mean':0.,'std':1.} for c in CHANNELS}})
        (w/'state_normalization.json').write_text(json.dumps(norm))
        (w/'state_window_manifest.json').write_text(json.dumps(dict(channels=list(CHANNELS),num_nodes=16,n_his=12,target_mode='last-observed-step')))
        (w/'validation_summary.json').write_text('{"overall_validation_passed":true,"failures":[]}')
        (w/'node_mapping.csv').write_text(nodes)
    (tmp_path/'nodes.csv').write_text(nodes)
    adj=np.zeros((16,16),np.float32)
    for i in range(15):adj[i,i+1]=1
    np.savez(tmp_path/'graph.npz',adjacency=adj,edge_ids=np.array([str(i) for i in range(16)]))
    cfg=default_config();cfg.update(sampling_root=str(root),num_nodes=16,device='cpu',epochs=1,batch_size=2,eval_batch_size=2,
                                    graph_path=str(tmp_path/'graph.npz'),r_nodes=str(tmp_path/'nodes.csv'))
    train_task(cfg,task='I2I',rate='p05',output_root=tmp_path/'results')
    out=sampling_run_results_dir(tmp_path/'results','inflow','inflow','p05',42)
    history=json.loads((out/'training_history.json').read_text())
    report=json.loads((out/'validation_sampling_metrics.json').read_text())
    assert history[0]['selection_value']==pytest.approx(report['mean']['mae'],rel=1e-4)
    assert history[0]['mean_validation_mae']==pytest.approx(report['mean']['mae'],rel=1e-4)
    assert (out/'test_sampling_metrics.json').is_file()
    for seed in SAMPLING_SEEDS:assert (out/'predictions'/f'seed_{seed}'/'test.npz').is_file()


def test_rf_minimal_fit_and_five_sampling(tmp_path,monkeypatch):
    from reimplementation.RF import pipeline as rf
    class TinyRFData:
        def __init__(self,sampling_seed=42):
            self.seed=sampling_seed;self.edges=['0','1','2'];self.days={s:[s] for s in ('train','validation','test')}
        def day_cube(self,rate,day,input_type,target_type):
            rng=np.random.default_rng(12)
            x=rng.uniform(1,5,(4,3,7)).astype(np.float32)
            y=2*x[...,0]+1
            x[...,0]*=self.seed/42
            return x,y
        def stack(self,rate,days,input_type,state):
            assert self.seed==42 and days==['train']
            x,y=self.day_cube(rate,days[0],input_type,state);return x.reshape(-1,7),y.ravel()
    monkeypatch.setattr(rf,'XuanchengRFData',TinyRFData)
    monkeypatch.setattr(rf,'N_TARGETS',4)
    monkeypatch.setattr(rf,'RESULTS_ROOT',tmp_path)
    out=tmp_path/'RF/in_inflow__out_inflow/p05/seed_42'
    rf.train_penetration(dict(penetration='p05',rate_dir=str(out),input_type='inflow',target_type='inflow',
                              seed=42,n_jobs=1,n_clusters=2,n_estimators=2,kmeans_n_init=1,force=False,evaluate_only=False))
    assert rf._complete(out,False)
    assert (out/'model/scaler.joblib').is_file()
    for split in ('validation','test'):
        report=json.loads((out/f'{split}_sampling_metrics.json').read_text())
        assert report['sampling_seeds']==list(SAMPLING_SEEDS)
        assert report['std_ddof']==1
    assert (out/'best_checkpoint.pt').is_file() and (out/'last_checkpoint.pt').is_file()
