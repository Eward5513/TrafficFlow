#!/usr/bin/env python3
"""Read-only CPU acceptance audit for the canonical five-sampling dataset."""
import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
from data_root import SAMPLING_SEEDS, sampling_window_root, xuancheng26_sampling_root, xuancheng26_training_3s_root
from reimplementation.common.data.state_windows import AlignedStateWindowDataset, CHANNELS
from reimplementation.common.utils.atomic_io import atomic_write_json
from reimplementation.common.utils.hashing import sha256_file


def digest(array):
    h=hashlib.sha256()
    for item in array: h.update(np.ascontiguousarray(item).tobytes())
    return h.hexdigest()


def main():
    root=xuancheng26_sampling_root(); shared=root.parent/'shared'
    manifest=json.loads((shared/'split_manifest.json').read_text()); scaler=json.loads((shared/'state_normalization.json').read_text())
    assert scaler['fit_split']=='train' and scaler['training_days']==manifest['train']
    assert tuple(scaler['channels'])==CHANNELS
    rates=('p05','p10','p20','p30','p40','p50','p70'); counts={'train':5540,'validation':1108,'test':1385}
    references={}; inputs={}; rows=[]; paths=0; csv_count=0; ids_count=0
    metadata=('split_manifest.json','state_normalization.json','node_mapping.csv','state_window_manifest.json','validation_summary.json')
    for seed in SAMPLING_SEEDS:
        win=sampling_window_root(root,seed)
        for name in metadata:
            assert sha256_file(win/name)==sha256_file(shared/name),(seed,name)
        for rate in rates:
            for split in counts:
                for day in manifest[split]:
                    for state in (*CHANNELS,'all'):
                        p=win/rate/split/day/f'{state}.npz'; assert p.is_file(),p; paths+=1
                    for state in CHANNELS:
                        p=win.parent/'partial_flow'/rate/day/f'{state}.csv'; assert p.is_file(),p;csv_count+=1
                    assert (win.parents[1]/'sampled_vehicle'/rate/f'{day}.csv').is_file();ids_count+=1
                if rate in ('p05','p70'):
                    data=AlignedStateWindowDataset(win,rate,split,input_type='all',target_type='all',expected_count=counts[split])
                    assert data.x.shape==(counts[split],12,1406,4)
                    assert data.y.shape==(counts[split],1,1406,4)
                    for field in ('x','x_raw','y','y_raw'): assert np.isfinite(getattr(data,field)).all()
                    yh=digest(data.y_raw); xh=digest(data.x_raw)
                    key=(rate,split); references.setdefault(key,yh); assert yh==references[key]
                    inputs.setdefault(key,[]).append(xh)
                    mean=np.array([scaler['target'][c]['mean'] for c in CHANNELS]);std=np.array([scaler['target'][c]['std'] for c in CHANNELS])
                    assert np.max(abs(data.y[:10]*std+mean-data.y_raw[:10]))<1e-3
                    rows.append(dict(seed=seed,rate=rate,split=split,count=len(data),x_shape=list(data.x.shape),y_shape=list(data.y.shape),x_sha256=xh,y_sha256=yh))
                    del data;gc.collect()
                # Every rate gets actual array and channel-order checks (first day each split).
                day=manifest[split][0]
                with np.load(win/rate/split/day/'all.npz') as a:
                    x=a['x'];y=a['y'];xr=a['x_raw'];yr=a['y_raw']
                    assert x.shape==(277,12,1406,4) and y.shape==(277,1,1406,4)
                    for i,c in enumerate(CHANNELS):
                        with np.load(win/rate/split/day/f'{c}.npz') as b:
                            assert np.array_equal(x[...,i:i+1],b['x'])
                            assert np.array_equal(yr[...,i:i+1],b['y_raw'])
                    assert np.array_equal(a['target_slot'],a['window_start_slot']+11)
                    key=('probe',rate,split); yh=digest(yr);references.setdefault(key,yh);assert yh==references[key]
                    inputs.setdefault(key,[]).append(digest(xr))
            print(f'checked seed {seed} {rate}',flush=True)
    assert all(len(set(v))==5 for v in inputs.values()),'accidentally duplicated inputs'
    assert paths==5075 and csv_count==4060 and ids_count==1015
    result=dict(passed=True,full_loads=rows,arrays_checked_all_rates=True,window_files=paths,flow_csv_files=csv_count,
                vehicle_files=ids_count,channel_order=list(CHANNELS),sample_counts=counts,
                scaler_sha256=sha256_file(shared/'state_normalization.json'),five_labels_equal=True,five_inputs_distinct=True)
    atomic_write_json(root.parent/'architecture_audit/reader_audit.json',result)
    print('PASS',flush=True)


if __name__=='__main__':main()
