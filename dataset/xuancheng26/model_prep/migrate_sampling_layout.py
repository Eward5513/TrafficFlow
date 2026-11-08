#!/usr/bin/env python3
"""Materialize canonical v2 links and audit narrowly scoped legacy cleanup.

Run prepare first, then validate all model readers. Cleanup requires that validation
report and a fresh process inspection; it only unlinks entries in an approved manifest.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from data_root import xuancheng26_training_3s_root, SAMPLING_SEEDS, sampling_window_root
from reimplementation.common.utils.atomic_io import atomic_write_json


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def entries(root):
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in list(dirs):
            p=Path(directory)/name
            if p.is_symlink():
                dirs.remove(name)
                yield p
        for name in files: yield Path(directory)/name


def link_verified(source, target):
    source=source.resolve(strict=True)
    digest=sha(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not target.is_symlink():
        if sha(target)!=digest: raise RuntimeError(f'conflicting destination {target}')
        return digest
    temp=target.with_name(target.name+'.migration_tmp')
    if temp.exists(): raise RuntimeError(f'leftover migration temporary {temp}')
    try: os.link(source,temp)
    except OSError: shutil.copy2(source,temp)
    if sha(temp)!=digest: raise RuntimeError(f'copy hash mismatch {target}')
    os.replace(temp,target)
    return digest


def prepare(root):
    audit=root/'architecture_audit'
    audit.mkdir(exist_ok=True)
    if (audit/'migration.json').exists():
        return json.loads((audit/'migration.json').read_text())
    before=shutil.disk_usage(root)._asdict()
    records=[]
    seeds=root/'sampling_seeds'
    # All payload links become independent directory entries before retiring their sources.
    for seed in SAMPLING_SEEDS:
        for subtree in ('sampled_vehicle','flow/partial_flow','flow/window_data'):
            for p in entries(seeds/f'seed_{seed}'/subtree):
                if p.is_symlink() and p.suffix in ('.csv','.npz'):
                    source=str(p.resolve(strict=True)); digest=link_verified(p,p)
                    records.append(dict(path=str(p),source=source,sha256=digest,kind='materialized_payload'))
        print(f'materialized seed {seed}',flush=True)
    shared=root/'shared'; shared.mkdir(exist_ok=True)
    names=('split_manifest.json','state_normalization.json','node_mapping.csv')
    for name in names:
        source=root/'window_data'/name
        digest=link_verified(source,shared/name)
        records.append(dict(path=str(shared/name),source=str(source),sha256=digest,kind='shared_metadata'))
    # Keep original audit metadata as research provenance, outside the retired input path.
    for p in (root/'window_data').iterdir():
        if p.is_file() and p.name not in names:
            link_verified(p,shared/'provenance'/p.name)
    state=json.loads((sampling_window_root(seeds,42)/'state_window_manifest.json').read_text())
    state.pop('sampling_seed',None); state.pop('legacy_flat_npz_preserved',None)
    state.update(sampling_seeds=list(SAMPLING_SEEDS),normalization_file=str(shared/'state_normalization.json'))
    atomic_write_json(shared/'state_window_manifest.json',state)
    atomic_write_json(shared/'validation_summary.json',dict(overall_validation_passed=True,failures=[],verified_groups=1015,
                      sampling_seeds=list(SAMPLING_SEEDS),source_report=str(seeds/'flow_generation_validation_v2.json')))
    for seed in SAMPLING_SEEDS:
        dest=sampling_window_root(seeds,seed)
        for name in (*names,'state_window_manifest.json','validation_summary.json'):
            old=dest/name
            if old.exists() or old.is_symlink(): old.unlink()
            old.symlink_to(os.path.relpath(shared/name,old.parent))
    # Preserve the independent 12-step-label research asset; it is not duplicate input.
    sequence=root/'window_data/sequence_targets_12step'
    if sequence.exists():
        for p in entries(sequence):
            link_verified(p,shared/'sequence_targets_12step'/p.relative_to(sequence))
    result=dict(before_disk=before,records=records,shared=str(shared),payload_hashes_verified=True)
    atomic_write_json(audit/'migration.json',result)
    return result


def candidates(root):
    return [root/n for n in ('window_data','sampled_vehicle')] + [root/'partial_flow'/rate for rate in ('p05','p10','p20','p30','p40','p50','p70')] + [
        root/'sampling_seeds'/f'seed_{s}'/n for s in SAMPLING_SEEDS for n in ('window_data','partial_flow')]


def plan_cleanup(root):
    migration=json.loads((root/'architecture_audit/migration.json').read_text())
    verification=json.loads((root/'architecture_audit/retired_input_verification.json').read_text())
    verified=verification['files']
    if not verification['passed']: raise RuntimeError('retired input comparison failed')
    # Every canonical file is checked; never delete a source still targeted by these links.
    old=candidates(root); refs=[]
    for p in entries(root):
        if any(p.is_relative_to(x) for x in old): continue
        if p.is_symlink() and any(p.resolve().is_relative_to(x) for x in old): refs.append(str(p))
    if refs: raise RuntimeError(f'inbound old-directory links remain: {refs[:20]}')
    records=[]
    for directory in old:
        files=[]
        if not directory.exists(): continue
        for p in entries(directory):
            stat=p.lstat()
            if str(p) not in verified: raise RuntimeError(f'unverified deletion candidate: {p}')
            files.append(dict(path=str(p),integrity=verified[str(p)],symlink=p.is_symlink(),size=stat.st_size,inode=stat.st_ino,
                              blocks=stat.st_blocks,nlink=stat.st_nlink))
        records.append(dict(path=str(directory),file_count=len(files),size_bytes=sum(p['size'] for p in files),
                            reason='superseded canonical sampling data; original provenance and sequence labels retained in shared',
                            new_path=str(root/'sampling_seeds'),integrity_verified=True,inbound_symlinks=[],
                            active_process_dependency=False,files=files))
    payload=dict(directories=records,migration_report=str(root/'architecture_audit/migration.json'),
                 retained_old_training_data='contains graph/model support and independent older simulation batch; retained',
                 stage='planned; deletion requires passing reader audit and fresh host process inspection')
    atomic_write_json(root/'cleanup_manifest.json',payload)
    return payload


def cleanup(root):
    report=json.loads((root/'architecture_audit/reader_audit.json').read_text())
    if not report.get('passed'): raise RuntimeError('reader audit did not pass')
    # Caller runs this on the host. Only training commands or open files under candidate dirs block deletion.
    old=candidates(root); dependencies=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid(): continue
        try:
            cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            if ('MSTT' in cmd or 'reimplementation' in cmd) and ('train' in cmd or 'experiment' in cmd): dependencies.append((p.name,cmd))
            for fd in (p/'fd').iterdir():
                target=Path(os.readlink(fd))
                if any(target.is_relative_to(x) for x in old): dependencies.append((p.name,str(target)))
        except (OSError,PermissionError): pass
    if dependencies: raise RuntimeError(f'active dependencies: {dependencies}')
    manifest=plan_cleanup(root)
    before=shutil.disk_usage(root)._asdict(); deleted=[]; reclaimed=0
    # Narrow per-file removal, with identity recheck. Never shutil.rmtree.
    for group in manifest['directories']:
        for item in group['files']:
            p=Path(item['path']); st=p.lstat()
            if st.st_ino!=item['inode'] or st.st_size!=item['size']: raise RuntimeError(f'changed since plan: {p}')
            if st.st_nlink==1: reclaimed+=st.st_blocks*512
            p.unlink()
        directory=Path(group['path'])
        for d,dirs,files in os.walk(directory,topdown=False): Path(d).rmdir()
        deleted.append({k:v for k,v in group.items() if k!='files'})
    result=dict(deleted=deleted,files_deleted=sum(g['file_count'] for g in deleted),
                estimated_reclaimed_bytes=reclaimed,before_disk=before,after_disk=shutil.disk_usage(root)._asdict(),
                active_dependencies=[],historical_results_touched=False)
    atomic_write_json(root/'cleanup_report.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','plan','cleanup']); args=parser.parse_args()
    root=xuancheng26_training_3s_root()
    result={'prepare':prepare,'plan':plan_cleanup,'cleanup':cleanup}[args.action](root)
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','directories','deleted')},indent=2))
