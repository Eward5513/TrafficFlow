#!/usr/bin/env python3
"""Compare every retired input to its surviving payload before cleanup."""
import json
import os
from pathlib import Path
import numpy as np
from migrate_sampling_layout import entries,candidates,sha,link_verified,atomic_write_json,xuancheng26_training_3s_root


def main():
    root=xuancheng26_training_3s_root()
    canonical=root/'sampling_seeds/seed_42/flow/window_data'
    manifest=json.loads((root/'shared/split_manifest.json').read_text())
    inodes={}
    for base in [root/'shared',*[root/'sampling_seeds'/f'seed_{s}'/n for s in range(42,47) for n in ('sampled_vehicle','flow')]]:
        for p in entries(base):
            if not p.is_symlink():
                st=p.stat();inodes[st.st_dev,st.st_ino]=str(p)
    records={}
    for directory in candidates(root):
        if not directory.exists():continue
        for p in entries(directory):
            if p.is_symlink():
                records[str(p)]=dict(method='obsolete_alias',target=os.readlink(p));continue
            st=p.stat();target=inodes.get((st.st_dev,st.st_ino))
            if target:
                records[str(p)]=dict(method='same_inode_verified_migration',new_path=target);continue
            if p.suffix=='.npz' and p.is_relative_to(root/'window_data'):
                parts=p.relative_to(root/'window_data').parts
                rate=parts[0]
                if len(parts)==2:
                    split=p.stem;days=manifest[split]
                elif len(parts)==3:
                    split=parts[1];days=[p.stem]
                else:raise RuntimeError(f'unexpected retired NPZ {p}')
                destinations=[canonical/rate/split/day/'inflow.npz' for day in days]
                with np.load(p,allow_pickle=False) as old:
                    # The merged legacy file and dated canonical files must agree for EVERY array.
                    for key in old.files:
                        values=old[key];offset=0
                        for dest in destinations:
                            with np.load(dest,allow_pickle=False) as new:
                                a=new[key]
                                if not np.array_equal(values[offset:offset+len(a)],a):raise RuntimeError(f'mismatch {p} {key} {dest}')
                                offset+=len(a)
                        if offset!=len(values):raise RuntimeError(f'count mismatch {p}')
                records[str(p)]=dict(method='all_arrays_exact',new_paths=list(map(str,destinations)),sha256=sha(p))
            elif p.suffix in ('.json','.csv'):
                # Summaries and scalers are small provenance records, not disposable research evidence.
                dest=root/'shared/provenance/retired_layout'/p.relative_to(root)
                checksum=link_verified(p,dest)
                records[str(p)]=dict(method='provenance_hardlink_sha256',new_path=str(dest),sha256=checksum)
            else:raise RuntimeError(f'unclassified old input {p}')
        print(f'verified {directory}',flush=True)
    atomic_write_json(root/'architecture_audit/retired_input_verification.json',dict(passed=True,files=records,
                      protected=['partial_flow/m-level','partial_flow/mr-level','full_flow','graph','config','all historical results']))
    print(f'PASS {len(records)} entries',flush=True)


if __name__=='__main__':main()
