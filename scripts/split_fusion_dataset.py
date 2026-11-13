"""Group-safe evaluation protocols; no fitted normalization or label sharing."""
from pathlib import Path
import sys
import hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read,dump
from src.netlist import topology_hash


def fraction(value):
    return int(hashlib.sha256(('42:'+value).encode()).hexdigest()[:16],16)/2**64


def build():
    base=ROOT/'data/fusion_v2';out=base/'splits';out.mkdir(exist_ok=True)
    index=pd.read_parquet(base/'index.parquet');catalog=read(base/'catalog.json')
    # A design shared between source variants always receives the same role.
    f=index.design_group.map(fraction)
    result=index[['sample_id','design_group','topology_id','connectivity_group']].copy()
    result['role']=np.where(f<.8,'train',np.where(f<.9,'validation','test'))
    # Tiny topologies otherwise easily receive no validation/test observations.
    # Assign globally by design_group so shared variants cannot leak.
    for key,part in result.groupby('topology_id',sort=True):
        if part.design_group.nunique()<3:continue
        for missing in ('validation','test','train'):
            current=result.loc[part.index]
            if missing in set(current.role):continue
            counts=current.groupby('role').design_group.nunique()
            eligible=current.loc[current.role.map(counts)>1,'design_group'].unique()
            chosen=min(eligible,key=fraction)
            result.loc[result.design_group==chosen,'role']=missing
    result.to_parquet(out/'id.parquet',index=False,compression='zstd')
    # Union actual full-port classes with shared conservative MOS-core classes.
    parent={s['topology_id']:s['topology_id'] for s in catalog}
    def root(k):
        while parent[k]!=k:k=parent[k]
        return k
    buckets={}
    for s in catalog:
        key=s['topology_id'];graph=read(ROOT/s['graph'])
        for token in ('full:'+s['connectivity_group'],'mos:'+topology_hash(graph,True)):
            if token in buckets:parent[root(key)]=root(buckets[token])
            else:buckets[token]=key
    groups={k:root(k) for k in parent}
    unique=sorted(set(groups.values()),key=fraction)
    folds={g:i%5 for i,g in enumerate(unique)}
    for fold in range(5):
        data=result.copy();data['holdout_group']=data.topology_id.map(groups)
        held=data.holdout_group.map(folds)==fold
        data['role']=np.where(held,'test',np.where(f<.1,'validation','train'))
        data.to_parquet(out/f'loto_fold_{fold}.parquet',index=False,compression='zstd')
    protocols={'id':{'status':'ready','file':'id.parquet','rule':'design_group hash about 80/10/10, seed 42; minimum one group per role for topologies with >=3 groups'},
        'loto':{'status':'ready','files':[f'loto_fold_{i}.parquet' for i in range(5)],
                'rule':'Hold out union of full-port connectivity and conservative MOS-core hash groups; WL collisions overgroup safely'},
        'cross_dataset':{'status':'blocked_no_verified_shared_targets',
                'reason':'No approved equal physical definitions AND verified test conditions across sources. Candidate files must not be used as comparable-label benchmark.'}}
    for source in sorted(index.source_dataset.unique()):
        data=result.copy();data['role']=np.where(index.source_dataset==source,'test',np.where(f<.1,'validation','train'))
        data.to_parquet(out/f'cross_dataset_candidate_{source}.parquet',index=False,compression='zstd')
    dump(out/'protocols.json',protocols)
    print('Splits written',len(index),'rows',len(unique),'conservative topology groups')


if __name__=='__main__':build()
