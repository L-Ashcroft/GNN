"""Input-stratified ID split, with global design groups as indivisible units."""
from collections import Counter
from pathlib import Path
import hashlib
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read
from src.analoggym_campaign import sha
from src.windows_campaign_io import atomic_json


def make_split(index):
    fields=['source_dataset','circuit_family','connectivity_group','topology_id','parameter_stratum']
    ordered=index.sort_values(fields+['design_group','sample_id']).copy()
    sizes=ordered.groupby('design_group').size()
    # 957 shared designs couple strata across datasets: one canonical stratum decides the whole group.
    primary=ordered.drop_duplicates('design_group').copy()
    primary['fine_stratum']=primary[fields].astype(str).agg('|'.join,axis=1)
    counts=primary.groupby('fine_stratum').size()
    sparse=primary.fine_stratum.map(counts)<10
    primary['split_stratum']=primary.fine_stratum
    coarse=primary[fields[:-1]].astype(str).agg('|'.join,axis=1)+'|sparse_parameter_pool'
    primary.loc[sparse,'split_stratum']=coarse.loc[sparse]
    primary['random_key']=primary.design_group.map(lambda x:hashlib.sha256(('stratified42:'+x).encode()).hexdigest())
    roles={};statistics=[]
    for name,part in primary.groupby('split_stratum',sort=True):
        part=part.sort_values('random_key');n=len(part)
        weights=np.array([.8,.1,.1])*n;base=np.floor(weights).astype(int)
        for i in np.argsort(-(weights-base),kind='stable')[:n-int(base.sum())]:base[i]+=1
        if n>=3:
            for i in range(3):
                if base[i]==0:
                    donor=int(base.argmax());base[donor]-=1;base[i]+=1
        labels=['train']*int(base[0])+['validation']*int(base[1])+['test']*int(base[2])
        roles.update(zip(part.design_group,labels))
        statistics.append({'stratum':name,'design_groups':n,'train_groups':int(base[0]),
            'validation_groups':int(base[1]),'test_groups':int(base[2])})
    result=index[['sample_id','design_group','source_dataset','circuit_family','topology_id','connectivity_group','parameter_stratum']].copy()
    result['role']=result.design_group.map(roles)
    assert not result.role.isna().any() and result.groupby('design_group').role.nunique().max()==1
    assert result.sample_id.is_unique
    return result,statistics,{'sparse_canonical_groups':int(sparse.sum()),'coupled_designs':int((sizes>1).sum())}


def main():
    base=ROOT/'data/fusion_v2';index=pd.read_parquet(base/'index.parquet')
    split,stats,extra=make_split(index)
    path=base/'splits/id_stratified_v1.parquet'
    split.to_parquet(path,index=False,compression='zstd')
    readback=pd.read_parquet(path);assert readback.equals(split)
    by_source=readback.groupby(['source_dataset','role']).size().unstack(fill_value=0).to_dict('index')
    by_topology=readback.groupby(['topology_id','role']).size().unstack(fill_value=0).to_dict('index')
    assert all(all(part.get(role,0)>0 for role in ['train','validation','test']) for part in by_topology.values())
    audit={'passed':True,'seed':42,'samples':len(split),'stratification_fields':
        ['source_dataset','circuit_family','connectivity_group','topology_id','parameter_stratum'],
        'counts':split.role.value_counts().to_dict(),'by_source':by_source,'by_topology':by_topology,
        'strata':stats,**extra,'sample_and_group_leakage':False,
        'sparse_policy':'Fewer than 10 canonical design groups: pool parameter cells within same source/function/structure/topology.',
        'coupling_policy':'A shared design is assigned once using its deterministic canonical stratum; groups never split across sources.',
        'ratio_policy':'Largest remainder near 80/10/10 within each group stratum; minimum one each if >=3 groups. Exact row ratios may differ.',
        'index_sha256':sha(base/'index.parquet'),'split_sha256':sha(path),'old_split_modified':False,'labels_used':False}
    atomic_json(base/'splits/id_stratified_v1_audit.json',audit)
    print('Stratified split verified:',audit['counts'],';',len(stats),'strata;',extra)


if __name__=='__main__':main()
