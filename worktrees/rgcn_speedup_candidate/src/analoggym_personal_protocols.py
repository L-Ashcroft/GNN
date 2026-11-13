"""Independent target-domain splits and nested few-shot budgets, no fitted scales."""
import hashlib
from pathlib import Path
import random
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .common import ROOT,read
from .analoggym_campaign import atomic_json,inside


def key(value):return hashlib.sha256(('personal42:'+value).encode()).hexdigest()


def nested_order(frame):
    rng=random.Random(42);names=sorted(frame.topology_id.unique());rng.shuffle(names)
    groups={n:frame.loc[frame.topology_id==n].sort_values('sample_id',key=lambda s:s.map(key)).sample_id.tolist() for n in names}
    result=[]
    for i in range(max(map(len,groups.values()))):
        for name in names:
            if i<len(groups[name]):result.append(groups[name][i])
    return result


def finalize(output):
    output=inside(output);manifest=read(output/'manifest.json');snapshot=read(output/'dataset.json')
    frames=[pq.read_table(ROOT/s['index']).to_pandas() for s in snapshot['shards']]
    index=pd.concat(frames,ignore_index=True)
    assert index.sample_id.is_unique and index.design_group.is_unique
    assert len(index)==20000 and index.topology_id.nunique()==10
    assert index.topology_id.value_counts().to_dict()==manifest['production_quotas']
    index.to_parquet(output/'personal_index.parquet',index=False,compression='zstd')
    chunks=[]
    for name,part in index.groupby('topology_id',sort=True):
        part=part.sort_values('design_group',key=lambda s:s.map(key)).copy();n=len(part)
        part['role']=['train']*int(.8*n)+['validation']*int(.1*n)+['test']*(n-int(.8*n)-int(.1*n))
        chunks.append(part)
    split=pd.concat(chunks,ignore_index=True);folder=output/'splits';folder.mkdir(exist_ok=True)
    split.to_parquet(folder/'id.parquet',index=False,compression='zstd')
    assert split.groupby('design_group').role.nunique().max()==1
    train=split.loc[split.role=='train'];ordered=nested_order(train);records=[];previous=set()
    for budget in [0,5,10,20,50,100,200,500,1000]:
        selected=set(ordered[:budget]);assert previous<=selected;previous=selected
        assert not selected&set(split.loc[split.role!='train','sample_id'])
        actual=split.loc[split.sample_id.isin(selected)]
        records.append({'budget':budget,'sample_ids':ordered[:budget],
            'topology_counts':actual.topology_id.value_counts().to_dict()})
    atomic_json(folder/'few_shot_global.json',{'seed':42,'scope':'total target-domain labels across all ten topologies',
        'test_file':'id.parquet','test_role':'test','validation_role':'validation',
        'selection':'Nested deterministic topology round-robin within train only; k<10 does not cover every topology',
        'budgets':records,'normalization_fitted':False})
    by_topology=[]
    for budget in [5,10,20,50,100]:
        ids=[]
        for name,part in train.groupby('topology_id',sort=True):ids+=nested_order(part)[:budget]
        by_topology.append({'budget_per_topology':budget,'total_budget':len(ids),'sample_ids':ids})
    atomic_json(folder/'few_shot_per_topology.json',{'scope':'k labels for EACH topology, not k total',
        'seed':42,'budgets':by_topology,'test_file':'id.parquet','test_role':'test'})
    atomic_json(folder/'protocols.json',{
        'id':{'status':'data_ready','file':'id.parquet','rule':'1600 train / 200 validation / 200 test per topology'},
        'cross_source_zero_shot':{'status':'blocked_pending_verified_shared_targets',
            'public_training':'fusion_v2 training partition only',
            'external_test':'id.parquet role=test; no target-domain fitting or tuning',
            'reason':'Existing 500pF/0.1Hz AnalogGym-derived fixture is not automatically comparable to LOADBench 1pF/1Hz or unpublished FALCON/AICircuit benches; no untrained target head or domain embedding may be counted as a zero-shot predictor.'},
        'few_shot':{'status':'data_ready_model_adapter_and_new_heads_required',
            'budget_files':['few_shot_global.json','few_shot_per_topology.json'],
            'constraints':'Separate target-domain head/semantics and actual testbench inputs; fit all learned statistics only on the declared training/adaptation rows. Validation labels are an additional declared target-domain budget, never claimed as part of k without disclosure.'}})
    atomic_json(output/'personal_statistics.json',{'complete':True,'samples':len(index),
        'topology_counts':index.topology_id.value_counts().to_dict(),'split_counts':split.role.value_counts().to_dict(),
        'independent_of_fusion_v2':True,'few_shot_budgets_nested':True,'normalization_fitted':False})
    return {'samples':len(index),'topologies':10}
