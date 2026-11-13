"""Additional split view: keep calibration centres in train; preserve all test IDs."""
from collections import Counter
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read
from src.analoggym_campaign import sha
from src.windows_campaign_io import atomic_json
from src.analoggym_personal_protocols import key,nested_order


def main():
    output=ROOT/'data/generated/analoggym_personal_20k_v1';source=output/'splits/id.parquet'
    original=pd.read_parquet(source);safe=original.copy()
    centres={a['candidate'] for a in read(output/'sampling_policy_v2.json')['centre_provenance']}
    is_anchor=(safe.topology_id=='ldo_folded_cascode')&safe.candidate.isin(centres)
    assert int(is_anchor.sum())==len(centres)==3
    assert not (safe.loc[is_anchor,'role']=='test').any(),'Do not silently change the fixed test set'
    changes=[]
    for name,part in safe.loc[is_anchor & (safe.role!='train')].groupby('topology_id'):
        promote=part.index.tolist()
        candidates=safe.loc[(safe.topology_id==name)&(safe.role=='train')&~is_anchor]
        demote=candidates.sort_values('design_group',key=lambda s:s.map(key)).index[:len(promote)].tolist()
        assert len(demote)==len(promote)
        for index in promote+demote:
            old=safe.at[index,'role'];new='train' if index in promote else 'validation'
            changes.append({'sample_id':safe.at[index,'sample_id'],'candidate':int(safe.at[index,'candidate']),
                'topology_id':name,'old_role':old,'new_role':new})
            safe.at[index,'role']=new
    assert (safe.loc[is_anchor,'role']=='train').all()
    assert set(original.loc[original.role=='test','sample_id'])==set(safe.loc[safe.role=='test','sample_id'])
    assert safe.sample_id.is_unique and safe.design_group.is_unique
    for _,part in safe.groupby('topology_id'):
        assert part.role.value_counts().to_dict()=={'train':1600,'validation':200,'test':200}
    folder=output/'splits/anchor_safe_v1';folder.mkdir(exist_ok=True)
    safe.to_parquet(folder/'id.parquet',index=False,compression='zstd')
    train=safe.loc[safe.role=='train'];ordered=nested_order(train);records=[];previous=set()
    for k in [0,5,10,20,50,100,200,500,1000]:
        selected=ordered[:k];assert previous<=set(selected)<=set(train.sample_id);previous=set(selected)
        records.append({'budget':k,'sample_ids':selected,'topology_counts':dict(Counter(
            safe.set_index('sample_id').loc[selected,'topology_id'].tolist())) if selected else {}})
    atomic_json(folder/'few_shot_global.json',{'seed':42,'scope':'k labels total across target domain',
        'budgets':records,'test_file':'id.parquet','test_role':'test','validation_role':'validation',
        'validation_labels_additional_budget':2000,'normalization_fitted':False})
    records=[];previous=set()
    for k in [5,10,20,50,100]:
        selected=[]
        for _,part in train.groupby('topology_id',sort=True):selected+=nested_order(part)[:k]
        assert previous<=set(selected)<=set(train.sample_id);previous=set(selected)
        assert set(Counter(safe.set_index('sample_id').loc[selected,'topology_id']).values())=={k}
        records.append({'budget_per_topology':k,'total_budget':len(selected),'sample_ids':selected})
    atomic_json(folder/'few_shot_per_topology.json',{'seed':42,'scope':'k per topology; total 10k',
        'budgets':records,'test_file':'id.parquet','test_role':'test','validation_labels_additional_budget':2000})
    audit={'passed':True,'original_split_sha256':sha(source),'new_split_sha256':sha(folder/'id.parquet'),
        'original_files_modified':False,'test_ids_unchanged':True,'all_three_sampling_centres_in_train':True,
        'role_changes':changes,'split_counts':safe.role.value_counts().to_dict(),
        'per_topology_counts':{'train':1600,'validation':200,'test':200},'few_shot_nested_and_train_only':True,
        'limitations':['Does not remove near-neighbour similarity within a local sampling region.',
            'Calibration labels used for data collection are separate from downstream adaptation-label budgets.',
            'Validation target labels must be counted as extra budget; new model adapter/heads still required.']}
    atomic_json(folder/'audit.json',audit)
    print('Anchor-safe split ready; test unchanged; reassigned',len(changes),'roles; all budgets rebuilt.')


if __name__=='__main__':main()
