"""Prepare leakage-safe LOTO protocols only for previously verified targets.

Metadata-only operation: no performance labels are fitted or test-tuned.
Existing dataset/ID split and semantic contracts are never overwritten.
"""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
import numpy as np
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/fusion_v2/splits/verified_loto_v1';OUT.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
source=ROOT/'data/fusion_v2/splits/id_stratified_v1.parquet';audit=read(ROOT/'data/fusion_v2/supervision_group_details.json');tasks=read(ROOT/'data/model_ready/fusion_v2_training_v1/tasks.json');raw=read(ROOT/'data/model_ready/fusion_v2_training_v1/manifest.json')
table=pq.read_table(source,columns=['sample_id','design_group','source_dataset','topology_id','connectivity_group','role']);columns={k:np.asarray(table[k].to_pylist(),dtype=object) for k in table.column_names}
assert len(columns['sample_id'])==500000 and len(set(columns['sample_id']))==500000
observed={}
for group,role in zip(columns['design_group'],columns['role']):
    if group in observed:assert observed[group]==role,'Original ID design-group leakage'
    else:observed[group]=role
position={str(s):i for i,s in enumerate(columns['sample_id'])};refs=np.full((500000,2),-1,dtype='int32');topologies=list(raw['topologies'])
for number,name in enumerate(topologies):
    ids=np.load(ROOT/raw['topologies'][name]['folder']/'sample_ids.npy',mmap_mode='r')
    for row,identity in enumerate(ids):refs[position[identity.decode()]]=[number,row]
    ids._mmap.close()
assert (refs>=0).all()
shared={};aliases={};folds=[];skipped=[]
for merge in audit['verified_merge_sets']:
    assert merge['status']=='verified_same_published_protocol_not_activated'
    alias='verified_LOADBench_'+merge['merge_id'];groups=merge['supervision_group_ids'];base=tasks[groups[0]]
    mapping={}
    for group in groups:
        aliases[group]=alias;slotmap={}
        for slot,rule in tasks[group]['slots'].items():
            name=rule['original_name'];assert name in merge['shared_targets']
            reference=next(r for r in base['slots'].values() if r['original_name']==name)
            assert rule['unit']==reference['unit'] and rule['scale']==reference['scale']
            slotmap[slot]={'original_semantic_id':rule['semantic_id'],'shared_semantic_id':alias+'::'+name,'unit':rule['unit'],'scale':rule['scale'],'original_name':name}
        mapping[group]=slotmap
    shared[alias]={'merge_id':merge['merge_id'],'protocol_signature':merge['protocol_signature'],'original_groups':groups,'slot_aliases':mapping,'proof':merge['proof']}
    done_structures=set()
    for target in merge['topology_ids']:
        structural=set(columns['connectivity_group'][columns['topology_id']==target]);assert len(structural)==1
        structure=next(iter(structural))
        if structure in done_structures:continue
        done_structures.add(structure);held=columns['connectivity_group']==structure
        target_topologies=[t for t in merge['topology_ids'] if np.any((columns['topology_id']==t)&held)]
        training_peers=[t for t in merge['topology_ids'] if t not in target_topologies and np.any((columns['topology_id']==t)&(columns['role']=='train')&~held)]
        if not training_peers:skipped.append({'merge':merge['merge_id'],'structure':structure,'reason':'No compatible training topology remains after actual-structure holdout'});continue
        test=held&np.isin(columns['topology_id'],target_topologies);train=(columns['role']=='train')&~held;val=(columns['role']=='validation')&~held;unused=~(test|train|val)
        assert not np.any(train&test) and not np.any(val&test)
        assert not set(columns['connectivity_group'][train])&set(columns['connectivity_group'][test])
        assert not set(columns['design_group'][train])&set(columns['design_group'][test])
        assert not set(columns['design_group'][val])&set(columns['design_group'][test])
        assert not set(columns['design_group'][train])&set(columns['design_group'][val])
        fold_id=merge['merge_id']+'__hold_'+target.removeprefix('LOADBench__');path=OUT/(fold_id+'.npz')
        buffers={}
        for role,selected in [('train',train),('validation',val),('test',test),('unused',unused)]:
            at=np.flatnonzero(selected).astype('int32');buffers[role+'_positions']=at
            if role!='unused':buffers[role+'_references']=refs[at]
        np.savez_compressed(path,**buffers)
        # Check coverage of every requested target in the compatible train peer.
        coverage=np.zeros(16,dtype='int64')
        for peer in training_peers:
            folder=ROOT/raw['topologies'][peer]['folder'];roles=np.load(folder/'role.npy',mmap_mode='r');mask=np.load(folder/'mask.npy',mmap_mode='r');coverage+=mask[roles==0].sum(0);roles._mmap.close();mask._mmap.close()
        requested=sorted(int(s) for s in base['slots']);untrained=[s for s in requested if not coverage[s]]
        fold={'fold_id':fold_id,'merge_id':merge['merge_id'],'heldout_structure':structure,'test_topologies':target_topologies,
            'excluded_structural_aliases':sorted(set(columns['topology_id'][held])-set(target_topologies)),'compatible_training_topologies':training_peers,
            'head_alias':alias,'requested_target_slots':requested,'untrained_target_slots':untrained,
            'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'counts':{r:int(buffers[r+'_positions'].size) for r in ['train','validation','test','unused']},
            'training_target_counts':coverage.tolist(),'sample_design_structure_leakage_checks':'passed',
            'training_configuration':{'raw_graph_cache':'data/model_ready/fusion_v2_training_v1','reference_override_required':True,
                'initialization':'random_seed42','pretrained_checkpoint':None,'precomputed_GNN_embeddings':None,
                'target_calibration':'MUST_FIT_FROM_THIS_FOLD_TRAIN_ONLY','test_usage':'After validation model selection only',
                'forbidden':'ID-trained checkpoints, ID frozen embeddings and ID calibration all include this heldout topology'}}
        folds.append(fold)
manifest={'prepared_at':datetime.now(timezone.utc).isoformat(),'state':'splits_prepared_not_trained','source_split':source.relative_to(ROOT).as_posix(),'source_split_sha256':sha(source),
    'source_compatibility_audit_sha256':sha(ROOT/'data/fusion_v2/supervision_group_details.json'),'raw_manifest_sha256':sha(ROOT/'data/model_ready/fusion_v2_training_v1/manifest.json'),
    'topologies':topologies,'shared_contracts':shared,'head_alias_map':aliases,'folds':folds,'skipped':skipped,
    'ID_split_checks':{'sample_ids_unique':True,'design_groups_disjoint':True,'rows':500000},
    'policies':['Only previously verified same-protocol LOADBench aliases apply in these new experiments; existing142contracts unchanged',
                'Hold out every source topology sharing the exact full-port connectivity structure',
                'Non-target original ID-test samples remain unused; target topology entire population becomes outer test',
                'No heldout labels, target normalization or graph pretraining may enter training',
                'Untrained metrics must be declared unsupported, never hidden by changing observed Mask'],
    'original_dataset_or_ID_split_modified':False,'labels_read_or_fitted':False}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'folds':len(folds),'skipped':len(skipped),'size_MiB':sum(p.stat().st_size for p in OUT.glob('*'))/2**20,'all_checks_passed':True,'untrained_target_folds':sum(bool(f['untrained_target_slots']) for f in folds)},indent=2))
