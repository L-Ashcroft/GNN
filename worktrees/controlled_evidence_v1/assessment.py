"""Frozen-only evaluation, original task-macro hits plus native sample-MRE.

No torch import, no inference/model selection, no overwriting parent outputs.
Actual topology is recovered from raw sample IDs, not a shared-head alias.
"""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,math,os
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import importlib.util
_spec=importlib.util.spec_from_file_location('frozen_reference_metrics',ROOT/'src/falcon_reference_metrics.py')
_reference=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_reference)
FalconReferenceAccumulator,regression,LOG_UNITS=_reference.FalconReferenceAccumulator,_reference.regression,_reference.LOG_UNITS
DEST=ROOT/'artifacts/controlled_evidence_v1/evaluation'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def atomic(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp')
    q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(q,p)
def ready(folder):
    if not (folder/'selection_frozen.json').exists() or not (folder/'heldout_predictions.parquet').exists():return False
    if not (folder/'heldout_test_metrics.json').exists():return False
    if (folder/'status.json').exists() and read(folder/'status.json').get('state')!='complete':return False
    f=read(folder/'selection_frozen.json')
    assert f['outer_test_used_for_selection'] is False
    assert sha(folder/'best.pt')==f['checkpoint_sha256'],'Frozen weight mismatch'
    recipe=read(folder/'recipe.json')
    digest=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
    assert f['recipe_digest']==digest,'Frozen recipe mismatch'
    return True
def load(folder):
    assert ready(folder),'No completed frozen test'
    import pyarrow.parquet as pq
    rows=pq.read_table(folder/'heldout_predictions.parquet').to_pylist()
    wanted={r['sample_id'] for r in rows};assert len(wanted)==len(rows),'Duplicate test ID'
    mapping={};manifest=read(ROOT/'data/model_ready/fusion_v2_training_v1/manifest.json')
    for top,spec in manifest['topologies'].items():
        p=ROOT/spec['folder']/'sample_ids.npy';a=np.load(p,mmap_mode='r')
        for b in a:
            sid=b.decode('utf-8')
            if sid in wanted:
                assert sid not in mapping,'Duplicate raw sample ID';mapping[sid]=top
        a._mmap.close()
    assert len(mapping)==len(rows),'Unresolved actual topology'
    for r in rows:
        actual=mapping[r['sample_id']]
        if r.get('topology_id'):assert r['topology_id']==actual
        r['topology_id']=actual;r['source_dataset']=actual.split('__')[0]
    return rows,read(folder/'task_registry.json')
def metrics(rows,tasks):
    acc=FalconReferenceAccumulator();physical={};mapping=read(ROOT/'data/fusion_v2/performance_mapping.json')['groups']
    for start in range(0,len(rows),256):
        groups={}
        for r in rows[start:start+256]:groups.setdefault((r['source_dataset'],r['topology_id'],r['supervision_group']),[]).append(r)
        for (source,top,g),items in groups.items():
            rules=json.loads(json.dumps(tasks[g]['slots']))
            for rule in rules.values():
                possibilities={(m['source_unit'],float(m['factor'])) for m in mapping.get(top,[]) if m['original_name']==rule['original_name'] and m['unit']==rule['unit']}
                assert len(possibilities)==1,('Missing/ambiguous native units',top,rule)
                unit,factor=next(iter(possibilities));rule['native_unit']=unit;rule['native_multiplier']=1/factor
            p=np.array([r['prediction'] for r in items],dtype='float64');y=np.array([r['target'] for r in items],dtype='float64');mask=np.array([r['mask'] for r in items],dtype=bool)
            acc.update(p,y,mask,[r['sample_id'] for r in items],source,top,g,rules)
            for slot,rule in rules.items():
                j=int(slot);valid=mask[:,j]
                if valid.any():
                    key=(source,g,rule['semantic_id'])
                    record=physical.setdefault(key,{'rule':rule,'parts':[]});record['parts'].append((y[valid,j],p[valid,j]))
    task_rows=[]
    for (source,g,semantic),record in sorted(physical.items()):
        rule=record['rule'];y=np.concatenate([a for a,b in record['parts']]);p=np.concatenate([b for a,b in record['parts']])
        stats=regression(y,p);valid=(np.abs(y)>1e-8*rule['scale'])&(rule['unit'] not in LOG_UNITS)
        rel=np.abs(p[valid]-y[valid])/np.abs(y[valid])
        task_rows.append(dict(source_dataset=source,supervision_group=g,semantic_id=semantic,original_name=rule['original_name'],unit=rule['unit'],
              count=len(y),mae=stats['mae'],rmse=stats['rmse'],r2=stats['r2_mathematical'],
              relative_count=int(valid.sum()),hit5=float(np.mean(rel<=.05)) if len(rel) else None,
              hit10=float(np.mean(rel<=.1)) if len(rel) else None))
    def summary(part):
        def average(k):
            values=[r[k] for r in part if r[k] is not None];return float(np.mean(values)) if values else None
        r2=[r['r2'] for r in part if r['r2'] is not None]
        return dict(macro_hit5=average('hit5'),macro_hit10=average('hit10'),macro_r2=average('r2'),
              median_r2=float(np.median(r2)) if r2 else None,negative_r2_tasks=sum(v<0 for v in r2),
              undefined_r2_tasks=len(part)-len(r2),relative_tasks=sum(r['hit5'] is not None for r in part),tasks=len(part))
    falcon,native_tasks,_,samples=acc.finish()
    return {'task_macro':summary(task_rows),'by_source':{s:summary([r for r in task_rows if r['source_dataset']==s]) for s in sorted({r['source_dataset'] for r in rows})},
        'sample_equal_native_MRE':falcon['overall'],'sample_count':len(rows),
        'aggregation':'task macro over (source,approved supervision group,semantic); undefined R2 omitted, negatives retained',
        'MRE_policy':'Original Mask and nonzero native label; epsilon1e-8; log labels numerical only; full untrimmed'},task_rows,native_tasks,samples
def evaluate_run(name):
    folder=ROOT/'artifacts'/name
    if not ready(folder):return False
    out=DEST/name
    paths=[folder/n for n in ['best.pt','recipe.json','selection_frozen.json','heldout_predictions.parquet','heldout_test_metrics.json','task_registry.json']]
    paths += [ROOT/'data/model_ready/fusion_v2_training_v1/manifest.json',ROOT/'data/fusion_v2/performance_mapping.json',Path(__file__),ROOT/'src/falcon_reference_metrics.py']
    pins={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    if (out/'receipt.json').exists():assert read(out/'receipt.json')['input_sha256']==pins;return True
    rows,tasks=load(folder);result,physical,native,samples=metrics(rows,tasks)
    original=read(folder/'heldout_test_metrics.json')
    # Training and unified reporting must use the same task-macro semantics.
    for k in ('macro_hit5','macro_hit10','macro_r2','median_r2'):
        if original.get(k) is not None:assert math.isclose(original[k],result['task_macro'][k],rel_tol=1e-8,abs_tol=1e-9),(name,k,original[k],result['task_macro'][k])
    import pyarrow as pa,pyarrow.parquet as pq
    out.mkdir(parents=True,exist_ok=True)
    for n,records in [('per_task',physical),('native_per_topology_task',native),('per_sample',samples)]:pq.write_table(pa.Table.from_pylist(records),out/(n+'.parquet'),compression='zstd')
    h=read(folder/'history.json') if (folder/'history.json').exists() else []
    cfg=read(folder/'recipe.json')['config'];status=read(folder/'status.json')
    labels=hashlib.sha256()
    for row in sorted(rows,key=lambda r:r['sample_id']):labels.update(json.dumps([row['sample_id'],row['target'],row['mask']],separators=(',',':')).encode())
    result.update(run=name,selected_epoch=read(folder/'selection_frozen.json')['best_epoch'],parameters=status.get('model_parameters'),
         training_steps=sum(e['training']['steps'] for e in h),training_step_seconds=sum(e['training']['seconds'] for e in h),
         training_cost_scope='Completed epoch training steps only; excludes validation, baseline, resource waits',
         selected_updates=sum(e['training']['steps'] for e in h if e['epoch']<=read(folder/'selection_frozen.json')['best_epoch']),
         labels_mask_sha256=labels.hexdigest(),checkpoint_sha256=sha(folder/'best.pt'),config=cfg,test_ids_sha256=hashlib.sha256('\n'.join(sorted(r['sample_id'] for r in rows)).encode()).hexdigest())
    atomic(out/'summary.json',result)
    assert all(sha(ROOT/p)==h for p,h in pins.items())
    atomic(out/'receipt.json',{'input_sha256':pins,'parent_outputs_unchanged':True,'frozen_only':True,'at':datetime.now(timezone.utc).isoformat()})
    return True
def compare_intersection(idname,lotoname):
    a,b=ROOT/'artifacts'/idname,ROOT/'artifacts'/lotoname
    if not (ready(a) and ready(b)):return False
    dest=DEST/'matched_ID_LOTO'/(lotoname+'.json')
    inputs={str(p.relative_to(ROOT)):sha(p) for f in (a,b) for p in (f/'heldout_predictions.parquet',f/'task_registry.json',f/'selection_frozen.json')}
    if dest.exists():assert read(dest)['input_sha256']==inputs;return True
    ar,at=load(a);br,bt=load(b);aa={r['sample_id']:r for r in ar};bb={r['sample_id']:r for r in br};ids=sorted(aa.keys()&bb.keys())
    assert ids,'No common test IDs'
    for sid in ids:
        x,y=aa[sid],bb[sid];assert x['topology_id']==y['topology_id']
        assert x['mask']==y['mask'],'Mask mismatch'
        valid=np.asarray(x['mask'],dtype=bool)
        assert np.array_equal(np.asarray(x['target'])[valid],np.asarray(y['target'])[valid]),'Observed target mismatch'
        for j,m in enumerate(x['mask']):
            if m:
                rx=at[x['supervision_group']]['slots'][str(j)];ry=bt[y['supervision_group']]['slots'][str(j)]
                assert all(rx[k]==ry[k] for k in ('semantic_id','unit','scale','original_name')),'Incompatible common test target'
    ma=metrics([aa[s] for s in ids],at)[0];mb=metrics([bb[s] for s in ids],bt)[0]
    atomic(dest,{'input_sha256':inputs,'common_test_samples':len(ids),'ID':ma,'LOTO':mb,
       'test_set':'Exact sample_id intersection; label/Mask/semantic identity checked',
       'remaining_confounders':['Training population/size differs by topology holdout','ID selects global validation, LOTO compatible-head validation','OldM04 uses historical implementation/microbatch; not a perfectly controlled rerun']})
    return True
