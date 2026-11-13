"""Versioned review flags/reference cohorts; never rewrite targets or missingness."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,re
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/audit/loadbench_physical_v1'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def native_number(s):
    s=s.strip().lower();m=re.fullmatch(r'([0-9.e+-]+)([a-z]*)',s)
    if not m:raise ValueError('Unsupported sweep bound '+s)
    return float(m[1])*{'':1.,'k':1e3,'meg':1e6,'g':1e9,'m':1e-3,'u':1e-6,'n':1e-9,'p':1e-12}[m[2]]

def flags(gain,ft,phase,fmin,fmax):
    # Screening triggers are evidence requests, never universal invalidity rules.
    result=[]
    if not np.isfinite([gain,ft,phase]).all():result.append('nonfinite_AC_metric')
    if gain<=0:result.append('nonpositive_DC_gain_review')
    if ft<fmin or ft>fmax:result.append('reported_ft_outside_measured_sweep')
    if abs(phase)>360:result.append('absolute_reported_phase_over360_review')
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    policy={'version':'loadbench_physical_screen_v1','rule_origin':'Published AC measurement window and unit definitions; no test prediction errors used',
      'triggers':['nonfinite_AC_metric','nonpositive_DC_gain_review','reported_ft_outside_measured_sweep','absolute_reported_phase_over360_review'],
      'interpretation':'Review cohort, not an automatic proof of failed simulation. No-trigger cohort is not physics-certified.',
      'feasible_false_alone_excluded':False,'performance_labels_modified':False,'missingness_mask_modified':False,
      'train_val_test_roles_modified':False,'frozen_ID_or_LOTO_artifacts_modified':False,
      'scope':'LOADBench only; other sources carried through with scope=not_assessed_by_this_policy'}
    canonical=json.dumps(policy,sort_keys=True).encode();policy['policy_sha256']=hashlib.sha256(canonical).hexdigest()
    (OUT/'policy.json').write_text(json.dumps(policy,ensure_ascii=False,indent=2),encoding='utf-8')
    schema=pa.schema([('sample_id',pa.string()),('source_dataset',pa.string()),('topology_id',pa.string()),
      ('source_partition',pa.string()),('partition_row',pa.int64()),('baseline_ID_role',pa.string()),
      ('assessment_scope',pa.string()),('review_flags',pa.list_(pa.string())),('needs_review',pa.bool_()),
      ('sample_component',pa.string()),('source_feasible',pa.bool_())])
    id_schema=pa.schema([('sample_id',pa.string()),('source_partition',pa.string()),('partition_row',pa.int64()),('baseline_ID_role',pa.string())])
    files={k:OUT/(k+'.parquet') for k in ['quality_index','review_references','no_trigger_references']}
    writers={'quality_index':pq.ParquetWriter(files['quality_index'],schema,compression='zstd'),
             'review_references':pq.ParquetWriter(files['review_references'],id_schema,compression='zstd'),
             'no_trigger_references':pq.ParquetWriter(files['no_trigger_references'],id_schema,compression='zstd')}
    cache=read(ROOT/'data/model_ready/fusion_v2_training_v1/manifest.json');catalog={s['topology_id']:s for s in read(ROOT/'data/fusion_v2/catalog.json')}
    totals=Counter();reasons=Counter();roles=defaultdict(Counter);per_topology=[];input_hashes={}
    try:
        for top,spec in cache['topologies'].items():
            partition=ROOT/'data/fusion_v2/sampled'/(top+'.parquet');before=sha(partition);input_hashes[partition.relative_to(ROOT).as_posix()]=before
            source=top.split('__')[0];folder=ROOT/spec['folder'];role=np.load(folder/'role.npy',mmap_mode='r');ids=np.load(folder/'sample_ids.npy',mmap_mode='r')
            fmin=fmax=None
            if source=='LOADBench':
                tb=ROOT/'data/raw/LOADBench/subset'/top.removeprefix('LOADBench__')/'CombinedTestBench.cir'
                m=re.search(r'(?im)^\s*ac\s+dec\s+\d+\s+(\S+)\s+(\S+)',tb.read_text(encoding='utf-8'))
                assert m,'No declared AC sweep';fmin,fmax=map(native_number,[m[1],m[2]])
            offset=0;local=Counter()
            for batch in pq.ParquetFile(partition).iter_batches(batch_size=2048,columns=['sample_id','performance','sample_component','source_feasible']):
                metadata=[];review=[];plain=[]
                for i,row in enumerate(batch.to_pylist()):
                    j=offset+i;sid=ids[j].decode() if isinstance(ids[j],bytes) else str(ids[j]);assert sid==row['sample_id']
                    r={0:'train',1:'validation',2:'test'}[int(role[j])]
                    trigger=flags(row['performance'][0],row['performance'][1],row['performance'][3],fmin,fmax) if source=='LOADBench' else []
                    entry={'sample_id':sid,'source_dataset':source,'topology_id':top,'source_partition':partition.relative_to(ROOT).as_posix(),
                      'partition_row':j,'baseline_ID_role':r,'assessment_scope':'LOADBench_AC_screen' if source=='LOADBench' else 'not_assessed_by_this_policy',
                      'review_flags':trigger,'needs_review':bool(trigger),'sample_component':row['sample_component'],'source_feasible':row['source_feasible']}
                    ref={k:entry[k] for k in id_schema.names};metadata.append(entry);(review if trigger else plain).append(ref)
                    totals[source]+=1;totals['all_rows']+=1;local['rows']+=1
                    if trigger:totals['needs_review']+=1;local['needs_review']+=1;roles[source][r+'_review']+=1
                    else:totals['no_trigger']+=1;roles[source][r+'_no_trigger']+=1
                    reasons.update(trigger)
                writers['quality_index'].write_table(pa.Table.from_pylist(metadata,schema=schema))
                if review:writers['review_references'].write_table(pa.Table.from_pylist(review,schema=id_schema))
                if plain:writers['no_trigger_references'].write_table(pa.Table.from_pylist(plain,schema=id_schema))
                offset+=len(batch)
            assert offset==spec['rows'];role._mmap.close();ids._mmap.close();assert before==sha(partition)
            per_topology.append({'topology_id':top,'source':source,'ac_window_Hz':[fmin,fmax] if fmin else None,**dict(local)})
    finally:
        for writer in writers.values():writer.close()
    assert totals['all_rows']==500000 and totals['needs_review']+totals['no_trigger']==500000
    result={'state':'review_index_complete','policy':policy,'counts':dict(totals),'flag_occurrences':dict(reasons),
      'ID_role_counts':{k:dict(v) for k,v in roles.items()},'per_topology':per_topology,'source_partition_sha256':input_hashes,
      'output_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files.values()},
      'outputs_bytes':sum(p.stat().st_size for p in files.values()),
      'training_recommendation':'Preserve original benchmark; review flagged records before claiming physical ft/phase validity. No new model automatically trained on an unverified filter.'}
    (OUT/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'quality_index':result['state'],'counts':dict(totals),'flags':dict(reasons),'disk_MiB':result['outputs_bytes']/2**20},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
