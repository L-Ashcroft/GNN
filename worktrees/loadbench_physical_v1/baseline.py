"""Train-only, single-thread paired ridge diagnostic; never reads test labels.

Compare original versus no-screening-trigger TRAIN pools on identical no-trigger
VALIDATION samples. This is a bounded parameter baseline, not a replacement GNN.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
from collections import defaultdict
import argparse,hashlib,json,time
import numpy as np
import pyarrow.parquet as pq
import psutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/loadbench_physical_baseline_v1'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
SCALES={'m':1e-6,'F':1e-12,'H':1e-9,'ohm':1e3,'V':1.,'A':1e-6,'Hz':1e6,'1':1.}
LOG={'dB','dBm','dBc/Hz','source_label_dBm_compression_function'}

def score(pred,y,rule):
    error=pred-y;sse=float(np.square(error).sum());m2=float(np.square(y-y.mean()).sum());absolute=np.abs(error)
    rel=np.abs(y)>1e-8*rule['scale'];log=rule['unit'] in LOG
    return {'count':len(y),'mae':float(absolute.mean()),'rmse':float(np.sqrt(sse/len(y))),
            'r2':1-sse/m2 if len(y)>1 and m2>0 else None,
            'hit5':float((absolute[rel]/np.abs(y[rel])<=.05).mean()) if rel.any() and not log else None,
            'unit':rule['unit'],'relative_count':int(rel.sum()) if not log else 0}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--probe',action='store_true');args=parser.parse_args()
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    if psutil.virtual_memory().available<950*2**20:raise SystemExit('Deferred: require >=950MiB available host RAM; no training modified')
    tick=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    cache=read(ROOT/'data/model_ready/fusion_v2_training_v1/manifest.json');tasks=read(ROOT/'data/model_ready/fusion_v2_training_v1/tasks.json')
    quality=ROOT/'data/audit/loadbench_physical_v1';policy=read(quality/'policy.json');manifest=read(quality/'manifest.json')
    assert sha(quality/'review_references.parquet')==manifest['output_sha256']['data/audit/loadbench_physical_v1/review_references.parquet']
    review=set(pq.read_table(quality/'review_references.parquet',columns=['sample_id'])['sample_id'].to_pylist())
    chosen=[n for n in cache['topologies'] if n.startswith('LOADBench__')]
    if args.probe:chosen=['LOADBench__SimpleOpAmp4','LOADBench__SimpleOpAmp130']
    lambdas=[.01,1.,100.];metrics={f'{mode}:{lam}':[] for mode in ['original_train','no_trigger_train'] for lam in lambdas}
    counts=defaultdict(int);unsupported=[];proof=[]
    for top in chosen:
        if psutil.virtual_memory().available<640*2**20:raise SystemExit('Stopped diagnostic at memory guard; original trainer unaffected')
        folder=ROOT/cache['topologies'][top]['folder'];partition=ROOT/'data/fusion_v2/sampled'/(top+'.parquet')
        # Read roles/identities first. Parameters/targets are selected from train
        # and validation only; no target.npy or test prediction file is opened.
        roles=np.load(folder/'role.npy',mmap_mode='r');groups=np.load(folder/'group.npy',mmap_mode='r')
        table=pq.read_table(partition,columns=['sample_id','parameters','parameter_units','parameter_names'])
        ids=table['sample_id'].to_pylist();clean=np.array([sid not in review for sid in ids]);train=np.asarray(roles)==0;val=(np.asarray(roles)==1)&clean
        train_pos=np.flatnonzero(train);val_pos=np.flatnonzero(val)
        # Read existing cache TRAIN and VALIDATION targets by role, never TEST.
        # Arrow predicate filtering uses sample IDs from these two roles only.
        labels=pq.read_table(partition,columns=['sample_id','performance','mask'],filters=[('sample_id','in',[ids[i] for i in np.r_[train_pos,val_pos]])])
        label_by_id={r['sample_id']:r for r in labels.to_pylist()}
        requested={ids[i] for i in np.r_[train_pos,val_pos]}
        assert set(label_by_id)==requested
        assert not requested&{ids[i] for i in np.flatnonzero(np.asarray(roles)==2)}
        assert sha(partition)==manifest['source_partition_sha256'][partition.relative_to(ROOT).as_posix()]
        x=np.asarray(table['parameters'].to_pylist(),dtype='float64');units=table['parameter_units'][0].as_py()
        reference=np.array([SCALES.get(u,1.) for u in units]);x=np.sign(x)*np.log1p(abs(x)/reference)
        assert np.isfinite(x).all()
        counts['original_train_rows']+=int(train.sum());counts['no_trigger_train_rows']+=int((train&clean).sum());counts['shared_validation_rows']+=int(val.sum())
        for code in np.unique(np.asarray(groups)[val_pos]):
            group=cache['group_names'][int(code)];task=tasks[group];observations=np.asarray(groups)==code
            fit_positions={'original_train':np.flatnonzero(train&observations),'no_trigger_train':np.flatnonzero(train&clean&observations)}
            v=np.flatnonzero(val&observations)
            if not len(v):continue
            for slot,rule in task['slots'].items():
                j=int(slot);vv=np.array([i for i in v if label_by_id[ids[i]]['mask'][j]],dtype='int64')
                if not len(vv):continue
                positions={mode:np.array([i for i in ix if label_by_id[ids[i]]['mask'][j]],dtype='int64') for mode,ix in fit_positions.items()}
                if any(not len(ix) for ix in positions.values()):
                    unsupported.append({'topology':top,'group':group,'slot':j,'validation_observations':len(vv),'train_counts':{k:len(a) for k,a in positions.items()}});continue
                yv=np.array([label_by_id[ids[i]]['performance'][j] for i in vv]);scale=rule['scale']
                for mode,ix in positions.items():
                    xt=x[ix];center=xt.mean(0);std=xt.std(0);active=std>1e-9;xx=(xt[:,active]-center[active])/std[active];xv=(x[vv][:,active]-center[active])/std[active]
                    y=np.array([label_by_id[ids[i]]['performance'][j] for i in ix]);t=np.sign(y/scale)*np.log1p(abs(y/scale));mu=float(t.mean());sd=max(float(t.std()),1e-6)
                    a=xx.T@xx;b=xx.T@((t-mu)/sd)
                    for lam in lambdas:
                        w=np.linalg.solve(a+lam*np.eye(a.shape[0]),b) if len(b) else np.empty(0)
                        z=mu+sd*(xv@w);p=np.sign(z)*np.expm1(np.minimum(abs(z),80))*scale
                        result=score(p,yv,rule);result.update(topology=top,supervision_group=group,slot=j,original_name=rule['original_name'],training_observations=len(ix))
                        metrics[f'{mode}:{lam}'].append(result)
        proof.append({'topology':top,'fit_train_only':True,'test_label_rows_returned':False,'validation_sample_count':int(val.sum()),
          'requested_label_ids_sha256':hashlib.sha256('\n'.join(sorted(requested)).encode()).hexdigest(),
          'source_partition_sha256':sha(partition)})
        roles._mmap.close();groups._mmap.close()
        del table,labels,label_by_id,x
        if len(proof)%15==0:print('ridge_topologies',len(proof),flush=True)
    summary={}
    for key,rows in metrics.items():
        hits=[r['hit5'] for r in rows if r['hit5'] is not None];r2=[r['r2'] for r in rows if r['r2'] is not None]
        summary[key]={'macro_hit5':float(np.mean(hits)) if hits else None,'median_r2':float(np.median(r2)) if r2 else None,
                      'macro_r2':float(np.mean(r2)) if r2 else None,'negative_r2_targets':sum(a<0 for a in r2),'target_pairs':len(rows)}
    result={'state':'probe_complete' if args.probe else 'validation_diagnostic_complete','model':'Per-contract train-normalized parameter ridge; no GNN weights used',
      'role_policy':'Both TRAIN pools scored on exact same no-trigger VALIDATION IDs; no test labels read',
      'train_only_parameters':'Input center/std/active columns and symlog target center/std fitted separately on each TRAIN pool',
      'label_source':'Original sampled Parquet float64 label rows; only requested TRAIN/VALIDATION IDs returned by Arrow filter',
      'prediction_numeric_guard':'Absolute predicted symlog limited to80 before expm1; no source labels changed',
      'quality_policy_sha256':policy['policy_sha256'],'configuration_lambdas':lambdas,'counts':dict(counts),'unsupported_target_pairs':unsupported,
      'implementation_sha256':sha(Path(__file__)),'original_split_sha256':cache['split_sha256'],
      'summary':summary,'per_target':metrics,'proof':proof,'seconds':time.monotonic()-tick,
      'limitations':'Diagnostic cannot establish GNN improvement, causal source-domain transfer, or a physics-certified clean dataset. Different TRAIN sizes reported. No promotion based on test scores.'}
    output=OUT/('probe.json' if args.probe else 'validation_comparison.json');output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_target','proof']},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
