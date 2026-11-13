"""Bounded, resumable LOADBench validation interventions. Never opens test labels."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import argparse,hashlib,json,time,sys
import numpy as np
import pyarrow.parquet as pq
import psutil
from readout import predict,normalize

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/loadbench_remediation_v1'
RAW=ROOT/'data/model_ready/fusion_v2_training_v1'
META=ROOT/'data/model_ready/fusion_v2_frozen_e19_bf0d71ad'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,p)

LOG={'dB','dBm','dBc/Hz','source_label_dBm_compression_function'}
SCALES={'m':1e-6,'F':1e-12,'H':1e-9,'ohm':1e3,'V':1.,'A':1e-6,'Hz':1e6,'1':1.}
MODES=['published','design_knn8','design_affine32','graph_design_affine32','graph_design_review_sensitivity']

def score(pred,y,rule):
    err=abs(pred-y);m2=float(np.square(y-y.mean()).sum());rel=abs(y)>1e-8*rule['scale']
    return {'count':len(y),'mae':float(err.mean()),'rmse':float(np.sqrt(np.mean(err**2))),
      'p95_absolute_error':float(np.quantile(err,.95)),'maximum_absolute_error':float(err.max()),
      'r2':1-float(np.square(err).sum())/m2 if len(y)>1 and m2>0 else None,
      'hit5':float((err[rel]/abs(y[rel])<=.05).mean()) if rule['unit'] not in LOG and rel.any() else None,
      'relative_count':int(rel.sum()) if rule['unit'] not in LOG else 0,
      'hit_abs_1_unit':float((err<=1).mean()),'unit':rule['unit']}

def summary(rows):
    hit=[r['hit5'] for r in rows if r['hit5'] is not None];r2=[r['r2'] for r in rows if r['r2'] is not None]
    return {'macro_hit5':float(np.mean(hit)) if hit else None,'median_r2':float(np.median(r2)) if r2 else None,
      'macro_r2':float(np.mean(r2)) if r2 else None,'negative_r2_targets':sum(x<0 for x in r2),
      'target_pairs':len(rows),'relative_target_pairs':len(hit),'observed_scalars':sum(r['count'] for r in rows)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--probe',action='store_true');args=ap.parse_args()
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    if psutil.virtual_memory().available<1.1*2**30:raise SystemExit('Deferred: require 1.1GiB available RAM; trainer untouched')
    OUT.mkdir(parents=True,exist_ok=True)
    bundle_path=ROOT/'artifacts/experiment1_rgcn_local_head_v1/bundle.json'
    bundle=read(bundle_path);manifest=read(RAW/'manifest.json');fm=read(META/'manifest.json');tasks=read(RAW/'tasks.json')
    recipe={'version':'LOADBench_ID_readout_remediation_v1','script_sha256':sha(Path(__file__)),
      'readout_sha256':sha(Path(__file__).with_name('readout.py')),'original_bundle_sha256':sha(bundle_path),
      'split_sha256':fm['split_sha256'],'cache_manifest_sha256':sha(RAW/'manifest.json'),
      'quality_manifest_sha256':sha(ROOT/'data/audit/loadbench_physical_v1/manifest.json'),
      'modes':MODES,'ridge':.01,'k_affine':32,'k_knn':8,'test_access':False,
      'selection':'On all-original same validation targets: highest macro_hit5 among non-review variants with median_R2>=published, macro_R2>=published and negative_R2_count<=published. No per-target selection.',
      'quality':'Sensitivity only: exclude review TRAIN neighbors for ft/phasemargin, not other targets; no claim all review records invalid; mask unchanged.',
      'feature_transform':'SI design parameters signed log1p(unit reference); mean/std/active fitted to TRAIN only. Frozen GNN embedding uses its existing TRAIN transforms.',
      'fallback':'All variants retain published readout for contracts with fewer than256 TRAIN rows; no source label changes.',
      'scope':'Known-topology ID only; new readout, not an end-to-end GNN or LOTO improvement.'}
    rp=OUT/'recipe.json'
    if rp.exists():assert read(rp)==recipe,'Recipe changed: use a new version'
    else:save(rp,recipe)
    meta={k:np.load(META/'validation'/(k+'.npy'),mmap_mode='r') for k in ['embedding','group','target','mask','refs']}
    old=np.load(ROOT/'artifacts/experiment1_rgcn_local_head_v1/validation_prediction_symlog.npy',mmap_mode='r')
    topologies=[n for n in manifest['topologies'] if n.startswith('LOADBench__')]
    if args.probe:topologies=['LOADBench__SimpleOpAmp4','LOADBench__SimpleOpAmp130']
    begin=time.monotonic();proof=[];peak=0
    for top in topologies:
        dest=OUT/(top+'.json')
        if dest.exists():proof.append(read(dest));continue
        if psutil.virtual_memory().available<700*2**20:raise SystemExit('Memory guard; partial results resumable, trainer untouched')
        folder=ROOT/manifest['topologies'][top]['folder'];partition=ROOT/'data/fusion_v2/sampled'/(top+'.parquet')
        data=pq.read_table(partition,columns=['sample_id','parameters','parameter_units'])
        ids=data['sample_id'].to_pylist();idmap={s:i for i,s in enumerate(ids)}
        params=np.asarray(data['parameters'].to_pylist(),dtype=np.float64);units=data['parameter_units'][0].as_py()
        refs=np.array([SCALES.get(u,1.) for u in units]);params=np.sign(params)*np.log1p(abs(params)/refs)
        roles=np.load(folder/'role.npy',mmap_mode='r');numbers=np.load(folder/'group.npy',mmap_mode='r')
        # Read only train/validation quality references; test labels/flags are not requested.
        quality=pq.read_table(ROOT/'data/audit/loadbench_physical_v1/quality_index.parquet',
          columns=['sample_id','needs_review'],filters=[('topology_id','=',top),('baseline_ID_role','in',['train','validation'])])
        review={r['sample_id'] for r in quality.to_pylist() if r['needs_review']}
        rows={mode:{cohort:[] for cohort in ['all_original','AC_no_review']} for mode in MODES}
        checks=[]
        for code in np.unique(numbers[np.asarray(roles)==0]):
            group=manifest['group_names'][int(code)];info=bundle['contracts'][group]
            group_number=fm['groups'].index(group);vr=np.flatnonzero(np.asarray(meta['group'])==group_number)
            if not len(vr):continue
            pos=np.asarray(meta['refs'][vr,1],dtype=int);assert np.all(roles[pos]==1)
            path=ROOT/info['index'];assert sha(path)==info['index_sha256']
            with np.load(path,allow_pickle=False) as stored:b={k:stored[k] for k in stored.files}
            tr=np.array([idmap[s.decode()] for s in b['train_ids']],dtype=int)
            assert np.all(roles[tr]==0) and not set(tr)&set(pos)
            expected_tr=np.flatnonzero((np.asarray(roles)==0)&(np.asarray(numbers)==code))
            np.testing.assert_array_equal(tr,expected_tr)
            x,q,mu,sd,active=normalize(params[tr],params[pos])
            e=b['train_embedding'].astype(np.float64);a=b['embedding_active']
            ev=(np.asarray(meta['embedding'][vr],dtype=np.float64)[:,a]-b['embedding_mean'][a])/b['embedding_std'][a]
            combined=np.column_stack([x/max(1,np.sqrt(x.shape[1])),e/max(1,np.sqrt(e.shape[1]))])
            cv=np.column_stack([q/max(1,np.sqrt(q.shape[1])),ev/max(1,np.sqrt(ev.shape[1]))])
            pred={'published':np.asarray(old[vr]).copy()};obs=b['train_mask'];z=b['train_z']
            eligible=obs.copy();train_review=np.array([ids[i] in review for i in tr]);val_review=np.array([ids[i] in review for i in pos])
            for slot,rule in tasks[group]['slots'].items():
                if rule['original_name'] in ['ft','phasemargin']:eligible[:,int(slot)]&=~train_review
            if len(tr)>=256:
                for mode,tx,vx,mask,affine,k in [
                  ('design_knn8',x,q,obs,False,8),('design_affine32',x,q,obs,True,32),
                  ('graph_design_affine32',combined,cv,obs,True,32),
                  ('graph_design_review_sensitivity',combined,cv,eligible,True,32)]:
                    prediction,count=predict(tx,vx,z,mask,k=k,affine=affine)
                    prediction[:,count==0]=pred['published'][:,count==0]
                    pred[mode]=prediction
            else:
                for mode in MODES[1:]:pred[mode]=pred['published'].copy()
            # Store an independent deployable TRAIN feature index; original bank remains immutable.
            np.savez_compressed(OUT/(group+'.npz'),train_parameters=x,center=mu,std=sd,active=active,
              parameter_units=np.asarray(units),train_review=train_review,train_ids=b['train_ids'])
            y=np.asarray(meta['target'][vr],dtype=np.float64);maskv=np.asarray(meta['mask'][vr],dtype=bool)
            for mode,p in pred.items():
                np.save(OUT/(group+'__'+mode+'.npy'),p,allow_pickle=False)
                physical=np.sign(p)*np.expm1(abs(p))*b['scales'];assert np.isfinite(physical).all()
                for slot,rule in tasks[group]['slots'].items():
                    j=int(slot)
                    for cohort in ['all_original','AC_no_review']:
                        valid=maskv[:,j].copy()
                        if cohort=='AC_no_review' and rule['original_name'] in ['ft','phasemargin']:valid&=~val_review
                        if not valid.any():continue
                        r=score(physical[valid,j],y[valid,j],rule)
                        r.update(supervision_group=group,topology=top,slot=j,original_name=rule['original_name'])
                        rows[mode][cohort].append(r)
            checks.append({'group':group,'train_rows':len(tr),'validation_rows':len(vr),
              'train_ids_sha256':hashlib.sha256(b['train_ids'].tobytes()).hexdigest(),
              'validation_ids_sha256':hashlib.sha256('\n'.join(ids[i] for i in pos).encode()).hexdigest(),
              'source_bank_sha256':info['index_sha256'],'missing_mask_unchanged':True,'test_labels_read':False})
        record={'topology':top,'metrics':rows,'proof':checks,'recipe_sha256':sha(rp),'source_partition_sha256':sha(partition)}
        save(dest,record);proof.append(record)
        peak=max(peak,psutil.Process().memory_info().rss)
        roles._mmap.close();numbers._mmap.close()
        save(OUT/'progress.json',{'topologies_complete':len(proof),'requested':len(topologies),'last':top,'seconds':time.monotonic()-begin,'peak_RSS_MiB':peak/2**20})
        print('completed',len(proof),top,flush=True)
    allrows={mode:{cohort:[r for p in proof for r in p['metrics'][mode][cohort]] for cohort in ['all_original','AC_no_review']} for mode in MODES}
    sums={mode:{c:summary(r) for c,r in cohorts.items()} for mode,cohorts in allrows.items()}
    base=sums['published']['all_original'];accepted=[]
    for mode in MODES[1:-1]:
        s=sums[mode]['all_original']
        if s['macro_hit5']>base['macro_hit5'] and s['median_r2']>=base['median_r2'] and s['macro_r2']>=base['macro_r2'] and s['negative_r2_targets']<=base['negative_r2_targets']:accepted.append(mode)
    selected=max(accepted,key=lambda mode:sums[mode]['all_original']['macro_hit5']) if accepted else None
    result={'state':'probe_complete' if args.probe else 'validation_complete','summaries':sums,'selected':selected,
      'topologies':len(proof),'train_rows':sum(c['train_rows'] for p in proof for c in p['proof']),
      'validation_rows':sum(c['validation_rows'] for p in proof for c in p['proof']),
      'seconds':time.monotonic()-begin,'peak_RSS_MiB':peak/2**20,'recipe_sha256':sha(rp),
      'test_labels_read':False,'original_model_and_data_modified':False,'per_target':allrows}
    save(OUT/('probe.json' if args.probe else 'validation_comparison.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='per_target'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
