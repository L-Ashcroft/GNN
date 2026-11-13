"""Train-internal group routing; outer validation selects one global blend."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import json,time,hashlib
import numpy as np
import psutil
from readout import predict
from run import ROOT,OUT,RAW,META,read,save,sha,score,summary

DEST=ROOT/'artifacts/loadbench_remediation_v1/calibration'

def old_knn(a,q,z,mask):
    result=[];norm=(a*a).sum(1);center=np.divide((z*mask).sum(0),mask.sum(0),out=np.zeros(16),where=mask.sum(0)>0)
    for start in range(0,len(q),32):
        v=q[start:start+32];d=np.maximum(0.,(v*v).sum(1)[:,None]+norm-2*v@a.T);k=min(8,len(a))
        ix=np.argpartition(d,k-1,axis=1)[:,:k];w=(1/(np.take_along_axis(d,ix,1)+1e-4))[:,:,None]*mask[ix];den=w.sum(1)
        result.append(np.divide((w*z[ix]).sum(1),den,out=np.broadcast_to(center,(len(v),16)).copy(),where=den>0))
    return np.concatenate(result)

def main():
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    if psutil.virtual_memory().available<1.1*2**30:raise SystemExit('Memory guard: calibration deferred')
    DEST.mkdir(parents=True,exist_ok=True);tick=time.monotonic()
    bundle=read(ROOT/'artifacts/experiment1_rgcn_local_head_v1/bundle.json');tasks=read(RAW/'tasks.json');fm=read(META/'manifest.json')
    recipe={'implementation':sha(Path(__file__)),'source_recipe':sha(OUT/'recipe.json'),'readout':sha(Path(__file__).with_name('readout.py')),
      'inner_split':'seed42, 20% of each original TRAIN contract; original encoder already trained on all original TRAIN labels, so this is readout calibration, not independently held-out encoder validation',
      'group_weights':[0.,.25,.5,.75,1.], 'outer_global_weights':[0.,.25,.5,.75,1.],
      'selection':'Each group maximizes inner hit5 among blends with inner mean R2>=old and negative target count<=old; outer selection uses original four global guards.',
      'no_test_access':True,'not_using_quality_filter':True}
    if (DEST/'recipe.json').exists():assert read(DEST/'recipe.json')==recipe
    else:save(DEST/'recipe.json',recipe)
    val={k:np.load(META/'validation'/(k+'.npy'),mmap_mode='r') for k in ['group','target','mask']}
    metrics={str(a):[] for a in recipe['outer_global_weights']};routes=[];collisions=[];peak=0
    for g,info in bundle['contracts'].items():
        if not info['topology'].startswith('LOADBench__'):continue
        result_path=DEST/(g+'.json')
        if result_path.exists():
            r=read(result_path)
            for k,rows in r['metrics'].items():metrics[k]+=rows
            routes.append(r['route']);collisions.append(r['collisions']);continue
        if psutil.virtual_memory().available<700*2**20:raise SystemExit('Memory guard; resumable calibration')
        with np.load(ROOT/info['index'],allow_pickle=False) as a:b={k:a[k] for k in a.files}
        with np.load(OUT/(g+'.npz'),allow_pickle=False) as a:d={k:a[k] for k in a.files}
        x=d['train_parameters'];e=b['train_embedding'].astype(np.float64);device=b['train_device'].astype(np.float64)
        combined=np.column_stack([x/max(1,np.sqrt(x.shape[1])),e/max(1,np.sqrt(e.shape[1]))])
        previous=np.column_stack([e/max(1,np.sqrt(e.shape[1])),device/max(1,np.sqrt(device.shape[1]))])
        n=len(x);rng=np.random.default_rng(42);order=rng.permutation(n);nn=max(1,n//5);hold=order[:nn];fit=order[nn:]
        rules=tasks[g]['slots'];z=b['train_z'];mask=b['train_mask'];weight=0.;trials=[]
        if n>=256 and len(fit)>32:
            p0=old_knn(previous[fit],previous[hold],z[fit],mask[fit]);p1,_=predict(combined[fit],combined[hold],z[fit],mask[fit])
            y=np.sign(z[hold])*np.expm1(abs(z[hold]))*b['scales']
            for alpha in recipe['group_weights']:
                p=(1-alpha)*p0+alpha*p1;physical=np.sign(p)*np.expm1(abs(p))*b['scales'];rr=[]
                for slot,rule in rules.items():
                    j=int(slot);use=mask[hold,j]
                    if use.any():rr.append(score(physical[use,j],y[use,j],rule))
                trials.append(dict(alpha=alpha,**summary(rr)))
            base=trials[0];accepted=[t for t in trials if t['macro_hit5'] is not None and t['macro_r2'] is not None and t['macro_r2']>=base['macro_r2'] and t['negative_r2_targets']<=base['negative_r2_targets']]
            if accepted:weight=max(accepted,key=lambda t:t['macro_hit5'])['alpha']
        code=fm['groups'].index(g);vr=np.flatnonzero(np.asarray(val['group'])==code)
        p0=np.load(OUT/(g+'__published.npy'));p1=np.load(OUT/(g+'__graph_design_affine32.npy'))
        assert len(vr)==len(p0)
        y=np.asarray(val['target'][vr]);maskv=np.asarray(val['mask'][vr]);rows={}
        for alpha in recipe['outer_global_weights']:
            p=p0+(alpha*weight)*(p1-p0);physical=np.sign(p)*np.expm1(abs(p))*b['scales'];rr=[]
            for slot,rule in rules.items():
                j=int(slot);use=maskv[:,j]
                if use.any():
                    r=score(physical[use,j],y[use,j],rule);r.update(supervision_group=g,slot=j,original_name=rule['original_name']);rr.append(r)
            rows[str(alpha)]=rr;metrics[str(alpha)]+=rr
        # Exact transformed-design input collisions: audit TRAIN only.
        _,inverse,counts=np.unique(x,axis=0,return_inverse=True,return_counts=True)
        contradictory=0
        for key in np.flatnonzero(counts>1):
            ix=np.flatnonzero(inverse==key)
            for j in map(int,rules):
                observed=ix[mask[ix,j]]
                if len(observed)>1 and np.ptp(z[observed,j])>1e-8:contradictory+=1
        collision={'group':g,'train_rows':n,'duplicate_input_rows':int(np.maximum(counts-1,0).sum()),'conflicting_input_target_pairs':contradictory}
        route={'group':g,'weight':weight,'inner_holdout_rows':len(hold),'inner_fit_rows':len(fit),'trials':trials,
          'inner_fit_ids_sha256':hashlib.sha256(b['train_ids'][fit].tobytes()).hexdigest(),'inner_holdout_ids_sha256':hashlib.sha256(b['train_ids'][hold].tobytes()).hexdigest()}
        routes.append(route);collisions.append(collision);save(result_path,{'metrics':rows,'route':route,'collisions':collision})
        peak=max(peak,psutil.Process().memory_info().rss)
        if len(routes)%20==0:print('calibrated',len(routes),flush=True)
    summaries={k:summary(v) for k,v in metrics.items()};base=summaries['0.0'];accepted=[]
    for key,s in summaries.items():
        if s['macro_hit5']>base['macro_hit5'] and s['median_r2']>=base['median_r2'] and s['macro_r2']>=base['macro_r2'] and s['negative_r2_targets']<=base['negative_r2_targets']:accepted.append(key)
    selected=max(accepted,key=lambda k:summaries[k]['macro_hit5']) if accepted else None
    result={'state':'complete','summary':summaries,'selected_global_weight':selected,'routes':routes,'train_collision_audit':collisions,
      'test_read':False,'seconds':time.monotonic()-tick,'peak_RSS_MiB':peak/2**20,'recipe_sha256':sha(DEST/'recipe.json')}
    save(DEST/'selection.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['routes','train_collision_audit']},indent=2),flush=True)

if __name__=='__main__':main()
