"""Bounded local graph-feature readouts and train-range safeguards, ID only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,ast,hashlib,time,statistics
import psutil
import numpy as np
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/experiment1_representation_diagnostic_v1';OUT=ROOT/'artifacts/experiment1_local_readout_v1';OUT.mkdir(parents=True,exist_ok=True)
if psutil.virtual_memory().available<1.2*2**30:raise SystemExit(75)
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8');os.replace(q,p)
RAW=ROOT/'data/model_ready/fusion_v2_training_v1';META=ROOT/'data/model_ready/fusion_v2_frozen_e19_bf0d71ad'
manifest=read(RAW/'manifest.json');fm=read(META/'manifest.json');tasks=read(RAW/'tasks.json');LOG={'dB','dBm','dBc/Hz','source_label_dBm_compression_function'}
assert read(BASE/'summary.json')['state']=='complete'
tree=ast.parse((HERE/'run.py').read_text(encoding='utf-8'));namespace=globals()
for node in tree.body:
    if isinstance(node,ast.FunctionDef) and node.name in {'metrics','summarize'}:exec(compile(ast.Module(body=[node],type_ignores=[]),'preserved_diagnostic_metrics','exec'),namespace)
meta={r:{k:np.load(META/r/(k+'.npy'),mmap_mode='r') for k in ['refs','embedding']} for r in ['train','validation']}
variants=['embedding_linear_trainrange','embedding_device_RFF_trainrange','embedding_local8','device_local8','combined_local8']
recipe={'variant':'local_bounded_readouts_ID_v1','source_diagnostic_recipe_sha256':sha(BASE/'recipe.json'),'script_sha256':sha(Path(__file__)),
    'metrics_sha256':sha(HERE/'run.py'),'raw_manifest_sha256':sha(RAW/'manifest.json'),'k':8,'inverse_squared_distance_epsilon':1e-4,
    'source_fit':'train-only normalization, observed output bounds and neighbor labels','test_read':False,
    'limitation':'Known semantic contracts and canonical devices only; not validated for LOTO/unseen dataset',
    'prediction_policy':'Train-range clipping changes predictions, never labels, Mask or evaluation formulas'}
if (OUT/'recipe.json').exists():assert read(OUT/'recipe.json')==recipe
else:save(OUT/'recipe.json',recipe)
result={m:[] for m in variants};groups_seen=set();tick=time.perf_counter();peak=0
for number,(name,spec) in enumerate(manifest['topologies'].items(),1):
    folder=ROOT/spec['folder'];arrays={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['values','known','zero','target','mask','role','group','sample_ids']}
    train=np.flatnonzero(arrays['role']==0);val=np.flatnonzero(arrays['role']==1)
    assert not set(arrays['sample_ids'][train].tolist())&set(arrays['sample_ids'][val].tolist())
    topo=fm['topologies'].index(name);map_rows={}
    for role,indices in [('train',train),('validation',val)]:
        select=np.flatnonzero(meta[role]['refs'][:,0]==topo);lookup=np.full(len(arrays['role']),-1,dtype=np.int64);lookup[meta[role]['refs'][select,1]]=select
        assert (lookup[indices]>=0).all();map_rows[role]=lookup[indices]
    local={m:[] for m in variants}
    for code in np.unique(arrays['group'][train]):
        group=manifest['group_names'][int(code)];assert group not in groups_seen;groups_seen.add(group)
        tr=train[arrays['group'][train]==code];vr=val[arrays['group'][val]==code]
        ti=np.flatnonzero(arrays['group'][train]==code);vi=np.flatnonzero(arrays['group'][val]==code)
        if not len(vr):continue
        record=np.load(BASE/(group+'_embedding_linear.npz'),allow_pickle=False)
        cols=record['raw_columns'];blocks=[];vblocks=[]
        for key in ['values','known','zero']:
            c=np.array([int(c) for k,c in cols if k==key],dtype=np.int64)
            if not len(c):continue
            a=arrays[key].reshape(len(arrays[key]),-1);blocks.append(np.asarray(a[tr[:,None],c[None,:]],dtype=np.float64));vblocks.append(np.asarray(a[vr[:,None],c[None,:]],dtype=np.float64))
        x=np.concatenate(blocks,1);v=np.concatenate(vblocks,1);active=record['device_active'];mean=record['device_mean'];std=record['device_std']
        x=(x[:,active]-mean[active])/std[active];v=(v[:,active]-mean[active])/std[active]
        e=np.asarray(meta['train']['embedding'][map_rows['train'][ti]],dtype=np.float64);ev=np.asarray(meta['validation']['embedding'][map_rows['validation'][vi]],dtype=np.float64)
        ea=record['embedding_active'];em=record['embedding_mean'];es=record['embedding_std'];e=(e[:,ea]-em[ea])/es[ea];ev=(ev[:,ea]-em[ea])/es[ea]
        target=np.asarray(arrays['target'][tr],dtype=np.float64);targetv=np.asarray(arrays['target'][vr],dtype=np.float64)
        mask=np.asarray(arrays['mask'][tr],dtype=bool);maskv=np.asarray(arrays['mask'][vr],dtype=bool)
        scales=np.array([tasks[group]['slots'].get(str(j),{}).get('scale',1.) for j in range(16)],dtype=np.float32).astype(np.float64)
        z=np.sign(target)*np.log1p(abs(target/scales));bounds=np.zeros((2,16));center=np.zeros(16)
        for slot in map(int,tasks[group]['slots']):
            obs=z[mask[:,slot],slot]
            if len(obs):bounds[:,slot]=[obs.min(),obs.max()];center[slot]=obs.mean()
        preds={}
        for kind in ['embedding_linear','embedding_device_RFF']:
            rec=np.load(BASE/(group+'_'+kind+'.npz'),allow_pickle=False)
            if kind=='embedding_linear':features=ev
            else:
                rff=np.sqrt(2.)*np.cos(v@rec['omega']+rec['phase']);features=np.column_stack([ev,v,rff])
            p=np.column_stack([np.ones(len(vr)),features])@rec['coefficients'];preds[kind+'_trainrange']=np.clip(p,bounds[0],bounds[1])
        for kind,(a,b) in {'embedding_local8':(e/max(1,np.sqrt(e.shape[1])),ev/max(1,np.sqrt(e.shape[1]))),
                            'device_local8':(x/max(1,np.sqrt(x.shape[1])),v/max(1,np.sqrt(v.shape[1]))),
                            'combined_local8':(np.column_stack([e/max(1,np.sqrt(e.shape[1])),x/max(1,np.sqrt(x.shape[1]))]),np.column_stack([ev/max(1,np.sqrt(e.shape[1])),v/max(1,np.sqrt(v.shape[1]))]))}.items():
            query=[];k=min(8,len(tr));norm=np.square(a).sum(1)
            for start in range(0,len(b),64):
                q=b[start:start+64];distance=np.maximum(0.,np.square(q).sum(1)[:,None]+norm[None,:]-2*q@a.T)
                ii=np.argpartition(distance,k-1,axis=1)[:,:k];dd=np.take_along_axis(distance,ii,axis=1);weights=1/(dd+1e-4)
                valid=mask[ii];w=weights[:,:,None]*valid;den=w.sum(1)
                pred=np.divide((w*z[ii]).sum(1),den,out=np.broadcast_to(center,(len(q),16)).copy(),where=den>0);query.append(pred)
            preds[kind]=np.concatenate(query)
        for model,p in preds.items():
            physical=np.sign(p)*np.expm1(abs(p))*scales;rr=metrics(physical,targetv,maskv,group);result[model].extend(rr);local[model].extend(rr)
        np.savez_compressed(OUT/(group+'.npz'),embedding_mean=em,embedding_std=es,embedding_active=ea,train_embedding=e.astype('float32'),
            device_mean=mean,device_std=std,device_active=active,raw_columns=cols,train_device=x.astype('float32'),
            train_z=z,train_mask=mask,train_ids=arrays['sample_ids'][tr],scales=scales,bounds=bounds,training_center=center)
    save(OUT/(name+'.json'),{'metrics':local,'topology':name});peak=max(peak,psutil.Process().memory_info().rss)
    for a in arrays.values():a._mmap.close()
    if number%20==0:print(number,name,flush=True)
summary={'state':'complete','updated_at':datetime.now(timezone.utc).isoformat(),'test_read':False,'contracts':len(groups_seen),
    'models':{m:summarize(rr) for m,rr in result.items()},'seconds':time.perf_counter()-tick,'peak_RSS_MiB':peak/2**20,'recipe':recipe}
assert len(groups_seen)==142
assert all(len(rr)==1491 and len({(r['supervision_group'],r['semantic_id']) for r in rr})==1491 for rr in result.values())
save(OUT/'summary.json',summary);save(ROOT/'reports/experiment1_local_readout.json',summary);print(json.dumps(summary['models'],indent=2),flush=True)
