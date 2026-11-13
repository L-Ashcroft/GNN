"""Train-only fitted ID diagnostics: frozen GNN vs actual device parameters.

No test references/labels are opened. Independent contract ridge readouts are
diagnostics, not a replacement claimed to generalize to unknown topologies.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import sys,json,time,hashlib,re,threading,statistics
from datetime import datetime,timezone
import psutil
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/experiment1_representation_diagnostic_v1';OUT.mkdir(parents=True,exist_ok=True)
if psutil.virtual_memory().available<1.2*2**30:raise SystemExit(75)
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
import numpy as np
import pyarrow.parquet as pq
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(t,p)
def now():return datetime.now(timezone.utc).isoformat()
RAW=ROOT/'data/model_ready/fusion_v2_training_v1';META=ROOT/'data/model_ready/fusion_v2_frozen_e19_bf0d71ad'
manifest=read(RAW/'manifest.json');meta_manifest=read(META/'manifest.json');tasks=read(RAW/'tasks.json');schema=read(RAW/'schema.json')
assert sha(RAW/'manifest.json')==meta_manifest['source_cache_manifest_sha256']
assert not meta_manifest['test_features_or_labels_cached']
meta={r:{k:np.load(META/r/(k+'.npy'),mmap_mode='r') for k in ['refs','group','embedding']} for r in ['train','validation']}
recipe={'variant':'independent_contract_ID_representation_ridge_v1','ridge_alpha':10.,'RFF_features':256,'RFF_seed':42,
    'raw_manifest_sha256':sha(RAW/'manifest.json'),'metadata_manifest_sha256':sha(META/'manifest.json'),'script_sha256':sha(Path(__file__)),
    'feature_fit':'train only, per-topology/device/feature mean and std; drop train-constant dimensions',
    'objective':'masked symlog ridge per contract/target; independent output rows; explicit L2 differs from neural optimizer',
    'models':['embedding_linear','device_linear','embedding_device_RFF'],'test_read':False,
    'limitation':'Known topology/device ordering only; diagnostic ID models, not LOTO-compatible graph encoders'}
if (OUT/'recipe.json').exists():assert read(OUT/'recipe.json')==recipe
else:save(OUT/'recipe.json',recipe)
done=threading.Event();resource={'peak_RSS':0,'min_available':psutil.virtual_memory().available}
def guard():
    while not done.wait(.2):
        av=psutil.virtual_memory().available;resource['min_available']=min(resource['min_available'],av);resource['peak_RSS']=max(resource['peak_RSS'],psutil.Process().memory_info().rss)
        if av<384*2**20:os._exit(86)
threading.Thread(target=guard,daemon=True).start()
def standardize(train,val):
    center=train.mean(0);std=train.std(0);active=std>np.maximum(1e-10,abs(center)*1e-7)
    return (train[:,active]-center[active])/std[active],(val[:,active]-center[active])/std[active],center,std,active
def ridge_features(x,v):return np.column_stack([np.ones(len(x)),x]),np.column_stack([np.ones(len(v)),v])
LOG={'dB','dBm','dBc/Hz','source_label_dBm_compression_function'}
def metrics(pred,y,mask,group):
    rows=[]
    assert np.isfinite(pred[mask]).all() and np.isfinite(y[mask]).all()
    for slot,rule in tasks[group]['slots'].items():
        j=int(slot);valid=mask[:,j];p=pred[valid,j];t=y[valid,j].astype(np.float64)
        if not len(t):continue
        error=np.abs(p-t);m2=np.square(t-t.mean()).sum();relative=np.abs(t)>1e-8*rule['scale'];count=int(relative.sum())
        if rule['unit'] in LOG:count=0
        rows.append({'supervision_group':group,'semantic_id':rule['semantic_id'],'original_name':rule['original_name'],'unit':rule['unit'],
            'count':len(t),'mae':float(error.mean()),'r2':float(1.-np.square(error).sum()/m2) if len(t)>=2 and m2>0 else None,
            'relative_count':count,'hit5':float(np.mean(error[relative]/abs(t[relative])<=.05)) if count else None,
            'hit10':float(np.mean(error[relative]/abs(t[relative])<=.1)) if count else None})
    return rows
def summarize(rows):
    def part(rr):
        hit=[r['hit5'] for r in rr if r['hit5'] is not None];r2=[r['r2'] for r in rr if r['r2'] is not None]
        return {'macro_hit5':statistics.mean(hit) if hit else None,'median_r2':statistics.median(r2) if r2 else None,
                'macro_r2':statistics.mean(r2) if r2 else None,'targets':len(rr),'negative_r2_targets':sum(v<0 for v in r2)}
    result=part(rows);result['by_source']={s:part([r for r in rows if tasks[r['supervision_group']]['contract']['source_dataset']==s]) for s in ['FALCON','AICircuit','LOADBench']};return result
allrows={m:[] for m in recipe['models']};audit=[];completed=0;tick=time.perf_counter();fit_rows=0;validation_rows=0
try:
    for number,(name,spec) in enumerate(manifest['topologies'].items(),1):
        receipt=OUT/(name+'.json')
        if receipt.exists():
            item=read(receipt)
            for model,rows in item['metrics'].items():allrows[model].extend(rows)
            audit.append(item['parameter_audit']);fit_rows+=item['train_rows'];validation_rows+=item['validation_rows'];completed+=1;continue
        folder=ROOT/spec['folder'];arrays={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['values','known','zero','target','mask','role','group']}
        for k in arrays:assert sha(folder/(k+'.npy'))==spec['cache_sha256'][k+'.npy'],(name,k)
        train=np.flatnonzero(arrays['role']==0);val=np.flatnonzero(arrays['role']==1)
        graph=read(ROOT/'data/fusion_v2/graphs'/(name+'.json'))
        # Coverage audit reads source parameter columns only; only train rows
        # are converted or used to determine varying design dimensions.
        source=ROOT/'data/fusion_v2/sampled'/(name+'.parquet')
        assert sha(source)==spec['inputs']['source']
        tab=pq.read_table(source,columns=['parameter_names','parameters']);names=tab.column('parameter_names')[int(train[0])].as_py()
        params=np.asarray(tab.column('parameters').take(train).to_pylist(),dtype=np.float64);vary=np.ptp(params,axis=0)>0
        symbols=set(re.findall(r'[A-Za-z_]\w*',' '.join(str(e) for row in graph['feature_expressions'] for e in row if e is not None)))
        missing=[p for p,v in zip(names,vary) if v and p not in symbols]
        pa={'topology':name,'source':name.split('__')[0],'varying_design_parameters':int(vary.sum()),'varying_parameter_names_absent_from_graph_expressions':missing}
        del tab,params
        # Keep only train-varying raw feature columns; never fit on val/test.
        xt=[];xv=[];raw_columns=[]
        for key in ['values','known','zero']:
            flat=arrays[key].reshape(len(arrays[key]),-1);minimum=None;maximum=None
            for start in range(0,len(train),512):
                b=np.asarray(flat[train[start:start+512]],dtype=np.float64);lo=b.min(0);hi=b.max(0)
                minimum=lo if minimum is None else np.minimum(minimum,lo);maximum=hi if maximum is None else np.maximum(maximum,hi)
            columns=np.flatnonzero(maximum>minimum)
            if len(columns):
                xt.append(np.asarray(flat[train[:,None],columns[None,:]],dtype=np.float64));xv.append(np.asarray(flat[val[:,None],columns[None,:]],dtype=np.float64));raw_columns.extend([(key,int(c)) for c in columns])
        x=np.concatenate(xt,1) if xt else np.zeros((len(train),0));v=np.concatenate(xv,1) if xv else np.zeros((len(val),0))
        x,v,mean,std,active=standardize(x,v);pa['varying_device_feature_dimensions']=x.shape[1]
        topo=meta_manifest['topologies'].index(name);pairs={}
        for role,indices in [('train',train),('validation',val)]:
            select=np.flatnonzero(meta[role]['refs'][:,0]==topo);refs=np.asarray(meta[role]['refs'][select,1]);lookup=np.full(len(arrays['role']),-1,dtype=np.int64);lookup[refs]=select
            assert (lookup[indices]>=0).all();pairs[role]=lookup[indices]
        e=np.asarray(meta['train']['embedding'][pairs['train']],dtype=np.float64);ev=np.asarray(meta['validation']['embedding'][pairs['validation']],dtype=np.float64)
        assert set(np.asarray(arrays['group'][train]).tolist())==set(np.asarray(meta['train']['group'][pairs['train']]).tolist())
        et,ee,em,es,ea=standardize(e,ev)
        rng=np.random.default_rng(42);omega=rng.normal(size=(x.shape[1],256))/max(1.,np.sqrt(x.shape[1]));phase=rng.uniform(0,2*np.pi,256)
        rt=np.sqrt(2.)*np.cos(x@omega+phase);rv=np.sqrt(2.)*np.cos(v@omega+phase)
        variants={'embedding_linear':(et,ee),'device_linear':(x,v),'embedding_device_RFF':(np.concatenate([et,x,rt],1),np.concatenate([ee,v,rv],1))}
        local={m:[] for m in recipe['models']}
        for group_number in np.unique(arrays['group'][train]):
            group=manifest['group_names'][int(group_number)];ti=np.flatnonzero(arrays['group'][train]==group_number);vi=np.flatnonzero(arrays['group'][val]==group_number)
            if not len(vi):continue
            target=np.asarray(arrays['target'][train[ti]],dtype=np.float64);target_val=np.asarray(arrays['target'][val[vi]],dtype=np.float64)
            mask=np.asarray(arrays['mask'][train[ti]],dtype=bool);mask_val=np.asarray(arrays['mask'][val[vi]],dtype=bool)
            scales=np.array([tasks[group]['slots'].get(str(j),{}).get('scale',1.) for j in range(16)],dtype=np.float32).astype(np.float64)
            y=np.sign(target)*np.log1p(np.abs(target/scales));predicts={}
            for model,(a,b) in variants.items():
                a,b=ridge_features(a[ti],b[vi]);coeff=np.zeros((a.shape[1],16));cache={}
                for slot in map(int,tasks[group]['slots']):
                    observed=mask[:,slot]
                    if not observed.any():continue
                    key=observed.tobytes()
                    if key not in cache:
                        aa=a[observed];regularizer=np.eye(a.shape[1])*10.;regularizer[0,0]=0.;cache[key]=(aa,aa.T@aa+regularizer)
                    aa,gram=cache[key];coeff[:,slot]=np.linalg.solve(gram,aa.T@y[observed,slot])
                z=b@coeff;physical=np.sign(z)*np.expm1(abs(z))*scales
                rows=metrics(physical,target_val,mask_val,group);local[model].extend(rows);allrows[model].extend(rows)
                np.savez_compressed(OUT/(group+'_'+model+'.npz'),coefficients=coeff,device_mean=mean,device_std=std,device_active=active,
                    raw_columns=np.asarray(raw_columns,dtype='U32'),embedding_mean=em,embedding_std=es,embedding_active=ea,omega=omega,phase=phase)
        item={'topology':name,'train_rows':len(train),'validation_rows':len(val),'parameter_audit':pa,'metrics':local}
        save(receipt,item);audit.append(pa);fit_rows+=len(train);validation_rows+=len(val);completed+=1
        for a in arrays.values():a._mmap.close()
        state={'state':'running','updated_at':now(),'completed_topologies':completed,'total_topologies':len(manifest['topologies']),
            'latest_topology':name,'elapsed_seconds':time.perf_counter()-tick,'partial_metrics':{k:summarize(v) for k,v in allrows.items()}}
        save(OUT/'status.json',state);print(completed,name,'raw_features',x.shape[1],'missing_parameters',len(missing),flush=True)
    assert fit_rows==400083 and validation_rows==50861
    summary={'state':'complete','at':now(),'recipe':recipe,'train_rows':fit_rows,'validation_rows':validation_rows,'test_read':False,
        'models':{k:summarize(v) for k,v in allrows.items()},'parameter_coverage':audit,
        'topologies_with_missing_varying_parameter_inputs':sum(bool(a['varying_parameter_names_absent_from_graph_expressions']) for a in audit),
        'resource':{'peak_RSS_MiB':resource['peak_RSS']/2**20,'minimum_available_RAM_MiB':resource['min_available']/2**20},'seconds':time.perf_counter()-tick}
    save(OUT/'summary.json',summary);save(OUT/'status.json',{'state':'complete','updated_at':now(),'completed_topologies':completed,'models':summary['models']})
    save(ROOT/'reports/experiment1_representation_diagnostic.json',summary);print(json.dumps(summary['models'],indent=2),flush=True)
finally:
    done.set()
    for role in meta.values():
        for a in role.values():a._mmap.close()
