"""Predeclared global mixtures and train-count fallback; validation only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import sys,json,ast,statistics,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from model import LocalDecoder,load_model,sha
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
OUT=ROOT/'artifacts/experiment1_rgcn_local_head_v1';OUT.mkdir(parents=True,exist_ok=True)
RAW=ROOT/'data/model_ready/fusion_v2_training_v1';META=ROOT/'data/model_ready/fusion_v2_frozen_e19_bf0d71ad'
manifest=read(RAW/'manifest.json');fm=read(META/'manifest.json');tasks=read(RAW/'tasks.json');cal=read(ROOT/'data/model_ready/fusion_v2_accuracy_v2/target_calibration.json')
policy={'global_graph_local_weights':[0.,.25,.5,.75,1.],'minimum_local_training_rows':256,'neighbors':8,'distance_epsilon':1e-4,
    'sparse_policy':'Use unchanged parent GNN/head for every target of any contract with fewer256 training rows',
    'selection':'Max validation macro_hit5 among candidates passing source/median/negative-count guards and macro_R2>=parent',
    'no_targetwise_selection':True,'test_read':False,'script_sha256':sha(Path(__file__)),'model_sha256':sha(HERE/'model.py')}
(OUT/'policy.json').write_text(json.dumps(policy,indent=2),encoding='utf-8')
bundle={'format':'edge_RGCN_local_decoder_v1','graph_checkpoint':'artifacts/experiment1_autonomous_head_v1/best.pt',
    'graph_checkpoint_sha256':sha(ROOT/'artifacts/experiment1_autonomous_head_v1/best.pt'),'device_features':read(RAW/'schema.json')['device_features'],
    'neighbors':8,'distance_epsilon':1e-4,'minimum_local_training_rows':256,'graph_local_weight':1.,'contracts':{},'topologies':{},
    'raw_manifest_sha256':sha(RAW/'manifest.json'),'split_sha256':fm['split_sha256'],'test_used':False,
    'limitations':'Known contracts and canonical device ordering; no claim of unseen-topology/contract zero-shot capability'}
for name,spec in manifest['topologies'].items():
    record=read(ROOT/'artifacts/experiment1_local_readout_v1'/(name+'.json'));static=read(ROOT/spec['folder']/'structure.json');d=max(static['device_index'])+1
    first=[static['device_index'].index(i) for i in range(d)];bundle['topologies'][name]={'structure':static,'first_device_edges':first}
    for row in record['metrics']['combined_local8']:
        g=row['supervision_group'];index=ROOT/'artifacts/experiment1_local_readout_v1'/(g+'.npz')
        if g not in bundle['contracts']:bundle['contracts'][g]={'topology':name,'training_rows':cal['groups'][g]['training_rows'],'index':index.relative_to(ROOT).as_posix(),'index_sha256':sha(index)}
(OUT/'bundle.json').write_text(json.dumps(bundle,indent=2),encoding='utf-8')
net,_=load_model();decoder=net.decoder
namespace={'np':np,'statistics':statistics,'tasks':tasks,'LOG':{'dB','dBm','dBc/Hz','source_label_dBm_compression_function'}}
for n in ast.parse((ROOT/'worktrees/representation_diagnostic_v1/run.py').read_text(encoding='utf-8')).body:
    if isinstance(n,ast.FunctionDef) and n.name in {'metrics','summarize'}:exec(compile(ast.Module(body=[n],type_ignores=[]),'verified_metrics','exec'),namespace)
meta={k:np.load(META/'validation'/(k+'.npy'),mmap_mode='r') for k in ['embedding','group','target','mask','refs']}
weights=policy['global_graph_local_weights'];rows={str(a):[] for a in weights};predictions={str(a):np.zeros((50861,16),dtype=np.float64) for a in weights};tick=time.perf_counter();seen=0
for number,g in enumerate(fm['groups']):
    ids=np.flatnonzero(meta['group']==number)
    if not len(ids):continue
    info=bundle['contracts'][g];embedding=np.asarray(meta['embedding'][ids],dtype=np.float64);bank=decoder.bank(g)
    if info['training_rows']<256:
        with torch.no_grad():prior=net.encoder.head(torch.tensor(embedding,dtype=torch.float32),[g]*len(ids)).to(torch.float64).numpy()
        candidates={str(a):prior for a in weights}
    else:
        spec=manifest['topologies'][info['topology']];folder=ROOT/spec['folder'];raw_indices=np.asarray(meta['refs'][ids,1]);blocks=[]
        for key in ['values','known','zero']:
            columns=np.array([int(c) for k,c in bank['raw_columns'] if k==key],dtype=np.int64)
            if not len(columns):continue
            mm=np.load(folder/(key+'.npy'),mmap_mode='r');flat=mm.reshape(len(mm),-1);blocks.append(np.asarray(flat[raw_indices[:,None],columns[None,:]],dtype='float64'));mm._mmap.close()
        raw=np.concatenate(blocks,1);device,combined=decoder.query(g,embedding,raw);candidates={str(a):(1-a)*device+a*combined for a in weights}
    target=np.asarray(meta['target'][ids]);mask=np.asarray(meta['mask'][ids]);scales=bank['scales']
    for key,z in candidates.items():
        predictions[key][ids]=z;physical=np.sign(z)*np.expm1(abs(z))*scales;rows[key]+=namespace['metrics'](physical,target,mask,g)
    seen+=len(ids)
assert seen==50861 and all(len(r)==1491 for r in rows.values())
parent=read(ROOT/'reports/experiment1_validation_review.json');trials=[]
for key,value in rows.items():
    score=namespace['summarize'](value)
    guards=all(score['by_source'][s]['macro_hit5']>=r['macro_hit5']-.01 and score['by_source'][s]['median_r2']>=r['median_r2']-.03 for s,r in parent['sources'].items())
    accepted=score['macro_hit5']>parent['macro_hit5']+1e-4 and score['median_r2']>=parent['median_r2']-.01 and score['macro_r2']>=parent['macro_r2'] and score['negative_r2_targets']<=parent['negative_r2_targets'] and guards
    trials.append({'graph_local_weight':float(key),'validation':score,'passed':accepted})
accepted=[t for t in trials if t['passed']]
chosen=max(accepted,key=lambda t:t['validation']['macro_hit5']) if accepted else None
if chosen:
    bundle['graph_local_weight']=chosen['graph_local_weight'];bundle['validation_selection']=chosen['validation'];(OUT/'bundle.json').write_text(json.dumps(bundle,indent=2),encoding='utf-8')
    key=str(chosen['graph_local_weight']);np.save(OUT/'validation_prediction_symlog.npy',predictions[key],allow_pickle=False)
    (OUT/'validation_metrics.json').write_text(json.dumps(dict(chosen['validation'],per_target=rows[key]),indent=2),encoding='utf-8')
summary={'state':'complete','selected':chosen,'trials':trials,'policy':policy,'seconds':time.perf_counter()-tick,'validation_rows':seen,'test_read':False,'sparse_contracts':sum(s['training_rows']<256 for s in bundle['contracts'].values())}
(OUT/'selection.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');(ROOT/'reports/rgcn_local_head_selection.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
