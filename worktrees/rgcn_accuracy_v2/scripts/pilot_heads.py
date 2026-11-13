"""Bounded CPU head trial using a read-only trained backbone and real held-out data.

No CUDA, no test labels, no production writes. This is a diagnostic validation
subset, not a replacement for full-validation model selection.
"""
import os
import sys
from pathlib import Path
import argparse
import ast
import copy
import gc
import hashlib
import json
import threading
import time
from types import SimpleNamespace,MethodType
import psutil

PACKAGE=Path(__file__).resolve().parents[1];ROOT=PACKAGE.parents[1]
sys.path.insert(0,str(PACKAGE))

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--train-per-group',type=int,default=16)
    parser.add_argument('--validation-per-group',type=int,default=8);parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--max-seconds',type=int,default=480);args=parser.parse_args()
    if psutil.virtual_memory().available<1.5*2**30:raise SystemExit('CPU pilot deferred: need 1.5 GiB available')
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    output=PACKAGE/'results/head_pilot_v1'
    if output.exists():raise FileExistsError('Pilot results already exist; do not overwrite')
    output.mkdir();stop=threading.Event();tick=time.perf_counter();resources=[]
    def guard():
        process=psutil.Process()
        while not stop.wait(.1):
            available=psutil.virtual_memory().available;resources.append((process.memory_info().rss,available))
            if available<768*2**20 or time.perf_counter()-tick>args.max_seconds:
                write(output/'summary.json',{'status':'aborted_resource_guard','available_RAM_GiB':available/2**30,
                    'wall_seconds':time.perf_counter()-tick,'production_touched':False,'CUDA_used':False});os._exit(86)
    threading.Thread(target=guard,daemon=True).start()
    # Copy one atomic best checkpoint; the running trainer remains unmodified.
    production=ROOT/'artifacts/experiment1_fusion_id_fast_v1';original=production/'best.pt';checkpoint=output/'parent_best.pt'
    before=sha(original)
    with original.open('rb') as source,checkpoint.open('xb') as dest:
        for block in iter(lambda:source.read(2**20),b''):dest.write(block)
    if sha(checkpoint)!=before:raise RuntimeError('Parent checkpoint changed during copy; retry separately')
    import numpy as np
    import torch
    from src.model.network import EdgeCentricRGCN
    from src.model.contract_head import ContractResidualHead
    from src.graph.builder import CircuitBatch
    from src.train.losses import symlog
    from src.train.metrics import MetricAccumulator
    from src.train.accuracy_loss import supervision_loss
    from src.train.target_calibration import validate
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(42)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True);config=saved['config']
    cache=ROOT/config['cache'];manifest=read(cache/'manifest.json');schema=read(cache/'schema.json');tasks=read(cache/'tasks.json')
    calibration=read(ROOT/'data/model_ready/fusion_v2_accuracy_v2/target_calibration.json');validate(calibration,tasks,manifest)
    if saved['schema']!=schema or saved['tasks']!=tasks:raise ValueError('Parent vocabulary differs')
    model=EdgeCentricRGCN(schema,tasks,**{k:config[k] for k in
        ['hidden_dim','num_layers','dropout','update_edges','use_domain_emb','use_film','collapse_relations']})
    model.load_state_dict(saved['model'],strict=True);model.eval();parent_head=copy.deepcopy(model.head)
    parent_epoch=saved['epoch'];del saved;gc.collect()
    # Exact existing cache batch method, with only selected training/validation rows.
    tree=ast.parse((PACKAGE/'src/train/fusion_cache.py').read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CachedFusion')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='batch')
    namespace={'np':np,'torch':torch,'CircuitBatch':CircuitBatch}
    exec(compile(ast.Module(body=[method],type_ignores=[]),'unchanged_CachedFusion_batch','exec'),namespace)
    probe=SimpleNamespace(manifest=manifest,schema=schema,tasks=tasks,arrays={},static={})
    probe.batch=MethodType(namespace['batch'],probe);candidates={0:{},1:{}}
    for name,spec in sorted(manifest['topologies'].items()):
        folder=ROOT/spec['folder'];a={key:np.load(folder/(key+'.npy'),mmap_mode='r') for key in
            ['values','known','zero','target','mask','role','group','domain','sample_ids']}
        probe.arrays[name]=a;probe.static[name]=read(folder/'structure.json')
        for role in [0,1]:
            rows=np.flatnonzero(a['role']==role)
            for number in np.unique(a['group'][rows]):
                group=manifest['group_names'][int(number)]
                candidates[role].setdefault(group,[]).extend((name,int(i)) for i in rows[a['group'][rows]==number])
    rng=np.random.default_rng(42);refs={}
    for role,size in [(0,args.train_per_group),(1,args.validation_per_group)]:
        refs[role]=[]
        for group,items in sorted(candidates[role].items()):
            choice=rng.choice(len(items),min(size,len(items)),replace=False)
            refs[role].extend(items[int(i)] for i in choice)
    del candidates
    extracted={}
    with torch.no_grad():
        for role,label in [(0,'train'),(1,'validation')]:
            chunks=[]
            for start in range(0,len(refs[role]),8):
                batch=probe.batch(refs[role][start:start+8],label);states=model(batch,True)
                chunks.append({'embedding':states['embedding'].clone(),'targets':batch.targets,'mask':batch.mask,'scales':batch.scales,
                    'group_ids':batch.group_ids,'semantic_ids':batch.semantic_ids,'sample_ids':batch.sample_ids})
                del batch,states
            extracted[label]={key:torch.cat([c[key] for c in chunks]) if key in ['embedding','targets','mask','scales']
                else sum((c[key] for c in chunks),[]) for key in chunks[0]}
            print('Extracted',label,len(refs[role]),'real graphs',round(time.perf_counter()-tick,2),'s',flush=True)
    train_ids=set(extracted['train']['sample_ids']);val_ids=set(extracted['validation']['sample_ids'])
    assert not train_ids&val_ids
    # All graph tensors/weights can now be released; only small cached embeddings remain.
    del model,probe;gc.collect()
    def batch_rows(data,indices,role):
        groups=[data['group_ids'][i] for i in indices];n=len(indices)
        return CircuitBatch(x=torch.zeros(n,34),edge_index=torch.empty(2,0,dtype=torch.long),
            edge_attr=torch.empty(0,90),edge_type=torch.empty(0,dtype=torch.long),graph_index=torch.arange(n),
            targets=data['targets'][indices],mask=data['mask'][indices],scales=data['scales'][indices],
            group_ids=groups,semantic_ids=[data['semantic_ids'][i] for i in indices],sample_ids=[data['sample_ids'][i] for i in indices],split_role=role)
    # The zero-edge batch above carries head loss/routing metadata only. Real
    # netlist embeddings were computed above; it is never sent through a GNN.
    def evaluate(head,data):
        head.eval();metric=MetricAccumulator(tasks);sums={}
        with torch.no_grad():
            for start in range(0,len(data['embedding']),64):
                indices=list(range(start,min(start+64,len(data['embedding']))));batch=batch_rows(data,indices,'validation')
                prediction=head(data['embedding'][indices],batch.group_ids);metric.update(prediction,batch)
        result=metric.compute();valid=[r['r2'] for r in result['per_target'] if r['r2'] is not None]
        result['median_r2']=float(np.median(valid)) if valid else None;result['by_source']={}
        for source in ['FALCON','AICircuit','LOADBench']:
            rows=[r for r in result['per_target'] if tasks[r['supervision_group']]['contract']['source_dataset']==source]
            def average(field):
                values=[r[field] for r in rows if r[field] is not None]
                return float(np.mean(values)) if values else None
            r2=[r['r2'] for r in rows if r['r2'] is not None]
            result['by_source'][source]={'target_combinations':len(rows),'macro_hit5':average('hit5'),
                'median_r2':float(np.median(r2)) if r2 else None}
        return result
    baseline=ContractResidualHead(128,tasks,calibration,base=copy.deepcopy(parent_head),contract_residual=False,calibrated_targets=False)
    initial=evaluate(baseline,extracted['validation']);write(output/'unchanged_parent_validation.json',initial)
    train=extracted['train'];training_diagnostics={};reset_rows=[]
    with torch.no_grad():
        for start in range(0,len(train['embedding']),64):
            ix=list(range(start,min(start+64,len(train['embedding']))));p=parent_head(train['embedding'][ix],[train['group_ids'][i] for i in ix])
            y=symlog(train['targets'][ix]/train['scales'][ix])
            for j,i in enumerate(ix):
                group=train['group_ids'][i]
                for slot in torch.nonzero(train['mask'][i]).flatten().tolist():
                    record=calibration['groups'][group]['slots'][str(slot)];item=training_diagnostics.setdefault((group,slot),[0.,0.,0])
                    item[0]+=float((p[j,slot]-y[j,slot]).square());item[1]+=float((record['center']-y[j,slot]).square());item[2]+=1
    for key,(error,constant,n) in training_diagnostics.items():
        if error>constant:reset_rows.append(key)
    variants={};initial_max_error={}
    for name,residual,normalized in [('control',False,False),('head_only',True,False),('calibration_only',False,True),('combined',True,True)]:
        torch.manual_seed(42)
        head=ContractResidualHead(128,tasks,calibration,base=copy.deepcopy(parent_head),contract_residual=residual,calibrated_targets=normalized)
        if normalized:
            # Explicit weights-only warm start: preserve every learned prediction
            # except rows worse than the fixed training-derived mean on training
            # examples. No validation metric participates in this initialization.
            with torch.no_grad():
                for group,families in head.base.slots.items():
                    for family,slots in families.items():
                        layer=head.base.readouts[group][family]
                        for row,slot in enumerate(slots):
                            value=calibration['groups'][group]['slots'][str(slot)];scale=value['std'];center=value['center']
                            if (group,slot) in reset_rows:layer.weight[row].zero_();layer.bias[row].zero_()
                            else:layer.weight[row].div_(scale);layer.bias[row].sub_(center).div_(scale)
        optimizer=torch.optim.AdamW(head.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        local_rng=np.random.default_rng(42);losses=[];head.train();started=time.perf_counter()
        for step in range(args.steps):
            indices=local_rng.choice(len(train['embedding']),min(64,len(train['embedding'])),replace=False).tolist()
            batch=batch_rows(train,indices,'train');optimizer.zero_grad(set_to_none=True)
            prediction=head(train['embedding'][indices],batch.group_ids);loss,_=supervision_loss(prediction,batch,head)
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);optimizer.step()
            losses.append(float(loss.detach()))
        result=evaluate(head,extracted['validation']);result.update(training_steps=args.steps,training_wall_seconds=time.perf_counter()-started,
            first_training_loss=losses[0],last_training_loss=losses[-1],train_only_reset_rows=len(reset_rows) if normalized else 0)
        write(output/(name+'_validation.json'),result)
        variants[name]={k:v for k,v in result.items() if k!='per_target'}
        print(name,'hit5',result['macro_hit5'],'medianR2',result['median_r2'],'seconds',round(result['training_wall_seconds'],2),flush=True)
        torch.save({'head':head.state_dict(),'variant':name,'training_steps':args.steps},output/(name+'_head.pt'))
        del head,optimizer,batch,prediction,loss;gc.collect()
    stop.set()
    summary={'status':'complete','scope':'CPU frozen-trained-backbone head pilot, stratified real training/validation subset; not full validation',
        'seed':42,'parent_checkpoint_sha256':before,'parent_best_epoch':parent_epoch,'parent_checkpoint_file':str(checkpoint.relative_to(ROOT)),
        'train_rows':len(train_ids),'validation_rows':len(val_ids),'train_validation_sample_overlap':0,
        'test_labels_used':False,'graph_backbone_frozen':True,'initial_parent_validation':{k:v for k,v in initial.items() if k!='per_target'},
        'normalized_initialization':'weights-only affine reparameterization; reset rows worse than train-derived mean on selected train examples',
        'train_only_reset_rows':len(reset_rows),'variants':variants,'CUDA_used':False,'production_touched':False,
        'elapsed_seconds':time.perf_counter()-tick,'CPU_peak_RSS_MiB':max(v[0] for v in resources)/2**20,
        'minimum_available_RAM_MiB':min(v[1] for v in resources)/2**20,
        'training_sample_ids':sorted(train_ids),'validation_sample_ids':sorted(val_ids),
        'warning':'Subset has few observations per target. Its R2 is unstable. Pilot variants are diagnostic, not directly comparable to full-dataset metrics or fresh-start configs.'}
    write(output/'summary.json',summary);print('Pilot complete',output,flush=True)

if __name__=='__main__':main()
