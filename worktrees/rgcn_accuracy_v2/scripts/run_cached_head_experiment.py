"""Frozen pretrained RGCN + complete real train/validation cached-feature stage.

CUDA runs only after the baseline has saved and exited and >=4 GiB RAM is free.
This explicit new stage trains heads only, has its own recipe/checkpoints, and
never uses test labels unless full-validation promotion criteria are satisfied.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse
import sys
import json
import hashlib
import time
import threading
import copy
import gc
from types import SimpleNamespace
from datetime import datetime,timezone
import psutil
ROOT=Path(__file__).resolve().parents[3];PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE));sys.path.append(str(ROOT))
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for value in iter(lambda:f.read(2**20),b''):h.update(value)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def gate():
    for p in psutil.process_iter(['name','cmdline']):
        if p.pid in {os.getpid(),os.getppid()}:continue
        if (p.info['name'] or '').lower() in {'python.exe','pythonw.exe'} and any(
            Path(a.replace('\\','/')).name in {'train_experiment1.py','train_experiment1_fast.py','train_accuracy_v2.py','run_cached_head_experiment.py'} for a in p.info['cmdline'] or []):
            raise RuntimeError('Other trainer still running: '+str(p.pid))
    if psutil.virtual_memory().available<4*2**30:raise RuntimeError('CUDA gate: need at least 4 GiB available RAM')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);args=parser.parse_args();gate()
    plan=read(ROOT/args.plan);output=ROOT/plan['output'];cache=ROOT/plan['feature_cache']
    output.mkdir(parents=True,exist_ok=True)
    if (output/'recipe.json').exists():raise FileExistsError('Fresh stage only; refusing overwrite/resume without explicit validated implementation')
    import numpy as np
    import torch
    import src.common as common
    common.ROOT=ROOT
    from src.windows_campaign_io import atomic_json,replace_with_retry
    from src.train.fusion_cache import CachedFusion
    from src.model.network import EdgeCentricRGCN
    from src.model.accuracy_network import AccuracyRGCN
    from src.model.grouped_contract_head import forward_grouped
    from src.graph.builder import CircuitBatch
    from src.model.execution import prepare_batch
    from src.train.target_calibration import validate as validate_calibration
    from src.train.accuracy_loss import supervision_loss as graph_loss
    from src.train.metrics import MetricAccumulator
    from src.train.accuracy_loop import evaluate as evaluate_graph
    from src.train.loop import TrainingCoverage
    torch.set_num_threads(2);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.80)
    status={'pid':os.getpid(),'state':'initializing','updated_at':now(),'test_used':False,'protocol':'frozen_pretrained_backbone_calibrated_head_stage'}
    def update(**values):status.update(values,updated_at=now());atomic_json(output/'status.json',status)
    def save(path,value):
        temporary=path.with_suffix(path.suffix+'.tmp');torch.save(value,temporary);replace_with_retry(temporary,path)
    stop=threading.Event();resources=[]
    def resource_guard():
        process=psutil.Process()
        while not stop.wait(.2):
            available=psutil.virtual_memory().available;resources.append((process.memory_info().rss,available))
            if available<640*2**20:
                atomic_json(output/'resource_abort.json',{'at':now(),'reason':'Diagnostic/stage exited below 640 MiB; parent remains recoverable'});os._exit(86)
    threading.Thread(target=resource_guard,daemon=True).start()
    parent_path=ROOT/plan['parent_checkpoint'];assert sha(parent_path)==plan['parent_checkpoint_sha256']
    parent=torch.load(parent_path,map_location='cpu',weights_only=True);config=parent['config'];calibration=read(ROOT/plan['target_calibration'])
    data=CachedFusion(config);validate_calibration(calibration,data.tasks,data.manifest)
    assert sha(data.path/'manifest.json')==calibration['cache_manifest_sha256']
    schema,tasks=data.schema,data.tasks;names=sorted(tasks);group_index={g:i for i,g in enumerate(names)}
    centers=np.zeros((len(names),16),dtype=np.float64);scales=np.ones_like(centers)
    for group in names:
        for slot,rule in tasks[group]['slots'].items():
            centers[group_index[group],int(slot)]=calibration['groups'][group]['slots'][slot]['center'];scales[group_index[group],int(slot)]=rule['scale']
    model_keys=['hidden_dim','num_layers','dropout','update_edges','use_domain_emb','use_film','collapse_relations']
    body=EdgeCentricRGCN(schema,tasks,**{k:config[k] for k in model_keys}).to('cuda');body.load_state_dict(parent['model'],strict=True)
    body.eval();body.requires_grad_(False);cache.mkdir(parents=True,exist_ok=True)
    parent_metric=MetricAccumulator(tasks);model_sse=np.zeros_like(centers);constant_sse=np.zeros_like(centers);nobs=np.zeros_like(centers,dtype=np.int64)
    cache_specs={};topologies=list(data.arrays);top_index={name:i for i,name in enumerate(topologies)}
    build_started=time.perf_counter()
    for role in ['train','validation']:
        update(state='building_feature_cache',role=role,samples_done=0)
        refs=data.references(role,42);folder=cache/role;folder.mkdir(exist_ok=True);count=len(refs)
        arrays={key:np.lib.format.open_memmap(folder/(key+'.npy'),mode='w+',dtype=dtype,shape=shape) for key,dtype,shape in
            [('embedding','float32',(count,128)),('target','float32',(count,16)),('mask','bool',(count,16)),('group','int16',(count,)),('refs','int32',(count,2))]}
        with torch.no_grad():
            for start in range(0,count,256):
                selected=refs[start:start+256];batch=data.batch(selected,role).to('cuda');states=body(batch,True)
                end=start+len(selected);embedding=states['embedding'].cpu().numpy();p=states['prediction'].cpu().numpy().astype(np.float64)
                target=batch.targets.cpu().numpy();mask=batch.mask.cpu().numpy();groups=np.array([group_index[g] for g in batch.group_ids])
                arrays['embedding'][start:end]=embedding;arrays['target'][start:end]=target;arrays['mask'][start:end]=mask;arrays['group'][start:end]=groups
                arrays['refs'][start:end]=np.array([(top_index[name],row) for name,row in selected],dtype=np.int32)
                if role=='train':
                    z=np.sign(target)*np.log1p(np.abs(target.astype(np.float64)/scales[groups]))
                    np.add.at(model_sse,groups,np.square(p-z)*mask);np.add.at(constant_sse,groups,np.square(centers[groups]-z)*mask);np.add.at(nobs,groups,mask)
                else:parent_metric.update(states['prediction'],batch)
                if start%2560==0 or end==count:
                    elapsed=time.perf_counter()-build_started
                    update(role=role,samples_done=end,role_samples=count,cache_build_seconds=elapsed)
                    print(now(),'cache',role,end,'/',count,flush=True)
                del batch,states
        for array in arrays.values():array.flush()
        del arrays,refs;gc.collect()
        cache_specs[role]={'rows':count,'files':{p.name:sha(p) for p in folder.glob('*.npy')}}
    original_validation=parent_metric.compute()
    def add_summary(metrics):
        rows=metrics['per_target'];r2=[r['r2'] for r in rows if r['r2'] is not None]
        metrics['median_r2']=float(np.median(r2)) if r2 else None;metrics['by_source']={}
        for source in ['FALCON','AICircuit','LOADBench']:
            subset=[r for r in rows if tasks[r['supervision_group']]['contract']['source_dataset']==source]
            hits=[r['hit5'] for r in subset if r['hit5'] is not None];r2s=[r['r2'] for r in subset if r['r2'] is not None]
            metrics['by_source'][source]={'macro_hit5':float(np.mean(hits)) if hits else None,'median_r2':float(np.median(r2s)) if r2s else None}
        return metrics
    add_summary(original_validation);atomic_json(output/'parent_validation.json',original_validation)
    cache_manifest={'parent_checkpoint_sha256':plan['parent_checkpoint_sha256'],'parent_epoch':parent['epoch'],
        'source_cache_manifest_sha256':sha(data.path/'manifest.json'),'split_sha256':data.manifest['split_sha256'],
        'groups':names,'topologies':topologies,'roles':cache_specs,'dropout_in_backbone':'eval; all embedding-producing weights frozen',
        'test_features_or_labels_cached':False,'feature_shape':128}
    atomic_json(cache/'manifest.json',cache_manifest)
    # Explicit warm start, not optimizer/resume migration. Graph weights stay exact.
    candidate_config=dict(config,output=plan['output'],contract_residual=False,calibrated_targets=True,adapter_hidden=32,adapter_min_train_samples=256,
        target_calibration=plan['target_calibration'],head_batch_size=plan['head_batch_size'],epochs=plan['head_epochs'],patience=plan['patience'],
        stage='frozen_all_embedding_parameters_train_head',precision='FP32; train-only symlog calibration')
    candidate=AccuracyRGCN(schema,tasks,calibration,**{k:config[k] for k in model_keys},contract_residual=False,calibrated_targets=True).to('cuda')
    current=candidate.state_dict();mapped={('head.base.'+k[5:] if k.startswith('head.') else k):v for k,v in parent['model'].items()}
    assert set(mapped)<=set(current);current.update(mapped);candidate.load_state_dict(current,strict=True)
    resets=[]
    with torch.no_grad():
        for group,families in candidate.head.base.slots.items():
            for family,slots in families.items():
                layer=candidate.head.base.readouts[group][family]
                for i,slot in enumerate(slots):
                    record=calibration['groups'][group]['slots'][str(slot)]
                    if nobs[group_index[group],slot] and model_sse[group_index[group],slot]>constant_sse[group_index[group],slot]:
                        layer.weight[i].zero_();layer.bias[i].zero_();resets.append((group,slot))
                    else:layer.weight[i].div_(record['std']);layer.bias[i].sub_(record['center']).div_(record['std'])
    candidate.requires_grad_(False);candidate.head.requires_grad_(True);candidate.eval()
    del body,parent;gc.collect();torch.cuda.empty_cache()
    head=candidate.head;grouped=False;comparisons=[]
    def compare(name,a,b,atol,rtol):
        a=a.detach().cpu().double();b=b.detach().cpu().double();error=(a-b).abs()
        maximum=float(error.max()) if a.numel() else 0.;ratio=float((error/(atol+rtol*a.abs())).max()) if a.numel() else 0.
        comparisons.append({'name':name,'max_abs':maximum,'max_relative_floor_1e8':float((error/torch.maximum(torch.maximum(a.abs(),b.abs()),torch.tensor(1e-8))).max()) if a.numel() else 0.,'acceptance_ratio':ratio,'atol':atol,'rtol':rtol})
        if ratio>1 or not torch.isfinite(b).all():raise AssertionError(name+' exceeds unchanged tolerance: '+str(comparisons[-1]))
    fixture=torch.load(ROOT/'reports/experiment1_performance/fixtures/batch_256_11.pt',map_location='cpu',weights_only=True)
    cpu_batch=CircuitBatch(**fixture);gpu_batch=cpu_batch.to('cuda');backup=copy.deepcopy(head.state_dict())
    cpu_head=copy.deepcopy(head).cpu();cpu_head.train();head.train()
    with torch.no_grad():states=candidate(gpu_batch,True);embedding=states['embedding'].detach()
    compare('cached_head_vs_full_graph_forward',states['prediction'],head(embedding,gpu_batch.group_ids),5e-6,5e-5)
    def verify(use_grouped):
        head.load_state_dict(backup);cpu_head.load_state_dict({k:v.cpu() for k,v in backup.items()});head.zero_grad(set_to_none=True);cpu_head.zero_grad(set_to_none=True)
        p=forward_grouped(head,embedding,gpu_batch.group_ids,prepare_batch(gpu_batch).grouping()) if use_grouped else head(embedding,gpu_batch.group_ids)
        q=cpu_head(embedding.cpu(),cpu_batch.group_ids);compare('GPU_vs_CPU_prediction_'+str(use_grouped),q,p,5e-6,5e-5)
        left,_=graph_loss(q,cpu_batch,cpu_head);right,_=graph_loss(p,gpu_batch,head);compare('GPU_vs_CPU_loss_'+str(use_grouped),left,right,2e-6,2e-5)
        left.backward();right.backward()
        for (name,a),(other,b) in zip(cpu_head.named_parameters(),head.named_parameters()):
            assert name==other and (a.grad is None)==(b.grad is None)
            if a.grad is not None:compare('gradient/'+name,a.grad,b.grad,2e-5,2e-3)
        torch.nn.utils.clip_grad_norm_(cpu_head.parameters(),5.,error_if_nonfinite=True);torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True)
        opt_cpu=torch.optim.AdamW(cpu_head.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        opt_gpu=torch.optim.AdamW(head.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay']);opt_cpu.step();opt_gpu.step()
        for (name,a),(other,b) in zip(cpu_head.named_parameters(),head.named_parameters()):compare('adamw/'+name,a,b,2e-5,2e-4)
    update(state='GPU_acceptance')
    verify(False)
    try:verify(True);grouped=True
    except AssertionError as error:
        atomic_json(output/'grouped_execution_rejected.json',{'reason':str(error),'fallback':'original per-sample head; no tolerance relaxation'})
    head.load_state_dict(backup);head.zero_grad(set_to_none=True);del cpu_head,backup,states,gpu_batch,cpu_batch,embedding;gc.collect();torch.cuda.empty_cache()
    atomic_json(output/'GPU_acceptance.json',{'status':'passed','batch_size':256,'relations':48,'FP32':True,'deterministic':True,
        'grouped_head_accepted':grouped,'comparisons':comparisons,'scope':'CPU/GPU new head prediction/loss/gradient/AdamW; full real GNN vs cached-embedding forward',
        'frozen_body_unchanged':True})
    # Only small metadata arrays stay mapped during head training.
    ids={name:a['sample_ids'] for name,a in data.arrays.items()};del data;gc.collect()
    feature_data={role:{key:np.load(cache/role/(key+'.npy'),mmap_mode='r') for key in ['embedding','target','mask','group','refs']} for role in ['train','validation']}
    engineering=torch.tensor(scales,dtype=torch.float32,device='cuda')
    def make_batch(role,indices):
        arrays=feature_data[role];numbers=np.asarray(arrays['group'][indices]);group_ids=[names[int(i)] for i in numbers]
        mask=np.asarray(arrays['mask'][indices]);order=[];loss_slices=[];row_order=[];group_slices=[];coverage={}
        for number in sorted(set(numbers.tolist())):
            group=names[number];rows=np.flatnonzero(numbers==number);observed=mask[rows]
            positions=(rows[:,None]*16+np.arange(16)[None,:])[observed];start=len(order);order.extend(positions.tolist())
            if len(positions):loss_slices.append((group,start,len(order)))
            start=len(row_order);row_order.extend(rows.tolist());group_slices.append((group,start,len(row_order)))
            coverage[group]=np.flatnonzero(observed.any(0)).tolist()
        refs=np.asarray(arrays['refs'][indices]);sample_ids=[ids[topologies[int(t)]][int(row)].decode('utf-8') for t,row in refs]
        meta=SimpleNamespace(targets=torch.from_numpy(np.asarray(arrays['target'][indices]).copy()).to('cuda'),mask=torch.from_numpy(mask.copy()).to('cuda'),
            scales=engineering[torch.tensor(numbers.astype(np.int64),device='cuda')],group_ids=group_ids,
            sample_ids=sample_ids,semantic_ids=[[tasks[g]['slots'].get(str(i),{}).get('semantic_id') for i in range(16)] for g in group_ids],
            loss_order=torch.tensor(order,dtype=torch.long,device='cuda'),loss_slices=loss_slices,
            group_plan=(group_slices,torch.tensor(row_order,dtype=torch.long,device='cuda')),coverage=coverage)
        x=torch.from_numpy(np.asarray(arrays['embedding'][indices]).copy()).to('cuda');return x,meta
    def predict(x,meta):return forward_grouped(head,x,meta.group_ids,meta.group_plan) if grouped else head(x,meta.group_ids)
    def objective(prediction,meta):
        head.check_observed(meta.coverage);target=meta.targets.sign()*torch.log1p((meta.targets/meta.scales).abs())
        _,std=head.affine(meta.group_ids);losses=[];finite=[]
        for group,start,end in meta.loss_slices:
            positions=meta.loss_order[start:end];p=prediction.reshape(-1)[positions];y=target.reshape(-1)[positions];s=std.reshape(-1)[positions]
            finite.append(torch.isfinite(p).all() & torch.isfinite(y).all());losses.append(((p-y)/s).square().mean())
        if not losses or not torch.stack(finite).all():raise ValueError('Invalid observed cached-head loss')
        return torch.stack(losses).mean()
    # Formula cross-check on the same real fixture, before processing the stage.
    with torch.no_grad():
        x,meta=make_batch('train',list(range(256)));prediction=predict(x,meta);assert torch.isfinite(objective(prediction,meta))
    source_files=sorted((PACKAGE/'src').rglob('*.py'))+[Path(__file__).resolve()]
    recipe={'execution_variant':'frozen_rgcn_calibrated_head_v1','config':candidate_config,'plan':plan,
        'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in source_files},
        'feature_cache_manifest_sha256':sha(cache/'manifest.json'),'target_calibration_sha256':sha(ROOT/plan['target_calibration']),
        'torch':torch.__version__,'GPU':torch.cuda.get_device_name(0),'grouped_execution':grouped,
        'initialization':'Explicit pretrained weights; affine reparameterization, reset rows worse than fixed mean using full training labels only; new head optimizer',
        'reset_rows':resets,'validation_for_selection_only':True,'test_policy':'once only if predeclared promotion criteria pass'}
    atomic_json(output/'recipe.json',recipe);atomic_json(output/'config.json',candidate_config)
    optimizer=torch.optim.AdamW(head.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=plan['head_epochs'])
    best=-float('inf');best_epoch=None;bad=0;history=[]
    def save_best(epoch,metrics):
        save(output/'best.pt',{'model':candidate.state_dict(),'target_calibration':calibration,'schema':schema,'tasks':tasks,'config':candidate_config,
            'status':'trained','model_class':'AccuracyRGCN','epoch':epoch,'recipe_digest':digest(recipe),
            'trained_group_ids':names,'trained_target_slots':{g:sorted(candidate.head.trained_slots[g]) for g in names},
            'trained_domain_ids':[d['id'] for d in schema['domains']],'parent_pretraining_epoch':plan['parent_best_epoch'],
            'validation':{k:v for k,v in metrics.items() if k!='per_target'}})
    for epoch in range(plan['head_epochs']):
        head.train();indices=np.random.default_rng(42+epoch).permutation(cache_specs['train']['rows']);began=time.perf_counter();loss_sum=0.;steps=0
        for start in range(0,len(indices),plan['head_batch_size']):
            if (output/'STOP').exists():raise RuntimeError('Cached-head STOP requested; controller restores parent')
            selected=indices[start:start+plan['head_batch_size']];x,meta=make_batch('train',selected)
            optimizer.zero_grad(set_to_none=True);prediction=predict(x,meta);loss=objective(prediction,meta);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);optimizer.step();loss_sum+=float(loss.detach());steps+=1
            if steps%20==0 or start+len(selected)==len(indices):
                seconds=time.perf_counter()-began;update(state='training_heads',epoch=epoch+1,max_epochs=plan['head_epochs'],batch=steps,
                    batches_per_epoch=int(np.ceil(len(indices)/plan['head_batch_size'])),samples_done=start+len(selected),
                    samples_per_second=(start+len(selected))/seconds,training_loss=loss_sum/steps,body_frozen=True)
                print(now(),'head epoch',epoch+1,'samples',start+len(selected),'loss',loss_sum/steps,flush=True)
        train_seconds=time.perf_counter()-began;head.eval();metric=MetricAccumulator(tasks);update(state='validating',epoch=epoch+1)
        with torch.no_grad():
            for start in range(0,cache_specs['validation']['rows'],plan['head_batch_size']):
                ix=np.arange(start,min(start+plan['head_batch_size'],cache_specs['validation']['rows']));x,meta=make_batch('validation',ix)
                head.check_observed(meta.coverage);metric.update(predict(x,meta),meta)
        validation=add_summary(metric.compute());score=validation['macro_hit5']
        record={'epoch':epoch+1,'training_loss_mean_of_batch_group_means':loss_sum/steps,'training_seconds':train_seconds,
            'validation':validation,'learning_rate':optimizer.param_groups[0]['lr']};history.append(record)
        if score>best+1e-4:best=score;best_epoch=epoch+1;bad=0;save_best(epoch+1,validation)
        else:bad+=1
        scheduler.step();save(output/'last.pt',{'recipe_digest':digest(recipe),'model':candidate.state_dict(),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'next_epoch':epoch+1,'history':history,'best_epoch':best_epoch,'best_score':best,'bad':bad,
            'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'protocol':recipe['execution_variant']})
        atomic_json(output/'history.json',history);update(state='epoch_complete',completed_epochs=epoch+1,best_epoch=best_epoch,best_validation_score=best)
        print(now(),'validated head epoch',epoch+1,'hit5',score,'medianR2',validation['median_r2'],flush=True)
        if bad>=plan['patience']:break
    selected=torch.load(output/'best.pt',map_location='cuda',weights_only=True);candidate.load_state_dict(selected['model']);chosen=history[best_epoch-1]['validation']
    eligible=chosen['macro_hit5']>=original_validation['macro_hit5']+plan['minimum_hit5_gain'] and chosen['median_r2']>=original_validation['median_r2']-plan['maximum_median_r2_drop']
    promotion={'accepted':eligible,'at':now(),'parent_best_epoch':plan['parent_best_epoch'],'head_best_epoch':best_epoch,
        'parent_validation':{k:v for k,v in original_validation.items() if k!='per_target'},'candidate_validation':{k:v for k,v in chosen.items() if k!='per_target'},
        'criteria':{'minimum_hit5_gain':plan['minimum_hit5_gain'],'maximum_median_r2_drop':plan['maximum_median_r2_drop']},
        'test_used_for_selection':False}
    atomic_json(output/'promotion.json',promotion)
    if not eligible:
        update(state='rejected_restore_parent',test_used=False);stop.set();print('No promotion: controller restores parent.',flush=True);return
    update(state='testing_once',test_used=True);candidate.eval();data=CachedFusion(config);coverage=TrainingCoverage()
    coverage.slots={g:set(candidate.head.trained_slots[g]) for g in names};coverage.domains={d['id'] for d in schema['domains']}
    test=evaluate_graph(candidate,data.batches('test',256),coverage,device='cuda',role='test');add_summary(test)
    test.update(recipe_digest=digest(recipe),best_epoch=best_epoch,parent_pretraining_epoch=plan['parent_best_epoch'])
    atomic_json(output/'test_metrics.json',test);update(state='complete',completed_epochs=len(history),best_epoch=best_epoch,
        test_metrics='test_metrics.json',test_used=True,original_training_frozen=True)
    atomic_json(output/'resource_summary.json',{'CPU_peak_RSS_MiB':max(v[0] for v in resources)/2**20,
        'minimum_available_RAM_MiB':min(v[1] for v in resources)/2**20,'GPU_peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30})
    stop.set();print('Cached-head experiment complete; selected model evaluated on test once.',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        print('Cached-head stage error:',repr(error),flush=True);raise
