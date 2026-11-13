"""Small cached-head continuation; train/validation only, never reads test data.

Explicit new recipe inherits weights, AdamW moments and RNG from the six-epoch
stage. Its exhausted cosine schedule is replaced by a declared 5e-5 restart and
validation plateau schedule. Old artifacts and evaluated model stay immutable.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from pathlib import Path
import json
import hashlib
import time
import threading
import sys
import math
import tempfile
import psutil
from types import SimpleNamespace
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];PACKAGE=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(2**20),b''):h.update(data)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def replace_with_retry(source,target):
    source=Path(source).resolve();target=Path(target).resolve()
    if not source.is_relative_to(ROOT) or not target.is_relative_to(ROOT):raise ValueError('Write outside EDA')
    deadline=time.monotonic()+15
    while True:
        try:os.replace(source,target);return
        except OSError as error:
            if getattr(error,'winerror',None) not in {5,32,33} or time.monotonic()>=deadline:raise
            time.sleep(.1)
def atomic_json(path,value):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Write outside EDA')
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno());temporary=Path(stream.name)
    replace_with_retry(temporary,path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    cfg=read(PACKAGE/'config_cpu.json');output=(ROOT/cfg['output']).resolve()
    if output.parent!=ROOT/'artifacts' or output.name!='experiment1_cached_head_cpu_continue_v1':raise ValueError('Independent output required')
    if psutil.virtual_memory().available<cfg['minimum_start_RAM_GiB']*2**30:raise SystemExit(75)
    for process in psutil.process_iter(['name','cmdline']):
        if process.pid in {os.getpid(),os.getppid()}:continue
        if (process.info['name'] or '').lower() in {'python.exe','pythonw.exe'} and any(
            Path(a.replace('\\','/')).name in {'train_experiment1.py','train_experiment1_fast.py','train_accuracy_v2.py','run_cached_head_experiment.py'}
            or str(a).replace('\\','/').endswith('cached_head_continue_v1/train.py') for a in process.info['cmdline'] or []):
            raise RuntimeError('Another training process is active')
    source=ROOT/cfg['parent_output'];cache=ROOT/cfg['feature_cache'];version=ROOT/cfg['implementation']
    assert sha(source/'recipe.json')==cfg['parent_recipe_sha256']
    assert sha(source/'last.pt')==cfg['parent_last_sha256'] and sha(source/'best.pt')==cfg['parent_best_sha256']
    assert sha(cache/'manifest.json')==cfg['feature_cache_manifest_sha256']
    parent_recipe=read(source/'recipe.json');manifest=read(cache/'manifest.json')
    publication=read(version/'publication.json')
    for path,value in publication['source_sha256'].items():assert sha(version/path)==value,path
    for role in ['train','validation']:
        for name,value in manifest['roles'][role]['files'].items():assert sha(cache/role/name)==value,name
    files=sorted((version/'src').rglob('*.py'))+[Path(__file__),PACKAGE/'config_cpu.json']
    recipe={'execution_variant':'cached_head_validation_continue_cpu_v1','config':cfg,'parent_recipe_digest':digest(parent_recipe),
        'parent_last_sha256':cfg['parent_last_sha256'],'parent_best_sha256':cfg['parent_best_sha256'],
        'feature_cache_manifest_sha256':cfg['feature_cache_manifest_sha256'],
        'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files},
        'optimizer_policy':'Preserve AdamW moments and step counters; restart LR at 5e-5, ReduceLROnPlateau validation-only',
        'test_policy':cfg['test_policy'],'data_access_roles':['train','validation'],
        'memory_scope':'CPU-only cached head, measured ~304 MiB small-step RSS; no CUDA context; retain graph parameters unchanged'}
    output.mkdir(exist_ok=True,parents=True)
    if (output/'recipe.json').exists():
        if recipe!=read(output/'recipe.json'):raise ValueError('Recipe mismatch; no cross-version resume')
        if not args.resume:raise ValueError('Existing run requires --resume')
    elif args.resume:raise ValueError('No continuation recipe to resume')
    else:(output/'recipe.json').write_text(json.dumps(recipe,indent=2),encoding='utf-8')
    import numpy as np
    import torch
    import msvcrt
    sys.path.insert(0,str(version))
    from src.model.contract_head import ContractResidualHead
    from src.model.grouped_contract_head import forward_grouped
    from src.train.metrics import MetricAccumulator
    from src.train.target_calibration import validate
    lock=(output/'training.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    status={'state':'initializing','pid':os.getpid(),'create_time':psutil.Process().create_time(),'updated_at':now(),
        'test_used':False,'test_policy':'No further test evaluation','protocol':'frozen_backbone_cached_head_continuation_CPU','device':'cpu',
        'max_additional_epochs':cfg['additional_epochs'],'body_frozen':True}
    def update(**values):status.update(values,updated_at=now());atomic_json(output/'status.json',status)
    def save(path,value):
        temporary=path.with_suffix(path.suffix+'.tmp');torch.save(value,temporary);replace_with_retry(temporary,path)
    stop=threading.Event();resource=[]
    def guard():
        process=psutil.Process()
        while not stop.wait(.2):
            available=psutil.virtual_memory().available;resource.append((process.memory_info().rss,available))
            if available<256*2**20:
                atomic_json(output/'resource_abort.json',{'at':now(),'reason':'Head-only worker exited below384MiB; recover from last validated cursor'});os._exit(86)
    threading.Thread(target=guard,daemon=True).start()
    try:
        update();torch.set_num_threads(cfg['cpu_threads']);torch.use_deterministic_algorithms(True);torch.manual_seed(cfg['seed'])
        # CPU backend: no CUDA initialization or GPU allocations.
        parent=torch.load(source/'last.pt',map_location='cpu',weights_only=True)
        assert parent['recipe_digest']==digest(parent_recipe)
        base=torch.load(source/'best.pt',map_location='cpu',weights_only=True)
        schema,tasks,calibration=base['schema'],base['tasks'],base['target_calibration'];names=manifest['groups']
        assert names==sorted(tasks) and not manifest['test_features_or_labels_cached']
        assert int(manifest['roles']['train']['rows'])==400083 and int(manifest['roles']['validation']['rows'])==50861
        original_manifest=read(ROOT/base['config']['cache']/'manifest.json');validate(calibration,tasks,original_manifest)
        for role in ['train','validation']:
            refs=np.load(cache/role/'refs.npy',mmap_mode='r');encoded=(refs[:,0].astype(np.uint64)<<32)|refs[:,1].astype(np.uint64)
            if len(np.unique(encoded))!=len(encoded):raise ValueError('Duplicate cached sample references')
            if role=='train':train_keys=encoded
            else:
                if len(np.intersect1d(train_keys,encoded)):raise ValueError('Train/validation reference overlap')
        del train_keys,encoded,refs
        inherited_total=parent['next_epoch'];body_state={k:v for k,v in parent['model'].items() if not k.startswith('head.')}
        old_head={k[5:]:v for k,v in parent['model'].items() if k.startswith('head.')}
        head=ContractResidualHead(128,tasks,calibration,contract_residual=False,calibrated_targets=True).to('cpu')
        head.load_state_dict(old_head,strict=True)
        parameter_names=[name for name,_ in head.named_parameters()]
        old_order=[name for name in old_head if name in set(parameter_names)]
        if parameter_names!=old_order:raise ValueError('Inherited optimizer parameter order mismatch')
        optimizer=torch.optim.AdamW(head.parameters(),lr=cfg['initial_learning_rate'],weight_decay=base['config']['weight_decay'])
        optimizer.load_state_dict(parent['optimizer'])
        for group in optimizer.param_groups:group['lr']=cfg['initial_learning_rate'];group['initial_lr']=cfg['initial_learning_rate']
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=cfg['reduce_lr_factor'],
            patience=cfg['reduce_lr_patience'],threshold=1e-4,threshold_mode='abs',min_lr=cfg['minimum_learning_rate'])
        initial_best=parent['best_score'];best=initial_best;best_total=parent['best_epoch'];bad=0;history=[];start_epoch=0;start_batch=0;partial=None
        scheduler.best=initial_best
        model_config=dict(base['config'],output=cfg['output'],stage='validation_only_cached_head_continuation',
            learning_rate=cfg['initial_learning_rate'],epochs=cfg['additional_epochs'],additional_epochs=cfg['additional_epochs'],
            total_head_epoch_budget=inherited_total+cfg['additional_epochs'],patience=cfg['patience'],test_policy=cfg['test_policy'])
        initial_head_equal=all(torch.equal(v.cpu(),head.state_dict()[k].cpu()) for k,v in old_head.items())
        assert initial_head_equal
        def state_digest(state):
            h=hashlib.sha256()
            for key,value in sorted(state.items()):
                h.update(str(key).encode())
                if isinstance(value,dict):
                    for name,item in sorted(value.items()):
                        h.update(str(name).encode())
                        if isinstance(item,torch.Tensor):h.update(item.detach().cpu().contiguous().numpy().tobytes())
                        else:h.update(repr(item).encode())
            return h.hexdigest()
        assert state_digest(parent['optimizer']['state'])==state_digest(optimizer.state_dict()['state'])
        if args.resume and not (output/'last.pt').exists():args.resume=False
        if args.resume:
            saved=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
            if saved['recipe_digest']!=digest(recipe):raise ValueError('Continuation checkpoint fingerprint mismatch')
            head.load_state_dict(saved['head'],strict=True);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
            start_epoch=saved['next_additional_epoch'];start_batch=saved['next_batch'];partial=saved['partial']
            history=saved['history'];best=saved['best_score'];best_total=saved['best_total_head_epoch'];bad=saved['bad']
            torch.set_rng_state(saved['rng_cpu'].cpu());inherited_cuda_rng=saved['rng_cuda']
        else:
            torch.set_rng_state(parent['rng_cpu'].cpu());inherited_cuda_rng=parent['rng_cuda']
            backend=ROOT/cfg['backend_parent_output']
            assert sha(backend/'recipe.json')==cfg['backend_parent_recipe_sha256']
            assert sha(backend/'last.pt')==cfg['backend_parent_last_sha256']
            inherited=torch.load(backend/'last.pt',map_location='cpu',weights_only=True)
            assert inherited['recipe_digest']==digest(read(backend/'recipe.json'))
            head.load_state_dict(inherited['head'],strict=True);optimizer.load_state_dict(inherited['optimizer']);scheduler.load_state_dict(inherited['scheduler'])
            start_epoch=inherited['next_additional_epoch'];start_batch=inherited['next_batch'];partial=inherited['partial']
            history=inherited['history'];best=inherited['best_score'];best_total=inherited['best_total_head_epoch'];bad=inherited['bad']
            torch.set_rng_state(inherited['rng_cpu'].cpu());inherited_cuda_rng=inherited['rng_cuda']
            assert state_digest(inherited['optimizer']['state'])==state_digest(optimizer.state_dict()['state'])
            assert all(torch.equal(v,head.state_dict()[k]) for k,v in inherited['head'].items())
            atomic_json(output/'backend_migration.json',{'source':cfg['backend_parent_output'],'source_last_sha256':cfg['backend_parent_last_sha256'],
                'weights_and_AdamW_moments_exactly_inherited':True,'next_additional_epoch':start_epoch,'next_batch':start_batch,
                'partial_samples':partial['samples'] if partial else 0,'backend':'CPU FP32','test_data_or_scores_used':False})
            del inherited
            atomic_json(output/'fork_receipt.json',{'at':now(),'parent_last_sha256':cfg['parent_last_sha256'],
                'head_weights_exactly_inherited':True,'AdamW_moments_and_steps_exactly_inherited':True,
                'learning_rate_explicitly_restarted_from':parent['optimizer']['param_groups'][0]['lr'],
                'learning_rate_new':cfg['initial_learning_rate'],'previous_cosine_T_max':parent['scheduler']['T_max'],
                'inherited_head_epochs':inherited_total,'graph_pretraining_epochs':base['parent_pretraining_epoch'],
                'test_data_or_scores_used':False,'no_cross_recipe_resume':True,'original_artifacts_unchanged':True})
            initial=dict(base,config=model_config,recipe_digest=digest(recipe),continuation_parent=cfg['parent_output'],additional_head_epochs=0)
            save(output/'best.pt',initial)
        schema_metadata={'schema':schema,'tasks':tasks,'target_calibration':calibration,'config':model_config,
            'status':'trained','model_class':'AccuracyRGCN','trained_group_ids':base['trained_group_ids'],
            'trained_target_slots':base['trained_target_slots'],'trained_domain_ids':base['trained_domain_ids'],
            'parent_pretraining_epoch':base['parent_pretraining_epoch'],'parent_head_epochs':inherited_total}
        def save_cursor(epoch,batch,stats):
            save(output/'last.pt',{'recipe_digest':digest(recipe),'protocol':recipe['execution_variant'],'head':head.state_dict(),
                'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'next_additional_epoch':epoch,'next_batch':batch,
                'partial':stats,'history':history,'best_score':best,'best_total_head_epoch':best_total,'bad':bad,
                'rng_cpu':torch.get_rng_state(),'rng_cuda':inherited_cuda_rng})
        if not args.resume:save_cursor(start_epoch,start_batch,partial)
        del parent,base,old_head;import gc;gc.collect()
        data={role:{key:np.load(cache/role/(key+'.npy'),mmap_mode='r') for key in ['embedding','target','mask','group']} for role in ['train','validation']}
        engineering=np.ones((len(names),16),dtype=np.float32)
        for i,g in enumerate(names):
            for slot,rule in tasks[g]['slots'].items():engineering[i,int(slot)]=rule['scale']
        engineering=torch.from_numpy(engineering).to('cpu')
        def batch_data(role,indices):
            a=data[role];numbers=np.asarray(a['group'][indices]);groups=[names[int(i)] for i in numbers];mask=np.asarray(a['mask'][indices])
            order=[];loss_slices=[];row_order=[];group_slices=[];coverage={}
            for number in sorted(set(numbers.tolist())):
                group=names[number];rows=np.flatnonzero(numbers==number);valid=mask[rows]
                positions=(rows[:,None]*16+np.arange(16)[None,:])[valid];offset=len(order);order.extend(positions.tolist())
                if len(positions):loss_slices.append((group,offset,len(order)))
                offset=len(row_order);row_order.extend(rows.tolist());group_slices.append((group,offset,len(row_order)))
                coverage[group]=np.flatnonzero(valid.any(0)).tolist()
            meta=SimpleNamespace(targets=torch.from_numpy(np.asarray(a['target'][indices]).copy()).to('cpu'),
                mask=torch.from_numpy(mask.copy()).to('cpu'),scales=engineering[torch.tensor(numbers.astype(np.int64),device='cpu')],
                group_ids=groups,loss_slices=loss_slices,loss_order=torch.tensor(order,dtype=torch.long,device='cpu'),
                group_plan=(group_slices,torch.tensor(row_order,dtype=torch.long,device='cpu')),coverage=coverage)
            return torch.from_numpy(np.asarray(a['embedding'][indices]).copy()).to('cpu'),meta
        def forward(x,meta):return forward_grouped(head,x,meta.group_ids,meta.group_plan) if parent_recipe['grouped_execution'] else head(x,meta.group_ids)
        def objective(prediction,meta):
            head.check_observed(meta.coverage);target=meta.targets.sign()*torch.log1p((meta.targets/meta.scales).abs());_,std=head.affine(meta.group_ids)
            losses=[];finite=[]
            for _,start,end in meta.loss_slices:
                indices=meta.loss_order[start:end];p=prediction.reshape(-1)[indices];y=target.reshape(-1)[indices];scale=std.reshape(-1)[indices]
                finite.append(torch.isfinite(p).all() & torch.isfinite(y).all());losses.append(((p-y)/scale).square().mean())
            if not losses or not torch.stack(finite).all():raise ValueError('Nonfinite observed continuation loss')
            return torch.stack(losses).mean()
        def summary(metric):
            result=metric.compute();r2=[r['r2'] for r in result['per_target'] if r['r2'] is not None];result['median_r2']=float(np.median(r2)) if r2 else None
            result['by_source']={}
            for source_name in ['FALCON','AICircuit','LOADBench']:
                rows=[r for r in result['per_target'] if tasks[r['supervision_group']]['contract']['source_dataset']==source_name]
                hits=[r['hit5'] for r in rows if r['hit5'] is not None];r2=[r['r2'] for r in rows if r['r2'] is not None]
                result['by_source'][source_name]={'macro_hit5':float(np.mean(hits)) if hits else None,'median_r2':float(np.median(r2)) if r2 else None}
            return result
        n=manifest['roles']['train']['rows'];batch_size=cfg['batch_size'];batches=int(math.ceil(n/batch_size))
        update(state='training_heads',role='train',epoch=inherited_total+start_epoch+1,additional_epoch=start_epoch+1,
            inherited_head_epochs=inherited_total,batch=start_batch,batches_per_epoch=batches,role_samples=n,samples_done=min(start_batch*batch_size,n),
            learning_rate=optimizer.param_groups[0]['lr'],best_validation_score=best,best_total_head_epoch=best_total)
        for epoch in range(start_epoch,cfg['additional_epochs']):
            stats=partial if epoch==start_epoch and partial is not None else {'loss_sum':0.,'steps':0,'samples':0,'elapsed':0.}
            base_elapsed=stats['elapsed'];began=time.perf_counter();head.train();indices=np.random.default_rng(cfg['seed']+inherited_total+epoch).permutation(n)
            first_batch=start_batch if epoch==start_epoch else 0
            for b in range(first_batch,batches):
                memory_pause=psutil.virtual_memory().available<cfg['pause_below_RAM_MiB']*2**20;user_stop=(output/'STOP').exists()
                if memory_pause or user_stop:
                    stats['elapsed']=base_elapsed+time.perf_counter()-began;save_cursor(epoch,b,stats)
                    update(state='paused',reason='memory_pressure' if memory_pause else 'user_stop',batch=b);return
                selected=indices[b*batch_size:(b+1)*batch_size];x,meta=batch_data('train',selected)
                optimizer.zero_grad(set_to_none=True);prediction=forward(x,meta);loss=objective(prediction,meta);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);optimizer.step()
                stats['loss_sum']+=float(loss.detach());stats['steps']+=1;stats['samples']+=len(selected);stats['elapsed']=base_elapsed+time.perf_counter()-began
                if (b+1)%20==0 or b+1==batches:
                    update(state='training_heads',role='train',epoch=inherited_total+epoch+1,additional_epoch=epoch+1,batch=b+1,
                        batches_per_epoch=batches,samples_done=stats['samples'],role_samples=n,training_loss=stats['loss_sum']/stats['steps'],
                        samples_per_second=stats['samples']/stats['elapsed'],learning_rate=optimizer.param_groups[0]['lr'])
                    print(now(),'head epoch',inherited_total+epoch+1,'batch',b+1,'/',batches,'loss',stats['loss_sum']/stats['steps'],flush=True)
                if (b+1)%cfg['checkpoint_every_batches']==0:save_cursor(epoch,b+1,stats)
            assert stats['samples']==n
            head.eval();metric=MetricAccumulator(tasks);validation_n=manifest['roles']['validation']['rows']
            update(state='validating',role='validation',samples_done=0,role_samples=validation_n,batch=0,batches_per_epoch=int(math.ceil(validation_n/batch_size)))
            with torch.no_grad():
                for start in range(0,validation_n,batch_size):
                    ix=np.arange(start,min(start+batch_size,validation_n));x,meta=batch_data('validation',ix);head.check_observed(meta.coverage);metric.update(forward(x,meta),meta)
            val=summary(metric);score=val['macro_hit5']
            if score is None or not math.isfinite(score):raise ValueError('Invalid validation selection score')
            record={'head_epoch_total':inherited_total+epoch+1,'additional_epoch':epoch+1,'training_seconds':stats['elapsed'],
                'training_loss_mean_of_batch_group_means':stats['loss_sum']/stats['steps'],'learning_rate':optimizer.param_groups[0]['lr'],'validation':val}
            history.append(record)
            if score>best+1e-4:
                best=score;best_total=inherited_total+epoch+1;bad=0
                full=dict(body_state);full.update({'head.'+k:v for k,v in head.state_dict().items()})
                save(output/'best.pt',dict(schema_metadata,model=full,epoch=best_total,additional_head_epochs=epoch+1,
                    recipe_digest=digest(recipe),validation={k:v for k,v in val.items() if k!='per_target'}))
            else:bad+=1
            scheduler.step(score);save_cursor(epoch+1,0,None);atomic_json(output/'history.json',history)
            update(state='epoch_complete',completed_additional_epochs=epoch+1,completed_total_head_epochs=inherited_total+epoch+1,
                best_validation_score=best,best_total_head_epoch=best_total,validation_hit5=score,validation_median_r2=val['median_r2'],
                bad_epochs=bad,learning_rate=optimizer.param_groups[0]['lr'])
            print(now(),'validated head epoch',inherited_total+epoch+1,'hit5',score,'medianR2',val['median_r2'],'best',best,flush=True)
            if bad>=cfg['patience']:break
        update(state='complete',role=None,samples_done=0,role_samples=0,test_used=False,completion_reason='validation_early_stop' if bad>=cfg['patience'] else 'additional_epoch_budget',
            best_validation_score=best,best_total_head_epoch=best_total,completed_additional_epochs=len(history),
            prior_test_report_applies_to_parent_model_only=True)
        atomic_json(output/'resource_summary.json',{'peak_RSS_MiB':max(v[0] for v in resource)/2**20,'minimum_available_RAM_MiB':min(v[1] for v in resource)/2**20,
            'GPU_peak_allocated_MiB':0.})
    except BaseException as error:
        update(state='error',error_type=type(error).__name__,error=str(error));raise
    finally:
        stop.set();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()

if __name__=='__main__':main()
