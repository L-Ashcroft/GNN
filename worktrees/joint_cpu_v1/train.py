"""CPU joint RGCN refinement with streaming graphs; original CUDA admission gate stays unchanged.

All 4 layers, 128 dimensions and 48 relations remain. Each effective batch's
contract/Missing-mask MSE is accumulated exactly before clipping/AdamW. Graph
embeddings are recomputed; frozen embedding files are never model inputs here.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys
import json
import argparse
import hashlib
import time
import threading
from datetime import datetime,timezone
from collections import defaultdict
import psutil
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from stream_graphs import read,sha,StreamGraphs,weighted_micro_loss
def now():return datetime.now(timezone.utc).isoformat()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    deadline=time.monotonic()+15
    while True:
        try:os.replace(temp,path);return
        except PermissionError:
            if time.monotonic()>deadline:raise
            time.sleep(.1)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    config_path=ROOT/args.config;cfg=read(config_path);output=ROOT/cfg['output']
    if not output.resolve().is_relative_to(ROOT/'artifacts'):raise ValueError('Output outside artifacts')
    if psutil.virtual_memory().available<cfg['minimum_start_RAM_GiB']*2**30:raise SystemExit(75)
    for p in psutil.process_iter(['name','cmdline']):
        if p.pid in {os.getpid(),os.getppid()}:continue
        if (p.info['name'] or '').lower() in {'python.exe','pythonw.exe'} and any(
            str(a).replace('\\','/').endswith(('autonomous_training_v1/head_train.py','autonomous_training_v1/joint_train.py','cached_head_continue_v1/train_cpu.py','joint_cpu_v1/train.py')) for a in p.info['cmdline'] or []):
            raise RuntimeError('Another model trainer is active')
    parent_path=ROOT/cfg['parent_checkpoint']
    if sha(parent_path)!=cfg['parent_checkpoint_sha256']:raise ValueError('Warm-start checkpoint changed')
    output.mkdir(parents=True,exist_ok=True)
    import numpy as np
    import torch
    import msvcrt
    version=ROOT/'versions/rgcn_calibrated_head_v1';sys.path.insert(0,str(version))
    import src.common as common
    common.ROOT=ROOT
    from src.model.accuracy_network import AccuracyRGCN
    from src.model.grouped_contract_head import forward_grouped
    from src.model.execution import prepare_batch
    from src.train.accuracy_loss import supervision_loss
    from src.train.metrics import MetricAccumulator
    from src.train.stage import build_optimizer
    from types import MethodType
    lock=(output/'training.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    state={'state':'initializing','pid':os.getpid(),'create_time':psutil.Process().create_time(),'updated_at':now(),
        'protocol':'streamed_joint_finetune_validation_only','body_frozen':False,'test_used':False,'device':'cpu'}
    def update(**values):state.update(values,updated_at=now());atomic(output/'status.json',state)
    def save(path,value):
        temp=path.with_suffix(path.suffix+'.tmp');torch.save(value,temp);os.replace(temp,path)
    stop=threading.Event();telemetry={'peak_RSS':0,'minimum_available':psutil.virtual_memory().available}
    def memory_guard():
        while not stop.wait(.2):
            available=psutil.virtual_memory().available;telemetry.update(peak_RSS=max(telemetry['peak_RSS'],psutil.Process().memory_info().rss),minimum_available=min(telemetry['minimum_available'],available))
            if available<512*2**20:
                atomic(output/'resource_abort.json',{'at':now(),'reason':'Joint worker alone exits below512MiB; previous model preserved'});os._exit(86)
    threading.Thread(target=memory_guard,daemon=True).start()
    reader=None
    try:
        update();torch.set_num_threads(cfg['cpu_threads']);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);torch.use_deterministic_algorithms(True);torch.manual_seed(42)
        # CPU-only run: no CUDA context or CUDA memory allocation.
        parent=torch.load(parent_path,map_location='cpu',weights_only=True)
        keys=['hidden_dim','num_layers','dropout','update_edges','use_domain_emb','use_film','collapse_relations','contract_residual','calibrated_targets','adapter_hidden','adapter_min_train_samples']
        model=AccuracyRGCN(parent['schema'],parent['tasks'],parent['target_calibration'],**{k:parent['config'][k] for k in keys}).to('cpu')
        model.load_state_dict(parent['model'],strict=True)
        # Same parameters/state_dict as normal inference; only grouped execution.
        def grouped(self,embedding,groups,group_plan=None):
            if group_plan is None:raise ValueError('Full network must supply cached group routing')
            return forward_grouped(self,embedding,groups,group_plan)
        model.head.forward=MethodType(grouped,model.head)
        reader=StreamGraphs(ROOT,'data/model_ready/fusion_v2_training_v1',cfg['metadata_cache'],'versions/rgcn_calibrated_head_v1',capacity=cfg['lru_topologies'],verify=True)
        if reader.schema!=parent['schema'] or reader.tasks!=parent['tasks']:raise ValueError('Warm-start vocabulary differs from data')
        files=sorted((version/'src').rglob('*.py'))+[Path(__file__),HERE/'stream_graphs.py',config_path]
        recipe={'execution_variant':'streamed_joint_CPU_v1','config':cfg,'parent_checkpoint_sha256':cfg['parent_checkpoint_sha256'],
            'implementation_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files},'torch':torch.__version__,
            'cache_manifest_sha256':sha(reader.path/'manifest.json'),'metadata_manifest_sha256':sha(reader.meta_path/'manifest.json'),
            'target_calibration':'Inherited train-only statistics; unchanged','test_policy':'No test records or scores are read',
            'optimizer_policy':'Explicit weights-only warm start; new two-LR AdamW for changed joint stage',
            'microbatch_policy':'Sum each micro group MSE times its observed-label fraction within effective group, then divide by effective group count; clip once',
            'dropout_policy':'Active in training; microbatch RNG trajectory is this new recipe, not the old full-batch trajectory',
            'structure_sha256':{name:sha(ROOT/spec['folder']/'structure.json') for name,spec in reader.manifest['topologies'].items()}}
        if (output/'recipe.json').exists():
            if read(output/'recipe.json')!=recipe or not args.resume:raise ValueError('Joint recipe/resume mismatch')
        elif args.resume:raise ValueError('No joint recipe to resume')
        else:atomic(output/'recipe.json',recipe)
        if args.resume and not (output/'last.pt').exists():
            args.resume=False # Replay an interrupted deterministic pilot, never cross recipes.
        def summarize(metric):
            result=metric.compute();r2=[r['r2'] for r in result['per_target'] if r['r2'] is not None]
            result['median_r2']=float(np.median(r2)) if r2 else None;result['by_source']={}
            for source in ['FALCON','AICircuit','LOADBench']:
                rows=[r for r in result['per_target'] if reader.tasks[r['supervision_group']]['contract']['source_dataset']==source]
                hits=[r['hit5'] for r in rows if r['hit5'] is not None];r2s=[r['r2'] for r in rows if r['r2'] is not None]
                result['by_source'][source]={'macro_hit5':float(np.mean(hits)) if hits else None,'median_r2':float(np.median(r2s)) if r2s else None}
            return result
        def evaluate(indices,phase):
            model.eval();metric=MetricAccumulator(reader.tasks);total=0.;count=0;update(state=phase,role='validation',samples_done=0,role_samples=len(indices))
            with torch.no_grad():
                for start in range(0,len(indices),cfg['validation_microbatch']):
                    selected=indices[start:start+cfg['validation_microbatch']];batch=reader.batch('validation',selected).to('cpu')
                    model.head.check_observed(prepare_batch(batch).coverage);prediction=model(batch);loss,_=supervision_loss(prediction,batch,model.head)
                    metric.update(prediction,batch);total+=float(loss)*len(selected);count+=len(selected)
                    if start%1024==0:update(samples_done=count)
            result=summarize(metric);result['diagnostic_mean_microbatch_loss']=total/count;return result
        def build_opt(scale=1.):return build_optimizer(model,'finetune_all',lr_head=cfg['head_lr']*scale,lr_backbone=cfg['backbone_lr']*scale,weight_decay=.01)
        def train_indices(indices,optimizer,epoch,start_update=0,stats=None,checkpoint=None):
            model.train();stats=stats or {'loss_sum':0.,'updates':0,'samples':0,'seconds':0.};base_seconds=stats['seconds'];tick=time.perf_counter()
            effective=cfg['effective_batch_size'];micro=cfg['training_microbatch']
            update(state='joint_training',role='train',epoch=epoch+1,batch=start_update,batches_per_epoch=int(np.ceil(len(indices)/effective)),samples_done=stats['samples'],role_samples=len(indices))
            for update_index,start in enumerate(range(start_update*effective,len(indices),effective),start_update+1):
                if (output/'STOP').exists() or psutil.virtual_memory().available<640*2**20:
                    stats['seconds']=base_seconds+time.perf_counter()-tick
                    if checkpoint:checkpoint(epoch,update_index-1,stats)
                    update(state='paused',reason='controller_or_memory_stop',batch=update_index-1);return None
                chosen=indices[start:start+effective];full_counts=reader.counts('train',chosen);optimizer.zero_grad(set_to_none=True);total_loss=0.
                for offset in range(0,len(chosen),micro):
                    cpu=reader.batch('train',chosen[offset:offset+micro]);batch=cpu.to('cpu');prediction=model(batch)
                    _,group_losses=supervision_loss(prediction,batch,model.head);loss=weighted_micro_loss(group_losses,prepare_batch(batch).loss_slices,full_counts)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite weighted joint loss')
                    loss.backward();total_loss+=float(loss.detach());del cpu,batch,prediction,group_losses,loss
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
                stats['loss_sum']+=total_loss;stats['updates']+=1;stats['samples']+=len(chosen);stats['seconds']=base_seconds+time.perf_counter()-tick
                if update_index%10==0 or start+len(chosen)==len(indices):
                    update(batch=update_index,samples_done=stats['samples'],training_loss=stats['loss_sum']/stats['updates'],samples_per_second=stats['samples']/stats['seconds'],
                        learning_rates={g['name']:g['lr'] for g in optimizer.param_groups},gradient_norm=float(norm))
                    print(now(),'joint epoch',epoch+1,'update',update_index,'samples',stats['samples'],flush=True)
                if checkpoint and update_index%cfg['checkpoint_every_updates']==0:checkpoint(epoch,update_index,stats)
            return stats
        # Fixed train/validation subsets only. A cheap pilot filters ineffective
        # joint changes before spending full 400k-sample epochs.
        pilot_train=reader.stratified_indices('train',cfg['pilot_train_samples'],42)
        pilot_val=reader.stratified_indices('validation',cfg['pilot_validation_samples'],43)
        if not args.resume:
            baseline=evaluate(pilot_val,'pilot_baseline');atomic(output/'pilot_baseline.json',baseline)
            successes=[];trials=[]
            for number,scale in enumerate(cfg['pilot_lr_scales']):
                model.load_state_dict(parent['model'],strict=True);torch.manual_seed(42);optimizer=build_opt(scale)
                stats=train_indices(pilot_train,optimizer,0)
                if stats is None:return
                validation=evaluate(pilot_val,'pilot_validation');trial={'trial':number,'lr_scale':scale,'training':stats,'validation':validation};trials.append(trial)
                save(output/('pilot_'+str(number)+'.pt'),{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'lr_scale':scale})
                gain=validation['macro_hit5']-baseline['macro_hit5'];r2gain=validation['median_r2']-baseline['median_r2']
                loss_gain=1-validation['diagnostic_mean_microbatch_loss']/baseline['diagnostic_mean_microbatch_loss']
                acceptable=(gain>=cfg['pilot_min_hit_gain'] or loss_gain>=cfg['pilot_min_loss_gain']) and r2gain>=-cfg['pilot_max_median_drop']
                trial['accepted']=acceptable
                if acceptable:successes.append(trial)
                atomic(output/'pilot_trials.json',trials)
            if not successes:update(state='pilot_rejected',reason='No validation-qualified joint gain',test_used=False);return
            selected=max(successes,key=lambda t:t['validation']['macro_hit5']);saved=torch.load(output/('pilot_'+str(selected['trial'])+'.pt'),map_location='cpu',weights_only=True)
            model.load_state_dict(saved['model']);optimizer=build_opt(selected['lr_scale']);optimizer.load_state_dict(saved['optimizer'])
            baseline_full=evaluate(reader.indices('validation'),'joint_initial_full_validation');atomic(output/'initial_full_validation.json',baseline_full)
            scale=selected['lr_scale'];history=[];start_epoch=0;start_update=0;partial=None;bad=0
            parent_validation=parent.get('validation')
            if parent_validation is None:raise ValueError('Parent lacks validation selection metadata')
            best=parent_validation['macro_hit5'];best_epoch=-1
            # Keep the already better parent unless full validation improves.
            initial=dict(parent,config=dict(parent['config'],output=cfg['output'],joint_stage=True),recipe_digest=digest(recipe))
            save(output/'best.pt',initial)
            if baseline_full['macro_hit5']>best+1e-4:
                best=baseline_full['macro_hit5'];best_epoch=0
                save(output/'best.pt',dict(initial,model=model.state_dict(),joint_epoch=0,
                    validation={k:v for k,v in baseline_full.items() if k!='per_target'},joint_selection='full-validation-qualified pilot'))
        else:
            saved=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
            if saved['recipe_digest']!=digest(recipe):raise ValueError('Joint checkpoint digest mismatch')
            model.load_state_dict(saved['model']);scale=saved['lr_scale'];optimizer=build_opt(scale);optimizer.load_state_dict(saved['optimizer'])
            history=saved['history'];start_epoch=saved['next_epoch'];start_update=saved['next_update'];partial=saved['partial'];bad=saved['bad'];best=saved['best_score'];best_epoch=saved['best_epoch']
            torch.set_rng_state(saved['rng_cpu'].cpu());assert saved['rng_cuda']==[]
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=2,threshold=1e-4,threshold_mode='abs',min_lr=cfg['minimum_lr'])
        scheduler.best=best
        if args.resume:scheduler.load_state_dict(saved['scheduler'])
        def checkpoint(epoch,update_index,stats):
            save(output/'last.pt',{'recipe_digest':digest(recipe),'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'lr_scale':scale,'next_epoch':epoch,'next_update':update_index,'partial':stats,'history':history,'bad':bad,'best_score':best,'best_epoch':best_epoch,
                'rng_cpu':torch.get_rng_state(),'rng_cuda':[]})
        if not args.resume:checkpoint(0,0,None)
        practical_stop=False
        for epoch in range(start_epoch,cfg['max_epochs']):
            indices=reader.indices('train',42+epoch);stats=train_indices(indices,optimizer,epoch,start_update if epoch==start_epoch else 0,
                partial if epoch==start_epoch else None,checkpoint)
            if stats is None:return
            assert stats['samples']==400083
            validation=evaluate(reader.indices('validation'),'joint_full_validation');score=validation['macro_hit5']
            if not np.isfinite(score):raise ValueError('Invalid validation score')
            record={'epoch':epoch+1,'training':stats,'validation':validation};history.append(record)
            if score>best+1e-4:
                best=score;best_epoch=epoch+1;bad=0
                save(output/'best.pt',dict(parent,model=model.state_dict(),config=dict(parent['config'],output=cfg['output'],joint_stage=True),
                    recipe_digest=digest(recipe),joint_epoch=epoch+1,validation={k:v for k,v in validation.items() if k!='per_target'}))
            else:bad+=1
            scheduler.step(score);practical_stop=False
            window=cfg['window_epochs'];last_action=max([r['epoch'] for r in history if r.get('window_lr_action')],default=0)
            if len(history)>=window and epoch+1-last_action>=window:
                earlier=max([parent['validation']['macro_hit5']]+[r['validation']['macro_hit5'] for r in history[:-window]])
                gain=best-earlier;record['window_validation_gain']=gain
                if gain<cfg['minimum_window_gain']:
                    backbone_lr=next(g['lr'] for g in optimizer.param_groups if g['name']=='backbone')
                    if backbone_lr>cfg['window_stop_backbone_lr']:
                        for group in optimizer.param_groups:group['lr']=max(cfg['minimum_lr'],group['lr']*.5)
                        scheduler._last_lr=[g['lr'] for g in optimizer.param_groups];record['window_lr_action']='reduce_on_practical_plateau'
                    else:practical_stop=True;record['window_lr_action']='stop_on_practical_plateau'
            checkpoint(epoch+1,0,None);atomic(output/'history.json',history)
            update(state='joint_epoch_complete',completed_epochs=epoch+1,best_validation_score=best,best_joint_epoch=best_epoch,bad_epochs=bad,
                validation_hit5=score,validation_median_r2=validation['median_r2'])
            if bad>=cfg['patience'] or practical_stop:break
        update(state='complete',test_used=False,best_validation_score=best,best_joint_epoch=best_epoch,
            completion_reason='validation_window_plateau' if practical_stop else 'validation_early_stop' if bad>=cfg['patience'] else 'safety_budget_review')
    except BaseException as error:
        update(state='error',error_type=type(error).__name__,error=str(error));raise
    finally:
        stop.set()
        atomic(output/'resource_summary.json',{'peak_RSS_MiB':telemetry['peak_RSS']/2**20,'minimum_available_RAM_MiB':telemetry['minimum_available']/2**20,'CUDA_used':False})
        if reader is not None:reader.close()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()

if __name__=='__main__':main()
