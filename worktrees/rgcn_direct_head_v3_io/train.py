"""Train independent direct residual GNN heads, using only train/validation."""
from pathlib import Path
import os,sys,json,time,hashlib,argparse,ast,statistics,threading,msvcrt
from datetime import datetime,timezone
import psutil
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(2**20),b''):h.update(data)
    return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def atomic(p,v):
    q=p.with_suffix('.train.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
    for attempt in range(50):
        try:os.replace(q,p);return
        except PermissionError:time.sleep(.1)
    raise RuntimeError('Cannot publish status')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args();cfg=read(HERE/'config.json');output=ROOT/cfg['output']
    if psutil.virtual_memory().available<cfg['minimum_start_RAM_GiB']*2**30:raise SystemExit(75)
    output.mkdir(parents=True,exist_ok=True);lock=(output/'training.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    state={'state':'initializing','pid':os.getpid(),'create_time':psutil.Process().create_time(),'updated_at':now(),'device':'cpu',
        'protocol':'RGCN_frozen_backbone_independent_direct_residual_head','body_frozen':True,'test_used':False}
    def update(**v):state.update(v,updated_at=now());atomic(output/'status.json',state)
    done=threading.Event();resource={'peak_RSS':0,'minimum_available':psutil.virtual_memory().available}
    def guard():
        while not done.wait(.2):
            available=psutil.virtual_memory().available;resource['peak_RSS']=max(resource['peak_RSS'],psutil.Process().memory_info().rss);resource['minimum_available']=min(resource['minimum_available'],available)
            if available<384*2**20:os._exit(86)
    threading.Thread(target=guard,daemon=True).start();arrays={}
    try:
        update();psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        import numpy as np
        import torch
        torch.set_num_threads(cfg['cpu_threads']);torch.use_deterministic_algorithms(True);torch.manual_seed(cfg['seed'])
        sys.path.insert(0,str(HERE));from model import DirectResidualHead
        tasks=read(ROOT/'data/model_ready/fusion_v2_training_v1/tasks.json');calibration=read(ROOT/cfg['target_calibration']);cache=ROOT/cfg['feature_cache'];manifest=read(cache/'manifest.json')
        for role in ['train','validation']:
            arrays[role]={}
            for filename,expected in manifest['roles'][role]['files'].items():
                assert sha(cache/role/filename)==expected,('feature integrity',role,filename)
            for name in ['embedding','target','mask','group']:arrays[role][name]=np.load(cache/role/(name+'.npy'),mmap_mode='r')
        assert len(arrays['train']['group'])==400083 and len(arrays['validation']['group'])==50861
        groups=manifest['groups'];indices={role:{g:np.flatnonzero(data['group']==i) for i,g in enumerate(groups)} for role,data in arrays.items()}
        parent=read(ROOT/'reports/experiment1_validation_review.json')
        head=DirectResidualHead(ROOT,tasks,calibration,cfg['hidden'],cfg['minimum_residual_training_rows'])
        initial_hash=hashlib.sha256()
        for name,value in head.state_dict().items():initial_hash.update(name.encode());initial_hash.update(value.numpy().tobytes())
        files=[HERE/name for name in ['model.py','train.py','supervisor.py','config.json','checks.py']]
        recipe={'variant':'edge_RGCN_direct_residual_head_v3','config':cfg,'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files},
            'cache_manifest_sha256':sha(cache/'manifest.json'),'graph_checkpoint_sha256':sha(ROOT/cfg['graph_checkpoint']),
            'target_calibration_sha256':sha(ROOT/cfg['target_calibration']),'initial_ridge_recipe_sha256':sha(ROOT/'artifacts/experiment1_representation_diagnostic_v1/recipe.json'),
            'bounds_recipe_sha256':sha(ROOT/'artifacts/experiment1_local_readout_v1/recipe.json'),'torch':torch.__version__,
            'initial_head_sha256':initial_hash.hexdigest(),'metric_source_sha256':sha(ROOT/'worktrees/representation_diagnostic_v1/run.py'),
            'objective':'Observed per-contract mean squared error in train-calibrated symlog units; no label/mask changes',
            'changes':'Independent128D direct skips/residual heads; FP32 normalized trainable layers, FP64 target affine; eval-only train-range output safeguard',
            'test_policy':'No test arrays or scores read; validation only','LOTO_policy':'Unknown semantic heads rejected; this ID candidate does not solve unseen-contract zero shot'}
        digest=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
        if (output/'recipe.json').exists():assert args.resume and read(output/'recipe.json')==recipe,'Recipe mismatch'
        else:assert not args.resume;atomic(output/'recipe.json',recipe)
        direct=[p for g in head.contracts.values() for p in g.direct.parameters()];residual=[p for g in head.contracts.values() if g.residual is not None for p in g.residual.parameters()]
        optimizer=torch.optim.AdamW([{'params':direct,'lr':cfg['direct_learning_rate'],'name':'direct'},
            {'params':residual,'lr':cfg['residual_learning_rate'],'name':'residual'}],weight_decay=cfg['weight_decay'])
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=2,factor=.5,min_lr=1e-6,threshold=cfg['minimum_gain'],threshold_mode='abs')
        LOG={'dB','dBm','dBc/Hz','source_label_dBm_compression_function'};namespace={'np':np,'statistics':statistics,'tasks':tasks,'LOG':LOG}
        for node in ast.parse((ROOT/'worktrees/representation_diagnostic_v1/run.py').read_text(encoding='utf-8')).body:
            if isinstance(node,ast.FunctionDef) and node.name in {'metrics','summarize'}:exec(compile(ast.Module(body=[node],type_ignores=[]),'verified_metric_definitions','exec'),namespace)
        # Bind evaluated metric source to recipe independently of training code.
        assert namespace['metrics'] is not None
        scales={g:np.asarray([tasks[g]['slots'].get(str(j),{}).get('scale',1.) for j in range(16)],dtype=np.float32) for g in groups}
        def data(role,g,ids):
            a=arrays[role];x=torch.from_numpy(a['embedding'][ids].copy());y=torch.from_numpy(a['target'][ids].copy());m=torch.from_numpy(a['mask'][ids].copy())
            scale=torch.from_numpy(np.broadcast_to(scales[g],(len(ids),16)).copy());return x,y,m,scale
        def evaluate(epoch):
            head.eval();rows=[];update(state='validating',epoch=epoch,role='validation',samples_done=0,role_samples=50861);seen=0
            with torch.no_grad():
                for g in groups:
                    ids=indices['validation'][g]
                    if not len(ids):continue
                    x,y,m,s=data('validation',g,ids);z=head.contracts[g](x).numpy();physical=np.sign(z)*np.expm1(abs(z))*s.numpy().astype('float64')
                    rows+=namespace['metrics'](physical,y.numpy(),m.numpy(),g);seen+=len(ids)
            assert seen==50861 and len(rows)==1491
            result=namespace['summarize'](rows);result['per_target']=rows;return result
        def torch_save(p,value):
            q=p.with_suffix('.pt.tmp');torch.save(value,q)
            for attempt in range(100):
                try:os.replace(q,p);return
                except PermissionError:time.sleep(.1)
            raise RuntimeError('Atomic checkpoint replace failed after10seconds: '+str(p))
        base_info={'config':cfg,'recipe_digest':digest,'tasks':tasks,'graph_checkpoint':cfg['graph_checkpoint'],'graph_checkpoint_sha256':recipe['graph_checkpoint_sha256'],
            'head_type':'independent_direct_residual_v3','inference_class':'model.DirectResidualHead.from_state','test_used':False}
        history=[];start_epoch=0;start_batch=0;partial=None;bad=0
        if args.resume and (output/'last.pt').exists():
            saved=torch.load(output/'last.pt',map_location='cpu',weights_only=True);assert saved['recipe_digest']==digest
            head.load_state_dict(saved['head']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);torch.set_rng_state(saved['rng'])
            history=saved['history'];start_epoch=saved['next_epoch'];start_batch=saved['next_batch'];partial=saved['partial'];best=saved['best'];best_epoch=saved['best_epoch'];bad=saved['bad']
        elif cfg.get('migration_checkpoint'):
            for field in ['checkpoint','best','recipe']:
                assert sha(ROOT/cfg['migration_'+field])==cfg['migration_'+field+'_sha256'],'Migration source changed'
            source_recipe=read(ROOT/cfg['migration_recipe'])
            old_digest=hashlib.sha256(json.dumps(source_recipe,sort_keys=True).encode()).hexdigest()
            saved=torch.load(ROOT/cfg['migration_checkpoint'],map_location='cpu',weights_only=True)
            assert saved['recipe_digest']==old_digest
            head.load_state_dict(saved['head'],strict=True);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);torch.set_rng_state(saved['rng'])
            for name,value in head.state_dict().items():torch.testing.assert_close(value,saved['head'][name],atol=0,rtol=0)
            history=saved['history'];start_epoch=saved['next_epoch'];start_batch=saved['next_batch'];partial=saved['partial'];best=saved['best'];best_epoch=saved['best_epoch'];bad=saved['bad']
            prior_best=torch.load(ROOT/cfg['migration_best'],map_location='cpu',weights_only=True)
            torch_save(output/'best.pt',dict(base_info,head=prior_best['head'],epoch=prior_best['epoch'],validation=prior_best['validation'],migration_from=cfg['migration_best']))
            atomic(output/'history.json',history)
            atomic(output/'migration.json',{'source_recipe_digest':old_digest,'new_recipe_digest':digest,'head_exact':True,'optimizer_restored':True,'scheduler_restored':True,'RNG_restored':True,'next_epoch':start_epoch,'next_batch':start_batch,'old_artifacts_unchanged':True})
        else:
            initial=evaluate(0);atomic(output/'initial_validation.json',initial);best=initial['macro_hit5'];best_epoch=0
            torch_save(output/'best.pt',dict(base_info,head=head.state_dict(),epoch=0,validation={k:v for k,v in initial.items() if k!='per_target'}))
            scheduler.best=best
        def checkpoint(epoch,batch,stats):
            torch_save(output/'last.pt',{'recipe_digest':digest,'head':head.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'rng':torch.get_rng_state(),'history':history,'next_epoch':epoch,'next_batch':batch,'partial':stats,'best':best,'best_epoch':best_epoch,'bad':bad})
        if not args.resume:checkpoint(0,0,None)
        for epoch in range(start_epoch,cfg['max_epochs']):
            rng=np.random.default_rng(cfg['seed']+epoch);order=[]
            for g in groups:
                ix=rng.permutation(indices['train'][g])
                for offset in range(0,len(ix),cfg['batch_size']):order.append((g,ix[offset:offset+cfg['batch_size']]))
            rng.shuffle(order);stats=partial if epoch==start_epoch and partial else {'samples':0,'loss_sum':0.,'steps':0,'seconds':0.}
            cursor=start_batch if epoch==start_epoch else 0;tick=time.perf_counter();elapsed=stats['seconds'];head.train()
            update(state='training_heads',epoch=epoch+1,batch=cursor,batches_per_epoch=len(order),role='train',role_samples=400083,samples_done=stats['samples'],best_validation_score=best,best_head_epoch=best_epoch)
            for b in range(cursor,len(order)):
                if (output/'STOP').exists() or psutil.virtual_memory().available<512*2**20:
                    stats['seconds']=elapsed+time.perf_counter()-tick;checkpoint(epoch,b,stats);update(state='paused',reason='stop_or_low_memory');return
                g,ids=order[b];x,y,m,s=data('train',g,ids)
                allowed={int(j) for j,v in calibration['groups'][g]['slots'].items() if v['count']>0}
                assert set(np.flatnonzero(m.numpy().any(0)))<=allowed
                optimizer.zero_grad(set_to_none=True);loss=head.contracts[g].loss(x,y,m,s);loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.contracts[g].parameters(),5.,error_if_nonfinite=True);optimizer.step()
                stats['samples']+=len(ids);stats['loss_sum']+=float(loss.detach());stats['steps']+=1;stats['seconds']=elapsed+time.perf_counter()-tick
                if (b+1)%50==0 or b+1==len(order):update(batch=b+1,samples_done=stats['samples'],training_loss=stats['loss_sum']/stats['steps'],samples_per_second=stats['samples']/stats['seconds'],gradient_norm=float(norm))
                if (b+1)%cfg['checkpoint_every_batches']==0:checkpoint(epoch,b+1,stats)
            assert stats['samples']==400083
            validation=evaluate(epoch+1);score=validation['macro_hit5'];history.append({'epoch':epoch+1,'training':stats,'validation':validation})
            if score>best+cfg['minimum_gain']:
                best=score;best_epoch=epoch+1;bad=0;torch_save(output/'best.pt',dict(base_info,head=head.state_dict(),epoch=best_epoch,validation={k:v for k,v in validation.items() if k!='per_target'}))
            else:bad+=1
            scheduler.step(score);checkpoint(epoch+1,0,None);atomic(output/'history.json',history)
            update(state='epoch_complete',epoch=epoch+1,completed_epochs=epoch+1,best_validation_score=best,best_head_epoch=best_epoch,
                validation_hit5=score,validation_median_r2=validation['median_r2'],validation_macro_r2=validation['macro_r2'],bad_epochs=bad)
            print(now(),'epoch',epoch+1,'hit5',score,'medianR2',validation['median_r2'],'best',best,flush=True)
            if bad>=cfg['patience']:break
        update(state='complete',best_validation_score=best,best_head_epoch=best_epoch,completion_reason='validation_early_stop' if bad>=cfg['patience'] else 'safety_budget')
    except BaseException as error:update(state='error',error_type=type(error).__name__,error=str(error));raise
    finally:
        done.set();atomic(output/'resource_summary.json',{'peak_RSS_MiB':resource['peak_RSS']/2**20,'minimum_available_RAM_MiB':resource['minimum_available']/2**20,'CUDA_used':False})
        for a in arrays.values():
            for v in a.values():v._mmap.close()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
if __name__=='__main__':main()
