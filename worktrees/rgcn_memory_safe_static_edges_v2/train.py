"""Public500k LOTO training, strict fold roles and resumable CPU/CUDA execution."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys,json,time,argparse,hashlib,threading,msvcrt
from datetime import datetime,timezone
import psutil
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def replace(a,b):
    for i in range(100):
        try:os.replace(a,b);return
        except PermissionError:time.sleep(.1)
    raise RuntimeError('Atomic file replacement timed out: '+str(b))
def atomic(p,v):
    q=p.with_suffix('.train.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8');replace(q,p)
def recipe_for(cfg,reader,stats):
    names=['runtime.py','data.py','fit.py','model.py','train.py','qualify.py','supervisor.py','config.json']
    files=[HERE/n for n in names]+list((ROOT/'versions/rgcn_calibrated_head_v1/src').rglob('*.py'))
    return {'experiment':cfg['experiment_kind']+'_controlled_repair_v1','config':cfg,'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files},
        'protocol_sha256':sha(reader.protocol_path),'fold_references_sha256':reader.fold['sha256'],'normalization_sha256':sha(ROOT/cfg['output']/'normalization.json'),
        'raw_graph_manifest_sha256':sha(reader.path/'manifest.json'),'initialization':'All neural parameters random seed42; no pretrained state or cached embeddings',
        'supported_runtime_SHA256':{n:sha(ROOT/n) for n in ['.venv/Lib/site-packages/torch/version.py','.venv-gpu/Lib/site-packages/torch/version.py']},
        'backend_policy':'Same pinned recipe supports CPU/CUDA; migration restores weights/AdamW/scheduler/cursor, logs device and derived dropout RNG transition',
        'selection':cfg['selection_metric']+'; fixed test after selection freeze only',
        'normalization_proof':stats['proof'],'private_AnalogGym_used':False}
def build_model(reader,stats,cfg):
    from model import FoldRGCN
    return FoldRGCN(reader.schema,reader.tasks,stats,**{k:cfg[k] for k in ['hidden_dim','num_layers','dropout','update_edges','use_domain_emb','use_film','collapse_relations']})
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=['cpu','cuda'],default='cpu');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    cfg=read(HERE/'config.json');output=ROOT/cfg['output'];output.mkdir(parents=True,exist_ok=True)
    if cfg['private_AnalogGym_used'] or cfg['source_population']!='public_fusion_500000_only':raise ValueError('Wrong source population')
    gate=cfg['minimum_CUDA_start_RAM_GiB'] if args.backend=='cuda' else cfg['minimum_CPU_start_RAM_GiB']
    if psutil.virtual_memory().available<gate*2**30:raise SystemExit(75)
    lock=(output/'training.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    state={'state':'initializing','pid':os.getpid(),'create_time':psutil.Process().create_time(),'updated_at':now(),'device':args.backend,
        'experiment':cfg['experiment_kind'],'fold_id':cfg['fold_id'],'body_frozen':False,'private_AnalogGym_used':False,'heldout_test_used':False}
    def update(**v):state.update(v,updated_at=now());atomic(output/'status.json',state)
    done=threading.Event();resource={'peak_RSS':0,'min_available':psutil.virtual_memory().available}
    def guard():
        while not done.wait(.2):
            av=psutil.virtual_memory().available;resource['peak_RSS']=max(resource['peak_RSS'],psutil.Process().memory_info().rss);resource['min_available']=min(resource['min_available'],av)
            if av<512*2**20:resource['memory_pressure']=True
    threading.Thread(target=guard,daemon=True).start();reader=None
    try:
        update();import runtime
        import numpy as np
        import torch
        from data import FoldData
        from fit import fit_stats
        from model import loss_by_contract,weighted_loss
        from src.model.execution import prepare_batch
        from src.train.stage import build_optimizer
        from src.train.metrics import MetricAccumulator
        assert torch.__version__ in cfg['supported_torch_builds'],'Unsupported Torch build'
        torch.set_num_threads(cfg['cpu_threads']);torch.use_deterministic_algorithms(True);torch.manual_seed(cfg['seed']);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        if args.backend=='cuda':
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.8)
            acceptance=read(output/'cuda_acceptance.json')
            if not acceptance['passed'] or acceptance['model_source_sha256']!=sha(HERE/'model.py') or acceptance['normalization_sha256']!=sha(output/'normalization.json'):raise ValueError('CUDA not qualified for current implementation')
        reader=FoldData(cfg['fold_id'],cfg['lru_topologies'],verify=True);stats=fit_stats(reader,output/'normalization.json')
        recipe=recipe_for(cfg,reader,stats);digest=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
        if (output/'recipe.json').exists():assert args.resume and read(output/'recipe.json')==recipe,'Pinned recipe changed'
        else:assert not args.resume;atomic(output/'recipe.json',recipe)
        if args.resume and not (output/'last.pt').exists():args.resume=False # Replay only an interrupted pre-training baseline under the same recipe.
        atomic(output/'task_registry.json',reader.tasks);atomic(output/'schema.json',reader.schema)
        model=build_model(reader,stats,cfg).to(args.backend);optimizer=build_optimizer(model,'finetune_all',cfg['adaptation_lr'],cfg['backbone_lr'],cfg['weight_decay'])
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=2,threshold=cfg['minimum_gain'],threshold_mode='abs',min_lr=cfg['minimum_lr'])
        def save(p,value):
            q=p.with_suffix('.pt.tmp');torch.save(value,q);replace(q,p)
        def summary(metric):
            m=metric.compute();r2=[r['r2'] for r in m['per_target'] if r['r2'] is not None];m['median_r2']=float(np.median(r2)) if r2 else None
            compatible=m['per_target'] if cfg['experiment_kind'].startswith('ID') else [r for r in m['per_target'] if r['supervision_group']==reader.fold['head_alias']]
            def reduce(rows):
                hits=[r['hit5'] for r in rows if r['hit5'] is not None];rr=[r['r2'] for r in rows if r['r2'] is not None]
                return {'macro_hit5':float(np.mean(hits)) if hits else None,'median_r2':float(np.median(rr)) if rr else None,'targets':len(rows)}
            m['compatible_validation']=reduce(compatible);m['by_source']={}
            for source in ['FALCON','AICircuit','LOADBench']:m['by_source'][source]=reduce([r for r in m['per_target'] if reader.tasks[r['supervision_group']]['contract']['source_dataset']==source])
            return m
        def evaluate(role,epoch,baseline=False):
            if role=='test' and not (output/'selection_frozen.json').exists():raise ValueError('Outer test access before selection freeze')
            model.eval();metric=MetricAccumulator(reader.tasks);n=len(reader.refs[role]);update(state='baseline_validation' if baseline else 'evaluating_heldout_test' if role=='test' else 'validating',role=role,epoch=epoch,samples_done=0,role_samples=n)
            predictions=[]
            with torch.no_grad():
                for start in range(0,n,cfg['validation_microbatch']):
                    if psutil.virtual_memory().available<512*2**20:raise MemoryError('Validation memory guard; prior checkpoint retained')
                    b=reader.batch(role,np.arange(start,min(start+cfg['validation_microbatch'],n))).to(args.backend)
                    model.head.check_observed(prepare_batch(b).coverage)
                    pred=model.head.affine(b.group_ids)[0] if baseline else model(b);metric.update(pred,b)
                    if role=='test':
                        physical=(pred.sign()*torch.expm1(pred.abs())*b.scales.to(torch.float64)).cpu().numpy()
                        for i,sid in enumerate(b.sample_ids):predictions.append({'sample_id':sid,'supervision_group':b.group_ids[i],'prediction':physical[i].tolist(),'target':b.targets[i].cpu().tolist(),'mask':b.mask[i].cpu().tolist()})
                    if start%2048==0:update(samples_done=start+len(b.group_ids))
            result=summary(metric)
            if role=='test':
                import pyarrow as pa
                import pyarrow.parquet as pq
                q=output/'heldout_predictions.parquet.tmp';pq.write_table(pa.Table.from_pylist(predictions),q,compression='zstd');replace(q,output/'heldout_predictions.parquet')
            return result
        history=[];start_epoch=0;start_update=0;partial=None;best=-1.;best_epoch=0;bad=0;updates=0;backend_events=[]
        if args.resume:
            saved=torch.load(output/'last.pt',map_location='cpu',weights_only=True);assert saved['recipe_digest']==digest,'Checkpoint recipe mismatch'
            model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
            history=saved['history'];start_epoch=saved['next_epoch'];start_update=saved['next_update'];partial=saved['partial'];best=saved['best'];best_epoch=saved['best_epoch'];bad=saved['bad'];updates=saved['updates'];backend_events=saved['backend_events']
            torch.set_rng_state(saved['rng_cpu'].cpu())
            if saved['backend']==args.backend:
                if args.backend=='cuda':torch.cuda.set_rng_state_all([r.cpu() for r in saved['rng_cuda']])
            else:
                derived=cfg['seed']+updates+1000003
                if args.backend=='cuda':torch.cuda.manual_seed_all(derived)
                else:torch.manual_seed(derived)
                backend_events.append({'at':now(),'from':saved['backend'],'to':args.backend,'completed_updates':updates,'dropout_seed':derived,'note':'Declared same-recipe backend transition, not bitwise CPU/GPU dropout equivalence'})
                atomic(output/'backend_events.json',backend_events)
        else:
            if args.backend=='cuda':torch.cuda.manual_seed_all(cfg['seed'])
            baseline=evaluate('validation',0,True);atomic(output/'baseline_validation.json',baseline);best=baseline['compatible_validation']['macro_hit5']
            if best is None:raise ValueError('No compatible validation metrics')
            scheduler.best=best;save(output/'best.pt',{'model':model.state_dict(),'epoch':0,'recipe_digest':digest,'validation':baseline,'config':cfg})
        def checkpoint(epoch,update_index,stats):
            save(output/'last.pt',{'recipe_digest':digest,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all() if args.backend=='cuda' else [],'backend':args.backend,
                'next_epoch':epoch,'next_update':update_index,'partial':stats,'history':history,'best':best,'best_epoch':best_epoch,'bad':bad,'updates':updates,'backend_events':backend_events})
        if not args.resume:checkpoint(0,0,None)
        stop_before_test=False
        end_epoch=start_epoch if start_epoch>=cfg['minimum_epochs'] and bad>=cfg['patience'] else cfg['max_epochs']
        for epoch in range(start_epoch,end_epoch):
            indices=np.random.default_rng(cfg['seed']+epoch).permutation(len(reader.refs['train']));effective=cfg['effective_batch_size'];micro=cfg['training_microbatch'];cursor=start_update if epoch==start_epoch else 0
            stats_epoch=partial if epoch==start_epoch and partial else {'samples':0,'steps':0,'loss_sum':0.,'seconds':0.};elapsed=stats_epoch['seconds'];tick=time.perf_counter();model.train()
            update(state='training',role='train',epoch=epoch+1,batch=cursor,batches_per_epoch=int(np.ceil(len(indices)/effective)),samples_done=stats_epoch['samples'],role_samples=len(indices),
                best_compatible_validation_hit5=best,best_epoch=best_epoch,model_parameters=sum(p.numel() for p in model.parameters()),heldout_topologies=reader.fold['test_topologies'])
            for number,start in enumerate(range(cursor*effective,len(indices),effective),cursor+1):
                if (output/'STOP').exists() or resource.get('memory_pressure',False) or psutil.virtual_memory().available<640*2**20:
                    checkpoint(epoch,number-1,stats_epoch);update(state='paused',reason='supervisor_or_memory_stop',batch=number-1);return
                if updates<cfg['warmup_updates']:
                    scale=max(.1,(updates+1)/cfg['warmup_updates'])
                    for group in optimizer.param_groups:group['lr']=cfg['backbone_lr' if group['name']=='backbone' else 'adaptation_lr']*scale
                step_tick=time.perf_counter();chosen=indices[start:start+effective];counts=reader.counts('train',chosen);optimizer.zero_grad(set_to_none=True);total=0.
                for off in range(0,len(chosen),micro):
                    b=reader.batch('train',chosen[off:off+micro]).to(args.backend);_,per=loss_by_contract(model(b),b,model.head);loss=weighted_loss(per,prepare_batch(b).loss_slices,counts);loss.backward();total+=float(loss.detach());del b,per,loss
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step();updates+=1
                stats_epoch['samples']+=len(chosen);stats_epoch['steps']+=1;stats_epoch['loss_sum']+=total;stats_epoch['seconds']+=time.perf_counter()-step_tick
                if number%10==0 or start+len(chosen)==len(indices):
                    update(batch=number,samples_done=stats_epoch['samples'],training_loss=stats_epoch['loss_sum']/stats_epoch['steps'],samples_per_second=stats_epoch['samples']/stats_epoch['seconds'],gradient_norm=float(norm),updates=updates,learning_rates={g['name']:g['lr'] for g in optimizer.param_groups})
                    print(now(),args.backend,'epoch',epoch+1,'batch',number,'samples',stats_epoch['samples'],flush=True)
                if number%cfg['checkpoint_every_updates']==0:checkpoint(epoch,number,stats_epoch)
            assert stats_epoch['samples']==reader.fold['counts']['train']
            # Validation-phase failure can restart from the fully completed
            # epoch cursor without repeating the preceding training updates.
            checkpoint(epoch,int(np.ceil(len(indices)/effective)),stats_epoch)
            validation=evaluate('validation',epoch+1);score=validation['compatible_validation']['macro_hit5']
            if score is None or not np.isfinite(score):raise ValueError('Invalid compatible validation score')
            history.append({'epoch':epoch+1,'training':stats_epoch,'validation':validation,'backend':args.backend})
            if score>best+cfg['minimum_gain']:
                best=score;best_epoch=epoch+1;bad=0;save(output/'best.pt',{'model':model.state_dict(),'epoch':best_epoch,'recipe_digest':digest,'validation':validation,'config':cfg})
            else:bad+=1
            scheduler.step(score);checkpoint(epoch+1,0,None);atomic(output/'history.json',history)
            update(state='epoch_complete',completed_epochs=epoch+1,compatible_validation_hit5=score,global_validation_hit5=validation['macro_hit5'],compatible_validation_median_r2=validation['compatible_validation']['median_r2'],best_compatible_validation_hit5=best,best_epoch=best_epoch,bad_epochs=bad)
            print(now(),'validation compatiblehit',score,'globalhit',validation['macro_hit5'],flush=True)
            if epoch+1>=cfg['minimum_epochs'] and bad>=cfg['patience']:break
        if (output/'STOP').exists():update(state='paused',reason='stop_before_outer_test');return
        atomic(output/'selection_frozen.json',{'at':now(),'best_epoch':best_epoch,'best_compatible_validation_hit5':best,'checkpoint_sha256':sha(output/'best.pt'),'recipe_digest':digest,'outer_test_used_for_selection':False})
        saved=torch.load(output/'best.pt',map_location=args.backend,weights_only=True);model.load_state_dict(saved['model']);outer=evaluate('test',best_epoch);atomic(output/'heldout_test_metrics.json',outer)
        update(state='complete',best_epoch=best_epoch,best_compatible_validation_hit5=best,heldout_test_used=True,test_samples=reader.fold['counts']['test'],
            test_hit5=outer['macro_hit5'],test_median_r2=outer['median_r2'],completion_reason='validation_early_stop' if bad>=cfg['patience'] else 'epoch_budget')
        atomic(output/'final_summary.json',{'fold':reader.fold,'model_selection':read(output/'selection_frozen.json'),'test_metrics':{k:v for k,v in outer.items() if k!='per_target'},
            'initialization':'random; no pretraining','normalization_training_rows':stats['proof']['training_rows'],'private_AnalogGym_used':False,'backend_events':backend_events})
    except BaseException as e:update(state='error',error_type=type(e).__name__,error=str(e));raise
    finally:
        done.set();atomic(output/'resource_summary.json',{'peak_RSS_MiB':resource['peak_RSS']/2**20,'minimum_available_RAM_MiB':resource['min_available']/2**20,'backend':args.backend})
        if reader is not None:reader.close()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
if __name__=='__main__':main()
