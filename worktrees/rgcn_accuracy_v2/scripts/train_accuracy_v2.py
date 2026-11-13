"""Resumable public-fusion ID training, with live status and fixed test isolation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from datetime import datetime,timezone
import argparse
import copy
import math
from pathlib import Path
import sys
import time
import msvcrt
import psutil
# Resource/process guard runs before importing torch or writing artifacts.
if __name__=='__main__' and '--help' not in sys.argv and '-h' not in sys.argv:
    probe=argparse.ArgumentParser(add_help=False);probe.add_argument('--config',required=True)
    preliminary,_=probe.parse_known_args()
    trial_root=Path(__file__).resolve().parents[3]
    trial_config=__import__('json').loads((trial_root/preliminary.config).read_text(encoding='utf-8'))
    if trial_config['device']=='cuda':
        for process in psutil.process_iter(['name','cmdline']):
            if process.pid in {os.getpid(),os.getppid()}:continue
            command=process.info['cmdline'] or []
            if (process.info['name'] or '').lower() in {'python.exe','pythonw.exe'} and any(
                Path(a.replace('\\','/')).name in {'train_experiment1.py','train_experiment1_fast.py','train_accuracy_v2.py'} for a in command):
                raise SystemExit('CUDA start deferred: another training process is running; existing training untouched')
        if psutil.virtual_memory().available<4*2**30:
            raise SystemExit('CUDA start deferred: at least 4 GiB host RAM required')

import torch
PACKAGE_ROOT=Path(__file__).resolve().parents[1]
ROOT=PACKAGE_ROOT.parents[1]
sys.path.insert(0,str(PACKAGE_ROOT));sys.path.append(str(ROOT))
import src.common as _data_paths
_data_paths.ROOT=ROOT
from src.common import read,digest
from src.analoggym_campaign import sha
from src.windows_campaign_io import atomic_json,replace_with_retry
from src.train.fusion_cache import CachedFusion
from src.model.accuracy_network import AccuracyRGCN as EdgeCentricRGCN
from src.train.target_calibration import validate as validate_calibration
from src.train.accuracy_loss import supervision_loss
from src.train.accuracy_loop import TrainingCoverage,LossAccumulator,evaluate
from src.train.stage import build_optimizer

CONFIG=None
KEYS=['hidden_dim','num_layers','dropout','update_edges','use_domain_emb','use_film','collapse_relations','contract_residual','calibrated_targets','adapter_hidden','adapter_min_train_samples']
SOURCES=sorted({p.relative_to(ROOT).as_posix() for p in (PACKAGE_ROOT/'src').rglob('*.py')}|{Path(__file__).resolve().relative_to(ROOT).as_posix()})


def now():return datetime.now(timezone.utc).isoformat()
def torch_atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(value,temp);replace_with_retry(temp,path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');parser.add_argument('--config',required=True);args=parser.parse_args()
    CONFIG=(ROOT/args.config).resolve()
    if not CONFIG.is_relative_to(ROOT):raise ValueError('Configuration outside workspace')
    config=read(CONFIG);output=(ROOT/config['output']).resolve()
    if output.parent!=ROOT/'artifacts' or not output.name.startswith('experiment1_accuracy_v2_'):raise ValueError('Accuracy v2 requires its own experiment1_accuracy_v2_ output namespace')
    if args.resume and not (output/'last.pt').exists():raise ValueError('Resume requires an existing accuracy-v2 checkpoint')
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'training.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    status={'state':'initializing','pid':os.getpid(),'updated_at':now(),'test_used':False}
    def update(**fields):
        status.update(fields,updated_at=now());atomic_json(output/'status.json',status)
    try:
        if (output/'status.json').exists() and read(output/'status.json').get('state')=='complete':
            print('Experiment already complete; refusing to rerun the fixed test.',flush=True);return
        update()
        torch.set_num_threads(config['cpu_threads']);torch.manual_seed(config['seed'])
        if config['device']=='cuda':
            if not torch.cuda.is_available():raise RuntimeError('GPU training requested but CUDA is unavailable')
            torch.cuda.set_per_process_memory_fraction(.85);torch.backends.cuda.matmul.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        device=config['device'];data=CachedFusion(config)
        calibration=read(ROOT/config['target_calibration']);validate_calibration(calibration,data.tasks,data.manifest)
        if sha(data.path/'manifest.json')!=calibration['cache_manifest_sha256']:raise ValueError('Calibration cache fingerprint mismatch')
        model=EdgeCentricRGCN(data.schema,data.tasks,calibration=calibration,**{k:config[k] for k in KEYS}).to(device)
        optimizer=build_optimizer(model,'pretrain',config['learning_rate'],weight_decay=config['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['epochs'])
        recipe={'config':config,'execution_variant':'edge_rgcn_accuracy_v2','data_root':str(ROOT),'config_sha256':sha(CONFIG),'target_calibration_sha256':sha(ROOT/config['target_calibration']),'parent_migration_sha256':sha(output/'migration_lineage.json') if (output/'migration_lineage.json').exists() else None,'cache_manifest_sha256':sha(data.path/'manifest.json'),
            'split_sha256':sha(ROOT/config['split']),'implementation_sha256':{p:sha(ROOT/p) for p in SOURCES},
            'torch':torch.__version__,'GPU':torch.cuda.get_device_name(0) if device=='cuda' else None}
        if (output/'recipe.json').exists():
            if read(output/'recipe.json')!=recipe:raise ValueError('Training recipe changed; use a fresh output directory')
        else:atomic_json(output/'recipe.json',recipe)
        coverage=TrainingCoverage();history=[];best=-math.inf;best_epoch=None;bad=0;epoch0=0;batch0=0;partial=None
        checkpoint=output/'last.pt'
        if args.resume and checkpoint.exists():
            previous=torch.load(checkpoint,map_location=device,weights_only=True)
            if previous['recipe_digest']!=digest(recipe):raise ValueError('Resume fingerprint mismatch')
            model.load_state_dict(previous['model']);optimizer.load_state_dict(previous['optimizer']);scheduler.load_state_dict(previous['scheduler'])
            coverage.slots={g:set(s) for g,s in previous['coverage_slots'].items()};coverage.domains=set(previous['coverage_domains'])
            epoch0=previous['next_epoch'];batch0=previous['next_batch'];partial=previous['partial']
            history=previous['history'];best=previous['best'];best_epoch=previous['best_epoch'];bad=previous['bad']
            torch.set_rng_state(previous['rng_cpu'].cpu())
            if device=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in previous['rng_cuda']])
        elif checkpoint.exists():raise ValueError('Existing checkpoint: pass --resume or choose another output')
        def save(next_epoch,next_batch,stats):
            torch_atomic(checkpoint,{'recipe_digest':digest(recipe),'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),'coverage_slots':{g:sorted(s) for g,s in coverage.slots.items()},
                'coverage_domains':sorted(coverage.domains),'next_epoch':next_epoch,'next_batch':next_batch,'partial':stats,
                'history':history,'best':best,'best_epoch':best_epoch,'bad':bad,
                'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all() if device=='cuda' else []})
        total_train=data.manifest['role_counts']['train'];total_batches=math.ceil(total_train/config['batch_size'])
        update(state='training',epoch=epoch0+1,max_epochs=config['epochs'],batch=batch0,batches_per_epoch=total_batches,
            train_samples=total_train,validation_samples=data.manifest['role_counts']['validation'],test_samples=data.manifest['role_counts']['test'],
            device=device,GPU=recipe['GPU'],batch_size=config['batch_size'],best_epoch=best_epoch)
        if bad>=config['patience']:epoch0=config['epochs']
        for epoch in range(epoch0,config['epochs']):
            accumulated=LossAccumulator(model.head)
            stats=partial if epoch==epoch0 and partial is not None else {'samples':0,'grad_sum':0.,'grad_count':0,'grad_max':0.,'elapsed':0.,'loss_values':{}}
            accumulated.values=stats['loss_values'];base_elapsed=stats['elapsed'];started=time.perf_counter();last_status=started
            model.train();start_batch=batch0 if epoch==epoch0 else 0
            segment_samples=0;segment_started=time.perf_counter()
            for batch_number,batch in enumerate(data.batches('train',config['batch_size'],config['seed']+epoch,start_batch),start_batch+1):
                if (output/'STOP').exists() or psutil.virtual_memory().available<256*2**20:
                    stats['elapsed']=base_elapsed+time.perf_counter()-started;stats['loss_values']=accumulated.values
                    save(epoch,batch_number-1,stats);update(state='paused',reason='STOP requested or available RAM below 256 MiB',batch=batch_number-1);return
                if batch.split_role!='train':raise ValueError('Training role mismatch')
                batch=batch.to(device);optimizer.zero_grad(set_to_none=True)
                prediction=model(batch);loss,_=supervision_loss(prediction,batch,model.head);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
                coverage.observe(batch);accumulated.update(prediction.detach(),batch)
                segment_samples+=batch.num_graphs
                stats['samples']+=batch.num_graphs;stats['grad_sum']+=float(norm);stats['grad_count']+=1;stats['grad_max']=max(stats['grad_max'],float(norm))
                elapsed=base_elapsed+time.perf_counter()-started;stats['elapsed']=elapsed;stats['loss_values']=accumulated.values
                if time.perf_counter()-last_status>=10 or batch_number==total_batches:
                    update(state='training',epoch=epoch+1,batch=batch_number,samples_processed_in_epoch=stats['samples'],
                        epoch_training_seconds=elapsed,training_loss=accumulated.mean(),
                        samples_per_second=stats['samples']/elapsed,execution_variant='edge_rgcn_accuracy_v2',active_segment_samples=segment_samples,active_segment_seconds=time.perf_counter()-segment_started,active_segment_samples_per_second=segment_samples/max(1e-9,time.perf_counter()-segment_started),estimated_training_seconds_per_epoch=total_train*elapsed/max(1,stats['samples']),
                        gradient_norm=float(norm),GPU_peak_GiB=torch.cuda.max_memory_allocated()/2**30 if device=='cuda' else None)
                    print(now(),'epoch',epoch+1,'batch',batch_number,'/',total_batches,'loss',accumulated.mean(),flush=True);last_status=time.perf_counter()
                if batch_number%config['checkpoint_every_batches']==0:save(epoch,batch_number,stats)
            assert stats['samples']==total_train
            segment_seconds=time.perf_counter()-segment_started
            update(state='validating',epoch=epoch+1);validation_started=time.perf_counter()
            validation=evaluate(model,data.batches('validation',config['batch_size']),coverage,device=device)
            score=validation['macro_hit5'];metric='macro_hit5'
            if score is None:score=-validation['loss'];metric='negative_validation_loss'
            if not math.isfinite(score):raise ValueError('Invalid validation selection score')
            record={'epoch':epoch+1,'training_loss':accumulated.mean(),'training_samples':stats['samples'],
                'training_seconds':stats['elapsed'],'execution_variant':'edge_rgcn_accuracy_v2','implementation_switch_batch':start_batch,'active_segment_seconds':segment_seconds,'active_segment_samples':segment_samples,'gradient_norm_mean':stats['grad_sum']/stats['grad_count'],
                'gradient_norm_max':stats['grad_max'],'validation_seconds':time.perf_counter()-validation_started,
                'validation':validation,'selection_metric':metric,'learning_rate':optimizer.param_groups[0]['lr']}
            history.append(record)
            if score>best+1e-4:
                best=score;best_epoch=epoch+1;bad=0
                torch_atomic(output/'best.pt',{'model':model.state_dict(),'target_calibration':calibration,'schema':data.schema,'tasks':data.tasks,'config':config,
                    'status':'trained','model_class':'AccuracyRGCN','epoch':best_epoch,'recipe_digest':digest(recipe),
                    'trained_group_ids':sorted(coverage.slots),'trained_target_slots':{g:sorted(s) for g,s in coverage.slots.items()},
                    'trained_domain_ids':sorted(coverage.domains),'dataset_manifest_sha256':sha(ROOT/config['dataset']/'manifest.json')})
            else:bad+=1
            scheduler.step();save(epoch+1,0,None)
            atomic_json(output/'history.json',history);update(best_epoch=best_epoch,best_validation_score=best,completed_epochs=epoch+1)
            print(now(),'validated epoch',epoch+1,'score',score,'best',best,flush=True)
            if bad>=config['patience']:break
        if best_epoch is None:raise ValueError('No validation-selected checkpoint')
        best_checkpoint=torch.load(output/'best.pt',map_location=device,weights_only=True);model.load_state_dict(best_checkpoint['model'])
        coverage.slots={g:set(s) for g,s in best_checkpoint['trained_target_slots'].items()};coverage.domains=set(best_checkpoint['trained_domain_ids'])
        update(state='testing_once',test_used=True,best_epoch=best_epoch)
        if (output/'test_metrics.json').exists():
            test=read(output/'test_metrics.json')
            if test.get('recipe_digest')!=digest(recipe) or test.get('best_epoch')!=best_epoch:raise ValueError('Existing test result belongs to another model')
        else:
            test=evaluate(model,data.batches('test',config['batch_size']),coverage,device=device,role='test')
            test.update(recipe_digest=digest(recipe),best_epoch=best_epoch)
            atomic_json(output/'test_metrics.json',test)
        update(state='complete',completed_epochs=len(history),best_epoch=best_epoch,test_metrics='test_metrics.json',test_used=True)
        print('Experiment 1 complete; fixed test evaluated once.',flush=True)
    except BaseException as error:
        update(state='error',error_type=type(error).__name__,error=str(error));raise
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':main()
