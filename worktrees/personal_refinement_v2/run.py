"""Train revised known-topology GNN readouts; validation-only model selection."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ.setdefault(key,'1')
import argparse
import gc
import math
import shutil
import time
import numpy as np
import psutil
from common import HERE, ROOT, OUTPUT, CACHE, NEW_CACHE, PARENT_OUTPUT, read, atomic, sha, digest, pins, now, replace
from features import Inputs, ridge, denormalize, metric_rows, summarize
from neural import Readout


def save_arrays(path, arrays):
    temp=path.with_suffix('.npz.tmp')
    with temp.open('wb') as handle: np.savez(handle,**arrays)
    replace(temp,path)


def run(job, resume=False):
    folder=OUTPUT/job['id'];folder.mkdir(parents=True,exist_ok=True)
    state={'state':'initializing','pid':os.getpid(),'create_time':psutil.Process().create_time(),'job_id':job['id'],
           'test_used':False,'device':'cpu','backend':'NumPy FP64 residual neural readout'}
    def update(**values):
        state.update(values,updated_at=now(),available_RAM_GiB=psutil.virtual_memory().available/2**30,
                     RSS_MiB=psutil.Process().memory_info().rss/2**20)
        atomic(folder/'status.json',state)
    update();psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    try:
        if job['mode']!='physical':
            # Run extraction as an independent worker so Torch memory is released
            # before this lightweight neural optimizer starts.
            if not (NEW_CACHE/job['parent']/'manifest.json').exists():
                update(state='waiting_embeddings');return 76
        data=Inputs(job['budget'])
        if job['parent']: data.parent_check(job['parent'])
        blocks={};heads={};rng=np.random.default_rng(job['seed'])
        for name in sorted(data.structures):
            if psutil.virtual_memory().available<320*2**20:
                update(state='paused',reason='initialization memory gate');return 75
            block=data.block(name,parent=job['parent'],mode=job['mode']); w,b=ridge(block,job['ridge_alpha'])
            blocks[name]=block;heads[name]=Readout(block,w,b,rng,hidden=64,dropout=.05)
            gc.collect()
        recipe={'job':job,'source_sha256':pins(),'source_cache_manifest_sha256':sha(CACHE/'manifest.json'),
                'training_ids_sha256':data.training_id_hash_cache_order,'target_training_labels':len(data.allowed),
                'validation_labels_additional':2000,'test_samples_used':0,'supported_protocols':['known_topology_ID','labeled_few_shot'],
                'not_supported':'LOTO or zero-shot with untrained target heads',
                'loss':'Topology-equal observed target MSE; topology-local train-only target scales; missing-only mask unchanged',
                'model':'Frozen parent edge-RGCN plus standardized numeric-device/condition skip and residual neural output',
                'target_policy':'Linear for signed voltage/phase and logarithmic units; symlog for remaining units; inference clipped to training support',
                'input_policy':'First19 numeric device fields in pinned device order +46 known conditions; graph branch retains original90 fields and all52 relations',
                'normalization':{name:block['normalization'] for name,block in blocks.items()},
                'parent_checkpoint_sha256':sha(PARENT_OUTPUT/job['parent']/'best.pt') if job['parent'] else None,
                'selection':'Validation hit5 improvement1e-4; median/mean R2 cannot fall more0.02 from initialized ridge, negative targets at most+2',
                'final_test_policy':'Not accessed by this training program; historical test is not a new blind test'}
        if (folder/'recipe.json').exists(): assert resume and read(folder/'recipe.json')==recipe,'Pinned recipe changed'
        else: assert not resume;atomic(folder/'recipe.json',recipe)
        recipe_hash=digest(recipe)
        def evaluate():
            rows=[]
            for name,head in heads.items():
                normalized,_=head.forward(blocks[name]['validation_x'])
                rows.extend(metric_rows(name,denormalize(normalized,blocks[name]),blocks[name]))
            return summarize(rows)
        def arrays(): return {name+'___'+key:value for name,head in heads.items() for key,value in head.state().items()}
        history=[];best_epoch=0;bad=0;updates=0;first_epoch=0;factor=1.;schedule_bad=0;best_score=-1.
        restoring=resume and (folder/'cursor.json').exists()
        if restoring:
            initial=read(folder/'initial_validation.json')
            assert (folder/'best.npz').exists(), 'Missing selected checkpoint'
        else:
            initial=evaluate();atomic(folder/'initial_validation.json',initial)
            best_score=initial['macro_hit5'];save_arrays(folder/'best.npz',arrays())
        def checkpoint(next_epoch):
            stamp=time.time_ns()
            saved_path=folder/f'state_epoch_{next_epoch:04d}_{stamp}.npz'
            save_arrays(saved_path,arrays())
            best_path=folder/f'selected_at_epoch_{next_epoch:04d}_{stamp}.npz'
            shutil.copyfile(folder/'best.npz',best_path)
            atomic(folder/'cursor.json',{'recipe_hash':recipe_hash,'next_epoch':next_epoch,'history':history,
                  'best_score':best_score,'best_epoch':best_epoch,'bad':bad,'updates':updates,'factor':factor,
                  'schedule_bad':schedule_bad,'rng':rng.bit_generator.state,
                  'state_file':saved_path.name,'state_sha256':sha(saved_path),
                  'best_file':best_path.name,'best_sha256':sha(best_path)})
        if restoring:
            cursor=read(folder/'cursor.json');assert cursor['recipe_hash']==recipe_hash
            state_path=folder/cursor['state_file'];assert sha(state_path)==cursor['state_sha256']
            best_path=folder/cursor['best_file'];assert sha(best_path)==cursor['best_sha256']
            shutil.copyfile(best_path,folder/'best.npz')
            with np.load(state_path,allow_pickle=False) as saved:
                for name,head in heads.items():head.restore({key:saved[name+'___'+key] for key in head.state()})
            history=cursor['history'];first_epoch=cursor['next_epoch'];best_score=cursor['best_score'];best_epoch=cursor['best_epoch']
            bad=cursor['bad'];updates=cursor['updates'];factor=cursor['factor'];schedule_bad=cursor['schedule_bad']
            rng.bit_generator.state=cursor['rng']
        checkpoint(first_epoch)
        tick=time.perf_counter()
        end=first_epoch if bad>=job['patience'] else job['max_epochs']
        for epoch in range(first_epoch,end):
            if (folder/'STOP').exists() or psutil.virtual_memory().available<320*2**20:
                checkpoint(epoch);update(state='paused',reason='Memory or explicit STOP',epoch=epoch);return 75
            orders={name:rng.permutation(len(block['train_x'])) for name,block in blocks.items()}
            steps=max(math.ceil(len(v)/32) for v in orders.values());epoch_loss=0.
            update(state='training',role='train',epoch=epoch+1,max_epochs=job['max_epochs'],batches_per_epoch=steps,
                   train_samples=len(data.allowed),validation_label_budget=2000,total_target_supervision_budget=len(data.allowed)+2000,
                   best_validation_hit5=best_score,best_epoch=best_epoch,mode=job['mode'])
            for step in range(steps):
                gradients={};losses=[]
                for name,head in heads.items():
                    indices=orders[name][step*32:(step+1)*32]
                    if not len(indices):continue
                    value,grad=head.backward(blocks[name]['train_x'][indices],indices,rng)
                    gradients[name]=grad;losses.append(value)
                n=len(gradients);norm=np.sqrt(sum(float(np.square(g).sum())/n**2 for part in gradients.values() for g in part.values()))
                clip=min(1.,5./max(norm,1e-12))
                for name,grad in gradients.items():heads[name].update(grad,1e-4*factor,1e-3*factor,multiplier=clip/n)
                updates+=1;epoch_loss+=sum(losses)/n
                if step%10==0:update(batch=step+1,role_samples=len(data.allowed),samples_done=min((step+1)*32*len(heads),len(data.allowed)),training_loss=epoch_loss/(step+1))
            current=evaluate();atomic(folder/'latest_validation.json',current)
            score=current['macro_hit5']
            admissible=(current['median_r2']>=initial['median_r2']-.02 and current['macro_r2']>=initial['macro_r2']-.02
                        and current['negative_r2_targets']<=initial['negative_r2_targets']+2)
            history.append({'epoch':epoch+1,'training_loss':epoch_loss/steps,'validation_hit5':score,
                            'validation_median_r2':current['median_r2'],'validation_macro_r2':current['macro_r2'],
                            'negative_r2_targets':current['negative_r2_targets'],'admissible':admissible})
            if score>best_score+1e-4 and admissible:
                best_score=score;best_epoch=epoch+1;bad=0;schedule_bad=0;save_arrays(folder/'best.npz',arrays())
            else:bad+=1;schedule_bad+=1
            if schedule_bad>10:factor=max(factor*.5,.001);schedule_bad=0
            if (epoch+1)%5==0 or bad>=job['patience']:checkpoint(epoch+1)
            atomic(folder/'history.json',history)
            update(state='epoch_complete',completed_epochs=epoch+1,best_validation_hit5=best_score,best_epoch=best_epoch,
                   validation_hit5=score,validation_median_r2=current['median_r2'],bad_epochs=bad,updates=updates)
            if (epoch+1)%10==0:print(now(),job['id'],'epoch',epoch+1,'best_val_hit5',best_score,flush=True)
            if bad>=job['patience']:break
        checkpoint(len(history))
        with np.load(folder/'best.npz',allow_pickle=False) as saved:
            for name,head in heads.items():head.restore({key:saved[name+'___'+key] for key in head.state()})
        selected=evaluate();atomic(folder/'selected_validation.json',selected)
        atomic(folder/'selection_frozen.json',{'at':now(),'recipe_hash':recipe_hash,'best_epoch':best_epoch,
               'best_checkpoint_sha256':sha(folder/'best.npz'),'validation_hit5':selected['macro_hit5'],
               'validation_median_r2':selected['median_r2'],'test_used':False})
        update(state='complete',best_epoch=best_epoch,best_validation_hit5=selected['macro_hit5'],
               selected_validation_median_r2=selected['median_r2'],selected_validation_macro_r2=selected['macro_r2'],
               validation_negative_r2_targets=selected['negative_r2_targets'],test_used=False,
               seconds=time.perf_counter()-tick,trainable_parameters=sum(v.size for h in heads.values() for v in h.parameters.values()))
        return 0
    except BaseException as exc:
        update(state='error',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    job=next(x for x in read(HERE/'suite.json')['jobs'] if x['id']==args.job)
    raise SystemExit(run(job,args.resume))
