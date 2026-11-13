"""Persistent train/validation-only optimization controller inside D:/EDA.

Observe the CPU head stage, coordinate checkpointed GPU handoff at the original
4-GiB gate, run a bounded joint pilot, retain validation-best weights, and recover
known resource failures. Numerical/integrity failures are never silently ignored.
"""
from pathlib import Path
from datetime import datetime,timezone
import os
import json
import time
import subprocess
import hashlib
import ctypes
import msvcrt
import psutil
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
CONTROL=ROOT/'monitoring/autonomous_training_v1';HEAD=ROOT/'artifacts/experiment1_autonomous_head_v1'
HEAD_MONITOR=ROOT/'monitoring/autonomous_head_v1';HANDOFF='autonomous_joint_handoff_v1'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(2**20),b''):h.update(part)
    return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.auto.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    deadline=time.time()+15
    while True:
        try:os.replace(temporary,path);return
        except PermissionError:
            if time.time()>deadline:raise
            time.sleep(.1)
def matching(status):
    try:return psutil.Process(status['pid']).create_time()==status['create_time']
    except (KeyError,psutil.NoSuchProcess):return False
def main():
    CONTROL.mkdir(parents=True,exist_ok=True);lock=(CONTROL/'controller.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    pins={p.relative_to(ROOT).as_posix():sha(p) for p in [HERE/'head_train.py',HERE/'config_head.json',HERE/'head_supervisor.py',HERE/'stream_graphs.py',HERE/'joint_train.py',Path(__file__)]}
    for path in (ROOT/'versions/rgcn_calibrated_head_v1/src').rglob('*.py'):pins[path.relative_to(ROOT).as_posix()]=sha(path)
    for name in ['.venv/Lib/site-packages/torch/version.py','.venv-gpu/Lib/site-packages/torch/version.py',
                 'data/model_ready/fusion_v2_accuracy_v2/target_calibration.json',
                 'data/model_ready/fusion_v2_frozen_e19_bf0d71ad/manifest.json',
                 'data/model_ready/fusion_v2_training_v1/manifest.json','data/fusion_v2/splits/id_stratified_v1.parquet']:
        pins[name]=sha(ROOT/name)
    state={'state':'initializing','controller_pid':os.getpid(),'controller_create_time':psutil.Process().create_time(),
        'updated_at':now(),'authorization':'User explicitly grants autonomous model/training decisions within EDA',
        'test_policy':'Train/validation only. Previously evaluated model/test report remain immutable.'}
    def update(**fields):state.update(fields,updated_at=now());atomic(CONTROL/'status.json',state)
    atomic(CONTROL/'policy.json',{'at':now(),'authorization':state['authorization'],'pinned_source_sha256':pins,
        'head_policy':read(HERE/'config_head.json'),'joint_policy':{'minimum_RAM_GiB':4,'effective_batch':256,'microbatch':128,
            'head_learning_rate':2.5e-5,'backbone_learning_rate':1e-5,'pilot_train_samples':5120,'pilot_validation_samples':5120,
            'max_epochs':48,'patience':6,'window_epochs':4,'minimum_window_gain':.002,'window_stop_backbone_lr':2.5e-6,
            'max_resource_retries':2,'test_evaluation':False},
        'model_selection':'Validation only; source guard and median-R2 guard for final recommendation',
        'numerical_or_integrity_failure':'stop affected stage, preserve checkpoints and report; no tolerance relaxation'})
    awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001));joint=None;joint_cfg=None;resource_retries=0;last_attempt=0.;stable=None;joint_disabled=False;last_gpu=0.;gpu_free=0.;last_data=0.
    head_last=dict(read(HEAD/'status.json')) if (HEAD/'status.json').exists() else {}
    def event(code,detail):
        with (CONTROL/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'at':now(),'code':code,'detail':detail})+'\n')
    def launch_head():
        if (HEAD/'STOP').exists():
            if (HEAD/'STOP').read_text(encoding='utf-8')!=HANDOFF:return False
            (HEAD/'STOP').unlink()
        if (HEAD/'status.json').exists() and read(HEAD/'status.json')['state']=='complete':return False
        mon=read(HEAD_MONITOR/'status.json') if (HEAD_MONITOR/'status.json').exists() else {}
        try:
            if psutil.Process(mon['supervisor_pid']).create_time()==mon['supervisor_create_time']:return True
        except (KeyError,psutil.NoSuchProcess):pass
        with (CONTROL/'head_supervisor_stdout.log').open('ab') as out,(CONTROL/'head_supervisor_stderr.log').open('ab') as err:
            subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-X','utf8','-B','-u',str(HERE/'head_supervisor.py')],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        return True
    def pointer(status,kind):
        run='experiment1_autonomous_head_v1' if kind=='head' else 'experiment1_autonomous_joint_v1'
        output=HEAD if kind=='head' else ROOT/joint_cfg['output']
        tested='artifacts/experiment1_accuracy_v2_cached_head/best.pt'
        atomic(ROOT/'monitoring/current_training.json',{'run_id':run,'state':status.get('state','initializing'),'updated_at':now(),
            'runtime_status':(output/'status.json').relative_to(ROOT).as_posix(),'monitor_status':'monitoring/autonomous_training_v1/status.json',
            'model':(output/'best.pt').relative_to(ROOT).as_posix() if (output/'best.pt').exists() else tested,
            'model_evaluation_status':'Validation-selected candidate; no new test evaluation','device':'cpu' if kind=='head' else 'cuda',
            'body_frozen':kind=='head','epoch':status.get('epoch'),'batch':status.get('batch'),'candidate_validation_hit5':status.get('best_validation_score'),
            'previous_tested_model':tested,'previous_test_metrics':'artifacts/experiment1_accuracy_v2_cached_head/test_metrics.json',
            'candidate_test_metrics':None,'test_policy':state['test_policy'],'results_report':'reports/autonomous_training.md'})
    try:
        launch_head()
        while True:
            if (CONTROL/'STOP').exists():
                if matching(head_last):(HEAD/'STOP').write_text('User stopped autonomous training',encoding='utf-8')
                if joint is not None and joint.poll() is None:(ROOT/joint_cfg['output']/'STOP').write_text('User stopped autonomous training',encoding='utf-8')
                update(state='controller_stop_requested');return
            if any(sha(ROOT/p)!=value for p,value in pins.items()):
                if matching(head_last):(HEAD/'STOP').write_text('Pinned source/config changed; preserve checkpoint for review',encoding='utf-8')
                if joint is not None and joint.poll() is None:(ROOT/joint_cfg['output']/'STOP').write_text('Pinned source/config changed',encoding='utf-8')
                update(state='source_changed_requires_review');return
            if time.time()-last_data>=900:
                folder=ROOT/'data/model_ready/fusion_v2_frozen_e19_bf0d71ad';manifest=read(folder/'manifest.json')
                bad=[f'{role}/{name}' for role,spec in manifest['roles'].items() for name,value in spec['files'].items() if sha(folder/role/name)!=value]
                if bad:
                    if matching(head_last):(HEAD/'STOP').write_text('Cached data integrity changed',encoding='utf-8')
                    if joint is not None and joint.poll() is None:(ROOT/joint_cfg['output']/'STOP').write_text('Cached data integrity changed',encoding='utf-8')
                    update(state='cached_data_changed_requires_review',changed=bad);return
                last_data=time.time()
            if (HEAD/'status.json').exists():head_last=read(HEAD/'status.json')
            available=psutil.virtual_memory().available
            if time.time()-last_gpu>=60:
                try:
                    result=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8,creationflags=subprocess.CREATE_NO_WINDOW)
                    gpu_free=float(result.stdout.strip().splitlines()[0]) if not result.returncode else 0.
                except (ValueError,IndexError,OSError,subprocess.TimeoutExpired):gpu_free=0.
                last_gpu=time.time()
            if joint is not None:
                output=ROOT/joint_cfg['output'];js=read(output/'status.json') if (output/'status.json').exists() else {}
                update(state='joint_stage_running',joint=js,available_RAM_GiB=available/2**30);pointer(js,'joint')
                if joint.poll() is None:time.sleep(20);continue
                if js.get('state')=='complete':
                    parent_meta=read(CONTROL/'joint_parent_metadata.json');parent_val=parent_meta['validation'];best_score=js.get('best_validation_score',0.)
                    epoch=js.get('best_joint_epoch',-1)
                    if epoch==0:chosen=read(output/'initial_full_validation.json')
                    elif epoch>0:chosen=next(r['validation'] for r in read(output/'history.json') if r['epoch']==epoch)
                    else:chosen=parent_val
                    source_ok=all(chosen['by_source'][s]['macro_hit5']>=parent_val['by_source'][s]['macro_hit5']-.01 and
                        chosen['by_source'][s]['median_r2']>=parent_val['by_source'][s]['median_r2']-.03 for s in parent_val['by_source'])
                    accepted=best_score>parent_val['macro_hit5']+1e-4 and chosen['median_r2']>=parent_val['median_r2']-.01 and source_ok
                    result={'at':now(),'parent_validation_hit5':parent_val['macro_hit5'],'best_joint_validation_hit5':best_score,
                        'improved':best_score>parent_val['macro_hit5']+1e-4,'accepted':accepted,'source_guard_passed':source_ok,
                        'candidate_validation':{k:v for k,v in chosen.items() if k!='per_target'},
                        'selected_checkpoint':joint_cfg['output']+'/best.pt' if accepted else 'artifacts/experiment1_autonomous_head_v1/best.pt','test_evaluated':False}
                    atomic(CONTROL/'joint_result.json',result)
                    if accepted:update(state='joint_refinement_complete',joint=js,result=result);pointer(js,'joint');return
                    event('joint_full_validation_not_promoted',result);joint=None;joint_disabled=True;launch_head();continue
                if js.get('state')=='pilot_rejected':
                    event('joint_pilot_rejected',js);joint=None;joint_disabled=True;last_attempt=time.time();update(state='joint_pilot_no_gain',joint=js)
                    launch_head();time.sleep(20);continue
                text=(js.get('error_type','')+' '+js.get('error','')).lower()
                transient=joint.returncode in {75,86} or js.get('state')=='paused' or any(x in text for x in ['out of memory','memoryerror','winerror 32'])
                if transient and resource_retries<2:
                    resource_retries+=1;event('joint_resource_retry',{'attempt':resource_retries,'status':js});update(state='waiting_joint_recovery',joint=js)
                    stable=None
                    while True:
                        if (CONTROL/'STOP').exists():update(state='controller_stop_requested');return
                        memory=psutil.virtual_memory().available
                        try:
                            probe=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8,creationflags=subprocess.CREATE_NO_WINDOW)
                            free=float(probe.stdout.strip().splitlines()[0]) if not probe.returncode else 0.
                        except (ValueError,IndexError,OSError,subprocess.TimeoutExpired):free=0.
                        ok=memory>=4*2**30 and free>=2048;stable=(stable or time.time()) if ok else None
                        update(available_RAM_GiB=memory/2**30)
                        if stable and time.time()-stable>=60:break
                        time.sleep(20)
                    if (output/'STOP').exists():(output/'STOP').unlink()
                    command=[str(ROOT/'.venv-gpu/Scripts/python.exe'),'-X','utf8','-B','-u',str(HERE/'joint_train.py'),'--config',str((HERE/'config_joint.json').relative_to(ROOT))]
                    if (output/'recipe.json').exists():command.append('--resume')
                    with (CONTROL/'joint_stdout.log').open('ab') as out,(CONTROL/'joint_stderr.log').open('ab') as err:
                        joint=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
                    continue
                event('joint_failure_requires_review',js);joint=None;joint_disabled=True;last_attempt=time.time();launch_head()
                update(state='joint_failed_parent_preserved',joint=js);time.sleep(20);continue
            pointer(head_last,'head');alive=matching(head_last)
            update(state='head_refinement_running' if alive else 'waiting_for_joint_resources',head=head_last,available_RAM_GiB=available/2**30,
                joint_start_requirement_GiB=4.,head_identity_matches=alive,GPU_free_MiB=gpu_free,joint_disabled=joint_disabled)
            if head_last.get('state')=='error':event('head_error',head_last);update(state='head_error_requires_review');return
            # If releasing this verified CPU worker is enough, checkpoint it
            # before checking the actual 4 GiB gate. No CUDA test overlaps it.
            rss=psutil.Process(head_last['pid']).memory_info().rss if alive else 0
            potential=not joint_disabled and gpu_free>=2048 and (available>=4.1*2**30 or alive and available>=3.5*2**30 and available+rss>=4.2*2**30)
            stable=(stable or time.time()) if potential else None
            if stable and time.time()-stable>=60 and time.time()-last_attempt>=3600:
                if alive:
                    flag=HEAD/'STOP'
                    if flag.exists():update(state='foreign_stop_marker');return
                    flag.write_text(HANDOFF,encoding='utf-8');update(state='checkpointing_head_for_joint');deadline=time.time()+180
                    while matching(head_last) and time.time()<deadline:time.sleep(2)
                    if matching(head_last):update(state='head_pause_timeout');return
                time.sleep(5);available=psutil.virtual_memory().available
                if available<4*2**30:
                    last_attempt=time.time();stable=None;launch_head();event('joint_gate_deferred',{'RAM_GiB':available/2**30});continue
                if (CONTROL/'joint_parent.pt').exists():update(state='existing_joint_artifacts_require_review');return
                source=HEAD/'best.pt';snapshot=CONTROL/'joint_parent.pt'
                with source.open('rb') as inp,snapshot.open('xb') as out:
                    for value in iter(lambda:inp.read(2**20),b''):out.write(value)
                if sha(source)!=sha(snapshot):raise RuntimeError('Head snapshot copy mismatch')
                history=read(HEAD/'history.json');chosen=next(r for r in reversed(history) if r['head_epoch_total']==head_last['best_total_head_epoch'])
                atomic(CONTROL/'joint_parent_metadata.json',{'source':source.relative_to(ROOT).as_posix(),'sha256':sha(snapshot),
                    'validation':{k:v for k,v in chosen['validation'].items() if k!='per_target'}})
                joint_cfg={'parent_checkpoint':snapshot.relative_to(ROOT).as_posix(),'parent_checkpoint_sha256':sha(snapshot),
                    'metadata_cache':'data/model_ready/fusion_v2_frozen_e19_bf0d71ad','output':'artifacts/experiment1_autonomous_joint_v1',
                    'effective_batch_size':256,'training_microbatch':128,'validation_microbatch':128,'lru_topologies':8,
                    'head_lr':2.5e-5,'backbone_lr':1e-5,'minimum_lr':1e-6,'max_epochs':48,'patience':6,'checkpoint_every_updates':100,
                    'window_epochs':4,'minimum_window_gain':.002,'window_stop_backbone_lr':2.5e-6,
                    'pilot_train_samples':5120,'pilot_validation_samples':5120,'pilot_lr_scales':[1.,.5],
                    'pilot_min_hit_gain':1e-4,'pilot_min_loss_gain':.005,'pilot_max_median_drop':.01,'test_policy':'No test evaluation'}
                atomic(HERE/'config_joint.json',joint_cfg)
                with (CONTROL/'joint_stdout.log').open('ab') as out,(CONTROL/'joint_stderr.log').open('ab') as err:
                    joint=subprocess.Popen([str(ROOT/'.venv-gpu/Scripts/python.exe'),'-X','utf8','-B','-u',str(HERE/'joint_train.py'),
                        '--config',str((HERE/'config_joint.json').relative_to(ROOT))],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
                update(state='joint_launched',joint_launcher_pid=joint.pid);continue
            # Respect actual convergence; do not revive completed head stages
            # merely because their optimizer can still take steps.
            if not alive and head_last.get('state') not in {'complete','error'}:launch_head()
            if not alive and head_last.get('state')=='complete' and joint_disabled:
                update(state='head_converged_joint_rejected',selected_checkpoint='artifacts/experiment1_autonomous_head_v1/best.pt');return
            time.sleep(20)
    except BaseException as error:
        update(state='controller_error',error_type=type(error).__name__,error=str(error));raise
    finally:
        if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()

if __name__=='__main__':main()
