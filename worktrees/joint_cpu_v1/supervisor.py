"""Supervise one bounded CPU joint worker; no test access or duplicate GPU job."""
from pathlib import Path
from datetime import datetime,timezone
import json
import time
import os
import hashlib
import subprocess
import ctypes
import msvcrt
import psutil
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
CONTROL=ROOT/'monitoring/joint_cpu_v1';OUTPUT=ROOT/'artifacts/experiment1_joint_cpu_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(2**20),b''):h.update(data)
    return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.supervisor.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    for attempt in range(30):
        try:os.replace(temp,path);return
        except PermissionError:time.sleep(.1)
    raise RuntimeError('Cannot atomically publish '+str(path))
def identity(status):
    try:
        p=psutil.Process(status['pid'])
        return abs(p.create_time()-status['create_time'])<.001 and any(str(a).replace('\\','/').endswith('joint_cpu_v1/train.py') for a in p.cmdline())
    except (KeyError,psutil.NoSuchProcess):return False
def main():
    CONTROL.mkdir(parents=True,exist_ok=True);lock=(CONTROL/'supervisor.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    cfg=read(HERE/'config.json');paths=[HERE/'train.py',HERE/'stream_graphs.py',HERE/'config.json',Path(__file__)]+list((ROOT/'versions/rgcn_calibrated_head_v1/src').rglob('*.py'))
    paths += [ROOT/cfg['parent_checkpoint'],ROOT/'data/model_ready/fusion_v2_training_v1/manifest.json',ROOT/'data/fusion_v2/splits/id_stratified_v1.parquet',ROOT/'.venv/Lib/site-packages/torch/version.py']
    pins={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    policy={'pinned_SHA256':pins,'backend':'CPU only','CUDA_minimum_available_RAM_GiB':4,'minimum_CPU_restart_RAM_GiB':1.5,'max_resource_restarts':3,
        'retained':'4layers/128hidden/48relations/FP32/deterministic/originalcontracts/splits/Mask','test_used':False,
        'GPU_plan':'CUDA remains gated; CPU worker will not overlap a large CUDA test. No silent cross-recipe migration.'}
    if (CONTROL/'policy.json').exists() and read(CONTROL/'policy.json')!=policy:raise RuntimeError('Supervisor policy/source changed; review lineage first')
    atomic(CONTROL/'policy.json',policy)
    state={'state':'initializing','supervisor_pid':os.getpid(),'supervisor_create_time':psutil.Process().create_time(),'updated_at':now(),'restarts':0}
    def update(**v):state.update(v,updated_at=now());atomic(CONTROL/'status.json',state)
    awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001));child=None;stable=None;restarts=0;attempted=False
    try:
        while True:
            status=read(OUTPUT/'status.json') if (OUTPUT/'status.json').exists() else {}
            alive=identity(status);available=psutil.virtual_memory().available
            if (CONTROL/'STOP').exists() or any(sha(ROOT/p)!=h for p,h in pins.items()):
                if alive:(OUTPUT/'STOP').write_text('Supervisor stop/source change: preserve checkpoint',encoding='utf-8')
                update(state='safe_stop_requested',training=status);return
            if status:
                atomic(ROOT/'monitoring/current_training.json',{
                    'run_id':'experiment1_joint_cpu_v1','state':status['state'],'updated_at':now(),'runtime_status':'artifacts/experiment1_joint_cpu_v1/status.json',
                    'monitor_status':'monitoring/joint_cpu_v1/status.json','device':'cpu','body_frozen':False,
                    'epoch':status.get('epoch'),'batch':status.get('batch'),'best_validation_score':status.get('best_validation_score'),
                    'model':'artifacts/experiment1_joint_cpu_v1/best.pt' if (OUTPUT/'best.pt').exists() else cfg['parent_checkpoint'],
                    'candidate_test_metrics':None,'test_policy':'Training/validation only; preserve all previous test results',
                    'results_report':'reports/joint_cpu_training.md'})
            if alive:
                update(state='joint_training_active',training=status,worker_identity_matches=True,available_RAM_GiB=available/2**30)
                time.sleep(15);continue
            if status.get('state') in {'complete','pilot_rejected'}:
                parent=read(ROOT/'reports/experiment1_validation_review.json');selection={'at':now(),'parent_checkpoint':cfg['parent_checkpoint'],'parent_validation_hit5':parent['macro_hit5'],'accepted':False,'test_used':False}
                if status['state']=='complete':
                    epoch=status['best_joint_epoch']
                    if epoch==0:best=read(OUTPUT/'initial_full_validation.json')
                    elif epoch>0:best=next(r['validation'] for r in read(OUTPUT/'history.json') if r['epoch']==epoch)
                    else:best=None
                    if best is not None:
                        guards=all(best['by_source'][s]['macro_hit5']>=r['macro_hit5']-.01 and best['by_source'][s]['median_r2']>=r['median_r2']-.03 for s,r in parent['sources'].items())
                        selection.update(candidate_validation_hit5=best['macro_hit5'],candidate_median_r2=best['median_r2'],source_guards=guards,
                            accepted=best['macro_hit5']>parent['macro_hit5']+1e-4 and best['median_r2']>=parent['median_r2']-.01 and guards)
                selection['selected_checkpoint']='artifacts/experiment1_joint_cpu_v1/best.pt' if selection['accepted'] else cfg['parent_checkpoint']
                atomic(CONTROL/'selection.json',selection);update(state='stage_complete' if status['state']=='complete' else 'pilot_no_validation_gain',training=status,selection=selection);return
            if attempted and child is not None and child.poll() is None:
                update(state='worker_initializing',available_RAM_GiB=available/2**30);time.sleep(5);continue
            if attempted:
                code=child.poll() if child is not None else None
                error=(status.get('error','')+' '+status.get('error_type','')).lower()
                recoverable=code in {75,86} or status.get('state')=='paused' or any(t in error for t in ['memoryerror','out of memory','winerror 32'])
                if not recoverable:update(state='worker_error_requires_review',training=status,exit_code=code);return
                if restarts>=3:update(state='resource_retry_limit',training=status);return
            stable=(stable or time.time()) if available>=cfg['minimum_start_RAM_GiB']*2**30 else None
            update(state='waiting_CPU_resources',available_RAM_GiB=available/2**30,training=status)
            if not stable or time.time()-stable<15:time.sleep(5);continue
            if (OUTPUT/'STOP').exists():update(state='explicit_stop_preserved',training=status);return
            if attempted:restarts+=1
            command=[str(ROOT/'.venv/Scripts/python.exe'),'-X','utf8','-B','-u',str(HERE/'train.py'),'--config','worktrees/joint_cpu_v1/config.json']
            if (OUTPUT/'recipe.json').exists():command.append('--resume')
            with (CONTROL/'training_stdout.log').open('ab') as out,(CONTROL/'training_stderr.log').open('ab') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
            attempted=True;stable=None;update(state='worker_launched',launcher_pid=child.pid,restarts=restarts);time.sleep(5)
    except BaseException as error:
        update(state='supervisor_error',error_type=type(error).__name__,error=str(error));raise
    finally:
        if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
if __name__=='__main__':main()
