"""Bounded native Windows supervisor for the small validation-only head stage."""
from pathlib import Path
from datetime import datetime,timezone
import json
import time
import hashlib
import subprocess
import ctypes
import os
import psutil
import msvcrt
ROOT=Path(__file__).resolve().parents[2];PACKAGE=Path(__file__).resolve().parent
CONTROL=ROOT/'monitoring/cached_head_cpu_continue_v1'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for v in iter(lambda:f.read(2**20),b''):h.update(v)
    return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');os.replace(temp,path)
def main():
    CONTROL.mkdir(parents=True,exist_ok=True);lock=(CONTROL/'supervisor.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    cfg=read(PACKAGE/'config_cpu.json');output=ROOT/cfg['output'];state={'state':'initializing','supervisor_pid':os.getpid(),
        'supervisor_create_time':psutil.Process().create_time(),'updated_at':now(),'output':cfg['output'],
        'policy':'bounded auto-resume, source/config/recipe unchanged, no test evaluation','restarts':0}
    pins={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [PACKAGE/'train_cpu.py',PACKAGE/'config_cpu.json',Path(__file__)]}
    write(CONTROL/'policy.json',{'authorized_by':'User allowed autonomous training changes and recovery without confirmation',
        'pinned_source_sha256':pins,'minimum_RAM_GiB':cfg['minimum_start_RAM_GiB'],'memory_scope':'CPU-only cached head; no CUDA context',
        'maximum_restarts':3,'restart_resources_stable_seconds':60,'keep_system_awake':True,'test_policy':cfg['test_policy']})
    def update(**v):state.update(v,updated_at=now());write(CONTROL/'status.json',state)
    awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001));child=None;restarts=0
    try:
        while True:
            if any(sha(ROOT/p)!=v for p,v in pins.items()):update(state='source_changed_requires_review');return
            if (CONTROL/'STOP').exists():update(state='supervisor_stop_requested');return
            if (output/'STOP').exists():update(state='user_stop_requested');return
            available=psutil.virtual_memory().available
            if available<cfg['minimum_start_RAM_GiB']*2**30:
                update(state='waiting_RAM',available_RAM_GiB=available/2**30);time.sleep(15);continue
            # A verified existing worker may be adopted if this observer restarts.
            existing=None
            if (output/'status.json').exists():
                status=read(output/'status.json')
                if status['state']=='complete':update(state='complete',training=status);return
                try:
                    process=psutil.Process(status['pid'])
                    if process.create_time()==status['create_time'] and any(str(a).replace('\\','/').endswith('cached_head_continue_v1/train_cpu.py') for a in process.cmdline()):existing=process
                except psutil.NoSuchProcess:pass
            if existing is None:
                command=[str(ROOT/'.venv/Scripts/python.exe'),'-X','utf8','-B','-u',str(PACKAGE/'train_cpu.py')]
                if (output/'recipe.json').exists():command.append('--resume')
                with (CONTROL/'training_stdout.log').open('ab') as out,(CONTROL/'training_stderr.log').open('ab') as err:
                    child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
                update(state='worker_launched',launcher_pid=child.pid,command=command)
            else:update(state='verified_worker_adopted',training_pid=existing.pid)
            while child.poll() is None if child is not None and existing is None else existing.is_running():
                if (output/'status.json').exists():
                    try:
                        status=read(output/'status.json');process=psutil.Process(status['pid'])
                        identity=process.create_time()==status['create_time']
                        update(state='running',training=status,training_pid=status['pid'],training_identity_matches=identity,
                            available_RAM_GiB=psutil.virtual_memory().available/2**30)
                    except (psutil.NoSuchProcess,PermissionError,json.JSONDecodeError):pass
                if (CONTROL/'STOP').exists():update(state='observer_stopped_worker_left_running');return
                time.sleep(15)
            status=read(output/'status.json') if (output/'status.json').exists() else {}
            if status.get('state')=='complete':update(state='complete',training=status);return
            if status.get('reason')=='user_stop' or (output/'STOP').exists():update(state='user_stop_requested',training=status);return
            error=(status.get('error_type','')+' '+status.get('error','')).lower()
            if status.get('state')=='error' and not any(k in error for k in ['out of memory','memoryerror','winerror 32','sharing violation']):
                update(state='worker_error_requires_review',training=status);return
            restarts+=1
            if restarts>3:update(state='restart_limit_reached',training=status);return
            if (output/'recipe.json').exists():
                recipe=read(output/'recipe.json')
                if any(sha(ROOT/p)!=v for p,v in recipe['source_sha256'].items()):update(state='recipe_source_changed');return
            update(state='waiting_stable_resources',training=status,restarts=restarts)
            stable=None
            while stable is None or time.time()-stable<60:
                if (CONTROL/'STOP').exists() or (output/'STOP').exists():update(state='user_stop_requested');return
                ok=psutil.virtual_memory().available>=cfg['minimum_start_RAM_GiB']*2**30
                stable=(stable or time.time()) if ok else None;time.sleep(15)
            child=None
    finally:
        if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()

if __name__=='__main__':main()
