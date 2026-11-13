"""Launch the approved derived run and its isolated bounded observer, hidden on Windows."""
from pathlib import Path
import hashlib
import json
import subprocess
import time
from datetime import datetime,timezone
import psutil

ROOT=Path(__file__).resolve().parents[3]
OUTPUT=ROOT/'artifacts/experiment1_fusion_id_fast_v1'
MONITOR=ROOT/'monitoring/experiment1_fusion_id_fast_v1'
SCRIPTS=ROOT/'versions/rgcn_cached_fast_v1/scripts'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def main():
    if (OUTPUT/'launch.json').exists():raise FileExistsError('Launch already recorded; never launch twice')
    for p in psutil.process_iter(['name','cmdline']):
        if (p.info['name'] or '').lower() in {'python.exe','pythonw.exe'} and any(
            Path(a.replace('\\','/')).name in {'train_experiment1.py','train_experiment1_fast.py'} for a in p.info['cmdline'] or []):
            raise RuntimeError(f'Training command still alive: {p.pid}')
    if psutil.virtual_memory().available<3*2**30:raise RuntimeError('Need 3 GiB RAM to resume accepted formal run')
    receipt=read(OUTPUT/'migration_receipt.json');assert receipt['every_state_component_unchanged'] and receipt['speedup']>1.10
    assert read(MONITOR/'migration_checkpoint_validation.json')['status']=='valid'
    assert read(MONITOR/'release_integration_checks.json')['status']=='passed'
    recipe=read(OUTPUT/'recipe.json')
    for path,value in recipe['implementation_sha256'].items():assert sha(ROOT/path)==value,path
    assert sha(ROOT/'config/experiment1_fusion_id_fast_v1.json')==recipe['config_sha256']
    assert sha(OUTPUT/'last.pt')==receipt['new_checkpoint_sha256']
    command=[str(ROOT/'.venv-gpu/Scripts/python.exe'),'-B','-u',str(SCRIPTS/'train_experiment1_fast.py'),
             '--config','config/experiment1_fusion_id_fast_v1.json','--resume']
    launch={'at_UTC':now(),'command':command,'authorized_by':'User approved pause, acceptance and migration after measured improvement',
            'parent_cursor':{'epoch':16,'completed_batches':1024},'state':'launching'}
    write(OUTPUT/'launch.json',launch)
    with (OUTPUT/'training_stdout.log').open('ab') as out,(OUTPUT/'training_stderr.log').open('ab') as err:
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
            close_fds=True,creationflags=subprocess.CREATE_NO_WINDOW)
    launch['launcher_pid']=process.pid;write(OUTPUT/'launch.json',launch)
    for _ in range(120):
        if (OUTPUT/'status.json').exists():
            status=read(OUTPUT/'status.json')
            if status.get('state')=='error':raise RuntimeError(f'Formal initialization failed: {status}')
            if status.get('state')=='training' and status.get('batch',0)>=1024:break
        if process.poll() is not None:raise RuntimeError('Trainer exited during initialization')
        time.sleep(1)
    else:raise RuntimeError('Initialization timeout; inspect output, do not spawn again')
    worker=psutil.Process(status['pid']);created=worker.create_time()
    assert any(Path(a.replace('\\','/')).name=='train_experiment1_fast.py' for a in worker.cmdline())
    old_auth=read(ROOT/'monitoring/experiment1_fusion_id_v1/recovery_authorization.json')
    protected=dict(recipe['implementation_sha256'])
    for path in ['config/experiment1_fusion_id_fast_v1.json','artifacts/experiment1_fusion_id_fast_v1/recipe.json',
                 'artifacts/experiment1_fusion_id_fast_v1/migration_lineage.json']:
        protected[path]=sha(ROOT/path)
    for path,value in old_auth['protected_sha256'].items():
        if path.startswith('data/') or path.startswith('config/gnn_'):protected[path]=value
    monitor_sources={p.relative_to(ROOT).as_posix():sha(p) for p in SCRIPTS.glob('*.py') if p.name!='train_experiment1_fast.py'}
    auth={'enabled':True,'authorized_at_UTC':now(),'authorization':'Inherited user-approved bounded checkpoint recovery, for the explicitly approved migrated version only.',
          'initial_training_pid':worker.pid,'initial_training_create_time':created,'GPU':recipe['GPU'],'torch':recipe['torch'],
          'protected_sha256':protected,'recovery_implementation_sha256':monitor_sources,'policy':old_auth['policy']}
    write(MONITOR/'recovery_authorization.json',auth)
    observer_command=[str(ROOT/'.venv/Scripts/python.exe'),'-X','utf8','-B','-u',str(SCRIPTS/'monitor_experiment1.py'),'--auto-resume']
    with (MONITOR/'observer_stdout.log').open('ab') as out,(MONITOR/'observer_stderr.log').open('ab') as err:
        observer=subprocess.Popen(observer_command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
            close_fds=True,creationflags=subprocess.CREATE_NO_WINDOW)
    launch.update(state='training',training_pid=worker.pid,training_create_time=created,observer_launcher_pid=observer.pid,observer_command=observer_command)
    write(OUTPUT/'launch.json',launch)
    print(json.dumps(launch,indent=2))

if __name__=='__main__':main()
