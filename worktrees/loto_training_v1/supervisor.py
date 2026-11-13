"""Native Windows LOTO supervisor; safe CPU fallback and qualified GPU handoff."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time,os,subprocess,msvcrt,ctypes
import psutil
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def atomic(p,v):
    q=p.with_suffix('.supervisor.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
    for i in range(100):
        try:os.replace(q,p);return
        except PermissionError:time.sleep(.1)
    raise RuntimeError('Cannot publish supervisor status')
def same(s):
    try:
        p=psutil.Process(s['pid']);return abs(p.create_time()-s['create_time'])<.001 and any(str(a).replace('\\','/').endswith('loto_training_v1/train.py') for a in p.cmdline())
    except (KeyError,psutil.NoSuchProcess):return False
def gpu_free():
    try:
        r=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8,creationflags=subprocess.CREATE_NO_WINDOW)
        return float(r.stdout.strip().splitlines()[0]) if r.returncode==0 else 0.
    except (OSError,ValueError,IndexError,subprocess.TimeoutExpired):return 0.
def main():
    cfg=read(HERE/'config.json');output=ROOT/cfg['output'];control=ROOT/'monitoring/experiment2_loto_v1'/cfg['fold_id'];control.mkdir(parents=True,exist_ok=True)
    lock=(control/'supervisor.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    protocol=read(ROOT/cfg['protocol']);fold=next(f for f in protocol['folds'] if f['fold_id']==cfg['fold_id'])
    paths=[HERE/n for n in ['runtime.py','data.py','fit.py','model.py','train.py','qualify.py','supervisor.py','config.json']]
    paths+=list((ROOT/'versions/rgcn_calibrated_head_v1/src').rglob('*.py'))
    paths += [ROOT/cfg['protocol'],ROOT/fold['path'],output/'normalization.json',ROOT/'.venv/Lib/site-packages/torch/version.py',ROOT/'.venv-gpu/Lib/site-packages/torch/version.py']
    pins={p.relative_to(ROOT).as_posix():sha(p) for p in paths};policy={'source_sha256':pins,'fold_id':cfg['fold_id'],'counts':fold['counts'],
        'CPU_minimum_start_RAM_GiB':cfg['minimum_CPU_start_RAM_GiB'],'CUDA_minimum_actual_RAM_GiB':4.,'max_resource_restarts':3,
        'CUDA_handoff':'Checkpoint CPU, verify actual RAM>=4GiB, qualify implementation, restore same pinned recipe; otherwise resumeCPU',
        'heldout_test':'Never used for fitting or epoch selection','private_data_used':False,'pretrained_ID_state_used':False}
    if (control/'policy.json').exists():assert read(control/'policy.json')==policy,'Supervisor sources/recipe changed'
    else:atomic(control/'policy.json',policy)
    state={'state':'initializing','supervisor_pid':os.getpid(),'supervisor_create_time':psutil.Process().create_time(),'updated_at':now(),'restarts':0,'fold_id':cfg['fold_id']}
    def update(**v):state.update(v,updated_at=now());atomic(control/'status.json',state)
    def event(code,detail):
        with (control/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'at':now(),'code':code,'detail':detail})+'\n')
    def launch(backend):
        exe=ROOT/('.venv-gpu/Scripts/python.exe' if backend=='cuda' else '.venv/Scripts/python.exe')
        command=[str(exe),'-X','utf8','-B','-u',str(HERE/'train.py'),'--backend',backend]
        if (output/'recipe.json').exists():command.append('--resume')
        with (control/'training_stdout.log').open('ab') as out,(control/'training_stderr.log').open('ab') as err:
            p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        event('launch',{'backend':backend,'launcher_pid':p.pid,'command':command});return p
    def qualify():
        if psutil.virtual_memory().available<4*2**30 or gpu_free()<3072:return False
        with (control/'cuda_qualification_stdout.log').open('ab') as out,(control/'cuda_qualification_stderr.log').open('ab') as err:
            p=subprocess.Popen([str(ROOT/'.venv-gpu/Scripts/python.exe'),'-X','utf8','-B','-u',str(HERE/'qualify.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        deadline=time.time()+180
        while p.poll() is None:
            update(state='qualifying_CUDA',available_RAM_GiB=psutil.virtual_memory().available/2**30)
            if time.time()>deadline:raise RuntimeError('CUDA qualifier timed out; no competing trainer will be started')
            time.sleep(5)
        accepted=(output/'cuda_acceptance.json').exists() and read(output/'cuda_acceptance.json')['passed'] and p.returncode==0
        event('cuda_qualification',{'passed':accepted,'returncode':p.returncode});return accepted
    child=None;backend='cpu';attempted=False;stable=None;retry=0;cuda_disabled=False;last_cuda_attempt=0.;cuda_stable=None;marker='LOTO_qualified_CUDA_handoff'
    awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    try:
        while True:
            s=read(output/'status.json') if (output/'status.json').exists() else {};alive=same(s);available=psutil.virtual_memory().available
            if (control/'STOP').exists() or any(sha(ROOT/p)!=h for p,h in pins.items()):
                if alive:(output/'STOP').write_text('User stop or pinned source change; preserve fold checkpoint',encoding='utf-8')
                update(state='safe_stop_requested',training=s);return
            if s:
                atomic(ROOT/'monitoring/current_training.json',{'run_id':'experiment2_loto_v1','state':s['state'],'updated_at':now(),'fold_id':cfg['fold_id'],
                    'runtime_status':(output/'status.json').relative_to(ROOT).as_posix(),'monitor_status':(control/'status.json').relative_to(ROOT).as_posix(),
                    'device':s['device'],'epoch':s.get('epoch'),'batch':s.get('batch'),'heldout_topologies':fold['test_topologies'],
                    'counts':fold['counts'],'private_AnalogGym_used':False,'ID_model_preserved':'versions/rgcn_local_id_v1/publication.json','results_report':'reports/experiment2_loto_training.md'})
            if alive:
                backend=s['device'];update(state='training_active',training=s,worker_identity_matches=True,available_RAM_GiB=available/2**30,CUDA_disabled=cuda_disabled)
                # Host RAM released by the verified CPU worker may make the
                # original4GiB gate attainable; verify it again after exit.
                rss=psutil.Process(s['pid']).memory_info().rss
                potential=backend=='cpu' and not cuda_disabled and (output/'last.pt').exists() and available>=3.5*2**30 and available+rss>=4.3*2**30 and gpu_free()>=3072
                cuda_stable=(cuda_stable or time.time()) if potential else None
                if cuda_stable and time.time()-cuda_stable>=60 and time.time()-last_cuda_attempt>3600:
                    (output/'STOP').write_text(marker,encoding='utf-8');update(state='checkpointing_CPU_for_CUDA');deadline=time.time()+150
                    while same(s) and time.time()<deadline:time.sleep(2)
                    if same(s):raise RuntimeError('CPU pause timed out; no CUDA process launched')
                    time.sleep(5);last_cuda_attempt=time.time();cuda_stable=None
                    accepted=qualify() if psutil.virtual_memory().available>=4*2**30 else False
                    if (output/'STOP').exists() and (output/'STOP').read_text(encoding='utf-8')==marker:(output/'STOP').unlink()
                    if accepted:backend='cuda';child=launch('cuda');attempted=True;stable=None;time.sleep(5);continue
                    event('cuda_handoff_deferred_CPU_retained',{'available_RAM_GiB':psutil.virtual_memory().available/2**30})
                    if (output/'cuda_acceptance.json').exists() and not read(output/'cuda_acceptance.json')['passed']:cuda_disabled=True
                    child=None;attempted=False;stable=None
                else:time.sleep(15);continue
            if s.get('state')=='complete':
                update(state='fold_complete',training=s,summary=(output/'final_summary.json').relative_to(ROOT).as_posix());return
            if attempted and child is not None and child.poll() is None:update(state='worker_initializing',available_RAM_GiB=available/2**30);time.sleep(5);continue
            if attempted:
                error=(s.get('error_type','')+' '+s.get('error','')).lower();code=child.poll() if child is not None else None
                transient=code in {75,86} or s.get('state')=='paused' or any(k in error for k in ['memoryerror','out of memory','winerror 32','winerror 5'])
                if not transient:update(state='worker_error_requires_review',training=s,exit_code=code);return
                if retry>=3:update(state='resource_retry_limit',training=s);return
                retry+=1;event('resource_recovery',{'attempt':retry,'previous_backend':backend,'status':s});backend='cpu';cuda_disabled=True;attempted=False;child=None;stable=None
            if (output/'STOP').exists():update(state='explicit_stop_preserved',training=s);return
            stable=(stable or time.time()) if available>=cfg['minimum_CPU_start_RAM_GiB']*2**30 else None
            update(state='waiting_resources',available_RAM_GiB=available/2**30,restarts=retry,training=s)
            if not stable or time.time()-stable<15:time.sleep(5);continue
            # First launch can use CUDA only after the same admission/qualification.
            if not attempted and not (output/'last.pt').exists() and not cuda_disabled and available>=4.2*2**30 and gpu_free()>=3072:
                backend='cuda' if qualify() else 'cpu'
            else:backend='cpu'
            child=launch(backend);attempted=True;stable=None;update(state='worker_launched',backend=backend,launcher_pid=child.pid);time.sleep(5)
    except BaseException as e:update(state='supervisor_error',error_type=type(e).__name__,error=str(e));raise
    finally:
        if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
if __name__=='__main__':main()
