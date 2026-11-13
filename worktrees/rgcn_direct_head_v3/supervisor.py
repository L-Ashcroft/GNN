"""Persistent safe supervision for the independent direct-head candidate."""
from pathlib import Path
import os,json,time,hashlib,subprocess,ctypes,msvcrt
from datetime import datetime,timezone
import psutil
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
CONTROL=ROOT/'monitoring/rgcn_direct_head_v3';OUTPUT=ROOT/'artifacts/experiment1_rgcn_direct_head_v3'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def atomic(p,v):
    q=p.with_suffix('.supervisor.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
    for i in range(40):
        try:os.replace(q,p);return
        except PermissionError:time.sleep(.1)
    raise RuntimeError('Cannot update '+str(p))
def same(s):
    try:
        p=psutil.Process(s['pid']);return abs(p.create_time()-s['create_time'])<.001 and any(str(a).replace('\\','/').endswith('rgcn_direct_head_v3/train.py') for a in p.cmdline())
    except (psutil.NoSuchProcess,KeyError):return False
def main():
    CONTROL.mkdir(parents=True,exist_ok=True);lock=(CONTROL/'supervisor.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    cfg=read(HERE/'config.json');paths=[HERE/n for n in ['model.py','train.py','checks.py','supervisor.py','config.json']]
    paths+=[ROOT/p for p in ['worktrees/representation_diagnostic_v1/run.py',cfg['target_calibration'],cfg['graph_checkpoint'],cfg['feature_cache']+'/manifest.json','.venv/Lib/site-packages/torch/version.py']]
    pins={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    atomic(CONTROL/'policy.json',{'pinned_SHA256':pins,'backend':'CPU','max_resource_restarts':3,'minimum_start_RAM_GiB':cfg['minimum_start_RAM_GiB'],
        'test_used':False,'parent_preserved':True,'model_changes':'Explicit independent direct/residual heads, same frozen 4layer/48relation RGCN',
        'unknown_contracts':'Rejected; no untrained output head evaluation'})
    state={'state':'initializing','supervisor_pid':os.getpid(),'supervisor_create_time':psutil.Process().create_time(),'updated_at':now(),'restarts':0}
    def update(**v):state.update(v,updated_at=now());atomic(CONTROL/'status.json',state)
    child=None;attempted=False;stable=None;restarts=0;awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    try:
        while True:
            s=read(OUTPUT/'status.json') if (OUTPUT/'status.json').exists() else {};alive=same(s);available=psutil.virtual_memory().available
            if (CONTROL/'STOP').exists() or any(sha(ROOT/p)!=h for p,h in pins.items()):
                if alive:(OUTPUT/'STOP').write_text('Supervisor/source-change stop; preserve checkpoint',encoding='utf-8')
                update(state='safe_stop_requested',training=s);return
            if s:
                atomic(ROOT/'monitoring/current_training.json',{'run_id':'experiment1_rgcn_direct_head_v3','state':s['state'],'updated_at':now(),
                    'runtime_status':'artifacts/experiment1_rgcn_direct_head_v3/status.json','monitor_status':'monitoring/rgcn_direct_head_v3/status.json',
                    'device':'cpu','body_frozen':True,'head_type':'independent_direct_residual','epoch':s.get('epoch'),'batch':s.get('batch'),
                    'candidate_validation_hit5':s.get('best_validation_score'),'candidate_model':'artifacts/experiment1_rgcn_direct_head_v3/best.pt',
                    'previous_selected_model':cfg['graph_checkpoint'],'candidate_test_metrics':None,'results_report':'reports/rgcn_readout_revision_v3.md'})
            if alive:update(state='training_active',training=s,worker_identity_matches=True,available_RAM_GiB=available/2**30);time.sleep(15);continue
            if s.get('state')=='complete':
                epoch=s['best_head_epoch'];h=read(OUTPUT/'history.json') if (OUTPUT/'history.json').exists() else []
                best=read(OUTPUT/'initial_validation.json') if epoch==0 else next(r['validation'] for r in h if r['epoch']==epoch)
                parent=read(ROOT/'reports/experiment1_validation_review.json')
                guards=all(best['by_source'][g]['macro_hit5']>=r['macro_hit5']-.01 and best['by_source'][g]['median_r2']>=r['median_r2']-.03 for g,r in parent['sources'].items())
                accepted=best['macro_hit5']>parent['macro_hit5']+1e-4 and best['median_r2']>=parent['median_r2']-.01 and guards and best['negative_r2_targets']<=parent['negative_r2_targets']
                result={'at':now(),'accepted':accepted,'parent_validation_hit5':parent['macro_hit5'],'candidate_validation':{k:v for k,v in best.items() if k!='per_target'},
                    'source_guards_passed':guards,'negative_R2_guard':best['negative_r2_targets']<=parent['negative_r2_targets'],
                    'selected_checkpoint':'artifacts/experiment1_rgcn_direct_head_v3/best.pt' if accepted else cfg['graph_checkpoint'],'test_evaluated':False}
                atomic(CONTROL/'selection.json',result);update(state='stage_complete',training=s,selection=result);return
            if attempted and child is not None and child.poll() is None:update(state='worker_initializing',available_RAM_GiB=available/2**30);time.sleep(5);continue
            if attempted:
                code=child.poll();text=(s.get('error_type','')+' '+s.get('error','')).lower()
                if not (code in {75,86} or s.get('state')=='paused' or any(k in text for k in ['memoryerror','out of memory','winerror 32'])):update(state='worker_error_requires_review',training=s,exit_code=code);return
                if restarts>=3:update(state='resource_retry_limit',training=s);return
            stable=(stable or time.time()) if available>=cfg['minimum_start_RAM_GiB']*2**30 else None
            update(state='waiting_resources',available_RAM_GiB=available/2**30,training=s)
            if not stable or time.time()-stable<15:time.sleep(5);continue
            if (OUTPUT/'STOP').exists():update(state='explicit_stop_preserved',training=s);return
            command=[str(ROOT/'.venv/Scripts/python.exe'),'-X','utf8','-B','-u',str(HERE/'train.py')]
            if (OUTPUT/'recipe.json').exists():command.append('--resume')
            if attempted:restarts+=1
            with (CONTROL/'training_stdout.log').open('ab') as out,(CONTROL/'training_stderr.log').open('ab') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
            attempted=True;stable=None;update(state='worker_launched',launcher_pid=child.pid,restarts=restarts);time.sleep(5)
    except BaseException as e:update(state='supervisor_error',error_type=type(e).__name__,error=str(e));raise
    finally:
        if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
if __name__=='__main__':main()
