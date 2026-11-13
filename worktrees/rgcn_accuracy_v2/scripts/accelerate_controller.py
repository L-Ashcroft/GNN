"""Authorized bounded handoff: save parent, run verified head stage, or restore it.

Native Windows, independent directory, no process killing and no source edits.
"""
from pathlib import Path
from datetime import datetime,timezone
import json
import hashlib
import subprocess
import time
import ctypes
import os
import psutil

ROOT=Path(__file__).resolve().parents[3];PACKAGE=Path(__file__).resolve().parents[1]
CONTROL=ROOT/'monitoring/accuracy_acceleration_v1'
PARENT=ROOT/'artifacts/experiment1_fusion_id_fast_v1'
OLD_MONITOR=ROOT/'monitoring/experiment1_fusion_id_fast_v1'
MARKER='Authorized cached-head efficiency experiment; parent checkpoint preserved; autonomous restore on failure.'

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for value in iter(lambda:stream.read(2**20),b''):h.update(value)
    return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');os.replace(temp,path)
def main():
    if CONTROL.exists():raise FileExistsError('Controller directory already exists; inspect before another handoff')
    CONTROL.mkdir();identity=read(PARENT/'launch.json');parent_pid=identity['training_pid'];created=identity['training_create_time']
    recipe=read(PARENT/'recipe.json');protected=recipe['implementation_sha256']
    status={'controller_pid':os.getpid(),'controller_create_time':psutil.Process().create_time(),'updated_at':now(),
        'state':'preparing','authorization':'User approved autonomous training/model changes and replacement without further questions',
        'parent_pid':parent_pid,'parent_create_time':created,'parent_output':str(PARENT.relative_to(ROOT)),
        'candidate_output':'artifacts/experiment1_accuracy_v2_cached_head','test_used_for_selection':False}
    def update(**values):status.update(values,updated_at=now());atomic(CONTROL/'status.json',status)
    def pins_ok():return all((ROOT/p).exists() and sha(ROOT/p)==h for p,h in protected.items())
    def alive():
        try:return psutil.Process(parent_pid).create_time()==created
        except psutil.NoSuchProcess:return False
    def own_flags():return all(p.exists() and p.read_text(encoding='utf-8')==MARKER for p in [PARENT/'STOP',OLD_MONITOR/'DISABLE_AUTO_RESUME'])
    def restore(reason):
        stage=ROOT/status['candidate_output']
        if (stage/'status.json').exists() and read(stage/'status.json').get('test_used'):
            update(state='test_phase_requires_integrity_review',reason=reason);return
        if alive():update(state='parent_still_alive_no_duplicate',reason=reason);return
        if not pins_ok() or not own_flags():update(state='restore_guard_blocked',reason=reason);return
        update(state='waiting_to_restore_parent',reason=reason)
        stable=None;deadline=time.time()+900
        while time.time()<deadline:
            healthy=psutil.virtual_memory().available>=3*2**30
            stable=(stable or time.time()) if healthy else None
            if stable and time.time()-stable>=120:break
            update(available_RAM_GiB=psutil.virtual_memory().available/2**30);time.sleep(15)
        else:update(state='waiting_resources_parent_preserved',reason=reason);return
        if not pins_ok() or not own_flags() or alive():update(state='restore_prelaunch_guard_blocked');return
        # Delete only our exact two marker files, never directories.
        for p in [PARENT/'STOP',OLD_MONITOR/'DISABLE_AUTO_RESUME']:
            if not p.resolve().is_relative_to(ROOT) or p.read_text(encoding='utf-8')!=MARKER:raise RuntimeError('Foreign control marker')
            p.unlink()
        command=[str(ROOT/'.venv-gpu/Scripts/python.exe'),'-B','-u',str(ROOT/'versions/rgcn_cached_fast_v1/scripts/train_experiment1_fast.py'),
            '--config','config/experiment1_fusion_id_fast_v1.json','--resume']
        with (PARENT/'training_stdout.log').open('ab') as out,(PARENT/'training_stderr.log').open('ab') as err:
            p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        update(state='parent_resume_launched',launcher_pid=p.pid,reason=reason)
    awake=False
    try:
        if not alive() or not pins_ok():raise RuntimeError('Parent identity or source fingerprint mismatch')
        if read(PARENT/'status.json').get('test_used'):raise RuntimeError('Parent already entered fixed test phase')
        pilot=read(PACKAGE/'results/head_pilot_v1/summary.json');assert pilot['status']=='complete' and not pilot['test_labels_used']
        if pilot['variants']['calibration_only']['macro_hit5']<=pilot['initial_parent_validation']['macro_hit5']:
            raise RuntimeError('No pilot gain to justify further stage')
        for path in [PARENT/'STOP',OLD_MONITOR/'DISABLE_AUTO_RESUME']:
            if path.exists():raise RuntimeError('Existing control flag: '+str(path))
        awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        atomic(CONTROL/'authorization.json',{'at':now(),'user_scope':'Autonomous training changes authorized; prioritize efficient reliable experimental results',
            'protected_parent_recipe_sha256':sha(PARENT/'recipe.json'),'controller_sha256':sha(Path(__file__)),
            'pause':'graceful checkpoint only','fallback':'unchanged parent --resume after stable resources','parent_optimizer_not_migrated_to_new_objective':True})
        (OLD_MONITOR/'DISABLE_AUTO_RESUME').write_text(MARKER,encoding='utf-8');(PARENT/'STOP').write_text(MARKER,encoding='utf-8')
        update(state='waiting_parent_checkpoint_exit');deadline=time.time()+600
        while alive() and time.time()<deadline:time.sleep(2)
        if alive():raise RuntimeError('Graceful pause timeout; parent not killed')
        paused=read(PARENT/'status.json')
        if paused.get('state')!='paused':raise RuntimeError('Parent exited without expected paused status')
        if not pins_ok():raise RuntimeError('Parent source changed')
        atomic(CONTROL/'parent_paused.json',{'status':paused,'last_checkpoint_sha256':sha(PARENT/'last.pt'),'best_checkpoint_sha256':sha(PARENT/'best.pt'),
            'recipe_sha256':sha(PARENT/'recipe.json'),'at':now()})
        parent_copy=CONTROL/'parent_best.pt'
        with (PARENT/'best.pt').open('rb') as source,parent_copy.open('xb') as dest:
            for value in iter(lambda:source.read(2**20),b''):dest.write(value)
        if sha(parent_copy)!=sha(PARENT/'best.pt'):raise RuntimeError('Best checkpoint copy mismatch')
        update(state='checking_CUDA_resources',available_RAM_GiB=psutil.virtual_memory().available/2**30)
        stable=None;deadline=time.time()+180
        while time.time()<deadline:
            healthy=psutil.virtual_memory().available>=4*2**30
            stable=(stable or time.time()) if healthy else None
            if stable and time.time()-stable>=10:break
            update(available_RAM_GiB=psutil.virtual_memory().available/2**30);time.sleep(5)
        else:restore('4 GiB CUDA gate not met after parent release');return
        best_epoch=paused['best_epoch']
        plan={'parent_checkpoint':str(parent_copy.relative_to(ROOT)).replace('\\','/'),'parent_checkpoint_sha256':sha(parent_copy),
            'parent_best_epoch':best_epoch,'target_calibration':'data/model_ready/fusion_v2_accuracy_v2/target_calibration.json',
            'output':status['candidate_output'],'feature_cache':f'data/model_ready/fusion_v2_frozen_e{best_epoch}_{sha(parent_copy)[:8]}',
            'head_batch_size':1024,'head_epochs':6,'patience':2,'minimum_hit5_gain':.005,'maximum_median_r2_drop':.01,
            'selection':'full validation only; no test label for selection','warm_start':'reuse trained graph body, train-derived head calibration/reset; new optimizer'}
        atomic(CONTROL/'plan.json',plan)
        command=[str(ROOT/'.venv-gpu/Scripts/python.exe'),'-X','utf8','-B','-u',str(PACKAGE/'scripts/run_cached_head_experiment.py'),
                 '--plan',str((CONTROL/'plan.json').relative_to(ROOT))]
        with (CONTROL/'stage_stdout.log').open('ab') as out,(CONTROL/'stage_stderr.log').open('ab') as err:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        update(state='cached_head_stage_running',stage_launcher_pid=child.pid,command=command,available_RAM_GiB=psutil.virtual_memory().available/2**30)
        while child.poll() is None:
            stage_status=ROOT/status['candidate_output']/'status.json'
            if stage_status.exists():
                try:update(stage=read(stage_status),available_RAM_GiB=psutil.virtual_memory().available/2**30)
                except (json.JSONDecodeError,PermissionError):pass
            time.sleep(15)
        stage_output=ROOT/status['candidate_output'];final=read(stage_output/'status.json') if (stage_output/'status.json').exists() else {}
        if child.returncode==0 and final.get('state')=='complete' and (stage_output/'test_metrics.json').exists():
            if not pins_ok():raise RuntimeError('Parent source changed during stage')
            update(state='complete_promoted',stage=final,first_experiment_result=str((stage_output/'test_metrics.json').relative_to(ROOT)),
                old_training_preserved_paused=True,accepted_new_protocol=True)
            (OLD_MONITOR/'STOP_MONITOR').write_text('Accepted independent cached-head stage complete; original run preserved paused.',encoding='utf-8')
        else:restore('Stage ended without accepted complete result: '+str(child.returncode)+' '+str(final.get('state')))
    except BaseException as error:
        update(state='controller_error',error=repr(error))
        if own_flags():restore(repr(error))
        else:raise
    finally:
        if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':main()
