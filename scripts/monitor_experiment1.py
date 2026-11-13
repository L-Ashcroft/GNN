"""Windows observer with explicitly authorized, bounded checkpoint recovery."""
import argparse
import ctypes
from datetime import datetime,timezone
import gc
import json
import msvcrt
from pathlib import Path
import os
import subprocess
import time
import traceback
import psutil
from experiment1_health import ROOT,read,atomic,now,digest,runtime_findings,history_summary,quality_findings,nonfinite_paths

OUTPUT=ROOT/'monitoring/experiment1_fusion_id_v1'
PRODUCTION=ROOT/'artifacts/experiment1_fusion_id_v1'

class Awake:
    def __init__(self):
        self.enabled=False;self.error=None;self.function=ctypes.WinDLL('kernel32',use_last_error=True).SetThreadExecutionState
        self.function.argtypes=[ctypes.c_uint];self.function.restype=ctypes.c_uint
    def set(self,enable):
        if self.enabled==enable:return
        result=self.function(0x80000001 if enable else 0x80000000)
        if result:self.enabled=enable;self.error=None
        else:self.error=ctypes.get_last_error()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--interval',type=int,default=30)
    parser.add_argument('--once',action='store_true');parser.add_argument('--no-keep-awake',action='store_true')
    parser.add_argument('--auto-resume',action='store_true');args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    lock=(OUTPUT/'monitor.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    snapshot=read(ROOT/'worktrees/rgcn_optimization/snapshot.json');expected_pid=snapshot['production_before']['pid'];expected_created=snapshot['pid_create_time']
    recipe=read(PRODUCTION/'recipe.json');protected=dict(recipe['implementation_sha256'])
    for name in ['config/experiment1_fusion.json','artifacts/experiment1_fusion_id_v1/recipe.json']:
        protected[name]=digest(ROOT/name)
    controller=None
    if args.auto_resume:
        from experiment1_recovery import RecoveryController
        controller=RecoveryController(OUTPUT);protected=controller.auth['protected_sha256']
        expected_pid,expected_created=controller.expected
    recovery_policy='bounded_checkpoint_auto_resume' if controller else 'observe_only_no_restart'
    known_events={};summaries=[];history_stamp=None;last_gpu=last_hash=0.;gpu=None;hash_ok=True
    awake=Awake();last_poll=time.time();iteration=0
    atomic(OUTPUT/'policy.json',{'monitor_pid':os.getpid(),'expected_training_pid':expected_pid,'expected_training_create_time':expected_created,
        'poll_seconds':args.interval,'recovery_policy':recovery_policy,'keep_awake_while_training':not args.no_keep_awake,
        'protected_sha256':protected,'training_files_written':False,'test_set_used':False,
        'GPU_context_created':False,'script_sha256':digest(Path(__file__)),
        'health_module_sha256':digest(ROOT/'scripts/experiment1_health.py'),
        'recovery_module_sha256':digest(ROOT/'scripts/experiment1_recovery.py') if controller else None,
        'checkpoint_validator_sha256':digest(ROOT/'scripts/validate_experiment1_resume.py') if controller else None,
        'training_log_append_on_authorized_resume':bool(controller)})
    def event(code,severity,message):
        key=(code,severity,message if code in {'extreme_negative_r2','validation_hit5_drop','validation_loss_jump'} else '')
        if time.time()-known_events.get(key,0)<1800:return
        known_events[key]=time.time()
        with (OUTPUT/'alerts.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'at_UTC':now(),'code':code,'severity':severity,'message':message},ensure_ascii=False)+'\n')
    try:
        while True:
            iteration+=1;runtime=[];tick=time.time();status=None;identity=False
            try:
                if (OUTPUT/'STOP_MONITOR').exists():
                    atomic(OUTPUT/'monitor_stopped.json',{'at_UTC':now(),'reason':'Observer-only STOP_MONITOR requested'});break
                if tick-last_poll>max(180,args.interval*5):event('monitor_gap','warning',f'Observer polling gap {tick-last_poll:.0f}s; possible sleep/scheduling interruption')
                last_poll=tick;status=read(PRODUCTION/'status.json')
                recovery=None
                if controller:
                    recovery=controller.step(status);expected_pid,expected_created=controller.expected
                try:
                    process=psutil.Process(expected_pid);identity=process.is_running() and process.create_time()==expected_created and status['pid']==expected_pid
                except psutil.NoSuchProcess:identity=False
                available=psutil.virtual_memory().available;disk=psutil.disk_usage(str(ROOT)).free
                active=identity and status['state'] not in {'complete','paused','error'}
                waiting_recovery=bool(recovery and (recovery['state'].startswith('waiting_') or recovery['state'] in
                    {'validating_checkpoint','replacement_launched','launch_intent_recorded','replacement_training_adopted'}))
                awake.set((active or waiting_recovery) and not args.no_keep_awake and not args.once)
                validation_seconds=summaries[-1]['validation_seconds'] if summaries else 300
                runtime=runtime_findings(status,identity,available,disk,validation_seconds=validation_seconds)
                path=PRODUCTION/'history.json';stamp=(path.stat().st_mtime_ns,path.stat().st_size) if path.exists() else None
                if stamp!=history_stamp and stamp is not None and available>768*2**20:
                    history=read(path);bad=nonfinite_paths(history)
                    if bad:runtime.append({'code':'nonfinite_history','severity':'critical','message':str(bad[:5])})
                    else:
                        summaries=history_summary(history);atomic(OUTPUT/'history_summary.json',summaries);history_stamp=stamp
                    del history;gc.collect()
                if tick-last_hash>=300:
                    mismatches=[name for name,expected in protected.items() if digest(ROOT/name)!=expected]
                    hash_ok=not mismatches;last_hash=tick
                    if mismatches:event('implementation_changed','critical',str(mismatches))
                if not hash_ok:runtime.append({'code':'implementation_changed','severity':'critical','message':'Pinned implementation/config changed; current recipe must not be bypassed'})
                if tick-last_gpu>=300:
                    try:
                        command=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,temperature.gpu',
                            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,
                            creationflags=subprocess.CREATE_NO_WINDOW)
                        gpu={'sampled_at_UTC':now(),'raw':command.stdout.strip(),'returncode':command.returncode};last_gpu=tick
                    except (OSError,subprocess.TimeoutExpired) as error:
                        gpu={'error':str(error)};last_gpu=tick;event('gpu_query_error','warning',str(error))
                stderr=PRODUCTION/'training_stderr.log'
                if stderr.exists() and stderr.stat().st_size:
                    runtime.append({'code':'training_stderr_nonempty','severity':'warning','message':f'Training stderr has {stderr.stat().st_size} bytes'})
                last_checkpoint=PRODUCTION/'last.pt'
                checkpoint={'exists':last_checkpoint.exists(),'age_seconds':tick-last_checkpoint.stat().st_mtime if last_checkpoint.exists() else None}
                if active and checkpoint['age_seconds'] is not None and checkpoint['age_seconds']>3600:
                    runtime.append({'code':'checkpoint_old','severity':'warning','message':'Last checkpoint more than one hour old; may include sleep time'})
                quality=quality_findings(summaries)
                for item in runtime:event(item['code'],item['severity'],item['message'])
                for item in quality:event(item['code'],item['severity'],f"epoch {summaries[-1]['epoch']}: "+item['message'])
                atomic(OUTPUT/'status.json',{'updated_at_UTC':now(),'monitor_pid':os.getpid(),'monitor_create_time':psutil.Process().create_time(),
                    'monitor_state':'running' if active else 'recovery_observing' if controller else 'terminal_or_process_missing','poll_count':iteration,
                    'training_pid':expected_pid,'training_identity_matches':identity,'training':status,
                    'runtime_health':'critical' if any(x['severity']=='critical' for x in runtime) else 'warning' if runtime else 'ok',
                    'runtime_findings':runtime,'quality_health':'needs_attention' if quality else 'no_heuristic_alert',
                    'quality_findings':quality,'last_completed_validation':summaries[-1] if summaries else None,
                    'available_RAM_GiB':available/2**30,'disk_free_GiB':disk/2**30,'gpu_readonly_sample':gpu,
                    'checkpoint':checkpoint,'protected_fingerprints_match':hash_ok,
                    'temporary_system_awake_request':awake.enabled,'awake_request_error':awake.error,
                    'recovery_policy':recovery_policy,'recovery':recovery,'test_set_used':status.get('test_used')})
                atomic(OUTPUT/'last_good_training_status.json',status)
                if args.once or status['state']=='complete' or (not active and controller is None):break
            except Exception as error:
                event('monitor_read_error','warning',str(error))
                atomic(OUTPUT/'observer_error.json',{'updated_at_UTC':now(),'error':str(error),'traceback':traceback.format_exc()})
                if not psutil.pid_exists(expected_pid):awake.set(False)
                if args.once:raise
            time.sleep(max(5,args.interval))
    finally:
        awake.set(False);lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()

if __name__=='__main__':main()
