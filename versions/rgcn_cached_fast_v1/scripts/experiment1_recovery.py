"""Bounded auto-resume controller. Never terminates a training process."""
import ast
from datetime import datetime,timezone
import json
import msvcrt
from pathlib import Path
import subprocess
import time
import psutil
from experiment1_health import ROOT,read,atomic,digest,now

def retryable_status(status):
    if status.get('test_used') or status.get('state') in {'complete','testing_once'}:return False,'terminal_or_test_phase'
    if status.get('state') not in {'training','validating','initializing','paused','error'}:return False,'unknown_training_state'
    if status.get('state')=='error':
        message=(status.get('error_type','')+' '+status.get('error','')).lower()
        if any(s in message for s in ['nonfinite','nan','inf observed','mismatch','invalid','assertion','illegal memory','device-side assert']):
            return False,'numerical_data_or_integrity_error'
        if not any(s in message for s in ['out of memory','memoryerror','winerror 32','sharing violation','winerror 5']):
            return False,'unclassified_training_error_requires_review'
    return True,None

def recovery_gate(context,policy):
    if context['training_alive']:return 'training_alive'
    if context.get('other_training'):return 'other_training_present'
    if context.get('test_seen'):return 'test_phase_requires_manual_review'
    if context.get('stop_requested'):return 'user_stop_requested'
    if not context.get('fingerprints_ok'):return 'fingerprint_mismatch'
    allowed,reason=retryable_status(context['status'])
    if not allowed:return reason
    if context.get('attempts_24h',0)>=policy['max_restarts_24h']:return 'restart_limit_reached'
    if context.get('seconds_since_attempt',float('inf'))<policy['cooldown_seconds']:return 'waiting_cooldown'
    if context['available_ram']<policy['minimum_available_RAM_GiB']*2**30:return 'waiting_RAM'
    if context['free_disk']<policy['minimum_disk_GiB']*2**30:return 'waiting_disk'
    if context.get('resource_stable_seconds',0)<policy['resource_stable_seconds']:return 'waiting_stable_resources'
    if context.get('dead_seconds',0)<policy['death_grace_seconds']:return 'waiting_exit_confirmation'
    if not context.get('checkpoint_exists'):return 'checkpoint_missing'
    if context.get('checkpoint_age',0)<30:return 'waiting_stable_checkpoint'
    return 'eligible_for_checkpoint_validation'

class RecoveryController:
    def __init__(self,output):
        self.output=Path(output);self.auth=read(self.output/'recovery_authorization.json')
        if self.auth.get('enabled') is not True:raise ValueError('Auto-resume not authorized')
        self.policy=self.auth['policy'];self.path=self.output/'recovery_state.json'
        self.state=read(self.path) if self.path.exists() else {'expected_pid':self.auth['initial_training_pid'],
            'expected_create_time':self.auth['initial_training_create_time'],'attempts':[],'dead_since':None,'stable_since':None,'test_seen':False}
        self.info={};self.validator_process=None

    @property
    def expected(self):return self.state['expected_pid'],self.state['expected_create_time']

    def alive(self):
        try:return psutil.Process(self.expected[0]).create_time()==self.expected[1]
        except psutil.NoSuchProcess:return False

    def fingerprints_ok(self):
        pinned=dict(self.auth['protected_sha256']);pinned.update(self.auth.get('recovery_implementation_sha256',{}))
        return all((ROOT/name).exists() and digest(ROOT/name)==value for name,value in pinned.items())

    def training_processes(self):
        result=[]
        for process in psutil.process_iter(['pid','name','cmdline','create_time']):
            try:
                info=process.info;args=info['cmdline'] or []
                if (info['name'] or '').lower() not in {'python.exe','pythonw.exe'}:continue
                if not any(str(a).replace('\\','/').split('/')[-1] in {'train_experiment1.py','train_experiment1_fast.py'} for a in args):continue
                if Path(process.cwd()).resolve()!=ROOT:continue
                result.append({'pid':process.pid,'create_time':process.create_time(),'own_version':any(str(a).replace('\\','/').endswith('versions/rgcn_cached_fast_v1/scripts/train_experiment1_fast.py') for a in args)})
            except psutil.NoSuchProcess:continue
            except psutil.AccessDenied:result.append({'pid':process.pid,'identity_unverified':True})
        return result

    def save(self,state,**details):
        self.info={'state':state,'updated_at_UTC':now(),'policy':'bounded_checkpoint_auto_resume','attempts_total':len(self.state['attempts']),**details}
        self.state['last_decision']=self.info;atomic(self.path,self.state);return self.info

    def step(self,status):
        tick=time.time();self.state['test_seen']=self.state['test_seen'] or bool(status.get('test_used')) or (ROOT/'artifacts/experiment1_fusion_id_fast_v1/test_metrics.json').exists()
        previous_probe=self.state.get('last_probe_at')
        if previous_probe is not None and (tick-previous_probe>90 or tick<previous_probe):self.state['stable_since']=None
        self.state['last_probe_at']=tick
        if self.alive():
            self.state['dead_since']=self.state['stable_since']=None
            return self.save('training_alive')
        active=self.training_processes()
        # Follow a verified replacement, whether launched here or by the user;
        # never spawn another trainer alongside it.
        candidate=next((p for p in active if p['pid']==status.get('pid') and p.get('own_version') and 'create_time' in p),None)
        if candidate and status.get('state') not in {'complete','error','paused'} and self.fingerprints_ok():
            if datetime.fromisoformat(status['updated_at']).timestamp()>=candidate['create_time']-1:
                self.state.update(expected_pid=candidate['pid'],expected_create_time=candidate['create_time'],dead_since=None,stable_since=None)
                self.state.pop('launch_pending',None)
                return self.save('replacement_training_adopted',identity=candidate)
        self.state['dead_since']=self.state['dead_since'] or tick
        pending=self.state.get('launch_pending')
        if pending:
            return self.save('waiting_replacement_startup' if tick-pending['at']<600 else 'replacement_startup_requires_review',launch=pending)
        production=ROOT/'artifacts/experiment1_fusion_id_fast_v1';checkpoint=production/'last.pt'
        ram=psutil.virtual_memory().available;disk=psutil.disk_usage(str(ROOT)).free
        resources=ram>=self.policy['minimum_available_RAM_GiB']*2**30 and disk>=self.policy['minimum_disk_GiB']*2**30
        self.state['stable_since']=(self.state['stable_since'] or tick) if resources else None
        recent=[a for a in self.state['attempts'] if tick-a['at']<86400]
        context={'status':status,'training_alive':False,'other_training':bool(active),'test_seen':self.state['test_seen'],
            'stop_requested':(production/'STOP').exists() or (self.output/'DISABLE_AUTO_RESUME').exists(),
            'fingerprints_ok':self.fingerprints_ok(),'attempts_24h':len(recent),
            'seconds_since_attempt':tick-self.state['attempts'][-1]['at'] if self.state['attempts'] else float('inf'),
            'available_ram':ram,'free_disk':disk,'resource_stable_seconds':tick-self.state['stable_since'] if self.state['stable_since'] else 0,
            'dead_seconds':tick-self.state['dead_since'],'checkpoint_exists':checkpoint.exists(),
            'checkpoint_age':tick-checkpoint.stat().st_mtime if checkpoint.exists() else 0}
        decision=recovery_gate(context,self.policy)
        if decision!='eligible_for_checkpoint_validation':return self.save(decision,resources={'available_RAM_GiB':ram/2**30,'disk_GiB':disk/2**30})
        signature=digest(checkpoint)
        if any(a['checkpoint_sha256']==signature for a in self.state['attempts']):return self.save('unchanged_checkpoint_already_attempted')
        if self.state.get('invalid_checkpoint_sha256')==signature:return self.save('checkpoint_invalid_requires_review')
        try:
            gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,creationflags=subprocess.CREATE_NO_WINDOW)
            name,free=gpu.stdout.strip().splitlines()[0].rsplit(',',1)
            if gpu.returncode or name.strip()!=self.auth['GPU'] or float(free)<self.policy['minimum_free_GPU_MiB']:
                return self.save('waiting_GPU_resources')
        except (ValueError,IndexError,OSError,subprocess.TimeoutExpired):return self.save('waiting_GPU_query')
        version_file=ROOT/'.venv-gpu/Lib/site-packages/torch/version.py'
        tree=ast.parse(version_file.read_text());version=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='__version__' for t in n.targets))
        if version!=self.auth['torch']:return self.save('torch_environment_changed')
        attempt_dir=self.output/'recovery'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');attempt_dir.mkdir(parents=True)
        validation=attempt_dir/'checkpoint_validation.json';self.save('validating_checkpoint',checkpoint_sha256=signature)
        try:
            with (attempt_dir/'validator_stdout.log').open('wb') as out,(attempt_dir/'validator_stderr.log').open('wb') as err:
                result=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-X','utf8','-B',str(ROOT/'versions/rgcn_cached_fast_v1/scripts/validate_experiment1_resume.py'),
                    '--checkpoint',str(checkpoint),'--output',str(validation)],cwd=ROOT,stdout=out,stderr=err,timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
        except (OSError,subprocess.TimeoutExpired) as error:
            self.state['invalid_checkpoint_sha256']=signature
            return self.save('checkpoint_validator_requires_review',error=str(error))
        evidence=read(validation) if validation.exists() else {}
        if result.returncode or evidence.get('status')!='valid' or evidence.get('checkpoint_sha256')!=signature:
            if evidence.get('status')=='deferred':self.state['stable_since']=None;return self.save('waiting_validator_resources')
            self.state['invalid_checkpoint_sha256']=signature;return self.save('checkpoint_validation_failed',evidence=str(validation))
        # Recheck after validation; nothing may have changed while it ran.
        if self.training_processes() or not self.fingerprints_ok() or digest(checkpoint)!=signature:return self.save('prelaunch_state_changed')
        if psutil.virtual_memory().available<self.policy['minimum_available_RAM_GiB']*2**30:
            self.state['stable_since']=None;return self.save('waiting_RAM_after_validation')
        if (production/'STOP').exists() or (self.output/'DISABLE_AUTO_RESUME').exists():return self.save('user_stop_requested')
        # Check original runner lock, then let the unchanged runner acquire it.
        with (production/'training.lock').open('r+b') as lock:
            try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            except OSError:return self.save('training_lock_busy')
            msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
        attempt={'at':time.time(),'at_UTC':now(),'checkpoint_sha256':signature,'validation':str(validation),
                 'command':['.venv-gpu/Scripts/python.exe','-B','-u','versions/rgcn_cached_fast_v1/scripts/train_experiment1_fast.py','--config','config/experiment1_fusion_id_fast_v1.json','--resume']}
        self.state['attempts'].append(attempt);self.save('launch_intent_recorded')
        with (production/'training_stdout.log').open('ab') as stdout,(production/'training_stderr.log').open('ab') as stderr:
            process=subprocess.Popen([str(ROOT/'.venv-gpu/Scripts/python.exe'),'-B','-u','versions/rgcn_cached_fast_v1/scripts/train_experiment1_fast.py','--config','config/experiment1_fusion_id_fast_v1.json','--resume'],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,close_fds=True,creationflags=subprocess.CREATE_NO_WINDOW)
        attempt['launcher_pid']=process.pid;self.state['launch_pending']={'at':time.time(),'launcher_pid':process.pid,'checkpoint_sha256':signature}
        with (self.output/'recovery_events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(attempt)+'\n')
        return self.save('replacement_launched',launch=self.state['launch_pending'])
