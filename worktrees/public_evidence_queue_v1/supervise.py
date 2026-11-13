"""One pinned Windows CPU worker; resource cooldown, real process identities."""
from pathlib import Path
from datetime import datetime, timezone
import os
import sys
import json
import hashlib
import subprocess
import time
import ctypes
import msvcrt
import psutil

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONTROL = ROOT / 'monitoring/public_evidence_queue_v1'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def atomic(path, obj):
    temp = path.with_suffix('.queue.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for _ in range(50):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            time.sleep(.1)
    raise RuntimeError('Status replacement timed out: ' + str(path))

def live_identity(status, script):
    try:
        process = psutil.Process(status['pid'])
        return abs(process.create_time() - status['create_time']) < .01 and any(
            str(a).replace('\\', '/').lower() == script.as_posix().lower() for a in process.cmdline())
    except (KeyError, psutil.NoSuchProcess, psutil.AccessDenied):
        return False

def other_workers():
    result = []
    for process in psutil.process_iter(['name', 'cmdline', 'create_time']):
        if process.pid in {os.getpid(), os.getppid()}:
            continue
        args = [str(a).replace('\\', '/').lower() for a in process.info['cmdline'] or []]
        prefix = (ROOT / 'worktrees').as_posix().lower() + '/'
        if (process.info['name'] or '').lower() in {'python.exe', 'pythonw.exe'} and any(
                a.startswith(prefix) and a.endswith(('/train.py', '/watch.py', '/supervise.py', '/supervisor.py')) for a in args):
            result.append({'pid': process.pid, 'create_time': process.info['create_time']})
    return result

def main():
    CONTROL.mkdir(parents=True, exist_ok=True)
    lock = (CONTROL / 'supervisor.lock').open('a+b')
    lock.seek(0)
    if not lock.read(1):
        lock.write(b'0')
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    jobs = read(HERE / 'queue.json')['jobs']
    files = list(HERE.glob('*.py')) + [HERE / 'queue.json']
    for job in jobs:
        files += list((ROOT / job['worktree']).glob('*.py')) + [ROOT / job['worktree'] / 'config.json']
    files += list((ROOT / 'versions/rgcn_calibrated_head_v1/src').rglob('*.py'))
    pins = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(files))}
    policy = {'source_sha256': pins, 'queue': jobs, 'serial_only': True, 'CPU_initial_gate_GiB': 1.4,
              'stable_memory_seconds': 20, 'CUDA_used': False, 'CUDA_gate_not_relaxed_GiB': 4.,
              'resource_recovery': 'Checkpoint then cooldown; after two exits wait for 1.8GiB, after four wait for 2.2GiB',
              'no_cross_recipe_resume': True, 'warm_ID_candidate_excluded_from_cold_controls': True}
    if (CONTROL / 'policy.json').exists():
        assert read(CONTROL / 'policy.json') == policy, 'Pinned queue/source changed'
    else:
        atomic(CONTROL / 'policy.json', policy)
    state = {'supervisor_pid': os.getpid(), 'supervisor_create_time': psutil.Process().create_time(),
             'run_id': 'public_evidence_queue_v1', 'training_active': False, 'worker_active': False}
    def update(**values):
        state.update(values, updated_at=now(), available_RAM_GiB=psutil.virtual_memory().available / 2**30)
        atomic(CONTROL / 'status.json', state)
        atomic(ROOT / 'monitoring/current_training.json', state)
    def stopped():
        return (CONTROL / 'STOP').exists() or any(sha(ROOT / p) != h for p, h in pins.items())
    def event(code, **values):
        with (CONTROL / 'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'at': now(), 'event': code, **values}, ensure_ascii=False) + '\n')
    def wait_ready(gate, job):
        stable = None
        while True:
            if stopped():
                raise RuntimeError('Explicit stop or pinned source change')
            others = other_workers()
            ready = not others and psutil.virtual_memory().available >= gate * 2**30
            stable = (stable or time.monotonic()) if ready else None
            update(state='waiting_existing_worker' if others else 'waiting_resources', current_job=job,
                   other_workers=others, training_active=False, worker_active=False, required_RAM_GiB=gate)
            if stable and time.monotonic() - stable >= 20:
                return
            time.sleep(5)
    awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    child = None
    try:
        assert read(ROOT / 'artifacts/public_evidence_queue_v1/preflight.json')['passed']
        assert read(ROOT / 'artifacts/public_evidence_queue_v1/preflight.json')['queue_sha256'] == sha(HERE / 'queue.json')
        for job in jobs:
            output = ROOT / job['output']
            script = ROOT / job['worktree'] / 'train.py'
            failures = 0
            while True:
                latest = read(output / 'status.json') if (output / 'status.json').exists() else {}
                if latest.get('state') == 'complete':
                    assert (output / 'selection_frozen.json').exists() and (output / 'heldout_test_metrics.json').exists()
                    event('job_complete', job=job['id'], summary=job['output'] + '/final_summary.json')
                    break
                if (output / 'STOP').exists():
                    raise RuntimeError('Explicit task STOP retained: ' + str(output))
                gate = 2.2 if failures >= 4 else 1.8 if failures >= 2 else 1.4
                wait_ready(gate, job['id'])
                cmd = [str(ROOT / '.venv/Scripts/python.exe'), '-X', 'utf8', '-B', '-u', str(script), '--backend', 'cpu']
                if (output / 'recipe.json').exists():
                    cmd.append('--resume')
                env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
                with (CONTROL / (job['id'] + '_stdout.log')).open('ab') as out, (CONTROL / (job['id'] + '_stderr.log')).open('ab') as err:
                    child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                             creationflags=subprocess.CREATE_NO_WINDOW)
                event('launch', job=job['id'], command=cmd, launcher_pid=child.pid)
                while child.poll() is None:
                    if stopped():
                        (output / 'STOP').write_text('Evidence queue stop or fingerprint change; checkpoint at batch boundary', encoding='utf-8')
                    latest = read(output / 'status.json') if (output / 'status.json').exists() else {}
                    identity = live_identity(latest, script)
                    update(state='working', current_job=job['id'], command=cmd, launcher_pid=child.pid,
                           worker_pid=latest.get('pid') if identity else None,
                           worker_create_time=latest.get('create_time') if identity else None,
                           worker_identity_matches=identity, worker_active=True,
                           training_active=identity and latest.get('state') == 'training', worker_status=latest,
                           runtime_status=job['output'] + '/status.json')
                    time.sleep(10)
                latest = read(output / 'status.json') if (output / 'status.json').exists() else {}
                update(training_active=False, worker_active=False, last_exit=child.returncode, worker_status=latest)
                if stopped():
                    raise RuntimeError('Stopped safely; checkpoint retained')
                if child.returncode == 0 and latest.get('state') == 'complete':
                    continue
                error = (latest.get('error_type', '') + ' ' + latest.get('error', '')).lower()
                resource_exit = child.returncode in {75, 86} or latest.get('state') == 'paused' or any(
                    word in error for word in ['memoryerror', 'out of memory', 'paging file', '页面文件'])
                if not resource_exit:
                    raise RuntimeError(f"{job['id']} exited {child.returncode}; see dedicated stderr")
                failures += 1
                event('resource_pause', job=job['id'], failures=failures, exit=child.returncode)
                for _ in range(12):
                    if stopped():
                        raise RuntimeError('Stopped during resource cooldown')
                    update(state='resource_cooldown', current_job=job['id'], resource_exits=failures,
                           training_active=False, worker_active=False)
                    time.sleep(10)
        update(state='suite_complete', current_job=None, training_active=False, worker_active=False)
    except BaseException as error:
        update(state='requires_review', training_active=False, worker_active=False,
               error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()

if __name__ == '__main__':
    main()
