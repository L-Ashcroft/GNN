"""Native Windows serial coordinator; resource wait preserves checkpoints."""
import os
import sys
import time
import subprocess
import ctypes
import msvcrt
import psutil
from common import ROOT, HERE, OUT, CONTROL, read, sha, atomic, freeze, now


def main():
    CONTROL.mkdir(parents=True, exist_ok=True)
    lock = (CONTROL / 'watch.lock').open('a+b'); lock.seek(0)
    if not lock.read(1): lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    queue = [{'id': 'rgcn_memory_safe_id_v2', 'worktree': 'worktrees/rgcn_memory_safe_id_v2', 'output': 'artifacts/rgcn_memory_safe_id_v2'}]
    queue += read(OUT / 'remaining_queue.json')['jobs']
    paths = list(HERE.glob('*.py')) + list((ROOT / 'worktrees/rgcn_metric_v4').glob('*.py'))
    for job in queue: paths += list((ROOT / job['worktree']).glob('*.py')) + [ROOT / job['worktree'] / 'config.json']
    paths += list((ROOT / 'versions/rgcn_calibrated_head_v1/src').rglob('*.py'))
    pins = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(paths))}
    freeze(CONTROL / 'policy.json', {'sources': pins, 'queue': queue, 'CPU_start_GiB': 1.4, 'CUDA_start_GiB': 4.,
           'resource_exit': 'Pause/wait; after3 resource exits require stable2.2GiB before another attempt, no endless crash loop',
           'max_retries_per_job': 8, 'old_supervisor_not_restarted': True})
    state = {'supervisor_pid': os.getpid(), 'supervisor_create_time': psutil.Process().create_time(), 'run_id': 'rgcn_accuracy_v3'}
    def update(**v):
        state.update(v, updated_at=now(), available_RAM_GiB=psutil.virtual_memory().available/2**30)
        atomic(CONTROL / 'status.json', state); atomic(ROOT / 'monitoring/current_training.json', state)
    def stop(): return (CONTROL / 'STOP').exists() or any(sha(ROOT / p) != h for p, h in pins.items())
    def other_workers(script):
        result = []
        for p in psutil.process_iter(['name', 'cmdline', 'create_time']):
            if p.pid in {os.getpid(), os.getppid()}: continue
            if (p.info['name'] or '').lower() in ['python.exe', 'pythonw.exe'] and any(str(a).replace('\\', '/').endswith(script) for a in p.info['cmdline'] or []):
                result.append((p.pid, p.info['create_time']))
        return result
    def run(name, script, output, terminal, gate, train=False):
        for attempt in range(8):
            status_path = output / ('status.json' if train else 'fixed_test/status.json')
            if status_path.exists() and read(status_path).get('state') in terminal: return
            while True:
                if stop(): raise RuntimeError('Explicit stop or source fingerprint changed')
                existing = other_workers(script)
                needed = max(gate, 2.2 if attempt >= 3 else gate)
                if not existing and psutil.virtual_memory().available >= needed*2**30: break
                update(state='waiting_existing_worker' if existing else 'waiting_resources', current_job=name,
                       existing_processes=existing, training_active=False, worker_active=bool(existing), required_RAM_GiB=needed)
                time.sleep(15)
                if status_path.exists() and read(status_path).get('state') in terminal: return
            if train and (output / 'STOP').exists(): raise RuntimeError('Preserved explicit task STOP: '+str(output))
            output.mkdir(parents=True, exist_ok=True)
            command = [str(ROOT / '.venv/Scripts/python.exe'), '-X', 'utf8', '-B', '-u', str(ROOT / script)]
            if train:
                command += ['--backend', 'cpu']
                if (output / 'recipe.json').exists(): command.append('--resume')
            with (CONTROL / (name+'_stdout.log')).open('ab') as out, (CONTROL / (name+'_stderr.log')).open('ab') as err:
                p = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
            update(state='starting', current_job=name, worker_pid=p.pid, worker_create_time=psutil.Process(p.pid).create_time(), command=command, worker_active=True)
            while p.poll() is None:
                if stop():
                    (output / 'STOP').write_text('Coordinator/source stop; preserve checkpoint', encoding='utf-8')
                    update(state='stop_requested', training_active=False)
                latest = read(status_path) if status_path.exists() else {}
                update(state='training' if train else 'evaluating', current_job=name, worker_status=latest,
                       training_active=bool(train and latest.get('state')=='training'), worker_active=True)
                time.sleep(10)
            latest = read(status_path) if status_path.exists() else {}
            if p.returncode == 0 and latest.get('state') in terminal:
                update(state='job_complete', current_job=name, training_active=False, worker_active=False); return
            if p.returncode not in [75, 86] and latest.get('state')!='paused': raise RuntimeError(f'{name}: exit{p.returncode}; inspect stderr')
            update(state='waiting_resources', current_job=name, last_exit=p.returncode, attempt=attempt+1, training_active=False, worker_active=False)
            time.sleep(30)
        raise RuntimeError('Resource retry limit; preserved checkpoint')
    awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        run('metric_fixed_test', 'worktrees/rgcn_metric_v4/evaluate.py', ROOT / 'artifacts/rgcn_metric_v4', {'complete'}, 1.2)
        for job in queue:
            run(job['id'], job['worktree']+'/train.py', ROOT/job['output'], {'complete'}, 1.4, True)
        update(state='suite_complete', current_job=None, training_active=False, worker_active=False)
    except BaseException as e:
        update(state='requires_review', training_active=False, worker_active=False, error_type=type(e).__name__, error=str(e)); raise
    finally:
        if awake: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__=='__main__': main()
