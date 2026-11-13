"""One serial Windows worker, pinned code, bounded retries, honest live status."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'
import sys
import time
import subprocess
import msvcrt
import ctypes
import psutil
from core import ROOT, HERE, OUT, CONTROL, read, sha, atomic, freeze, now


def main():
    CONTROL.mkdir(parents=True, exist_ok=True)
    lock = (CONTROL / 'supervisor.lock').open('a+b'); lock.seek(0)
    if not lock.read(1):
        lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    jobs = read(OUT / 'public_queue.json')['jobs']
    paths = list(HERE.glob('*.py'))
    for job in jobs:
        paths += list((ROOT / job['worktree']).glob('*.py')) + [ROOT / job['worktree'] / 'config.json']
    repeats = read(OUT / 'personal_repeats_queue.json')
    paths += list((ROOT / repeats['worktree']).glob('*.py')) + [ROOT / repeats['worktree'] / 'suite.json']
    paths += list((ROOT / 'versions/rgcn_calibrated_head_v1/src').rglob('*.py'))
    policy = {'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(paths))},
              'public_queue_sha256': sha(OUT / 'public_queue.json'), 'max_resource_retries': 3,
              'automatic_model_change': 'None within a run; any new recipe requires a new output directory',
              'GPU_gate_GiB': 4., 'serial_only': True}
    freeze(CONTROL / 'policy.json', policy)
    python = ROOT / '.venv/Scripts/python.exe'
    state = {'supervisor_pid': os.getpid(), 'supervisor_create_time': psutil.Process().create_time(), 'state': 'initializing'}
    def update(**kwargs):
        state.update(kwargs, updated_at=now(), available_RAM_GiB=psutil.virtual_memory().available / 2**30)
        atomic(CONTROL / 'status.json', state)
        atomic(ROOT / 'monitoring/current_training.json', {'run_id': 'research_repair_v1', **state,
               'monitor_status': (CONTROL / 'status.json').relative_to(ROOT).as_posix()})
    def report():
        subprocess.run([str(python), '-X', 'utf8', '-B', str(HERE / 'report.py')], cwd=ROOT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    def check_pins():
        return all(sha(ROOT / p) == h for p, h in policy['source_sha256'].items())
    def stopped():
        return (CONTROL / 'STOP').exists() or not check_pins()
    def run(name, command, status_path, terminal, gate_gib=1.6, public_output=None):
        for attempt in range(4):
            if status_path.exists() and read(status_path).get('state') in terminal:
                return
            while psutil.virtual_memory().available < gate_gib * 2**30:
                update(state='waiting_resources', current_job=name, training_active=False)
                if stopped():
                    raise RuntimeError('Explicit stop or pinned source changed')
                time.sleep(15)
            if stopped():
                raise RuntimeError('Explicit stop or pinned source changed')
            cmd = list(command)
            if public_output and (public_output / 'recipe.json').exists():
                cmd.append('--resume')
            with (CONTROL / (name + '_stdout.log')).open('ab') as out, (CONTROL / (name + '_stderr.log')).open('ab') as err:
                child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                         creationflags=subprocess.CREATE_NO_WINDOW)
            update(state='worker_starting', current_job=name, worker_pid=child.pid,
                   worker_create_time=psutil.Process(child.pid).create_time(), command=cmd,
                   worker_active=True, training_active=False, attempt=attempt)
            while child.poll() is None:
                if stopped():
                    if public_output:
                        (public_output / 'STOP').write_text('Repair supervisor stop/source change; save checkpoint', encoding='utf-8')
                    update(state='stop_requested', current_job=name)
                    # Never launch a replacement while an earlier child is alive.
                    child.wait(); raise RuntimeError('Stopped safely; no replacement launched')
                latest = read(status_path) if status_path.exists() else {}
                update(state='working', current_job=name, worker_status=latest, worker_active=True,
                       training_active=bool(public_output and latest.get('state') == 'training'))
                time.sleep(10)
            latest = read(status_path) if status_path.exists() else {}
            if child.returncode == 0 and latest.get('state') in terminal:
                update(state='job_complete', current_job=name, training_active=False, worker_active=False, worker_status=latest)
                report()
                return
            transient = child.returncode in {75, 86} or latest.get('state') == 'paused'
            if not transient or attempt == 3:
                raise RuntimeError(f'{name} failed: exit={child.returncode}; inspect dedicated stderr log')
            update(state='resource_retry', current_job=name, exit_code=child.returncode, training_active=False)
            time.sleep(30)
        raise RuntimeError('Retry limit')
    awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        # Short fits precede long graph runs. No concurrent CPU/GPU pressure.
        run('fewshot_calibration', [str(python), '-X', 'utf8', '-B', '-u', str(HERE / 'fewshot_select.py')],
            OUT / 'fewshot_calibration/status.json', {'validation_complete'}, 1.1)
        run('calibration_guard', [str(python), '-X', 'utf8', '-B', '-u', str(HERE / 'qualify_candidates.py')],
            OUT / 'fewshot_calibration/promotion_guard.json', {'complete'}, 1.1)
        run('personal_test', [str(python), '-X', 'utf8', '-B', '-u', str(HERE / 'personal_test.py')],
            OUT / 'personal_test/status.json', {'complete'}, 1.6)
        for job in repeats['jobs']:
            folder = ROOT / repeats['worktree']; output = ROOT / repeats['output'] / job['id']
            output.mkdir(parents=True, exist_ok=True)
            run(job['id'], [str(python), '-X', 'utf8', '-B', '-u', str(folder / 'train.py'), '--job', job['id']],
                output / 'status.json', {'complete'}, 1.6, output)
        for job in jobs:
            folder = ROOT / job['worktree']; output = ROOT / job['output']; output.mkdir(parents=True, exist_ok=True)
            cfg = job['config']
            if not (output / 'preflight.json').exists():
                run(job['id'] + '_check', [str(python), '-X', 'utf8', '-B', '-u', str(HERE / 'public_check.py'), job['id']],
                    output / 'preflight.json', {'complete'}, 1.6)
            backend = 'cpu'; executable = python
            if psutil.virtual_memory().available >= 4.2 * 2**30:
                # No other worker is alive here. Failed numerical acceptance
                # cannot silently enable CUDA or alter tolerance.
                with (CONTROL / (job['id'] + '_cuda.log')).open('ab') as log:
                    q = subprocess.run([str(ROOT / '.venv-gpu/Scripts/python.exe'), '-X', 'utf8', '-B', '-u', str(folder / 'qualify.py')],
                                       cwd=ROOT, stdout=log, stderr=log, stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
                if q.returncode == 0 and (output / 'cuda_acceptance.json').exists() and read(output / 'cuda_acceptance.json')['passed']:
                    backend = 'cuda'; executable = ROOT / '.venv-gpu/Scripts/python.exe'
            run(job['id'], [str(executable), '-X', 'utf8', '-B', '-u', str(folder / 'train.py'), '--backend', backend],
                output / 'status.json', {'complete'}, 4. if backend == 'cuda' else 1.6, output)
        update(state='suite_complete', current_job=None, training_active=False)
    except BaseException as e:
        update(state='requires_review', training_active=False, error_type=type(e).__name__, error=str(e))
        raise
    finally:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == '__main__':
    main()
