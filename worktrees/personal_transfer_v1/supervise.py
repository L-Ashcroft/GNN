"""Bounded-memory native Windows queue, pinned recipes and explicit recovery."""
from pathlib import Path
import ctypes
import msvcrt
import os
import subprocess
import time
import psutil
from common import ROOT, HERE, CACHE, OUTPUT, PRETRAINED, atomic, read, sha, pins, now

CONTROL = ROOT / 'monitoring/experiment34_personal_v1'


def identity(status, job):
    try:
        process = psutil.Process(status['pid'])
        args = process.cmdline()
        return abs(process.create_time() - status['create_time']) < .001 and job in args and any(
            str(a).replace('\\', '/').endswith('personal_transfer_v1/train.py') for a in args)
    except (KeyError, psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def snapshot(suite):
    rows = []
    for job in suite['jobs']:
        folder = OUTPUT / job['id']
        state = read(folder / 'status.json') if (folder / 'status.json').exists() else {'state': 'queued'}
        rows.append({'job': job, 'status': state})
    atomic(OUTPUT / 'summary.json', {'updated_at': now(), 'jobs': rows,
           'completed': sum(r['status']['state'] == 'complete' for r in rows), 'total': len(rows),
           'zero_shot': 'No verified comparable source/target target contract; no fabricated zero-shot metrics',
           'target_validation_labels_additional': 2000, 'test_labels_for_selection': 0})
    lines = ['# 实验三与实验四进度', '', '自动更新：' + now(), '',
             '实验三已完成监督兼容准入检查：当前可验证共享的指标为0；因此尚不能报告有效的零样本预测精度。',
             '实验四使用独立个人2万条记录，不改变公开50万条、原模型、既有测试结果或论文。', '',
             '预算为每拓扑k条（十个拓扑合计10k）；另使用2,000条验证标签，总监督预算10k+2,000。',
             '各预算分别比较公开预训练初始化与随机初始化，架构、训练步预算和优化设置相同；另训练全量目标域参考模型。',
             '目前仅seed42；全部配置在第一次测试前冻结，不宣称多随机种子稳健性。', '',
             '|任务|初始化|每拓扑训练数|状态|验证hit@5%|测试hit@5%|',
             '|---|---|---:|---|---:|---:|']
    def fmt(value):
        return '—' if value is None else f'{100*value:.2f}%'
    for row in rows:
        job, state = row['job'], row['status']
        lines.append('|{}|{}|{}|{}|{}|{}|'.format(job['id'], job['initialization'],
            job['budget_per_topology'], state['state'], fmt(state.get('best_validation_hit5')), fmt(state.get('test_hit5'))))
    lines += ['', '状态入口：`monitoring/experiment34_personal_v1/status.json`。',
              '各任务历史、验证指标、冻结选择、独立测试及预测文件保存在 `artifacts/experiment4_personal_transfer_v1/<任务>/`。',
              'MAE/RMSE按各指标原单位报告；dB类指标采用绝对误差，近零分母不计相对误差命中率。',
              '全量目标域参考模型并非保证优于迁移模型的理论上界；没有有效零样本下界时不计算适配效率η。',
              '调度只自动恢复同recipe的资源暂停；数值异常、源码/数据变化及非资源错误停队待排查。']
    (ROOT / 'reports/experiment34_progress.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    CONTROL.mkdir(parents=True, exist_ok=True); OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (CONTROL / 'supervisor.lock').open('a+b')
    lock.seek(0)
    if not lock.read(1):
        lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    suite = read(HERE / 'suite.json')
    assert read(HERE / 'checks.json')['passed'], 'Real-batch qualification required'
    source_pins = pins()
    policy = {'source_sha256': source_pins, 'suite_sha256': sha(HERE / 'suite.json'),
              'cache_manifest_sha256': sha(CACHE / 'manifest.json'), 'source_checkpoint_sha256': sha(PRETRAINED),
              'qualification_sha256': sha(HERE / 'checks.json'), 'maximum_concurrent_trainers': 1,
              'CPU_threads': 2, 'CUDA_used': False, 'start_RAM_GiB': 1.6,
              'safe_pause_RAM_MiB': 640, 'worker_hard_admission_RAM_MiB': 512,
              'target_validation_labels_additional_to_k': 2000, 'test_used_for_model_selection': False}
    if (CONTROL / 'policy.json').exists():
        assert read(CONTROL / 'policy.json') == policy, 'Pinned suite/implementation changed'
    else:
        atomic(CONTROL / 'policy.json', policy)
    state = {'state': 'initializing', 'pid': os.getpid(), 'create_time': psutil.Process().create_time(),
             'experiment3': 'not_admissible_current_contracts', 'experiment4': 'qualified_queue'}
    def update(**values):
        state.update(values, updated_at=now(), available_RAM_GiB=psutil.virtual_memory().available / 2**30)
        atomic(CONTROL / 'status.json', state)
    awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    current = None
    try:
        for job in suite['jobs']:
            folder = OUTPUT / job['id']; folder.mkdir(parents=True, exist_ok=True)
            retries = 0; child = None; attempted = False
            while True:
                s = read(folder / 'status.json') if (folder / 'status.json').exists() else {}
                alive = identity(s, job['id'])
                changed = sha(HERE / 'suite.json') != policy['suite_sha256'] or any(sha(ROOT / p) != h for p, h in source_pins.items())
                if (CONTROL / 'STOP').exists() or changed:
                    if alive:
                        (folder / 'STOP').write_text('Suite stop or pinned source change', encoding='utf-8')
                    update(state='stopped', reason='Explicit STOP or pinned input changed', current_job=job['id'])
                    return
                if s.get('state') == 'complete':
                    assert (folder / 'selection_frozen.json').exists() and (folder / 'test_metrics.json').exists()
                    snapshot(suite)
                    break
                if alive:
                    if psutil.virtual_memory().available < 640 * 2**20 and not (folder / 'STOP').exists():
                        (folder / 'STOP').write_text('automatic_resource_pause', encoding='utf-8')
                    update(state='training_active', current_job=job['id'], worker_identity_matches=True, training=s)
                    atomic(ROOT / 'monitoring/current_training.json', {'run_id': 'experiment4_personal_transfer_v1',
                           'state': s['state'], 'updated_at': now(), 'runtime_status': (folder / 'status.json').relative_to(ROOT).as_posix(),
                           'monitor_status': 'monitoring/experiment34_personal_v1/status.json', 'private_AnalogGym_used': True,
                           'public_500k_retrained': False, 'device': 'cpu', 'epoch': s.get('epoch'), 'batch': s.get('batch')})
                    snapshot(suite); time.sleep(10); continue
                if child is not None and child.poll() is None:
                    update(state='worker_initializing', current_job=job['id']); time.sleep(5); continue
                if attempted:
                    code = child.poll() if child is not None else None
                    if s.get('state') != 'paused' and code not in [75, 86]:
                        update(state='error_requires_review', current_job=job['id'], exit_code=code, training=s)
                        snapshot(suite); return
                    retries += 1
                    if retries > 3:
                        update(state='resource_retry_limit', current_job=job['id'], training=s); return
                    attempted = False; child = None
                if (folder / 'STOP').exists():
                    if (folder / 'STOP').read_text(encoding='utf-8') == 'automatic_resource_pause':
                        if psutil.virtual_memory().available >= 1.8 * 2**30:
                            (folder / 'STOP').unlink()
                    else:
                        update(state='explicit_job_stop_preserved', current_job=job['id']); return
                if psutil.virtual_memory().available < 1.6 * 2**30 or (folder / 'STOP').exists():
                    update(state='waiting_memory', current_job=job['id']); time.sleep(10); continue
                command = [str(ROOT / '.venv/Scripts/python.exe'), '-X', 'utf8', '-B', '-u', str(HERE / 'train.py'), '--job', job['id']]
                if (folder / 'recipe.json').exists():
                    assert (folder / 'last.pt').exists() or s.get('state') == 'paused', 'Unexpected interrupted initialization'
                    command.append('--resume')
                with (folder / 'stdout.log').open('ab') as out, (folder / 'stderr.log').open('ab') as err:
                    child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                             creationflags=subprocess.CREATE_NO_WINDOW)
                current = child; attempted = True
                update(state='worker_launched', current_job=job['id'], launcher_pid=child.pid, retries=retries)
                time.sleep(5)
        snapshot(suite)
        update(state='suite_complete', completed_jobs=len(suite['jobs']), training=None)
    except BaseException as exc:
        update(state='supervisor_error', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == '__main__':
    main()
