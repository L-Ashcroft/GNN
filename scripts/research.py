"""Windows workspace entry point; route to pinned implementations, never bypass recipes."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / '.venv/Scripts/python.exe'
P1 = ROOT / 'worktrees/public_evidence_queue_v1/queue.json'
P3 = ROOT / 'worktrees/controlled_evidence_v1/suite.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def identity(pid, created):
    import psutil
    try:
        return abs(psutil.Process(pid).create_time() - created) < .01
    except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
        return False


def status():
    import psutil
    result = {'available_RAM_GiB': psutil.virtual_memory().available / 2**30, 'queues': {}}
    for name in ('public_evidence_queue_v1', 'controlled_evidence_v1'):
        path = ROOT / 'monitoring' / name / 'status.json'
        if not path.exists():
            continue
        s = read(path)
        pid = s.get('supervisor_pid', s.get('pid'))
        created = s.get('supervisor_create_time', s.get('create_time'))
        result['queues'][name] = {
            'state': s.get('state'), 'pid': pid,
            'identity_matches': identity(pid, created),
            'job': s.get('current_job', s.get('original_job')),
            'progress': s.get('worker_status', s.get('original_state', {})),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def restore_gpu_receipt():
    """Restore only historical metadata for CPU recipes, never an installed runtime."""
    import sqlite3
    name = '.venv-gpu/Lib/site-packages/torch/version.py'
    with sqlite3.connect(ROOT / 'monitoring/workspace_cleanup_v2/inventory.sqlite') as db:
        row = db.execute('SELECT json FROM documents WHERE name=?', ('runtime_receipts',)).fetchone()
    receipt = json.loads(row[0])[name]
    destination = ROOT / name
    if destination.exists():
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == receipt['sha256']
    else:
        if (ROOT / '.venv-gpu/Scripts/python.exe').exists() or list(destination.parent.glob('_C*.pyd')):
            raise SystemExit('An installed CUDA environment exists; reinstall it instead of patching its version metadata.')
        assert destination.resolve().is_relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = receipt['utf8_text'].encode('utf-8')
        assert hashlib.sha256(payload).hexdigest() == receipt['sha256']
        destination.write_bytes(payload)
    print('Original CUDA version receipt verified. CPU recipes may reference it; no GPU runtime is installed by this command.')


def jobs():
    result = {}
    for j in read(P1)['jobs']:
        cfgpath = ROOT / j['worktree'] / 'config.json'
        result[j['id']] = {'script': ROOT / j['worktree'] / 'train.py', 'config': cfgpath, 'args': []}
    for j in read(P3)['jobs']:
        result[j['id']] = {'script': ROOT / 'worktrees/controlled_evidence_v1/train.py',
                          'config': ROOT / j['config'], 'args': ['--config', str(ROOT / j['config'])]}
    return result


ALIASES = {
    'id': 'rgcn_memory_safe_id_control_v2',
    'no-film': 'rgcn_memory_safe_no_film_v2',
    'static-edge': 'rgcn_memory_safe_static_edges_v2',
    'merged-relation': 'rgcn_memory_safe_collapsed_relations_v2',
    'loto': 'rgcn_memory_safe_loto_v2',
    'single': 'controlled_single_rgcn_lb_v1',
    'multi': 'controlled_multi_rgcn_lb_v1',
    'mlp': 'controlled_single_mlp_lb_v1',
    'gcn': 'controlled_single_gcn_lb_v1',
}


def guarded_run(command, training=False):
    if training:
        import psutil
        # Do not race the serial queues, even during a resource cooldown.
        for name in ('public_evidence_queue_v1', 'controlled_evidence_v1'):
            p = ROOT / 'monitoring' / name / 'status.json'
            if p.exists():
                s = read(p)
                if identity(s.get('supervisor_pid', s.get('pid')),
                            s.get('supervisor_create_time', s.get('create_time'))):
                    raise SystemExit('The serial queue already owns these jobs. Use status; do not start a second trainer.')
        for proc in psutil.process_iter(['name', 'cmdline']):
            if proc.pid in (os.getpid(), os.getppid()):
                continue
            for arg in proc.info['cmdline'] or []:
                value = str(arg).replace('\\', '/').lower()
                if value.startswith(ROOT.as_posix().lower() + '/') and value.endswith('/train.py'):
                    raise SystemExit('Another training process is running; serial execution required.')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return subprocess.call([str(x) for x in command], cwd=ROOT, env=env)


def prepare_loto(holdout):
    """Share immutable source bytes via NTFS hard links; a new recipe is explicit."""
    base = ROOT / 'worktrees/rgcn_memory_safe_loto_v2'
    cfg = read(base / 'config.json')
    protocol = read(ROOT / cfg['protocol'])
    wanted = holdout.removeprefix('LOADBench__')
    options = [f for f in protocol['folds'] if any(t.removeprefix('LOADBench__') == wanted
               for t in f['test_topologies'])]
    if len(options) != 1:
        raise SystemExit('Holdout must match exactly one verified fold; no new split is generated.')
    fold = options[0]
    if fold['fold_id'] == cfg['fold_id']:
        return {'script': base / 'train.py', 'config': base / 'config.json', 'args': []}
    key = 'loto_' + hashlib.sha256(fold['fold_id'].encode()).hexdigest()[:12]
    directory = ROOT / 'worktrees' / key
    cfg['fold_id'] = fold['fold_id']
    cfg['output'] = 'artifacts/' + key
    if (directory / 'config.json').exists():
        if read(directory / 'config.json') != cfg:
            raise SystemExit('Existing fold recipe differs; refusing overwrite.')
    else:
        directory.mkdir(exist_ok=True)
        for p in base.glob('*.py'):
            destination = directory / p.name
            if not destination.exists():
                os.link(p, destination)
        (directory / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8')
    return {'script': directory / 'train.py', 'config': directory / 'config.json', 'args': []}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('status')
    sub.add_parser('list')
    sub.add_parser('restore-gpu-receipt')
    queue = sub.add_parser('queue'); queue.add_argument('--which', choices=['current', 'followup'], default='current')
    train = sub.add_parser('train')
    train.add_argument('experiment', choices=list(ALIASES))
    train.add_argument('--holdout', help='Exact topology in the existing verified LOTO protocol')
    train.add_argument('--prepare-only', action='store_true')
    train.add_argument('--resume', action='store_true')
    evaluation = sub.add_parser('evaluate'); evaluation.add_argument('--job', required=True)
    verify = sub.add_parser('verify'); verify.add_argument('--stage', choices=['data', 'models', 'metrics', 'pins', 'all'], default='all')
    a = p.parse_args()
    if a.command == 'status':
        status(); return 0
    if a.command == 'restore-gpu-receipt':
        restore_gpu_receipt(); return 0
    if a.command == 'list':
        print(json.dumps({name: {'id': ALIASES[name], 'config': str(jobs()[ALIASES[name]]['config'].relative_to(ROOT))}
                          for name in ALIASES}, indent=2)); return 0
    if a.command == 'queue':
        folder, file = ('public_evidence_queue_v1', 'supervise.py') if a.which == 'current' else ('controlled_evidence_v1', 'followup.py')
        control = ROOT / 'monitoring' / folder / 'status.json'
        if control.exists():
            s = read(control)
            if identity(s.get('supervisor_pid', s.get('pid')), s.get('supervisor_create_time', s.get('create_time'))):
                status(); return 0
        return guarded_run([PYTHON, '-X', 'utf8', '-B', '-u', ROOT / 'worktrees' / folder / file])
    if a.command == 'verify':
        return guarded_run([PYTHON, '-X', 'utf8', '-B', ROOT / 'scripts/verify_workspace.py', '--stage', a.stage])
    if a.command == 'evaluate':
        config = read(ROOT / 'config/falcon_reference_eval_v1.json')
        selected = [j for j in config['jobs'] if j['id'] == a.job]
        if not selected:
            raise SystemExit('Unknown frozen evaluation job.')
        j = selected[0]
        if j.get('pending') and not all((ROOT / q).exists() for q in j['proofs']):
            raise SystemExit('This test is not frozen; evaluation refused.')
        return guarded_run([PYTHON, '-X', 'utf8', '-B', ROOT / 'scripts/evaluate_falcon_reference.py', '--job', a.job, '--no-plots'])
    job = prepare_loto(a.holdout) if a.experiment == 'loto' and a.holdout else jobs()[ALIASES[a.experiment]]
    cfg = read(job['config']); output = ROOT / cfg['output']
    if (output / 'selection_frozen.json').exists() or ((output / 'status.json').exists() and read(output / 'status.json').get('state') == 'complete'):
        raise SystemExit('This run is frozen. Evaluate its saved predictions; further training requires an independent run configuration.')
    command = [PYTHON, '-X', 'utf8', '-B', '-u', job['script'], *job['args'], '--backend', 'cpu']
    if a.resume:
        if not (output / 'last.pt').exists():
            raise SystemExit('No recovery checkpoint; refusing --resume.')
        command.append('--resume')
    elif (output / 'recipe.json').exists():
        raise SystemExit('Existing recipe found. Use --resume for the same run; never overwrite it.')
    if a.prepare_only:
        print(subprocess.list2cmdline([str(x) for x in command])); return 0
    return guarded_run(command, training=True)


if __name__ == '__main__':
    raise SystemExit(main())
