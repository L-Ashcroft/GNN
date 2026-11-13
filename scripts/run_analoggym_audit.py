"""Repeatable supplemental audit; fail fast, no simulations or GNN training."""
import hashlib
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import dump


def snapshot():
    paths = sorted((ROOT / 'data/audit/analoggym_replay').glob('*.parquet'))
    paths += [ROOT / 'reports/analoggym_replay_audit.json']
    result = {}
    for p in paths:
        if p.exists():
            with p.open('rb') as f: result[p.relative_to(ROOT).as_posix()] = hashlib.file_digest(f, 'sha256').hexdigest()
    return result


def main():
    before = snapshot()
    steps = [
        ['scripts/acquire_sources.py', '--lock', 'config/analoggym_audit_sources.lock.json', '--check-only'],
        ['-m', 'unittest', 'discover', '-s', 'tests', '-v'],
        ['scripts/audit_analoggym_replay.py'],
        ['scripts/check_analoggym_audit.py'], ['scripts/build_analoggym_reports.py'],
        ['scripts/build_reports.py'],
        ['scripts/acquire_sources.py', '--lock', 'config/analoggym_audit_sources.lock.json', '--check-only'],
    ]
    for args in steps:
        subprocess.run([sys.executable, *args], cwd=ROOT, check=True)
    after = snapshot()
    if before:
        assert all(after.get(k) == v for k, v in before.items()), 'Existing audit outputs changed; inspect before certifying reproducibility'
    dump(ROOT / 'reports/analoggym_audit_reproducibility.json', {
        'status': 'identical' if before else 'baseline_created_rerun_to_verify',
        'checked_files': len(before), 'output_files': len(after), 'sha256': after})
    print('Supplemental audit complete:', len(after), 'deterministic artifacts')


if __name__ == '__main__': main()
