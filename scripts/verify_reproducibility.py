"""Rerun the sampler/split/scaler stages using cached conversions, then compare artifacts."""
from pathlib import Path
import hashlib
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]


def hashes():
    paths = []
    for name in ['processed', 'strict']:
        base = ROOT / 'data' / name
        paths += [base / 'index.parquet', base / 'sampling_summary.json']
        paths += sorted((base / 'sampled').glob('*.parquet'))
        paths += [base / 'catalog.json']
        paths += sorted((base / 'graphs').glob('*.json'))
    for name in ['splits', 'strict_splits']:
        paths += sorted((ROOT / 'data' / name).rglob('*.json'))
        paths += sorted((ROOT / 'data' / name).glob('*.parquet'))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    before = hashes()
    subprocess.run([sys.executable, 'scripts/run_pipeline.py'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, 'scripts/run_pipeline.py', '--from-converted', 'data/processed', '--require-complete-inputs',
                    '--output', 'data/strict', '--splits', 'data/strict_splits'], cwd=ROOT, check=True)
    after = hashes()
    changed = [p for p in before if before[p] != after.get(p)]
    if changed:
        raise AssertionError('Reproducibility mismatch: ' + ', '.join(changed))
    import json
    result = {'status': 'passed', 'checked_files': len(before), 'sha256': after,
              'scope': 'Two repeated complete sampling/split/normalization runs using unchanged pinned inputs and conversion cache'}
    (ROOT / 'reports/reproducibility.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Reproducible SHA256 for', len(before), 'artifacts')


if __name__ == '__main__':
    main()
