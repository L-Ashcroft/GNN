"""Prove graph migration preserves sampled rows, labels, and split/scaler files."""
import argparse
import hashlib
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import dump, read


def snapshot():
    paths = []
    for name in ('processed', 'strict'):
        base = ROOT / 'data' / name
        paths += [base / 'index.parquet', *sorted((base / 'sampled').glob('*.parquet'))]
    for name in ('splits', 'strict_splits'):
        base = ROOT / 'data' / name
        paths += sorted(base.glob('*.parquet')) + sorted(base.rglob('*.json'))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--snapshot', action='store_true'); args = parser.parse_args()
    baseline = ROOT / 'reports/graph_migration_baseline.json'
    if args.snapshot:
        if baseline.exists(): raise FileExistsError('Refusing to replace pre-migration evidence')
        dump(baseline, snapshot()); print('Pre-migration snapshot saved'); return
    before, after = read(baseline), snapshot()
    assert before == after, 'Sample rows, labels, memberships or training statistics changed'
    result = []
    for spec in read(ROOT / 'data/processed/catalog.json'):
        graph = read(ROOT / spec['graph'])
        assert graph['representation'] == 'net_device_multigraph_v2'
        result.append({'topology_id': spec['topology_id'], 'num_nodes': graph['num_nodes'],
                       'num_devices': graph['num_devices'], 'num_edges': graph['num_edges'],
                       'mos_count': sum(d['model'] in {'nmos', 'pmos'} for d in graph['devices'])})
    dump(ROOT / 'reports/graph_migration.json', {'status': 'passed', 'representation': 'net_device_multigraph_v2',
         'unchanged_sample_split_scaler_files': len(before), 'graphs': result,
         'port_policy': 'FALCON style, extended with DB/GB/SB to retain bulk', 'devices_are_nodes': False})
    print('Graph migration checked;', len(before), 'sample/split/scaler files unchanged')


if __name__ == '__main__': main()
