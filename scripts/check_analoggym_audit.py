"""Check every supplemental output row and prevent accidental training admission."""
import math
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import read, dump
from src.load_analoggym import iter_replay_candidates


def main():
    report = read(ROOT / 'reports/analoggym_replay_audit.json')
    ids = set()
    rows = list(iter_replay_candidates(allow_quarantined=True))
    assert len(rows) == report['written_transitions'] == 20
    for row in rows:
        assert row['sample_id'] not in ids and not row['supervised_ready']
        ids.add(row['sample_id'])
        names, mask = list(row['raw_metric_names']), list(row['raw_presence_mask'])
        assert len(names) == len(mask) == len(row['performance_raw'])
        assert set(names) == set(row['performance_raw']) and all(mask)
        assert row['quality_flags'].size > 0
        assert all(math.isfinite(v) for v in row['performance_raw'].values())
        assert all(math.isfinite(v) for v in row['candidate_parameters'].values())
    main_ids = pd.read_parquet(ROOT / 'data/strict/index.parquet', columns=['sample_id']).sample_id
    assert len(main_ids) == 500000 and not set(main_ids) & ids
    for p in (ROOT / 'data/strict_splits').glob('*.parquet'):
        split_ids = pd.read_parquet(p, columns=['sample_id']).sample_id
        assert not set(split_ids) & ids
    # Loader explicitly refuses default admission of quarantined replay data.
    try:
        next(iter_replay_candidates())
    except ValueError:
        pass
    else:
        raise AssertionError('Quarantine gate did not reject default loading')
    dump(ROOT / 'reports/analoggym_audit_validation.json', {
        'status': 'passed',
        'replay_transitions_checked': len(rows), 'new_supervised_samples': 0,
        'raw_values_finite': True, 'mask_only_missingness': True, 'unique_ids': True,
        'main_dataset_and_strict_splits_have_no_audit_ids': True,
        'scope': 'data integrity and quarantine, not historical simulation convergence'})
    print('20 official replay transitions: export/provenance/mask/quarantine checks passed')


if __name__ == '__main__': main()
