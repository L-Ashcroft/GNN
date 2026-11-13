import numpy as np
import pandas as pd
from .common import dump, read


def grouped_roles(index, seed=42):
    assignments = {}
    for _, table in index.groupby('topology_group', sort=True):
        groups = sorted(table.design_group.unique())
        rng = np.random.default_rng(seed)
        rng.shuffle(groups)
        nt = max(1, int(.8 * len(groups)))
        nv = max(1, int(.1 * len(groups))) if len(groups) >= 3 else 0
        for i, group in enumerate(groups):
            assignments[group] = 'train' if i < nt else 'validation' if i < nt + nv else 'test'
    return index.design_group.map(assignments)


def make_splits(output, split_dir, seed=42):
    split_dir.mkdir(parents=True, exist_ok=True)
    ix = pd.read_parquet(output / 'index.parquet')
    specs = read(output / 'catalog.json')
    manifest = {}
    base = ix[['sample_id', 'source_dataset', 'topology_id', 'topology_group', 'design_group']].copy()
    base['role'] = grouped_roles(ix, seed)
    base.to_parquet(split_dir / 'id.parquet', index=False, compression='zstd')
    manifest['id'] = {'status': 'ready', 'file': 'id.parquet', 'seed': seed,
                       'evaluation_status': 'within_source_topology_semantic_groups',
                       'rule': '80/10/10 unique design groups per conservative topology group; group counts rounded, not exact row fractions'}
    held = []
    families = set()
    # Three independent full held-out core groups, preferring distinct families.
    for s in specs:
        if not s['input_complete']:
            continue
        if s['circuit_family'] not in families and s['topology_group'] not in held:
            held.append(s['topology_group']); families.add(s['circuit_family'])
        if len(held) == 3:
            break
    for i, group in enumerate(held):
        fold = base.copy()
        target = fold.topology_group.eq(group)
        fold.loc[target, 'role'] = 'test'
        # Remaining groups: 90/10 train/validation; no non-target test samples.
        rest = fold.loc[~target]
        role = grouped_roles(rest, seed+i+1)
        fold.loc[~target, 'role'] = role.replace({'test': 'train'}).to_numpy()
        name = f'loto_{i:02d}'
        fold.to_parquet(split_dir / (name + '.parquet'), index=False, compression='zstd')
        manifest[name] = {'status': 'ready', 'file': name + '.parquet', 'heldout_topology_groups': [group],
                          'heldout_source_topologies': sorted(fold.loc[target, 'topology_id'].unique()), 'seed': seed+i+1,
                          'readiness_scope': 'leakage_checked_membership_only',
                          'evaluation_status': 'blocked_no_verified_cross_topology_targets', 'verified_shared_semantic_ids': [],
                          'reason': 'The held-out membership is usable, but a shared physical target head requires verified compatible extraction and test conditions. Current metric semantic IDs remain topology-scoped; do not report a cross-topology score from equal vector slots.'}
    # Candidate cross-dataset membership is useful for later verified mappings, but NOT an evaluable benchmark.
    for source in ['FALCON', 'AICircuit', 'AnalogGym']:
        fold = base.copy()
        target = fold.source_dataset.eq(source)
        test_designs = set(fold.loc[target, 'design_group'])
        embargo = ~target & fold.design_group.isin(test_designs)
        fold.loc[target, 'role'] = 'test'
        fold.loc[~target, 'role'] = fold.loc[~target, 'role'].replace({'test': 'train'})
        fold.loc[embargo, 'role'] = 'excluded_duplicate_design'
        name = 'cross_' + source.lower()
        fold.to_parquet(split_dir / (name + '_candidate.parquet'), index=False, compression='zstd')
        manifest[name] = {'status': 'blocked_incompatible_or_unverified', 'file': name + '_candidate.parquet',
                          'heldout_source': source, 'comparable_semantic_ids': [], 'test_samples': int(target.sum()),
                          'embargoed_samples': int(embargo.sum()),
                          'reason': 'No cross-source metric has verified equal physical extraction AND test conditions. AnalogGym has zero verified supervised rows. Candidate membership must not be used to report a cross-dataset score.'}
    dump(split_dir / 'protocols.json', manifest)
    return manifest
