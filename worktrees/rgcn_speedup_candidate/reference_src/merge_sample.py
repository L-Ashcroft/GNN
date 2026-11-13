import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .common import dump, read


def balanced_budgets(counts, target):
    budgets = {k: 0 for k in sorted(counts)}
    left = min(target, sum(counts.values()))
    while left:
        active = [k for k in budgets if budgets[k] < counts[k]]
        share = max(1, left // len(active))
        for key in active:
            n = min(share, counts[key] - budgets[key], left)
            budgets[key] += n
            left -= n
            if not left:
                break
    return budgets


def merge_sample(output, target=500000, seed=42, converted_dir=None):
    converted_dir = converted_dir or output / 'converted'
    specs = read(output / 'catalog.json')
    small = {}
    duplicates = []
    for s in specs:
        key = s['topology_id']
        df = pd.read_parquet(converted_dir / (key + '.parquet'),
                             columns=['sample_id', 'record_hash', 'design_group', 'source_row'])
        dupe = df.duplicated('record_hash', keep='first')
        duplicates.extend({'sample_id': sid, 'reason': 'identical_parameters_and_labels_same_source_topology'} for sid in df.loc[dupe, 'sample_id'])
        small[key] = df[~dupe]
    # Give each distinct full connectivity class a budget, then balance its source variants.
    groups = {}
    for s in specs:
        groups.setdefault(s['connectivity_group'], []).append(s['topology_id'])
    counts = {g: sum(len(small[k]) for k in ks) for g, ks in groups.items()}
    budgets = balanced_budgets(counts, target)
    quotas = {}
    for g, ks in groups.items():
        quotas.update(balanced_budgets({k: len(small[k]) for k in ks}, budgets[g]))
    index = []
    statistics = []
    (output / 'sampled').mkdir(exist_ok=True)
    for s in specs:
        key = s['topology_id']
        # A per-topology seed makes reruns independent of table traversal order.
        rng = np.random.default_rng(seed + int(__import__('hashlib').sha256(key.encode()).hexdigest()[:8], 16))
        candidates = small[key]
        selected = sorted(rng.choice(candidates['source_row'].to_numpy(), size=quotas[key], replace=False).tolist())
        table = pd.read_parquet(converted_dir / (key + '.parquet'))
        table = table[table.source_row.isin(selected)].sort_values('source_row').reset_index(drop=True)
        table.to_parquet(output / 'sampled' / (key + '.parquet'), index=False, compression='zstd')
        cols = ['sample_id', 'source_dataset', 'topology_id', 'topology_group', 'connectivity_group', 'design_group',
                'compatibility_group', 'input_complete', 'source_path', 'source_row', 'record_hash']
        ix = table[cols].copy()
        ix['partition'] = 'sampled/' + key + '.parquet'; ix['partition_row'] = np.arange(len(table))
        index.append(ix)
        statistics.append({'source_dataset': s['source_dataset'], 'topology_id': key, 'unique_candidates': len(candidates),
                           'sampled': len(table), 'input_complete': s['input_complete'],
                           'connectivity_group': s['connectivity_group'], 'topology_group': s['topology_group']})
    merged = pd.concat(index, ignore_index=True)
    merged.to_parquet(output / 'index.parquet', index=False, compression='zstd')
    merged[merged.input_complete].to_parquet(output / 'input_complete_index.parquet', index=False, compression='zstd')
    pd.DataFrame(duplicates, columns=['sample_id', 'reason']).to_parquet(output / 'duplicate_records.parquet', index=False)
    pd.DataFrame(statistics).to_csv(output / 'sampling_statistics.csv', index=False)
    summary = {'seed': seed, 'target': target, 'sampled': len(merged), 'input_complete_sampled': int(merged.input_complete.sum()),
               'unique_candidates': sum(len(v) for v in small.values()), 'exact_duplicate_records_removed': len(duplicates),
               'source_topologies': len(specs), 'connectivity_classes': len(groups),
               'conservative_mos_core_classes': merged.topology_group.nunique(),
               'sources': merged.source_dataset.value_counts().to_dict(),
               'same_design_multiple_rows': int(merged.duplicated('design_group', keep=False).sum())}
    dump(output / 'sampling_summary.json', summary)
    return summary
