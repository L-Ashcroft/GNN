"""Frozen-candidate replay using existing test predictions; no encoder rerun."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from core import ROOT, HERE, OUT, read, sha, atomic, freeze, metrics, summarize, gate, now


def main():
    gate(1.)
    candidate = OUT / 'loadbench_shared_v2'; dest = candidate / 'fixed_test'
    summaries = read(candidate / 'summary.json'); assert summaries['state'] == 'validation_complete'
    details = read(ROOT / 'data/fusion_v2/supervision_group_details.json')
    tasks = read(ROOT / 'data/model_ready/fusion_v2_training_v1/tasks.json')
    old = ROOT / 'artifacts/loadbench_remediation_v1/fixed_test'
    mapping = {}; paths = [candidate / 'recipe.json', candidate / 'summary.json', HERE / 'core.py', HERE / 'loadbench_test.py']
    for merge in details['verified_merge_sets']:
        folder = candidate / merge['merge_id']; selection = read(folder / 'selection.json'); paths.append(folder / 'selection.json')
        assert selection['selected']['kind'] in ['identity', 'topology_train_bounds'], 'This evaluator needs a new explicit implementation for a learned shared head'
        if selection['selected']['kind'] == 'topology_train_bounds':
            paths.append(folder / 'bounds.npz')
            with np.load(folder / 'bounds.npz', allow_pickle=False) as f:
                for g, top in zip(merge['supervision_group_ids'], merge['topology_ids']):
                    mapping[top] = (g, {k: f[g + '__' + k].copy() for k in ['own_low', 'own_high', 'observed']})
    paths += list(old.glob('*.parquet'))
    recipe = {'sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
              'selection': 'Frozen validation candidates only; no test selection or candidate changes',
              'scope': 'LOADBench known-topology ID replay; historical test reused, not a new blind test',
              'no_label_mask_or_split_changes': True, 'new_encoder_execution': False}
    freeze(dest / 'evaluation_frozen.json', recipe)
    by_topology = {t['contract']['topology_id']: (g, t) for g, t in tasks.items() if t['contract']['source_dataset'] == 'LOADBench'}
    before = []; after = []; safe_before = []; safe_after = []; total = 0; changed = 0
    for p in sorted(old.glob('*.parquet')):
        top = p.stem; g, task = by_topology[top]; table = pq.read_table(p); values = table.to_pylist()
        ids = [r['sample_id'] for r in values]; assert len(ids) == len(set(ids))
        y = np.array([r['target'] for r in values], dtype=np.float64)
        mask = np.array([r['mask'] for r in values], dtype=bool)
        base = np.array([r['prediction'] for r in values], dtype=np.float64); pred = base.copy()
        if top in mapping:
            _, bounds = mapping[top]
            scale = np.array([task['slots'].get(str(j), {}).get('scale', 1.) for j in range(16)])
            lo = np.sign(bounds['own_low']) * np.expm1(abs(bounds['own_low'])) * scale
            hi = np.sign(bounds['own_high']) * np.expm1(abs(bounds['own_high'])) * scale
            observed = bounds['observed']
            pred[:, observed] = np.clip(pred[:, observed], lo[observed], hi[observed])
        a = metrics(y, base, mask, task['slots'], top); b = metrics(y, pred, mask, task['slots'], top)
        for rows in [a, b]:
            for row in rows:
                row['r2_small_sample_warning'] = row['count'] < 30
                row['r2_low_variance_warning'] = row['target_std'] < 1e-6 * task['slots'][str(row['slot'])]['scale']
        before += a; after += b; total += len(values); changed += int(np.any(pred != base, axis=1).sum())
        # Separate sensitivity view, never replace the original observed mask.
        q = pq.read_table(ROOT / 'data/audit/loadbench_physical_v1/quality_index.parquet',
                          columns=['sample_id', 'needs_review'], filters=[('topology_id', '=', top), ('baseline_ID_role', '=', 'test')])
        review = {r['sample_id'] for r in q.to_pylist() if r['needs_review']}
        eligible = mask.copy()
        for slot, rule in task['slots'].items():
            if rule['original_name'] in ['ft', 'phasemargin']:
                eligible[:, int(slot)] &= np.array([s not in review for s in ids])
        safe_before += metrics(y, base, eligible, task['slots'], top)
        safe_after += metrics(y, pred, eligible, task['slots'], top)
        for i, r in enumerate(values):
            r['prediction'] = pred[i].tolist(); r['engineering_eligible'] = eligible[i].tolist()
        pq.write_table(pa.Table.from_pylist(values), dest / p.name, compression='zstd')
    assert total == 19301
    result = {'state': 'complete', 'samples': total, 'topologies': len(by_topology), 'changed_predictions_samples': changed,
              'raw_all_original': {'baseline': summarize(before), 'candidate': summarize(after)},
              'AC_no_review_sensitivity_only': {'baseline': summarize(safe_before), 'candidate': summarize(safe_after)},
              'r2_n_ge_30_sensitivity_only': {'baseline': summarize([r for r in before if r['count'] >= 30]),
                                            'candidate': summarize([r for r in after if r['count'] >= 30])},
              'per_target_before': before, 'per_target_after': after,
              'mask_preserved': True, 'test_reused_for_evaluation_only': True, 'test_not_used_to_choose_or_promote': True,
              'all_LOADBench_issues_resolved': False, 'at': now()}
    atomic(dest / 'metrics.json', result)
    atomic(dest / 'status.json', {'state': 'complete', 'samples': total, 'updated_at': now()})
    print({k: v for k, v in result.items() if not k.startswith('per_target')})


if __name__ == '__main__':
    main()
