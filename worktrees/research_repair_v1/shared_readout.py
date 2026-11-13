"""Train/validation-only shared ridge heads for the 11 verified LOADBench sets.

These are independent ID readouts, not LOTO-trained encoders or zero-shot heads.
No output is shared unless the existing protocol proves target/fixture equality.
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import numpy as np
from core import ROOT, OUT, HERE, read, sha, atomic, freeze, gate, metrics, summarize, now

DEST = OUT / 'loadbench_shared_v2'
RAW = ROOT / 'data/model_ready/fusion_v2_training_v1'
META = ROOT / 'data/model_ready/fusion_v2_frozen_e19_bf0d71ad'
OLD = ROOT / 'artifacts/loadbench_remediation_v1'


def fit(x, y, mask, alpha):
    """Observed-only standardization + ridge with unpenalized intercept."""
    center = x.mean(0); std = x.std(0); active = np.flatnonzero(std >= 1e-6)
    a = (x[:, active] - center[active]) / std[active]
    coef = np.zeros((len(active), 16)); bias = np.zeros(16); low = np.zeros(16); high = np.zeros(16)
    for j in range(16):
        valid = mask[:, j]
        if not valid.any():
            continue
        xx = a[valid]; yy = y[valid, j]; mx = xx.mean(0); my = yy.mean()
        xx = xx - mx; yy = yy - my
        coef[:, j] = np.linalg.solve(xx.T @ xx + alpha * np.eye(len(active)), xx.T @ yy)
        bias[j] = my - mx @ coef[:, j]
        low[j], high[j] = y[valid, j].min(), y[valid, j].max()
    return dict(center=center, std=std, active=active, coefficient=coef, bias=bias, low=low, high=high)


def predict(state, embedding):
    x = (np.asarray(embedding, dtype=np.float64)[:, state['active']] - state['center'][state['active']]) / state['std'][state['active']]
    return np.clip(x @ state['coefficient'] + state['bias'], state['low'], state['high'])


def bounded_blend(base, shared, weight, low, high, observed):
    result = base + weight * (shared - base)
    # Explicitly opt in to range projection; never modify labels or missingness.
    result[:, observed] = np.clip(result[:, observed], low[observed], high[observed])
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--checks', action='store_true'); args = parser.parse_args()
    if args.checks:
        x = np.array([[0., 2.], [1., 2.], [2., 2.], [3., 2.]])
        y = np.tile(np.arange(16.), (4, 1)); y[:, 0] = [0., 1., 2., 3.]
        mask = np.ones_like(y, dtype=bool); mask[-1, 0] = False
        a = fit(x, y, mask, 1.); y[-1, 0] = 1e30; b = fit(x, y, mask, 1.)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        p = predict(a, np.array([[1e9, 2.]])); assert np.isfinite(p).all() and p[0, 0] <= 2.
        atomic(DEST / 'checks.json', {'passed': True, 'missing_label_has_no_effect': True,
                                   'constant_input_safe': True, 'train_range_projection': True})
        print('shared-readout checks passed'); return
    gate(1.1)
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    details_path = ROOT / 'data/fusion_v2/supervision_group_details.json'
    details = read(details_path); fm = read(META / 'manifest.json'); tasks = read(RAW / 'tasks.json')
    bundle_path = ROOT / 'artifacts/experiment1_rgcn_local_head_v1/bundle.json'
    bundle = read(bundle_path); corrected = read(OLD / 'bundle.json')
    paths = [HERE / n for n in ['core.py', 'shared_readout.py']]
    paths += [details_path, META / 'manifest.json', RAW / 'manifest.json', RAW / 'tasks.json', bundle_path, OLD / 'bundle.json']
    for merge in details['verified_merge_sets']:
        for g in merge['supervision_group_ids']:
            paths.append(ROOT / bundle['contracts'][g]['index'])
            paths.append(OLD / (g + '__graph_design_affine32.npy'))
    recipe = {'protocol': 'verified_LOADBench_shared_ridge_v1', 'merge_ids': [m['merge_id'] for m in details['verified_merge_sets']],
              'sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
              'alphas': [.1, 10., 100.], 'weights': [0., .1, .25, .5, 1.],
              'fit_labels': 'Immutable training only, observed-only, raw engineering scales then signed-log1p',
              'selection': 'Per verified set: maximize validation macro hit5 under non-worse median R2, mean R2 and negative-count; per-target RMSE guard <=1.05x baseline. Tie: lower mean RMSE ratio then smaller weight. Identity baseline always available.',
              'range_projection': 'Candidates: own-topology TRAIN support or pooled verified-set TRAIN support; identity baseline untouched',
              'quality_policy': 'No rows dropped, raw original label mask unchanged; validation all-original only',
              'scope': 'Known-topology ID only, frozen public ID encoder; cannot claim LOTO or zero-shot',
              'test_labels_read': False}
    freeze(DEST / 'recipe.json', recipe)
    maps = {role: {k: np.load(META / role / (k + '.npy'), mmap_mode='r') for k in ['embedding', 'group', 'target', 'mask', 'refs']}
            for role in ['train', 'validation']}
    old_prediction = np.load(ROOT / 'artifacts/experiment1_rgcn_local_head_v1/validation_prediction_symlog.npy', mmap_mode='r')
    results = {}; before_all = []; after_all = []
    for merge in details['verified_merge_sets']:
        gate(.7); mid = merge['merge_id']; train_x = []; train_z = []; train_mask = []; parts = []
        for g, top in zip(merge['supervision_group_ids'], merge['topology_ids']):
            bank_path = ROOT / bundle['contracts'][g]['index']
            assert sha(bank_path) == bundle['contracts'][g]['index_sha256']
            code = fm['groups'].index(g)
            ti = np.flatnonzero(np.asarray(maps['train']['group']) == code)
            vi = np.flatnonzero(np.asarray(maps['validation']['group']) == code)
            with np.load(bank_path, allow_pickle=False) as f:
                z = f['train_z'].copy(); mask = f['train_mask'].copy(); scales = f['scales'].copy()
                ids = f['train_ids'].copy()
            # Banks and frozen encoder cache must refer to identical ordered TRAIN records.
            manifest = read(RAW / 'manifest.json'); folder = ROOT / manifest['topologies'][top]['folder']
            raw_ids = np.load(folder / 'sample_ids.npy', mmap_mode='r')
            raw_roles = np.load(folder / 'role.npy', mmap_mode='r')
            tr = np.asarray(maps['train']['refs'][ti, 1], dtype=int)
            vr = np.asarray(maps['validation']['refs'][vi, 1], dtype=int)
            assert np.all(raw_roles[tr] == 0) and np.all(raw_roles[vr] == 1) and not set(tr) & set(vr)
            # The frozen feature cache and neighbor bank use different TRAIN
            # orderings. Align by immutable sample ID, never by row position.
            bank_lookup = {bytes(s): i for i, s in enumerate(ids)}
            ordered_ids = raw_ids[tr]
            assert len(bank_lookup) == len(ids) == len(ordered_ids)
            assert set(bank_lookup) == set(map(bytes, ordered_ids))
            order = np.array([bank_lookup[bytes(s)] for s in ordered_ids])
            z, mask = z[order], mask[order]
            np.testing.assert_array_equal(mask, maps['train']['mask'][ti])
            yy = np.asarray(maps['train']['target'][ti], dtype=np.float64) / scales
            np.testing.assert_allclose(z[mask], (np.sign(yy) * np.log1p(abs(yy)))[mask], rtol=2e-6, atol=2e-6)
            assert len(ti) == len(z)
            train_x.append(np.asarray(maps['train']['embedding'][ti], dtype=np.float64)); train_z.append(z); train_mask.append(mask)
            base = np.asarray(old_prediction[vi], dtype=np.float64).copy()
            weight = corrected['groups'].get(g, {}).get('effective_weight', 0.)
            if weight:
                correction = np.load(OLD / (g + '__graph_design_affine32.npy'))
                assert correction.shape == base.shape
                base += weight * (correction - base)
            own_low = np.array([z[mask[:, j], j].min() if mask[:, j].any() else 0. for j in range(16)])
            own_high = np.array([z[mask[:, j], j].max() if mask[:, j].any() else 0. for j in range(16)])
            parts.append({'group': g, 'topology': top, 'indices': vi, 'base': base, 'scales': scales,
                          'own_low': own_low, 'own_high': own_high, 'observed': mask.any(0),
                          'train_rows': len(ti), 'validation_rows': len(vi), 'train_ids_sha256': sha(bank_path)})
            raw_ids._mmap.close(); raw_roles._mmap.close()
        x = np.concatenate(train_x); z = np.concatenate(train_z); mask = np.concatenate(train_mask)
        def score(predictions):
            rows = []
            for part, pred in zip(parts, predictions):
                vi = part['indices']; y = np.asarray(maps['validation']['target'][vi]); m = np.asarray(maps['validation']['mask'][vi])
                physical = np.sign(pred) * np.expm1(abs(pred)) * part['scales']
                rows += metrics(y, physical, m, tasks[part['group']]['slots'], part['topology'])
            return rows, summarize(rows)
        base_rows, base_summary = score([p['base'] for p in parts]); before_all += base_rows
        candidates = [{'alpha': None, 'weight': 0., 'kind': 'identity', 'summary': base_summary, 'accepted': True}]
        best = candidates[0]; best_rows = base_rows; best_predictions = [p['base'] for p in parts]; best_state = None
        def ranking(candidate, rows):
            ratios = [r['rmse'] / max(b['rmse'], 1e-12) for r, b in zip(rows, base_rows)]
            return (candidate['summary']['macro_hit5'], -float(np.mean(ratios)), -candidate['weight'])
        best_rank = ranking(best, best_rows)
        own_predictions = [bounded_blend(p['base'], p['base'], 0., p['own_low'], p['own_high'], p['observed']) for p in parts]
        own_rows, own_summary = score(own_predictions)
        good = (own_summary['median_r2'] >= base_summary['median_r2'] - 1e-10 and
                own_summary['macro_r2'] >= base_summary['macro_r2'] - 1e-10 and
                own_summary['negative_r2_targets'] <= base_summary['negative_r2_targets'] and
                all(r['rmse'] <= max(b['rmse'] * 1.05, 1e-12) for r, b in zip(own_rows, base_rows)))
        own_candidate = {'alpha': None, 'weight': 0., 'kind': 'topology_train_bounds', 'summary': own_summary, 'accepted': bool(good)}
        candidates.append(own_candidate)
        if good and ranking(own_candidate, own_rows) > best_rank:
            best, best_rows, best_predictions = own_candidate, own_rows, own_predictions
            best_rank = ranking(best, best_rows)
        for alpha in recipe['alphas']:
            state = fit(x, z, mask, alpha)
            shared = [predict(state, maps['validation']['embedding'][p['indices']]) for p in parts]
            for w in recipe['weights']:
                predictions = [bounded_blend(p['base'], s, w, state['low'], state['high'], mask.any(0)) for p, s in zip(parts, shared)]
                rows, summary = score(predictions)
                good = (summary['median_r2'] >= base_summary['median_r2'] - 1e-10 and
                        summary['macro_r2'] >= base_summary['macro_r2'] - 1e-10 and
                        summary['negative_r2_targets'] <= base_summary['negative_r2_targets'] and
                        all(r['rmse'] <= max(b['rmse'] * 1.05, 1e-12) for r, b in zip(rows, base_rows)))
                candidate = {'alpha': alpha, 'weight': w, 'kind': 'bounded_shared', 'summary': summary, 'accepted': bool(good)}
                candidates.append(candidate)
                if good and ranking(candidate, rows) > best_rank:
                    best = candidate; best_rows = rows; best_predictions = predictions; best_state = state; best_rank = ranking(candidate, rows)
        folder = DEST / mid; folder.mkdir(parents=True, exist_ok=True)
        if best_state is not None:
            np.savez_compressed(folder / 'head.npz', **best_state)
        if best['kind'] == 'topology_train_bounds':
            np.savez_compressed(folder / 'bounds.npz', **{p['group'] + '__' + k: p[k] for p in parts for k in ['own_low', 'own_high', 'observed']})
        for part, prediction in zip(parts, best_predictions):
            np.save(folder / (part['group'] + '__validation.npy'), prediction, allow_pickle=False)
        after_all += best_rows
        record = {'merge_id': mid, 'baseline': base_summary, 'selected': best, 'candidates': candidates,
                  'train_rows': len(x), 'validation_rows': sum(p['validation_rows'] for p in parts),
                  'members': [{k: v for k, v in p.items() if k not in ['indices', 'base', 'scales', 'own_low', 'own_high', 'observed']} for p in parts],
                  'per_target_before': base_rows, 'per_target_after': best_rows, 'test_labels_read': False,
                  'missing_mask_changed': False, 'recipe_sha256': sha(DEST / 'recipe.json')}
        atomic(folder / 'selection.json', record); results[mid] = {k: record[k] for k in ['train_rows', 'validation_rows', 'baseline', 'selected']}
        atomic(DEST / 'status.json', {'state': 'validating', 'sets_complete': len(results), 'total': len(recipe['merge_ids']), 'updated_at': now()})
        print(mid, 'baseline', base_summary, 'selected', best, flush=True)
    atomic(DEST / 'summary.json', {'state': 'validation_complete', 'sets': results,
                                 'baseline': summarize(before_all), 'selected': summarize(after_all),
                                 'test_labels_read': False, 'new_test_evaluation_required': True,
                                 'ready_for_deployment': False, 'at': now()})
    atomic(DEST / 'status.json', {'state': 'validation_complete', 'sets_complete': len(results), 'updated_at': now()})


if __name__ == '__main__':
    main()
