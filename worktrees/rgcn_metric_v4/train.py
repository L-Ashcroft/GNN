"""Train-supervised metric + validation-selected masked local experts.

Frozen RGCN and physical skip remain distinct. Canonical topology ID only.
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[key] = '1'
from pathlib import Path
import sys
import argparse
import time
import numpy as np
import psutil
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[2]; HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'worktrees/rgcn_accuracy_v3'))
from common import read, sha, atomic, freeze, save_npz, now, RAW, META
from decoder import transform, physical
sys.path.insert(0, str(ROOT / 'worktrees/research_repair_v1'))
from core import metrics, summarize
sys.path.insert(0, str(ROOT / 'worktrees/loadbench_bounds_inference_v1'))
from interface import BoundedPhysicalReadout
sys.path.insert(0, str(ROOT / 'worktrees/loadbench_remediation_v1'))
from readout import predict
OUT = ROOT / 'artifacts/rgcn_metric_v4'
UNIT = {'m': 1e-6, 'F': 1e-12, 'H': 1e-9, 'ohm': 1e3, 'V': 1., 'A': 1e-6, 'Hz': 1e6, '1': 1.}
PROBE = ['LOADBench__SimpleOpAmp4', 'LOADBench__SimpleThreeStageOpAmp_SimpleOpAmp1_1_10',
         'LOADBench__SimpleTwoStageOpAmp_SimpleOpAmp92_9', 'LOADBench__SymmetricalOpAmp12']


def fit_projection(x, target, mask):
    weight = np.zeros((x.shape[1], 16)); intercept = np.zeros(16)
    for j in range(16):
        selected = mask[:, j]
        if not selected.any(): continue
        a = x[selected].astype('float64'); y = target[selected, j]
        mx, my = a.mean(0), y.mean(); a -= mx
        weight[:, j] = np.linalg.solve(a.T @ a + 10. * np.eye(x.shape[1]), a.T @ (y - my))
        intercept[j] = my - mx @ weight[:, j]
    return weight, intercept


def feature_modes(parameter, embedding, projection):
    a = parameter / max(1., np.sqrt(parameter.shape[1])); b = embedding / max(1., np.sqrt(embedding.shape[1]))
    p = projection / 4.
    return {'physical': a, 'graph_physical': np.column_stack([a, b]),
            'supervised': np.column_stack([a, b * .25, p]),
            'supervised_physical': np.column_stack([a, p])}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--probe', action='store_true'); args = ap.parse_args()
    if psutil.virtual_memory().available < .7 * 2**30: raise SystemExit(75)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    spec = read(ROOT / 'worktrees/rgcn_accuracy_v3/suite.json')
    recipe = {'version': 'supervised_neighborhood_masked_local_expert_v4', 'topologies': spec['topologies'],
              'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in [HERE / 'train.py', ROOT / 'worktrees/rgcn_accuracy_v3/common.py',
                              ROOT / 'worktrees/rgcn_accuracy_v3/decoder.py', ROOT / 'worktrees/loadbench_remediation_v1/readout.py', RAW / 'manifest.json', META / 'manifest.json',
                              ROOT / 'artifacts/research_repair_v1/loadbench_shared_v2/deployment.json']},
              'metric_training': 'Ridge10 on masked TRAIN-only standardized physical labels to learn projection of parameters+frozenGNN; no query labels',
              'features': ['physical', 'graph_physical', 'supervised', 'supervised_physical'], 'neighbors': [8, 32, 64],
              'local_ridge': .01, 'blend_weights': [.25, .5, 1.],
              'selection': 'Per contract+metric validation only: strictly improve hit5, RMSE and MAE no worse than fixed baseline. For dB: improve MAE, RMSE no worse. Identity available.',
              'sparse_gate': 'At least64TRAIN rows and30observed VALIDATION labels per metric; otherwise retain baseline',
              'model_scope': 'Frozen edge-RGCN + input-aware local experts, known-topology ID only',
              'validation_selection_caveat': 'Multiple candidate and per-metric validation selection; requires frozen follow-up test, no claim of independent blind test',
              'target_normalization_floor': .02, 'missing_mask_changed': False, 'test_labels_used': False, 'seed': 42}
    freeze(OUT / 'recipe.json', recipe)
    fm = read(META / 'manifest.json'); manifest = read(RAW / 'manifest.json'); tasks = read(RAW / 'tasks.json')
    meta = {r: {k: np.load(META / r / (k + '.npy'), mmap_mode='r') for k in ['embedding', 'refs', 'target', 'mask', 'group']} for r in ['train', 'validation']}
    bounds = BoundedPhysicalReadout(); completed = {}; all_before = []; all_after = []; begin = time.monotonic()
    for name in PROBE if args.probe else spec['topologies']:
        output = OUT / name; output.mkdir(parents=True, exist_ok=True)
        if (output / 'selection_frozen.json').exists():
            r = read(output / 'selection_frozen.json'); completed[name] = r['selected_summary']; all_before += r['baseline_rows']; all_after += r['selected_rows']; continue
        if (OUT / 'STOP').exists() or psutil.virtual_memory().available < 512 * 2**20: raise SystemExit(75)
        group = next(g for g, t in tasks.items() if t['contract']['topology_id'] == name); code = fm['groups'].index(group); rules = tasks[group]['slots']
        scales = np.array([rules.get(str(j), {}).get('scale', 1.) for j in range(16)])
        kinds = ['linear' if rules.get(str(j), {}).get('unit') in {'dB', 'dBm', 'dBc/Hz', 'source_label_dBm_compression_function', 'V', 'degree', '1/K', 'V/V', 'V/A'} else 'symlog' for j in range(16)]
        table = pq.read_table(ROOT / 'data/fusion_v2/sampled' / (name + '.parquet'), columns=['sample_id', 'parameters', 'parameter_units', 'parameter_names'])
        params = np.array(table['parameters'].to_pylist()); units = table['parameter_units'][0].as_py()
        parameter_scales = np.array([UNIT.get(u, 1.) for u in units]); params = np.sign(params) * np.log1p(abs(params) / parameter_scales)
        folder = ROOT / manifest['topologies'][name]['folder']; ids = np.load(folder / 'sample_ids.npy', mmap_mode='r'); roles = np.load(folder / 'role.npy', mmap_mode='r')
        partition_ids = table['sample_id'].to_pylist(); assert [s.decode() for s in ids] == partition_ids
        ix = {r: np.flatnonzero(np.asarray(meta[r]['group']) == code) for r in meta}
        pos = {r: np.asarray(meta[r]['refs'][i, 1], dtype=int) for r, i in ix.items()}
        assert np.all(roles[pos['train']] == 0) and np.all(roles[pos['validation']] == 1) and not set(pos['train']) & set(pos['validation'])
        pcenter = params[pos['train']].mean(0); pstd = params[pos['train']].std(0); pa = np.flatnonzero(pstd > 1e-6)
        txp, vxp = [(params[pos[r]][:, pa] - pcenter[pa]) / pstd[pa] for r in ['train', 'validation']]
        te = np.asarray(meta['train']['embedding'][ix['train']], dtype='float64'); ve = np.asarray(meta['validation']['embedding'][ix['validation']], dtype='float64')
        ec = te.mean(0); es = te.std(0); ea = np.flatnonzero(es > 1e-6); te = (te[:, ea] - ec[ea]) / es[ea]; ve = (ve[:, ea] - ec[ea]) / es[ea]
        y = np.asarray(meta['train']['target'][ix['train']], dtype='float64'); tm = np.asarray(meta['train']['mask'][ix['train']], dtype=bool)
        vy = np.asarray(meta['validation']['target'][ix['validation']], dtype='float64'); vm = np.asarray(meta['validation']['mask'][ix['validation']], dtype=bool)
        zz = transform(y, scales, kinds); count = tm.sum(0)
        mu = np.divide(np.where(tm, zz, 0.).sum(0), count, out=np.zeros(16), where=count > 0)
        sd = np.maximum(np.sqrt(np.divide(np.where(tm, (zz - mu)**2, 0.).sum(0), count, out=np.zeros(16), where=count > 0)), .02)
        combined = np.column_stack([txp, te]); combined_v = np.column_stack([vxp, ve])
        w, b = fit_projection(combined, (zz - mu) / sd, tm)
        features = feature_modes(txp, te, combined @ w + b); queries = feature_modes(vxp, ve, combined_v @ w + b)
        basefile = ROOT / 'artifacts/loadbench_remediation_v1' / (group + '__accepted.npy')
        if not basefile.exists(): basefile = basefile.with_name(group + '__published.npy')
        prior_z = np.load(basefile); prior = bounds(group, np.sign(prior_z) * np.expm1(abs(prior_z)) * scales)
        before = metrics(vy, prior, vm, rules, name); selected = [dict(r) for r in before]; prediction = prior.copy(); choices = {str(r['slot']): {'mode': 'identity'} for r in before}
        if len(y) >= 64:
            for mode, train_x in features.items():
                for k in recipe['neighbors']:
                    local, _ = predict(train_x, queries[mode], zz, tm, k=k, ridge=.01)
                    candidate = physical(local, scales, kinds)
                    for weight in recipe['blend_weights']:
                        blend = prior * (1. - weight) + candidate * weight
                        rows = metrics(vy, blend, vm, rules, name)
                        for j, (base, old, new) in enumerate(zip(before, selected, rows)):
                            if new['count'] < 30: continue
                            if new['rmse'] > base['rmse'] + 1e-12 or new['mae'] > base['mae'] + 1e-12: continue
                            improve = new['hit5'] is not None and new['hit5'] > old['hit5'] + 1e-4
                            improve |= new['hit5'] is None and new['mae'] < old['mae'] - 1e-6
                            if improve:
                                slot = new['slot']; selected[j] = new; prediction[:, slot] = blend[:, slot]
                                choices[str(slot)] = dict(mode=mode, neighbors=k, weight=weight)
                    atomic(OUT / 'status.json', {'state': 'training_metric_experts', 'pid': os.getpid(), 'create_time': psutil.Process().create_time(),
                           'topology': name, 'mode': mode, 'neighbors': k, 'topologies_complete': len(completed), 'total': len(spec['topologies']),
                           'baseline_hit5': summarize(before)['macro_hit5'], 'selected_hit5': summarize(selected)['macro_hit5'],
                           'RSS_MiB': psutil.Process().memory_info().rss / 2**20, 'updated_at': now()})
        save_npz(output / 'model.npz', **{} ) if False else None
        save_npz(output / 'model.npz', dict(parameter_center=pcenter, parameter_std=pstd, parameter_active=pa,
                 embedding_center=ec, embedding_std=es, embedding_active=ea, projection_weight=w, projection_bias=b,
                 train_parameter=txp, train_embedding=te, train_z=zz, train_mask=tm, scales=scales))
        norm = {'group': group, 'parameter_names': table['parameter_names'][0].as_py(), 'parameter_scales': parameter_scales.tolist(), 'kinds': kinds}
        atomic(output / 'normalization.json', norm); np.save(output / 'validation_prediction.npy', prediction, allow_pickle=False)
        result = {'topology': name, 'group': group, 'choices': choices, 'baseline_summary': summarize(before), 'selected_summary': summarize(selected),
                  'baseline_rows': before, 'selected_rows': selected, 'train_rows': len(y), 'validation_rows': len(vy),
                  'test_used': False, 'model_sha256': sha(output / 'model.npz'), 'recipe_sha256': sha(OUT / 'recipe.json'),
                  'normalization_sha256': sha(output / 'normalization.json'), 'partition_sha256': sha(ROOT / 'data/fusion_v2/sampled' / (name + '.parquet')),
                  'training_ids_sha256': __import__('hashlib').sha256(ids[pos['train']].tobytes()).hexdigest()}
        atomic(output / 'selection_frozen.json', result); completed[name] = result['selected_summary']; all_before += before; all_after += selected
        ids._mmap.close(); roles._mmap.close()
        atomic(OUT / ('probe_summary.json' if args.probe else 'summary.json'), {'state': 'training', 'topologies_complete': len(completed),
               'baseline': summarize(all_before), 'selected': summarize(all_after), 'jobs': completed, 'test_used': False})
        print(name, result['baseline_summary']['macro_hit5'], '->', result['selected_summary']['macro_hit5'], flush=True)
    summaryfile = OUT / ('probe_summary.json' if args.probe else 'summary.json'); summary = read(summaryfile)
    summary.update(state='probe_complete' if args.probe else 'validation_complete', seconds=time.monotonic() - begin)
    atomic(summaryfile, summary); atomic(OUT / 'status.json', {'state': summary['state'], 'topologies_complete': len(completed), 'updated_at': now()})


if __name__ == '__main__': main()
