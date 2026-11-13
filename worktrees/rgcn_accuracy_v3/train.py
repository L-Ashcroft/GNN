"""One-topology-at-a-time supervised neural decoder; validation selection only."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[key] = '1'
import sys
import argparse
import time
import gc
import numpy as np
import psutil
import pyarrow.parquet as pq
from common import ROOT, HERE, OUT, RAW, META, read, sha, atomic, freeze, now, save_npz
from decoder import Decoder, transform, physical
sys.path.insert(0, str(ROOT / 'worktrees/research_repair_v1'))
from core import metrics, summarize
sys.path.insert(0, str(ROOT / 'worktrees/loadbench_bounds_inference_v1'))
from interface import BoundedPhysicalReadout

UNIT = {'m': 1e-6, 'F': 1e-12, 'H': 1e-9, 'ohm': 1e3, 'V': 1., 'A': 1e-6, 'Hz': 1e6, '1': 1.}
PROBE = ['LOADBench__SimpleOpAmp4', 'LOADBench__SimpleThreeStageOpAmp_SimpleOpAmp1_1_10',
         'LOADBench__SimpleTwoStageOpAmp_SimpleOpAmp92_9', 'LOADBench__SymmetricalOpAmp12']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--probe', action='store_true'); args = ap.parse_args()
    if psutil.virtual_memory().available < .8 * 2**30: raise SystemExit(75)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    manifest = read(RAW / 'manifest.json'); fm = read(META / 'manifest.json'); tasks = read(RAW / 'tasks.json')
    bundle_path = ROOT / 'artifacts/experiment1_rgcn_local_head_v1/bundle.json'; bundle = read(bundle_path)
    meta = {role: {k: np.load(META / role / (k + '.npy'), mmap_mode='r') for k in ['embedding', 'refs', 'target', 'mask', 'group']}
            for role in ['train', 'validation']}
    spec = read(HERE / 'suite.json')
    paths = [HERE / n for n in ['common.py', 'decoder.py', 'train.py', 'checks.py']] + [HERE / 'suite.json', RAW / 'manifest.json', RAW / 'tasks.json', META / 'manifest.json', bundle_path,
            ROOT / 'artifacts/research_repair_v1/loadbench_shared_v2/deployment.json']
    recipe = {'sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)},
              'suite': spec, 'model': 'Frozen edge-RGCN128 embedding plus canonical SI-parameter skip, ridge-linear + SiLU64 residual',
              'loss': 'Missing-only masked Huber(delta2) on TRAIN-only standardized unit-aware transformed labels; new loss, old recipes unchanged',
              'target_std_floor': .02, 'input_statistics': 'TRAIN only', 'test_read': False,
              'scope': 'Known canonical topology ID only; not portable LOTO or zero-shot',
              'precision': 'FP64 decoder and target transforms, cached FP32 graph features',
              'selection': 'Validation macro hit5, median/meanR2 >= prior-.005, negative count <= prior, each target RMSE <=1.1*prior; identity always available',
              'epochs': 200, 'patience': 25, 'evaluation_interval': 5, 'batch': 128, 'seed': 42}
    freeze(OUT / 'recipe.json', recipe)
    bounds = BoundedPhysicalReadout(); completed = {}; baseline_rows = []; selected_rows = []; start = time.monotonic()
    names = PROBE if args.probe else spec['topologies']
    for name in names:
        output = OUT / name; output.mkdir(parents=True, exist_ok=True)
        if (output / 'selection_frozen.json').exists():
            saved = read(output / 'selection_frozen.json'); completed[name] = saved['selected_summary']
            baseline_rows += saved['baseline_rows']; selected_rows += saved['selected_rows']; continue
        if (OUT / 'STOP').exists() or psutil.virtual_memory().available < 512 * 2**20: raise SystemExit(75)
        code = fm['groups'].index(next(g for g, t in tasks.items() if t['contract']['topology_id'] == name))
        group = fm['groups'][code]; rules = tasks[group]['slots']; slots = sorted(map(int, rules))
        scales = np.array([rules.get(str(j), {}).get('scale', 1.) for j in range(16)])
        kinds = ['linear' if rules.get(str(j), {}).get('unit') in {'dB', 'dBm', 'dBc/Hz', 'source_label_dBm_compression_function', 'V', 'degree', '1/K', 'V/V', 'V/A'} else 'symlog' for j in range(16)]
        indices = {r: np.flatnonzero(np.asarray(meta[r]['group']) == code) for r in meta}
        table = pq.read_table(ROOT / 'data/fusion_v2/sampled' / (name + '.parquet'), columns=['sample_id', 'parameters', 'parameter_units', 'parameter_names'])
        params = np.asarray(table['parameters'].to_pylist(), dtype='float64'); units = table['parameter_units'][0].as_py()
        unit = np.array([UNIT.get(u, 1.) for u in units]); params = np.sign(params) * np.log1p(abs(params) / unit)
        folder = ROOT / manifest['topologies'][name]['folder']; roles = np.load(folder / 'role.npy', mmap_mode='r')
        ids = np.load(folder / 'sample_ids.npy', mmap_mode='r'); x = {}; target = {}; mask = {}; selected_ids = {}
        for r, role_number in [('train', 0), ('validation', 1)]:
            ix = indices[r]; local = np.asarray(meta[r]['refs'][ix, 1], dtype=int)
            assert np.all(roles[local] == role_number)
            selected_ids[r] = [s.decode() for s in ids[local]]
            x[r] = np.column_stack([params[local], np.asarray(meta[r]['embedding'][ix], dtype='float64')])
            target[r] = np.asarray(meta[r]['target'][ix], dtype='float64'); mask[r] = np.asarray(meta[r]['mask'][ix], dtype=bool)
        assert not set(selected_ids['train']) & set(selected_ids['validation'])
        train_count = len(x['train']); val_count = len(x['validation']); roles._mmap.close(); ids._mmap.close()
        old = ROOT / 'artifacts/loadbench_remediation_v1' / (group + '__accepted.npy')
        if old.exists(): prior_z = np.load(old)
        else: prior_z = np.load(ROOT / 'artifacts/loadbench_remediation_v1' / (group + '__published.npy'))
        prior = bounds(group, np.sign(prior_z) * np.expm1(abs(prior_z)) * scales)
        assert prior.shape == target['validation'].shape
        prior_rows = metrics(target['validation'], prior, mask['validation'], rules, name); prior_summary = summarize(prior_rows)
        best_summary = prior_summary; best_rows = prior_rows; best_pred = prior; best_state = None; best_weight = 0.; best_epoch = 0
        center = x['train'].mean(0); std = x['train'].std(0); active = np.flatnonzero(std >= 1e-6)
        tx = ((x['train'][:, active] - center[active]) / std[active]).astype('float32')
        vx = ((x['validation'][:, active] - center[active]) / std[active]).astype('float32')
        z = transform(target['train'], scales, kinds); observed_count = mask['train'].sum(0)
        mu = np.divide(np.where(mask['train'], z, 0.).sum(0), observed_count, out=np.zeros(16), where=observed_count > 0)
        sd = np.maximum(np.sqrt(np.divide(np.where(mask['train'], (z - mu)**2, 0.).sum(0), observed_count, out=np.zeros(16), where=observed_count > 0)), .02)
        low = np.array([z[mask['train'][:, j], j].min() if observed_count[j] else 0. for j in range(16)])
        high = np.array([z[mask['train'][:, j], j].max() if observed_count[j] else 0. for j in range(16)])
        history = []; best_raw_hit = -1.; bad = 0; factor = 1.
        if train_count >= 64 and val_count >= 30:
            model = Decoder(tx, (z[:, slots] - mu[slots]) / sd[slots], mask['train'][:, slots])
            for epoch in range(201):
                if epoch:
                    order = model.rng.permutation(train_count)
                    for off in range(0, train_count, 128):
                        ix = order[off:off + 128]; loss, grad = model.gradient(tx[ix], ix); model.update(grad, factor)
                if epoch % 5: continue
                if (OUT / 'STOP').exists() or psutil.virtual_memory().available < 384 * 2**20: raise SystemExit(75)
                p, _ = model.forward(vx); zz = np.broadcast_to(mu, (val_count, 16)).copy(); zz[:, slots] += p * sd[slots]
                neural = physical(np.clip(zz, low, high), scales, kinds)
                pure = summarize(metrics(target['validation'], neural, mask['validation'], rules, name))
                if pure['macro_hit5'] > best_raw_hit + 1e-4: best_raw_hit = pure['macro_hit5']; bad = 0
                else: bad += 5
                accepted = False
                for weight in [.1, .25, .5, .75, 1.]:
                    mixed = (1. - weight) * prior + weight * neural
                    rows = metrics(target['validation'], mixed, mask['validation'], rules, name); summary = summarize(rows)
                    guard = (summary['median_r2'] >= prior_summary['median_r2'] - .005 and
                             summary['macro_r2'] >= prior_summary['macro_r2'] - .005 and
                             summary['negative_r2_targets'] <= prior_summary['negative_r2_targets'] and
                             all(r['rmse'] <= max(1e-12, b['rmse'] * 1.1) for r, b in zip(rows, prior_rows)))
                    if guard and summary['macro_hit5'] > best_summary['macro_hit5'] + 1e-4:
                        best_summary, best_rows, best_pred = summary, rows, mixed
                        best_weight, best_epoch, best_state = weight, epoch, model.state(); accepted = True
                history.append({'epoch': epoch, 'neural_hit5': pure['macro_hit5'], 'selected_hit5': best_summary['macro_hit5'], 'selected_candidate_changed': accepted})
                if bad >= 15: factor = .3
                atomic(OUT / 'status.json', {'state': 'training', 'pid': os.getpid(), 'create_time': psutil.Process().create_time(), 'topology': name,
                       'epoch': epoch, 'train_rows': train_count, 'validation_rows': val_count, 'topologies_complete': len(completed),
                       'baseline_hit5': prior_summary['macro_hit5'], 'selected_hit5': best_summary['macro_hit5'],
                       'RSS_MiB': psutil.Process().memory_info().rss / 2**20, 'available_RAM_GiB': psutil.virtual_memory().available / 2**30, 'updated_at': now()})
                if epoch >= 30 and bad >= 25: break
        else: model = None
        if best_state is not None: save_npz(output / 'best.npz', best_state)
        norm = dict(center=center.tolist(), std=std.tolist(), active=active.tolist(), target_center=mu.tolist(), target_std=sd.tolist(),
                    low=low.tolist(), high=high.tolist(), scales=scales.tolist(), kinds=kinds, slots=slots, parameter_units=units,
                    parameter_names=table['parameter_names'][0].as_py(), group=group, train_rows=train_count,
                    training_ids_sha256=__import__('hashlib').sha256('\n'.join(selected_ids['train']).encode()).hexdigest())
        atomic(output / 'normalization.json', norm); atomic(output / 'history.json', history)
        np.save(output / 'validation_prediction.npy', best_pred, allow_pickle=False)
        selection = {'topology': name, 'group': group, 'state': 'frozen', 'best_epoch': best_epoch, 'neural_weight': best_weight,
                     'baseline_summary': prior_summary, 'selected_summary': best_summary, 'baseline_rows': prior_rows, 'selected_rows': best_rows,
                     'checkpoint_sha256': sha(output / 'best.npz') if best_state is not None else None,
                     'normalization_sha256': sha(output / 'normalization.json'), 'recipe_sha256': sha(OUT / 'recipe.json'),
                     'raw_partition_sha256': sha(ROOT / 'data/fusion_v2/sampled' / (name + '.parquet')),
                     'test_used': False, 'train_rows': train_count, 'validation_rows': val_count,
                     'fallback_reason': 'Sparse contract or candidate failed validation guard' if best_state is None else None}
        atomic(output / 'selection_frozen.json', selection)
        completed[name] = best_summary; baseline_rows += prior_rows; selected_rows += best_rows
        atomic(OUT / ('probe_summary.json' if args.probe else 'summary.json'), {'state': 'probe_complete' if args.probe else 'training', 'topologies_complete': len(completed),
               'baseline': summarize(baseline_rows), 'selected': summarize(selected_rows), 'jobs': completed, 'test_used': False})
        print(name, 'baseline', prior_summary['macro_hit5'], 'selected', best_summary['macro_hit5'], 'weight', best_weight, flush=True)
        del model, x, tx, vx, table; gc.collect()
    path = OUT / ('probe_summary.json' if args.probe else 'summary.json'); result = read(path)
    result.update(state='probe_complete' if args.probe else 'validation_complete', seconds=time.monotonic() - start)
    atomic(path, result); atomic(OUT / 'status.json', {'state': result['state'], 'topologies_complete': len(completed), 'updated_at': now()})


if __name__ == '__main__': main()
