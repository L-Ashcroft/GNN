"""Evaluate every pre-frozen personal readout, never choose a model on test."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import gc
import sys
import time
import numpy as np
from core import ROOT, HERE, OUT, atomic, freeze, gate, metrics, summarize, read, sha, now

V2 = ROOT / 'worktrees/personal_refinement_v2'
sys.path.insert(0, str(V2))
from common import CACHE, PARENT_OUTPUT, OUTPUT, setup_model_runtime
from predict import Predictor
from features import Inputs

DEST = OUT / 'personal_test'


def recipe():
    suite = read(V2 / 'suite.json')
    paths = [V2 / 'suite.json', CACHE / 'manifest.json']
    paths += list(V2.glob('*.py')) + list((ROOT / 'worktrees/personal_transfer_v1').glob('*.py'))
    paths += list((ROOT / 'versions/rgcn_calibrated_head_v1/src').rglob('*.py'))
    parents = {j['parent'] for j in suite['jobs'] if j['parent']}
    for job in suite['jobs']:
        folder = OUTPUT / job['id']
        selected = read(folder / 'selection_frozen.json')
        assert sha(folder / 'best.npz') == selected['best_checkpoint_sha256']
        paths += [folder / n for n in ['recipe.json', 'selection_frozen.json', 'best.npz']]
    for parent in sorted(parents):
        folder = PARENT_OUTPUT / parent
        assert sha(folder / 'best.pt') == read(folder / 'selection_frozen.json')['checkpoint_sha256']
        paths += [folder / n for n in ['recipe.json', 'normalization.json', 'selection_frozen.json', 'best.pt']]
    paths += [HERE / n for n in ['core.py', 'personal_test.py']]
    return {'protocol': 'all_18_pre_frozen_models_test_v1', 'jobs': suite['jobs'],
            'sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(paths))},
            'selection': 'All 18 jobs already frozen by validation; no test-based selection or further tuning',
            'test_caveat': 'Same fixed 2000 samples previously evaluated by v1; not a newly blind test',
            'cache_policy': 'New test embeddings only; existing train/validation caches untouched',
            'dropout': 'eval mode; deterministic CPU FP32 backbone; unchanged FP64 readout'}


def embedding(parent, reader, model_module, smoke=False):
    import torch
    folder = DEST / 'embeddings' / parent
    source = PARENT_OUTPUT / parent / 'best.pt'
    if (folder / 'manifest.json').exists() and not smoke:
        m = read(folder / 'manifest.json')
        assert m['parent_sha256'] == sha(source)
        assert all(sha(ROOT / p) == h for p, h in m['files'].items())
        return folder
    gate()
    stats = read(PARENT_OUTPUT / parent / 'normalization.json')
    model = model_module.TargetRGCN(reader.schema, reader.tasks, stats)
    state = torch.load(source, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'], strict=True); del state
    model.eval()
    # Validation probe tests the independent loader against the frozen cache.
    selected = reader.refs['validation'][:16]
    with torch.inference_mode():
        b = reader.batch(selected, 'validation'); got = model(b, return_states=True)['embedding'].numpy()
    expected = []
    for ref in selected:
        item = reader.index[int(ref)]
        mmap = np.load(ROOT / 'data/model_ready/personal_refinement_v2' / parent / item['topology_id'] / 'embedding.npy', mmap_mode='r')
        expected.append(mmap[item['local']].copy()); mmap._mmap.close()
    delta = float(np.max(np.abs(got - expected)))
    np.testing.assert_allclose(got, expected, atol=2e-5, rtol=2e-5)
    if smoke:
        return {'validation_embedding_max_absolute_error': delta, 'samples': len(selected)}
    folder.mkdir(parents=True, exist_ok=True)
    maps = {name: np.full((2000, 128), np.nan, dtype=np.float32) for name in reader.structures}
    refs = reader.refs['test']
    for start in range(0, len(refs), 16):
        gate(.7)
        chosen = refs[start:start + 16]
        with torch.inference_mode():
            b = reader.batch(chosen, 'test'); value = model(b, return_states=True)['embedding'].numpy()
        for i, ref in enumerate(chosen):
            item = reader.index[int(ref)]
            maps[item['topology_id']][item['local']] = value[i]
        if start % 256 == 0:
            atomic(DEST / 'status.json', {'state': 'extracting_test_embeddings', 'parent': parent,
                   'samples_done': start + len(chosen), 'role_samples': len(refs), 'updated_at': now()})
    files = {}
    for name, value in maps.items():
        # Only declared test positions are populated.
        local = [r['local'] for r in reader.index if r['topology_id'] == name and r['role'] == 'test']
        assert np.isfinite(value[local]).all() and len(local) == 200
        assert int(np.isfinite(value[:, 0]).sum()) == 200
        p = folder / (name + '.npy'); np.save(p, value, allow_pickle=False)
        files[p.relative_to(ROOT).as_posix()] = sha(p)
    atomic(folder / 'manifest.json', {'parent_sha256': sha(source), 'files': files,
           'validation_embedding_max_absolute_error': delta, 'role': 'test_only', 'rows': 2000})
    del model, maps; gc.collect()
    return folder


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    gate()
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    DEST.mkdir(parents=True, exist_ok=True)
    frozen = recipe()
    # Freeze all weights/recipes BEFORE opening the test reader or test labels.
    if not args.smoke:
        freeze(DEST / 'evaluation_frozen.json', frozen)
    data_module, model_module = setup_model_runtime()
    import torch
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    reader = data_module.Reader(verify=True)
    parents = list(dict.fromkeys(j['parent'] for j in frozen['jobs'] if j['parent']))
    if args.smoke:
        value = embedding(parents[0], reader, model_module, smoke=True)
        atomic(DEST / 'checks.json', {'passed': True, 'test_labels_read': False, **value})
        print(value, flush=True); return
    for parent in parents:
        embedding(parent, reader, model_module)
    import pyarrow as pa
    import pyarrow.parquet as pq
    results = {}; started = time.monotonic()
    for job in frozen['jobs']:
        result_path = DEST / job['id'] / 'metrics.json'
        if result_path.exists():
            results[job['id']] = read(result_path)['summary']; continue
        gate(.7)
        predictor = Predictor(job['id']); rows = []; predictions = []
        for name, arrays in reader.arrays.items():
            refs = [r for r in reader.index if r['role'] == 'test' and r['topology_id'] == name]
            local = [r['local'] for r in refs]
            emb = np.load(DEST / 'embeddings' / job['parent'] / (name + '.npy'))[local] if job['parent'] else None
            p, available = predictor(name, arrays['device'][local], arrays['conditions'][local], emb)
            y = arrays['target'][local]; mask = arrays['mask'][local]
            assert not np.any(mask & ~available), 'An observed test metric lacks a trained output'
            rules = reader.tasks[reader.structures[name]['group']]['slots']
            rows += metrics(y, p, mask, rules, name)
            for i, ref in enumerate(refs):
                predictions.append({'sample_id': ref['sample_id'], 'topology_id': name,
                                    'prediction': p[i].tolist(), 'target': y[i].tolist(), 'mask': mask[i].tolist()})
        assert len(predictions) == len({p['sample_id'] for p in predictions}) == 2000
        result_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(predictions), result_path.parent / 'predictions.parquet', compression='zstd')
        summary = summarize(rows); results[job['id']] = summary
        atomic(result_path, {'split': 'fixed_test', 'sample_count': 2000, 'summary': summary, 'per_target': rows,
                            'selected_before_test': True, 'evaluation_recipe_sha256': sha(DEST / 'evaluation_frozen.json')})
        atomic(DEST / 'status.json', {'state': 'evaluating_frozen_readouts', 'jobs_complete': len(results), 'total': 18,
                                    'last_job': job['id'], 'updated_at': now()})
        print(job['id'], summary, flush=True)
    atomic(DEST / 'summary.json', {'state': 'complete', 'jobs': results, 'test_samples_per_job': 2000,
                                 'test_used_for_model_selection': False, 'readout_evaluation_seconds': time.monotonic() - started,
                                 'completed_at': now()})
    atomic(DEST / 'status.json', {'state': 'complete', 'jobs_complete': 18, 'updated_at': now()})


if __name__ == '__main__':
    main()
