"""Validate interface, freeze ALL candidates, then one historical fixed test."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[key] = '1'
from pathlib import Path
import sys
import time
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[2]; HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'worktrees/rgcn_accuracy_v3'))
from common import read, sha, atomic, freeze, now, RAW, META
sys.path.insert(0, str(HERE))
from inference import MetricReadout
sys.path.insert(0, str(ROOT / 'worktrees/research_repair_v1'))
from core import metrics, summarize
sys.path.insert(0, str(ROOT / 'worktrees/loadbench_bounds_inference_v1'))
from interface import BoundedPhysicalReadout
OUT = ROOT / 'artifacts/rgcn_metric_v4'


def main():
    # CPU-only inference: microbatch16, one mapped topology, no optimizer.
    # The independent real-graph qualification measured ~341MiB process RSS.
    # Retain >=640MiB guard during evaluation; CUDA still requires4GiB.
    if psutil.virtual_memory().available < 1.1 * 2**30: raise SystemExit(75)
    assert read(OUT / 'summary.json')['state'] == 'validation_complete'
    # Every topology and output choice is frozen before opening TEST labels.
    recipe = read(OUT / 'recipe.json'); paths = [OUT / 'recipe.json', HERE / 'evaluate.py', HERE / 'inference.py', HERE / 'train.py']
    for name in recipe['topologies']:
        paths += [OUT / name / f for f in ['selection_frozen.json', 'model.npz', 'normalization.json']]
    freeze(OUT / 'fixed_test/evaluation_frozen.json', {'sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
           'test_policy': 'All validation-selected models, no test-based ranking, tuning or replacement',
           'historical_test_reused': True, 'not_an_independent_blind_test': True})
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    fm = read(META / 'manifest.json'); manifest = read(RAW / 'manifest.json'); tasks = read(RAW / 'tasks.json')
    meta = {k: np.load(META / 'validation' / (k + '.npy'), mmap_mode='r') for k in ['embedding', 'refs', 'group']}
    model = MetricReadout(); bounds = BoundedPhysicalReadout(); maximum = 0.; checked = 0
    for name in recipe['topologies']:
        selected = read(OUT / name / 'selection_frozen.json'); g = selected['group']; code = fm['groups'].index(g)
        ids = np.flatnonzero(np.asarray(meta['group']) == code); pos = np.asarray(meta['refs'][ids, 1], dtype=int)
        table = pq.read_table(ROOT / 'data/fusion_v2/sampled' / (name + '.parquet'), columns=['parameters', 'parameter_names'])
        params = np.array(table['parameters'].to_pylist())[pos]; names = table['parameter_names'][0].as_py()
        basefile = ROOT / 'artifacts/loadbench_remediation_v1' / (g + '__accepted.npy')
        if not basefile.exists(): basefile = basefile.with_name(g + '__published.npy')
        z = np.load(basefile); state = model.bank(name)[0]; base = bounds(g, np.sign(z) * np.expm1(abs(z)) * state['scales'])
        actual = model(name, params, meta['embedding'][ids], base, parameter_names=names)
        expected = np.load(OUT / name / 'validation_prediction.npy')
        delta = float(abs(actual - expected).max()); maximum = max(maximum, delta)
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-7); checked += len(ids)
    atomic(OUT / 'interface_checks.json', {'passed': True, 'real_validation_rows': checked, 'max_absolute_error': maximum,
           'rtol': 1e-9, 'atol': 1e-7, 'test_used_for_check': False})
    model.cache.clear(); del meta
    # Reuse the encoder only. Its validation feature replay was checked in v3.
    sys.path.insert(0, str(ROOT / 'versions/rgcn_local_id_v1'))
    sys.path.insert(0, str(ROOT / 'worktrees/autonomous_training_v1'))
    from model import load_model
    from stream_graphs import StreamGraphs
    import torch
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    net, _ = load_model(); reader = StreamGraphs(ROOT, 'data/model_ready/fusion_v2_training_v1',
                        'data/model_ready/fusion_v2_frozen_e19_bf0d71ad', 'versions/rgcn_calibrated_head_v1', capacity=1, verify=True)
    encoder = net.encoder; encoder.eval(); before = []; after = []; total = 0; begin = time.monotonic()
    for name in recipe['topologies']:
        if (OUT / 'STOP').exists() or psutil.virtual_memory().available < 640 * 2**20: raise SystemExit(75)
        selected = read(OUT / name / 'selection_frozen.json'); g = selected['group']; rules = tasks[g]['slots']
        output = OUT / 'fixed_test' / (name + '.parquet'); record_path = output.with_suffix('.json')
        if record_path.exists():
            old = read(record_path); assert sha(output) == old['prediction_sha256']
            before += old['baseline_rows']; after += old['selected_rows']; total += old['samples']; continue
        basepath = ROOT / 'artifacts/research_repair_v1/loadbench_shared_v2/fixed_test' / (name + '.parquet')
        rows = pq.read_table(basepath).to_pylist(); idmap = {r['sample_id']: i for i, r in enumerate(rows)}
        y = np.array([r['target'] for r in rows]); mask = np.array([r['mask'] for r in rows], dtype=bool)
        prior = np.array([r['prediction'] for r in rows]); prediction = prior.copy()
        table = pq.read_table(ROOT / 'data/fusion_v2/sampled' / (name + '.parquet'), columns=['sample_id', 'parameters', 'parameter_names'])
        all_ids = table['sample_id'].to_pylist(); positions = np.array([all_ids.index(r['sample_id']) for r in rows], dtype=int)
        params = np.array(table['parameters'].to_pylist())[positions]; names = table['parameter_names'][0].as_py()
        roles = np.load(ROOT / manifest['topologies'][name]['folder'] / 'role.npy', mmap_mode='r')
        assert np.all(roles[positions] == 2); roles._mmap.close()
        if any(c['mode'] != 'identity' for c in selected['choices'].values()):
            for start in range(0, len(rows), 16):
                ix = positions[start:start + 16]
                with torch.inference_mode():
                    batch = reader.batch_refs([(name, int(i)) for i in ix], 'test')
                    e = encoder(batch, return_states=True)['embedding'].numpy()
                assert [r['sample_id'] for r in rows[start:start + 16]] == batch.sample_ids
                prediction[start:start + 16] = model(name, params[start:start + 16], e, prior[start:start + 16], parameter_names=names)
        a = metrics(y, prior, mask, rules, name); b = metrics(y, prediction, mask, rules, name)
        for i, row in enumerate(rows): row['prediction'] = prediction[i].tolist()
        pq.write_table(pa.Table.from_pylist(rows), output, compression='zstd')
        record = {'samples': len(rows), 'baseline_rows': a, 'selected_rows': b, 'prediction_sha256': sha(output)}
        atomic(record_path, record); before += a; after += b; total += len(rows)
        atomic(OUT / 'fixed_test/status.json', {'state': 'evaluating_frozen_models', 'topology': name, 'samples_done': total,
               'total': 19301, 'updated_at': now(), 'RSS_MiB': psutil.Process().memory_info().rss / 2**20})
        print(name, 'fixed test', total, flush=True)
    assert total == 19301
    atomic(OUT / 'fixed_test/metrics.json', {'state': 'complete', 'samples': total, 'baseline': summarize(before), 'selected': summarize(after),
           'per_target_before': before, 'per_target_after': after, 'test_based_model_selection': False,
           'original_mask_preserved': True, 'scope': 'Known-topology LOADBench ID, not LOTO', 'seconds': time.monotonic() - begin})
    atomic(OUT / 'fixed_test/status.json', {'state': 'complete', 'samples': total, 'updated_at': now()}); reader.close()


if __name__ == '__main__': main()
