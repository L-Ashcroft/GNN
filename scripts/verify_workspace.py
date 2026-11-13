"""Bounded CPU checks for retained research inputs. Never infer on sealed test data."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'monitoring/workspace_cleanup_v2'
sys.path.insert(0, str(ROOT))
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def record(stage, value):
    DEST.mkdir(exist_ok=True)
    with sqlite3.connect(DEST / 'inventory.sqlite') as db:
        db.execute('CREATE TABLE IF NOT EXISTS documents(name TEXT PRIMARY KEY, json TEXT NOT NULL)')
        db.execute('INSERT OR REPLACE INTO documents VALUES(?,?)',
                   ('validation_' + stage, json.dumps(value, ensure_ascii=False, allow_nan=False)))


def pins():
    union = {}
    for path, key in [('monitoring/public_evidence_queue_v1/policy.json', 'source_sha256'),
                      ('monitoring/controlled_evidence_v1/policy.json', 'pins')]:
        union.update(read(ROOT / path)[key])
    for p, h in union.items():
        assert sha(ROOT / p) == h, p
    for name in ('rgcn_memory_safe_loto_v2', 'rgcn_memory_safe_id_control_v2'):
        recipe = read(ROOT / 'artifacts' / name / 'recipe.json')
        for path, h in recipe['supported_runtime_SHA256'].items():
            assert sha(ROOT / path) == h, path
    return {'source_pins': len(union), 'CPU_and_CUDA_version_receipts_unchanged': True}


def data():
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1)
    counts = Counter(); tops = Counter(); ids = set(); source_paths = set(); graph_paths = set()
    manifest = read(ROOT / 'data/fusion_v2/manifest.json')
    for path, expected in manifest['outputs_sha256'].items():
        assert sha(ROOT / path) == expected, path
    assert sha(ROOT / 'data/fusion_v2/index.parquet') == manifest['index_sha256']
    # Read every column and every row, but release each Arrow batch immediately.
    for p in sorted((ROOT / 'data/fusion_v2/sampled').glob('*.parquet')):
        for b in pq.ParquetFile(p).iter_batches(batch_size=256, use_threads=False):
            for row in b.to_pylist():
                assert row['sample_id'] not in ids
                ids.add(row['sample_id']); counts[row['source_dataset']] += 1; tops[row['topology_id']] += 1
                y, mask = row['performance'], row['mask']
                assert len(y) == len(mask) == len(row['semantic_ids']) == 16
                assert all((v is not None and np.isfinite(v)) if m else v is None for v, m in zip(y, mask))
                assert all(bool(s) == bool(m) for s, m in zip(row['semantic_ids'], mask))
                assert len(row['parameters']) == len(row['parameter_names']) == len(row['parameter_units'])
                assert np.isfinite(row['parameters']).all()
                graph_paths.add(row['graph']); source_paths.add(row['source_path'])
    assert counts == {'FALCON': 200000, 'AICircuit': 100000, 'LOADBench': 200000}
    assert len(ids) == 500000 and len(tops) == 121
    for p in graph_paths | source_paths:
        assert (ROOT / p).is_file(), p
    protocol = read(ROOT / 'data/fusion_v2/splits/verified_loto_full_v2/manifest.json')
    split = ROOT / protocol['source_split']
    assert sha(split) == protocol['source_split_sha256']
    role_ids = defaultdict(set); design_roles = {}; observed = 0
    for b in pq.ParquetFile(split).iter_batches(batch_size=2048, use_threads=False):
        for row in b.to_pylist():
            assert row['sample_id'] in ids
            role = row['role']; assert row['sample_id'] not in role_ids[role]
            role_ids[role].add(row['sample_id']); observed += 1
            if row['design_group'] in design_roles:
                assert design_roles[row['design_group']] == role
            design_roles[row['design_group']] = role
    assert observed == 500000 and sum(map(len, role_ids.values())) == len(set.union(*role_ids.values()))
    id_counts = {k: len(v) for k, v in role_ids.items()}
    assert id_counts == {'train': 400083, 'validation': 50861, 'test': 49056}
    del role_ids, design_roles, ids; gc.collect()
    fold_rows = []
    for fold in protocol['folds']:
        p = ROOT / fold['path']; assert sha(p) == fold['sha256']
        held = {protocol['topologies'].index(t) for t in fold['test_topologies'] + fold['excluded_structural_aliases']}
        packed = []
        with np.load(p, allow_pickle=False) as arrays:
            for role in ('train', 'validation', 'test', 'unused'):
                key = role + '_references'
                if key not in arrays:
                    assert fold['counts'].get(role, 0) == 0; continue
                a = arrays[key]
                assert len(a) == fold['counts'][role]
                if role in ('train', 'validation'):
                    assert not set(np.unique(a[:, 0])) & held
                packed.append(a[:, 0].astype('int64') * 10000000 + a[:, 1])
        joined = np.concatenate(packed)
        assert len(joined) == len(np.unique(joined)) == 500000
        fold_rows.append({'fold': fold['fold_id'], 'counts': fold['counts']})
    cache = read(ROOT / 'data/model_ready/fusion_v2_training_v1/manifest.json')
    cache_rows = 0
    for top, spec in cache['topologies'].items():
        folder = ROOT / spec['folder']
        for filename, h in spec['cache_sha256'].items():
            assert sha(folder / filename) == h, (top, filename)
        y = np.load(folder / 'target.npy', mmap_mode='r'); m = np.load(folder / 'mask.npy', mmap_mode='r')
        assert y.shape == m.shape and y.shape[1] == 16
        for i in range(0, len(y), 512):
            assert np.isfinite(y[i:i+512][m[i:i+512]]).all()
        cache_rows += len(y); y._mmap.close(); m._mmap.close()
    assert cache_rows == 500000
    personal = ROOT / 'data/generated/analoggym_personal_20k_v1'
    snapshot = read(personal / 'dataset.json'); personal_ids = set(); personal_tops = Counter(); archives = {}
    for shard in snapshot['shards']:
        n = 0
        for b in pq.ParquetFile(ROOT / shard['path']).iter_batches(batch_size=128, use_threads=False):
            for row in b.to_pylist():
                sid = row['sample_id']; assert sid not in personal_ids
                personal_ids.add(sid); personal_tops[row['topology_id']] += 1; n += 1
                assert row['generation_status'] == 'measured_standalone_not_merged'
                assert len(row['mask']) == len(row['performance']) == len(row['semantic_ids']) <= 16
                assert all((v is not None and np.isfinite(v)) if m else v is None for v, m in zip(row['performance'], row['mask']))
                assert (ROOT / row['graph']).exists() and (ROOT / row['source_path']).exists()
                archives[row['source_path']] = row['source_sha256']
        assert n == shard['rows']
    assert len(personal_ids) == 20000 and len(personal_tops) == 10 and set(personal_tops.values()) == {2000}
    for i, (p, h) in enumerate(archives.items(), 1):
        assert sha(ROOT / p) == h, p
        if i % 5000 == 0:
            print('Verified personal original archives:', i, flush=True)
    pm = read(ROOT / 'data/model_ready/personal_transfer_v1/manifest.json')
    for p, h in pm['files'].items():
        assert sha(ROOT / p) == h, p
    pmeta = read(ROOT / 'data/model_ready/analoggym_personal_20k_v1/manifest.json')
    psplit = ROOT / pmeta['split']['path'] if isinstance(pmeta['split'], dict) else ROOT / pmeta['split']
    # The exact training split is independently pinned in the transfer manifest.
    pcounts = Counter()
    for b in pq.ParquetFile(psplit).iter_batches(batch_size=2048, use_threads=False):
        for row in b.to_pylist():
            assert row['sample_id'] in personal_ids
            pcounts[row['role']] += 1
    assert dict(pcounts) == pm['split_counts']
    assert sha(psplit) == pm['split_sha256']
    return {'public_rows': 500000, 'public_sources': dict(counts), 'public_topology_entries': len(tops),
            'public_ID_counts': id_counts, 'LOTO_protocols_checked': fold_rows, 'raw_training_cache_rows': cache_rows,
            'personal_rows': len(personal_ids), 'personal_topologies': dict(personal_tops),
            'personal_split_counts': dict(pcounts), 'personal_archives_SHA256_verified': len(archives),
            'public_all_columns_read': True, 'missingness_and_finite_observed_labels_checked': True}


def models():
    import psutil
    assert psutil.virtual_memory().available >= 1.5 * 2**30, 'Defer CPU smoke checks: insufficient memory'
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    directory = ROOT / 'worktrees/controlled_evidence_v1'
    sys.path.insert(0, str(directory))
    import runtime
    import torch
    import numpy as np
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    from raw_data import FoldData
    from portable import FoldRGCN, loss_by_contract
    from src.train.stage import build_optimizer
    results = []; resume_check = {}; reader = FoldData('paired_ID_same_neural_architecture_seed42', 1, verify=False)
    assert len(reader.tasks) == 129
    indices = np.arange(2); batch = reader.batch('train', indices)
    for name in ('rgcn_memory_safe_id_control_v2', 'rgcn_memory_safe_no_film_v2',
                 'rgcn_memory_safe_static_edges_v2', 'rgcn_memory_safe_collapsed_relations_v2'):
        cfg = read(ROOT / 'worktrees' / name / 'config.json')
        stats = read(ROOT / 'artifacts' / name / 'normalization.json')
        assert stats['proof']['test_or_validation_labels_used'] is False
        assert stats['proof']['training_rows'] == len(reader.refs['train'])
        torch.manual_seed(42)
        model = FoldRGCN(reader.schema, reader.tasks, stats, **{k: cfg[k] for k in
                ('hidden_dim', 'num_layers', 'dropout', 'update_edges', 'use_domain_emb', 'use_film', 'collapse_relations')})
        assert model.use_film == cfg['use_film']
        assert model.collapse_relations == cfg['collapse_relations']
        optimizer = build_optimizer(model, 'finetune_all', cfg['adaptation_lr'], cfg['backbone_lr'], cfg['weight_decay'])
        if name == 'rgcn_memory_safe_id_control_v2':
            saved = torch.load(ROOT / 'artifacts' / name / 'last.pt', map_location='cpu', weights_only=True)
            required = {'model', 'optimizer', 'scheduler', 'rng_cpu', 'rng_cuda', 'next_epoch', 'next_update', 'partial', 'recipe_digest'}
            assert required <= set(saved)
            recipe = read(ROOT / 'artifacts' / name / 'recipe.json')
            assert saved['recipe_digest'] == hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
            model.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
            scheduler.load_state_dict(saved['scheduler']); torch.set_rng_state(saved['rng_cpu'])
            assert torch.equal(torch.get_rng_state(), saved['rng_cpu'])
            resume_check = {'recipe_digest_verified': True, 'weights_optimizer_scheduler_RNG_loaded': True,
                            'saved_epoch_index': saved['next_epoch'], 'saved_update_cursor': saved['next_update']}
            del saved, scheduler
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss, _ = loss_by_contract(model(batch), batch, model.head)
        assert torch.isfinite(loss); loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step(); model.eval()
        with torch.no_grad():
            expected = model(batch)
        temporary = DEST / 'smoke_state.pt'
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': torch.get_rng_state()}, temporary)
        saved = torch.load(temporary, map_location='cpu', weights_only=True)
        model.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
        with torch.no_grad():
            torch.testing.assert_close(expected, model(batch), rtol=0, atol=0)
        temporary.unlink()
        results.append({'id': name, 'parameters': sum(p.numel() for p in model.parameters()),
                        'loss': float(loss.detach()), 'AdamW_backward_checkpoint_roundtrip': True})
        del saved, model, optimizer, loss, expected; gc.collect()
    reader.close(); del batch, reader; gc.collect()
    # Frozen M04 formal checkpoint, forward on TRAIN rows only.
    folder = ROOT / 'artifacts/experiment2_loto_v1/LB_M04__hold_SimpleOpAmp4'
    cfg = read(folder / 'recipe.json')['config']; stats = read(folder / 'normalization.json')
    # M04 belongs to the original verified protocol, not the later full-pool one.
    # Override the reader's config only inside this independent check process.
    import raw_data
    original_read = raw_data.read
    raw_data.read = lambda p: cfg if Path(p).resolve() == directory / 'config.json' else original_read(p)
    try:
        reader = FoldData(cfg['fold_id'], 1, verify=False)
    finally:
        raw_data.read = original_read
    model = FoldRGCN(reader.schema, reader.tasks, stats, **{k: cfg[k] for k in
            ('hidden_dim', 'num_layers', 'dropout', 'update_edges', 'use_domain_emb', 'use_film', 'collapse_relations')})
    saved = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model'], strict=True); model.eval()
    with torch.no_grad():
        assert torch.isfinite(model(reader.batch('train', np.arange(1)))).all()
    results.append({'id': 'frozen_M04_checkpoint', 'strict_load_and_TRAIN_forward': True})
    reader.close(); del model, reader, saved; gc.collect()
    from data import ScopedData
    from model import make_model, loss_by_contract
    # Stats fitted only on the same tiny real TRAIN subsets as prior preflight.
    for job in read(directory / 'suite.json')['jobs']:
        cfg = read(ROOT / job['config']); reader = ScopedData(cfg, verify=False)
        stats = read(ROOT / 'artifacts/controlled_evidence_v1/diagnostics' / (cfg['id'] + '_v2_TRAIN_subset_stats.json'))
        batch = reader.batch('train', np.arange(2)); torch.manual_seed(42)
        model = make_model(reader, stats, cfg)
        optimizer = build_optimizer(model, 'finetune_all', 3e-4, 2e-4, .01)
        loss, _ = loss_by_contract(model(batch), batch, model.head)
        assert torch.isfinite(loss); loss.backward(); optimizer.step()
        results.append({'id': cfg['id'], 'parameters': sum(p.numel() for p in model.parameters()),
                        'two_real_TRAIN_rows_forward_backward_AdamW': True})
        reader.close(); del model, optimizer, batch, loss, reader; gc.collect()
    # Retain and verify all 15 selected few-shot/full neural models; no target
    # test inference and no refitting target-domain statistics.
    module('common', 'worktrees/personal_transfer_v1/common.py')
    pdata = module('cleanup_personal_data', 'worktrees/personal_transfer_v1/data.py')
    pmodel = module('cleanup_personal_model', 'worktrees/personal_transfer_v1/model.py')
    reader = pdata.Reader(verify=False)
    refs = [next(i for i in reader.refs['train'] if reader.index[int(i)]['topology_id'] == top)
            for top in ('Alfio_RAFFC_Pin_3', 'ldo_simple')]
    batch = reader.batch(refs, 'train'); private_checked = []
    for base in ('artifacts/experiment4_personal_transfer_v1', 'artifacts/research_repair_personal_repeats_v1'):
        for p in sorted((ROOT / base).glob('*/best.pt')):
            stats = read(p.parent / 'normalization.json')
            model = pmodel.TargetRGCN(reader.schema, reader.tasks, stats)
            saved = torch.load(p, map_location='cpu', weights_only=True)
            model.load_state_dict(saved['model'], strict=True); model.eval()
            with torch.no_grad():
                assert torch.isfinite(model(batch)).all()
            private_checked.append(p.relative_to(ROOT).as_posix())
            del model, saved; gc.collect()
    for arrays in reader.arrays.values():
        for a in arrays.values():
            a._mmap.close()
    assert len(private_checked) == 15
    return {'jobs': results, 'test_batches_inferred': 0, 'official_artifacts_modified': False,
            'private_selected_checkpoints_strict_load_TRAIN_forward': private_checked,
            'actual_paused_ID_resume_state': resume_check,
            'supervision_routes': 129, 'CPU_only': True, 'RSS_MiB': psutil.Process().memory_info().rss / 2**20}


def metrics():
    import numpy as np
    import pyarrow.parquet as pq
    reference = module('cleanup_falcon_evaluator', 'scripts/evaluate_falcon_reference.py')
    from src.falcon_reference_metrics import FalconReferenceAccumulator, regression, LOG_UNITS
    config = read(ROOT / 'config/falcon_reference_eval_v1.json'); mapping = read(ROOT / 'data/fusion_v2/performance_mapping.json')['groups']
    results = []
    for job in config['jobs']:
        if job['id'] not in ('public_id_published', 'loto_M04', 'personal_full_joint'):
            continue
        assert reference.ready(job)
        tasks = read(ROOT / job['tasks']); acc = FalconReferenceAccumulator(); physical = {}
        structures = {t: v['group'] for t, v in read(ROOT / job['topology_groups']).items()} if job.get('topology_groups') else {}
        topology_groups = defaultdict(list)
        for g, task in tasks.items():
            topology_groups[task.get('contract', {}).get('topology_id')].append(g)
        for p, spec in reference.inputs_for(job):
            default = p.stem if spec.get('topology') == 'file_stem' else spec.get('topology')
            for b in pq.ParquetFile(p).iter_batches(batch_size=256, use_threads=False):
                groups = defaultdict(list)
                for row in b.to_pylist():
                    top = row.get('topology_id') or default
                    group = row.get('supervision_group') or structures.get(top)
                    if not group:
                        assert len(topology_groups[top]) == 1
                        group = topology_groups[top][0]
                    contract = tasks[group]['contract']; top = top or contract.get('topology_id')
                    source = row.get('source_dataset') or contract.get('source_dataset') or contract.get('source')
                    groups[source, top, group].append(row)
                for (source, top, group), rows in groups.items():
                    rules = json.loads(json.dumps(tasks[group]['slots']))
                    for rule in rules.values():
                        candidates = {(m['source_unit'], float(m['factor'])) for m in mapping.get(top, [])
                                      if m['original_name'] == rule['original_name'] and m['unit'] == rule['unit']}
                        rule['native_unit'] = rule['unit']; rule['native_multiplier'] = 1.
                        if candidates:
                            assert len(candidates) == 1
                            unit, factor = next(iter(candidates)); rule['native_unit'] = unit; rule['native_multiplier'] = 1 / factor
                    acc.update([x['prediction'] for x in rows], [x['target'] for x in rows], [x['mask'] for x in rows],
                               [x['sample_id'] for x in rows], source, top, group, rules)
                    for slot, rule in rules.items():
                        j = int(slot); valid = [x for x in rows if x['mask'][j]]
                        if not valid:
                            continue
                        key = (top if source == 'AnalogGym_generated' else '', group, rule['semantic_id'])
                        item = physical.setdefault(key, {'rule': rule, 'parts': []})
                        item['parts'].append((np.array([x['target'][j] for x in valid]), np.array([x['prediction'][j] for x in valid])))
        fresh, target_rows, _, _ = acc.finish()
        old = read(ROOT / 'artifacts/falcon_reference_eval_v1' / job['id'] / 'summary.json')
        for key in ('mean_pct', 'median_pct', 'p95_pct'):
            assert np.isclose(fresh['overall']['untrimmed_sample_mre'][key], old['overall']['untrimmed_sample_mre'][key], rtol=1e-12, atol=1e-8)
        assert fresh['overall']['samples'] == old['overall']['samples']
        def task_key(x):
            return (x['source_dataset'], x['topology_id'], x['supervision_group'], x['semantic_id'])
        previous = {task_key(x): x for x in pq.read_table(ROOT / 'artifacts/falcon_reference_eval_v1' / job['id'] / 'per_target_metrics.parquet').to_pylist()}
        assert len(previous) == len(target_rows)
        max_physical_difference = 0.
        for x in target_rows:
            y = previous[task_key(x)]
            for scope in ('reference_nonzero', 'all_observed'):
                for field in ('mae', 'rmse', 'r2_mathematical'):
                    a, b = x[scope][field], y[scope][field]
                    if a is None or b is None:
                        assert a is None and b is None
                    else:
                        assert np.isclose(a, b, rtol=1e-12, atol=1e-10)
                        max_physical_difference = max(max_physical_difference, abs(a-b))
        task_stats = []
        for item in physical.values():
            y = np.concatenate([a for a, _ in item['parts']]); p = np.concatenate([b for _, b in item['parts']])
            rule = item['rule']; valid = (np.abs(y) > 1e-8 * rule['scale']) & (rule['unit'] not in LOG_UNITS)
            relative = np.abs(p[valid]-y[valid]) / np.abs(y[valid])
            task_stats.append({'hit5': float(np.mean(relative <= .05)) if len(relative) else None,
                               'hit10': float(np.mean(relative <= .1)) if len(relative) else None,
                               'r2': regression(y, p)['r2_mathematical']})
        def avg(field):
            values = [x[field] for x in task_stats if x[field] is not None]
            return float(np.mean(values)) if values else None
        r2 = [x['r2'] for x in task_stats if x['r2'] is not None]
        macro = {'macro_hit5': avg('hit5'), 'macro_hit10': avg('hit10'), 'macro_r2': avg('r2'),
                 'median_r2': float(np.median(r2)), 'negative_r2_tasks': sum(x < 0 for x in r2)}
        if job['id'] == 'public_id_published':
            baseline = read(ROOT / 'artifacts/loadbench_remediation_v1/combined_fixed_test_summary.json')
        elif job['id'] == 'personal_full_joint':
            baseline = read(ROOT / 'artifacts/research_repair_v1/personal_test/full_joint/metrics.json')['summary']
        else:
            baseline = read(ROOT / 'artifacts/experiment2_loto_v1/LB_M04__hold_SimpleOpAmp4/heldout_test_metrics.json')
        for field in ('macro_hit5', 'macro_r2', 'median_r2'):
            assert np.isclose(macro[field], baseline[field], rtol=1e-7, atol=1e-9), (job['id'], field, macro[field], baseline[field])
        results.append({'job': job['id'], 'samples': fresh['overall']['samples'],
                        'MRE_median_P95_recalculated_match': True,
                        'per_metric_MAE_RMSE_R2_match': True, 'max_physical_metric_absolute_difference': max_physical_difference,
                        'task_macro_recomputed': macro, **fresh['overall']['untrimmed_sample_mre']})
        del acc; gc.collect()
    assessment = module('cleanup_controlled_assessment', 'worktrees/controlled_evidence_v1/assessment.py')
    folder = ROOT / 'artifacts/experiment2_loto_v1/LB_M04__hold_SimpleOpAmp4'
    rows, tasks = assessment.load(folder); fresh = assessment.metrics(rows, tasks)[0]
    old = read(folder / 'heldout_test_metrics.json')
    for key in ('macro_hit5', 'macro_hit10', 'macro_r2', 'median_r2'):
        assert np.isclose(fresh['task_macro'][key], old[key], rtol=1e-10, atol=1e-10), key
    return {'frozen_reanalyses': results, 'M04_task_macro_recomputed': fresh['task_macro'],
            'unsealed_tests_accessed': False, 'new_predictions_generated': False}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', choices=['data', 'models', 'metrics', 'pins', 'all'], default='all')
    a = parser.parse_args()
    if a.stage == 'all':
        for stage in ('pins', 'data', 'models', 'metrics'):
            result = subprocess.call([sys.executable, '-X', 'utf8', '-B', str(Path(__file__).resolve()), '--stage', stage], cwd=ROOT)
            if result:
                return result
        return 0
    tick = time.monotonic()
    try:
        result = globals()[a.stage]()
    except Exception as e:
        record(a.stage, {'passed': False, 'error': repr(e), 'at': datetime.now(timezone.utc).isoformat()})
        raise
    result.update(passed=True, seconds=time.monotonic()-tick, at=datetime.now(timezone.utc).isoformat())
    record(a.stage, result); print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
