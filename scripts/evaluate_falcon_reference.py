"""Read frozen physical predictions and report FALCON Stage-2 metrics separately.

No model inference, fitting, split changes, or writes to original experiments.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
from pathlib import Path
from datetime import datetime, timezone
import sys
import json
import hashlib
import argparse
import time
import subprocess
import msvcrt
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / 'config/falcon_reference_eval_v1.json'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.analysis.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)

def inputs_for(job):
    paths = []
    for spec in job['inputs']:
        for path in sorted(ROOT.glob(spec['glob'])):
            paths.append((path, spec))
    if not paths:
        return []
    if len({p for p, _ in paths}) != len(paths):
        raise ValueError('Duplicate prediction file in job')
    return paths

def ready(job):
    paths = inputs_for(job)
    if not paths or any(not (ROOT / p).exists() for p in job['proofs']):
        return False
    if job.get('requires_selection_freeze'):
        freeze = read(ROOT / job['proofs'][0])
        if freeze.get('outer_test_used_for_selection') is not False:
            raise ValueError('No verified frozen validation selection')
        if sha(ROOT / job['checkpoint']) != freeze['checkpoint_sha256']:
            raise ValueError('Selected checkpoint does not match freeze record')
    if job.get('expected_files') and len(paths) != job['expected_files']:
        raise ValueError('Unexpected prediction partition count')
    return True

def evaluate(job, config_path, plots=True):
    if not ready(job):
        return False
    if psutil.virtual_memory().available < 1.0 * 2**30:
        raise SystemExit(75)
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.falcon_reference_metrics import FalconReferenceAccumulator
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    tasks = read(ROOT / job['tasks'])
    mapping = read(ROOT / 'data/fusion_v2/performance_mapping.json')['groups']
    structure_groups = {}
    if job.get('topology_groups'):
        structure_groups = {top: item['group'] for top, item in read(ROOT / job['topology_groups']).items()}
    topology_groups = {}
    for group, task in tasks.items():
        top = task.get('contract', {}).get('topology_id')
        if top:
            topology_groups.setdefault(top, []).append(group)
    paths = inputs_for(job)
    dependencies = [p for p, _ in paths] + [ROOT / p for p in job['proofs']]
    dependencies += [ROOT / job['tasks'], ROOT / 'data/fusion_v2/performance_mapping.json',
                     Path(__file__), ROOT / 'src/falcon_reference_metrics.py',
                     ROOT / 'scripts/render_error_distribution.ps1', config_path]
    if job.get('topology_groups'):
        dependencies.append(ROOT / job['topology_groups'])
    pins = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(dependencies))}
    out = ROOT / 'artifacts/falcon_reference_eval_v1' / job['id']
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'evaluation.lock').open('a+b')
    lock.seek(0)
    if not lock.read(1):
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    try:
        if (out / 'summary.json').exists():
            assert read(out / 'receipt.json')['input_sha256'] == pins, 'Analysis inputs changed; use a new profile/version'
            return True
        accumulator = FalconReferenceAccumulator()
        begin = time.perf_counter()
        restored_units = set()
        for path, spec in paths:
            default_topology = spec.get('topology')
            if default_topology == 'file_stem':
                default_topology = path.stem
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                records = batch.to_pylist()
                routed = {}
                for row in records:
                    topology = row.get('topology_id') or default_topology
                    group = row.get('supervision_group')
                    if not group:
                        group = structure_groups.get(topology)
                        if not group:
                            options = topology_groups.get(topology, [])
                            if len(options) != 1:
                                raise ValueError('Cannot unambiguously recover supervision group: ' + str(topology))
                            group = options[0]
                    if group not in tasks:
                        raise ValueError('Prediction group missing from task registry')
                    contract = tasks[group]['contract']
                    source = row.get('source_dataset') or contract.get('source_dataset') or contract.get('source')
                    topology = topology or contract.get('topology_id')
                    if not source or not topology or topology == 'verified_multi_topology':
                        raise ValueError('Actual source/topology required for evaluation')
                    routed.setdefault((source, topology, group), []).append(row)
                for (source, top, group), rows in routed.items():
                    rules = json.loads(json.dumps(tasks[group]['slots']))
                    for rule in rules.values():
                        candidates = [m for m in mapping.get(top, [])
                                      if m['original_name'] == rule['original_name'] and m['unit'] == rule['unit']]
                        rule['native_multiplier'] = 1.
                        rule['native_unit'] = rule['unit']
                        if candidates:
                            possibilities = {(m['source_unit'], float(m['factor'])) for m in candidates}
                            if len(possibilities) != 1:
                                raise ValueError('Ambiguous source-unit restoration')
                            native, factor = next(iter(possibilities))
                            rule['native_multiplier'] = 1. / factor
                            rule['native_unit'] = native
                            if factor != 1.:
                                restored_units.add((source, rule['original_name'], rule['unit'], native, factor))
                    accumulator.update([row['prediction'] for row in rows], [row['target'] for row in rows],
                                       [row['mask'] for row in rows], [row['sample_id'] for row in rows],
                                       source, top, group, rules)
                del records, routed
        summary, target_rows, native_rows, sample_rows = accumulator.finish()
        for row in target_rows + native_rows:
            display, factor = {'W': ('mW', 1000.), 'V': ('mV', 1000.), 'Hz': ('GHz', 1e-9)}.get(row['native_unit'], (row['native_unit'], 1.))
            row['display_unit'], row['display_multiplier'] = display, factor
            for field in ['reference_nonzero', 'all_observed']:
                row[field]['mae_display'] = row[field]['mae'] * factor if row[field]['mae'] is not None else None
                row[field]['rmse_display'] = row[field]['rmse'] * factor if row[field]['rmse'] is not None else None
        if job.get('expected_rows'):
            assert summary['overall']['samples'] == job['expected_rows'], 'Unexpected sample population'
        summary.update(job_id=job['id'], model_description=job['description'], at=now(),
                       analysis_scope=job['scope'], selection_used_this_analysis=False,
                       split_changed=False, masks_changed=False, training_recipe_changed=False,
                       source_unit_restorations=[list(v) for v in sorted(restored_units)])
        pq.write_table(pa.Table.from_pylist(target_rows), out / 'per_target_metrics.parquet', compression='zstd')
        if native_rows:
            pq.write_table(pa.Table.from_pylist(native_rows), out / 'falcon_native_fields.parquet', compression='zstd')
        pq.write_table(pa.Table.from_pylist(sample_rows), out / 'per_sample_errors.parquet', compression='zstd')
        table = ['# ' + job['description'], '', 'Historical fixed-test reanalysis; no model selection or new inference.', '',
                 '| Source | Samples | Full sample MRE (%) | Median (%) | P95 (%) |', '| --- | ---: | ---: | ---: | ---: |']
        for source, row in summary['by_source'].items():
            dist = row['untrimmed_sample_mre']
            table.append(f"| {source} | {row['samples']} | {dist['mean_pct']:.6g} | {dist['median_pct']:.6g} | {dist['p95_pct']:.6g} |")
        if native_rows:
            table += ['', 'FALCON-only original-field pooling, following the reference table convention. This does not merge supervision.', '',
                      '| Metric | Display unit | Nonzero n | R² | RMSE | MAE | Full MRE (%) |',
                      '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
            for row in native_rows:
                q = row['reference_nonzero']
                def fmt(v):
                    return 'undefined' if v is None else f'{v:.6g}'
                table.append(f"| {row['original_name']} | {row['display_unit']} | {row['reference_count']} | {fmt(q['r2'])} | {fmt(q['rmse_display'])} | {fmt(q['mae_display'])} | {fmt(row['mean_relative_error_pct'])} |")
        table += ['', 'Absolute errors above use the explicitly listed display units; relative error uses restored native labels before display conversion. Physical diagnostics including observed zeros are retained in per_target_metrics.parquet.',
                  'Plot truncation is visualization only. Full summary metrics are untrimmed; the official-code tiny-trim variant is a separately named field.', '']
        (out / 'tables.md').write_text('\n'.join(table), encoding='utf-8')
        if plots:
            plot_samples(sample_rows, out / 'sample_relative_error.png', job['description'])
        receipt = {'input_sha256': pins, 'seconds': time.perf_counter() - begin,
                   'process_RSS_MiB': psutil.Process().memory_info().rss / 2**20,
                   'no_torch_or_model_inference': True, 'parent_outputs_unchanged': all(sha(ROOT / p) == h for p, h in pins.items()),
                   'plots_are_not_scalar_metric_trimming': True}
        assert receipt['parent_outputs_unchanged']
        atomic(out / 'receipt.json', receipt)
        atomic(out / 'summary.json', summary)
        print(json.dumps({'job': job['id'], 'samples': summary['overall']['samples'],
                          'full_sample_mre_pct': summary['overall']['untrimmed_sample_mre']['mean_pct'],
                          'seconds': receipt['seconds']}, ensure_ascii=False), flush=True)
        return True
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()

def plot_samples(rows, output, title):
    import numpy as np
    x = np.array([r['sample_mre_pct'] for r in rows if r['sample_mre_pct'] is not None])
    if not len(x):
        return
    ordered = np.sort(x)
    indices = np.unique(np.linspace(0, len(x) - 1, min(1001, len(x))).astype(int))
    positive = x[x > 0]
    floor = max(float(positive.min()) / 10, 1e-12) if len(positive) else 1e-6
    cdf = [[max(float(ordered[i]), floor), float(i + 1) / len(x)] for i in indices]
    cutoff = float(np.percentile(x, 95))
    trimmed = x[x <= cutoff]
    density, edges = np.histogram(trimmed, bins=45,
                                 range=(0., max(cutoff, 1e-6)), density=True)
    payload = {'title': title + f'\nFull mean={x.mean():.5g}%, median={np.median(x):.5g}%, n={len(x)}',
               'cdf': cdf, 'histogram': [[float((a + b) / 2), float(y)] for a, b, y in zip(edges[:-1], edges[1:], density)],
               'footer': f'CDF shown at <=1001 ranks; zero-error x shown at {floor:.3g}%. Scalars use all samples. Plot cutoff={cutoff:.5g}%.',
               'backend': 'Windows .NET System.Windows.Forms.DataVisualization',
               'all_samples_in_scalar_statistics': True, 'plot_only_zero_coordinate_floor_pct': floor}
    json_path = output.with_suffix('.plot.json')
    atomic(json_path, payload)
    subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                    str(ROOT / 'scripts/render_error_distribution.ps1'), '-InputPath', str(json_path),
                    '-OutputPath', str(output)], cwd=ROOT, check=True, capture_output=True,
                   creationflags=subprocess.CREATE_NO_WINDOW)

def dashboard(config):
    lines = ['# FALCON参考口径的独立评价', '', '依据FALCON Stage 2：逐指标R²／MAE／RMSE／MRE，以及逐样本平均相对误差。',
             '下表均为完整、未经截尾的样本平均相对误差；不是5%命中率，也不能用100减该值称为准确率。',
             '已有测试预测的重新统计不等于新的盲测。不同数据来源与训练协议的数值不构成同条件模型排名。', '',
             '| 实验 | 样本数 | 完整MRE (%) | 样本中位误差 (%) | P95 (%) | 状态 |', '| --- | ---: | ---: | ---: | ---: | --- |']
    for job in config['jobs']:
        path = ROOT / 'artifacts/falcon_reference_eval_v1' / job['id'] / 'summary.json'
        if path.exists():
            q = read(path)['overall']; d = q['untrimmed_sample_mre']
            lines.append(f"| {job['description']} | {q['samples']} | {d['mean_pct']:.6g} | {d['median_pct']:.6g} | {d['p95_pct']:.6g} | 已统计 |")
        else:
            lines.append(f"| {job['description']} | — | — | — | — | 等待选模冻结与测试预测 |")
    lines += ['', '每个实验的逐任务表、原始样本误差和分布图位于`artifacts/falcon_reference_eval_v1/<job>/`。',
              'dB等标签的相对误差按FALCON原代码的数值约定计算，物理解释仍依赖MAE／RMSE。',
              '当前模型选择和数据划分保持原配方；新增统计不回用于已冻结模型的选择。', '']
    (ROOT / 'reports/falcon_reference_results.md').write_text('\n'.join(lines), encoding='utf-8')

def watch(config_path):
    control = ROOT / 'monitoring/falcon_reference_eval_v1'
    control.mkdir(parents=True, exist_ok=True)
    lock = (control / 'watch.lock').open('a+b'); lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    pinned = {str(p): sha(p) for p in [config_path, Path(__file__), ROOT / 'src/falcon_reference_metrics.py', ROOT / 'scripts/render_error_distribution.ps1']}
    state = {'pid': os.getpid(), 'create_time': psutil.Process().create_time()}
    try:
        while True:
            if (control / 'STOP').exists() or any(sha(p) != h for p, h in pinned.items()):
                atomic(control / 'status.json', {**state, 'state': 'stopped', 'updated_at': now()}); return
            config = read(config_path); pending = []
            for job in config['jobs']:
                if (ROOT / 'artifacts/falcon_reference_eval_v1' / job['id'] / 'summary.json').exists():continue
                pending.append(job['id'])
                if ready(job) and psutil.virtual_memory().available >= 1.0 * 2**30:
                    with (control / 'evaluation_stdout.log').open('ab') as out, (control / 'evaluation_stderr.log').open('ab') as err:
                        code = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(Path(__file__)), '--config', str(config_path), '--job', job['id']],
                                              cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW).returncode
                    if code not in {0, 75}:raise RuntimeError(f"Evaluation {job['id']} exited {code}")
            dashboard(config)
            atomic(control / 'status.json', {**state, 'state': 'waiting_frozen_predictions' if pending else 'complete',
                                           'updated_at': now(), 'pending_jobs': pending, 'no_training_config_change': True})
            if not pending:return
            time.sleep(30)
    except BaseException as error:
        atomic(control / 'status.json', {**state, 'state': 'requires_review', 'updated_at': now(), 'error': str(error)});raise
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1);lock.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--job')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if args.watch:
        watch(args.config.resolve()); return
    config = read(args.config)
    selected = [job for job in config['jobs'] if not args.job or job['id'] == args.job]
    if not selected:raise ValueError('Unknown job')
    for job in selected:
        if not evaluate(job, args.config.resolve(), not args.no_plots):
            print(json.dumps({'job': job['id'], 'state': 'pending_frozen_prediction'}))
    dashboard(config)

if __name__ == '__main__':
    main()
