"""FALCON Stage-2 reporting on physical predictions; no torch or model fitting.

Canonical tables retain topology and supervision semantics. A separate FALCON-only
view pools the original source fields to mirror the paper's Table 2 convention.
"""
import math
from collections import defaultdict
import numpy as np

EPSILON = 1e-8
LOG_UNITS = {'dB', 'dBm', 'dBc/Hz', 'source_label_dBm_compression_function'}

def relative_errors(prediction, target, mask):
    p = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    observed = np.asarray(mask, dtype=bool)
    if p.shape != y.shape or p.shape != observed.shape or p.ndim != 2:
        raise ValueError('Expected matching sample-by-metric arrays')
    if not np.isfinite(p[observed]).all() or not np.isfinite(y[observed]).all():
        raise ValueError('Nonfinite observed prediction/target; do not hide with Mask')
    valid = observed & (y != 0)
    error = np.zeros_like(y)
    error[valid] = np.abs(p[valid] - y[valid]) / (np.abs(y[valid]) + EPSILON)
    counts = valid.sum(axis=1)
    per_sample = np.divide(error.sum(axis=1), counts, out=np.full(len(y), np.nan), where=counts > 0)
    return error, valid, per_sample

def distribution(values):
    x = np.asarray(values, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Distribution must contain only defined finite errors')
    if not len(x):
        return {'count': 0, 'mean_pct': None, 'median_pct': None, 'p95_pct': None}
    x = x * 100.
    return {'count': len(x), 'mean_pct': float(x.mean()), 'median_pct': float(np.median(x)),
            'std_across_samples_pct': float(x.std()), 'p90_pct': float(np.percentile(x, 90)),
            'p95_pct': float(np.percentile(x, 95)), 'p99_pct': float(np.percentile(x, 99)),
            'max_pct': float(x.max())}

def regression(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if not len(y):
        return {'count': 0, 'mae': None, 'rmse': None, 'r2': None, 'r2_mathematical': None,
                'constant_target': False}
    difference = p - y
    sse = float(np.square(difference).sum())
    m2 = float(np.square(y - y.mean()).sum())
    if not math.isfinite(sse) or not math.isfinite(m2):
        raise ValueError('Nonfinite regression statistic')
    mathematical = 1. - sse / m2 if len(y) >= 2 and m2 > 0 else None
    # sklearn r2_score defaults to force_finite=True in the official code.
    reference = mathematical if m2 > 0 else (1. if sse == 0 else 0.)
    if len(y) < 2:
        reference = None
    return {'count': len(y), 'mae': float(np.abs(difference).mean()),
            'rmse': math.sqrt(sse / len(y)), 'r2': reference,
            'r2_mathematical': mathematical, 'constant_target': bool(len(y) >= 2 and m2 == 0)}

def target_report(chunks):
    y = np.concatenate([v[0] for v in chunks])
    p = np.concatenate([v[1] for v in chunks])
    valid = y != 0
    rel = np.abs(p[valid] - y[valid]) / (np.abs(y[valid]) + EPSILON)
    reference = regression(y[valid], p[valid])
    observed = regression(y, p)
    kept = int(len(rel) * (1 - 1e-5))
    code_trimmed = float(np.sort(rel)[:kept].mean() * 100) if kept else None
    return {'observed_count': len(y), 'reference_count': int(valid.sum()),
            'observed_zero_count': int((~valid).sum()), 'reference_nonzero': reference,
            'all_observed': observed, 'mean_relative_error_pct': float(rel.mean() * 100) if len(rel) else None,
            'official_code_table_trimmed_mre_pct': code_trimmed,
            'official_code_table_trim_removed': len(rel) - kept,
            'epsilon_dominated_nonzero_count': int((np.abs(y[valid]) <= EPSILON).sum())}

class FalconReferenceAccumulator:
    def __init__(self):
        self.samples = []
        self.identities = set()
        self.targets = defaultdict(list)
        self.metadata = {}
        self.falcon_fields = defaultdict(list)
        self.falcon_field_units = {}

    def update(self, prediction, target, mask, sample_ids, source, topology, group, rules):
        p = np.asarray(prediction, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)
        m = np.asarray(mask, dtype=bool)
        if p.shape != y.shape or p.shape != m.shape or p.shape[0] != len(sample_ids):
            raise ValueError('Invalid array shapes')
        factors = np.ones(p.shape[1], dtype=np.float64)
        for key, rule in rules.items():
            factors[int(key)] = rule.get('native_multiplier', 1.)
        if not np.isfinite(factors).all() or (factors <= 0).any():
            raise ValueError('Invalid audited unit conversion')
        pn, yn = p * factors, y * factors
        relative, valid, sample_error = relative_errors(pn, yn, m)
        for i, sample_id in enumerate(sample_ids):
            if sample_id in self.identities:
                raise ValueError('Duplicate sample_id within one model evaluation: ' + sample_id)
            self.identities.add(sample_id)
            self.samples.append({'sample_id': sample_id, 'source_dataset': source, 'topology_id': topology,
                                 'supervision_group': group, 'observed_metrics': int(m[i].sum()),
                                 'relative_valid_metrics': int(valid[i].sum()),
                                 'observed_zero_metrics': int((m[i] & ~valid[i]).sum()),
                                 'sample_mre_pct': float(sample_error[i] * 100) if np.isfinite(sample_error[i]) else None})
        for j in np.flatnonzero(m.any(axis=0)):
            if str(j) not in rules:
                raise ValueError('Observed metric lacks a registered definition')
            rule = rules[str(j)]
            selected = m[:, j]
            key = (source, topology, group, rule['semantic_id'])
            native_unit = rule.get('native_unit', rule['unit'])
            meta = {'source_dataset': source, 'topology_id': topology, 'supervision_group': group,
                    'semantic_id': rule['semantic_id'], 'slot': int(j), 'original_name': rule['original_name'],
                    'stored_unit': rule['unit'], 'native_unit': native_unit, 'native_multiplier': float(factors[j]),
                    'logarithmic_label': native_unit in LOG_UNITS}
            if key in self.metadata and self.metadata[key] != meta:
                raise ValueError('Task meaning or unit changed within one task')
            self.metadata[key] = meta
            pair = (yn[selected, j].copy(), pn[selected, j].copy())
            self.targets[key].append(pair)
            if source == 'FALCON':
                field = rule['original_name']
                if field in self.falcon_field_units and self.falcon_field_units[field] != native_unit:
                    raise ValueError('FALCON source field has inconsistent native units')
                self.falcon_field_units[field] = native_unit
                self.falcon_fields[field].append(pair)

    def finish(self):
        targets = [{**self.metadata[k], **target_report(v)} for k, v in sorted(self.targets.items())]
        pooled = [{'original_name': name, 'native_unit': self.falcon_field_units[name],
                   'pooling_scope': 'FALCON original source field only; statistical view, no supervision merge',
                   **target_report(chunks)} for name, chunks in sorted(self.falcon_fields.items())]
        def sample_summary(rows):
            errors = [s['sample_mre_pct'] / 100 for s in rows if s['sample_mre_pct'] is not None]
            x = np.asarray(errors)
            full = distribution(x)
            cut = float(np.percentile(x, 95)) if len(x) else None
            visual = x[x <= cut] if len(x) else x
            return {'samples': len(rows), 'undefined_sample_mre': len(rows) - len(errors),
                    'untrimmed_sample_mre': full,
                    'plot_only_95th_percentile_view': {**distribution(visual), 'removed_samples': len(x) - len(visual)}}
        result = {'profile': 'falcon_stage2_reference_v1', 'epsilon_in_native_label_units': EPSILON,
                  'relative_validity': 'original Mask AND nonzero native target; no Mask mutation',
                  'main_scalar': 'Untrimmed mean of per-sample mean relative errors, all native labels including log units',
                  'log_unit_interpretation': 'Relative error of reported log-label numbers, not error of linear voltage/power ratios',
                  'constant_target_r2': 'Reference matches sklearn force_finite=True; mathematical R2 retained separately',
                  'canonical_task_key': 'source + topology + supervision group + semantic_id',
                  'overall': sample_summary(self.samples),
                  'by_source': {source: sample_summary([s for s in self.samples if s['source_dataset'] == source])
                                for source in sorted({s['source_dataset'] for s in self.samples})},
                  'by_topology': {top: sample_summary([s for s in self.samples if s['topology_id'] == top])
                                  for top in sorted({s['topology_id'] for s in self.samples})},
                  'target_tasks': len(targets), 'falcon_native_fields': len(pooled),
                  'observed_scalars': sum(t['observed_count'] for t in targets),
                  'reference_nonzero_scalars': sum(t['reference_count'] for t in targets),
                  'zero_observations_retained_in_physical_diagnostics': sum(t['observed_zero_count'] for t in targets)}
        return result, targets, pooled, self.samples
