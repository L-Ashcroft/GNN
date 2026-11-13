"""Train-only, topology-local conditioning and fixed physical-unit validation."""
from collections import Counter
import numpy as np
from common import ROOT, SOURCE, CACHE, NEW_CACHE, PARENT_OUTPUT, read, digest, sha

LOG_UNITS = {'dB', 'dBm', 'dBc/Hz', 'source_label_dBm_compression_function'}


class Inputs:
    def __init__(self, budget):
        manifest = read(CACHE / 'manifest.json')
        for file, expected in manifest['files'].items():
            assert sha(ROOT / file) == expected, 'Input cache changed: ' + file
        self.index = read(CACHE / 'index.json')
        self.structures = read(CACHE / 'structures.json')
        self.tasks = read(CACHE / 'tasks.json')
        if budget == 'full':
            self.allowed = {r['sample_id'] for r in self.index if r['role'] == 'train'}
        else:
            record = next(r for r in read(SOURCE / 'splits/anchor_safe_v1/few_shot_per_topology.json')['budgets']
                          if r['budget_per_topology'] == int(budget))
            self.allowed = set(record['sample_ids'])
        assert all(r['role'] == 'train' for r in self.index if r['sample_id'] in self.allowed)
        assert len(self.allowed) == (16000 if budget == 'full' else 10 * int(budget))
        self.budget = budget
        self.roles = {}
        for name in self.structures:
            self.roles[name] = {}
            for role in ['train', 'validation']:
                self.roles[name][role] = [r for r in self.index if r['topology_id'] == name and r['role'] == role
                                         and (role != 'train' or r['sample_id'] in self.allowed)]
        selected = [r['sample_id'] for r in self.index if r['sample_id'] in self.allowed]
        self.training_id_hash_cache_order = digest(selected)

    def parent_check(self, parent):
        recipe = read(PARENT_OUTPUT / parent / 'recipe.json')
        assert recipe['job']['budget_per_topology'] == self.budget, 'Target-label budget changed'
        # Parent few-shot refs follow the declared budget-file order, not cache order.
        if self.budget == 'full':
            ids = [r['sample_id'] for r in self.index if r['role'] == 'train']
        else:
            ids = next(r['sample_ids'] for r in read(SOURCE / 'splits/anchor_safe_v1/few_shot_per_topology.json')['budgets']
                       if r['budget_per_topology'] == int(self.budget))
        assert recipe['statistics']['training_ids_sha256'] == digest(ids), 'Parent has different adaptation labels'
        return recipe

    def raw(self, name, role):
        assert role in ['train', 'validation'], 'Test data excluded from refinement'
        local = np.array([r['local'] for r in self.roles[name][role]], dtype=np.int64)
        arrays = {}
        for key in ['device', 'conditions', 'target', 'mask']:
            value = np.load(CACHE / name / (key + '.npy'), mmap_mode='r')
            arrays[key] = np.array(value[local], copy=True)
            value._mmap.close()
        # Numeric device fields only: flags and type fields are invariant within
        # the pinned topology. They remain present in the unchanged GNN branch.
        x = np.concatenate([arrays.pop('device')[:, :, :19].reshape(len(local), -1), arrays.pop('conditions')], axis=1)
        group = self.structures[name]['group']
        scales = np.array([self.tasks[group]['slots'].get(str(j), {}).get('scale', 1.) for j in range(16)])
        return x, arrays['target'], arrays['mask'], scales

    def block(self, name, parent=None, mode='joint'):
        x, y, mask, scales = self.raw(name, 'train')
        v, vy, vm, _ = self.raw(name, 'validation')
        physical_dimension = x.shape[1]
        if mode in ['joint', 'gnn']:
            assert parent is not None
            manifest = read(NEW_CACHE / parent / 'manifest.json')
            assert manifest['training_ids_sha256'] == self.training_id_hash_cache_order
            value = np.load(NEW_CACHE / parent / name / 'embedding.npy', mmap_mode='r')
            train_local = [r['local'] for r in self.roles[name]['train']]
            val_local = [r['local'] for r in self.roles[name]['validation']]
            a, b = np.array(value[train_local]), np.array(value[val_local]); value._mmap.close()
            assert np.isfinite(a).all() and np.isfinite(b).all()
            if mode == 'joint':
                x = np.concatenate([x, a], axis=1); v = np.concatenate([v, b], axis=1)
            else:
                x, v = a, b
        assert mode in ['joint', 'physical', 'gnn']
        x, v = x.astype(np.float64), v.astype(np.float64)
        center = x.mean(0); std = x.std(0)
        active = np.flatnonzero(std >= 1e-6)
        assert len(active), 'No varying known input features'
        x = (x[:, active] - center[active]) / std[active]
        v = (v[:, active] - center[active]) / std[active]
        scaled = y / scales
        rules = self.tasks[self.structures[name]['group']]['slots']
        # dB is already logarithmic; voltage errors and phase are signed.
        # A further sign-log bends these near-zero targets unnecessarily.
        kinds = ['linear' if rules.get(str(j), {}).get('unit') in LOG_UNITS | {'V','degree','V/V','V/A','1/K'}
                 else 'symlog' for j in range(16)]
        z = np.sign(scaled) * np.log1p(abs(scaled))
        for j, kind in enumerate(kinds):
            if kind == 'linear': z[:, j] = scaled[:, j]
        count = mask.sum(0)
        target_center = np.divide(np.where(mask, z, 0.).sum(0), count, out=np.zeros(16), where=count > 0)
        target_var = np.divide(np.where(mask, (z-target_center)**2, 0.).sum(0), count, out=np.zeros(16), where=count > 0)
        target_std = np.maximum(np.sqrt(target_var), 1e-3)
        low = [float(z[mask[:, j], j].min()) if count[j] else 0. for j in range(16)]
        high = [float(z[mask[:, j], j].max()) if count[j] else 0. for j in range(16)]
        slots = sorted(int(j) for j in self.tasks[self.structures[name]['group']]['slots'])
        assert all(count[j] > 0 for j in slots), 'No trained output for observed slot'
        state = {'active_columns': active.tolist(), 'input_center': center[active].tolist(),
                 'input_std': std[active].tolist(), 'target_center': target_center.tolist(),
                 'target_std': target_std.tolist(), 'target_count': count.tolist(), 'slots': slots,
                 'physical_dimension': physical_dimension, 'feature_mode': mode, 'target_std_floor': 1e-3,
                 'output_transform': kinds, 'training_target_min': low, 'training_target_max': high,
                 'inference_range_policy': 'Clamp predictions to observed training support; never change labels/mask',
                 'training_ids_sha256': digest([r['sample_id'] for r in self.roles[name]['train']]),
                 'training_rows': len(x), 'validation_or_test_statistics_used': False}
        return {'train_x': x.astype('float32'), 'validation_x': v.astype('float32'), 'train_z': z,
                'train_y': y, 'validation_y': vy, 'train_mask': mask, 'validation_mask': vm,
                'scales': scales, 'normalization': state, 'rules': self.tasks[self.structures[name]['group']]['slots']}


def ridge(block, alpha=10.):
    x = block['train_x'].astype('float64'); norm = block['normalization']
    z = (block['train_z'] - np.array(norm['target_center'])) / np.array(norm['target_std'])
    coefficient = np.zeros((len(norm['slots']), x.shape[1])); intercept = np.zeros(len(norm['slots']))
    for i, slot in enumerate(norm['slots']):
        observed = block['train_mask'][:, slot]
        a = x[observed]; b = z[observed, slot]
        center_x, center_y = a.mean(0), b.mean()
        a = a - center_x; b = b - center_y
        if len(a) < a.shape[1]:
            weight = a.T @ np.linalg.solve(a @ a.T + alpha * np.eye(len(a)), b)
        else:
            weight = np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ b)
        coefficient[i] = weight; intercept[i] = center_y - center_x @ weight
    return coefficient, intercept


def denormalize(normalized, block):
    norm = block['normalization']; slots = norm['slots']
    output = np.broadcast_to(norm['target_center'], (len(normalized), 16)).copy()
    output[:, slots] += normalized.astype('float64') * np.array(norm['target_std'])[slots]
    return np.clip(output, norm['training_target_min'], norm['training_target_max'])


def metric_rows(name, prediction_z, block):
    assert np.isfinite(prediction_z).all()
    transformed = prediction_z.copy()
    for j, kind in enumerate(block['normalization']['output_transform']):
        if kind == 'symlog': transformed[:, j] = np.sign(prediction_z[:, j])*np.expm1(abs(prediction_z[:, j]))
    physical = transformed * block['scales']
    rows = []
    for label, rule in block['rules'].items():
        j = int(label); observed = block['validation_mask'][:, j]
        y = block['validation_y'][observed, j]; p = physical[observed, j]
        assert len(y) and np.isfinite(p).all()
        err = p - y; sst = np.square(y-y.mean()).sum()
        relative = (abs(y) > 1e-8 * rule['scale']) & (rule['unit'] not in LOG_UNITS)
        relerr = abs(err[relative] / y[relative])
        rows.append({'topology_id': name, 'original_name': rule['original_name'], 'unit': rule['unit'],
                     'count': len(y), 'mae': float(abs(err).mean()), 'rmse': float(np.sqrt(np.square(err).mean())),
                     'r2': float(1.-np.square(err).sum()/sst) if len(y)>1 and sst>0 else None,
                     'hit5': float((relerr <= .05).mean()) if len(relerr) else None,
                     'hit10': float((relerr <= .1).mean()) if len(relerr) else None,
                     'hit_abs_1_unit': float((abs(err)<=1.).mean()) if rule['unit'] in LOG_UNITS else None,
                     'prediction_std': float(p.std()), 'target_std': float(y.std())})
    return rows


def summarize(rows):
    values = [r['r2'] for r in rows if r['r2'] is not None]
    hit = [r['hit5'] for r in rows if r['hit5'] is not None]
    return {'macro_hit5': float(np.mean(hit)), 'median_r2': float(np.median(values)),
            'macro_r2': float(np.mean(values)), 'negative_r2_targets': sum(v < 0 for v in values),
            'per_target': rows, 'split': 'validation_only', 'test_samples_used': 0}
