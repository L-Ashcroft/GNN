"""Memory-mapped inputs, unchanged anchor-safe roles and explicit label budgets."""
from collections import Counter
import gc
import math
import numpy as np
import pyarrow.parquet as pq
import torch
from common import ROOT, SOURCE, META, CACHE, atomic, read, sha, now
from src.personal_gnn_data import PersonalDataset, PersonalBatch, context_values
from src.graph_enrichment import transform_features
from src.graph.builder import feature_vector


def prepare():
    if (CACHE / 'manifest.json').exists():
        return Reader(verify=True).manifest
    config = read(ROOT / 'config/personal_rgcn.json')
    dataset = PersonalDataset(config, verify=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    schema, tasks = dataset.schema, dataset.tasks
    snapshot = read(SOURCE / 'dataset.json')
    entries, index, structures = {}, [], {}
    split = {r['sample_id']: r for r in dataset.split}
    names = sorted(dataset.catalog)
    shapes = {}
    for name in names:
        graph = read(ROOT / dataset.catalog[name]['graph'])
        mp = graph['message_passing']
        assert not graph['device_references'], 'Explicit coupling adapter needed'
        structures[name] = {
            'x': [[math.log1p(v) if n == 'connected_device_count' or n.startswith('terminal_count:') else float(v)
                   for n, v in zip(graph['node_feature_names'], row)] for row in graph['node_features']],
            'edge_index': mp['edge_index'], 'edge_type': mp['edge_type'],
            'device_index': mp['edge_device_index'], 'group': dataset.catalog[name]['group']}
        count, devices = 2000, len(graph['devices'])
        shapes[name] = {'device': (count, devices, 90), 'target': (count, 16),
                        'mask': (count, 16), 'conditions': (count, 46)}
        folder = CACHE / name
        folder.mkdir(exist_ok=True)
        entries[name] = {k: np.lib.format.open_memmap(folder / (k + '.npy'), mode='w+',
                         dtype=np.bool_ if k == 'mask' else np.float64 if k == 'target' else np.float32,
                         shape=shape) for k, shape in shapes[name].items()}
    counts = Counter()
    columns = [c for c in pq.ParquetFile(ROOT / snapshot['shards'][0]['path']).schema_arrow.names if c != 'device_features']
    for shard in snapshot['shards']:
        for part in pq.ParquetFile(ROOT / shard['path']).iter_batches(batch_size=64, columns=columns):
            for row in part.to_pylist():
                name = row['topology_id']
                local = counts[name]
                graph, features = dataset.materialize(row)
                values, known, zero = transform_features(features, graph['feature_names'])
                entries[name]['device'][local] = [feature_vector(v, k, z, graph['feature_names'])
                                                   for v, k, z in zip(values, known, zero)]
                entries[name]['conditions'][local] = context_values(row)
                entries[name]['target'][local] = [0. if v is None else v for v in row['performance']]
                entries[name]['mask'][local] = row['mask']
                item = split[row['sample_id']]
                assert item['topology_id'] == name and item['design_group'] == row['design_group']
                index.append({k: item[k] for k in ['sample_id', 'design_group', 'topology_id', 'role']} |
                             {'local': local})
                counts[name] += 1
        atomic(CACHE / 'prepare_status.json', {'state': 'building', 'rows': len(index), 'counts': dict(counts), 'updated_at': now()})
    assert counts == Counter({n: 2000 for n in names})
    assert len({r['sample_id'] for r in index}) == len({r['design_group'] for r in index}) == 20000
    assert Counter(r['role'] for r in index) == {'train': 16000, 'validation': 2000, 'test': 2000}
    for maps in entries.values():
        for array in maps.values():
            array.flush()
    entries.clear()
    gc.collect()
    for filename, value in [('index.json', index), ('structures.json', structures), ('schema.json', schema), ('tasks.json', tasks)]:
        atomic(CACHE / filename, value)
    files = [p for n in names for p in (CACHE / n).glob('*.npy')] + [CACHE / n for n in ['index.json', 'structures.json', 'schema.json', 'tasks.json']]
    manifest = {'rows': 20000, 'topologies': names, 'split_counts': {'train': 16000, 'validation': 2000, 'test': 2000},
                'input_manifest_sha256': sha(META / 'manifest.json'),
                'split_sha256': sha(SOURCE / 'splits/anchor_safe_v1/id.parquet'),
                'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in files},
                'normalization_fitted': False, 'original_samples_or_roles_modified': False,
                'test_policy': 'Label copying for immutable cache only; no fitted statistics or model selection on test',
                'bytes': sum(p.stat().st_size for p in files)}
    atomic(CACHE / 'manifest.json', manifest)
    atomic(CACHE / 'prepare_status.json', {'state': 'complete', 'rows': 20000, 'updated_at': now()})
    return manifest


class Reader:
    def __init__(self, verify=False):
        self.manifest = read(CACHE / 'manifest.json')
        if verify:
            assert sha(META / 'manifest.json') == self.manifest['input_manifest_sha256']
            assert sha(SOURCE / 'splits/anchor_safe_v1/id.parquet') == self.manifest['split_sha256']
            for name, expected in self.manifest['files'].items():
                assert sha(ROOT / name) == expected, name
        self.index = read(CACHE / 'index.json')
        self.schema, self.tasks = read(CACHE / 'schema.json'), read(CACHE / 'tasks.json')
        self.structures = read(CACHE / 'structures.json')
        self.by_id = {r['sample_id']: i for i, r in enumerate(self.index)}
        self.refs = {role: np.array([i for i, row in enumerate(self.index) if row['role'] == role], dtype=np.int64)
                     for role in ['train', 'validation', 'test']}
        self.arrays = {name: {key: np.load(CACHE / name / (key + '.npy'), mmap_mode='r')
                       for key in ['device', 'target', 'mask', 'conditions']} for name in self.structures}
        self.scales = {g: np.array([t['slots'].get(str(j), {}).get('scale', 1.) for j in range(16)]) for g, t in self.tasks.items()}

    def training_refs(self, budget):
        if budget == 'full':
            return self.refs['train'].copy()
        record = next(x for x in read(SOURCE / 'splits/anchor_safe_v1/few_shot_per_topology.json')['budgets']
                      if x['budget_per_topology'] == int(budget))
        refs = np.array([self.by_id[s] for s in record['sample_ids']], dtype=np.int64)
        assert len(refs) == 10 * int(budget)
        assert all(self.index[i]['role'] == 'train' for i in refs)
        assert set(Counter(self.index[i]['topology_id'] for i in refs).values()) == {int(budget)}
        return refs

    def batch(self, refs, role):
        xs, srcs, dsts, attributes, types, graph_ids = [], [], [], [], [], []
        ys, masks, scales, groups, semantics, sample_ids, contexts, topologies = [], [], [], [], [], [], [], []
        offset = 0
        for i, ref in enumerate(refs):
            row = self.index[int(ref)]
            assert row['role'] == role
            name, local = row['topology_id'], row['local']
            structure, arrays = self.structures[name], self.arrays[name]
            x = structure['x']; edge = np.asarray(structure['edge_index'])
            xs.extend(x); graph_ids.extend([i] * len(x))
            srcs.extend(edge[0] + offset); dsts.extend(edge[1] + offset)
            attributes.extend(arrays['device'][local][structure['device_index']]); types.extend(structure['edge_type'])
            groups.append(structure['group']); scales.append(self.scales[groups[-1]])
            ys.append(arrays['target'][local]); masks.append(arrays['mask'][local]); contexts.append(arrays['conditions'][local])
            sample_ids.append(row['sample_id']); topologies.append(name)
            semantics.append([self.tasks[groups[-1]]['slots'].get(str(j), {}).get('semantic_id') for j in range(16)])
            offset += len(x)
        result = PersonalBatch(x=torch.tensor(np.asarray(xs), dtype=torch.float32),
             edge_index=torch.tensor(np.asarray([srcs, dsts]), dtype=torch.long),
             edge_attr=torch.tensor(np.asarray(attributes), dtype=torch.float32),
             edge_type=torch.tensor(types, dtype=torch.long), graph_index=torch.tensor(graph_ids, dtype=torch.long),
             targets=torch.tensor(np.asarray(ys), dtype=torch.float64), mask=torch.tensor(np.asarray(masks), dtype=torch.bool),
             scales=torch.tensor(np.asarray(scales), dtype=torch.float64), group_ids=groups, semantic_ids=semantics,
             sample_ids=sample_ids, domain_ids=torch.zeros(len(refs), dtype=torch.long), split_role=role,
             conditions=torch.tensor(np.asarray(contexts), dtype=torch.float32))
        result.topology_ids = topologies
        return result

    def statistics(self, refs):
        buckets = {g: [] for g in self.tasks}
        for ref in refs:
            row = self.index[int(ref)]
            assert row['role'] == 'train', 'Statistics must use declared adaptation training rows only'
            name, local = row['topology_id'], row['local']; group = self.structures[name]['group']
            y = self.arrays[name]['target'][local] / self.scales[group]
            z = np.sign(y) * np.log1p(abs(y))
            buckets[group].append((z, self.arrays[name]['mask'][local]))
        stats = {}
        for group, rows in buckets.items():
            assert rows, 'Training budget lacks a required supervision group'
            z, mask = np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])
            count = mask.sum(0); total = np.where(mask, z, 0.).sum(0)
            center = np.divide(total, count, out=np.zeros(16), where=count > 0)
            var = np.divide(np.where(mask, (z - center)**2, 0.).sum(0), count, out=np.zeros(16), where=count > 0)
            # Fixed new recipe: prevent near-constant tiny-budget targets from dominating.
            std = np.maximum(np.sqrt(var), .1)
            wanted = [int(j) for j in self.tasks[group]['slots']]
            assert all(count[j] > 0 for j in wanted), 'Insufficient observed-slot coverage; do not invent zero-shot outputs'
            stats[group] = {'center': center.tolist(), 'std': std.tolist(), 'count': count.tolist()}
        return {'groups': stats, 'training_rows': len(refs), 'std_floor': .1,
                'validation_or_test_statistics_used': False,
                'training_ids_sha256': __import__('common').digest([self.index[int(i)]['sample_id'] for i in refs])}
