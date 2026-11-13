"""Full-data invariants, raw-row reconciliation, graph and split leakage checks."""
import argparse
from pathlib import Path
import sys
import hashlib
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from src.common import ROOT, read, dump, digest
from src.netlist import parse_spectre, make_graph, FEATURE_NAMES
from src.dataset import materialize


def check(output, split_dir):
    specs = read(output / 'catalog.json')
    ref = output / 'conversion_reference.json'
    converted_dir = ROOT / read(ref)['converted_dir'] if ref.exists() else output / 'converted'
    catalog = {s['topology_id']: s for s in specs}
    rules = read(ROOT / 'config/performance_mapping.json')['groups']
    ix = pd.read_parquet(output / 'index.parquet')
    assert ix.sample_id.is_unique, 'duplicate sample ID'
    assert ix.record_hash.is_unique, 'duplicate record survived'
    assert not ix[['source_path', 'source_row']].duplicated().any()
    totals = []
    for s in specs:
        key = s['topology_id']
        graph = read(ROOT / s['graph'])
        raw_graph = make_graph(parse_spectre((ROOT / s['netlist']).read_text()))
        for field in ['representation', 'nets', 'devices', 'edges', 'edge_index', 'edge_device_index', 'device_terminals', 'device_references', 'num_nodes', 'num_devices', 'num_nets', 'num_edges']:
            assert graph[field] == raw_graph[field], (key, field)
        assert graph['representation'] == 'net_device_multigraph_v2'
        assert graph['num_nodes'] == graph['num_nets']
        assert graph['num_edges'] == len(graph['edges'])
        assert graph['num_terminals'] == sum(len(d['ports']) for d in graph['devices'])
        assert graph['num_device_references'] == len(graph['device_references'])
        assert graph['edge_index'] == [[e['source'] for e in graph['edges']], [e['target'] for e in graph['edges']]]
        assert graph['edge_device_index'] == [e['device'] for e in graph['edges']]
        assert graph['num_nets'] == len(set(graph['nets']))
        for i, device in enumerate(graph['devices']):
            incidences = [e for e in graph['device_terminals'] if e['device'] == i]
            assert len(incidences) == len(device['nets'])
            assert [(e['port'], graph['nets'][e['net']]) for e in incidences] == list(zip(device['ports'], device['nets']))
            if device['model'] in {'nmos', 'pmos'}:
                assert device['ports'] == ['D', 'G', 'S', 'B']
                own_edges = [e for e in graph['edges'] if e['device'] == i]
                assert [e['ports'] for e in own_edges] == [['D','G'], ['D','S'], ['D','B'], ['G','S'], ['G','B'], ['S','B']]
            for edge in (e for e in graph['edges'] if e['device'] == i):
                a, b = edge['ports']
                assert graph['nets'][edge['source']] == device['nets'][device['ports'].index(a)]
                assert graph['nets'][edge['target']] == device['nets'][device['ports'].index(b)]
                assert edge['name'] == device['name']
        full = pd.read_csv(ROOT / s['csv'])
        inventory = next(t for t in read(ROOT / 'reports/source_inventory.json')['tables'] if
                         t['source'] == s['source_dataset'] and t['path'] == s['csv'].split('/', 2)[2])
        assert hashlib.sha256((ROOT / s['csv']).read_bytes()).hexdigest() == inventory['sha256']
        count = 0
        for batch in pq.ParquetFile(converted_dir / (key + '.parquet')).iter_batches(batch_size=10000):
            table = batch.to_pandas()
            assert table.sample_id.is_unique
            src = full.iloc[table.source_row.to_numpy()-1]
            params = np.stack(table.parameters).astype(float)
            raw = np.stack(table.raw_performance).astype(float)
            assert np.isfinite(params).all()
            np.testing.assert_array_equal(params, src[s['parameters']].to_numpy(float))
            np.testing.assert_array_equal(raw, src[s['metric_names']].to_numpy(float))
            labels = np.stack(table.performance).astype(float)
            mask = np.stack(table['mask'])
            assert mask.shape == labels.shape == (len(table), 16)
            assert np.isfinite(labels[mask]).all()
            assert np.isnan(labels[~mask]).all()
            assert np.array_equal(mask[:, :len(s['metric_names'])], ~np.isnan(raw))
            assert not mask[:, len(s['metric_names']):].any()
            for rule in rules[key]:
                j = rule['slot']
                np.testing.assert_array_equal(labels[:, j], raw[:, j] * rule['factor'])
            assert table.source_dataset.eq(s['source_dataset']).all()
            assert table.topology_id.eq(key).all()
            count += len(table)
        selected = pd.read_parquet(output / 'sampled' / (key + '.parquet'))
        ixpart = ix[ix.topology_id == key]
        assert selected.sample_id.tolist() == ixpart.sample_id.tolist()
        assert len(selected) <= count
        assert selected.record_hash.is_unique
        if len(selected):
            # Independent MOS D/G/S/B pin oracle for flat sources, directly from raw text.
            import re
            text = (ROOT / s['netlist']).read_text()
            if not re.search(r'^\s*subckt\b', text, re.M):
                expected = {m[1]: m[2].split() for m in re.finditer(r'^\s*(\S+)\s+\(([^)]*)\)\s+[np]mos\b', text, re.M)}
                actual = {d['name']: d['nets'] for d in graph['devices'] if d['model'] in {'nmos', 'pmos'}}
                assert expected == actual
            for row in selected.iloc[[0, len(selected)//2, len(selected)-1]].to_dict('records'):
                item = materialize(row, catalog)
                for resolved in item['resolved_device_parameters']:
                    if s['input_complete']:
                        assert all(v is not None for v in resolved['attributes'].values()), (key, resolved)
                assert len(item['device_features']) == graph['num_devices']
                assert len(item['edge_features']) == graph['num_edges']
                assert len(item['edge_port_pairs']) == graph['num_edges']
                for edge, values, known in zip(graph['edges'], item['edge_features'], item['edge_feature_known']):
                    assert values == item['device_features'][edge['device']]
                    assert known == item['device_feature_known'][edge['device']]
                for values, known in zip(item['device_features'], item['device_feature_known']):
                    assert len(values) == len(FEATURE_NAMES)
                    assert all((v is not None) == k for v, k in zip(values, known))
                    assert all(np.isfinite(v) for v in values if v is not None)
                if s['input_complete']:
                    for d, values in zip(graph['devices'], item['device_features']):
                        if d['model'] in {'nmos', 'pmos'}:
                            assert all(v is not None and v > 0 for v in values[:3])
        totals.append({'topology_id': key, 'converted_rows_checked': count, 'sampled_rows_checked': len(selected)})
        print('checked', key, count, flush=True)
    protocols = read(split_dir / 'protocols.json')
    split_results = {}
    for name, meta in protocols.items():
        split = pd.read_parquet(split_dir / meta['file'])
        assert split.sample_id.is_unique
        assert set(split.sample_id) == set(ix.sample_id)
        active = split[split.role.isin(['train', 'validation', 'test'])]
        assert active.groupby('design_group').role.nunique().max() == 1
        for role in ['train', 'validation', 'test']:
            ids = set(active.loc[active.role == role, 'sample_id'])
            for other in ['train', 'validation', 'test']:
                if other != role:
                    assert ids.isdisjoint(active.loc[active.role == other, 'sample_id'])
        if name.startswith('loto'):
            test = set(active.loc[active.role == 'test', 'topology_group'])
            assert test == set(meta['heldout_topology_groups'])
            assert test.isdisjoint(active.loc[active.role != 'test', 'topology_group'])
        if name.startswith('cross'):
            assert active.loc[active.role == 'test', 'source_dataset'].eq(meta['heldout_source']).all()
            assert not active.loc[active.role != 'test', 'source_dataset'].eq(meta['heldout_source']).any()
            assert meta['comparable_semantic_ids'] == [] and meta['status'] != 'ready'
        if meta['status'] == 'ready':
            norm = read(split_dir / 'normalization' / (name + '.json'))
            train_ids = set(active.loc[active.role == 'train', 'sample_id'])
            assert norm['training_ids_sha256'] == digest(sorted(train_ids))
            counts = active[active.role == 'train'].topology_id.value_counts().to_dict()
            classes = {}
            for spec in specs:
                if counts.get(spec['topology_id'], 0):
                    classes.setdefault(spec['connectivity_group'], []).append(spec['topology_id'])
            expected_weights = {k: len(train_ids)/(len(classes)*len(keys)*counts[k]) for keys in classes.values() for k in keys}
            assert norm['sampling_weight_per_topology'] == expected_weights
            for s in specs:
                selected = pd.read_parquet(output / 'sampled' / (s['topology_id'] + '.parquet'))
                selected = selected[selected.sample_id.isin(train_ids)]
                if selected.empty:
                    assert s['topology_id'] not in norm['groups']; continue
                # Recompute from training rows and verify count/mean/std for all targets and input parameters.
                group = norm['groups'][s['topology_id']]
                for category, values, names in [('parameters', np.stack(selected.parameters), s['parameters']),
                                                 ('performance', np.stack(selected.performance).astype(float), [str(i) for i in range(16)])]:
                    for j, field in enumerate(names):
                        col = values[:, j]; col = col[np.isfinite(col)]
                        if len(col):
                            stat = group[category][field]
                            assert stat['count'] == len(col)
                            np.testing.assert_allclose([stat['mean'], stat['std']], [np.mean(col), float(np.std(col)) or 1.], rtol=1e-12)
        split_results[name] = {'status': meta['status'], 'roles': split.role.value_counts().to_dict(), 'leakage_checks': 'passed'}
    result = {'status': 'passed', 'sampled_ids_checked': len(ix), 'graphs_checked': len(specs), 'tables': totals,
              'protocols': split_results, 'limits': 'Graph templates and all accepted rows checked; materialized numeric features spot-checked at 3 rows per topology. Convergence cannot be reverified without original simulator logs.'}
    dump(output / 'quality_checks.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', default='data/processed'); p.add_argument('--splits', default='data/splits')
    a = p.parse_args()
    print(check(ROOT / a.output, ROOT / a.splits)['status'])
