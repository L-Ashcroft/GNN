"""Direct-loading API; feature recipes avoid repeating fixed devices per row."""
from functools import lru_cache
import numpy as np
import pandas as pd
from .common import ROOT, read
from .netlist import evaluate, resolve_device_attributes


@lru_cache(maxsize=128)
def graph_at(path):
    return read(ROOT / path)


def materialize(row, catalog):
    row = dict(row)
    row['mask'] = [bool(v) for v in row['mask']]
    # pandas represents Arrow nulls as NaN; restore explicit missingness at the API boundary.
    row['performance'] = [float(v) if known else None for v, known in zip(row['performance'], row['mask'])]
    row['raw_performance'] = [None if v is None or not np.isfinite(v) else float(v) for v in row['raw_performance']]
    row['parameters'] = [float(v) for v in row['parameters']]
    spec = catalog[row['topology_id']]
    graph = graph_at(row['graph'])
    env = {**spec['defaults'], **dict(zip(row['parameter_names'], row['parameters']))}
    features = []
    known = []
    for expressions in graph['feature_expressions']:
        values, valid = [], []
        for exp in expressions:
            value = None
            if exp is not None:
                try:
                    value = evaluate(exp, env)
                except KeyError:
                    pass
            values.append(value); valid.append(value is not None)
        features.append(values); known.append(valid)
    row['graph'] = graph
    row['device_features'] = features
    row['device_feature_known'] = known
    # Each device may own several port-pair edges; all share its parameters.
    # Devices are a parameter/provenance table, NOT graph nodes.
    row['edge_index'] = graph['edge_index']
    row['edge_device_index'] = graph['edge_device_index']
    row['edge_port_pairs'] = [e['ports'] for e in graph['edges']]
    row['edge_features'] = [features[i] for i in graph['edge_device_index']]
    row['edge_feature_known'] = [known[i] for i in graph['edge_device_index']]
    row['resolved_device_parameters'] = [
        {'name': d['name'], 'model': d['model'], 'attributes': resolve_device_attributes(d, env)} for d in graph['devices']]
    row['semantic_ids'] = [None] * 16
    for rule in read(ROOT / 'config/performance_mapping.json')['groups'][row['topology_id']]:
        row['semantic_ids'][rule['slot']] = rule['semantic_id']
    return row


def load_split(protocol='id', role='train', output='data/strict', split_root='data/strict_splits', ready_only=True):
    base = ROOT / output
    manifest = read(ROOT / split_root / 'protocols.json')
    meta = manifest[protocol]
    if meta['status'] != 'ready':
        raise ValueError(meta['reason'])
    split = pd.read_parquet(ROOT / split_root / meta['file'])
    ids = set(split.loc[split['role'] == role, 'sample_id'])
    catalog = {s['topology_id']: s for s in read(base / 'catalog.json')}
    weights = read(ROOT / split_root / 'normalization' / (protocol + '.json'))['sampling_weight_per_topology']
    for s in catalog.values():
        table = pd.read_parquet(base / 'sampled' / (s['topology_id'] + '.parquet'))
        table = table[table.sample_id.isin(ids)]
        if ready_only:
            table = table[table.input_complete]
        for row in table.to_dict('records'):
            row['sampling_weight'] = weights.get(row['topology_id']) if role == 'train' else None
            yield materialize(row, catalog)
