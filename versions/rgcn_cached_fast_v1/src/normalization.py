"""Per-protocol, per-semantic-group scaling fitted exclusively to training IDs."""
import numpy as np
import pandas as pd
from .common import dump, read, digest


def fit_training_scalers(output, split_dir):
    protocols = read(split_dir / 'protocols.json')
    specs = read(output / 'catalog.json')
    for name, meta in protocols.items():
        if meta['status'] != 'ready':
            continue
        split = pd.read_parquet(split_dir / meta['file'])
        train_ids = set(split.loc[split.role == 'train', 'sample_id'])
        result = {'protocol': name, 'fit_role': 'train', 'training_ids_sha256': digest(sorted(train_ids)), 'groups': {}}
        train_counts = split[split.role == 'train'].topology_id.value_counts().to_dict()
        by_connectivity = {}
        for spec in specs:
            if train_counts.get(spec['topology_id'], 0):
                by_connectivity.setdefault(spec['connectivity_group'], []).append(spec['topology_id'])
        result['sampling_weight_per_topology'] = {
            key: len(train_ids) / (len(by_connectivity) * len(keys) * train_counts[key])
            for keys in by_connectivity.values() for key in keys}
        for spec in specs:
            key = spec['topology_id']
            table = pd.read_parquet(output / 'sampled' / (key + '.parquet'))
            table = table[table.sample_id.isin(train_ids)]
            if table.empty:
                continue
            p = np.stack(table.parameters)
            y = np.stack(table.performance).astype(float)
            m = np.stack(table['mask'])
            def stats(values):
                return {'count': len(values), 'mean': float(np.mean(values)), 'std': float(np.std(values)) or 1.0}
            result['groups'][key] = {'parameters': {n: stats(p[:, i]) for i, n in enumerate(spec['parameters'])},
                                    'performance': {str(i): stats(y[m[:, i], i]) for i in range(16) if m[:, i].any()}}
        dump(split_dir / 'normalization' / (name + '.json'), result)
