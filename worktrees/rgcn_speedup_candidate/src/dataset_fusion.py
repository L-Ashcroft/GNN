"""Streaming interface for the independent fusion_v2 corpus (no normalization)."""
from pathlib import Path
import hashlib
import pyarrow.parquet as pq
from .common import ROOT, read
from .netlist import evaluate, resolve_device_attributes


class FusionDataset:
    def __init__(self, path='data/fusion_v2', require_validated=True):
        self.path = ROOT / path
        self.manifest = read(self.path / 'manifest.json')
        if require_validated:
            if self.manifest['state'] != 'complete':
                raise ValueError('Dataset build is incomplete')
            report = read(self.path / 'quality_checks.json')
            if (report['status'] != 'passed' or report['index_sha256'] != self.manifest['index_sha256']
                    or report.get('manifest_sha256') != hashlib.sha256((self.path/'manifest.json').read_bytes()).hexdigest()):
                raise ValueError('Run check_fusion_dataset.py on this build first')
        self.catalog = {s['topology_id']: s for s in read(self.path / 'catalog.json')}
        self.graphs = {}

    def supervision_batches(self, batch_size=128, **sample_options):
        """Return batches with a single conservative supervision contract.

        Use distinct heads/losses per supervision_group. No cross-source target
        equivalence has been approved; raw samples() is only a storage interface.
        """
        from .supervision_fusion import SupervisionRouter
        sample_options.setdefault('materialize', False)
        yield from SupervisionRouter(self.catalog).batches(self.samples(**sample_options), batch_size)

    def materialize(self, row):
        row = dict(row)
        key = row['topology_id']; spec = self.catalog[key]
        if key not in self.graphs:
            self.graphs[key] = read(ROOT / spec['graph'])
        graph = self.graphs[key]
        env = {**spec['defaults'], **dict(zip(row['parameter_names'], row['parameters']))}
        features = []
        for expressions in graph['feature_expressions']:
            values = []
            for expression in expressions:
                try:
                    value = None if expression is None else evaluate(expression, env)
                except KeyError:
                    value = None
                values.append(value)
            features.append(values)
        row['graph'] = graph
        row['device_features'] = features
        row['device_feature_known'] = [[x is not None for x in f] for f in features]
        mp = graph.get('message_passing')
        row['physical_edge_index'] = graph['edge_index']
        row['physical_edge_device_index'] = graph['edge_device_index']
        row['edge_index'] = mp['edge_index'] if mp else graph['edge_index']
        row['edge_device_index'] = mp['edge_device_index'] if mp else graph['edge_device_index']
        row['edge_port_pairs'] = mp['edge_port_pairs'] if mp else [e['ports'] for e in graph['edges']]
        row['edge_features'] = [features[i] for i in row['edge_device_index']]
        row['edge_feature_known'] = [row['device_feature_known'][i] for i in row['edge_device_index']]
        if mp:
            from .graph_enrichment import transform_features
            transformed,known,zero = transform_features(features,graph['feature_names'])
            row.update(node_features=graph['node_features'],node_feature_names=graph['node_feature_names'],
                node_annotations=graph['node_annotations'],device_feature_names=graph['feature_names'],
                edge_types=mp['edge_type'],edge_relation_names=graph['feature_vocabulary']['relations'],
                edge_reverse_index=mp['reverse_edge_index'],edge_base_index=mp['edge_base_index'],
                edge_normalization_weight=mp['symmetric_norm_weight'],
                edge_features_transformed=[transformed[i] for i in row['edge_device_index']],
                edge_transformed_known=[known[i] for i in row['edge_device_index']],
                edge_feature_zero=[zero[i] for i in row['edge_device_index']])
            from .graph_enrichment import LOG10, LOG2
            row['edge_transformed_feature_names'] = [
                ('log10_'+n if n in LOG10 else 'log2_'+n if n in LOG2 else n) for n in graph['feature_names']]
        row['resolved_device_parameters'] = [
            {'name': d['name'], 'model': d['model'], 'attributes': resolve_device_attributes(d, env)}
            for d in graph['devices']]
        row['metrics_by_semantic_id'] = {
            name: value for name, value, known in zip(row['semantic_ids'], row['performance'], row['mask']) if known}
        row['conditions'] = spec.get('conditions', {})
        return row

    def samples(self, source=None, topology=None, materialize=True, split=None, role=None):
        """Yield rows, optionally selected by a split parquet and role.

        Sixteen slots are local to each semantic group. Compare semantic IDs,
        never slot numbers, when constructing a shared prediction target.
        """
        ids = None
        if split is not None:
            if role is None:
                raise ValueError('role is required with split')
            frame = pq.read_table(ROOT / split, columns=['sample_id', 'role']).to_pandas()
            ids = set(frame.loc[frame.role == role, 'sample_id'])
        for key, spec in self.catalog.items():
            if source and spec['source_dataset'] != source:
                continue
            if topology and key != topology:
                continue
            path = self.path / 'sampled' / (key + '.parquet')
            for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
                for row in batch.to_pylist():
                    if ids is None or row['sample_id'] in ids:
                        yield self.materialize(row) if materialize else row
