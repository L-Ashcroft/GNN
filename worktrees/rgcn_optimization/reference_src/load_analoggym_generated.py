"""Load measured pilot samples independently of the historical master dataset."""
import pandas as pd
from .common import ROOT, read


def iter_generated_samples(output='data/generated/analoggym_nmcf_pilot_v2'):
    folder = (ROOT/output).resolve()
    if not folder.is_relative_to(ROOT): raise ValueError('Output must remain inside workspace')
    if (folder/'dataset.json').exists():
        yield from iter_campaign_samples(output)
        return
    if not (folder/'samples.parquet').exists():
        raise FileNotFoundError('No exported accepted simulation samples in ' + str(folder))
    graph = read(folder/'graph.json')
    for row in pd.read_parquet(folder/'samples.parquet').to_dict('records'):
        row['performance'] = [float(v) if bool(k) else None for v,k in zip(row['performance'],row['mask'])]
        row['mask'] = [bool(k) for k in row['mask']]
        row['device_features'] = [[None if pd.isna(v) else float(v) for v in r] for r in row['device_features']]
        row['graph'] = graph
        row['edge_index'] = graph['edge_index']
        row['edge_device_index'] = graph['edge_device_index']
        row['edge_port_pairs'] = [e['ports'] for e in graph['edges']]
        row['edge_features'] = [row['device_features'][i] for i in graph['edge_device_index']]
        yield row


def iter_campaign_samples(output='data/generated/analoggym_campaign_v1', topologies=None):
    """Stream recorded shards; graph features are resolved from the stored design.

    Simulation waveforms are provenance only and are never graph input features.
    A running campaign's dataset.json is the latest completed export snapshot.
    """
    import pyarrow.parquet as pq
    folder=(ROOT/output).resolve()
    if not folder.is_relative_to(ROOT): raise ValueError('Dataset must remain inside workspace')
    manifest=read(folder/'dataset.json'); cache={}
    for shard in manifest['shards']:
        path=(ROOT/shard['path']).resolve()
        if not path.is_relative_to(folder): raise ValueError('Shard escapes dataset directory')
        if topologies and path.parent.name not in topologies: continue
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                key=row['graph']
                graph_path=(ROOT/key).resolve()
                if not graph_path.is_relative_to(folder): raise ValueError('Graph escapes dataset directory')
                if key not in cache: cache[key]=read(graph_path)
                graph=cache[key]; row['graph']=graph
                row['edge_index']=graph['edge_index']; row['edge_device_index']=graph['edge_device_index']
                row['edge_port_pairs']=[e['ports'] for e in graph['edges']]
                row['edge_features']=[row['device_features'][i] for i in graph['edge_device_index']]
                yield row
