"""Common converter. Source modules discover only audited official artifacts."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .common import ROOT, dump, digest, read
from .netlist import evaluate


SCHEMA = pa.schema([
    ('sample_id', pa.string()), ('source_dataset', pa.string()), ('topology_id', pa.string()),
    ('topology_group', pa.string()), ('connectivity_group', pa.string()), ('circuit_family', pa.string()),
    ('process', pa.string()), ('PDK', pa.string()), ('temperature', pa.float64()), ('supply_voltage', pa.float64()),
    ('testbench_id', pa.string()), ('compatibility_group', pa.string()), ('graph', pa.string()),
    ('device_features', pa.string()), ('input_complete', pa.bool_()), ('source_path', pa.string()),
    ('source_sha256', pa.string()), ('source_row', pa.int64()), ('parameter_names', pa.list_(pa.string())),
    ('parameters', pa.list_(pa.float64())), ('metric_names', pa.list_(pa.string())),
    ('raw_performance', pa.list_(pa.float64())), ('performance', pa.list_(pa.float64(), 16)),
    ('mask', pa.list_(pa.bool_(), 16)), ('design_group', pa.string()), ('record_hash', pa.string())])


def find_ambiguous_input_rows(frame, parameters, metrics):
    repeated = frame.duplicated(parameters, keep=False)
    if not repeated.any():
        return set()
    repeats = frame[repeated]
    conflicts = repeats.groupby(parameters, dropna=False)[metrics].transform('nunique').gt(1).any(axis=1)
    return set((repeats.index[conflicts] + 1).tolist())


def convert(spec, output, limit=None):
    key = spec['topology_id']
    dest = output / 'converted' / (key + '.parquet')
    statsfile = output / 'converted' / (key + '.json')
    source_info = next(t for t in read(ROOT / 'reports/source_inventory.json')['tables'] if
                       t['source'] == spec['source_dataset'] and t['path'] == spec['csv'].split('/', 2)[2])
    fingerprint = digest([spec, source_info['sha256'], limit, 'converter-v2-ambiguous-input-quarantine'])
    if dest.exists() and statsfile.exists() and read(statsfile).get('fingerprint') == fingerprint:
        return read(statsfile)
    dest.parent.mkdir(parents=True, exist_ok=True)
    mappings = read(ROOT / 'config/performance_mapping.json')['groups'][key]
    stats = {'topology_id': key, 'fingerprint': fingerprint, 'raw_rows': source_info['rows'], 'processed_rows': 0,
             'accepted_rows': 0, 'rejected_rows': 0, 'input_complete': spec['input_complete'], 'reasons': {}}
    rejected = []
    # A deterministic predictor cannot identify different labels for exactly the
    # same recorded inputs and fixed source defaults. Keep source rows intact,
    # quarantine every contradictory group rather than choosing a label.
    audit_frame = pd.read_csv(ROOT / spec['csv'], usecols=spec['parameters'] + spec['metric_names'])
    ambiguous_rows = find_ambiguous_input_rows(audit_frame, spec['parameters'], spec['metric_names'])
    stats['source_ambiguous_input_rows'] = len(ambiguous_rows)
    writer = pq.ParquetWriter(str(dest) + '.partial', SCHEMA, compression='zstd')
    position = 0
    try:
        for frame in pd.read_csv(ROOT / spec['csv'], chunksize=10000, nrows=limit):
            records = []
            for row in frame.to_dict('records'):
                position += 1
                stats['processed_rows'] += 1
                reason = None
                if position in ambiguous_rows:
                    reason = 'ambiguous_recorded_input_labels'
                parameters = [float(row[p]) for p in spec['parameters']]
                raw = [None if pd.isna(row[m]) else float(row[m]) for m in spec['metric_names']]
                if not all(np.isfinite(parameters)):
                    reason = 'nonfinite_parameter'
                if any(v is not None and not np.isfinite(v) for v in raw):
                    reason = 'nonfinite_label'
                if not any(v is not None for v in raw):
                    reason = 'no_labels'
                for m, v in zip(spec['metric_names'], raw):
                    if v is not None and (m in {'Bandwidth', 'OscillationFrequency'} and v <= 0 or
                                          m in {'DCPowerConsumption', 'PowerConsumption', 'TuningRange'} and v < 0):
                        reason = 'nonphysical_' + m
                if reason:
                    stats['rejected_rows'] += 1
                    stats['reasons'][reason] = stats['reasons'].get(reason, 0) + 1
                    rejected.append({'source_path': spec['csv'], 'source_row': position, 'reason': reason})
                    continue
                perf = [None] * 16
                mask = [False] * 16
                for rule, value in zip(mappings, raw):
                    if value is not None:
                        perf[rule['slot']] = value * rule['factor']; mask[rule['slot']] = True
                env = dict(zip(spec['parameters'], parameters))
                supply = spec['supply_voltage_default']
                if spec['supply_voltage_expression']:
                    try:
                        supply = evaluate(spec['supply_voltage_expression'], {**spec['defaults'], **env})
                    except KeyError:
                        pass
                # Canonicalize case only for leakage grouping, never for expression resolution.
                # This is intentionally conservative: same core + same named sizing stays together,
                # even if unrecorded fixed settings differ.
                design = digest([spec['topology_group'], sorted((k.lower(), format(v, '.12g')) for k, v in env.items())])
                sid = spec['source_dataset'] + ':' + source_info['sha256'][:16] + ':' + str(position)
                records.append({'sample_id': sid, 'source_dataset': spec['source_dataset'], 'topology_id': key,
                    'topology_group': spec['topology_group'], 'connectivity_group': spec['connectivity_group'],
                    'circuit_family': spec['circuit_family'], 'process': spec['process'], 'PDK': spec['process'],
                    'temperature': spec['temperature'], 'supply_voltage': supply, 'testbench_id': spec['testbench_id'],
                    'compatibility_group': spec['compatibility_group'], 'graph': spec['graph'],
                    'device_features': key + ':feature_recipe_v1', 'input_complete': spec['input_complete'],
                    'source_path': spec['csv'], 'source_sha256': source_info['sha256'], 'source_row': position,
                    'parameter_names': spec['parameters'], 'parameters': parameters, 'metric_names': spec['metric_names'],
                    'raw_performance': raw, 'performance': perf, 'mask': mask, 'design_group': design,
                    'record_hash': digest([key, parameters, raw])})
            if records:
                writer.write_table(pa.Table.from_pylist(records, schema=SCHEMA))
                stats['accepted_rows'] += len(records)
    finally:
        writer.close()
    __import__('pathlib').Path(str(dest) + '.partial').replace(dest)
    pd.DataFrame(rejected, columns=['source_path', 'source_row', 'reason']).to_parquet(output / 'converted' / (key + '_rejected.parquet'), index=False)
    dump(statsfile, stats)
    return stats
