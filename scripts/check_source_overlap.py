"""Full source overlap audit, before sampling. Numeric equality is not semantic equivalence."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.common import ROOT, read, dump

PARAM_ALIASES = {'FALCON__DBAMixer': {'WN1_RF': 'WRF', 'WN2_LO': 'WLO'}}
METRIC_ALIASES = {'PowerConsumption': 'DCPowerConsumption', 'IFVoltageSwing': 'VoltageSwing',
                  'LargeSignalPowerGain': 'PowerGain', 'GTMax': 'PowerGain', 'S11Min': 'S11', 'NFMin': 'NoiseFigure'}


def main():
    catalog = read(ROOT / 'data/processed/catalog.json')
    pairs = []
    within = []
    for spec in catalog:
        frame = pd.read_csv(ROOT / spec['csv'])
        duplicate = frame.duplicated(spec['parameters'], keep=False)
        if duplicate.any():
            repeated = frame.loc[duplicate]
            within.append({'topology_id': spec['topology_id'], 'rows_with_repeated_parameter_vector': int(duplicate.sum()),
                'repeated_parameter_groups': repeated.groupby(spec['parameters'], dropna=False).ngroups,
                'input_complete': spec['input_complete'],
                'interpretation': 'Exact full rows were checked separately. Different labels for the same recorded parameters indicate unrecorded settings or measurement differences; never average them or put them in different splits.'})
    for a in catalog:
        if a['source_dataset'] != 'FALCON':
            continue
        for b in catalog:
            if b['source_dataset'] != 'AICircuit' or a['topology_group'] != b['topology_group']:
                continue
            aliases = PARAM_ALIASES.get(a['topology_id'], {})
            anames = [aliases.get(p, p).lower() for p in a['parameters']]
            bnames = [p.lower() for p in b['parameters']]
            item = {'a': a['topology_id'], 'b': b['topology_id'], 'same_full_connectivity': a['connectivity_group'] == b['connectivity_group'],
                    'same_mos_core': True, 'parameter_aliases_for_overlap_only': aliases,
                    'warning': 'Matching named sizes or numeric labels does not verify process/fixed parameters/testbench equivalence.'}
            if set(anames) != set(bnames):
                item['status'] = 'parameter_sets_differ'; pairs.append(item); continue
            frames = []
            metricnames = []
            for spec, names in [(a, anames), (b, bnames)]:
                frame = pd.read_csv(ROOT / spec['csv'])
                p = frame[spec['parameters']].copy(); p.columns = names
                keys = [tuple(format(float(v), '.12g') for v in row) for row in p[sorted(names)].to_numpy()]
                metricmap = {METRIC_ALIASES.get(m, m): m for m in spec['metric_names']}
                signatures = {k: tuple(format(float(v), '.12g') for v in row) for k, row in zip(keys, frame[[metricmap[m] for m in sorted(metricmap)]].to_numpy())}
                frames.append(signatures); metricnames.append(sorted(metricmap))
            common = set(frames[0]) & set(frames[1])
            item['shared_named_parameter_vectors'] = len(common)
            item['same_complete_numeric_label_vector'] = sum(frames[0][k] == frames[1][k] for k in common) if metricnames[0] == metricnames[1] else None
            item['numeric_metric_aliases_for_comparison_only'] = METRIC_ALIASES
            item['status'] = 'audited'
            pairs.append(item)
    dump(ROOT / 'reports/cross_source_overlap.json', {'rounding': '12 significant digits for candidate grouping only; stored data is unchanged',
        'pairs': pairs, 'within_source_repeated_parameters': within,
        'policy': 'Semantic definitions remain source scoped. Equal sized related designs are grouped for leakage control, not averaged or relabeled.'})
    for p in pairs:
        print(p['a'], p['b'], p.get('shared_named_parameter_vectors'), p.get('same_complete_numeric_label_vector'))


if __name__ == '__main__':
    main()
