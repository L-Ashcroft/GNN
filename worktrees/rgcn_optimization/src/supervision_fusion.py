"""Conservative target routing. Does not train models or infer label equivalence."""
import hashlib
import json
import math


class SupervisionRouter:
    """Keep topology, provenance, testbench, PVT and metric semantics separate.

    Same units or slot numbers never grant supervision compatibility. This
    default deliberately over-separates targets until a reviewed equivalence
    mapping supports sharing. Missingness does not change the output head.
    """
    def __init__(self, catalog):
        self.catalog = catalog

    def describe(self, row):
        spec = self.catalog[row['topology_id']]
        if row['source_dataset'] != spec['source_dataset']:
            raise ValueError('Source does not match topology catalog')
        semantics = [None] * 16
        units = [None] * 16
        originals = [None] * 16
        for rule in spec['metric_mapping']:
            i = rule['slot']
            if not 0 <= i < 16 or semantics[i] is not None:
                raise ValueError('Invalid or duplicate metric slot')
            semantics[i] = rule['semantic_id']
            units[i] = rule['unit']
            originals[i] = rule['original_name']
        if not all(len(row[k]) == 16 for k in ('performance', 'mask', 'semantic_ids')):
            raise ValueError('Expected sixteen performance/mask/semantic slots')
        for i, (value, known, semantic) in enumerate(zip(row['performance'], row['mask'], row['semantic_ids'])):
            if known:
                if value is None or not math.isfinite(float(value)) or semantic != semantics[i] or semantic is None:
                    raise ValueError('Known label has invalid value or incompatible semantic ID')
            elif value is not None or semantic is not None:
                raise ValueError('Mask=False is reserved for absent labels')
        # Unknown historical conditions are isolated by source topology, never
        # treated as verified matches to another topology's unknown conditions.
        contract = {k: row.get(k) for k in ('source_dataset', 'topology_id', 'compatibility_group',
                    'testbench_id', 'process', 'PDK', 'temperature', 'supply_voltage')}
        contract.update(target_semantic_ids=semantics, target_units=units,
                        catalog_conditions=spec.get('conditions', {}))
        encoded = json.dumps(contract, sort_keys=True, allow_nan=False, separators=(',', ':'))
        return {'supervision_group': hashlib.sha256(encoded.encode()).hexdigest(),
                'contract': contract, 'target_semantic_ids': semantics,
                'target_units': units, 'original_names': originals}

    def batch(self, rows):
        rows = list(rows)
        if not rows:
            raise ValueError('Empty supervision batch')
        descriptions = [self.describe(row) for row in rows]
        if len({d['supervision_group'] for d in descriptions}) != 1:
            raise ValueError('Incompatible supervision groups: route to separate heads/losses; Mask cannot resolve semantic differences')
        return {**descriptions[0], 'samples': rows,
                'performance': [r['performance'] for r in rows],
                'mask': [r['mask'] for r in rows]}

    def batches(self, samples, batch_size=128):
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        pending = {}
        for row in samples:
            key = self.describe(row)['supervision_group']
            group = pending.setdefault(key, [])
            group.append(row)
            if len(group) == batch_size:
                yield self.batch(group)
                del pending[key]
        for key in sorted(pending):
            yield self.batch(pending[key])
