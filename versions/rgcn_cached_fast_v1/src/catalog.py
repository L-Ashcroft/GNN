import re
import pandas as pd
import networkx as nx
from .common import ROOT, dump, rel, digest
from .load_falcon import discover as falcon
from .load_aicircuit import discover as aicircuit
from .load_analoggym import audit as analog_audit
from .netlist import parse_spectre, make_graph, feature_expressions, FEATURE_NAMES, evaluate, topology_graph, topology_hash, expression
import ast


def prepare(output):
    specs = [*falcon(), *aicircuit()]
    graphs = {}
    mapping = {'version': 1, 'max_dimensions': 16, 'mask_semantics': 'true iff source metric exists and is finite; invalid rows are quarantined, never hidden with mask',
               'joining_rule': 'Only semantic_id equality permits sharing a target. Equal slots or similar names never imply compatibility.',
               'cross_dataset_approved_pairs': [], 'groups': {}}
    for s in specs:
        key = s['source_dataset'] + '__' + s['source_topology']
        s['topology_id'] = key
        s['graph'] = rel(output / 'graphs' / (key + '.json'))
        df = pd.read_csv(ROOT / s['csv'], nrows=1)
        s['metric_names'] = [c for c in df.columns if c in s['metric_units']]
        s.setdefault('parameters', [c for c in df.columns if c not in s['metric_names']])
        if set(df.columns) != set(s['parameters'] + s['metric_names']):
            raise ValueError('Unclassified columns: ' + key)
        g = make_graph(parse_spectre((ROOT / s['netlist']).read_text()))
        g.update(source_netlist=s['netlist'], feature_names=FEATURE_NAMES, feature_expressions=feature_expressions(g),
                 netlist_sha256=__import__('hashlib').sha256((ROOT / s['netlist']).read_bytes()).hexdigest())
        s['full_connectivity_hash'] = topology_hash(g)
        s['mos_core_hash'] = topology_hash(g, True)
        unresolved = {}
        env = {**s['defaults'], **df[s['parameters']].iloc[0].to_dict()}
        for d, row in zip(g['devices'], g['feature_expressions']):
            for name, exp in zip(FEATURE_NAMES, row):
                if exp is not None:
                    try:
                        evaluate(exp, env)
                    except (KeyError, SyntaxError, ValueError) as e:
                        unresolved[d['name'] + '.' + name] = {'expression': exp, 'reason': str(e)}
        s['unresolved_device_features'] = unresolved
        s['input_complete'] = not unresolved
        s['testbench_id'] = key + ':' + digest([s['testbench_files'], s['netlist'], s['defaults'], s['temperature']])[:12]
        s['compatibility_group'] = key  # Conservative until extraction/conditions are demonstrably equal.
        s['parameter_units'] = {}
        for p in s['parameters']:
            uses = set()
            for row in g['feature_expressions']:
                for feature, exp in zip(FEATURE_NAMES, row):
                    if exp and p in {n.id for n in ast.walk(expression(exp)) if isinstance(n, ast.Name)}:
                        uses.add(feature)
            s['parameter_units'][p] = {'source_unit': 'SI', 'factor': 1.0, 'feature_uses': sorted(uses), 'evidence': s['netlist']}
        s['supply_voltage_default'] = None
        s['supply_voltage_expression'] = None
        # Only explicit top-level VDD source, never a guessed process nominal voltage.
        for d in g['devices']:
            if d['model'] == 'vsource' and d['nets'] and d['nets'][0] in {'VDD', 'vdd'}:
                exp = d['attributes'].get('dc', '0')
                s['supply_voltage_expression'] = exp
                try:
                    s['supply_voltage_default'] = evaluate(exp, s['defaults'])
                except KeyError:
                    pass
        s['num_devices'] = g['num_devices']; s['num_nets'] = g['num_nets']; s['num_edges'] = g['num_edges']
        s['graph_representation'] = g['representation']
        graphs[key] = g
        metrics = []
        for slot, name in enumerate(s['metric_names']):
            if slot >= 16:
                raise ValueError('Need explicit overflow policy')
            unit = s['metric_units'][name]
            reason = 'Source topology scoped: extraction and complete test conditions are not verified across topologies/datasets.'
            if name == 'NFMin':
                reason += ' Ocean adds +1 to ymin(dB10(F)); cannot equal conventional NF.'
            if name == 'PSAT' and s['source_dataset'] == 'AICircuit':
                reason += ' Ocean uses compressionVRICurves gcomp=1; cannot claim saturation power.'
            if name == 'Bandwidth' and s['source_dataset'] == 'AICircuit' and s['source_topology'] == 'CVA':
                reason += ' Ocean denominator /Vin- is absent from supplied top-level netlist; unverified source extraction.'
            metrics.append({'slot': slot, 'original_name': name, 'semantic_id': key + '::' + name,
                            'source_unit': unit, 'unit': '1' if unit == '%' else unit, 'factor': .01 if unit == '%' else 1.,
                            'cross_dataset_compatible': False, 'reason': reason,
                            'evidence': s['testbench_files'] or ['third_party/FALCON/config/data_config.yaml']})
        mapping['groups'][key] = metrics
        dump(ROOT / s['graph'], g)
    # Exact connectivity classes: WL is only a candidate filter; VF2 establishes equality.
    # MOS cores are a conservative additional grouping to keep similar circuits together in LOTO.
    for field, mos in [('connectivity_group', False), ('topology_group', True)]:
        representatives = []
        for s in specs:
            g = topology_graph(graphs[s['topology_id']], mos)
            found = None
            for other, og in representatives:
                hashfield = 'mos_core_hash' if mos else 'full_connectivity_hash'
                if s[hashfield] == other[hashfield] and nx.is_isomorphic(g, og, node_match=nx.algorithms.isomorphism.categorical_node_match('color', '')):
                    found = other[field]; break
            s[field] = found or ('core_' if mos else 'conn_') + s['topology_id']
            if found is None:
                representatives.append((s, g))
    dump(output / 'catalog.json', specs)
    dump(ROOT / 'config/performance_mapping.json', mapping)
    dump(ROOT / 'reports/analoggym_audit.json', analog_audit())
    return specs, graphs, mapping
