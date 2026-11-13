"""Strict Spectre parser and net-node, device-edge multigraph.

Each terminal gets its own incidence even when S and B touch the same net.
Non-MOS multi-terminal primitives retain indexed ports rather than invented roles.
Subcircuits are flattened with lexical local nets; only node 0 is global unless
declared global. GND and 0 are deliberately NOT shorted by their spelling.
"""
import ast
import math
import operator
import re
from functools import lru_cache
import networkx as nx
from .common import digest

SCALE = {'T': 1e12, 'G': 1e9, 'M': 1e6, 'K': 1e3, 'k': 1e3,
         'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15, 'a': 1e-18}
NUM = re.compile(r'(?<![\w.])((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([TGMKkmunpfa])\b')
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}


@lru_cache(maxsize=16384)
def expression(text):
    return ast.parse(NUM.sub(lambda m: str(float(m[1]) * SCALE[m[2]]), text.strip().strip("'\"{}")), mode='eval').body


def evaluate(text, env):
    def walk(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name):
            return float(env[n.id])
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](walk(n.left), walk(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return (-1 if isinstance(n.op, ast.USub) else 1) * walk(n.operand)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'sqrt' and len(n.args) == 1 and not n.keywords:
            return math.sqrt(walk(n.args[0]))
        raise ValueError('Unsupported expression: ' + text)
    value = walk(expression(str(text)))
    if not math.isfinite(value):
        raise ValueError('nonfinite expression')
    return value


def attributes(text):
    matches = list(re.finditer(r'(\w+)\s*=\s*', text))
    return {m[1]: text[m.end():matches[i+1].start() if i+1 < len(matches) else len(text)].strip()
            for i, m in enumerate(matches)}


def logical_lines(text):
    buf = ''
    for line in text.splitlines():
        line = line.split('//')[0].strip()
        if not line or line.startswith('*'):
            continue
        if line.endswith('\\'):
            buf += line[:-1] + ' '
        else:
            yield buf + line
            buf = ''
    if buf:
        raise ValueError('Unterminated continuation')


def parse_spectre(text):
    definitions = {}
    top = []
    current = top
    globals_ = {'0'}
    for line in logical_lines(text):
        fields = line.split()
        if fields[0] == 'global':
            globals_.update(fields[1:]); continue
        if fields[0] == 'subckt':
            if current is not top:
                raise ValueError('Nested subckt definition')
            definitions[fields[1]] = (fields[2:], [])
            current = definitions[fields[1]][1]; continue
        if fields[0] == 'ends':
            current = top; continue
        m = re.fullmatch(r'(\S+)\s+\(([^)]*)\)\s+(\S+)(.*)', line)
        if m:
            current.append({'name': m[1], 'nets': m[2].split(), 'model': m[3], 'attributes': attributes(m[4])})
        elif len(fields) > 1 and fields[1] == 'mutual_inductor':
            current.append({'name': fields[0], 'nets': [], 'model': 'mutual_inductor', 'attributes': attributes(' '.join(fields[2:]))})
        elif fields[0] not in {'simulator', 'parameters'}:
            raise ValueError('Unparsed netlist line: ' + line)
    if current is not top:
        raise ValueError('Unclosed subckt')
    devices = []
    def expand(items, prefix='', ports=None, stack=()):
        ports = ports or {}
        def net(n):
            return ports.get(n, n if n in globals_ or not prefix else prefix + ':' + n)
        for item in items:
            name = prefix + '/' + item['name'] if prefix else item['name']
            nets = [net(n) for n in item['nets']]
            if item['model'] in definitions:
                model = item['model']
                if model in stack:
                    raise ValueError('recursive subckt')
                formal, body = definitions[model]
                if len(formal) != len(nets) or item['attributes']:
                    raise ValueError('Subckt arity mismatch or unsupported local parameters')
                expand(body, name, dict(zip(formal, nets)), stack + (model,))
            else:
                mos = item['model'] in {'nmos', 'pmos'}
                if mos and len(nets) != 4:
                    raise ValueError('MOS must have D G S B')
                roles = ['D', 'G', 'S', 'B'] if mos else (['P', 'N'] if len(nets) == 2 else [f'P{i+1}' for i in range(len(nets))])
                refs = {k: (prefix + '/' if prefix else '') + v for k, v in item['attributes'].items() if k in {'ind1', 'ind2'}}
                devices.append({**item, 'name': name, 'nets': nets, 'ports': roles, 'device_references': refs})
    expand(top)
    if not devices or len({d['name'] for d in devices}) != len(devices):
        raise ValueError('No devices or duplicate instance names')
    return devices


def make_graph(devices):
    nets = sorted({n for d in devices for n in d['nets']})
    idx = {n: i for i, n in enumerate(nets)}
    terminals = [{'device': i, 'net': idx[n], 'port': role} for i, d in enumerate(devices) for role, n in zip(d['ports'], d['nets'])]
    edges = []
    for i, d in enumerate(devices):
        for a in range(len(d['ports'])):
            for b in range(a + 1, len(d['ports'])):
                # FALCON uses a star for baluns, pair expansion for MOS. Extend
                # MOS to all four terminals: DG, DS, DB, GS, GB, SB.
                if d['model'] == 'balun' and a != 0:
                    continue
                pair = [d['ports'][a], d['ports'][b]]
                edges.append({'source': idx[d['nets'][a]], 'target': idx[d['nets'][b]],
                              'device': i, 'name': d['name'], 'ports': pair,
                              'component': d['model'] + '_' + ''.join(pair)})
    didx = {d['name']: i for i, d in enumerate(devices)}
    refs = [{'source': i, 'target': didx[name], 'role': role} for i, d in enumerate(devices) for role, name in d['device_references'].items()]
    return {'representation': 'net_device_multigraph_v2', 'directed': False,
            'port_policy': 'MOS all six port pairs including bulk; balun first-port star; other devices all port pairs; tied ports retain self-loops',
            'nets': nets, 'devices': devices, 'edges': edges,
            'edge_index': [[e['source'] for e in edges], [e['target'] for e in edges]],
            'edge_device_index': [e['device'] for e in edges],
            'device_terminals': terminals, 'device_references': refs,
            'num_nets': len(nets), 'num_devices': len(devices), 'num_nodes': len(nets),
            'num_terminals': len(terminals), 'num_edges': len(edges),
            'num_device_references': len(refs)}


def topology_graph(graph, mos_only=False):
    # Separate terminal nodes make tied terminals and their roles visible to isomorphism.
    g = nx.Graph()
    for i, d in enumerate(graph['devices']):
        if mos_only and d['model'] not in {'nmos', 'pmos'}:
            continue
        g.add_node(('d', i), color=d['model'])
        for j, (role, net) in enumerate(zip(d['ports'], d['nets'])):
            g.add_node(('p', i, j), color='port:' + role)
            g.add_node(('n', net), color='net')
            g.add_edge(('d', i), ('p', i, j)); g.add_edge(('p', i, j), ('n', net))
    if not mos_only:
        for j, r in enumerate(graph['device_references']):
            g.add_node(('ref', j), color=r['role'])
            g.add_edge(('d', r['source']), ('ref', j)); g.add_edge(('ref', j), ('d', r['target']))
    return g


def topology_hash(graph, mos_only=False):
    return nx.weisfeiler_lehman_graph_hash(topology_graph(graph, mos_only), node_attr='color', iterations=5)


FEATURE_NAMES = ['width_m', 'length_m', 'multiplicity', 'resistance_ohm', 'capacitance_F', 'inductance_H',
                 'dc_voltage_V', 'dc_current_A', 'ac_magnitude', 'frequency_Hz', 'area_source_m2', 'area_drain_m2',
                 'perimeter_source_m', 'perimeter_drain_m', 'quality_factor', 'phase_degree', 'coupling_coefficient']


def feature_expressions(graph):
    matrix = []
    for d in graph['devices']:
        a, model = d['attributes'], d['model']
        mapping = {'m': 'multiplicity', 'as': 'area_source_m2', 'ad': 'area_drain_m2', 'ps': 'perimeter_source_m',
                   'pd': 'perimeter_drain_m', 'r': 'resistance_ohm', 'c': 'capacitance_F', 'mag': 'ac_magnitude',
                   'freq': 'frequency_Hz', 'fq': 'frequency_Hz', 'q': 'quality_factor', 'phase': 'phase_degree', 'coupling': 'coupling_coefficient'}
        if model in {'nmos', 'pmos'}:
            mapping.update(w='width_m', l='length_m')
        if model == 'inductor':
            mapping['l'] = 'inductance_H'
        if model in {'vsource', 'isource'}:
            mapping['dc'] = 'dc_voltage_V' if model == 'vsource' else 'dc_current_A'
        row = {mapping[k]: v for k, v in a.items() if k in mapping}
        if model in {'nmos', 'pmos'}:
            row.setdefault('multiplicity', '1')
        matrix.append([row.get(n) for n in FEATURE_NAMES])
    return matrix


def resolve_device_attributes(device, env):
    """Preserve parameters beyond the compact feature vector, e.g. RF port dBm.

    Categorical source/region settings and inductor instance references are not
    numeric expressions. Unknown numeric values stay explicit None.
    """
    result = {}
    for key, exp in device['attributes'].items():
        if key in {'region', 'type', 'fundname', 'ind1', 'ind2'}:
            result[key] = exp.strip('"')
        else:
            try:
                result[key] = evaluate(exp, env)
            except KeyError:
                result[key] = None
    return result
