"""Strict source-dialect adapters; no tensors, neural layers or silent dropping.

The original parsers are pinned by the AnalogGym campaign and stay unchanged.
"""
from pathlib import Path
import re
from ..netlist import parse_spectre, make_graph, evaluate, attributes


def read_logical_lines(path):
    """Join SPICE '+' continuation lines; reject orphan continuations."""
    result=[];current=''
    for raw in Path(path).read_text(encoding='utf-8').splitlines():
        line=raw.strip()
        if not line or line.startswith('*'):continue
        if line.startswith('+'):
            if not current:raise ValueError('Continuation without preceding SPICE line')
            current+=' '+line[1:].strip()
        else:
            if current:result.append(current)
            current=line
    if current:result.append(current)
    return result


def spice_number(value):
    """SPICE suffixes: M is milli; Spectre expressions use their own parser."""
    match=re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([a-zA-Z]*)',value.strip())
    if not match:raise ValueError('Invalid SPICE number: '+value)
    scale={'':1.,'f':1e-15,'p':1e-12,'n':1e-9,'u':1e-6,'m':1e-3,'k':1e3,'meg':1e6,'g':1e9,'t':1e12}
    suffix=match[2].lower()
    if suffix not in scale:raise ValueError('Unsupported suffix: '+suffix)
    return float(match[1])*scale[suffix]


def parse_netlist(path,dialect):
    """Return existing audited graph; caller must explicitly choose dialect."""
    text=Path(path).read_text(encoding='utf-8')
    if dialect=='spectre':return make_graph(parse_spectre(text))
    if dialect=='loadbench_combined':
        from ..load_loadbench import graph_from_combined
        return graph_from_combined('\n'.join(read_logical_lines(path)))[0]
    raise ValueError('Unsupported netlist dialect: '+dialect)
