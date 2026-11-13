"""Pure AST action-schema helper retained for frozen AnalogGym producers.

The completed historical replay audit lives in data/audit/analoggym_replay/.
Its previous CLI is preserved in workspace_cleanup_v2/critical_backup.zip;
obsolete probe scripts are not required to import or train personal models.
"""
import ast
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'third_party/AnalogGym/RGNN_RL'


def action_schema(env_file, graph_class):
    """Only inspect AST; no import of upstream environments, torch, or simulators."""
    env_tree = ast.parse(env_file.read_text(encoding='utf-8'))
    method = next(n for n in ast.walk(env_tree) if isinstance(n, ast.FunctionDef) and n.name == '_do_simulation')
    unpack = next(n for n in method.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Name) and n.value.id == 'action')
    names = [n.id for n in unpack.targets[0].elts]
    integer = [n.targets[0].id for n in method.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
               and isinstance(n.value.func, ast.Name) and n.value.func.id == 'int']
    mapping = {}
    for n in ast.walk(method):
        if not isinstance(n, ast.JoinedStr): continue
        prev = ''
        for part in n.values:
            if isinstance(part, ast.Constant): prev = part.value
            elif isinstance(part, ast.FormattedValue) and isinstance(part.value, ast.Name):
                match = re.search(r'([A-Za-z_0-9]+)=$', prev)
                if match: mapping[part.value.id] = match.group(1)
    assert set(names) == set(mapping)
    tree = ast.parse((BASE / 'ckt_graphs.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == graph_class)
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    constants = {}
    for n in init.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant):
            constants[n.targets[0].id] = n.value.value
    bounds = {}
    for n in init.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Attribute) and n.targets[0].attr in ('action_space_low', 'action_space_high'):
            values = n.value.args[0].elts
            bounds[n.targets[0].attr] = [constants[v.id] if isinstance(v, ast.Name) else ast.literal_eval(v) for v in values]
    return names, integer, mapping, bounds
