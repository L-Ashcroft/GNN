"""CLI shared by each source adapter; common schema is never reimplemented."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, read, dump
from src.catalog import prepare
from src.convert import convert
from src.load_analoggym import audit


def main(source=None):
    p = argparse.ArgumentParser()
    if source is None:
        p.add_argument('source', choices=['FALCON', 'AICircuit', 'AnalogGym'])
    p.add_argument('--output', default='data/processed')
    p.add_argument('--limit-per-topology', type=int)
    a = p.parse_args()
    source = source or a.source
    output = (ROOT / a.output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Output must remain in workspace')
    output.mkdir(parents=True, exist_ok=True)
    if source == 'AnalogGym':
        result = audit()
        dump(output / 'analoggym_conversion_status.json', result)
        print(result['status'], 'accepted_samples=0')
        return
    specs = read(output / 'catalog.json') if (output / 'catalog.json').exists() else prepare(output)[0]
    for s in specs:
        if s['source_dataset'] == source:
            print(convert(s, output, a.limit_per_topology))


if __name__ == '__main__':
    main()
