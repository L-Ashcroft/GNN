"""Resumable native Windows pipeline; never launches upstream simulation/training."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, dump, read, rel
from src.catalog import prepare
from src.convert import convert
from src.merge_sample import merge_sample
from src.splits import make_splits
from src.normalization import fit_training_scalers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit-per-topology', type=int)
    parser.add_argument('--target', type=int, default=500000)
    parser.add_argument('--output', default='data/processed')
    parser.add_argument('--splits', default='data/splits')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--from-converted', help='Reuse an existing audited converted corpus without copying raw rows')
    parser.add_argument('--require-complete-inputs', action='store_true')
    args = parser.parse_args()
    output, split_dir = (ROOT / args.output).resolve(), (ROOT / args.splits).resolve()
    if not output.is_relative_to(ROOT) or not split_dir.is_relative_to(ROOT):
        raise ValueError('All outputs must remain inside workspace')
    if args.target <= 0 or args.limit_per_topology is not None and args.limit_per_topology < 1:
        raise ValueError('Counts must be positive')
    output.mkdir(parents=True, exist_ok=True)
    converted_dir = None
    if args.from_converted:
        source = (ROOT / args.from_converted).resolve()
        if not source.is_relative_to(ROOT):
            raise ValueError('Source must be inside workspace')
        specs = read(source / 'catalog.json')
        if args.require_complete_inputs:
            specs = [s for s in specs if s['input_complete']]
        dump(output / 'catalog.json', specs)
        converted_dir = source / 'converted'
        dump(output / 'conversion_reference.json', {'converted_dir': rel(converted_dir),
            'source_catalog': rel(source / 'catalog.json'), 'require_complete_inputs': args.require_complete_inputs})
        keys = {s['topology_id'] for s in specs}
        statistics = [s for s in read(source / 'conversion_statistics.json') if s['topology_id'] in keys]
    else:
        specs, _, _ = prepare(output)
        statistics = []
        for s in specs:
            stat = convert(s, output, args.limit_per_topology)
            statistics.append(stat)
            print(s['topology_id'], stat['accepted_rows'], 'accepted,', stat['rejected_rows'], 'rejected', flush=True)
        if args.require_complete_inputs:
            specs = [s for s in specs if s['input_complete']]
            keys = {s['topology_id'] for s in specs}
            statistics = [s for s in statistics if s['topology_id'] in keys]
            dump(output / 'catalog.json', specs)
    dump(output / 'conversion_statistics.json', statistics)
    print(merge_sample(output, args.target, args.seed, converted_dir), flush=True)
    make_splits(output, split_dir, args.seed)
    fit_training_scalers(output, split_dir)
    print('Conversion, sampling, splits and train-only scaling complete.', flush=True)


if __name__ == '__main__':
    main()
