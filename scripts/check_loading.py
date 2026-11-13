"""Integration smoke test for both ready views and explicit blocked cross-source protocols."""
from pathlib import Path
import sys
import math
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import load_split
from src.common import ROOT, read


def main():
    checked = 0
    for output, splits in [('data/processed', 'data/splits'), ('data/strict', 'data/strict_splits')]:
        protocols = read(ROOT / splits / 'protocols.json')
        for name, meta in protocols.items():
            if meta['status'] != 'ready':
                try:
                    next(load_split(name, 'test', output=output, split_root=splits))
                except ValueError as error:
                    assert 'No cross-source metric' in str(error)
                else:
                    raise AssertionError('Blocked cross-source protocol was loadable')
                continue
            for role in ['train', 'validation', 'test']:
                item = next(load_split(name, role, output=output, split_root=splits))
                assert item['input_complete']
                assert item['graph']['representation'] == 'net_device_multigraph_v2'
                assert item['graph']['num_nodes'] == item['graph']['num_nets']
                assert len(item['edge_features']) == len(item['edge_port_pairs']) == item['graph']['num_edges']
                assert all(len(x) == item['graph']['num_edges'] for x in item['edge_index'])
                assert len(item['performance']) == len(item['mask']) == len(item['semantic_ids']) == 16
                assert all((v is not None) == known for v, known in zip(item['performance'], item['mask']))
                assert all(math.isfinite(v) for v in item['performance'] if v is not None)
                if role == 'train':
                    assert item['sampling_weight'] > 0
                checked += 1
    print('Loaded and checked', checked, 'protocol/role/view combinations; blocked cross-source protocols correctly refused.')


if __name__ == '__main__':
    main()
