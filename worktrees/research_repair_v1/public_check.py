"""Small real-TRAIN forward/backward and exact protocol role checks."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
import sys
import argparse
from core import ROOT, OUT, atomic, read, sha, gate


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('job'); args = ap.parse_args(); gate(1.6)
    folder = ROOT / 'worktrees' / args.job; cfg = read(folder / 'config.json')
    sys.path.insert(0, str(folder))
    import runtime
    import torch
    import numpy as np
    from data import FoldData
    from fit import fit_stats
    from train import build_model
    from model import loss_by_contract
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True); torch.manual_seed(42)
    reader = FoldData(cfg['fold_id'], 2, verify=True)
    refs = reader.refs
    sets = {role: set(map(tuple, value.tolist())) for role, value in refs.items()}
    assert not sets['train'] & sets['validation'] and not sets['train'] & sets['test'] and not sets['validation'] & sets['test']
    assert all(len(sets[r]) == len(refs[r]) for r in refs)
    assert sum(len(v) for v in sets.values()) == 500000
    stats = fit_stats(reader, ROOT / cfg['output'] / 'normalization.json')
    model = build_model(reader, stats, cfg)
    # At least one real train graph per source. No test tensor construction.
    selected = []
    for source in ['FALCON', 'AICircuit', 'LOADBench']:
        number = next(i for i, name in enumerate(reader.topologies) if name.startswith(source + '__') and i in refs['train'][:, 0])
        selected.append(int(np.flatnonzero(refs['train'][:, 0] == number)[0]))
    batch = reader.batch('train', np.array(selected)); result = model(batch, return_states=True)
    loss, groups = loss_by_contract(result['prediction'], batch, model.head); loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert cfg['experiment_kind'].startswith('ID') or not reader.held_topologies & {reader.topologies[int(n)] for n in refs['train'][:, 0]}
    record = {'state': 'complete', 'passed': True, 'job': args.job, 'training_graphs': 3, 'counts': {r: len(v) for r, v in refs.items()},
              'parameters': sum(p.numel() for p in model.parameters()), 'finite_loss': float(loss.detach()),
              'finite_gradients': True, 'exact_role_disjoint': True, 'test_batches_constructed': 0,
              'selection_metric': cfg['selection_metric'], 'config_sha256': sha(folder / 'config.json')}
    atomic(ROOT / cfg['output'] / 'preflight.json', record); print(record); reader.close()


if __name__ == '__main__':
    main()
