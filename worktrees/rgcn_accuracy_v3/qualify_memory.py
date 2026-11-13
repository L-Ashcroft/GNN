"""Validate microbatch loss equivalence on real TRAIN graphs, no test access."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
import sys
import time
import psutil
import numpy as np
from common import ROOT, OUT, atomic, read


def main():
    if psutil.virtual_memory().available < 1.4 * 2**30: raise SystemExit(75)
    folder = ROOT / 'worktrees/rgcn_memory_safe_id_v2'; sys.path.insert(0, str(folder))
    import runtime
    import torch
    from data import FoldData
    from model import loss_by_contract, weighted_loss
    from train import build_model
    from src.model.execution import prepare_batch
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    cfg = read(folder / 'config.json'); reader = FoldData(cfg['fold_id'], 1, verify=True)
    stats = read(ROOT / cfg['output'] / 'normalization.json'); model = build_model(reader, stats, cfg)
    saved = torch.load(ROOT / cfg['initial_checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(saved['model'], strict=True); del saved
    chosen = []
    for source in ['FALCON', 'AICircuit', 'LOADBench']:
        number = next(i for i, n in enumerate(reader.topologies) if n.startswith(source + '__'))
        chosen.extend(np.flatnonzero(reader.refs['train'][:, 0] == number)[:4].tolist())
    chosen = np.array(chosen); counts = reader.counts('train', chosen); model.eval()
    rss_before = psutil.Process().memory_info().rss; begin = time.perf_counter()
    b = reader.batch('train', chosen); _, per = loss_by_contract(model(b), b, model.head)
    full = weighted_loss(per, prepare_batch(b).loss_slices, counts); full.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    expected = float(full.detach()); model.zero_grad(set_to_none=True); del b, per, full
    total = 0.
    for off in range(0, len(chosen), 3):
        b = reader.batch('train', chosen[off:off + 3]); _, per = loss_by_contract(model(b), b, model.head)
        part = weighted_loss(per, prepare_batch(b).loss_slices, counts); part.backward(); total += float(part.detach())
    maximum = 0.; ratio = 0.
    for n, p in model.named_parameters():
        if n not in grads: assert p.grad is None; continue
        delta = abs(p.grad - grads[n]); maximum = max(maximum, float(delta.max()))
        ratio = max(ratio, float((delta / (2e-5 + 2e-4 * abs(grads[n]))).max()))
    assert abs(total - expected) < 2e-6 and ratio <= 1.
    record = {'passed': True, 'real_train_graphs': len(chosen), 'sources': 3, 'full_loss': expected, 'micro_loss': total,
              'max_gradient_absolute_error': maximum, 'max_tolerance_ratio': ratio, 'gradient_atol': 2e-5, 'gradient_rtol': 2e-4,
              'dropout_disabled_for_partition_equivalence_only': True, 'normal_training_dropout_unchanged': True,
              'RSS_MiB': psutil.Process().memory_info().rss / 2**20, 'RSS_increase_MiB': (psutil.Process().memory_info().rss - rss_before) / 2**20,
              'seconds': time.perf_counter() - begin, 'test_accessed': False, 'original_128_microbatch_not_benchmarked': True,
              'effective_training_batch': 256, 'new_microbatch': 16}
    atomic(OUT / 'memory_checks.json', record); print(record); reader.close()


if __name__ == '__main__': main()
