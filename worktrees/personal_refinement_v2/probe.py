"""Small-memory real training of a physical skip baseline; no Torch or test reads."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(name, '1')
import argparse
import gc
import time
import psutil
import numpy as np
from common import HERE, OUTPUT, ROOT, read, atomic, now
from features import Inputs, ridge, denormalize, metric_rows, summarize


def main(budget):
    data = Inputs(budget); all_rows, means = [], []
    tick = time.perf_counter(); dimensions = {}
    for name in sorted(data.structures):
        if psutil.virtual_memory().available < 320 * 2**20:
            raise MemoryError('Low-memory diagnostic safe gate')
        block = data.block(name, mode='physical'); w, b = ridge(block)
        prediction = denormalize(block['validation_x'] @ w.T + b, block)
        all_rows.extend(metric_rows(name, prediction, block))
        mean = np.broadcast_to(block['normalization']['target_center'], (len(prediction),16))
        means.extend(metric_rows(name, mean, block))
        dimensions[name] = {'active_physical_columns': w.shape[1], 'training_rows': len(block['train_x'])}
        del block, prediction, w, b
        gc.collect()
    result = {'at': now(), 'budget_per_topology': budget, 'ridge_alpha': 10.,
              'training_samples': len(data.allowed), 'validation_labels_additional': 2000,
              'test_samples_used': 0, 'physical_skip': summarize(all_rows),
              'topology_training_mean': summarize(means), 'dimensions': dimensions,
              'seconds': time.perf_counter()-tick, 'RSS_MiB': psutil.Process().memory_info().rss/2**20,
              'interpretation': 'Known-topology parameter baseline; not evidence of GNN or LOTO gains'}
    atomic(OUTPUT / ('physical_probe_'+str(budget)+'.json'), result)
    print({'budget': budget, 'physical_skip': {k:v for k,v in result['physical_skip'].items() if k!='per_target'},
           'topology_training_mean': {k:v for k,v in result['topology_training_mean'].items() if k!='per_target'},
           'seconds':result['seconds'], 'RSS_MiB':result['RSS_MiB']})


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--budget',default='full');args=parser.parse_args()
    main('full' if args.budget=='full' else int(args.budget))
