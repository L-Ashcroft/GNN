"""Finite difference proof for the new loss and actual validation-cache replay."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[key] = '1'
import sys
import numpy as np
from common import ROOT, OUT, META, read, atomic
from decoder import Decoder, transform, physical


def main():
    rng = np.random.default_rng(42); x = rng.normal(size=(7, 4)); y = rng.normal(size=(7, 3)); mask = rng.random(y.shape) > .25
    model = Decoder(x, y, mask, hidden=5); model.p['o'][:] = rng.normal(scale=.1, size=model.p['o'].shape)
    loss, grad = model.gradient(x, np.arange(len(x))); maximum = 0.
    for key, p in model.p.items():
        for index in np.ndindex(p.shape):
            old = p[index]; p[index] = old + 1e-5; a = model.gradient(x, np.arange(len(x)))[0]
            p[index] = old - 1e-5; b = model.gradient(x, np.arange(len(x)))[0]; p[index] = old
            error = abs((a - b) / 2e-5 - grad[key][index]); maximum = max(maximum, error)
    assert maximum < 1e-7
    model.target[~mask] = 1e20; changed, gg = model.gradient(x, np.arange(len(x)))
    assert changed == loss
    for key in grad: np.testing.assert_array_equal(gg[key], grad[key])
    values = np.array([[-1e5, 0., .8], [1e5, 10., -.5]])
    scale = np.array([1e3, 1., 1.]); kinds = ['symlog', 'linear', 'linear']
    np.testing.assert_allclose(physical(transform(values, scale, kinds), scale, kinds), values, rtol=1e-12, atol=1e-12)
    # One independent process loads Torch then exits, so the NumPy optimizer
    # does not retain the framework allocator.
    sys.path.insert(0, str(ROOT / 'versions/rgcn_local_id_v1'))
    sys.path.insert(0, str(ROOT / 'worktrees/autonomous_training_v1'))
    from model import load_model
    from stream_graphs import StreamGraphs
    import torch
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    net, _ = load_model(); reader = StreamGraphs(ROOT, 'data/model_ready/fusion_v2_training_v1',
                         'data/model_ready/fusion_v2_frozen_e19_bf0d71ad', 'versions/rgcn_calibrated_head_v1', capacity=1)
    rows = np.array([0, 137, 4000, 15000, 35000, 50000])
    with torch.inference_mode():
        expected = np.load(META / 'validation/embedding.npy', mmap_mode='r')[rows]
        actual = net.encoder(reader.batch('validation', rows, sort_for_io=False), return_states=True)['embedding'].numpy()
    difference = float(abs(actual - expected).max()); np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    reader.close()
    atomic(OUT / 'checks.json', {'passed': True, 'finite_difference_max_abs': maximum, 'gradient_threshold': 1e-7,
           'missing_labels_no_effect': True, 'unit_transform_round_trip': True,
           'real_validation_graphs': len(rows), 'cached_graph_embedding_max_abs': difference, 'test_labels_read': False})
    print('Checks passed:', maximum, difference)


if __name__ == '__main__': main()
