"""Small real TRAIN-only CPU preflight for the controlled public experiments."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import sys
import json
import time
import gc
import hashlib
import psutil

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/public_evidence_queue_v1'

def main():
    if psutil.virtual_memory().available < 1.4 * 2**30:
        raise SystemExit(75)
    sys.path.insert(0, str(ROOT / 'worktrees/rgcn_memory_safe_id_control_v2'))
    import runtime
    import numpy as np
    import torch
    from data import FoldData
    from fit import fit_stats
    from train import build_model
    from model import loss_by_contract
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    reader = FoldData('paired_ID_same_neural_architecture_seed42', 1, verify=True)
    cfg = json.loads((ROOT / 'worktrees/rgcn_memory_safe_id_control_v2/config.json').read_text())
    stats = fit_stats(reader, ROOT / cfg['output'] / 'normalization.json')
    assert {role: reader.fold['counts'][role] for role in ['train', 'validation', 'test']} == {
        'train': 400083, 'validation': 50861, 'test': 49056}
    possible = []
    for number in np.unique(reader.refs['train'][:, 0]):
        name = reader.topologies[int(number)]
        possible.append((int(number), set(reader.fetch(name)['static']['edge_type'])))
    uncovered = set(range(48))
    chosen = []
    while uncovered:
        number, types = max(possible, key=lambda item: len(item[1] & uncovered))
        assert types & uncovered, 'Training pool does not cover all 48 relations'
        chosen.append(int(np.flatnonzero(reader.refs['train'][:, 0] == number)[0]))
        uncovered -= types
    for source in ['FALCON', 'AICircuit', 'LOADBench']:
        number = next(i for i, n in enumerate(reader.topologies) if n.startswith(source + '__'))
        chosen.append(int(np.flatnonzero(reader.refs['train'][:, 0] == number)[0]))
    chosen = np.array(list(dict.fromkeys(chosen)))
    batch = reader.batch('train', chosen)
    assert set(batch.edge_type.tolist()) == set(range(48))
    results = []
    for job in json.loads((HERE / 'queue.json').read_text())['jobs']:
        if job['kind'] != 'ID':
            continue
        if psutil.virtual_memory().available < 768 * 2**20:
            raise SystemExit(75)
        config = json.loads((ROOT / job['worktree'] / 'config.json').read_text())
        for name in ['protocol', 'fold_id', 'backbone_lr', 'adaptation_lr', 'effective_batch_size',
                     'training_microbatch', 'max_epochs', 'minimum_epochs', 'patience', 'seed', 'selection_metric']:
            assert config[name] == cfg[name], (job['id'], name)
        torch.manual_seed(config['seed'])
        model = build_model(reader, stats, config)
        assert len(model.layers) == 4
        assert model.relation_encoder.num_embeddings == (1 if config['collapse_relations'] else 48)
        assert len(model.films) == (4 if config['use_film'] else 0)
        assert all((layer.edge_update is not None) == config['update_edges'] for layer in model.layers)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01)
        losses = []
        begin = time.perf_counter()
        model.train()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = model(batch, return_states=True)
            assert output['node_states'].shape[1] == output['edge_states'].shape[1] == 128
            if not config['update_edges']:
                assert torch.equal(output['edge_states'], output['initial_edge_states'])
            loss, per_group = loss_by_contract(output['prediction'], batch, model.head)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            assert torch.isfinite(loss) and torch.isfinite(norm)
            losses.append(float(loss.detach()))
            del output, loss, per_group, norm
        results.append({'id': job['id'], 'parameters': sum(p.numel() for p in model.parameters()),
                        'finite_two_training_steps': True, 'losses': losses,
                        'seconds': time.perf_counter() - begin})
        del optimizer, model
        gc.collect()
    receipt = {'passed': True, 'device': 'cpu', 'real_train_samples': len(chosen),
               'relations_in_batch': 48, 'sources': 3, 'dropout': .1, 'deterministic': True,
               'test_labels_accessed': False, 'checkpoint_or_official_results_modified': False,
               'qualification_only_not_experiment_scores': True, 'jobs': results,
               'RSS_MiB': psutil.Process().memory_info().rss / 2**20,
               'queue_sha256': hashlib.sha256((HERE / 'queue.json').read_bytes()).hexdigest()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'preflight.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)
    reader.close()

if __name__ == '__main__':
    main()
