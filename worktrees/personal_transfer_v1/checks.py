"""Real-data cache, transfer, gradients, budget and full-step qualification."""
import copy
import gc
import time
from collections import Counter
import numpy as np
import psutil
import torch
from common import ROOT, HERE, OUTPUT, PRETRAINED, read, atomic, sha, now
from data import Reader
from model import TargetRGCN, transfer, loss
from src.personal_gnn_data import PersonalDataset
from src.model.execution import prepare_batch
from src.train.stage import build_optimizer


def main():
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True); torch.manual_seed(42)
    reader = Reader(verify=True)
    data = PersonalDataset(read(ROOT / 'config/personal_rgcn.json'), verify=True)
    rows = list(data.rows('train', limit_per_topology=2))
    refs = np.array([reader.by_id[r['sample_id']] for r in rows])
    assert len(refs) == 20
    cached, original = reader.batch(refs, 'train'), data.tensorize(rows, 'train')
    for name in ['x', 'edge_index', 'edge_attr', 'edge_type', 'graph_index', 'mask', 'conditions']:
        torch.testing.assert_close(getattr(cached, name), getattr(original, name), rtol=0, atol=0)
    torch.testing.assert_close(cached.targets.float(), original.targets, rtol=0, atol=0)
    assert cached.group_ids == original.group_ids and cached.sample_ids == original.sample_ids
    seen = set()
    budgets = []
    for k in [5, 10, 20, 50, 100, 'full']:
        selected = reader.training_refs(k)
        ids = {reader.index[int(i)]['sample_id'] for i in selected}
        assert seen <= ids; seen = ids
        assert not ids & {reader.index[int(i)]['sample_id'] for i in reader.refs['validation']}
        assert not ids & {reader.index[int(i)]['sample_id'] for i in reader.refs['test']}
        stats = reader.statistics(selected)
        budgets.append({'budget_per_topology': k, 'train_labels': len(selected),
                        'validation_labels_additional': 2000, 'observed_slot_counts': {
                            g: s['count'] for g, s in stats['groups'].items()}})
    try:
        reader.statistics(reader.refs['validation'][:10])
        raise RuntimeError('Validation statistic guard not enforced')
    except AssertionError:
        pass
    stats = reader.statistics(reader.training_refs(100))
    model = TargetRGCN(reader.schema, reader.tasks, stats)
    source = torch.load(PRETRAINED, map_location='cpu', weights_only=True)
    receipt = transfer(model, source)
    old = read(ROOT / 'data/model_ready/fusion_v2_training_v1/schema.json')
    for i, relation in enumerate(old['relations']):
        j = reader.schema['relations'].index(relation)
        torch.testing.assert_close(model.relation_encoder.weight[j], source['model']['relation_encoder.weight'][i], rtol=0, atol=0)
        for number in range(4):
            torch.testing.assert_close(model.layers[number].relation_weight[j],
                                       source['model'][f'layers.{number}.relation_weight'][i], rtol=0, atol=0)
    del source, original, data
    gc.collect()
    optimizer = build_optimizer(model, 'finetune_all', lr_head=3e-4, lr_backbone=2e-4)
    times, losses = [], []
    for _ in range(3):
        model.train(); optimizer.zero_grad(set_to_none=True); tick = time.perf_counter()
        value, _ = loss(model(cached), cached, model.head)
        value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True); optimizer.step()
        times.append(time.perf_counter() - tick); losses.append(float(value.detach()))
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert model.condition_encoder[-1].weight.grad.abs().sum() > 0
    assert model.layers[0].relation_weight.grad.abs().sum() > 0
    # Compare the exact group-equal objective and gradients at fixed weights.
    model.eval(); optimizer.zero_grad(set_to_none=True)
    whole, _ = loss(model(cached), cached, model.head); whole.backward()
    gradients = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    counts = Counter()
    for g, a, b in prepare_batch(cached).loss_slices:
        counts[g] += b - a
    optimizer.zero_grad(set_to_none=True); total = 0.
    for chunk in [refs[:10], refs[10:]]:
        batch = reader.batch(chunk, 'train'); _, parts = loss(model(batch), batch, model.head)
        sizes = {g: b-a for g,a,b in prepare_batch(batch).loss_slices}
        value = torch.stack([v * (sizes[g]/counts[g])/len(counts) for g,v in parts.items()]).sum()
        value.backward(); total += float(value.detach())
    errors = []
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name], atol=3e-5, rtol=3e-4)
            errors.append(float((parameter.grad - gradients[name]).abs().max()))
    assert abs(float(whole.detach()) - total) < 1e-6
    folder = ROOT / 'artifacts/personal_transfer_qualification_v1'
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / 'model.pt'
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint)
    restored = TargetRGCN(reader.schema, reader.tasks, stats)
    restored.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'], strict=True)
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(cached), model(cached), atol=0, rtol=0)
    restored_optimizer = build_optimizer(restored, 'finetune_all', lr_head=3e-4, lr_backbone=2e-4)
    restored_optimizer.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['optimizer'])
    for candidate, optim in [(model, optimizer), (restored, restored_optimizer)]:
        optim.zero_grad(set_to_none=True)
        objective, _ = loss(candidate(cached), cached, candidate.head)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 5., error_if_nonfinite=True)
        optim.step()
    for (name, parameter), (other_name, other) in zip(model.named_parameters(), restored.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(parameter, other, atol=0, rtol=0)
    result = {'passed': True, 'at': now(), 'test_samples_used': 0, 'real_train_batch': 20,
              'cache_matches_original_tensorizer_exactly': True, 'mask_and_split_unchanged': True,
              'name_mapped_pretrained_relations_exact': True, 'transfer_receipt': receipt,
              'all_nested_budgets_covered_and_train_only': budgets,
              'full_vs_microbatch_loss_abs_error': abs(float(whole.detach())-total),
              'full_vs_microbatch_gradient_max_abs_error': max(errors),
              'gradient_tolerance': {'atol': 3e-5, 'rtol': 3e-4},
              'gradient_test_dropout': 'eval mode; training RNG trajectories need not match across microbatch sizes',
              'checkpoint_reload_exact': True, 'AdamW_resume_one_step_exact': True,
              'training_step_seconds': times, 'training_losses': losses,
              'measured_train_samples_per_second': 20 / float(np.mean(times[1:])),
              'RSS_MiB': psutil.Process().memory_info().rss / 2**20,
              'available_RAM_GiB': psutil.virtual_memory().available / 2**30,
              'schema': {'node': 35, 'edge': 90, 'relations': 52, 'conditions': 46},
              'source_checkpoint_sha256': sha(PRETRAINED)}
    if (HERE / 'checks.json').exists():
        previous = read(HERE / 'checks.json')
        if 'batch64_cpu_microbenchmark' in previous:
            result['batch64_cpu_microbenchmark'] = previous['batch64_cpu_microbenchmark']
    atomic(HERE / 'checks.json', result)
    print({k:v for k,v in result.items() if k not in ['all_nested_budgets_covered_and_train_only', 'transfer_receipt']})


if __name__ == '__main__':
    main()
