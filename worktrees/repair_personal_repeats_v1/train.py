"""Resumable CPU few-shot/full target training, frozen protocols and test gates."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import gc
import json
import math
import msvcrt
import time
from collections import Counter
import numpy as np
import psutil
import torch
from common import ROOT, HERE, CACHE, OUTPUT, PRETRAINED, read, atomic, sha, digest, pins, now, replace
from data import Reader
from model import TargetRGCN, transfer, loss
from src.model.execution import prepare_batch
from src.train.stage import build_optimizer
from src.train.metrics import MetricAccumulator


class SafePause(Exception):
    pass


def stop_requested(folder):
    return (folder / 'STOP').exists() or psutil.virtual_memory().available < 512 * 2**20


def atomic_checkpoint(path, value):
    temp = path.with_suffix('.pt.tmp')
    torch.save(value, temp)
    replace(temp, path)


def metrics_result(metric):
    result = metric.compute()
    for row in result['per_target']:
        acc = metric.values[(row['supervision_group'], row['semantic_id'])]
        row['rmse'] = math.sqrt(acc['sse'] / acc['n'])
    values = [r['r2'] for r in result['per_target'] if r['r2'] is not None]
    result['median_r2'] = float(np.median(values)) if values else None
    result['negative_r2_targets'] = sum(v < 0 for v in values)
    return result


def evaluate(model, reader, role, folder, update, micro=32, baseline=False, write_predictions=False):
    if role == 'test':
        assert (folder / 'selection_frozen.json').exists(), 'No test access before frozen selection'
    model.eval()
    metric = MetricAccumulator(reader.tasks)
    aliases = {g + '::' + name: reader.tasks[g] for name, s in reader.structures.items() for g in [s['group']]}
    stratified = MetricAccumulator(aliases)
    prediction_rows = []
    refs = reader.refs[role]
    with torch.no_grad():
        for start in range(0, len(refs), micro):
            if stop_requested(folder):
                raise SafePause('Memory or explicit STOP during evaluation')
            batch = reader.batch(refs[start:start + micro], role)
            pred = model.head.affine(batch.group_ids)[0] if baseline else model(batch)
            metric.update(pred, batch)
            grouped = copy.copy(batch)
            grouped.group_ids = [g + '::' + n for g, n in zip(batch.group_ids, batch.topology_ids)]
            stratified.update(pred, grouped)
            if write_predictions:
                physical = (pred.sign() * torch.expm1(pred.abs()) * batch.scales).tolist()
                for i, sid in enumerate(batch.sample_ids):
                    prediction_rows.append({'sample_id': sid, 'topology_id': batch.topology_ids[i],
                                            'supervision_group': batch.group_ids[i], 'prediction': physical[i],
                                            'target': batch.targets[i].tolist(), 'mask': batch.mask[i].tolist()})
            if start % 256 == 0 or start + micro >= len(refs):
                update(state='baseline_validation' if baseline else 'evaluating_' + role,
                       role=role, samples_done=min(start + micro, len(refs)), role_samples=len(refs))
            del batch, pred, grouped
    result = metrics_result(stratified)
    result['contract_pooled_metrics'] = metrics_result(metric)
    result['metric_aggregation'] = 'Macro over topology x supervision contract x physical metric; logarithmic units use absolute errors'
    result['samples'] = len(refs)
    if write_predictions:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(prediction_rows), folder / 'test_predictions.parquet', compression='zstd')
    return result


def run(job, resume=False):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(job['seed'])
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    folder = OUTPUT / job['id']
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'training.lock').open('a+b')
    if lock.tell() == 0:
        lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    state = {'state': 'initializing', 'pid': os.getpid(), 'create_time': psutil.Process().create_time(),
             'job_id': job['id'], 'device': 'cpu', 'test_used': False}
    def update(**values):
        state.update(values, updated_at=now(), available_RAM_GiB=psutil.virtual_memory().available / 2**30,
                     RSS_MiB=psutil.Process().memory_info().rss / 2**20)
        atomic(folder / 'status.json', state)
    update()
    try:
        if psutil.virtual_memory().available < 1.1 * 2**30:
            raise SafePause('CPU start memory gate')
        reader = Reader(verify=True)
        refs = reader.training_refs(job['budget_per_topology'])
        stats = reader.statistics(refs)
        recipe = {'job': job, 'implementation_sha256': pins(), 'cache_sha256': sha(CACHE / 'manifest.json'),
                  'source_checkpoint_sha256': sha(PRETRAINED) if job['initialization'] == 'public_pretrained' else None,
                  'statistics': stats, 'target_training_labels': len(refs), 'target_validation_labels': 2000,
                  'total_target_supervision_budget': len(refs) + 2000,
                  'test_samples': 2000, 'private_test_used_for_training_or_selection': False,
                  'same_structure_and_optimizer_across_paired_initializations': True,
                  'loss': 'train-only symlog center/std floor0.1; contract-equal masked MSE',
                  'precision': 'FP32 encoder; FP64 target transform and observed loss',
                  'cross_source_target_readouts_reused': False}
        if (folder / 'recipe.json').exists():
            assert read(folder / 'recipe.json') == recipe and resume, 'Recipe changed; never cross-version resume'
        else:
            assert not resume
            atomic(folder / 'recipe.json', recipe)
        recipe_hash = digest(recipe)
        atomic(folder / 'normalization.json', stats)
        model = TargetRGCN(reader.schema, reader.tasks, stats)
        receipt = {'source': 'random_initialization', 'target_readouts_transferred': False}
        if job['initialization'] == 'public_pretrained':
            source = torch.load(PRETRAINED, map_location='cpu', weights_only=True)
            receipt = transfer(model, source)
            del source
            gc.collect()
        atomic(folder / 'initialization_receipt.json', receipt)
        optimizer = build_optimizer(model, 'finetune_all', lr_head=job['adaptation_lr'],
                                    lr_backbone=job['backbone_lr'], weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5,
                       patience=2, threshold=1e-4, threshold_mode='abs', min_lr=2e-6)
        best, best_epoch, bad, updates, start_epoch, cursor, history = -1., 0, 0, 0, 0, 0, []
        batches = math.ceil(len(refs) / job['effective_batch'])
        warmup = min(100, 2 * batches)
        def save(epoch, next_batch):
            atomic_checkpoint(folder / 'last.pt', {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'recipe_hash': recipe_hash, 'epoch': epoch, 'cursor': next_batch,
                'best': best, 'best_epoch': best_epoch, 'bad': bad, 'updates': updates,
                'rng_cpu': torch.get_rng_state(), 'history': history})
        resume_checkpoint = resume and (folder / 'last.pt').exists()
        if resume_checkpoint:
            checkpoint = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
            assert checkpoint['recipe_hash'] == recipe_hash
            model.load_state_dict(checkpoint['model'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer']); scheduler.load_state_dict(checkpoint['scheduler'])
            torch.set_rng_state(checkpoint['rng_cpu'])
            best, best_epoch, bad, updates = [checkpoint[k] for k in ['best', 'best_epoch', 'bad', 'updates']]
            start_epoch, cursor, history = checkpoint['epoch'], checkpoint['cursor'], checkpoint['history']
            del checkpoint
        if not resume_checkpoint or not (folder / 'best.pt').exists():
            save(0, 0)
            baseline = evaluate(model, reader, 'validation', folder, update, baseline=True)
            atomic(folder / 'training_mean_validation.json', baseline)
            best = baseline['macro_hit5']
            assert best is not None
            scheduler.best = best
            atomic_checkpoint(folder / 'best.pt', {'model': model.state_dict(), 'recipe_hash': recipe_hash, 'epoch': 0})
            save(0, 0)
        tick, measured_samples = time.perf_counter(), 0
        epoch = start_epoch
        next_batch = cursor
        try:
            end_epoch = start_epoch if start_epoch >= 4 and bad >= job['patience'] else job['max_epochs']
            for epoch in range(start_epoch, end_epoch):
                order = np.random.default_rng(job['seed'] + epoch).permutation(refs)
                model.train()
                update(state='training', epoch=epoch + 1, max_epochs=job['max_epochs'], train_samples=len(refs),
                       validation_label_budget=2000, total_target_supervision_budget=len(refs) + 2000,
                       initialization=job['initialization'], best_validation_hit5=best, best_epoch=best_epoch,
                       batch=cursor, batches_per_epoch=batches)
                for number in range(cursor, batches):
                    if stop_requested(folder):
                        raise SafePause('Memory or explicit stop')
                    selected = order[number * job['effective_batch']:(number + 1) * job['effective_batch']]
                    counts = Counter()
                    for ref in selected:
                        item = reader.index[int(ref)]; name, local = item['topology_id'], item['local']
                        counts[reader.structures[name]['group']] += int(reader.arrays[name]['mask'][local].sum())
                    optimizer.zero_grad(set_to_none=True)
                    if updates < warmup:
                        factor = max(.1, (updates + 1) / warmup)
                        for group in optimizer.param_groups:
                            group['lr'] = job['backbone_lr' if group['name'] == 'backbone' else 'adaptation_lr'] * factor
                    value = 0.
                    for off in range(0, len(selected), job['microbatch']):
                        batch = reader.batch(selected[off:off + job['microbatch']], 'train')
                        _, parts = loss(model(batch), batch, model.head)
                        sizes = {g: b - a for g, a, b in prepare_batch(batch).loss_slices}
                        weighted = torch.stack([v * (sizes[g] / counts[g]) / len(counts) for g, v in parts.items()]).sum()
                        weighted.backward(); value += float(weighted.detach())
                        del batch, parts, weighted
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                    optimizer.step(); updates += 1; next_batch = number + 1; measured_samples += len(selected)
                    if updates % 20 == 0 or next_batch == batches:
                        save(epoch, next_batch)
                    if updates % 5 == 0 or next_batch == batches:
                        update(state='training', role='train', epoch=epoch + 1, batch=next_batch, updates=updates,
                               training_loss=value, gradient_norm=float(grad), samples_per_second=measured_samples / (time.perf_counter() - tick),
                               learning_rates={g['name']: g['lr'] for g in optimizer.param_groups})
                save(epoch, batches)
                validation = evaluate(model, reader, 'validation', folder, update)
                atomic(folder / 'latest_validation.json', validation)
                score = validation['macro_hit5']
                assert score is not None and math.isfinite(score)
                history.append({'epoch': epoch + 1, 'validation_hit5': score,
                                'validation_median_r2': validation['median_r2'], 'updates': updates})
                if score > best + 1e-4:
                    best, best_epoch, bad = score, epoch + 1, 0
                    atomic_checkpoint(folder / 'best.pt', {'model': model.state_dict(), 'recipe_hash': recipe_hash, 'epoch': best_epoch})
                else:
                    bad += 1
                scheduler.step(score); cursor = 0; next_batch = 0
                save(epoch + 1, 0); atomic(folder / 'history.json', history)
                update(state='epoch_complete', completed_epochs=epoch + 1, best_validation_hit5=best,
                       best_epoch=best_epoch, bad_epochs=bad, validation_hit5=score,
                       validation_median_r2=validation['median_r2'])
                print(now(), job['id'], 'epoch', epoch + 1, 'validation', score, flush=True)
                if epoch + 1 >= 4 and bad >= job['patience']:
                    break
        except SafePause as exc:
            save(epoch, next_batch)
            update(state='paused', reason=str(exc))
            return 75
        atomic(folder / 'selection_frozen.json', {'at': now(), 'best_epoch': best_epoch, 'best_validation_hit5': best,
               'checkpoint_sha256': sha(folder / 'best.pt'), 'recipe_hash': recipe_hash,
               'test_used_for_selection': False})
        checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        assert checkpoint['recipe_hash'] == recipe_hash
        model.load_state_dict(checkpoint['model'], strict=True)
        result = evaluate(model, reader, 'test', folder, update, write_predictions=True)
        atomic(folder / 'test_metrics.json', result)
        update(state='complete', test_used=True, best_epoch=best_epoch, best_validation_hit5=best,
               test_hit5=result['macro_hit5'], test_median_r2=result['median_r2'], test_samples=2000,
               completion_reason='early_stop' if bad >= job['patience'] else 'epoch_budget')
        return 0
    except SafePause as exc:
        update(state='paused', reason=str(exc))
        return 75
    except BaseException as exc:
        update(state='error', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--job', required=True); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    suite = read(HERE / 'suite.json'); selected = next(x for x in suite['jobs'] if x['id'] == args.job)
    raise SystemExit(run(selected, args.resume))
