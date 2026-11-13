"""Budget-preserving ridge calibration; validation only, no transfer hype."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[key] = '1'
import sys
import numpy as np
from core import ROOT, HERE, OUT, atomic, freeze, read, sha, gate, summarize, now
V2 = ROOT / 'worktrees/personal_refinement_v2'
sys.path.insert(0, str(V2))
from common import CACHE, NEW_CACHE, PARENT_OUTPUT
from features import Inputs, ridge, denormalize, metric_rows


def main():
    gate(1.1)
    dest = OUT / 'fewshot_calibration'
    suite = read(V2 / 'suite.json'); jobs = [j for j in suite['jobs'] if j['budget'] != 'full']
    paths = list(V2.glob('*.py')) + [HERE / 'core.py', HERE / 'fewshot_select.py', CACHE / 'manifest.json', V2 / 'suite.json']
    for parent in sorted({j['parent'] for j in jobs if j['parent']}):
        paths += [NEW_CACHE / parent / 'manifest.json', PARENT_OUTPUT / parent / 'recipe.json', PARENT_OUTPUT / parent / 'selection_frozen.json']
    recipe = {'protocol': 'train_only_ridge_calibration_fixed_fewshot_budgets_v1', 'budgets': [100, 50, 20, 10, 5],
              'alphas': [.1, 1., 10., 100., 1000.], 'modes': ['physical', 'pretrained_joint', 'scratch_joint'],
              'sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
              'selection': 'Within each budget AND mode, highest validation macro hit5, tie higher medianR2; target budget/parent unchanged',
              'target_label_budget': 'k train labels per topology PLUS the existing 200 validation labels per topology',
              'seed': 42, 'test_labels_read': False,
              'repeat_caveat': 'Single frozen backbone seed; hyperparameter trials are NOT independent seeds or confidence intervals',
              'transfer_caveat': 'Pretrained and scratch evaluated separately; fall back to physical predictor if graph transfer does not help validation'}
    freeze(dest / 'recipe.json', recipe); result = {}
    for budget in recipe['budgets']:
        inputs = Inputs(budget)
        for mode in recipe['modes']:
            gate(.7); parent = None if mode == 'physical' else ('pretrained' if mode.startswith('pretrained') else 'scratch') + '_k' + str(budget) + '_seed42'
            if parent:
                inputs.parent_check(parent)
            blocks = {name: inputs.block(name, parent, 'physical' if mode == 'physical' else 'joint') for name in inputs.structures}
            candidates = []; best = None; best_states = None; best_rows = None
            for alpha in recipe['alphas']:
                rows = []; states = {}
                for name, block in blocks.items():
                    w, b = ridge(block, alpha)
                    pred = block['validation_x'].astype('float64') @ w.T + b
                    rows += metric_rows(name, denormalize(pred, block), block)
                    states[name + '___weight'] = w; states[name + '___bias'] = b
                summary = summarize(rows); candidate = {'alpha': alpha, **summary}; candidates.append(candidate)
                if best is None or (summary['macro_hit5'], summary['median_r2']) > (best['macro_hit5'], best['median_r2']):
                    best, best_states, best_rows = candidate, states, rows
            job = 'k' + str(budget) + '_' + mode; folder = dest / job; folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / 'best.npz', **best_states)
            normalization = {name: block['normalization'] for name, block in blocks.items()}
            atomic(folder / 'normalization.json', normalization)
            selection = {'job': job, 'parent': parent, 'budget_per_topology': budget, 'training_labels': 10 * budget,
                         'validation_labels': 2000, 'selected': best, 'candidates': candidates, 'per_target': best_rows,
                         'best_checkpoint_sha256': sha(folder / 'best.npz'), 'normalization_sha256': sha(folder / 'normalization.json'),
                         'selected_without_test': True, 'recipe_sha256': sha(dest / 'recipe.json')}
            atomic(folder / 'selection_frozen.json', selection); result[job] = best
            atomic(dest / 'status.json', {'state': 'calibrating', 'jobs_complete': len(result), 'total': 15, 'updated_at': now()})
            print(job, best, flush=True)
    atomic(dest / 'summary.json', {'state': 'validation_complete', 'jobs': result, 'test_labels_read': False,
                                 'independent_seeds': 1, 'target_validation_labels': 2000,
                                 'no_claim_of_positive_transfer_without_paired_evidence': True})
    atomic(dest / 'status.json', {'state': 'validation_complete', 'jobs_complete': 15, 'updated_at': now()})


if __name__ == '__main__':
    main()
