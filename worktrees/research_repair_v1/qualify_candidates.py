"""Prevent a higher hit rate from silently replacing a more stable model."""
from core import ROOT, OUT, read, atomic, now


def main():
    folder = OUT / 'fewshot_calibration'
    if not (folder / 'summary.json').exists():
        raise RuntimeError('Calibration not complete')
    results = {}
    for name, candidate in read(folder / 'summary.json')['jobs'].items():
        baseline = read(ROOT / 'artifacts/personal_refinement_v2' / name / 'selected_validation.json')
        passed = (candidate['macro_hit5'] > baseline['macro_hit5'] and
                  candidate['median_r2'] >= baseline['median_r2'] and
                  candidate['macro_r2'] >= baseline['macro_r2'] and
                  candidate['negative_r2_targets'] <= baseline['negative_r2_targets'])
        results[name] = {'validation_candidate': candidate,
                         'baseline': {k: baseline[k] for k in ['macro_hit5', 'median_r2', 'macro_r2', 'negative_r2_targets']},
                         'validation_qualified': passed, 'action': 'eligible_for_frozen_test' if passed else 'retain_existing_selected_model',
                         'test_based_selection': False, 'deployed': False}
    atomic(folder / 'promotion_guard.json', {'state': 'complete', 'jobs': results, 'at': now(),
           'rule': 'Strictly improve validation hit5; no lower medianR2/meanR2 and no additional negative-R2 targets',
           'eligible_count': sum(r['validation_qualified'] for r in results.values()), 'test_labels_read': False})
    print('Validation qualified:', [k for k, v in results.items() if v['validation_qualified']])


if __name__ == '__main__':
    main()
