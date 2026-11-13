"""Publish the validation decision and replay its interface on all fixed tests."""
from pathlib import Path
import sys
import numpy as np
import pyarrow.parquet as pq
from interface import ROOT, read, sha, BoundedPhysicalReadout
sys.path.insert(0, str(ROOT / 'worktrees/research_repair_v1'))
from core import atomic, freeze


def main():
    folder = ROOT / 'artifacts/research_repair_v1/loadbench_shared_v2'
    source_tasks = ROOT / 'data/model_ready/fusion_v2_training_v1/tasks.json'
    details_path = ROOT / 'data/fusion_v2/supervision_group_details.json'
    tasks = read(source_tasks); details = read(details_path)
    paths = [source_tasks, details_path, folder / 'recipe.json', Path(__file__).with_name('interface.py')]
    groups = {}
    for merge in details['verified_merge_sets']:
        selected_path = folder / merge['merge_id'] / 'selection.json'; selected = read(selected_path); paths.append(selected_path)
        kind = selected['selected']['kind']
        assert kind in ['identity', 'topology_train_bounds']
        if kind == 'identity':
            continue
        bounds = selected_path.with_name('bounds.npz'); paths.append(bounds)
        for g in merge['supervision_group_ids']:
            groups[g] = {'bounds': bounds.relative_to(ROOT).as_posix(), 'merge_id': merge['merge_id']}
    publication = {'format': 'validation_selected_physical_bounds_v1', 'scope': 'Known-topology ID only',
                   'input': 'Raw-unit output from the existing calibrated LOADBench readout; 16 columns',
                   'source_bundle': 'artifacts/loadbench_remediation_v1/bundle.json',
                   'sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}, 'groups': groups,
                   'selection_uses': 'TRAIN and VALIDATION only, according to predeclared guard',
                   'original_checkpoints_masks_labels_roles_changed': False,
                   'test_used_to_select_or_promote': False, 'no_LOTO_or_zero_shot_claim': True}
    freeze(folder / 'deployment.json', publication)
    adapter = BoundedPhysicalReadout(); maximum = 0.; samples = 0
    by_top = {t['contract']['topology_id']: g for g, t in tasks.items() if t['contract']['source_dataset'] == 'LOADBench'}
    for p in sorted((folder / 'fixed_test').glob('*.parquet')):
        before = pq.read_table(ROOT / 'artifacts/loadbench_remediation_v1/fixed_test' / p.name).to_pylist()
        after = pq.read_table(p).to_pylist()
        assert [r['sample_id'] for r in before] == [r['sample_id'] for r in after]
        np.testing.assert_array_equal([r['mask'] for r in before], [r['mask'] for r in after])
        np.testing.assert_array_equal([r['target'] for r in before], [r['target'] for r in after])
        actual = adapter(by_top[p.stem], np.array([r['prediction'] for r in before]))
        expected = np.array([r['prediction'] for r in after])
        maximum = max(maximum, float(abs(actual - expected).max()))
        np.testing.assert_array_equal(actual, expected); samples += len(actual)
    for g, task in tasks.items():
        if task['contract']['source_dataset'] != 'LOADBench':
            probe = np.random.default_rng(42).normal(size=(3, 16))
            np.testing.assert_array_equal(adapter(g, probe), probe)
    atomic(folder / 'interface_checks.json', {'passed': True, 'real_test_samples_replayed': samples,
           'maximum_absolute_replay_error': maximum, 'mask_and_targets_exactly_preserved': True,
           'other_sources_exact_pass_through': True, 'test_based_candidate_selection': False,
           'checkpoint_compatibility': 'Postprocessor only; existing model weights and recipe unchanged'})
    print('Published independent adapter; replayed', samples, 'samples; max error', maximum)


if __name__ == '__main__':
    main()
