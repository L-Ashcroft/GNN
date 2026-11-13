"""Admission fixes: distinguish an unavailable zero-shot task from zero score."""
from core import ROOT, OUT, read, atomic, sha, now


def main():
    public_path = ROOT / 'data/model_ready/fusion_v2_training_v1/tasks.json'
    private_path = ROOT / 'data/model_ready/personal_transfer_v1/tasks.json'
    public, private = read(public_path), read(private_path)
    def semantics(tasks):
        return {r['semantic_id'] for t in tasks.values() for r in t['slots'].values()}
    common = semantics(public) & semantics(private)
    source_schema = read(public_path.with_name('schema.json'))
    target_schema = read(private_path.with_name('schema.json'))
    original = ROOT / 'artifacts/experiment3_cross_source_v1/compatibility_audit.json'
    evidence = read(original)
    assert not common and evidence['verified_shared_targets'] == 0
    result = {'state': 'zero_shot_blocked_by_physical_contracts', 'verified_shared_targets': len(common),
              'source_schema': {'node_features': len(source_schema['node_features']), 'relations': len(source_schema['relations'])},
              'target_schema': {'node_features': len(target_schema['node_features']), 'relations': len(target_schema['relations'])},
              'new_target_relations': sorted(set(target_schema['relations']) - set(source_schema['relations'])),
              'sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in [public_path, private_path, original]},
              'required_bridge_before_training': [
                  'Choose an available common PDK/model/corner and the SAME voltage, temperature, bias, load and extractor for BOTH domains',
                  'Resimulate public-source training circuits AND held-out target circuits under that common fixture; preserve raw archives',
                  'Fit shared output heads on source bridge labels only; fit normalization on source TRAIN only',
                  'Freeze a shared graph vocabulary and deterministic behavior for unseen node/port types; source-only domain inference',
                  'Hold out all target labels from weight fitting, calibration and epoch/model selection; predeclare target test IDs'],
              'current_cannot_fix_by': ['Rename gain metrics', 'Attach a randomly initialized target output head',
                                       'Fit target labels and call the result zero-shot', 'Use missing-only mask to hide condition mismatch'],
              'new_large_simulation_started': False,
              'reason': 'Existing private amplifier load is500pF versus public LOADBench1pF, model bundles and extraction differ; current output heads are not shared. No measured zero-shot result is available.',
              'at': now()}
    atomic(OUT / 'zero_shot_admission.json', result)
    print(result['state'], 'shared targets:', len(common))


if __name__ == '__main__':
    main()
