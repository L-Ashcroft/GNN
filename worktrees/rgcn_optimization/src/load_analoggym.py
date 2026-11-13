"""Audit interface for AnalogGym. Never invent rows from testbench templates.

Archived replay labels can be clipped/fallback values. They are excluded from the
supervised dataset until raw results and actions are independently aligned.
"""
from .common import ROOT, rel


def discover():
    return iter(())


def audit():
    repo = ROOT / 'third_party/AnalogGym'
    result = {'source_dataset': 'AnalogGym', 'status': 'no_verified_row_level_supervised_table', 'accepted_samples': 0,
            'netlist_files': [rel(p) for p in sorted((repo / 'AnalogGym').rglob('*')) if p.is_file() and
                              ('spice_netlist' in p.parts or p.name == 'topology.txt')],
            'reason': 'Template netlists, example logs and RL replay are present; no verified row-to-parameter-to-unclipped-label alignment. Reward and normalized observations are not performance labels.',
            'replay_evidence': ['third_party/AnalogGym/RGNN_RL/ddpg.py', 'third_party/AnalogGym/RGNN_RL/AMP_NMCF.py',
                                'third_party/AnalogGym/RGNN_RL/LDO_TB.py'],
            'archives_policy': 'ZIP central directories inspected; no arbitrary pickle execution; no sample count inferred from allocated buffers.'}
    detailed = ROOT / 'reports/analoggym_replay_audit.json'
    if detailed.exists():
        from .common import read
        replay = read(detailed)
        result.update(status='replay_recovered_but_quarantined',
                      written_replay_transitions=replay['written_transitions'],
                      accepted_samples=replay['supervised_accepted'],
                      detailed_replay_audit=rel(detailed),
                      reason='20 stored transitions; constant AMP labels, stale LDO DC evidence, fallback ambiguity and archived action ranges differ from current source.',
                      archives_policy=replay['pickle_policy'])
    return result


def iter_replay_candidates(*, allow_quarantined=False):
    """Explicit opt-in to forensic records; never silently serve training samples."""
    if not allow_quarantined:
        raise ValueError('Replay rows are quarantined; use allow_quarantined=True for audit only')
    import json
    import pandas as pd
    path = ROOT / 'data/audit/analoggym_replay/quarantined_transitions.parquet'
    for row in pd.read_parquet(path).to_dict('records'):
        row['performance_raw'] = json.loads(row.pop('performance_raw_json'))
        row['candidate_parameters'] = json.loads(row.pop('candidate_parameters_json'))
        yield row


if __name__ == '__main__':
    from scripts.convert_source import main
    main('AnalogGym')
