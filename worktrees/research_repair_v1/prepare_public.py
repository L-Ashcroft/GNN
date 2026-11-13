"""Create independent, genuinely retrained ID/LOTO/ablation jobs."""
import shutil
from core import ROOT, HERE, OUT, atomic, freeze, read, sha


def main():
    source = ROOT / 'worktrees/experiment2_paired_id_v1'
    config = read(source / 'config.json')
    names = ['runtime.py', 'data.py', 'fit.py', 'model.py', 'train.py', 'qualify.py', 'supervisor.py']
    jobs = [('repair_paired_id_v1', 'ID', {}),
            ('repair_loto_three_stage_v1', 'LOTO', {'fold_id': 'LB_M02__hold_SimpleThreeStageOpAmp_SimpleOpAmp1_1_10'}),
            ('repair_id_no_film_v1', 'ID_ablation', {'use_film': False}),
            ('repair_id_static_edges_v1', 'ID_ablation', {'update_edges': False})]
    records = []
    for name, kind, changes in jobs:
        folder = ROOT / 'worktrees' / name
        if folder.exists():
            assert (folder / 'build.json').exists(), 'Refuse to overwrite existing worktree'
            records.append(read(folder / 'build.json')); continue
        folder.mkdir()
        for n in names:
            shutil.copyfile(source / n, folder / n)
        cfg = dict(config); cfg.update(changes)
        cfg['output'] = 'artifacts/' + name
        cfg['experiment_kind'] = kind
        cfg['selection_metric'] = 'validation_macro_hit5' if kind.startswith('ID') else 'compatible_head_validation_macro_hit5'
        cfg['test_policy'] = 'Fixed test labels are evaluated only after validation selection is frozen'
        if kind == 'LOTO':
            cfg['protocol'] = 'data/fusion_v2/splits/verified_loto_full_v2/manifest.json'
        atomic(folder / 'config.json', cfg)
        p = folder / 'data.py'; text = p.read_text(encoding='utf-8')
        text = text.replace("self.protocol_path=ROOT/'data/fusion_v2/splits/verified_loto_full_v2/paired_ID_reference.json'",
                            "self.protocol_path=ROOT/read(Path(__file__).with_name('config.json'))['protocol']")
        p.write_text(text, encoding='utf-8')
        p = folder / 'train.py'; text = p.read_text(encoding='utf-8')
        text = text.replace("'experiment':'public500k_paired_ID_v1'", "'experiment':cfg['experiment_kind']+'_controlled_repair_v1'")
        text = text.replace("'selection':'Compatible-head validation hit5 on other topologies only; outer test after selection only'",
                            "'selection':cfg['selection_metric']+'; fixed test after selection freeze only'")
        text = text.replace("'experiment':'paired_ID_reference'", "'experiment':cfg['experiment_kind']")
        # The prepared legacy ID control selected on M04 alone. ID and all ID
        # ablations must instead use their entire fixed validation population.
        text = text.replace("compatible=[r for r in m['per_target'] if r['supervision_group']==reader.fold['head_alias']]",
                            "compatible=m['per_target'] if cfg['experiment_kind'].startswith('ID') else [r for r in m['per_target'] if r['supervision_group']==reader.fold['head_alias']]")
        text = text.replace("'All neural parameters random seed42; no ID-trained state or embeddings'",
                            "'All neural parameters random seed42; no pretrained state or cached embeddings'")
        text = text.replace("'initialization':'random; noIDpretraining'", "'initialization':'random; no pretraining'")
        p.write_text(text, encoding='utf-8')
        p = folder / 'qualify.py'; text = p.read_text(encoding='utf-8')
        text = text.replace("endswith('experiment2_paired_id_v1/train.py')", "endswith('/train.py')")
        text = text.replace("peer=reader.fold['compatible_training_topologies'][0]",
                            "peer=reader.fold.get('compatible_training_topologies',[reader.topologies[int(reader.refs['train'][0,0])]])[0]")
        p.write_text(text, encoding='utf-8')
        record = {'id': name, 'kind': kind, 'worktree': folder.relative_to(ROOT).as_posix(), 'output': cfg['output'],
                  'config': cfg, 'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(folder.glob('*')) if p.is_file()},
                  'init': 'From scratch; no previously ID-trained weights, embeddings, or normalization',
                  'test_selection': False, 'heldout_topology_representative_choice': 'Predeclared three-stage family vs completed two-stage M04; no claim of exhaustive 24-fold LOTO',
                  'estimated_max_CPU_hours': 45, 'estimated_output_GiB': .5}
        atomic(folder / 'build.json', record); records.append(record)
    freeze(OUT / 'public_queue.json', {'jobs': records, 'serial_only': True,
           'stop_file': 'monitoring/research_repair_v1/STOP', 'CPU_threads': 4,
           'policy': 'Same public population, 20 epochs maximum, 5-epoch patience, seed42; ablations retrain on identical ID roles and selection criterion',
           'estimated_max_CPU_hours': 180, 'estimated_output_GiB': 2,
           'CUDA_policy': 'Only >=4GiB available host memory plus current-implementation qualifier; otherwise CPU'})
    print('Prepared', [r['id'] for r in records])


if __name__ == '__main__':
    main()
