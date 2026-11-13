"""True paired model-seed repeats, not bootstrap or readout-seed substitutes."""
import shutil
from core import ROOT, OUT, atomic, freeze, read, sha


def main():
    source = ROOT / 'worktrees/personal_transfer_v1'
    folder = ROOT / 'worktrees/repair_personal_repeats_v1'
    if folder.exists():
        assert (OUT / 'personal_repeats_queue.json').exists()
        return
    folder.mkdir()
    for n in ['common.py', 'data.py', 'model.py', 'train.py']:
        shutil.copyfile(source / n, folder / n)
    p = folder / 'common.py'; text = p.read_text(encoding='utf-8')
    text = text.replace("OUTPUT = ROOT / 'artifacts/experiment4_personal_transfer_v1'",
                        "OUTPUT = ROOT / 'artifacts/research_repair_personal_repeats_v1'")
    p.write_text(text, encoding='utf-8')
    templates = read(source / 'suite.json')['jobs'][:2]
    jobs = []
    for seed in [43, 44]:
        for original in templates:
            j = dict(original); j['seed'] = seed; j['id'] = j['id'].replace('seed42', 'seed' + str(seed)); jobs.append(j)
    atomic(folder / 'suite.json', {'jobs': jobs, 'budget_policy': 'Same immutable 100 TRAIN labels per topology, plus 2000 validation labels',
           'paired_seed_policy': 'Independent full-network training seeds43/44 complement existing seed42; same data IDs within each pair',
           'test_policy': 'Freeze best validation checkpoint before evaluating each fixed test; fixed test previously used in earlier versions'})
    freeze(OUT / 'personal_repeats_queue.json', {'worktree': folder.relative_to(ROOT).as_posix(), 'jobs': jobs,
           'output': 'artifacts/research_repair_personal_repeats_v1',
           'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(folder.glob('*')) if p.is_file()},
           'scope': 'Original end-to-end transfer recipe; not independent repeats of the later topology-local hybrid readout',
           'estimated_max_CPU_hours': 4, 'estimated_output_GiB': 1.5})
    print('Prepared four independently initialized full-network paired repeats')


if __name__ == '__main__':
    main()
