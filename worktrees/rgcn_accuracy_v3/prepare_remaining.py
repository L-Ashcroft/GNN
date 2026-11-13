"""Keep the remaining controlled protocols, reduce activation/mapping memory."""
import shutil
from common import ROOT, OUT, read, sha, atomic, freeze


def main():
    jobs = []
    for old, new in [('repair_loto_three_stage_v1', 'rgcn_memory_safe_loto_v2'),
                     ('repair_paired_id_v1', 'rgcn_memory_safe_id_control_v2'),
                     ('repair_id_no_film_v1', 'rgcn_memory_safe_no_film_v2'),
                     ('repair_id_static_edges_v1', 'rgcn_memory_safe_static_edges_v2')]:
        source = ROOT / 'worktrees' / old; folder = ROOT / 'worktrees' / new
        cfg = read(source / 'config.json')
        cfg.update(output='artifacts/' + new, training_microbatch=16, validation_microbatch=16,
                   lru_topologies=1, cpu_threads=2, checkpoint_every_updates=10, minimum_CPU_start_RAM_GiB=1.4)
        if not folder.exists():
            folder.mkdir()
            for name in ['runtime.py', 'data.py', 'fit.py', 'model.py', 'train.py', 'qualify.py', 'supervisor.py']:
                shutil.copyfile(source / name, folder / name)
            atomic(folder / 'config.json', cfg)
            output = ROOT / cfg['output']; output.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / 'artifacts' / old / 'normalization.json', output / 'normalization.json')
            p = folder / 'train.py'; text = p.read_text(encoding='utf-8')
            text = text.replace('if av<384*2**20:os._exit(86)', "if av<512*2**20:resource['memory_pressure']=True")
            text = text.replace('or psutil.virtual_memory().available<640*2**20:', "or resource.get('memory_pressure',False) or psutil.virtual_memory().available<640*2**20:")
            text = text.replace("stats_epoch['seconds']=elapsed+time.perf_counter()-tick;checkpoint(epoch,number-1,stats_epoch)", 'checkpoint(epoch,number-1,stats_epoch)')
            text = text.replace('chosen=indices[start:start+effective];counts=', 'step_tick=time.perf_counter();chosen=indices[start:start+effective];counts=')
            text = text.replace("stats_epoch['seconds']=elapsed+time.perf_counter()-tick", "stats_epoch['seconds']+=time.perf_counter()-step_tick")
            p.write_text(text, encoding='utf-8')
        assert read(folder / 'config.json') == cfg
        jobs.append({'id': new, 'worktree': folder.relative_to(ROOT).as_posix(), 'output': cfg['output'],
                     'sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in folder.glob('*') if p.is_file()},
                     'init': 'Random seed42, no ID encoder or target topology weights', 'kind': cfg['experiment_kind']})
    freeze(OUT / 'remaining_queue.json', {'jobs': jobs, 'scope': 'Same original protocols, fresh independent outputs; effective256/micro16, no label/loss/split changes',
           'ID_reference': 'Use new cold ID control with two new cold ablations; warm-start candidate is a separate optimization study',
           'estimated_CPU_budget_hours': 180, 'private_samples_used': False})
    print('Prepared four memory-bounded, from-scratch control/fold/ablation tasks')


if __name__ == '__main__': main()
