"""New recipe + explicit weights-only initialization, never hash-bypass resume."""
import shutil
from common import ROOT, OUT, read, sha, atomic, freeze


def main():
    source = ROOT / 'worktrees/repair_paired_id_v1'
    folder = ROOT / 'worktrees/rgcn_memory_safe_id_v2'
    if folder.exists():
        assert (OUT / 'memory_safe_build.json').exists(); return
    folder.mkdir()
    for name in ['runtime.py', 'data.py', 'fit.py', 'model.py', 'train.py', 'qualify.py', 'supervisor.py']:
        shutil.copyfile(source / name, folder / name)
    cfg = read(source / 'config.json')
    cfg.update(output='artifacts/rgcn_memory_safe_id_v2', training_microbatch=16, validation_microbatch=16,
               lru_topologies=1, cpu_threads=2, checkpoint_every_updates=10, max_epochs=12,
               backbone_lr=1e-4, adaptation_lr=2e-4, warmup_updates=50,
               initial_checkpoint='artifacts/repair_paired_id_v1/best.pt',
               initial_checkpoint_sha256=sha(ROOT / 'artifacts/repair_paired_id_v1/best.pt'),
               parent_recipe_sha256=sha(ROOT / 'artifacts/repair_paired_id_v1/recipe.json'),
               parent_normalization_sha256=sha(ROOT / 'artifacts/repair_paired_id_v1/normalization.json'),
               experiment_kind='ID_memory_safe_finetuning', minimum_CPU_start_RAM_GiB=1.4)
    atomic(folder / 'config.json', cfg)
    target = ROOT / cfg['output']; target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / 'artifacts/repair_paired_id_v1/normalization.json', target / 'normalization.json')
    p = folder / 'train.py'; text = p.read_text(encoding='utf-8')
    text = text.replace("'initialization':'All neural parameters random seed42; no pretrained state or cached embeddings'",
                        "'initialization':{'method':'Explicit weights-only same-ID warm start, fresh optimizer/RNG/recipe','checkpoint':cfg['initial_checkpoint'],'sha256':cfg['initial_checkpoint_sha256'],'parent_recipe_sha256':cfg['parent_recipe_sha256']}")
    text = text.replace("'initialization':'random; no pretraining'", "'initialization':'same-ID weights-only finetuning; fresh optimizer'")
    text = text.replace("'experiment':cfg['experiment_kind']", "'experiment':cfg['experiment_kind']")
    text = text.replace("if av<384*2**20:os._exit(86)", "if av<512*2**20:resource['memory_pressure']=True")
    text = text.replace("model=build_model(reader,stats,cfg).to(args.backend);optimizer=", "model=build_model(reader,stats,cfg).to(args.backend)\n        if not args.resume:\n            parent_path=ROOT/cfg['initial_checkpoint']\n            assert sha(parent_path)==cfg['initial_checkpoint_sha256']\n            parent_recipe=read(ROOT/'artifacts/repair_paired_id_v1/recipe.json')\n            assert sha(ROOT/'artifacts/repair_paired_id_v1/recipe.json')==cfg['parent_recipe_sha256']\n            assert sha(output/'normalization.json')==cfg['parent_normalization_sha256']\n            assert parent_recipe['fold_references_sha256']==reader.fold['sha256'] and parent_recipe['raw_graph_manifest_sha256']==sha(reader.path/'manifest.json')\n            parent=torch.load(parent_path,map_location='cpu',weights_only=True,mmap=True)\n            assert parent['recipe_digest']==hashlib.sha256(json.dumps(parent_recipe,sort_keys=True).encode()).hexdigest()\n            model.load_state_dict(parent['model'],strict=True)\n            atomic(output/'initialization_receipt.json',{'method':'weights_only_new_recipe','parent_epoch':parent['epoch'],'parent_checkpoint_sha256':cfg['initial_checkpoint_sha256'],'no_parent_optimizer_or_cursor_restored':True,'same_ID_roles':True,'usable_for_LOTO':False})\n            del parent\n        optimizer=")
    text = text.replace("baseline=evaluate('validation',0,True)", "baseline=evaluate('validation',0,False)")
    text = text.replace("or psutil.virtual_memory().available<640*2**20:", "or resource.get('memory_pressure',False) or psutil.virtual_memory().available<640*2**20:")
    text = text.replace("stats_epoch['seconds']=elapsed+time.perf_counter()-tick;checkpoint(epoch,number-1,stats_epoch)", "checkpoint(epoch,number-1,stats_epoch)")
    text = text.replace("chosen=indices[start:start+effective];counts=", "step_tick=time.perf_counter();chosen=indices[start:start+effective];counts=")
    text = text.replace("stats_epoch['seconds']=elapsed+time.perf_counter()-tick", "stats_epoch['seconds']+=time.perf_counter()-step_tick")
    p.write_text(text, encoding='utf-8')
    # CUDA qualification must not mistake a different running graph trainer for
    # free resources. This qualifier is only used by the gated coordinator.
    p = folder / 'qualify.py'; text = p.read_text(encoding='utf-8').replace("endswith('/train.py')", "endswith('/train.py')")
    p.write_text(text, encoding='utf-8')
    freeze(OUT / 'memory_safe_build.json', {'worktree': folder.relative_to(ROOT).as_posix(), 'config': cfg,
           'sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(folder.glob('*')) if p.is_file()},
           'effective_batch_unchanged': 256, 'gradient_accumulation_loss': 'Effective group observed counts passed unchanged to all microbatches',
           'dropout_caveat': 'Changing microbatch partition changes RNG draws and floating order; no bitwise equivalence claim',
           'research_caveat': 'Same-ID finetuning candidate, not a new from-scratch ID/LOTO controlled baseline',
           'reproducibility': 'New recipe and output; explicit parent hash. --resume only allowed within the new recipe.',
           'predicted_optimization': 'Lower peak activation memory and fewer mapped topology arrays; excludes restart waiting from step timing',
           'remaining_queue': ['repair_loto_three_stage_v1', 'repair_id_no_film_v1', 'repair_id_static_edges_v1']})
    print('Prepared', folder, 'microbatch16, accumulation256, independent warm-start recipe')


if __name__ == '__main__': main()
