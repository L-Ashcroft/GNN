"""Explicit same-state migration after a transient Windows checkpoint lock."""
from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OLD=ROOT/'worktrees/rgcn_direct_head_v3'
assert not (HERE/'train.py').exists(),'Cannot overwrite running/published source'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=ROOT/'artifacts/experiment1_rgcn_direct_head_v3'
for name in ['model.py','checks.py']:shutil.copy2(OLD/name,HERE/name)
cfg=json.loads((OLD/'config.json').read_text());cfg.update(output='artifacts/experiment1_rgcn_direct_head_v3_io',checkpoint_every_batches=400,
    migration_checkpoint='artifacts/experiment1_rgcn_direct_head_v3/last.pt.tmp',migration_checkpoint_sha256=sha(out/'last.pt.tmp'),
    migration_best='artifacts/experiment1_rgcn_direct_head_v3/best.pt',migration_best_sha256=sha(out/'best.pt'),
    migration_recipe='artifacts/experiment1_rgcn_direct_head_v3/recipe.json',migration_recipe_sha256=sha(out/'recipe.json'))
(HERE/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
text=(OLD/'train.py').read_text(encoding='utf-8')
before="            q=p.with_suffix('.pt.tmp');torch.save(value,q);os.replace(q,p)"
after="""            q=p.with_suffix('.pt.tmp');torch.save(value,q)
            for attempt in range(100):
                try:os.replace(q,p);return
                except PermissionError:time.sleep(.1)
            raise RuntimeError('Atomic checkpoint replace failed after10seconds: '+str(p))"""
assert before in text;text=text.replace(before,after)
before="        else:\n            initial=evaluate(0);atomic(output/'initial_validation.json',initial);best=initial['macro_hit5'];best_epoch=0"
after="""        elif cfg.get('migration_checkpoint'):
            for field in ['checkpoint','best','recipe']:
                assert sha(ROOT/cfg['migration_'+field])==cfg['migration_'+field+'_sha256'],'Migration source changed'
            source_recipe=read(ROOT/cfg['migration_recipe'])
            old_digest=hashlib.sha256(json.dumps(source_recipe,sort_keys=True).encode()).hexdigest()
            saved=torch.load(ROOT/cfg['migration_checkpoint'],map_location='cpu',weights_only=True)
            assert saved['recipe_digest']==old_digest
            head.load_state_dict(saved['head'],strict=True);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);torch.set_rng_state(saved['rng'])
            for name,value in head.state_dict().items():torch.testing.assert_close(value,saved['head'][name],atol=0,rtol=0)
            history=saved['history'];start_epoch=saved['next_epoch'];start_batch=saved['next_batch'];partial=saved['partial'];best=saved['best'];best_epoch=saved['best_epoch'];bad=saved['bad']
            prior_best=torch.load(ROOT/cfg['migration_best'],map_location='cpu',weights_only=True)
            torch_save(output/'best.pt',dict(base_info,head=prior_best['head'],epoch=prior_best['epoch'],validation=prior_best['validation'],migration_from=cfg['migration_best']))
            atomic(output/'history.json',history)
            atomic(output/'migration.json',{'source_recipe_digest':old_digest,'new_recipe_digest':digest,'head_exact':True,'optimizer_restored':True,'scheduler_restored':True,'RNG_restored':True,'next_epoch':start_epoch,'next_batch':start_batch,'old_artifacts_unchanged':True})
        else:
            initial=evaluate(0);atomic(output/'initial_validation.json',initial);best=initial['macro_hit5'];best_epoch=0"""
assert before in text;text=text.replace(before,after)
(HERE/'train.py').write_text(text,encoding='utf-8')
text=(OLD/'supervisor.py').read_text(encoding='utf-8').replace('rgcn_direct_head_v3','rgcn_direct_head_v3_io')
text=text.replace("['memoryerror','out of memory','winerror 32']","['memoryerror','out of memory','winerror 32','winerror 5']")
(HERE/'supervisor.py').write_text(text,encoding='utf-8')
print('Prepared explicit state migration/new recipe; old files preserved.')
