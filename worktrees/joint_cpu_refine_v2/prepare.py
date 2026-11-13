"""Prepare a new conservative full-data recipe; retain failed short pilots."""
from pathlib import Path
import json
import hashlib
import shutil
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
old=ROOT/'worktrees/joint_cpu_v1'
assert not (HERE/'train.py').exists(),'Never overwrite a prepared/running version'
text=(old/'train.py').read_text(encoding='utf-8')
start=text.index('        # Fixed train/validation subsets only.')
end=text.index('        else:\n            saved=torch.load(output/\'last.pt\'', start)
replacement='''        # Short pilots were inconclusive after20 updates. This explicitly new
        # recipe gives low-LR/warmup full training one complete validation cycle.
        # The already measured parent remains best until full validation improves.
        if not args.resume:
            optimizer=build_opt();scale=1.;history=[];start_epoch=0;start_update=0;partial=None;bad=0
            best=parent['validation']['macro_hit5'];best_epoch=-1
            save(output/'best.pt',dict(parent,config=dict(parent['config'],output=cfg['output'],joint_stage=True),recipe_digest=digest(recipe)))
'''
text=text[:start]+replacement+text[end:]
before="                chosen=indices[start:start+effective];full_counts=reader.counts('train',chosen);optimizer.zero_grad(set_to_none=True);total_loss=0."
after="""                if epoch==0 and update_index<=cfg['warmup_updates']:
                    warmup=max(.1,update_index/cfg['warmup_updates'])
                    for group in optimizer.param_groups:
                        group['lr']=cfg['backbone_lr' if group['name']=='backbone' else 'head_lr']*warmup
                chosen=indices[start:start+effective];full_counts=reader.counts('train',chosen);optimizer.zero_grad(set_to_none=True);total_loss=0."""
assert before in text;text=text.replace(before,after)
text=text.replace("'streamed_joint_CPU_v1'","'streamed_joint_CPU_lower_lr_warmup_v2'")
text=text.replace("'joint_cpu_v1/train.py'","'joint_cpu_v1/train.py','joint_cpu_refine_v2/train.py'")
text=text.replace("            checkpoint(epoch+1,0,None);atomic(output/'history.json',history)","            if score<parent['validation']['macro_hit5']-.03 and validation['median_r2']<parent['validation']['median_r2']-.05:\n                practical_stop=True;record['stop_action']='severe_full_validation_regression'\n            checkpoint(epoch+1,0,None);atomic(output/'history.json',history)")
(HERE/'train.py').write_text(text,encoding='utf-8')
shutil.copy2(old/'stream_graphs.py',HERE/'stream_graphs.py')
cfg=json.loads((old/'config.json').read_text(encoding='utf-8'))
cfg.update(output='artifacts/experiment1_joint_cpu_refine_v2',head_lr=1e-5,backbone_lr=2e-6,warmup_updates=100,
           max_epochs=12,patience=3,window_epochs=3,minimum_window_gain=.001,window_stop_backbone_lr=1e-6)
for key in list(cfg):
    if key.startswith('pilot_'):del cfg[key]
cfg['selection_policy']='Keep original full-validation-best parent; evaluate all400083train/50861validation each epoch. No early20-update superiority filter.'
(HERE/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
supervisor=(old/'supervisor.py').read_text(encoding='utf-8').replace('joint_cpu_v1','joint_cpu_refine_v2')
supervisor=supervisor.replace('reports/joint_cpu_training.md','reports/joint_cpu_training.md')
(HERE/'supervisor.py').write_text(supervisor,encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(HERE/'origin.json').write_text(json.dumps({'source':str(old.relative_to(ROOT)),'source_SHA256':sha(old/'train.py'),
    'prior_pilots':'artifacts/experiment1_joint_cpu_v1/pilot_trials.json','short_pilot_promoted':False,
    'changes':['Lower backboneLR2e-6/headLR1e-5','100update linear warmup','One full training epoch before validation selection','patience3 and severe regression stop','Separate recipe/output; no test selection'],
    'mathematical_objective':'unchanged contract/Mask standardized symlog MSE; no label/split/model capacity change'},indent=2),encoding='utf-8')
print('Prepared conservative full-data joint recipe, source artifacts retained.')
