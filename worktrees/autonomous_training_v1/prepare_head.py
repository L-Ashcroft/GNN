"""Derive a state-preserving head phase with validation-window convergence."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
def main():
    source=ROOT/'worktrees/cached_head_continue_v1'
    text=(source/'train_cpu.py').read_text(encoding='utf-8')
    text=text.replace("PACKAGE/'config_cpu.json'","PACKAGE/'config_head.json'")
    text=text.replace('experiment1_cached_head_cpu_continue_v1','experiment1_autonomous_head_v1')
    text=text.replace("'cached_head_validation_continue_cpu_v1'","'autonomous_cached_head_v1'")
    text=text.replace("'optimizer_policy':'Preserve AdamW moments and step counters; restart LR at 5e-5, ReduceLROnPlateau validation-only'",
        "'optimizer_policy':'Exact state continuation; inherited LR and ReduceLROnPlateau plus declared window-plateau rule'")
    text=text.replace("initial=dict(base,config=model_config", "initial=dict(torch.load(backend/'best.pt',map_location='cpu',weights_only=True),config=model_config")
    text=text.replace("stage='validation_only_cached_head_continuation'","stage='autonomous_cached_head_refinement',device='cpu'")
    text=text.replace("for epoch in range(start_epoch,cfg['additional_epochs']):", "adaptive_stop=False\n        for epoch in range(start_epoch,cfg['additional_epochs']):")
    old="scheduler.step(score);save_cursor(epoch+1,0,None);atomic_json(output/'history.json',history)"
    new="""scheduler.step(score);adaptive_stop=False
            window=cfg['window_epochs'];total_epoch=inherited_total+epoch+1
            last_action=max([r['head_epoch_total'] for r in history if r.get('window_lr_action')],default=cfg['phase_start_head_epoch']-window)
            if len(history)>window and total_epoch-last_action>=window:
                baseline=max([initial_best]+[r['validation']['macro_hit5'] for r in history[:-window]])
                gain=best-baseline;record['window_validation_gain']=gain
                if gain<cfg['minimum_window_gain']:
                    if optimizer.param_groups[0]['lr']>cfg['window_stop_lr']:
                        for group in optimizer.param_groups:group['lr']=max(cfg['minimum_learning_rate'],group['lr']*.5)
                        scheduler._last_lr=[g['lr'] for g in optimizer.param_groups]
                        record['window_lr_action']='reduce_on_practical_plateau'
                    else:adaptive_stop=True;record['window_lr_action']='stop_on_practical_plateau'
            save_cursor(epoch+1,0,None);atomic_json(output/'history.json',history)"""
    assert old in text;text=text.replace(old,new)
    text=text.replace("if bad>=cfg['patience']:break","if bad>=cfg['patience'] or adaptive_stop:break")
    text=text.replace("completion_reason='validation_early_stop' if bad>=cfg['patience'] else 'additional_epoch_budget'",
        "completion_reason='validation_window_plateau' if adaptive_stop else 'validation_early_stop' if bad>=cfg['patience'] else 'safety_budget_review'")
    text=text.replace("'device':'cpu'","'device':'cpu','phase_inherited_head_epochs':cfg['phase_start_head_epoch']")
    compile(text,str(HERE/'head_train.py'),'exec');(HERE/'head_train.py').write_text(text,encoding='utf-8')
    watch=(source/'supervise_cpu.py').read_text(encoding='utf-8').replace('monitoring/cached_head_cpu_continue_v1','monitoring/autonomous_head_v1')
    watch=watch.replace("PACKAGE/'config_cpu.json'","PACKAGE/'config_head.json'").replace("PACKAGE/'train_cpu.py'","PACKAGE/'head_train.py'")
    watch=watch.replace('cached_head_continue_v1/train_cpu.py','autonomous_training_v1/head_train.py')
    compile(watch,str(HERE/'head_supervisor.py'),'exec');(HERE/'head_supervisor.py').write_text(watch,encoding='utf-8')
    print('Derived source-preserving head trainer and supervisor. Weights, AdamW, scheduler, RNG/cursor inherited from epoch26.')

if __name__=='__main__':main()
