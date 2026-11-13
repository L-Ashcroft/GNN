"""CPU-only check of the active cache's completed prefix and exact loss formula."""
from pathlib import Path
import ast
import json
import sys
from types import SimpleNamespace
import psutil
PACKAGE=Path(__file__).resolve().parents[1];ROOT=PACKAGE.parents[1];sys.path.insert(0,str(PACKAGE))

def main():
    if psutil.virtual_memory().available<1.2*2**30:raise SystemExit('CPU cache verification deferred for RAM')
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    import numpy as np
    import torch
    from src.graph.builder import CircuitBatch
    from src.model.execution import prepare_batch
    from src.model.contract_head import ContractResidualHead
    from src.train.accuracy_loss import supervision_loss
    torch.set_num_threads(1);read=lambda p:json.loads(p.read_text())
    plan=read(ROOT/'monitoring/accuracy_acceleration_v1/plan.json');out=ROOT/plan['output'];cache=ROOT/plan['feature_cache']
    status=read(out/'status.json');manifest=read(ROOT/'data/model_ready/fusion_v2_training_v1/manifest.json')
    tasks=read(ROOT/'data/model_ready/fusion_v2_training_v1/tasks.json');groups=sorted(tasks);topologies=list(manifest['topologies'])
    role=status.get('role','train');done=status.get('samples_done',0)
    if status['state']!='building_feature_cache':role='train';done=manifest['role_counts']['train']
    if done<3:raise RuntimeError('Completed cache prefix too small')
    arrays={k:np.load(cache/role/(k+'.npy'),mmap_mode='r') for k in ['embedding','target','mask','group','refs']}
    selected=sorted(set([0,done//4,done//2,3*done//4,done-1]));checks=[]
    for i in selected:
        number,row=arrays['refs'][i];name=topologies[int(number)];folder=ROOT/manifest['topologies'][name]['folder']
        a={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['target','mask','group','role','sample_ids']}
        assert int(a['role'][int(row)])=={'train':0,'validation':1}[role]
        assert np.array_equal(a['target'][int(row)],arrays['target'][i]);assert np.array_equal(a['mask'][int(row)],arrays['mask'][i])
        expected=manifest['group_names'][int(a['group'][int(row)])];assert groups[int(arrays['group'][i])]==expected
        assert arrays['embedding'][i].shape==(128,) and np.isfinite(arrays['embedding'][i]).all()
        checks.append({'cache_row':i,'sample_id':a['sample_ids'][int(row)].decode(),'group':expected,'role':role})
    # Execute the actual function body, not a reimplementation of its objective.
    tree=ast.parse((PACKAGE/'scripts/run_cached_head_experiment.py').read_text())
    main_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    function=next(n for n in main_node.body if isinstance(n,ast.FunctionDef) and n.name=='objective')
    calibration=read(ROOT/'data/model_ready/fusion_v2_accuracy_v2/target_calibration.json');head=ContractResidualHead(128,tasks,calibration,contract_residual=False)
    namespace={'torch':torch,'head':head};exec(compile(ast.Module(body=[function],type_ignores=[]),'actual_cached_head_objective','exec'),namespace)
    fields=torch.load(ROOT/'reports/experiment1_performance/fixtures/batch_256_11.pt',weights_only=True)
    batch=CircuitBatch(**fields);prepared=prepare_batch(batch)
    meta=SimpleNamespace(targets=batch.targets,scales=batch.scales,group_ids=batch.group_ids,coverage=prepared.coverage,
        loss_slices=prepared.loss_slices,loss_order=prepared.tensors['loss_order'])
    torch.manual_seed(42);a=torch.randn_like(batch.targets,requires_grad=True);b=a.detach().clone().requires_grad_()
    left,_=supervision_loss(a,batch,head);right=namespace['objective'](b,meta)
    torch.testing.assert_close(left,right,atol=0,rtol=0);left.backward();right.backward();torch.testing.assert_close(a.grad,b.grad,atol=0,rtol=0)
    report={'status':'passed','CPU_only':True,'CUDA_used':False,'completed_cache_prefix':done,'source_checks':checks,
        'real_loss_batch_size':256,'relations_in_fixture':48,'contracts_in_fixture':len(set(batch.group_ids)),
        'actual_cached_loss_vs_reference_max_abs':float((left-right).detach().abs()),'prediction_gradient_max_abs':float((a.grad-b.grad).abs().max()),
        'original_targets_masks_roles_and_group_routing_preserved':True}
    (PACKAGE/'results/cached_stage_CPU_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
