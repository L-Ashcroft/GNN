"""CPU checks for independent semantic routing, initialization and metrics."""
from pathlib import Path
import sys,json,copy
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'versions/rgcn_calibrated_head_v1'))
torch.set_num_threads(1);torch.manual_seed(42)
from model import DirectResidualHead
from src.train.metrics import MetricAccumulator
from types import SimpleNamespace
tasks=json.loads((ROOT/'data/model_ready/fusion_v2_training_v1/tasks.json').read_text());cal=json.loads((ROOT/'data/model_ready/fusion_v2_accuracy_v2/target_calibration.json').read_text())
head=DirectResidualHead(ROOT,tasks,cal);head.eval()
meta=ROOT/'data/model_ready/fusion_v2_frozen_e19_bf0d71ad';manifest=json.loads((meta/'manifest.json').read_text());groups=manifest['groups']
e=np.load(meta/'validation/embedding.npy',mmap_mode='r');numbers=np.load(meta/'validation/group.npy',mmap_mode='r');targets=np.load(meta/'validation/target.npy',mmap_mode='r');mask=np.load(meta/'validation/mask.npy',mmap_mode='r')
selected=[]
for number in [0,len(groups)//2,len(groups)-1]:selected.extend(np.flatnonzero(numbers==number)[:4].tolist())
x=torch.from_numpy(e[selected].copy());routed=[groups[int(numbers[i])] for i in selected];pred=head(x,routed).detach();expected=[]
for i,g in enumerate(routed):
    rec=np.load(ROOT/'artifacts/experiment1_representation_diagnostic_v1'/(g+'_embedding_linear.npz'),allow_pickle=False);idx=np.load(ROOT/'artifacts/experiment1_local_readout_v1'/(g+'.npz'),allow_pickle=False)
    active=rec['embedding_active'];features=(x[i].numpy().astype('float64')[active]-rec['embedding_mean'][active])/rec['embedding_std'][active]
    z=np.r_[1.,features]@rec['coefficients'];expected.append(np.clip(z,idx['bounds'][0],idx['bounds'][1]))
expected=torch.tensor(np.stack(expected));observed=torch.tensor(mask[selected].copy())
torch.testing.assert_close(pred[observed],expected[observed],atol=2e-5,rtol=2e-5)
max_error=float((pred[observed]-expected[observed]).abs().max())
clone=DirectResidualHead.from_state(tasks,head.state_dict());clone.eval();torch.testing.assert_close(clone(x,routed),pred,atol=0,rtol=0)
group=routed[0];block=head.contracts[group];before={g:[p.detach().clone() for p in b.parameters()] for g,b in head.contracts.items() if g!=group}
columns=sorted(map(int,tasks[group]['slots']));ix=[i for i,g in enumerate(routed) if g==group];xx=x[ix];yy=torch.tensor(targets[np.array(selected)[ix]].copy());mm=torch.tensor(mask[np.array(selected)[ix]].copy());scale=torch.tensor([[tasks[group]['slots'].get(str(j),{}).get('scale',1.) for j in range(16)]]*len(ix))
head.train();opt=torch.optim.AdamW(block.parameters(),lr=1e-4);loss=block.loss(xx,yy,mm,scale);loss.backward();torch.nn.utils.clip_grad_norm_(block.parameters(),5.,error_if_nonfinite=True);opt.step()
assert torch.isfinite(loss)
for g,old in before.items():
    for p,q in zip(head.contracts[g].parameters(),old):torch.testing.assert_close(p,q,atol=0,rtol=0)
bad=yy.clone();bad[~mm]=float('nan');head.eval();torch.testing.assert_close(block.loss(xx,yy,mm,scale),block.loss(xx,bad,mm,scale))
try:head(x,['unknown']*len(x));raise AssertionError('Unknown contract accepted')
except ValueError:pass
import ast,statistics
namespace={'np':np,'statistics':statistics,'tasks':tasks,'LOG':{'dB','dBm','dBc/Hz','source_label_dBm_compression_function'}}
for node in ast.parse((ROOT/'worktrees/representation_diagnostic_v1/run.py').read_text(encoding='utf-8')).body:
    if isinstance(node,ast.FunctionDef) and node.name=='metrics':exec(compile(ast.Module(body=[node],type_ignores=[]),'unchanged_metrics','exec'),namespace)
scales=torch.tensor([[tasks[g]['slots'].get(str(j),{}).get('scale',1.) for j in range(16)] for g in routed])
batch=SimpleNamespace(scales=scales,targets=torch.tensor(targets[selected].copy()),mask=observed,group_ids=routed)
original=MetricAccumulator(tasks);original.update(pred,batch);original={(r['supervision_group'],r['semantic_id']):r for r in original.compute()['per_target']}
physical=pred.sign().numpy()*np.expm1(abs(pred.numpy()))*scales.numpy().astype('float64')
for g in set(routed):
    ids=[i for i,q in enumerate(routed) if q==g]
    for row in namespace['metrics'](physical[ids],targets[np.array(selected)[ids]],observed.numpy()[ids],g):
        ref=original[(g,row['semantic_id'])]
        for key in ['mae','r2','hit5','hit10']:
            if row[key] is None:assert ref[key] is None
            else:assert abs(row[key]-ref[key])<=1e-8*max(1,abs(ref[key])),(key,row[key],ref[key])
result={'ridge_initialization_observed_max_symlog_abs_error':max_error,'ridge_initialization_tolerance':{'atol':2e-5,'rtol':2e-5},
    'state_reload_exact':True,'other_contract_parameters_unchanged':True,'missing_NaN_loss_safe':True,'unknown_contract_rejected':True,
    'finite_gradient_AdamW':True,'original_metric_accumulator_match':True,'label_free_inference_constructor_exact':True,'target_affine_precision':'float64; normalized MLP weights/activations float32',
    'test_read':False,'trainable_head_parameters':sum(p.numel() for p in head.parameters())}
(HERE/'checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
