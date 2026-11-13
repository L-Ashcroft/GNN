"""Resource-gated CUDA correctness gate on true fold-training graphs only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import json,time,copy
import psutil
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
cfg=json.loads((HERE/'config.json').read_text());out=ROOT/cfg['output']
if psutil.virtual_memory().available<cfg['minimum_CUDA_start_RAM_GiB']*2**30:raise SystemExit(75)
for p in psutil.process_iter(['name','cmdline']):
    if p.pid in {os.getpid(),os.getppid()}:continue
    if (p.info['name'] or '').lower() in {'python.exe','pythonw.exe'} and any(str(a).replace('\\','/').endswith('loto_training_v1/train.py') for a in p.info['cmdline'] or []):raise RuntimeError('A trainer is still running')
import runtime
import numpy as np
import torch
from data import FoldData,sha
from fit import fit_stats
from train import build_model,replace
from model import loss_by_contract
from src.train.stage import build_optimizer
torch.set_num_threads(cfg['cpu_threads']);torch.use_deterministic_algorithms(True);torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
reader=FoldData(cfg['fold_id'],2,verify=False);stats=fit_stats(reader,out/'normalization.json');cpu=build_model(reader,stats,cfg)
saved=torch.load(out/'last.pt',map_location='cpu',weights_only=True) if (out/'last.pt').exists() else None
if saved is not None:cpu.load_state_dict(saved['model'])
gpu=copy.deepcopy(cpu).to('cuda');op_cpu=build_optimizer(cpu,'finetune_all',cfg['adaptation_lr'],cfg['backbone_lr'],cfg['weight_decay']);op_gpu=build_optimizer(gpu,'finetune_all',cfg['adaptation_lr'],cfg['backbone_lr'],cfg['weight_decay'])
if saved is not None:op_cpu.load_state_dict(saved['optimizer']);op_gpu.load_state_dict(saved['optimizer'])
# Greedy train-only topology coverage includes all48relations without a large
# batch. Add a compatible peer of the actual unseen target's output contract.
choices=[];seen=set();uncovered=set(range(48))
for number in np.unique(reader.refs['train'][:,0]):
    name=reader.topologies[int(number)];st=reader.fetch(name)['static'];choices.append((int(number),set(st['edge_type'])))
selected=[]
while uncovered:
    number,types=max(choices,key=lambda item:len(item[1]&uncovered))
    if not types&uncovered:raise ValueError('Training structures do not cover all relations')
    selected.append(int(np.flatnonzero(reader.refs['train'][:,0]==number)[0]));uncovered-=types
peer=reader.fold['compatible_training_topologies'][0];number=reader.topologies.index(peer);selected.append(int(np.flatnonzero(reader.refs['train'][:,0]==number)[0]))
selected=list(dict.fromkeys(selected));b_cpu=reader.batch('train',np.array(selected));b_gpu=b_cpu.to('cuda');cpu.eval();gpu.eval();rows=[]
def compare(name,a,b,atol,rtol):
    aa=a.detach().cpu().to(torch.float64);bb=b.detach().cpu().to(torch.float64);difference=(aa-bb).abs();ratio=difference/(atol+rtol*aa.abs());passed=bool((ratio<=1).all())
    rows.append({'name':name,'max_absolute_error':float(difference.max()) if difference.numel() else 0.,'max_tolerance_ratio':float(ratio.max()) if ratio.numel() else 0.,'atol':atol,'rtol':rtol,'passed':passed})
with torch.no_grad():
    a=cpu(b_cpu,return_states=True);b=gpu(b_gpu,return_states=True)
    for key in ['prediction','node_states','edge_states','embedding','node_attention','edge_attention']:compare(key,a[key],b[key],5e-6,5e-5)
op_cpu.zero_grad(set_to_none=True);op_gpu.zero_grad(set_to_none=True)
loss_cpu,per_cpu=loss_by_contract(cpu(b_cpu),b_cpu,cpu.head);loss_gpu,per_gpu=loss_by_contract(gpu(b_gpu),b_gpu,gpu.head)
compare('masked_loss',loss_cpu,loss_gpu,2e-6,2e-5)
for g in per_cpu:compare('group_loss/'+g,per_cpu[g],per_gpu[g],2e-6,2e-5)
loss_cpu.backward();loss_gpu.backward()
for (n,p),(m,q) in zip(cpu.named_parameters(),gpu.named_parameters()):
    assert n==m and (p.grad is None)==(q.grad is None)
    if p.grad is not None:compare('gradient/'+n,p.grad,q.grad,2e-5,.002)
torch.nn.utils.clip_grad_norm_(cpu.parameters(),5.,error_if_nonfinite=True);torch.nn.utils.clip_grad_norm_(gpu.parameters(),5.,error_if_nonfinite=True);op_cpu.step();op_gpu.step()
for (n,p),(m,q) in zip(cpu.named_parameters(),gpu.named_parameters()):compare('AdamW/'+n,p,q,5e-6,5e-5)
# Determinism is tested within each backend. CPU and CUDA Dropout do not share
# a bitwise RNG stream; the recipe records any backend-transition seed.
cpu.train();gpu.train()
with torch.no_grad():
    torch.manual_seed(1234);a=cpu(b_cpu);torch.manual_seed(1234);aa=cpu(b_cpu)
    cpu_repeat=torch.equal(a,aa)
    torch.cuda.manual_seed_all(1234);b=gpu(b_gpu);torch.cuda.manual_seed_all(1234);bb=gpu(b_gpu);gpu_repeat=torch.equal(b,bb)
torch.cuda.synchronize();events=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
events[0].record();op_gpu.zero_grad(set_to_none=True);loss,_=loss_by_contract(gpu(b_gpu),b_gpu,gpu.head);loss.backward();op_gpu.step();events[1].record();events[1].synchronize()
result={'passed':all(r['passed'] for r in rows) and cpu_repeat and gpu_repeat,'model_source_sha256':sha(HERE/'model.py'),'normalization_sha256':sha(out/'normalization.json'),
    'checkpoint_sha256':sha(out/'last.pt') if (out/'last.pt').exists() else None,'all48relations':len(torch.unique(b_cpu.edge_type))==48,
    'real_training_graphs':len(selected),'comparisons':rows,'CPU_dropout_repeat_exact':cpu_repeat,'GPU_dropout_repeat_exact':gpu_repeat,
    'Dropout_cross_backend_bitwise_claim':False,'GPU_small_train_step_milliseconds_CUDA_events':events[0].elapsed_time(events[1]),
    'host_RSS_MiB':psutil.Process().memory_info().rss/2**20,'GPU_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
    'heldout_test_labels_used':False,'admission_available_RAM_GiB_requirement':4.}
path=out/'cuda_acceptance.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2),encoding='utf-8');replace(temp,path)
print(json.dumps({k:v for k,v in result.items() if k!='comparisons'},indent=2));reader.close()
if not result['passed']:raise SystemExit(2)
