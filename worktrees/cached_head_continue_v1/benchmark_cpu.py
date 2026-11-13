"""Small real cached-head CPU benchmark, no GNN and no test data."""
from pathlib import Path
import sys
import time
import json
import threading
import os
import psutil
ROOT=Path(__file__).resolve().parents[2];PACKAGE=Path(__file__).resolve().parent
if psutil.virtual_memory().available<1024*2**20:raise SystemExit(75)
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
stop=threading.Event();resources=[]
def guard():
    while not stop.wait(.1):
        memory=psutil.virtual_memory().available;resources.append((psutil.Process().memory_info().rss,memory))
        if memory<384*2**20:os._exit(86)
threading.Thread(target=guard,daemon=True).start()
import numpy as np
import torch
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
sys.path.insert(0,str(ROOT/'versions/rgcn_calibrated_head_v1'))
from src.model.contract_head import ContractResidualHead
from src.model.grouped_contract_head import forward_grouped
read=lambda p:json.loads(p.read_text())
config=read(PACKAGE/'config.json');parent=ROOT/config['parent_output'];saved=torch.load(parent/'best.pt',map_location='cpu',weights_only=True)
head=ContractResidualHead(128,saved['tasks'],saved['target_calibration'],contract_residual=False,calibrated_targets=True)
head.load_state_dict({k[5:]:v for k,v in saved['model'].items() if k.startswith('head.')})
tasks=saved['tasks'];del saved
cache=ROOT/config['feature_cache'];manifest=read(cache/'manifest.json');names=manifest['groups'];arrays={k:np.load(cache/'train'/(k+'.npy'),mmap_mode='r') for k in ['embedding','target','mask','group']}
indices=np.random.default_rng(42).choice(manifest['roles']['train']['rows'],1024,replace=False)
x=torch.from_numpy(np.asarray(arrays['embedding'][indices]).copy());target=torch.from_numpy(np.asarray(arrays['target'][indices]).copy());mask=np.asarray(arrays['mask'][indices]);numbers=np.asarray(arrays['group'][indices])
groups=[names[int(i)] for i in numbers];scale=torch.tensor([[tasks[g]['slots'].get(str(i),{}).get('scale',1.) for i in range(16)] for g in groups])
transformed=target.sign()*torch.log1p((target/scale).abs());_,std=head.affine(groups)
rows=[];slices=[];positions=[]
for number in sorted(set(numbers.tolist())):
    g=names[number];take=np.flatnonzero(numbers==number);begin=len(rows);rows.extend(take.tolist());slices.append((g,begin,len(rows)))
    positions.append(torch.tensor((take[:,None]*16+np.arange(16)[None,:])[mask[take]],dtype=torch.long))
group_plan=slices,torch.tensor(rows,dtype=torch.long);optimizer=torch.optim.AdamW(head.parameters(),lr=5e-5,weight_decay=.01)
measurements=[]
for step in range(5):
    tick=time.perf_counter();optimizer.zero_grad(set_to_none=True);p=forward_grouped(head,x,groups,group_plan)
    losses=[((p.reshape(-1)[ix]-transformed.reshape(-1)[ix])/std.reshape(-1)[ix]).square().mean() for ix in positions if len(ix)]
    loss=torch.stack(losses).mean();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);optimizer.step()
    if step:measurements.append(time.perf_counter()-tick)
    assert torch.isfinite(loss)
stop.set();result={'device':'cpu','CPU_threads':2,'batch_size':1024,'complete_train_step_seconds':measurements,'mean_seconds':sum(measurements)/len(measurements),
    'samples_per_second':1024/(sum(measurements)/len(measurements)),'real_training_graph_embeddings':True,'test_data_read':False,
    'peak_RSS_MiB':max(v[0] for v in resources)/2**20,'minimum_available_RAM_MiB':min(v[1] for v in resources)/2**20,
    'scope':'Fixed real cached Batch1024, no I/O in step timing; optimizer fresh, not predictive-quality assessment'}
(PACKAGE/'cpu_benchmark.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
