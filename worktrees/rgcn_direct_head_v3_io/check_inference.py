"""Actual raw graph vs frozen-feature CPU inference gate using prior tolerances."""
from pathlib import Path
import sys,json
import psutil
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
if psutil.virtual_memory().available<1.25*2**30:raise SystemExit(75)
import numpy as np
import torch
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(42)
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'worktrees/autonomous_training_v1'))
from inference import load_model
from stream_graphs import StreamGraphs
net,saved=load_model();reader=StreamGraphs(ROOT,'data/model_ready/fusion_v2_training_v1','data/model_ready/fusion_v2_frozen_e19_bf0d71ad','versions/rgcn_calibrated_head_v1',capacity=2)
idx=reader.stratified_indices('validation',32,42);batch=reader.batch('validation',idx,sort_for_io=False)
e=np.load(ROOT/'data/model_ready/fusion_v2_frozen_e19_bf0d71ad/validation/embedding.npy',mmap_mode='r')
with torch.no_grad():
    raw=net(batch,return_states=True);expected=torch.from_numpy(e[idx].copy());cached=net.head(expected,batch.group_ids)
comparisons=[];passed=True
for name,a,b in [('raw_graph_vs_cache_embedding',raw['embedding'],expected),('raw_graph_vs_cached_head_prediction',raw['prediction'],cached)]:
    error=(a-b).abs();ratio=error/(5e-6+5e-5*b.abs());ok=bool((ratio<=1).all());passed&=ok
    comparisons.append({'name':name,'max_absolute_error':float(error.max()),'max_tolerance_ratio':float(ratio.max()),'atol':5e-6,'rtol':5e-5,'passed':ok})
assert torch.isfinite(raw['prediction'][batch.mask]).all()
result={'passed':passed,'comparisons':comparisons,'real_graph_batch':len(idx),'sources':sorted({saved['tasks'][g]['contract']['source_dataset'] for g in batch.group_ids}),
    'relations_observed':len(torch.unique(batch.edge_type)),'checkpoint_epoch':saved['epoch'],'full_model_parameters':sum(p.numel() for p in net.parameters()),
    'head_parameters':sum(p.numel() for p in net.head.parameters()),'head_precision':'FP32 normalized network/FP64 target affine',
    'no_training_labels_needed_to_load_model':True,'test_read':False,'device':'cpu'}
(HERE/'inference_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2));reader.close()
if not passed:raise SystemExit(2)
