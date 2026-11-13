"""Small real CPU joint-training admission probe; never touches test data."""
from pathlib import Path
import sys
import json
import time
import threading
import os
import psutil
import argparse
ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--microbatch',type=int,default=32);parser.add_argument('--threads',type=int,default=2);parser.add_argument('--steps',type=int,default=3);args=parser.parse_args()
if psutil.virtual_memory().available<1.25*2**30:raise SystemExit(75)
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
minimum=psutil.virtual_memory().available;peak=0;done=threading.Event()
def guard():
    global minimum,peak
    while not done.wait(.1):
        available=psutil.virtual_memory().available;minimum=min(minimum,available);peak=max(peak,psutil.Process().memory_info().rss)
        if available<512*2**20:os._exit(86)
threading.Thread(target=guard,daemon=True).start()
sys.path.insert(0,str(ROOT/'versions/rgcn_calibrated_head_v1'))
sys.path.insert(0,str(ROOT/'worktrees/autonomous_training_v1'))
import torch
torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True);torch.manual_seed(42)
import src.common as common
common.ROOT=ROOT
from src.model.accuracy_network import AccuracyRGCN
from src.model.grouped_contract_head import forward_grouped
from src.model.execution import prepare_batch
from src.train.stage import build_optimizer
from src.train.accuracy_loss import supervision_loss
from stream_graphs import StreamGraphs,weighted_micro_loss
from types import MethodType
parent=torch.load(ROOT/'artifacts/experiment1_autonomous_head_v1/best.pt',map_location='cpu',weights_only=True)
keys=['hidden_dim','num_layers','dropout','update_edges','use_domain_emb','use_film','collapse_relations','contract_residual','calibrated_targets','adapter_hidden','adapter_min_train_samples']
model=AccuracyRGCN(parent['schema'],parent['tasks'],parent['target_calibration'],**{k:parent['config'][k] for k in keys})
model.load_state_dict(parent['model'],strict=True)
model.head.forward=MethodType(forward_grouped,model.head)
optimizer=build_optimizer(model,'finetune_all',lr_head=2.5e-5,lr_backbone=1e-5,weight_decay=.01)
reader=StreamGraphs(ROOT,'data/model_ready/fusion_v2_training_v1','data/model_ready/fusion_v2_frozen_e19_bf0d71ad','versions/rgcn_calibrated_head_v1',capacity=2)
indices=reader.stratified_indices('train',256*args.steps,42);steps=[];rel=set()
before=model.node_encoder.weight.detach().clone()
model.train()
for start in range(0,len(indices),256):
    selected=indices[start:start+256];counts=reader.counts('train',selected);tick=time.perf_counter();optimizer.zero_grad(set_to_none=True);value=0.
    for offset in range(0,len(selected),args.microbatch):
        batch=reader.batch('train',selected[offset:offset+args.microbatch]);rel.update(batch.edge_type.tolist())
        prediction=model(batch);_,groups=supervision_loss(prediction,batch,model.head)
        loss=weighted_micro_loss(groups,prepare_batch(batch).loss_slices,counts)
        assert torch.isfinite(loss);loss.backward();value+=float(loss.detach())
        del batch,prediction,groups,loss
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
    steps.append({'seconds':time.perf_counter()-tick,'loss':value,'gradient_norm':float(norm),'samples':len(selected)})
    print('joint CPU step',steps[-1],flush=True)
assert not torch.equal(before,model.node_encoder.weight.detach())
assert all(torch.isfinite(p).all() for p in model.parameters())
reader.close();done.set()
result={'status':'passed','device':'cpu','test_read':False,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
    'all_backbone_parameters_trainable':all(p.requires_grad for p in model.layers.parameters()),'node_encoder_updated':True,
    'effective_batch':256,'microbatch':args.microbatch,'threads':args.threads,'relations_observed':len(rel),'steps':steps,
    'peak_process_RSS_MiB':max(peak,psutil.Process().memory_info().rss)/2**20,'minimum_available_RAM_MiB':minimum/2**20,
    'samples_per_second':sum(s['samples'] for s in steps)/sum(s['seconds'] for s in steps)}
(HERE/f'probe_t{args.threads}_m{args.microbatch}.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)
