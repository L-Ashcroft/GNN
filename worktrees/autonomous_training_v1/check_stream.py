"""CPU checks for owned-memmap lifetimes and exact accumulated group weights."""
from pathlib import Path
import sys
import json
import copy
import gc
import psutil
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
if psutil.virtual_memory().available<1.1*2**30:raise SystemExit(75)
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
sys.path.insert(0,str(ROOT/'versions/rgcn_calibrated_head_v1'))
import torch
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
import src.common as common
common.ROOT=ROOT
from stream_graphs import StreamGraphs,weighted_micro_loss
from src.model.accuracy_network import AccuracyRGCN
from src.model.execution import prepare_batch
from src.train.accuracy_loss import supervision_loss
from src.model.grouped_contract_head import forward_grouped
from types import MethodType
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
feature='data/model_ready/fusion_v2_frozen_e19_bf0d71ad';raw='data/model_ready/fusion_v2_training_v1';version='versions/rgcn_calibrated_head_v1'
reader=StreamGraphs(ROOT,raw,feature,version,capacity=1)
fixtures=read(ROOT/'reports/experiment1_performance/fixtures/manifest.json')['selections'];item=next(x for x in fixtures if x['size']==256)
batch=reader.batch_refs(item['references'],'train');fields=torch.load(ROOT/'reports/experiment1_performance/fixtures'/item['file'],weights_only=True)
for name in ['x','edge_index','edge_attr','edge_type','graph_index','targets','mask','scales','domain_ids']:
    torch.testing.assert_close(getattr(batch,name),fields[name],atol=0,rtol=0)
for name in ['group_ids','semantic_ids','sample_ids']:assert getattr(batch,name)==fields[name]
assert len(reader.entries)==1 and reader.evictions>1
checks={'batch256_original_tensors_exact':True,'all_48_relations':len(torch.unique(batch.edge_type))==48,
        'eviction_count':reader.evictions,'owned_target_mask_views_safe_after_eviction':True}
del batch,fields;gc.collect()
# Analytic ragged-mask/group objective and gradients, unrelated to device backend.
torch.manual_seed(42);p=torch.randn(6,3,dtype=torch.float64,requires_grad=True);q=p.detach().clone().requires_grad_()
y=torch.randn(6,3,dtype=torch.float64);mask=torch.tensor([[1,0,1],[1,1,1],[0,1,0],[1,0,0],[1,1,0],[0,0,1]],dtype=torch.bool);groups=['a','b','a','b','a','c']
counts={g:int(mask[[i for i,x in enumerate(groups) if x==g]].sum()) for g in set(groups)}
full=torch.stack([((p[[i for i,x in enumerate(groups) if x==g]]-y[[i for i,x in enumerate(groups) if x==g]])[mask[[i for i,x in enumerate(groups) if x==g]]]).square().mean() for g in sorted(counts)]).mean();full.backward()
value=0.
for first,last in [(0,2),(2,5),(5,6)]:
    losses={};slices=[];offset=0
    for g in sorted(set(groups[first:last])):
        rows=[i for i in range(first,last) if groups[i]==g];n=int(mask[rows].sum())
        losses[g]=(q[rows]-y[rows])[mask[rows]].square().mean();slices.append((g,offset,offset+n));offset+=n
    loss=weighted_micro_loss(losses,slices,counts);value+=float(loss.detach());loss.backward()
torch.testing.assert_close(p.grad,q.grad,atol=1e-14,rtol=1e-14);assert abs(float(full.detach())-value)<1e-14
checks['ragged_mask_effective_loss_and_gradients_exact']=True
# Real four-layer model: compare one 8-graph batch and two 4-graph micros,
# with Dropout disabled for a deterministic mathematical check.
if psutil.virtual_memory().available>=800*2**20:
    parent=torch.load(ROOT/'artifacts/experiment1_cached_head_cpu_continue_v1/best.pt',map_location='cpu',weights_only=True)
    kwargs={k:parent['config'][k] for k in ['hidden_dim','num_layers','dropout','update_edges','use_domain_emb','use_film','collapse_relations','contract_residual','calibrated_targets','adapter_hidden','adapter_min_train_samples']}
    left=AccuracyRGCN(parent['schema'],parent['tasks'],parent['target_calibration'],**kwargs).double();left.load_state_dict(parent['model']);left.eval()
    right=copy.deepcopy(left)
    def grouped_forward(self,embedding,groups,group_plan=None):return forward_grouped(self,embedding,groups,group_plan)
    right.head.forward=MethodType(grouped_forward,right.head)
    refs=next(x for x in fixtures if x['size']==16)['references'][:8]
    def convert(b):
        b.x=b.x.double();b.edge_attr=b.edge_attr.double();b.targets=b.targets.double();b.scales=b.scales.double();return b
    whole=convert(reader.batch_refs(refs,'train'));prediction=left(whole);loss,_=supervision_loss(prediction,whole,left.head);loss.backward()
    plan=prepare_batch(whole);counts={g:end-start for g,start,end in plan.loss_slices};parts=[];total=0.
    for start in [0,4]:
        b=convert(reader.batch_refs(refs[start:start+4],'train'));prediction=right(b);_,per=supervision_loss(prediction,b,right.head)
        piece=weighted_micro_loss(per,prepare_batch(b).loss_slices,counts);piece.backward();total+=float(piece.detach());parts.append(prediction.detach())
    torch.testing.assert_close(left(whole).detach(),torch.cat(parts),atol=1e-9,rtol=1e-8)
    maximum=0.
    for (name,a),(other,b) in zip(left.named_parameters(),right.named_parameters()):
        assert name==other and (a.grad is None)==(b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad,b.grad,atol=1e-8,rtol=1e-7);maximum=max(maximum,float((a.grad-b.grad).abs().max()))
    checks['real_4layer128dim_full_vs_micro_gradient_max_abs']=maximum;checks['real_graph_micro_accumulation_passed']=True
    checks['grouped_head_and_backbone_gradients_checked']=True
else:checks['real_graph_micro_accumulation_passed']=False;checks['reason']='deferred RAM'
reader.close();(HERE/'stream_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');print(json.dumps(checks,indent=2))
