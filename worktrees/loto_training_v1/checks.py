"""Fold/normalizer/micro-loss CPU correctness and small real throughput probe."""
from pathlib import Path
import sys,json,copy,time,threading,os
import psutil
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
if psutil.virtual_memory().available<1.5*2**30:raise SystemExit(75)
import runtime
import numpy as np
import torch
from data import FoldData
from fit import fit_stats
from model import FoldRGCN,loss_by_contract,weighted_loss
from src.model.execution import prepare_batch
from src.train.stage import build_optimizer
torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.manual_seed(42)
cfg=json.loads((HERE/'config.json').read_text());out=ROOT/cfg['output'];out.mkdir(parents=True,exist_ok=True)
reader=FoldData(cfg['fold_id'],verify=True);stats=fit_stats(reader,out/'normalization.json')
assert stats['proof']['training_rows']==446984 and stats['proof']['test_or_validation_labels_used']==False
assert set(reader.tasks)==set(stats['targets']) and len(reader.tasks)==129
assert not set(reader.topologies[int(n)] for n in np.unique(reader.refs['train'][:,0]))&reader.held_topologies
original=reader.refs['train'][0].copy();reader.refs['train'][0]=reader.refs['test'][0]
try:reader.batch('train',np.array([0]));raise AssertionError('Heldout structure admitted to train')
except ValueError:pass
reader.refs['train'][0]=original
kw={k:cfg[k] for k in ['hidden_dim','num_layers','dropout','update_edges','use_domain_emb','use_film','collapse_relations']}
model=FoldRGCN(reader.schema,reader.tasks,stats,**kw);optimizer=build_optimizer(model,'finetune_all',cfg['adaptation_lr'],cfg['backbone_lr'],cfg['weight_decay'])
rng=np.random.default_rng(42);small=rng.choice(len(reader.refs['train']),8,replace=False)
batch=reader.batch('train',small)
for _ in range(2):
    model.train();optimizer.zero_grad(set_to_none=True);loss,_=loss_by_contract(model(batch),batch,model.head);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
left=model.eval();right=copy.deepcopy(model).eval();left.zero_grad(set_to_none=True);right.zero_grad(set_to_none=True)
pred=left(batch);full,_=loss_by_contract(pred,batch,left.head);full.backward();counts=reader.counts('train',small);total=0.
for offset in [0,4]:
    b=reader.batch('train',small[offset:offset+4]);_,per=loss_by_contract(right(b),b,right.head);part=weighted_loss(per,prepare_batch(b).loss_slices,counts);part.backward();total+=float(part.detach())
torch.testing.assert_close(full.detach(),torch.tensor(total,dtype=full.dtype),atol=2e-6,rtol=2e-5)
maximum=0.;ratio=0.;backbone_grad=0.
for (name,a),(other,b) in zip(left.named_parameters(),right.named_parameters()):
    assert name==other and (a.grad is None)==(b.grad is None)
    if a.grad is not None:
        err=(a.grad-b.grad).abs();maximum=max(maximum,float(err.max()));ratio=max(ratio,float((err/(2e-5+.002*a.grad.abs())).max()));torch.testing.assert_close(a.grad,b.grad,atol=2e-5,rtol=.002)
        if name.startswith('node_encoder'):backbone_grad+=float(a.grad.abs().sum())
assert backbone_grad>0
del right,left,batch,pred,full;model.zero_grad(set_to_none=True)
resource={'peak_RSS':0,'minimum_available':psutil.virtual_memory().available};done=threading.Event()
def guard():
    while not done.wait(.1):
        av=psutil.virtual_memory().available;resource['peak_RSS']=max(resource['peak_RSS'],psutil.Process().memory_info().rss);resource['minimum_available']=min(resource['minimum_available'],av)
        if av<512*2**20:os._exit(86)
threading.Thread(target=guard,daemon=True).start();benchmark=[];relations=set();before=model.node_encoder.weight.detach().clone()
for step in range(3):
    idx=rng.choice(len(reader.refs['train']),256,replace=False);counts=reader.counts('train',idx);model.train();optimizer.zero_grad(set_to_none=True);tick=time.perf_counter();value=0.
    for off in [0,128]:
        b=reader.batch('train',idx[off:off+128]);relations.update(b.edge_type.tolist());_,per=loss_by_contract(model(b),b,model.head);loss=weighted_loss(per,prepare_batch(b).loss_slices,counts);loss.backward();value+=float(loss.detach());del b,per,loss
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step();benchmark.append({'seconds':time.perf_counter()-tick,'normalized_loss':value,'gradient_norm':float(norm)})
    print('LOTO CPU step',step+1,benchmark[-1],flush=True)
done.set();assert not torch.equal(before,model.node_encoder.weight.detach())
result={'passed':True,'fold_id':cfg['fold_id'],'counts':reader.fold['counts'],'sources':['FALCON','AICircuit','LOADBench'],'private_data_used':False,
    'heldout_training_guard_passed':True,'alias_contracts':len(reader.tasks),'normalization_training_rows':stats['proof']['training_rows'],
    'no_ID_weights_or_embeddings_loaded':True,'micro_loss_equal':True,'micro_gradient_max_absolute_error':maximum,'micro_gradient_max_tolerance_ratio':ratio,
    'gradient_tolerance':{'atol':2e-5,'rtol':.002},'nonzero_graph_backbone_gradient':backbone_grad,
    'full_model_parameters':sum(p.numel() for p in model.parameters()),'all_parameters_trainable':all(p.requires_grad for p in model.parameters()),
    'CPU_effective_batch':256,'microbatch':128,'steps':benchmark,'samples_per_second':768/sum(s['seconds'] for s in benchmark),
    'estimated_training_epoch_minutes':len(reader.refs['train'])/(768/sum(s['seconds'] for s in benchmark))/60,
    'relations_observed_in_probe':len(relations),'peak_RSS_MiB':resource['peak_RSS']/2**20,'minimum_available_RAM_MiB':resource['minimum_available']/2**20,
    'heldout_test_labels_used':False}
(HERE/'checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2));reader.close()
