"""Portable edge-RGCN with shared-compatible direct/residual target heads."""
import copy
import torch
from torch import nn
from src.model.network import EdgeCentricRGCN
class PortableHead(nn.Module):
    def __init__(self,hidden,tasks,stats):
        super().__init__();self.groups=sorted(tasks);self.group_index={g:i for i,g in enumerate(self.groups)};self.slots={g:tuple(sorted(map(int,t['slots']))) for g,t in tasks.items()}
        centers=torch.zeros(len(tasks),16,dtype=torch.float64);std=torch.ones_like(centers);self.blocks=nn.ModuleDict();self.direct=nn.ModuleDict();self.trained_slots={}
        for g in self.groups:
            row=stats['targets'][g];centers[self.group_index[g]]=torch.tensor(row['center'],dtype=torch.float64);std[self.group_index[g]]=torch.tensor(row['std'],dtype=torch.float64)
            self.trained_slots[g]={j for j,c in enumerate(row['count']) if c>0}
            self.direct[g]=nn.Linear(hidden,len(self.slots[g]));nn.init.zeros_(self.direct[g].weight);nn.init.zeros_(self.direct[g].bias)
            self.blocks[g]=nn.Sequential(nn.Linear(hidden,64),nn.SiLU(),nn.Linear(64,len(self.slots[g])))
            nn.init.zeros_(self.blocks[g][-1].weight);nn.init.zeros_(self.blocks[g][-1].bias)
        self.register_buffer('center',centers);self.register_buffer('std',std)
    def affine(self,groups):
        index=torch.tensor([self.group_index[g] for g in groups],device=self.center.device);return self.center[index],self.std[index]
    def check_observed(self,coverage):
        for g,slots in coverage.items():
            if g not in self.trained_slots or not set(slots)<=self.trained_slots[g]:raise ValueError('Observed target has no fold-training calibration: '+g)
    def forward(self,embedding,groups,group_plan=None):
        if group_plan is None:
            mapping={}
            for i,g in enumerate(groups):mapping.setdefault(g,[]).append(i)
            blocks=[(g,torch.tensor(rows,device=embedding.device)) for g,rows in mapping.items()]
        else:
            slices,order=group_plan;blocks=[(g,order[start:end]) for g,start,end in slices]
        output=embedding.new_zeros(len(groups),16)
        for g,index in blocks:
            v=self.direct[g](embedding[index])+self.blocks[g](embedding[index]);columns=torch.tensor(self.slots[g],device=embedding.device)
            output=output.index_copy(0,index,embedding.new_zeros(len(index),16).index_copy(1,columns,v))
        center,std=self.affine(groups);return center+std*output.to(torch.float64)
class FoldRGCN(EdgeCentricRGCN):
    def __init__(self,schema,tasks,stats,hidden_dim=128,num_layers=4,dropout=.1,update_edges=True,use_domain_emb=True,use_film=True,collapse_relations=False):
        super().__init__(schema,tasks,hidden_dim,num_layers,dropout,update_edges,use_domain_emb,use_film,collapse_relations)
        self.head=PortableHead(hidden_dim,tasks,stats)
        self.register_buffer('input_center',torch.tensor([r['center'] for r in stats['inputs']],dtype=torch.float32))
        self.register_buffer('input_std',torch.tensor([r['std'] for r in stats['inputs']],dtype=torch.float32))
    def forward(self,batch,return_states=False):
        local=copy.copy(batch);f=self.input_center.shape[1];domain=batch.domain_ids[batch.graph_index[batch.edge_index[0]]]
        original=batch.edge_attr;values=original[:,:f];known=original[:,f:2*f].bool();zero=original[:,2*f:].bool()
        converted=(values-self.input_center[domain])/self.input_std[domain]
        columns=torch.arange(f,device=values.device)<19
        values=torch.where(known&~zero&columns[None,:],converted,values)
        local.edge_attr=torch.cat([values,original[:,f:]],dim=1)
        return super().forward(local,return_states)
def loss_by_contract(prediction,batch,head):
    from src.model.execution import prepare_batch
    plan=prepare_batch(batch);head.check_observed(plan.coverage)
    y=batch.targets.to(torch.float64)/batch.scales.to(torch.float64);z=y.sign()*torch.log1p(y.abs());_,std=head.affine(batch.group_ids);losses={}
    for g,start,end in plan.loss_slices:
        positions=plan.tensors['loss_order'][start:end];error=((prediction-z)/std).reshape(-1)[positions]
        if not torch.isfinite(error).all():raise ValueError('Nonfinite observed normalized loss')
        losses[g]=error.square().mean()
    if not losses:raise ValueError('No observed targets')
    return torch.stack(list(losses.values())).mean(),losses
def weighted_loss(losses,slices,full_counts):
    sizes={g:end-start for g,start,end in slices};parts=[]
    for g,loss in losses.items():
        if g not in full_counts or sizes[g]>full_counts[g]:raise ValueError('Effective contract counts mismatch')
        parts.append(loss*(sizes[g]/full_counts[g])/len(full_counts))
    return torch.stack(parts).sum()
