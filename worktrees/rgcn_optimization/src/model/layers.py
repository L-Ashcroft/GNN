"""Relation-grouped edge-state layers and graph-local attention pooling."""
import torch
from torch import nn


class EdgeRGCNLayer(nn.Module):
    def __init__(self,hidden_dim,num_relations,dropout=0.1,update_edges=True):
        super().__init__();self.update_edges=update_edges;self.num_relations=num_relations
        self.relation_weight=nn.Parameter(torch.empty(num_relations,2*hidden_dim,hidden_dim))
        for weight in self.relation_weight:nn.init.xavier_uniform_(weight)
        self.root=nn.Linear(hidden_dim,hidden_dim)
        self.node_norm=nn.LayerNorm(hidden_dim);self.edge_norm=nn.LayerNorm(hidden_dim)
        self.dropout=nn.Dropout(dropout)
        self.edge_update=nn.Sequential(nn.Linear(3*hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,hidden_dim)) if update_edges else None

    def forward(self,h,z,edge_index,edge_type,gamma=None,beta=None,relation_plan=None):
        src,dst=edge_index;aggregate=torch.zeros_like(h)
        messages=torch.cat([h[src],z],dim=-1)
        # Grouped GEMMs avoid materializing E x (2d) x d relation weights.
        if relation_plan is None:
            # Compatible standalone layer API; the full network supplies a plan.
            parts=[]
            for r in range(self.num_relations):
                indices=torch.nonzero(edge_type==r,as_tuple=False).flatten()
                if not indices.numel():continue
                destinations=dst[indices]
                counts=torch.bincount(destinations,minlength=h.shape[0]).clamp_min(1)[destinations]
                parts.append((r,indices,destinations,counts))
        else:
            slices,order,packed_dst,packed_counts=relation_plan
            parts=((r,order[start:end],packed_dst[start:end],packed_counts[start:end]) for r,start,end in slices)
        for r,indices,destinations,counts in parts:
            update=messages[indices]@self.relation_weight[r]
            update=update/counts[:,None].to(update.dtype)
            aggregate.index_add_(0,destinations,update)
        h=self.node_norm(h+self.dropout(torch.relu(self.root(h)+aggregate)))
        if gamma is not None:
            if beta is None or gamma.shape!=h.shape or beta.shape!=h.shape:raise ValueError('Invalid node FiLM shape')
            h=gamma*h+beta
        if self.update_edges:
            update=self.edge_update(torch.cat([h[src],h[dst],z],dim=-1))
            z=self.edge_norm(z+self.dropout(torch.relu(update)))
        return h,z


class AttentionPool(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__();self.score=nn.Sequential(nn.Linear(hidden_dim,hidden_dim//2),nn.Tanh(),nn.Linear(hidden_dim//2,1))

    def forward(self,h,graph_index,num_graphs,segment_plan=None):
        if segment_plan is None:
            # Standalone compatibility path; the network supplies CPU buckets.
            from types import SimpleNamespace
            from .execution import prepare_batch
            temporary=SimpleNamespace(x=h,edge_index=torch.empty((2,0),dtype=torch.long,device=h.device),
                edge_type=torch.empty(0,dtype=torch.long,device=h.device),graph_index=graph_index,
                group_ids=list(range(num_graphs)),mask=None,domain_ids=None)
            segment_plan=prepare_batch(temporary).pooling('node')
        buckets,order,graph_order=segment_plan
        logits=self.score(h).squeeze(-1)
        attention=torch.empty_like(logits);pooled=h.new_zeros((num_graphs,h.shape[1]))
        if not h.shape[0]:return pooled+(h.sum()+logits.sum())*0,attention
        # Equal-length segments use native batched softmax, preserving its
        # reduction/derivative path more closely than a hand-built exp/sum.
        for length,count,start,end,gstart,gend in buckets:
            selected=order[start:end]
            weights=torch.softmax(logits[selected].view(count,length),dim=1)
            values=(h[selected].view(count,length,-1)*weights[:,:,None]).sum(1)
            attention.index_copy_(0,selected,weights.reshape(-1))
            pooled.index_copy_(0,graph_order[gstart:gend],values)
        return pooled,attention

