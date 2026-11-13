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
            from .execution import segment_plan as make_segments
            order,_,_,lengths=make_segments(graph_index,num_graphs)
            offsets=[0]+lengths.cumsum(0).tolist()
            segment_plan=(tuple((i,offsets[i],offsets[i+1]) for i in range(num_graphs)),order.to(h.device))
        segments,order=segment_plan
        pooled=[];attention=torch.empty(h.shape[0],device=h.device,dtype=h.dtype)
        logits=self.score(h).squeeze(-1)
        for i,start,end in segments:
            selected=order[start:end]
            weights=torch.softmax(logits[selected],dim=0)
            attention[selected]=weights
            pooled.append((h[selected]*weights[:,None]).sum(0))
        return torch.stack(pooled),attention

