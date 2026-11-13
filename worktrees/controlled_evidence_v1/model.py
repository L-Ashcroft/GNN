"""Fair LOADBench controls: portable heads and shared observed conditions.

GCN uses D^-1/2(A+I)D^-1/2 H W; edge parameters enter at input and readout,
never relation-specific propagation. Residual/LN/FiLM scaffolding is shared.
MLP uses ordered unique devices, all90 fields + present flags; no adjacency.
"""
import torch
from torch import nn
from portable import FoldRGCN,PortableHead,loss_by_contract,weighted_loss
from src.model.layers import AttentionPool
from src.model.film import DomainEmbedding,FiLM
from src.model.execution import prepare_batch

def normalized_edges(batch,center,std):
    f=center.shape[1];domain=batch.domain_ids[batch.graph_index[batch.edge_index[0]]]
    values=batch.edge_attr[:,:f];known=batch.edge_attr[:,f:2*f].bool();zero=batch.edge_attr[:,2*f:].bool()
    use=known&~zero&(torch.arange(f,device=values.device)[None,:]<19)
    return torch.cat([torch.where(use,(values-center[domain])/std[domain],values),batch.edge_attr[:,f:]],1)

class ControlledRGCN(FoldRGCN):
    def __init__(self,schema,tasks,stats,**kwargs):
        super().__init__(schema,tasks,stats,**kwargs)
        self.condition_adapter=nn.Linear(8,kwargs['hidden_dim'])
    def forward(self,batch,return_states=False):
        plan=prepare_batch(batch);edge_type=batch.edge_type
        h=torch.relu(self.node_encoder(batch.x));z=torch.relu(self.edge_encoder(normalized_edges(batch,self.input_center,self.input_std))+self.relation_encoder(edge_type))
        initial=z;domain=self.domain_emb(batch.domain_ids)
        for i,layer in enumerate(self.layers):
            gamma,beta=self.films[i](domain)
            h,z=layer(h,z,batch.edge_index,edge_type,gamma[batch.graph_index],beta[batch.graph_index],relation_plan=plan.relations(False))
        p,a=self.pool(h,batch.graph_index,batch.num_graphs,segment_plan=plan.pooling('node'))
        ep,ea=self.edge_pool(z,batch.graph_index[batch.edge_index[0]],batch.num_graphs,segment_plan=plan.pooling('edge'))
        embedding=self.pool_fusion(torch.cat([p,ep],1))+domain+self.condition_adapter(batch.conditions)
        prediction=self.head(embedding,batch.group_ids,plan.grouping())
        if return_states:return dict(prediction=prediction,embedding=embedding,node_states=h,edge_states=z,initial_edge_states=initial,node_attention=a,edge_attention=ea)
        return prediction

def gcn_propagate(h,edge_index):
    """Weighted multigraph: preserve original loops/parallel edges, add one I."""
    n=len(h);loops=torch.arange(n,device=h.device)
    src=torch.cat([edge_index[0],loops]);dst=torch.cat([edge_index[1],loops])
    degree=torch.bincount(dst,minlength=n).to(h.dtype)
    weight=(degree[src]*degree[dst]).rsqrt()
    out=torch.zeros_like(h);out.index_add_(0,dst,h[src]*weight[:,None]);return out

class DenseBlock(nn.Module):
    def __init__(self,h,dropout):
        super().__init__();self.linear=nn.Linear(h,h);self.norm=nn.LayerNorm(h);self.dropout=nn.Dropout(dropout)
    def forward(self,x,edges=None):
        update=self.linear(gcn_propagate(x,edges) if edges is not None else x)
        return self.norm(x+self.dropout(torch.relu(update)))

class SimpleModel(nn.Module):
    def __init__(self,schema,tasks,stats,cfg,parameter_dim):
        super().__init__();h=cfg['hidden_dim'];self.schema=schema;self.kind=cfg['model_kind']
        self.register_buffer('input_center',torch.tensor([r['center'] for r in stats['inputs']],dtype=torch.float32))
        self.register_buffer('input_std',torch.tensor([r['std'] for r in stats['inputs']],dtype=torch.float32))
        self.node_encoder=nn.Linear(parameter_dim if self.kind=='mlp' else len(schema['node_features']),h)
        self.edge_encoder=nn.Identity() if self.kind=='mlp' else nn.Linear(3*len(schema['device_features']),h)
        self.relation_encoder=nn.Identity() # no relation lookup or separate weights
        self.layers=nn.ModuleList([DenseBlock(h,cfg['dropout']) for _ in range(cfg['num_layers'])])
        self.domain_emb=DomainEmbedding(len(schema['domains']),h)
        self.films=nn.ModuleList([FiLM(h,h) for _ in self.layers])
        self.condition_adapter=nn.Linear(8,h)
        if self.kind=='gcn':
            self.pool=AttentionPool(h);self.edge_pool=AttentionPool(h)
            self.pool_fusion=nn.Sequential(nn.Linear(2*h,h),nn.ReLU(),nn.LayerNorm(h))
        self.head=PortableHead(h,tasks,stats)
    def forward(self,batch,return_states=False):
        plan=prepare_batch(batch);domain=self.domain_emb(batch.domain_ids)
        if self.kind=='mlp':
            f=self.input_center.shape[1];features=batch.parameter_input.reshape(len(domain),-1,3*f+1)
            v,k,z=features[:,:,:f],features[:,:,f:2*f].bool(),features[:,:,2*f:3*f].bool()
            use=k&~z&(torch.arange(f,device=v.device)[None,None,:]<19)
            v=torch.where(use,(v-self.input_center[batch.domain_ids,None,:])/self.input_std[batch.domain_ids,None,:],v)
            h=torch.relu(self.node_encoder(torch.cat([v,features[:,:,f:]],2).flatten(1)))
            for layer,film in zip(self.layers,self.films):
                gamma,beta=film(domain);h=gamma*layer(h)+beta
            embedding=h+domain+self.condition_adapter(batch.conditions);z=None
        else:
            z=torch.relu(self.edge_encoder(normalized_edges(batch,self.input_center,self.input_std)))
            h=torch.relu(self.node_encoder(batch.x));src,dst=batch.edge_index
            incident=torch.zeros_like(h);incident.index_add_(0,dst,z)
            degree=torch.bincount(dst,minlength=len(h)).clamp_min(1).to(h.dtype)
            h=h+incident/degree[:,None]
            for layer,film in zip(self.layers,self.films):
                gamma,beta=film(domain);h=gamma[batch.graph_index]*layer(h,batch.edge_index)+beta[batch.graph_index]
            pooled,_=self.pool(h,batch.graph_index,batch.num_graphs,segment_plan=plan.pooling('node'))
            ep,_=self.edge_pool(z,batch.graph_index[src],batch.num_graphs,segment_plan=plan.pooling('edge'))
            embedding=self.pool_fusion(torch.cat([pooled,ep],1))+domain+self.condition_adapter(batch.conditions)
        prediction=self.head(embedding,batch.group_ids,plan.grouping())
        if return_states:return dict(prediction=prediction,embedding=embedding,node_states=h,edge_states=z)
        return prediction

def make_model(reader,stats,cfg):
    if cfg['model_kind']=='rgcn':return ControlledRGCN(reader.schema,reader.tasks,stats,**{k:cfg[k] for k in ['hidden_dim','num_layers','dropout','update_edges','use_domain_emb','use_film','collapse_relations']})
    return SimpleModel(reader.schema,reader.tasks,stats,cfg,reader.parameter_dim)
