"""Assemble the edge-centred relational network."""
import torch
from torch import nn
from .layers import EdgeRGCNLayer, AttentionPool
from .heads import SemanticGroupedHead
from .film import DomainEmbedding, FiLM
from .execution import prepare_batch

class EdgeCentricRGCN(nn.Module):
    def __init__(self,schema,tasks,hidden_dim=128,num_layers=4,dropout=0.1,update_edges=True,
                 use_domain_emb=False,use_film=False,collapse_relations=False):
        super().__init__();self.schema=schema;self.tasks=tasks
        if hidden_dim<4 or num_layers<1:raise ValueError('Invalid model dimensions')
        if use_film and not use_domain_emb:raise ValueError('FiLM requires domain embeddings; disable both for no-domain ablation')
        self.use_domain_emb=use_domain_emb;self.use_film=use_film;self.collapse_relations=collapse_relations
        self.node_encoder=nn.Linear(len(schema['node_features']),hidden_dim)
        self.edge_encoder=nn.Linear(3*len(schema['device_features']),hidden_dim)
        relations=1 if collapse_relations else len(schema['relations'])
        self.relation_encoder=nn.Embedding(relations,hidden_dim)
        self.layers=nn.ModuleList([EdgeRGCNLayer(hidden_dim,relations,dropout,update_edges) for _ in range(num_layers)])
        self.domain_emb=DomainEmbedding(len(schema['domains']),hidden_dim) if use_domain_emb else None
        self.films=nn.ModuleList([FiLM(hidden_dim,hidden_dim) for _ in range(num_layers)]) if use_film else nn.ModuleList()
        self.pool=AttentionPool(hidden_dim);self.edge_pool=AttentionPool(hidden_dim)
        self.pool_fusion=nn.Sequential(nn.Linear(2*hidden_dim,hidden_dim),nn.ReLU(),nn.LayerNorm(hidden_dim))
        self.head=SemanticGroupedHead(hidden_dim,tasks)

    def forward(self,batch,return_states=False):
        plan=prepare_batch(batch)
        relation_plan=plan.relations(self.collapse_relations)
        if relation_plan[0] and relation_plan[0][-1][0]>=self.relation_encoder.num_embeddings:
            raise ValueError('Relation id outside model vocabulary')
        edge_type=torch.zeros_like(batch.edge_type) if self.collapse_relations else batch.edge_type
        h=torch.relu(self.node_encoder(batch.x));z=torch.relu(self.edge_encoder(batch.edge_attr)+self.relation_encoder(edge_type))
        initial=z
        domain=self.domain_emb(batch.domain_ids) if self.use_domain_emb else None
        for i,layer in enumerate(self.layers):
            gamma,beta=self.films[i](domain) if self.use_film else (None,None)
            h,z=layer(h,z,batch.edge_index,edge_type,
                      gamma[batch.graph_index] if gamma is not None else None,
                      beta[batch.graph_index] if beta is not None else None,relation_plan=relation_plan)
        embedding,attention=self.pool(h,batch.graph_index,batch.num_graphs,segment_plan=plan.pooling('node'))
        edge_embedding,edge_attention=self.edge_pool(z,batch.graph_index[batch.edge_index[0]],batch.num_graphs,segment_plan=plan.pooling('edge'))
        embedding=self.pool_fusion(torch.cat([embedding,edge_embedding],dim=-1))
        if domain is not None:embedding=embedding+domain
        prediction=self.head(embedding,batch.group_ids,group_plan=plan.grouping())
        if return_states:return {'prediction':prediction,'node_states':h,'edge_states':z,'initial_edge_states':initial,
                                 'embedding':embedding,'node_attention':attention,'edge_attention':edge_attention}
        return prediction

