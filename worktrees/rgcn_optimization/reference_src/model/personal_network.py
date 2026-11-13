"""Edge-centred RGCN with explicit, label-independent fixture/bias conditioning."""
import torch
from torch import nn
from .network import EdgeCentricRGCN


class PersonalEdgeRGCN(EdgeCentricRGCN):
    def __init__(self,schema,tasks,**kwargs):
        super().__init__(schema,tasks,**kwargs)
        d=self.node_encoder.out_features
        self.condition_encoder=nn.Sequential(nn.Linear(schema['condition_input_dim'],d),nn.ReLU(),nn.Linear(d,d))

    def forward(self,batch,return_states=False):
        if batch.conditions is None or batch.conditions.shape!=(batch.num_graphs,self.schema['condition_input_dim']):
            raise ValueError('Explicit fixture/bias condition vector required')
        edge_type=torch.zeros_like(batch.edge_type) if self.collapse_relations else batch.edge_type
        h=torch.relu(self.node_encoder(batch.x));z=torch.relu(self.edge_encoder(batch.edge_attr)+self.relation_encoder(edge_type))
        initial=z;domain=self.domain_emb(batch.domain_ids) if self.use_domain_emb else None
        for i,layer in enumerate(self.layers):
            gamma,beta=self.films[i](domain) if self.use_film else (None,None)
            h,z=layer(h,z,batch.edge_index,edge_type,
                gamma[batch.graph_index] if gamma is not None else None,
                beta[batch.graph_index] if beta is not None else None)
        nodes,attention=self.pool(h,batch.graph_index,batch.num_graphs)
        edges,edge_attention=self.edge_pool(z,batch.graph_index[batch.edge_index[0]],batch.num_graphs)
        graph=self.pool_fusion(torch.cat([nodes,edges],dim=-1))
        if domain is not None:graph=graph+domain
        embedding=graph+self.condition_encoder(batch.conditions)
        prediction=self.head(embedding,batch.group_ids)
        if return_states:return {'prediction':prediction,'node_states':h,'edge_states':z,'initial_edge_states':initial,
            'embedding':embedding,'node_attention':attention,'edge_attention':edge_attention}
        return prediction


def initialize_backbone(model,checkpoint):
    """Explicit compatible-weight transfer, without target-equivalence claims."""
    old_schema=checkpoint['schema'];new_schema=model.schema
    if old_schema['device_features']!=new_schema['device_features']:raise ValueError('Device feature semantics differ')
    for key in ['node_features','relations']:
        if not set(old_schema[key])<=set(new_schema[key]):raise ValueError('Pretrained schema not contained: '+key)
    source=checkpoint['model'];target=model.state_dict();copied=[]
    if source['relation_encoder.weight'].shape[0]!=len(old_schema['relations']):
        raise ValueError('Collapsed-relation checkpoints need a separate transfer policy')
    for key,value in source.items():
        if key.startswith(('node_encoder.','edge_encoder.','relation_encoder.','layers.','pool.','edge_pool.','pool_fusion.','head.shared.','head.families.')):
            if key=='node_encoder.weight' and value.shape[0]==target[key].shape[0]:
                target[key].zero_()
                for old_index,name in enumerate(old_schema['node_features']):
                    target[key][:,new_schema['node_features'].index(name)]=value[:,old_index]
                copied.append(key)
            elif key=='relation_encoder.weight' or (key.startswith('layers.') and key.endswith('.relation_weight')):
                if value.shape[1:]!=target[key].shape[1:]:raise ValueError('Hidden dimensions differ')
                for old_index,name in enumerate(old_schema['relations']):
                    target[key][new_schema['relations'].index(name)]=value[old_index]
                copied.append(key)
            elif key in target and value.shape==target[key].shape:target[key]=value;copied.append(key)
    required=[key for key in target if key.startswith(('node_encoder.','edge_encoder.','relation_encoder.','layers.'))]
    if not set(required)<=set(copied):raise ValueError('Pretrained backbone incomplete or dimensions differ')
    model.load_state_dict(target)
    return {'copied_tensors':copied,'target_readouts_transferred':False,'domain_film_or_conditions_transferred':False,
        'added_relations':sorted(set(new_schema['relations'])-set(old_schema['relations'])),
        'added_node_features':sorted(set(new_schema['node_features'])-set(old_schema['node_features']))}
