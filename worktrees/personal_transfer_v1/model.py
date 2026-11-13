"""Conditioned personal RGCN; semantic name-based backbone transfer, fresh heads."""
import torch
from torch import nn
from common import ROOT, read
from src.model.network import EdgeCentricRGCN
from src.model.execution import prepare_batch


class TargetHead(nn.Module):
    def __init__(self, tasks, stats):
        super().__init__()
        self.groups = sorted(tasks); self.lookup = {g: i for i, g in enumerate(self.groups)}
        self.slots = {g: sorted(map(int, tasks[g]['slots'])) for g in self.groups}
        self.direct = nn.ModuleDict(); self.residual = nn.ModuleDict()
        for group in self.groups:
            n = len(self.slots[group])
            self.direct[group] = nn.Linear(128, n)
            self.residual[group] = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, n))
            for layer in [self.direct[group], self.residual[group][-1]]:
                nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
        self.register_buffer('center', torch.tensor([stats['groups'][g]['center'] for g in self.groups], dtype=torch.float64))
        self.register_buffer('std', torch.tensor([stats['groups'][g]['std'] for g in self.groups], dtype=torch.float64))

    def affine(self, groups):
        index = torch.tensor([self.lookup[g] for g in groups], device=self.center.device)
        return self.center[index], self.std[index]

    def forward(self, graph, groups, group_plan=None):
        if group_plan is None:
            mapping = {}
            for i, g in enumerate(groups):
                mapping.setdefault(g, []).append(i)
            blocks = [(g, torch.tensor(rows)) for g, rows in mapping.items()]
        else:
            slices, order = group_plan
            blocks = [(g, order[a:b]) for g, a, b in slices]
        output = graph.new_zeros(len(groups), 16)
        for g, indices in blocks:
            values = self.direct[g](graph[indices]) + self.residual[g](graph[indices])
            columns = torch.tensor(self.slots[g], device=graph.device)
            padded = graph.new_zeros(len(indices), 16).index_copy(1, columns, values)
            output = output.index_copy(0, indices, padded)
        center, std = self.affine(groups)
        return center + std * output.to(torch.float64)


class TargetRGCN(EdgeCentricRGCN):
    def __init__(self, schema, tasks, stats):
        super().__init__(schema, tasks, hidden_dim=128, num_layers=4, dropout=.1,
                         use_domain_emb=True, use_film=True, update_edges=True)
        self.head = TargetHead(tasks, stats)
        self.condition_encoder = nn.Sequential(nn.Linear(46, 128), nn.ReLU(), nn.Linear(128, 128))
        nn.init.zeros_(self.condition_encoder[-1].weight); nn.init.zeros_(self.condition_encoder[-1].bias)

    def forward(self, batch, return_states=False):
        plan = prepare_batch(batch)
        h = torch.relu(self.node_encoder(batch.x))
        z = torch.relu(self.edge_encoder(batch.edge_attr) + self.relation_encoder(batch.edge_type))
        domain = self.domain_emb(batch.domain_ids)
        for i, layer in enumerate(self.layers):
            gamma, beta = self.films[i](domain)
            h, z = layer(h, z, batch.edge_index, batch.edge_type, gamma[batch.graph_index], beta[batch.graph_index],
                         relation_plan=plan.relations())
        nodes, node_attention = self.pool(h, batch.graph_index, batch.num_graphs, segment_plan=plan.pooling('node'))
        edges, edge_attention = self.edge_pool(z, batch.graph_index[batch.edge_index[0]], batch.num_graphs,
                                               segment_plan=plan.pooling('edge'))
        embedding = self.pool_fusion(torch.cat([nodes, edges], -1)) + domain + self.condition_encoder(batch.conditions)
        prediction = self.head(embedding, batch.group_ids, group_plan=plan.grouping())
        if return_states:
            return {'prediction': prediction, 'node_states': h, 'edge_states': z, 'embedding': embedding,
                    'node_attention': node_attention, 'edge_attention': edge_attention}
        return prediction


def transfer(model, checkpoint):
    source = checkpoint['model']; old = read(ROOT / 'data/model_ready/fusion_v2_training_v1/schema.json')
    new = model.schema; target = model.state_dict(); copied = []
    assert old['device_features'] == new['device_features']
    assert set(old['node_features']) <= set(new['node_features'])
    assert set(old['relations']) <= set(new['relations'])
    for key in target:
        if not key.startswith(('node_encoder.', 'edge_encoder.', 'relation_encoder.', 'layers.',
                               'pool.', 'edge_pool.', 'pool_fusion.')):
            continue
        value = source[key]
        if key == 'node_encoder.weight':
            target[key].zero_()
            for i, name in enumerate(old['node_features']):
                target[key][:, new['node_features'].index(name)] = value[:, i]
        elif key == 'relation_encoder.weight' or (key.startswith('layers.') and key.endswith('relation_weight')):
            assert value.shape[0] == len(old['relations']) and value.shape[1:] == target[key].shape[1:]
            for i, name in enumerate(old['relations']):
                target[key][new['relations'].index(name)] = value[i]
        else:
            assert value.shape == target[key].shape, key
            target[key] = value
        copied.append(key)
    model.load_state_dict(target, strict=True)
    return {'copied_tensors': copied, 'source_relation_count': len(old['relations']),
            'new_relations': sorted(set(new['relations']) - set(old['relations'])),
            'new_node_features': sorted(set(new['node_features']) - set(old['node_features'])),
            'target_heads_domain_FiLM_and_conditions_transferred': False,
            'source_input_transform': 'ID raw deterministic engineering transforms; no LOTO fitted input stats'}


def loss(prediction, batch, head):
    y = batch.targets / batch.scales
    target = y.sign() * torch.log1p(y.abs())
    _, std = head.affine(batch.group_ids)
    error = ((prediction - target) / std).square()
    assert torch.isfinite(error[batch.mask]).all(), 'Nonfinite observed loss'
    plan = prepare_batch(batch); result = {}
    for group, a, b in plan.loss_slices:
        positions = plan.tensors['loss_order'][a:b]
        result[group] = error.reshape(-1)[positions].mean()
    return torch.stack(list(result.values())).mean(), result
