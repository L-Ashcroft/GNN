"""Explicit module freezing and stage-specific learning rates."""
import torch


def build_optimizer(model,stage='pretrain',lr_head=2e-4,lr_backbone=2e-5,weight_decay=1e-2):
    if stage not in {'pretrain','finetune_head','finetune_all'}:raise ValueError('Unknown training stage')
    backbone=[model.node_encoder,model.edge_encoder,model.relation_encoder,model.layers]
    backbone_ids={id(p) for module in backbone for p in module.parameters()}
    for p in model.parameters():p.requires_grad_(stage!='finetune_head' or id(p) not in backbone_ids)
    if stage=='finetune_all':
        groups=[{'params':[p for p in model.parameters() if id(p) in backbone_ids],'lr':lr_backbone,'name':'backbone'},
                {'params':[p for p in model.parameters() if id(p) not in backbone_ids],'lr':lr_head,'name':'adaptation'}]
    else:groups=[{'params':[p for p in model.parameters() if p.requires_grad],'lr':lr_head,'name':stage}]
    optimizer=torch.optim.AdamW(groups,weight_decay=weight_decay)
    selected=[id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(selected)==len(set(selected))
    assert set(selected)=={id(p) for p in model.parameters() if p.requires_grad}
    return optimizer


def parameter_counts(model):
    return {'total':sum(p.numel() for p in model.parameters()),
            'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)}
