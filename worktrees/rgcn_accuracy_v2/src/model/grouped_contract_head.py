"""Optional grouped execution of the same head parameters for cached features.

No shared semantic readouts: each contract still executes its own Linear.
Enable only after the GPU forward/gradient/AdamW gate in the stage runner.
"""
import torch


def forward_grouped(head,embedding,groups,group_plan):
    shared=head.base.shared(embedding)
    features={family:block(shared) for family,block in head.base.families.items()}
    slices,order=group_plan;output=embedding.new_zeros(len(groups),16)
    for group,start,end in slices:
        if group not in head.base.readouts:raise ValueError('Unknown supervision contract: '+group)
        indices=order[start:end];row=embedding.new_zeros(len(indices),16)
        for family,slots in head.base.slots[group].items():
            values=head.base.readouts[group][family](features[family][indices])
            columns=torch.tensor(slots,device=embedding.device)
            row=row.index_copy(1,columns,values)
        if group in head.adapters:
            values=head.adapters[group](embedding[indices]);columns=torch.tensor(head.slots[group],device=embedding.device)
            row=row.index_add(1,columns,values)
        output=output.index_copy(0,indices,row)
    center,std=head.affine(groups)
    return center+std*output
