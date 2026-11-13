"""Physical families with independent semantic-contract readouts."""
import torch
from torch import nn

class SemanticGroupedHead(nn.Module):
    """Share family feature extractors, never unapproved final target readouts."""
    def __init__(self,hidden_dim,tasks):
        super().__init__();self.tasks=tasks
        self.shared=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ReLU())
        families=sorted({r['family'] for task in tasks.values() for r in task['slots'].values()})
        self.families=nn.ModuleDict({f:nn.Sequential(nn.Linear(hidden_dim,hidden_dim//2),nn.ReLU()) for f in families})
        self.readouts=nn.ModuleDict();self.slots={}
        for group,task in tasks.items():
            family_slots={f:sorted(int(i) for i,r in task['slots'].items() if r['family']==f) for f in families}
            self.slots[group]={f:indices for f,indices in family_slots.items() if indices}
            self.readouts[group]=nn.ModuleDict({f:nn.Linear(hidden_dim//2,len(indices)) for f,indices in self.slots[group].items()})
        packed=[];self._slot_slices={}
        for group,family_slots in self.slots.items():
            for family,indices in family_slots.items():
                start=len(packed);packed.extend(indices);self._slot_slices[group,family]=(start,len(packed))
        self.register_buffer('_slot_indices',torch.tensor(packed,dtype=torch.long),persistent=False)

    def forward(self,embedding,groups,group_plan=None):
        shared=self.shared(embedding);features={f:block(shared) for f,block in self.families.items()}
        if group_plan is None:
            grouped={}
            for i,group in enumerate(groups):grouped.setdefault(group,[]).append(i)
            order=[];slices=[]
            for group,rows in sorted(grouped.items()):
                start=len(order);order.extend(rows);slices.append((group,start,len(order)))
            group_plan=(slices,torch.tensor(order,device=embedding.device))
        slices,order=group_plan;prediction=embedding.new_zeros(16*len(groups))
        for group,start,end in slices:
            if group not in self.readouts:raise ValueError('Unknown/untrained supervision head: '+group)
            rows=order[start:end]
            for family,slots in self.slots[group].items():
                values=self.readouts[group][family](features[family].index_select(0,rows))
                begin,finish=self._slot_slices[group,family];indices=self._slot_indices[begin:finish]
                locations=rows[:,None]*16+indices[None,:]
                prediction.index_copy_(0,locations.reshape(-1),values.reshape(-1))
        return prediction.view(len(groups),16)

