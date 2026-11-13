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

    def forward(self,embedding,groups):
        shared=self.shared(embedding);features={f:block(shared) for f,block in self.families.items()};rows=[]
        for i,group in enumerate(groups):
            if group not in self.readouts:raise ValueError('Unknown/untrained supervision head: '+group)
            row=embedding.new_zeros(16)
            for family,slots in self.slots[group].items():
                values=self.readouts[group][family](features[family][i])
                row=row.index_copy(0,torch.tensor(slots,device=row.device),values)
            rows.append(row)
        return torch.stack(rows)

