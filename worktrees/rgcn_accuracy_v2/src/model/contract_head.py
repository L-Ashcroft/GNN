"""Contract-local nonlinear residuals and reversible train-only target affine maps.

Final semantic readouts remain separate. No new source, topology or label value
is fed into the graph backbone. Group identity is the existing routing contract.
"""
import torch
from torch import nn
from .heads import SemanticGroupedHead


class ContractResidualHead(nn.Module):
    def __init__(self,hidden_dim,tasks,calibration,base=None,contract_residual=True,calibrated_targets=True,
                 adapter_hidden=32,adapter_min_train_samples=256):
        super().__init__()
        if adapter_hidden<1 or adapter_min_train_samples<0:raise ValueError('Invalid adapter configuration')
        if set(tasks)!=set(calibration['groups']):raise ValueError('Calibration contracts mismatch')
        self.base=base if base is not None else SemanticGroupedHead(hidden_dim,tasks)
        self.tasks=tasks;self.calibrated_targets=calibrated_targets
        self.groups=sorted(tasks);self.group_index={g:i for i,g in enumerate(self.groups)}
        self.slots={g:tuple(sorted(map(int,t['slots']))) for g,t in tasks.items()}
        self.trained_slots={g:frozenset(int(s) for s,v in calibration['groups'][g]['slots'].items() if v['count']>0) for g in tasks}
        center=torch.zeros(len(tasks),16);std=torch.ones(len(tasks),16)
        self.adapters=nn.ModuleDict()
        for group in self.groups:
            record=calibration['groups'][group]
            if set(record['slots'])!=set(tasks[group]['slots']):raise ValueError('Calibration slot mismatch')
            for slot,rule in tasks[group]['slots'].items():
                item=record['slots'][slot]
                if item['semantic_id']!=rule['semantic_id'] or item['engineering_scale']!=rule['scale']:
                    raise ValueError('Calibration semantics mismatch')
                if item['std']<=0:raise ValueError('Invalid calibration scale')
                if calibrated_targets:
                    center[self.group_index[group],int(slot)]=item['center'];std[self.group_index[group],int(slot)]=item['std']
            if contract_residual and record['training_rows']>=adapter_min_train_samples:
                block=nn.Sequential(nn.Linear(hidden_dim,adapter_hidden),nn.SiLU(),nn.Linear(adapter_hidden,len(self.slots[group])))
                # Preserve the shared head's initial function: new branch starts
                # at zero and learns contract-specific corrections.
                nn.init.zeros_(block[-1].weight);nn.init.zeros_(block[-1].bias)
                self.adapters[group]=block
        self.register_buffer('target_center',center);self.register_buffer('target_std',std)

    def affine(self,groups):
        try:indices=[self.group_index[g] for g in groups]
        except KeyError as error:raise ValueError('Unknown/untrained supervision head: '+str(error)) from error
        index=torch.tensor(indices,device=self.target_center.device,dtype=torch.long)
        return self.target_center[index],self.target_std[index]

    def check_observed(self,coverage):
        for group,slots in coverage.items():
            if group not in self.trained_slots or not set(slots)<=self.trained_slots[group]:
                raise ValueError('Observed target has no training calibration: '+group)

    def forward(self,embedding,groups,group_plan=None):
        prediction=self.base(embedding,groups)
        if group_plan is None:
            ordered={}
            for i,group in enumerate(groups):ordered.setdefault(group,[]).append(i)
            blocks=[(g,torch.tensor(rows,device=embedding.device)) for g,rows in sorted(ordered.items())]
        else:
            slices,order=group_plan;blocks=[(g,order[start:end]) for g,start,end in slices]
        correction=torch.zeros_like(prediction)
        for group,indices in blocks:
            if group not in self.adapters:continue
            values=self.adapters[group](embedding[indices])
            columns=torch.tensor(self.slots[group],device=embedding.device)
            padded=embedding.new_zeros(len(indices),16).index_copy(1,columns,values)
            correction=correction.index_copy(0,indices,padded)
        normalized=prediction+correction
        center,std=self.affine(groups)
        return center+std*normalized
