"""Independent, ridge-initialized residual readouts of frozen RGCN features."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn

class ContractReadout(nn.Module):
    def __init__(self,group,tasks,calibration,ridge,index,hidden=64,min_rows=256):
        super().__init__();self.slots=tuple(sorted(map(int,tasks[group]['slots'])))
        self.register_buffer('feature_center',torch.tensor(ridge['embedding_mean'],dtype=torch.float64))
        self.register_buffer('feature_std',torch.tensor(ridge['embedding_std'],dtype=torch.float64))
        active=torch.tensor(ridge['embedding_active'],dtype=torch.bool);self.register_buffer('feature_active',active)
        center=torch.tensor([calibration['groups'][group]['slots'][str(j)]['center'] for j in self.slots],dtype=torch.float64)
        std=torch.tensor([calibration['groups'][group]['slots'][str(j)]['std'] for j in self.slots],dtype=torch.float64)
        self.register_buffer('target_center',center);self.register_buffer('target_std',std)
        self.register_buffer('bounds',torch.tensor(index['bounds'][:,self.slots],dtype=torch.float64))
        self.direct=nn.Linear(int(active.sum()),len(self.slots))
        coefficient=np.asarray(ridge['coefficients'][:,self.slots],dtype=np.float64)
        with torch.no_grad():
            self.direct.weight.copy_(torch.tensor((coefficient[1:]/std.numpy()).T,dtype=torch.float32))
            self.direct.bias.copy_(torch.tensor((coefficient[0]-center.numpy())/std.numpy(),dtype=torch.float32))
        if calibration['groups'][group]['training_rows']>=min_rows:
            self.residual=nn.Sequential(nn.Linear(int(active.sum()),hidden),nn.SiLU(),nn.Dropout(.05),nn.Linear(hidden,len(self.slots)))
            nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
        else:self.residual=None

    def normalized(self,embedding):
        a=self.feature_active
        x=((embedding.to(torch.float64)[:,a]-self.feature_center[a])/self.feature_std[a]).to(torch.float32)
        value=self.direct(x)
        if self.residual is not None:value=value+self.residual(x)
        return value

    def forward(self,embedding):
        z=self.target_center+self.target_std*self.normalized(embedding).to(torch.float64)
        # Output safeguard, fitted exclusively on observed training labels.
        # It changes predictions, never target values or missingness.
        if not self.training:z=torch.maximum(torch.minimum(z,self.bounds[1]),self.bounds[0])
        output=z.new_zeros(len(z),16);output[:,self.slots]=z;return output

    def loss(self,embedding,target,mask,scales):
        columns=list(self.slots);observed=mask[:,columns]
        y=target[:,columns].to(torch.float64)/scales[:,columns].to(torch.float64)
        z=y.sign()*torch.log1p(y.abs());target_normalized=(z-self.target_center)/self.target_std
        prediction=self.normalized(embedding).to(torch.float64)
        p=prediction[observed];t=target_normalized[observed]
        if not len(p) or not torch.isfinite(p).all() or not torch.isfinite(t).all():raise ValueError('Invalid observed contract loss')
        return (p-t).square().mean()

class DirectResidualHead(nn.Module):
    def __init__(self,root,tasks,calibration,hidden=64,min_rows=256):
        super().__init__();root=Path(root);self.contracts=nn.ModuleDict()
        for group in sorted(tasks):
            with np.load(root/'artifacts/experiment1_representation_diagnostic_v1'/(group+'_embedding_linear.npz'),allow_pickle=False) as ridge,np.load(root/'artifacts/experiment1_local_readout_v1'/(group+'.npz'),allow_pickle=False) as index:
                self.contracts[group]=ContractReadout(group,tasks,calibration,ridge,index,hidden,min_rows)

    def forward(self,embedding,groups,group_plan=None):
        ordered={}
        for i,g in enumerate(groups):
            if g not in self.contracts:raise ValueError('Unknown/untrained semantic contract')
            ordered.setdefault(g,[]).append(i)
        output=torch.zeros(len(groups),16,dtype=torch.float64,device=embedding.device)
        for group,rows in ordered.items():
            idx=torch.tensor(rows,device=embedding.device);output=output.index_copy(0,idx,self.contracts[group](embedding[idx]))
        return output

    @classmethod
    def from_state(cls,tasks,state):
        """Inference constructor: no training neighbor index/labels are needed."""
        head=cls.__new__(cls);nn.Module.__init__(head);head.contracts=nn.ModuleDict()
        for group in sorted(tasks):
            prefix='contracts.'+group+'.';block=ContractReadout.__new__(ContractReadout);nn.Module.__init__(block)
            block.slots=tuple(sorted(map(int,tasks[group]['slots'])))
            for name in ['feature_center','feature_std','feature_active','target_center','target_std','bounds']:
                block.register_buffer(name,torch.zeros_like(state[prefix+name]))
            features=state[prefix+'direct.weight'].shape[1];block.direct=nn.Linear(features,len(block.slots))
            if prefix+'residual.0.weight' in state:
                hidden=state[prefix+'residual.0.weight'].shape[0]
                block.residual=nn.Sequential(nn.Linear(features,hidden),nn.SiLU(),nn.Dropout(.05),nn.Linear(hidden,len(block.slots)))
            else:block.residual=None
            head.contracts[group]=block
        head.load_state_dict(state,strict=True);return head
