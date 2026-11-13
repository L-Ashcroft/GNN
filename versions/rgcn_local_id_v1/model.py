"""Frozen edge-RGCN + train-only local decoder, with sparse-contract fallback.

The decoder is nonparametric and limited to known contracts/canonical topologies.
It does not claim unseen-topology zero-shot behavior.
"""
from pathlib import Path
from collections import OrderedDict
import json,hashlib,sys
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
class LocalDecoder:
    def __init__(self,bundle,root=ROOT,capacity=8):self.bundle=bundle;self.root=Path(root);self.banks=OrderedDict();self.capacity=capacity
    def bank(self,group):
        if group not in self.bundle['contracts']:raise ValueError('Unknown/untrained contract')
        if group in self.banks:self.banks.move_to_end(group);return self.banks[group]
        item=self.bundle['contracts'][group];path=self.root/item['index']
        if sha(path)!=item['index_sha256']:raise ValueError('Training-neighbor index changed')
        with np.load(path,allow_pickle=False) as stored:
            keys=['embedding_mean','embedding_std','embedding_active','train_embedding','device_mean','device_std','device_active','raw_columns','train_device','train_z','train_mask','scales','bounds','training_center']
            bank={k:stored[k] for k in keys}
        if len(self.banks)>=self.capacity:self.banks.popitem(last=False)
        self.banks[group]=bank;return bank
    def query(self,group,embedding,raw):
        b=self.bank(group);active=b['embedding_active'];e=(np.asarray(embedding,dtype='float64')[:,active]-b['embedding_mean'][active])/b['embedding_std'][active]
        active=b['device_active'];x=(np.asarray(raw,dtype='float64')[:,active]-b['device_mean'][active])/b['device_std'][active]
        train_e=b['train_embedding'].astype('float64');train_x=b['train_device'].astype('float64')
        e=e/max(1,np.sqrt(e.shape[1]));x=x/max(1,np.sqrt(x.shape[1]));train_e=train_e/max(1,np.sqrt(train_e.shape[1]));train_x=train_x/max(1,np.sqrt(train_x.shape[1]))
        outputs=[]
        for a,q in [(train_x,x),(np.column_stack([train_e,train_x]),np.column_stack([e,x]))]:
            norm=np.square(a).sum(1);rows=[];k=min(self.bundle['neighbors'],len(a))
            for start in range(0,len(q),64):
                v=q[start:start+64];dist=np.maximum(0.,np.square(v).sum(1)[:,None]+norm[None,:]-2*v@a.T)
                ii=np.argpartition(dist,k-1,axis=1)[:,:k];d=np.take_along_axis(dist,ii,axis=1);w=(1/(d+self.bundle['distance_epsilon']))[:,:,None]*b['train_mask'][ii];den=w.sum(1)
                pred=np.divide((w*b['train_z'][ii]).sum(1),den,out=np.broadcast_to(b['training_center'],(len(v),16)).copy(),where=den>0);rows.append(pred)
            outputs.append(np.concatenate(rows))
        return outputs
    def raw_from_batch(self,batch,group,graph_number):
        spec=self.bundle['contracts'][group];bank=self.bank(group);topology=self.bundle['topologies'][spec['topology']]
        static=topology['structure'];edges=torch.nonzero(batch.graph_index[batch.edge_index[0]]==graph_number).flatten()
        if len(edges)!=len(static['device_index']):raise ValueError('Canonical device graph differs from trained decoder')
        if batch.edge_type[edges].cpu().tolist()!=static['edge_type']:raise ValueError('Port relation ordering differs from canonical contract graph')
        nodes=torch.nonzero(batch.graph_index==graph_number).flatten();offset=int(nodes[0])
        if (batch.edge_index[:,edges]-offset).cpu().tolist()!=static['edge_index']:raise ValueError('Circuit connections differ from canonical contract')
        expected=torch.tensor(static['x'],dtype=batch.x.dtype,device=batch.x.device)
        if not torch.equal(batch.x[nodes],expected):raise ValueError('Net roles/structural features differ from canonical graph')
        first=topology['first_device_edges'];features=batch.edge_attr[edges].detach().cpu().numpy();f=len(self.bundle['device_features']);values=[]
        for kind,flat in bank['raw_columns']:
            device,field=divmod(int(flat),f);offset={'values':0,'known':f,'zero':2*f}[str(kind)]
            values.append(features[first[device],offset+field])
        return np.asarray(values,dtype='float64')

class LocalRGCN(nn.Module):
    def __init__(self,encoder,bundle):super().__init__();self.encoder=encoder;self.decoder=LocalDecoder(bundle);self.bundle=bundle
    def forward(self,batch,return_states=False):
        state=self.encoder(batch,return_states=True);embedding=state['embedding'].detach().cpu().numpy();prior=state['prediction'].detach().cpu().to(torch.float64).numpy()
        result=prior.copy();groups={}
        for i,g in enumerate(batch.group_ids):
            if g not in self.bundle['contracts']:raise ValueError('Unknown semantic contract; no zero-shot fallback')
            groups.setdefault(g,[]).append(i)
        for g,indices in groups.items():
            info=self.bundle['contracts'][g]
            if info['training_rows']<self.bundle['minimum_local_training_rows']:continue
            raw=np.stack([self.decoder.raw_from_batch(batch,g,i) for i in indices]);device,combined=self.decoder.query(g,embedding[indices],raw)
            alpha=self.bundle['graph_local_weight'];result[indices]=(1-alpha)*device+alpha*combined
        prediction=torch.from_numpy(result).to(batch.x.device);state['prediction']=prediction
        return state if return_states else prediction

def load_model(bundle_path=None,device='cpu'):
    path=Path(bundle_path) if bundle_path else ROOT/'artifacts/experiment1_rgcn_local_head_v1/bundle.json'
    bundle=json.loads(path.read_text(encoding='utf-8'));sys.path.insert(0,str(ROOT/'versions/rgcn_calibrated_head_v1'))
    import src.common as common
    common.ROOT=ROOT
    from src.model.accuracy_network import AccuracyRGCN
    parent_path=ROOT/bundle['graph_checkpoint']
    if sha(parent_path)!=bundle['graph_checkpoint_sha256']:raise ValueError('Parent GNN weights changed')
    parent=torch.load(parent_path,map_location='cpu',weights_only=True)
    keys=['hidden_dim','num_layers','dropout','update_edges','use_domain_emb','use_film','collapse_relations','contract_residual','calibrated_targets','adapter_hidden','adapter_min_train_samples']
    encoder=AccuracyRGCN(parent['schema'],parent['tasks'],parent['target_calibration'],**{k:parent['config'][k] for k in keys});encoder.load_state_dict(parent['model'],strict=True)
    model=LocalRGCN(encoder,bundle);model.to(device);model.eval();return model,bundle
