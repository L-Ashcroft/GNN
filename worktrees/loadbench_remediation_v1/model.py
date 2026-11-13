"""Independent calibrated readout for frozen GNN embeddings and SI parameters.

No change to the original graph encoder, checkpoint, masks, splits or contracts.
Inputs include the published model prediction so other sources remain identical.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,hashlib
import numpy as np
from readout import predict

ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class CorrectedReadout:
    def __init__(self,path=None):
        path=Path(path) if path else ROOT/'artifacts/loadbench_remediation_v1/bundle.json'
        self.bundle=json.loads(path.read_text(encoding='utf-8'))
        original=ROOT/self.bundle['original_bundle']
        if sha(original)!=self.bundle['original_bundle_sha256']:raise ValueError('Original model changed')
        self.original=json.loads(original.read_text(encoding='utf-8'))

    def predict(self,group,parameters,embedding,published_prediction,*,parameter_names):
        if group not in self.original['contracts']:raise ValueError('Unknown contract; no zero-shot fallback')
        base=np.asarray(published_prediction,dtype=np.float64).copy()
        if group not in self.bundle['groups']:return base
        item=self.bundle['groups'][group]
        if list(parameter_names)!=item['parameter_names']:raise ValueError('Parameter order differs from canonical topology')
        if item['effective_weight']==0:return base
        index=ROOT/item['index'];old=ROOT/self.original['contracts'][group]['index']
        if sha(index)!=item['index_sha256'] or sha(old)!=self.original['contracts'][group]['index_sha256']:raise ValueError('Training index changed')
        with np.load(index,allow_pickle=False) as f:d={k:f[k] for k in f.files}
        with np.load(old,allow_pickle=False) as f:b={k:f[k] for k in f.files}
        values=np.asarray(parameters,dtype=np.float64)
        if values.ndim!=2 or values.shape[1]!=len(item['parameter_names']) or not np.isfinite(values).all():raise ValueError('Invalid design parameters')
        units=np.asarray(item['unit_scales']);values=np.sign(values)*np.log1p(abs(values)/units)
        active=d['active'];q=(values[:,active]-d['center'][active])/d['std'][active]
        e=b['train_embedding'].astype(np.float64);active=b['embedding_active']
        ev=(np.asarray(embedding,dtype=np.float64)[:,active]-b['embedding_mean'][active])/b['embedding_std'][active]
        x=d['train_parameters']
        train=np.column_stack([x/max(1,np.sqrt(x.shape[1])),e/max(1,np.sqrt(e.shape[1]))])
        query=np.column_stack([q/max(1,np.sqrt(q.shape[1])),ev/max(1,np.sqrt(ev.shape[1]))])
        pred,counts=predict(train,query,b['train_z'],b['train_mask'])
        pred[:,counts==0]=base[:,counts==0]
        return base+item['effective_weight']*(pred-base)
