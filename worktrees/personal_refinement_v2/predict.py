"""Load selected revised readouts; preserve raw-unit outputs and missing-only mask."""
import numpy as np
from common import OUTPUT, CACHE, read, sha


class Predictor:
    def __init__(self, job_id):
        folder=OUTPUT/job_id
        self.recipe=read(folder/'recipe.json')
        selected=read(folder/'selection_frozen.json')
        assert sha(folder/'best.npz')==selected['best_checkpoint_sha256']
        self.normalization=self.recipe['normalization']
        self.tasks=read(CACHE/'tasks.json');self.structures=read(CACHE/'structures.json')
        with np.load(folder/'best.npz',allow_pickle=False) as state:
            self.parameters={name:{key:state[name+'___p__'+key].copy()
                for key in ['direct_w','direct_b','hidden_w','hidden_b','output_w','output_b']}
                for name in self.normalization}

    def __call__(self, topology_id, device, conditions, embedding=None):
        """Inputs use the original pinned model-ready units/column and device order.

        device: [N,D,90], conditions: [N,46], optional frozen-parent embedding [N,128].
        Returns raw-unit [N,16] predictions and a trained-slot mask. This mask is
        output availability, NOT a replacement for a sample's observed-label mask.
        """
        n=self.normalization[topology_id];p=self.parameters[topology_id]
        x=np.concatenate([device[:,:,:19].reshape(len(device),-1),conditions],axis=1)
        mode=n['feature_mode']
        if mode!='physical':
            assert embedding is not None and embedding.shape==(len(device),128)
            x=np.concatenate([x,embedding],axis=1) if mode=='joint' else embedding
        x=(x[:,n['active_columns']].astype(np.float64)-n['input_center'])/n['input_std']
        # Training stores standardized inputs as float32 before FP64 matrix ops.
        x=x.astype(np.float32)
        h=x@p['hidden_w']+p['hidden_b'];a=h*np.exp(-np.logaddexp(0.,-h))
        pred=x@p['direct_w']+p['direct_b']+a@p['output_w']+p['output_b']
        slots=n['slots'];z=np.broadcast_to(n['target_center'],(len(x),16)).copy()
        z[:,slots]+=pred*np.array(n['target_std'])[slots]
        z=np.clip(z,n['training_target_min'],n['training_target_max'])
        for j,kind in enumerate(n['output_transform']):
            if kind=='symlog':z[:,j]=np.sign(z[:,j])*np.expm1(abs(z[:,j]))
        rules=self.tasks[self.structures[topology_id]['group']]['slots']
        scales=np.array([rules.get(str(j),{}).get('scale',1.) for j in range(16)])
        available=np.zeros_like(z,dtype=bool);available[:,slots]=True
        return z*scales,available
