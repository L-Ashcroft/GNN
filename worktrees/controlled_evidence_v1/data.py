"""Scoped TRAIN-only controls; stable ID roles, no warm embeddings or test fit."""
from pathlib import Path
from dataclasses import dataclass
import json, hashlib, math
import numpy as np
from raw_data import FoldData, ROOT, sha

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def epoch_indices(n, rows, seed, epoch):
    """Same number of presentations/updates; repeat training pool, never clone data."""
    rng=np.random.default_rng(seed+epoch)
    return np.concatenate([rng.permutation(n)[:min(n,rows-i)] for i in range(0,rows,n)])

def condition_vector(spec):
    c=spec.get('conditions',{})
    values=[spec.get('temperature'),spec.get('supply_voltage_default'),
            c.get('load_capacitance_F'),c.get('bias_current_A')]
    scales=[100.,1.,1e-12,1e-5]
    result=[];known=[]
    for v,scale in zip(values,scales):
        valid=isinstance(v,(float,int)) and math.isfinite(v)
        result.append(math.copysign(math.log1p(abs(v/scale)),v) if valid else 0.)
        known.append(float(valid))
    return result+known

class ScopedData(FoldData):
    def __init__(self,cfg,verify=True):
        super().__init__('paired_ID_same_neural_architecture_seed42',cfg['lru_topologies'],verify)
        self.cfg=cfg
        catalog={v['topology_id']:v for v in read(ROOT/'data/fusion_v2/catalog.json')}
        self.conditions={k:condition_vector(v) for k,v in catalog.items()}
        for role in self.refs:
            sources=cfg['train_sources'] if role=='train' else ['LOADBench']
            selected=np.array([self.topologies[int(i)].split('__')[0] in sources for i in self.refs[role][:,0]])
            self.refs[role]=self.refs[role][selected]
        self.fold=json.loads(json.dumps(self.fold));self.fold['fold_id']=cfg['id']
        self.fold['counts']={k:len(v) for k,v in self.refs.items()}
        self.fold['sha256']=hashlib.sha256(b''.join(self.refs[k].tobytes() for k in ('train','validation','test'))).hexdigest()
        self.max_devices=0
        for name,_ in self.group_references('train'):
            array=np.load(ROOT/self.manifest['topologies'][name]['folder']/'values.npy',mmap_mode='r')
            self.max_devices=max(self.max_devices,array.shape[1]);array._mmap.close()
        self.parameter_dim=self.max_devices*(3*len(self.schema['device_features'])+1)
        from src.graph.builder import CircuitBatch
        import torch
        @dataclass
        class ControlBatch(CircuitBatch):
            parameter_input:torch.Tensor|None=None
            conditions:torch.Tensor|None=None
            topology_ids:list|None=None
        self.batch_type=ControlBatch
    def batch(self,role,indices):
        b=super().batch(role,indices)
        refs=sorted([(self.topologies[int(i)],int(j)) for i,j in self.refs[role][indices]],key=lambda p:p[0])
        import torch
        out=self.batch_type(**vars(b));out.topology_ids=[n for n,_ in refs]
        out.conditions=torch.tensor([self.conditions[n] for n,_ in refs],dtype=torch.float32)
        if self.cfg['model_kind']=='mlp':
            features=[]
            for name,row in refs:
                arrays=self.fetch(name)['arrays'];v=arrays['values'][row];k=arrays['known'][row];z=arrays['zero'][row]
                padded=np.zeros((self.max_devices,3*v.shape[1]+1),dtype='float32')
                if len(v)>self.max_devices: raise ValueError('Device count outside train-defined MLP support')
                padded[:len(v),:-1]=np.concatenate([v,k,z],axis=1);padded[:len(v),-1]=1.
                features.append(padded.reshape(-1))
            out.parameter_input=torch.from_numpy(np.stack(features))
        return out

def scope_receipt(reader):
    out={'counts':dict(reader.fold['counts']),'roles_sha256':reader.fold['sha256'],
         'original_ID_protocol_sha256':sha(reader.protocol_path),'sources':{},
         'train_test_refs_disjoint':True,'test_labels_opened':False}
    sets={r:set(map(tuple,a.tolist())) for r,a in reader.refs.items()}
    assert not (sets['train']&sets['validation'] or sets['train']&sets['test'] or sets['validation']&sets['test'])
    for role,a in reader.refs.items():
        out['sources'][role]={s:sum(reader.topologies[int(i)].startswith(s+'__') for i in a[:,0]) for s in ['FALCON','AICircuit','LOADBench']}
    return out
