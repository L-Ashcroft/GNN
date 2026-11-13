"""Stable train-only input and per-semantic-target statistics for oneLOTO fold."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
from data import ROOT,sha
class Moments:
    def __init__(self,dim):self.count=np.zeros(dim,dtype='int64');self.mean=np.zeros(dim);self.m2=np.zeros(dim)
    def update(self,x,mask):
        n=mask.sum(0).astype('int64');mu=np.divide(np.where(mask,x,0).sum(0),n,out=np.zeros(x.shape[1]),where=n>0)
        m2=np.where(mask,np.square(x-mu),0).sum(0);total=self.count+n;delta=mu-self.mean
        self.mean+=np.divide(delta*n,total,out=np.zeros_like(mu),where=total>0)
        self.m2+=m2+np.divide(delta*delta*self.count*n,total,out=np.zeros_like(mu),where=total>0);self.count=total
    def result(self,kind):
        std=np.sqrt(np.divide(self.m2,self.count,out=np.zeros_like(self.mean),where=self.count>0))
        if kind=='input':std=np.where(std>=1e-6,std,1.)
        else:std=np.where(self.count>0,np.maximum(std,1e-6),1.)
        return {'count':self.count.tolist(),'center':self.mean.tolist(),'std':std.tolist()}
def fit_stats(reader,path):
    proof={'fold_references_sha256':reader.fold['sha256'],'fold_id':reader.fold['fold_id'],'protocol_sha256':sha(reader.protocol_path),
        'raw_manifest_sha256':sha(reader.path/'manifest.json'),'fit_script_sha256':sha(Path(__file__)),'training_rows':len(reader.refs['train']),
        'test_or_validation_labels_used':False,'excluded_test_topologies':sorted(reader.held_topologies)}
    if path.exists():
        saved=json.loads(path.read_text(encoding='utf-8'));assert saved['proof']==proof;return saved
    inputs=[Moments(len(reader.schema['device_features'])) for _ in reader.schema['domains']];targets={g:Moments(16) for g in reader.tasks};visited=0;hash_ids=hashlib.sha256()
    for name,rows in reader.group_references('train'):
        assert name not in reader.held_topologies
        arrays=reader.fetch(name)['arrays']
        for start in range(0,len(rows),512):
            indices=rows[start:start+512];values=arrays['values'][indices].astype('float64');valid=arrays['known'][indices]&~arrays['zero'][indices]
            domain=arrays['domain'][indices];numbers=arrays['group'][indices]
            for number in np.unique(domain):
                selected=domain==number;inputs[int(number)].update(values[selected].reshape(-1,values.shape[-1]),valid[selected].reshape(-1,values.shape[-1]))
            y=arrays['target'][indices].astype('float64');mask=arrays['mask'][indices]
            if not np.isfinite(y[mask]).all():raise ValueError('Invalid observed training label')
            for number in np.unique(numbers):
                original=reader.groups[int(number)];g=reader.alias.get(original,original);selected=numbers==number
                scale=np.asarray([reader.tasks[g]['slots'].get(str(j),{}).get('scale',1.) for j in range(16)],dtype='float32').astype('float64')
                yy=y[selected]/scale;z=np.sign(yy)*np.log1p(abs(yy));targets[g].update(z,mask[selected])
            hash_ids.update(arrays['sample_ids'][indices].tobytes());visited+=len(indices)
    assert visited==len(reader.refs['train'])
    record={'proof':proof,'fitted_at':datetime.now(timezone.utc).isoformat(),'training_id_hash_topology_order':hash_ids.hexdigest(),
        'inputs':[m.result('input') for m in inputs],'targets':{g:m.result('target') for g,m in targets.items()},
        'input_policy':'Source/PDK-domain statistics on known nonzero unique-device values; only numeric0:19 normalized; flags/types unchanged',
        'target_policy':'Observed normalized symlog values, per approved semantic contract; std floor1e-6; missingness unchanged'}
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2),encoding='utf-8');temp.replace(path);return record
