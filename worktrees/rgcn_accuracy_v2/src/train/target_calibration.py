"""Train-only per-contract affine calibration in the existing symlog space.

NumPy/stdlib only: this module does not import torch or create a CUDA context.
The variance floor protects float32 roundoff, not a target-validity mask.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

FORMAT='contract_symlog_affine_v1'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(2**20),b''):value.update(data)
    return value.hexdigest()
def symlog_numpy(value):return np.sign(value)*np.log1p(np.abs(value))

def combine(n,mean,m2,values):
    """Stable population moments, combined independently of row chunk size."""
    values=np.asarray(values,dtype=np.float64)
    if not values.size:return n,mean,m2
    count=values.size;center=float(values.mean());variance_sum=float(np.square(values-center).sum())
    total=n+count;delta=center-mean
    return total,mean+delta*count/total,m2+variance_sum+delta*delta*n*count/total

def fit(root,cache,split,output,minimum_std=1e-6,roundoff_multiplier=32.):
    root=Path(root).resolve();cache=(root/cache).resolve();output=Path(output).resolve()
    if not output.is_relative_to(root):raise ValueError('Calibration output outside EDA')
    if output.exists():raise FileExistsError('Calibration artifact is immutable; use another output')
    manifest=read(cache/'manifest.json');tasks=read(cache/'tasks.json')
    if sha(root/split)!=manifest['split_sha256']:raise ValueError('Split hash mismatch')
    if sha(cache/'tasks.json')!=manifest['tasks_sha256']:raise ValueError('Task registry hash mismatch')
    groups=manifest['group_names'];moments={};training_rows={g:0 for g in groups};role_counts={'train':0,'validation':0,'test':0}
    training_ids=hashlib.sha256();inputs={};per_topology={}
    for name,spec in sorted(manifest['topologies'].items()):
        folder=root/spec['folder'];arrays={}
        for key in ['role','group','target','mask','sample_ids']:
            path=folder/(key+'.npy');signature=sha(path)
            if signature!=spec['cache_sha256'][key+'.npy']:raise ValueError('Cache changed: '+str(path))
            inputs[str(path.relative_to(root)).replace('\\','/')]=signature
            arrays[key]=np.load(path,mmap_mode='r',allow_pickle=False)
        roles=arrays['role'];per_topology[name]={}
        for code,label in enumerate(role_counts):
            count=int(np.count_nonzero(roles==code));role_counts[label]+=count;per_topology[name][label]=count
        # Selection is performed before touching target values. Neither held-out
        # role contributes to centering, scale, adapter eligibility or clipping.
        selected=np.flatnonzero(roles==0)
        for start in range(0,len(selected),4096):
            rows=selected[start:start+4096];indices=np.asarray(arrays['group'][rows])
            training_ids.update(np.asarray(arrays['sample_ids'][rows]).tobytes())
            for number in np.unique(indices):
                group=groups[int(number)];take=rows[indices==number];training_rows[group]+=len(take)
                for slot,rule in tasks[group]['slots'].items():
                    column=int(slot);observed=np.asarray(arrays['mask'][take,column],dtype=bool)
                    raw=np.asarray(arrays['target'][take[observed],column],dtype=np.float64)
                    if not np.isfinite(raw).all():raise ValueError('Nonfinite observed training label')
                    transformed=symlog_numpy(raw/rule['scale'])
                    item=moments.setdefault((group,slot),{'z':(0,0.,0.),'raw':(0,0.,0.)})
                    item['z']=combine(*item['z'],transformed);item['raw']=combine(*item['raw'],raw)
        del arrays
    if role_counts!=manifest['role_counts']:raise ValueError('Role counts changed')
    result={}
    for group in groups:
        slots={}
        for slot,rule in tasks[group]['slots'].items():
            item=moments.get((group,slot));n,center,m2=item['z'] if item else (0,0.,0.)
            if not n:
                # Declared slots may be missing for every sample. Do not invent
                # moments or remove their original missing-label mask.
                slots[slot]={'semantic_id':rule['semantic_id'],'count':0,'calibrated':False,'center':0.,'std':1.,
                    'empirical_std':None,'std_floor':None,'floor_applied':False,'training_raw_mean':None,
                    'training_raw_std':None,'engineering_scale':rule['scale']}
                continue
            std=float(np.sqrt(max(0.,m2/n)))
            floor=max(minimum_std,roundoff_multiplier*float(np.finfo(np.float32).eps)*max(1.,abs(center)))
            slots[slot]={'semantic_id':rule['semantic_id'],'count':n,'calibrated':True,'center':center,'std':max(std,floor),'empirical_std':std,
                'std_floor':floor,'floor_applied':std<floor,'training_raw_mean':item['raw'][1],
                'training_raw_std':float(np.sqrt(max(0.,item['raw'][2]/n))),'engineering_scale':rule['scale']}
        result[group]={'training_rows':training_rows[group],'source_dataset':tasks[group]['contract']['source_dataset'],'slots':slots}
    artifact={'format':FORMAT,'fit_role':'train','test_labels_used':False,'validation_labels_used':False,
        'transform':'z=sign(y/engineering_scale)*log1p(abs(y/engineering_scale)); u=(z-center)/std',
        'std_policy':{'minimum_std':minimum_std,'roundoff_multiplier':roundoff_multiplier,'population_variance':True,
                      'constant_targets':'kept observed; finite scale floor; mask unchanged'},
        'cache_manifest_sha256':sha(cache/'manifest.json'),'tasks_sha256':sha(cache/'tasks.json'),'split_sha256':manifest['split_sha256'],
        'training_sample_ids_sha256':training_ids.hexdigest(),'role_counts':role_counts,'input_sha256':inputs,
        'per_topology_role_counts':per_topology,'groups':result}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(artifact,indent=2,allow_nan=False),encoding='utf-8')
    return artifact

def validate(artifact,tasks,manifest):
    if artifact['format']!=FORMAT or artifact['fit_role']!='train':raise ValueError('Unsupported target calibration')
    if artifact['validation_labels_used'] or artifact['test_labels_used']:raise ValueError('Calibration used held-out labels')
    if artifact['tasks_sha256']!=manifest['tasks_sha256'] or artifact['split_sha256']!=manifest['split_sha256']:
        raise ValueError('Calibration task/split mismatch')
    if set(artifact['groups'])!=set(tasks):raise ValueError('Calibration contracts mismatch')
    for group,task in tasks.items():
        stats=artifact['groups'][group]['slots']
        if set(stats)!=set(task['slots']):raise ValueError('Calibration slots mismatch')
        for slot,rule in task['slots'].items():
            row=stats[slot]
            if row['semantic_id']!=rule['semantic_id'] or row['engineering_scale']!=rule['scale']:raise ValueError('Calibration semantics mismatch')
            if not np.isfinite([row['center'],row['std']]).all() or row['std']<=0:raise ValueError('Invalid calibration statistics')
            if row['count']==0:
                if row['calibrated'] or row['center']!=0. or row['std']!=1.:raise ValueError('Invented moments for unobserved target')
            elif row['count']<0 or not row['calibrated']:raise ValueError('Invalid calibration count')
