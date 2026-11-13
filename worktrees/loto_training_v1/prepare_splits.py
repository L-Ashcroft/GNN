"""Independent LOTO roles using the full public500k population.

All target structures held out; non-target former ID-test records may train in
this separate experiment. Original ID and previousLOTO files stay immutable.
"""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser();parser.add_argument('--output-version',default='verified_loto_full_v2');args=parser.parse_args()
assert '/' not in args.output_version and '\\' not in args.output_version and args.output_version not in {'.','..'}
OUT=ROOT/'data/fusion_v2/splits'/args.output_version;OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'manifest.json').exists():raise RuntimeError('Protocol already published; choose a new --output-version. Never rewrite live recipe inputs.')
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
oldpath=ROOT/'data/fusion_v2/splits/verified_loto_v1/manifest.json';old=read(oldpath);raw=read(ROOT/'data/model_ready/fusion_v2_training_v1/manifest.json')
source=ROOT/old['source_split'];assert sha(source)==old['source_split_sha256']
table=pq.read_table(source,columns=['sample_id','design_group','source_dataset','connectivity_group','topology_id','role']);c={k:np.asarray(table[k].to_pylist(),dtype=object) for k in table.column_names}
assert len(c['sample_id'])==500000 and set(c['source_dataset'])=={'FALCON','AICircuit','LOADBench'}
position={str(s):i for i,s in enumerate(c['sample_id'])};refs=np.full((500000,2),-1,dtype='int32')
for i,name in enumerate(old['topologies']):
    ids=np.load(ROOT/raw['topologies'][name]['folder']/'sample_ids.npy',mmap_mode='r')
    for j,s in enumerate(ids):refs[position[s.decode()]]=[i,j]
    ids._mmap.close()
assert (refs>=0).all();folds=[]
for original in old['folds']:
    fold=dict(original);held=c['connectivity_group']==fold['heldout_structure']
    test=held&np.isin(c['topology_id'],fold['test_topologies']);train=~held&(c['role']!='validation');val=~held&(c['role']=='validation');unused=~(train|val|test)
    parts={'train':train,'validation':val,'test':test,'unused':unused}
    design={role:set(c['design_group'][mask]) for role,mask in parts.items() if role!='unused'}
    assert not design['train']&design['validation'] and not design['train']&design['test'] and not design['validation']&design['test']
    assert not set(c['connectivity_group'][train])&set(c['connectivity_group'][test])
    payload={}
    for role,mask in parts.items():
        at=np.flatnonzero(mask).astype('int32');payload[role+'_positions']=at
        if role!='unused':payload[role+'_references']=refs[at]
    path=OUT/(fold['fold_id']+'.npz');np.savez_compressed(path,**payload)
    fold.update(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),counts={role:int(mask.sum()) for role,mask in parts.items()},
        source_counts={role:{s:int(np.sum(mask&(c['source_dataset']==s))) for s in ['FALCON','AICircuit','LOADBench']} for role,mask in parts.items()},
        role_policy='All non-held public records: former IDtrain+IDtest -> LOTOtrain, original IDvalidation -> LOTOvalidation; target all -> LOTOtest')
    coverage=np.zeros(16,dtype='int64')
    for peer in fold['compatible_training_topologies']:
        folder=ROOT/raw['topologies'][peer]['folder'];mask=np.load(folder/'mask.npy',mmap_mode='r');peer_number=old['topologies'].index(peer)
        peer_rows=refs[train&(refs[:,0]==peer_number),1];coverage+=mask[peer_rows].sum(0);mask._mmap.close()
    fold['training_target_counts']=coverage.tolist()
    folds.append(fold)
new=dict(old,prepared_at=datetime.now(timezone.utc).isoformat(),state='full_public500k_LOTO_roles_prepared',folds=folds,
    parent_protocol=oldpath.relative_to(ROOT).as_posix(),parent_protocol_sha256=sha(oldpath),source_population=500000,
    private_AnalogGym_used=False,original_dataset_or_ID_split_modified=False,
    policies=[p for p in old['policies'] if not p.startswith('Non-target original ID-test samples remain unused')]+['This independent protocol uses non-target former ID-test samples for training; never target records',
        'All statistics must be refit from these expanded train references; priorID/LOTO-v1calibration forbidden'])
(OUT/'manifest.json').write_text(json.dumps(new,indent=2),encoding='utf-8')
first=next(f for f in folds if f['fold_id']=='LB_M04__hold_SimpleOpAmp4')
print(json.dumps({'folds':len(folds),'first_fold':first['fold_id'],'counts':first['counts'],'source_counts':first['source_counts'],'size_MiB':sum(p.stat().st_size for p in OUT.iterdir())/2**20},indent=2))
