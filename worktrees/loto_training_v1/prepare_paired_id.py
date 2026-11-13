"""Prepare, but do not launch, a same-architecture ID reference for fair comparison."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;DEST=ROOT/'worktrees/experiment2_paired_id_v1';DEST.mkdir(parents=True,exist_ok=True)
if (ROOT/'artifacts/experiment2_paired_id_v1/recipe.json').exists():raise RuntimeError('Paired control already published or running; use a new version, do not overwrite sources')
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=read(ROOT/'data/fusion_v2/splits/verified_loto_full_v2/manifest.json');first=protocol['folds'][0]
refs=np.full((500000,2),-1,dtype='int32')
with np.load(ROOT/first['path'],allow_pickle=False) as f:
    for role in ['train','validation','test']:refs[f[role+'_positions']]=f[role+'_references']
assert (refs>=0).all()
roles=np.asarray(pq.read_table(ROOT/protocol['source_split'],columns=['role'])['role'].to_pylist(),dtype=object);payload={}
for role in ['train','validation','test']:
    ix=np.flatnonzero(roles==role).astype('int32');payload[role+'_positions']=ix;payload[role+'_references']=refs[ix]
payload['unused_positions']=np.array([],dtype='int32')
path=ROOT/'data/fusion_v2/splits/verified_loto_full_v2/paired_ID_reference.npz';np.savez_compressed(path,**payload)
fold={'fold_id':'paired_ID_same_neural_architecture_seed42','heldout_structure':None,'test_topologies':[],
    'excluded_structural_aliases':[],'head_alias':'verified_LOADBench_LB_M04','path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),
    'counts':{r:len(payload[r+'_positions']) for r in ['train','validation','test','unused']},
    'note':'ID fixed test spans all topologies; no topology holdout in this control. Same model/normalization algorithm; fit statistics separately.'}
control=dict(protocol,folds=[fold],state='paired_ID_reference_prepared_not_started',policies=['Independent originalID8:1:1reference; all topology classes may enter train','Every statistic fit only originalIDtrain400083','Approved target aliases only; original dataset andIDsplit remain unchanged','No ID-pretrained weights imported']);controlpath=ROOT/'data/fusion_v2/splits/verified_loto_full_v2/paired_ID_reference.json';controlpath.write_text(json.dumps(control,indent=2),encoding='utf-8')
cfg=read(HERE/'config.json');cfg.update(fold_id=fold['fold_id'],output='artifacts/experiment2_paired_id_v1',protocol=controlpath.relative_to(ROOT).as_posix())
for name in ['runtime.py','fit.py','model.py']:shutil.copy2(HERE/name,DEST/name)
text=(HERE/'data.py').read_text(encoding='utf-8').replace("splits/verified_loto_full_v2/manifest.json","splits/verified_loto_full_v2/paired_ID_reference.json");(DEST/'data.py').write_text(text,encoding='utf-8')
for name in ['train.py','qualify.py','supervisor.py']:
    text=(HERE/name).read_text(encoding='utf-8').replace('loto_training_v1/train.py','experiment2_paired_id_v1/train.py')
    text=text.replace("'experiment':'LOTO'","'experiment':'paired_ID_reference'").replace("'experiment':'public500k_verified_LOTO_v1'","'experiment':'public500k_paired_ID_v1'")
    (DEST/name).write_text(text,encoding='utf-8')
(DEST/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
(DEST/'comparison_contract.json').write_text(json.dumps({'state':'prepared_not_started','LOTO_config_sha256':sha(HERE/'config.json'),
    'model_code_identical':sha(HERE/'model.py')==sha(DEST/'model.py'),'model_sha256':sha(HERE/'model.py'),
    'differences':['Independent role split and fold calibration','ID includes target training designs;LOTO excludes whole target structure'],
    'prohibited_comparison':'Do not attribute52.05%hybrid-ID versus neural-LOTO difference solely to topology shift',
    'execution':'Run separately after firstLOTO, no parallel large training. No control model has been trained.'},indent=2),encoding='utf-8')
assert sha(HERE/'model.py')==sha(DEST/'model.py')
print(json.dumps({'prepared_only':True,'same_model_source':True,'ID_counts':fold['counts'],'private_data_used':False},indent=2))
