"""All-contract real graph checks and complete training-index leakage checks."""
from pathlib import Path
import sys,json,copy
import psutil
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
if psutil.virtual_memory().available<1.25*2**30:raise SystemExit(75)
import numpy as np
import torch
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(42)
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'worktrees/autonomous_training_v1'))
from model import load_model,sha
from stream_graphs import StreamGraphs
net,bundle=load_model();OUT=ROOT/'artifacts/experiment1_rgcn_local_head_v1'
RAW=ROOT/'data/model_ready/fusion_v2_training_v1';META=ROOT/'data/model_ready/fusion_v2_frozen_e19_bf0d71ad'
manifest=json.loads((RAW/'manifest.json').read_text());fm=json.loads((META/'manifest.json').read_text());group=np.load(META/'validation/group.npy',mmap_mode='r')
expected=np.load(OUT/'validation_prediction_symlog.npy',mmap_mode='r');indices=[int(np.flatnonzero(group==i)[0]) for i in range(len(fm['groups']))]
reader=StreamGraphs(ROOT,'data/model_ready/fusion_v2_training_v1','data/model_ready/fusion_v2_frozen_e19_bf0d71ad','versions/rgcn_calibrated_head_v1',capacity=2)
comparisons=[];relations=set();sources=set();passed=True
with torch.no_grad():
    for start in range(0,len(indices),16):
        ids=indices[start:start+16];batch=reader.batch('validation',ids,sort_for_io=False);pred=net(batch).numpy();mask=batch.mask.numpy()
        ref=np.asarray(expected[ids]);difference=abs(pred-ref);ratio=difference/(5e-6+5e-5*abs(ref));ok=bool((ratio[mask]<=1).all());passed&=ok
        comparisons.append({'start':start,'graphs':len(ids),'max_observed_symlog_absolute_error':float(difference[mask].max()),'maximum_tolerance_ratio':float(ratio[mask].max()),'passed':ok})
        relations.update(batch.edge_type.tolist());sources.update(reader.tasks[g]['contract']['source_dataset'] for g in batch.group_ids)
    batch=reader.batch('validation',indices[:16],sort_for_io=False);g=batch.group_ids[0]
    bad=copy.deepcopy(batch);bad.edge_index[1,0]=bad.edge_index[0,0]
    # A relation edit certainly changes the contract even if the first edge
    # was already a self-loop in the original topology.
    bad.edge_type[0]=(bad.edge_type[0]+1)%48
    try:net.decoder.raw_from_batch(bad,g,0);raise AssertionError('Changed topology accepted')
    except ValueError:pass
    try:net.decoder.bank('unknown');raise AssertionError('Unknown contract accepted')
    except ValueError:pass
total_train=0;contracts=0
for name,spec in manifest['topologies'].items():
    folder=ROOT/spec['folder'];role=np.load(folder/'role.npy',mmap_mode='r');ids=np.load(folder/'sample_ids.npy',mmap_mode='r');numbers=np.load(folder/'group.npy',mmap_mode='r')
    validation_ids=set(ids[role==1].tolist())
    for number in np.unique(numbers[role==0]):
        g=manifest['group_names'][int(number)];rows=np.flatnonzero((role==0)&(numbers==number));path=ROOT/bundle['contracts'][g]['index'];assert sha(path)==bundle['contracts'][g]['index_sha256']
        with np.load(path,allow_pickle=False) as bank:
            assert np.array_equal(bank['train_ids'],ids[rows]);assert not set(bank['train_ids'].tolist())&validation_ids
        total_train+=len(rows);contracts+=1
    for a in [role,ids,numbers]:a._mmap.close()
assert total_train==400083 and contracts==142
result={'raw_graph_cached_readout_tolerance_passed':passed,'atol':5e-6,'rtol':5e-5,'comparisons':comparisons,
    'all_contracts_checked':len(indices),'all_48_relations_checked':len(relations)==48,'sources':sorted(sources),
    'training_neighbor_rows_verified':total_train,'training_index_no_validation_overlap':True,
    'changed_connections_or_ports_rejected':True,'unknown_contract_rejected':True,'test_read':False,'device':'cpu'}
(OUT/'checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2));reader.close()
if not passed:raise SystemExit(2)
