"""One frozen CPU test of the validation-selected LOADBench correction.

Never fits parameters; original outputs are read-only. Per-topology recovery.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import argparse,importlib.util,json,sys,time,hashlib
import psutil
ROOT=Path(__file__).resolve().parents[1]
if psutil.virtual_memory().available<2*2**30:raise SystemExit('Deferred: require 2GiB host RAM for native CPU graph evaluation')
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
sys.path.insert(0,str(ROOT/'worktrees/loadbench_remediation_v1'))
from model import CorrectedReadout
from run import sha,read,save,summary
spec=importlib.util.spec_from_file_location('original_frozen_local_model',ROOT/'worktrees/rgcn_local_head_v1/model.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
sys.path.insert(0,str(ROOT/'worktrees/autonomous_training_v1'))
from stream_graphs import StreamGraphs

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--probe',action='store_true');args=ap.parse_args()
    base=ROOT/'artifacts/loadbench_remediation_v1';out=base/('native_probe' if args.probe else 'fixed_test');out.mkdir(exist_ok=True)
    correction=CorrectedReadout();bundle_path=base/'bundle.json'
    recipe={'bundle_sha256':sha(bundle_path),'evaluation_code_sha256':sha(Path(__file__)),
      'probe':args.probe,'batch':32,'device':'cpu','torch_threads':1,'deterministic':True,
      'test_is_previous_fixed_holdout_not_new_blind_data':True,'fit_or_selection_allowed':False}
    if (out/'recipe.json').exists():assert read(out/'recipe.json')==recipe
    else:save(out/'recipe.json',recipe)
    if (out/'summary.json').exists():print('Frozen evaluation already completed; retained.');return
    tick=time.monotonic();net,_=legacy.load_model(device='cpu')
    reader=StreamGraphs(ROOT,'data/model_ready/fusion_v2_training_v1','data/model_ready/fusion_v2_frozen_e19_bf0d71ad','versions/rgcn_calibrated_head_v1',capacity=1,verify=True)
    from src.train.metrics import MetricAccumulator
    oldrows=[];newrows=[];done=0;peak=0;comparisons=[]
    try:
        for top,specification in reader.manifest['topologies'].items():
            if not top.startswith('LOADBench__'):continue
            if args.probe and top!='LOADBench__SimpleOpAmp4':continue
            receipt=out/(top+'.json')
            if receipt.exists():
                r=read(receipt);assert r['bundle_sha256']==recipe['bundle_sha256']
                oldrows+=r['old'];newrows+=r['new'];done+=r['rows'];continue
            folder=ROOT/specification['folder'];roles=np.load(folder/'role.npy',mmap_mode='r');sid=np.load(folder/'sample_ids.npy',mmap_mode='r')
            positions=np.flatnonzero(roles==(1 if args.probe else 2))
            if args.probe:positions=positions[:32]
            assert not set(sid[positions].tolist())&set(sid[roles==0].tolist())
            raw=pq.read_table(ROOT/'data/fusion_v2/sampled'/(top+'.parquet'),columns=['parameters','parameter_names'])
            parameters=np.asarray(raw['parameters'].to_pylist());names=raw['parameter_names'][0].as_py()
            old_metric=MetricAccumulator(reader.tasks);new_metric=MetricAccumulator(reader.tasks);records=[]
            with torch.no_grad():
                for start in range(0,len(positions),32):
                    if psutil.virtual_memory().available<900*2**20:raise MemoryError('Resource guard; original trainer unaffected, completed shards resumable')
                    chosen=positions[start:start+32];batch=reader.batch_refs([(top,int(i)) for i in chosen],'validation' if args.probe else 'test')
                    state=net(batch,return_states=True);oldz=state['prediction'];basez=oldz.cpu().numpy();z=basez.copy();e=state['embedding'].cpu().numpy()
                    for g in set(batch.group_ids):
                        ix=np.array([i for i,v in enumerate(batch.group_ids) if v==g])
                        z[ix]=correction.predict(g,parameters[chosen[ix]],e[ix],basez[ix],parameter_names=names)
                    newz=torch.tensor(z,dtype=torch.float64);old_metric.update(oldz,batch);new_metric.update(newz,batch)
                    physical=(newz.sign()*torch.expm1(newz.abs())*batch.scales.to(torch.float64)).numpy()
                    for i,sample_id in enumerate(batch.sample_ids):records.append({'sample_id':sample_id,'prediction':physical[i].tolist(),'target':batch.targets[i].tolist(),'mask':batch.mask[i].tolist()})
                    peak=max(peak,psutil.Process().memory_info().rss)
            a=old_metric.compute();b=new_metric.compute();oldrows+=a['per_target'];newrows+=b['per_target'];done+=len(positions)
            parquet=out/(top+'.parquet');pq.write_table(pa.Table.from_pylist(records),parquet,compression='zstd')
            save(receipt,{'topology':top,'rows':len(positions),'old':a['per_target'],'new':b['per_target'],
              'bundle_sha256':recipe['bundle_sha256'],'predictions_sha256':sha(parquet)})
            roles._mmap.close();sid._mmap.close()
            save(out/'status.json',{'state':'evaluating','samples_done':done,'topology':top,'elapsed_seconds':time.monotonic()-tick})
            print(top,done,flush=True)
        result={'state':'complete','rows':done,'old':summary(oldrows),'new':summary(newrows),
          'old_per_target':oldrows,'new_per_target':newrows,'bundle_sha256':sha(bundle_path),
          'bundle_unchanged':sha(bundle_path)==recipe['bundle_sha256'],'seconds':time.monotonic()-tick,'peak_RSS_MiB':peak/2**20,
          'test_used_for_fitting_or_reselection':False,'previous_fixed_holdout_not_new_blind_test':True}
        save(out/'summary.json',result);save(out/'status.json',{'state':'complete','samples_done':done})
        print(json.dumps({k:v for k,v in result.items() if 'per_target' not in k},indent=2),flush=True)
    finally:reader.close()

if __name__=='__main__':main()
