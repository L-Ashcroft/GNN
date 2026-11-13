"""Freeze validation-selected correction and replay its public interface."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json,time
import numpy as np
import pyarrow.parquet as pq
from run import ROOT,OUT,RAW,META,SCALES,read,save,sha
from model import CorrectedReadout

def main():
    tick=time.monotonic();selected=read(OUT/'calibration/selection.json');alpha=selected['selected_global_weight']
    if alpha is None:raise SystemExit('No accepted correction; do not publish')
    original=ROOT/'artifacts/experiment1_rgcn_local_head_v1/bundle.json';bundle=read(original)
    groups={}
    for route in selected['routes']:
        g=route['group'];top=bundle['contracts'][g]['topology']
        table=pq.read_table(ROOT/'data/fusion_v2/sampled'/(top+'.parquet'),columns=['parameter_names','parameter_units'])
        names=table['parameter_names'][0].as_py();units=table['parameter_units'][0].as_py();index=OUT/(g+'.npz')
        groups[g]={'topology':top,'parameter_names':names,'unit_scales':[SCALES.get(u,1.) for u in units],
          'effective_weight':float(alpha)*route['weight'],'index':index.relative_to(ROOT).as_posix(),'index_sha256':sha(index)}
    publication={'format':'train_calibrated_GNN_local_affine_correction_v1','state':'validation_accepted_test_pending',
      'original_bundle':original.relative_to(ROOT).as_posix(),'original_bundle_sha256':sha(original),
      'groups':groups,'selected_global_weight':float(alpha),'validation':selected['summary'][alpha],
      'baseline_validation':selected['summary']['0.0'],'test_labels_read':False,'LOTO_supported':False,
      'unchanged_sources':['FALCON','AICircuit'],'original_labels_masks_splits_modified':False,
      'implementation_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
      'calibration_recipe_sha256':sha(OUT/'calibration/recipe.json')}
    path=OUT/'bundle.json'
    if path.exists():assert read(path)==publication,'Published correction changed; create a new version'
    else:save(path,publication)
    model=CorrectedReadout(path);fm=read(META/'manifest.json')
    meta={k:np.load(META/'validation'/(k+'.npy'),mmap_mode='r') for k in ['embedding','group','refs']}
    maximum=0.;rows=0
    for g,item in groups.items():
        positions=np.flatnonzero(np.asarray(meta['group'])==fm['groups'].index(g))
        raw=pq.read_table(ROOT/'data/fusion_v2/sampled'/(item['topology']+'.parquet'),columns=['parameters'])
        params=np.asarray(raw['parameters'].to_pylist())[np.asarray(meta['refs'][positions,1],dtype=int)]
        old=np.load(OUT/(g+'__published.npy'));new=np.load(OUT/(g+'__graph_design_affine32.npy'))
        result=model.predict(g,params,np.asarray(meta['embedding'][positions]),old,parameter_names=item['parameter_names'])
        expected=old+item['effective_weight']*(new-old)
        maximum=max(maximum,float(abs(result-expected).max()));rows+=len(result)
        np.testing.assert_allclose(result,expected,rtol=1e-10,atol=1e-10)
        np.save(OUT/(g+'__accepted.npy'),result,allow_pickle=False)
    # Other sources pass through exactly, not just approximately.
    other=next(g for g in bundle['contracts'] if g not in groups)
    baseline=np.random.default_rng(42).normal(size=(2,16))
    np.testing.assert_array_equal(model.predict(other,None,None,baseline,parameter_names=[]),baseline)
    checks={'state':'passed','real_LOADBench_validation_rows':rows,'contracts':len(groups),
      'maximum_symlog_absolute_replay_error':maximum,'atol':1e-10,'rtol':1e-10,
      'other_sources_exact_passthrough':True,'test_labels_read':False,'seconds':time.monotonic()-tick,
      'original_bundle_hash_unchanged':sha(original)==publication['original_bundle_sha256']}
    save(OUT/'interface_checks.json',checks);print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':main()
