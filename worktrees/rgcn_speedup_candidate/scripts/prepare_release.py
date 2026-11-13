"""Publish the verified CPU/GPU-equivalent implementation and derive a new run.

No trainer is started here. Existing production files are read only.
"""
from pathlib import Path
import hashlib
import json
import shutil
import statistics
import sys
from datetime import datetime, timezone

CANDIDATE=Path(__file__).resolve().parents[1]
ROOT=CANDIDATE.parents[1]
VERSION=ROOT/'versions/rgcn_cached_fast_v1'
OUTPUT=ROOT/'artifacts/experiment1_fusion_id_fast_v1'
MONITOR=ROOT/'monitoring/experiment1_fusion_id_fast_v1'
CONFIG=ROOT/'config/experiment1_fusion_id_fast_v1.json'
PARENT=ROOT/'artifacts/experiment1_fusion_id_v1'

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def rel(path):return path.relative_to(ROOT).as_posix()

def main():
    import psutil
    if psutil.virtual_memory().available<3*2**30:raise RuntimeError('Need 3 GiB RAM for CPU publication/validation')
    # Exact process identity and actual command, rather than a stale PID.
    for p in psutil.process_iter(['name','cmdline']):
        if (p.info['name'] or '').lower() in {'python.exe','pythonw.exe'} and any(
            Path(a.replace('\\','/')).name in {'train_experiment1.py','train_experiment1_fast.py'} for a in p.info['cmdline'] or []):
            raise RuntimeError(f'Trainer still running: {p.pid}')
    for path in [VERSION,OUTPUT,MONITOR,CONFIG]:
        if path.exists():raise FileExistsError(f'Refusing overwrite: {path}')
    proofs=[CANDIDATE/'results'/f'correctness_cuda_cached_v2_{kind}.json' for kind in ['fresh','parent']]
    for path in proofs:
        proof=read(path)
        assert proof['status']=='passed' and proof['device']=='cuda' and proof['batch_size']==256
        assert proof['deterministic'] and proof['production_hashes_unchanged']
        for name,value in proof['optimized_source_sha256'].items():assert sha(CANDIDATE/name)==value,name
    parent_proof=read(proofs[1]);parent_sha=sha(PARENT/'last.pt')
    assert parent_proof['parent_checkpoint_sha256']==parent_sha
    old=read(PARENT/'recipe.json');snapshot=read(ROOT/'monitoring/experiment1_speedup_migration/parent_snapshot.json')
    assert parent_sha==snapshot['parent_checkpoint_sha256']
    assert sha(PARENT/'best.pt')==snapshot['parent_best_sha256']
    assert sha(PARENT/'recipe.json')==snapshot['parent_recipe_sha256']
    for name,value in old['implementation_sha256'].items():assert sha(ROOT/name)==value,name
    bench_folder=CANDIDATE/'results/speedup_20261031'
    reference=read(bench_folder/'reference_v2_e2e.json');optimized=read(bench_folder/'optimized_v2_e2e.json')
    for b in [reference,optimized]:
        assert b['status']=='complete' and b['device']=='cuda' and b['batch_size']==256
        assert b['timing_mode']=='end_to_end' and b['cpu_threads']==2 and b['deterministic']
    assert reference['sample_ids']==optimized['sample_ids']
    for name,value in optimized['source_sha256'].items():assert sha(CANDIDATE/name)==value,name
    ref_times=[r['wall_seconds'] for r in reference['measurements']]
    opt_times=[r['wall_seconds'] for r in optimized['measurements']]
    speedup=statistics.mean(ref_times)/statistics.mean(opt_times)
    assert speedup>1.10 and min(ref_times)>max(opt_times),'No clear measured benefit: do not migrate'
    assert read(CANDIDATE/'results/edge_case_tests.json')['success']
    shutil.copytree(CANDIDATE/'src',VERSION/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    scripts=VERSION/'scripts';scripts.mkdir()
    entry=(CANDIDATE/'candidate/train_experiment1_fast.py').read_text(encoding='utf-8')
    entry=entry.replace("'config_sha256':sha(CONFIG),", "'config_sha256':sha(CONFIG),'parent_migration_sha256':sha(output/'migration_lineage.json') if (output/'migration_lineage.json').exists() else None,")
    entry=entry.replace("update(state='validating',epoch=epoch+1);validation_started=time.perf_counter()", "segment_seconds=time.perf_counter()-segment_started\n            update(state='validating',epoch=epoch+1);validation_started=time.perf_counter()")
    entry=entry.replace("'active_segment_seconds':time.perf_counter()-segment_started,'active_segment_samples':segment_samples", "'active_segment_seconds':segment_seconds,'active_segment_samples':segment_samples")
    (scripts/'train_experiment1_fast.py').write_text(entry,encoding='utf-8')
    # Adapt separate observer/recovery/validator copies; original monitor stays frozen.
    for name in ['experiment1_health.py','monitor_experiment1.py','experiment1_recovery.py','validate_experiment1_resume.py']:
        text=(ROOT/'scripts'/name).read_text(encoding='utf-8')
        text=text.replace('experiment1_fusion_id_v1','experiment1_fusion_id_fast_v1')
        text=text.replace('config/experiment1_fusion.json','config/experiment1_fusion_id_fast_v1.json')
        for module in ['experiment1_health.py','experiment1_recovery.py','validate_experiment1_resume.py']:
            text=text.replace('scripts/'+module,'versions/rgcn_cached_fast_v1/scripts/'+module)
        if name=='experiment1_health.py':
            text=text.replace('ROOT=Path(__file__).resolve().parents[1]','ROOT=Path(__file__).resolve().parents[3]')
        if name=='monitor_experiment1.py':
            text=text.replace("snapshot=read(ROOT/'worktrees/rgcn_optimization/snapshot.json');expected_pid=snapshot['production_before']['pid'];expected_created=snapshot['pid_create_time']",
                "identity=read(OUTPUT/'recovery_authorization.json');expected_pid=identity['initial_training_pid'];expected_created=identity['initial_training_create_time']")
        if name=='experiment1_recovery.py':
            text=text.replace("=='train_experiment1.py'", " in {'train_experiment1.py','train_experiment1_fast.py'}")
            text=text.replace("'scripts/train_experiment1.py','--resume'", "'versions/rgcn_cached_fast_v1/scripts/train_experiment1_fast.py','--config','config/experiment1_fusion_id_fast_v1.json','--resume'")
            # Preserve own expected process adoption: an unrelated old trainer cannot be adopted.
            text=text.replace("result.append({'pid':process.pid,'create_time':process.create_time()})",
                "result.append({'pid':process.pid,'create_time':process.create_time(),'own_version':any(str(a).replace('\\\\','/').endswith('versions/rgcn_cached_fast_v1/scripts/train_experiment1_fast.py') for a in args)})")
            text=text.replace("p['pid']==status.get('pid') and 'create_time' in p", "p['pid']==status.get('pid') and p.get('own_version') and 'create_time' in p")
        if name=='validate_experiment1_resume.py':
            text=text.replace("sys.path.insert(0,str(ROOT));main()", "sys.path.insert(0,str(Path(__file__).resolve().parents[1]));main()")
        (scripts/name).write_text(text,encoding='utf-8')
    config=read(ROOT/'config/experiment1_fusion.json');config['output']=rel(OUTPUT);write(CONFIG,config)
    OUTPUT.mkdir();MONITOR.mkdir()
    lineage={'at_UTC':datetime.now(timezone.utc).isoformat(),'kind':'explicit_verified_implementation_migration',
        'authorization':'User allowed graceful pause, GPU acceptance and independent fast-run migration; no cross-recipe --resume.',
        'parent_output':rel(PARENT),'parent_checkpoint_sha256':parent_sha,'parent_recipe_sha256':sha(PARENT/'recipe.json'),
        'parent_best_sha256':sha(PARENT/'best.pt'),'GPU_correctness':[rel(p) for p in proofs],
        'measured_speedup':speedup,'benchmark_files':[rel(bench_folder/p) for p in ['reference_v2_e2e.json','optimized_v2_e2e.json']],
        'frozen_parent_epochs':15,'implementation_switch_epoch':16,'implementation_switch_completed_batches':1024,
        'state_policy':'Preserve model, optimizer, scheduler, RNG, history, cursor, coverage and accumulators; derive new recipe digest only.'}
    write(OUTPUT/'migration_lineage.json',lineage)
    sources=sorted({rel(p) for p in (VERSION/'src').rglob('*.py')}|{rel(scripts/'train_experiment1_fast.py')})
    recipe={'config':config,'execution_variant':'edge_rgcn_cached_fast_v1','data_root':str(ROOT),'config_sha256':sha(CONFIG),
        'parent_migration_sha256':sha(OUTPUT/'migration_lineage.json'),
        'cache_manifest_sha256':sha(ROOT/config['cache']/'manifest.json'),'split_sha256':sha(ROOT/config['split']),
        'implementation_sha256':{p:sha(ROOT/p) for p in sources},'torch':old['torch'],'GPU':old['GPU']}
    write(OUTPUT/'recipe.json',recipe)
    import torch
    torch.set_num_threads(1)
    checkpoint=torch.load(PARENT/'last.pt',map_location='cpu',weights_only=True)
    assert checkpoint['recipe_digest']==digest(old)
    assert checkpoint['next_epoch']==15 and checkpoint['next_batch']==1024
    assert checkpoint['partial']['samples']==262144 and checkpoint['partial']['grad_count']==1024
    # Independently hash every preserved state component, including scalar cursors/RNG.
    def state_digest(value):
        h=hashlib.sha256()
        def walk(v):
            if isinstance(v,torch.Tensor):
                h.update(str((v.dtype,tuple(v.shape))).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
            elif isinstance(v,dict):
                for k,item in v.items():h.update(repr(k).encode());walk(item)
            elif isinstance(v,(list,tuple)):
                h.update(type(v).__name__.encode())
                for item in v:walk(item)
            else:h.update(repr(v).encode())
        walk(value);return h.hexdigest()
    before={k:state_digest(v) for k,v in checkpoint.items() if k!='recipe_digest'}
    checkpoint['recipe_digest']=digest(recipe);torch.save(checkpoint,OUTPUT/'last.pt')
    write(OUTPUT/'history.json',checkpoint['history'])
    reloaded=torch.load(OUTPUT/'last.pt',map_location='cpu',weights_only=True)
    after={k:state_digest(v) for k,v in reloaded.items() if k!='recipe_digest'}
    assert before==after
    del checkpoint,reloaded
    best=torch.load(PARENT/'best.pt',map_location='cpu',weights_only=True)
    best_model=state_digest(best['model']);best['recipe_digest']=digest(recipe);best['config']=config
    torch.save(best,OUTPUT/'best.pt');assert best_model==state_digest(torch.load(OUTPUT/'best.pt',map_location='cpu',weights_only=True)['model'])
    assert sha(PARENT/'last.pt')==parent_sha and sha(PARENT/'best.pt')==lineage['parent_best_sha256']
    receipt={'status':'prepared_not_started','at_UTC':datetime.now(timezone.utc).isoformat(),
        'new_recipe_digest':digest(recipe),'new_checkpoint_sha256':sha(OUTPUT/'last.pt'),'preserved_state_sha256':before,
        'every_state_component_unchanged':True,'best_model_state_unchanged':True,'original_files_unchanged':True,
        'source_files':len(sources),'execution_py_included':any(p.endswith('/model/execution.py') for p in sources),
        'speedup':speedup,'parent':lineage}
    write(OUTPUT/'migration_receipt.json',receipt)
    # Compile-only: no GPU allocation, no formal training invocation.
    for path in (VERSION).rglob('*.py'):compile(path.read_text(encoding='utf-8'),str(path),'exec')
    print(json.dumps({'status':'prepared_not_started','speedup':speedup,'recipe_digest':digest(recipe),'version':str(VERSION)},indent=2))

if __name__=='__main__':main()
