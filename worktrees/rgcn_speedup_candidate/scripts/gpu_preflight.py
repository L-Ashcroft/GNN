"""CPU-only GPU acceptance preflight. Never imports torch or starts CUDA work."""
import argparse
from collections import Counter
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import psutil

ROOT=Path(__file__).resolve().parents[1];WORKSPACE=ROOT.parents[1]
PRODUCTION=WORKSPACE/'artifacts/experiment1_fusion_id_v1'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def process_identity(pid,expected_creation=None):
    record={'pid':int(pid),'exists':False,'expected_create_time':expected_creation}
    try:
        process=psutil.Process(int(pid));created=process.create_time()
        record.update(exists=process.is_running(),create_time=created,
            same_creation_time=None if expected_creation is None else created==expected_creation,
            created_at_beijing=datetime.fromtimestamp(created,timezone(timedelta(hours=8))).isoformat())
        for name in ['name','cmdline','cwd']:
            try:record[name]=getattr(process,name)()
            except (psutil.AccessDenied,psutil.NoSuchProcess):record[name]=None
    except psutil.NoSuchProcess:pass
    except psutil.AccessDenied:record.update(exists=True,identity_unverified=True)
    return record

def is_training_command(arguments):
    # Match an argument token, not text embedded in a Python -c or shell command.
    return any(str(arg).replace('\\','/').split('/')[-1].lower()=='train_experiment1.py' for arg in arguments or [])

def blocking_reasons(original,active_training,available_bytes,required_ram_gib):
    reasons=[]
    if original.get('exists') and original.get('same_creation_time'):
        reasons.append('original_training_same_pid_and_creation_time_alive')
    if original.get('identity_unverified'):reasons.append('original_process_identity_unverified')
    if active_training:reasons.append('workspace_training_command_still_running')
    if available_bytes<required_ram_gib*2**30:reasons.append('insufficient_available_host_RAM')
    return reasons

def check_environment(required_ram_gib=4):
    snapshot=read(ROOT/'snapshot.json');status=read(PRODUCTION/'status.json')
    original=process_identity(snapshot['production_before']['pid'],snapshot['pid_create_time'])
    active=[];workspace_text=str(WORKSPACE).casefold()
    for process in psutil.process_iter(['pid','name','cmdline','create_time']):
        try:
            info=process.info
            if (info['name'] or '').lower() not in {'python.exe','pythonw.exe','python','python3'}:continue
            if not is_training_command(info['cmdline']):continue
            record=process_identity(info['pid'])
            if not record['exists']:continue
            cwd=record.get('cwd')
            if cwd is None or str(Path(cwd).resolve()).casefold()==workspace_text:active.append(record)
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    memory=psutil.virtual_memory();reasons=blocking_reasons(original,active,memory.available,required_ram_gib)
    # A live status PID whose creation/command cannot be inspected is uncertain.
    current=process_identity(status['pid'])
    if current.get('identity_unverified'):reasons.append('status_process_identity_unverified')
    now=datetime.now(timezone.utc)
    return {'checked_at_UTC':now.isoformat(),'checked_at_beijing':now.astimezone(timezone(timedelta(hours=8))).isoformat(),
        'platform':platform.platform(),'python_executable':sys.executable,
        'original_process':original,'active_training_processes':active,'status_process':current,'production_status':status,
        'available_RAM_bytes':memory.available,'available_RAM_GiB':memory.available/2**30,
        'total_RAM_bytes':memory.total,'required_available_RAM_GiB':required_ram_gib,
        'blocking_reasons':reasons,'eligible_for_CUDA':not reasons,'torch_imported':False,'CUDA_work_started':False}

def fingerprint_audit():
    snapshot=read(ROOT/'snapshot.json');cpu=read(ROOT/'results/correctness_4_final.json');checks={}
    for name,expected in snapshot['source_files'].items():
        checks['production/'+name]=sha(WORKSPACE/name)==expected
        checks['frozen/'+name]=sha(ROOT/'reference_src'/Path(name).relative_to('src'))==expected
    for name,expected in snapshot['protected'].items():checks['protected/'+name]=sha(WORKSPACE/name)==expected
    for name,expected in cpu['optimized_source_sha256'].items():checks['optimized/'+name]=sha(ROOT/name)==expected
    config=read(WORKSPACE/'config/experiment1_fusion.json');cache=WORKSPACE/config['cache'];manifest=read(cache/'manifest.json')
    fixture_manifest=read(WORKSPACE/'reports/experiment1_performance/fixtures/manifest.json')
    checks['fixture_cache_manifest']=sha(cache/'manifest.json')==fixture_manifest['cache_manifest_sha256']
    checks['fixture_training_config']=sha(WORKSPACE/'config/experiment1_fusion.json')==fixture_manifest['config_sha256']
    checks['cache_schema']=sha(cache/'schema.json')==manifest['schema_sha256']
    checks['cache_tasks']=sha(cache/'tasks.json')==manifest['tasks_sha256']
    checks['training_split']=sha(WORKSPACE/config['split'])==manifest['split_sha256']
    for fixture in fixture_manifest['selections']:
        checks['fixture/'+fixture['file']]=sha(WORKSPACE/'reports/experiment1_performance/fixtures'/fixture['file'])==fixture['sha256']
    return {'all_passed':all(checks.values()),'checks_count':len(checks),'checks':checks,
        'thresholds_atol_rtol':cpu['thresholds_atol_rtol'],
        'optimized_execution_sha256':sha(ROOT/'src/model/execution.py'),
        'production_recipe_sha256':sha(PRODUCTION/'recipe.json')}

def batch_audit():
    import numpy as np
    config=read(WORKSPACE/'config/experiment1_fusion.json');cache=WORKSPACE/config['cache'];manifest=read(cache/'manifest.json')
    selections=read(WORKSPACE/'reports/experiment1_performance/fixtures/manifest.json')['selections']
    selection=next(s for s in selections if s['size']==256)
    references=[tuple(pair) for pair in selection['references']]
    statics={};arrays={};relations=set();groups=set();sample_ids=[];sources=Counter();roles=Counter()
    nodes=edges=loops=parallel=typed_duplicates=observed=0;finite=True
    for name,row in references:
        if name not in arrays:
            folder=(WORKSPACE/manifest['topologies'][name]['folder']).resolve()
            if not folder.is_relative_to(WORKSPACE):raise ValueError('Cache path escapes workspace')
            statics[name]=read(folder/'structure.json')
            arrays[name]={k:np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False)
                for k in ['role','group','mask','target','sample_ids']}
        a=arrays[name];s=statics[name];role=int(a['role'][row]);roles[str(role)]+=1
        groups.add(manifest['group_names'][int(a['group'][row])]);mask=a['mask'][row].astype(bool)
        observed+=int(mask.sum());finite=finite and bool(np.isfinite(a['target'][row][mask]).all())
        sample_id=a['sample_ids'][row].decode('utf-8');sample_ids.append(sample_id);sources[sample_id.split(':')[0]]+=1
        pairs=list(zip(*s['edge_index']));triples=[(u,v,r) for (u,v),r in zip(pairs,s['edge_type'])]
        nodes+=len(s['x']);edges+=len(pairs);loops+=sum(u==v for u,v in pairs)
        parallel+=len(pairs)-len(set(pairs));typed_duplicates+=len(triples)-len(set(triples));relations.update(s['edge_type'])
    passed=(len(references)==len(set(sample_ids))==256 and roles=={'0':256} and len(relations)==48 and finite)
    return {'passed':passed,'fixture':selection['file'],'fixture_sha256':selection['sha256'],
        'batch_size':256,'source_topology_ids':len(statics),'sources':dict(sources),'role_counts':dict(roles),
        'unique_sample_ids':len(set(sample_ids)),'relations':len(relations),'relation_ids':sorted(relations),
        'supervision_contracts':len(groups),'nodes':nodes,'message_edges':edges,'self_loops':loops,
        'parallel_endpoint_edges':parallel,'duplicate_typed_edges':typed_duplicates,
        'observed_labels':observed,'missing_labels':256*16-observed,'observed_labels_all_finite':finite,
        'note':'CPU read-only audit of original cache rows; not a GNN forward/backward test.'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--required-ram-gib',type=int,choices=[4,6],default=4);args=parser.parse_args()
    environment=check_environment(args.required_ram_gib)
    fingerprints=fingerprint_audit();batch=batch_audit()
    hardware=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    after=check_environment(args.required_ram_gib)
    passed=fingerprints['all_passed'] and batch['passed']
    reasons=list(dict.fromkeys(environment['blocking_reasons']+after['blocking_reasons']))
    if not passed:reasons.append('fingerprint_or_real_batch_audit_failed')
    report={'status':'deferred' if reasons else 'preflight_passed','GPU_acceptance_passed':False,
        'environment_before':environment,'environment_after':after,'blocking_reasons':reasons,
        'fingerprints':fingerprints,'real_batch256_audit':batch,
        'nvidia_smi':{'returncode':hardware.returncode,'stdout':hardware.stdout.strip(),'stderr':hardware.stderr.strip()},
        'GPU_correctness':{'status':'not_run','max_abs':None,'max_relative':None},
        'GPU_performance':{'status':'not_run','speedup':None,'samples_per_second':None,'kernel_count':None,'sync_count':None},
        'formal_optimized_release_created':False,'formal_training_command_available':False,
        'CUDA_work_started':False,'torch_imported':'torch' in sys.modules,
        'policy':'Preflight only; no automatic testing, deployment, process signals or writes to production.'}
    output=ROOT/'results/gpu_acceptance';output.mkdir(exist_ok=True)
    name='preflight_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.json'
    path=output/name;path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    (output/'latest_preflight.json').write_text(json.dumps({'path':str(path),'status':report['status']},indent=2),encoding='utf-8')
    print(json.dumps({'report':str(path),'status':report['status'],'blocking_reasons':reasons,
        'available_RAM_GiB':after['available_RAM_GiB'],'source_checks_passed':passed,
        'batch':batch,'torch_imported':report['torch_imported'],'CUDA_work_started':False},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
