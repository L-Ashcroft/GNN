"""CPU-only evidence summary and untouched-production verification."""
import hashlib
import json
from pathlib import Path
import statistics
import psutil

ROOT=Path(__file__).resolve().parents[1];WORKSPACE=ROOT.parents[1]
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    snapshot=read(ROOT/'snapshot.json');changed=[]
    for path,expected in snapshot['source_files'].items():
        frozen=ROOT/'reference_src'/Path(path).relative_to('src')
        assert digest(frozen)==expected,'Frozen reference changed: '+path
        assert digest(WORKSPACE/path)==expected,'Production source changed: '+path
        if digest(ROOT/path)!=expected:changed.append(path)
    for path,expected in snapshot['protected'].items():assert digest(WORKSPACE/path)==expected,path
    correct=read(ROOT/'results/correctness_4_final.json')
    for path,expected in correct['optimized_source_sha256'].items():assert digest(ROOT/path)==expected,path
    test=read(ROOT/'results/edge_case_tests.json');assert test['success'] and test['tests_run']>=11
    benchmarks={}
    for name in ['reference','optimized']:
        result=read(ROOT/'results'/f'benchmark_cpu_{name}_16.json');assert result['status']=='complete'
        records=result['measurements'];times=[r['wall_seconds'] for r in records]
        benchmarks[name]={'mean_step_seconds':statistics.mean(times),'range_seconds':[min(times),max(times)],
            'throughput_samples_per_second':16/statistics.mean(times),'CPU_RSS_peak_MiB':result['CPU_RSS_peak_bytes']/2**20,
            'mean_stage_ms':{k:statistics.mean(r['stages'][k]['wall_ms'] for r in records) for k in records[0]['stages']},
            'forward_operator_counts':{k:result['CPU_forward_ops_batch16'].get(k,{'calls':0})['calls'] for k in
                ['aten::nonzero','aten::bincount','aten::_local_scalar_dense','aten::mm','aten::addmm','aten::index_add_']}}
    old=read(ROOT/'results/benchmark_cpu_reference_16.json');new=read(ROOT/'results/benchmark_cpu_optimized_16.json')
    assert old['sample_ids']==new['sample_ids']
    # Optimized benchmark must refer to the final validated implementation.
    for path,expected in new['source_sha256'].items():assert digest(ROOT/path)==expected,path
    status=read(WORKSPACE/'artifacts/experiment1_fusion_id_v1/status.json')
    assert status['pid']==snapshot['production_before']['pid'] and psutil.pid_exists(status['pid'])
    assert psutil.Process(status['pid']).create_time()==snapshot['pid_create_time']
    summary={'changed_files':changed+['src/model/execution.py'],'CPU_batch16':benchmarks,
        'CPU_batch16_measured_speedup':benchmarks['reference']['mean_step_seconds']/benchmarks['optimized']['mean_step_seconds'],
        'correctness_maxima':correct['maxima'],'edge_case_tests':test,
        'production_sources_and_config_unchanged':True,'frozen_reference_unchanged':True,
        'production_before':snapshot['production_before'],'production_after':status,'production_same_process_alive':True,
        'CPU_batch256':{name:read(ROOT/'results'/f'benchmark_cpu_{name}_256.json') for name in ['reference','optimized']},
        'GPU_batch256':{name:read(ROOT/'results'/f'benchmark_cuda_{name}_256.json') for name in ['reference','optimized']},
        'GPU_speedup':None,'GPU_peak_memory':None,'GPU_kernel_count':None,'GPU_sync_count':None,
        'acceptance':'CPU small-real-batch equivalence passed; CUDA and full Batch=256 acceptance deferred for memory/live-process safety.'}
    (ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({'CPU_batch16':benchmarks,'CPU_speedup':summary['CPU_batch16_measured_speedup'],
                      'live_training_epoch_batch':[status['epoch'],status['batch']],'changed_files':summary['changed_files']},indent=2))

if __name__=='__main__':main()
