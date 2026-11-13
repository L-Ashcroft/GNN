"""CPU-only trace reduction; never imports torch or touches live training state."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import psutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/experiment1_performance'

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def main():
    trace=read(OUT/'trace_batch16.json');events=trace['traceEvents']
    counts=Counter();inclusive=Counter();exclusive=Counter();runtime_count=Counter();runtime_us=Counter()
    external={};threads=defaultdict(list);scalar_parents=Counter();scalar_ancestors=Counter();sync_parents=Counter();annotations=[]
    for event in events:
        category=event.get('cat');duration=event.get('dur',0.)
        if category=='cpu_op':
            counts[event['name']]+=1;inclusive[event['name']]+=duration
            external[event.get('args',{}).get('External id')]=event['name']
        if category=='cuda_runtime':runtime_count[event['name']]+=1;runtime_us[event['name']]+=duration
        if category=='user_annotation':annotations.append({'name':event['name'],'cpu_wall_ms':duration/1000})
        if category in {'cpu_op','cuda_runtime','user_annotation'} and event.get('ph')=='X':
            threads[(event['pid'],event['tid'])].append(event)
    # CPU self wall time excludes nested CPU operators and CUDA API intervals.
    # This is wall time (may include waits), not CPU process-time utilization.
    for stream in threads.values():
        stream.sort(key=lambda e:(e['ts'],-e['dur']));stack=[]
        for event in stream:
            end=event['ts']+event['dur']
            while stack and event['ts']>=stack[-1]['end']-0.01:
                finished=stack.pop()
                if finished['event'].get('cat')=='cpu_op':exclusive[finished['event']['name']]+=max(0.,finished['event']['dur']-finished['children'])
            if stack and end<=stack[-1]['end']+0.1:
                stack[-1]['children']+=event['dur']
                if event['name']=='aten::_local_scalar_dense':
                    parent=next((s['event']['name'] for s in reversed(stack) if s['event'].get('cat')=='cpu_op'),'none')
                    scalar_parents[parent]+=1
                    ancestor=next((s['event']['name'] for s in reversed(stack) if s['event'].get('cat')=='cpu_op' and s['event']['name']!='aten::item'),'none')
                    scalar_ancestors[ancestor]+=1
                if event['name']=='cudaStreamSynchronize':
                    parent=next((s['event']['name'] for s in reversed(stack) if s['event'].get('cat')=='cpu_op'),'none')
                    sync_parents[parent]+=1
            stack.append({'event':event,'end':end,'children':0.})
        for finished in stack:
            if finished['event'].get('cat')=='cpu_op':exclusive[finished['event']['name']]+=max(0.,finished['event']['dur']-finished['children'])
    intervals=[];device_us=Counter();device_calls=Counter();kernel_names=Counter();kernel_count=0;copies=Counter()
    for event in events:
        if event.get('cat') in {'kernel','gpu_memcpy','gpu_memset'} and event.get('ph')=='X':
            intervals.append((event['ts'],event['ts']+event['dur']))
        if event.get('cat')=='kernel':
            kernel_count+=1;kernel_names[event['name']]+=event['dur']
            op=external.get(event.get('args',{}).get('External id'),'unmapped')
            device_us[op]+=event['dur'];device_calls[op]+=1
        if event.get('cat')=='gpu_memcpy':copies[event['name']]+=1
    intervals.sort();merged=[]
    for start,end in intervals:
        if merged and start<=merged[-1][1]:merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
        else:merged.append((start,end))
    union=sum(b-a for a,b in merged);span=merged[-1][1]-merged[0][0]
    operations=[{'name':name,'calls':counts[name],'inclusive_cpu_wall_ms':inclusive[name]/1000,
                 'self_cpu_wall_ms':exclusive[name]/1000,'direct_kernel_ms':device_us[name]/1000,
                 'direct_kernel_count':device_calls[name]} for name in counts]
    samples=read(OUT/'stage_measurements_partial.json');large=[s for s in samples if s['batch_size']==256]
    mean_total=statistics.mean(s['staged_wall_ms'] for s in large)
    stage_summary={}
    for stage in large[0]['stages']:
        values=[s['stages'][stage]['wall_ms'] for s in large]
        stage_summary[stage]={'mean_wall_ms':statistics.mean(values),'min_wall_ms':min(values),'max_wall_ms':max(values),
            'share_percent':100*statistics.mean(values)/mean_total,
            'mean_process_cpu_ms':statistics.mean(s['stages'][stage]['process_cpu_ms'] for s in large)}
        if 'cuda_stream_span_ms' in large[0]['stages'][stage]:
            stage_summary[stage]['mean_cuda_stream_span_ms']=statistics.mean(s['stages'][stage]['cuda_stream_span_ms'] for s in large)
    control=read(OUT/'control_before.json');recipe=read(ROOT/'artifacts/experiment1_fusion_id_v1/recipe.json')
    after=read(ROOT/'artifacts/experiment1_fusion_id_v1/status.json')
    assert psutil.pid_exists(after['pid']) and after['pid']==control['status']['pid']
    for path,expected in control['protected_sha256'].items():assert sha(ROOT/path)==expected,path
    for path,expected in recipe['implementation_sha256'].items():assert sha(ROOT/path)==expected,path
    fixture_manifest=read(OUT/'fixtures/manifest.json')
    for fixture in fixture_manifest['selections']:assert sha(OUT/'fixtures'/fixture['file'])==fixture['sha256']
    run_path=OUT/'profile_run.json';run=read(run_path) if run_path.exists() else None
    result={'created_at_UTC':datetime.now(timezone.utc).isoformat(),
        'measurements':samples,'batch256_count':len(large),'batch256_mean_staged_wall_ms':mean_total,
        'stage_summary':stage_summary,'forward_batch16':{'scope':'forward only, profiler overhead included',
            'CPU_annotations':annotations,'operations':operations,
            'CUDA_runtime':[{'name':name,'calls':runtime_count[name],'wall_ms':runtime_us[name]/1000} for name in runtime_count],
            'local_scalar_cpu_parents':dict(scalar_parents),'local_scalar_ancestors':dict(scalar_ancestors),
            'stream_synchronize_parent_ops':dict(sync_parents),'kernel_count':kernel_count,
            'kernel_sum_ms':sum(kernel_names.values())/1000,'device_activity_union_ms':union/1000,
            'device_activity_span_ms':span/1000,'own_device_activity_coverage_percent':union/span*100,
            'copies_by_kind':dict(copies),'top_kernels':kernel_names.most_common(20)},
        'trace_sha256':sha(OUT/'trace_batch16.json'),'stage_measurements_sha256':sha(OUT/'stage_measurements_partial.json'),
        'production_before':control['status'],'production_after':after,'production_same_pid_alive':True,
        'protected_source_config_hashes_unchanged':True,'matches_running_recipe':True,
        'diagnostic_run_metadata':run,
        'limits':['Shared GPU; diagnostic process briefly competes with live training.',
            'Stage end-event synchronization changes overlap; stage percentages are not exact production percentages.',
            'Only warm-cache data access measured; no forced cache eviction.',
            'Three full batches, one small forward trace; no epoch-level confidence interval.',
            'CUDA Event stream spans include launch gaps and scheduling; device activity coverage is only this diagnostic process.',
            'CPU operator inclusive times are nested and cannot be summed; self wall times exclude nested CPU/API intervals.',
            'Host RAM guard aborted prior diagnostic attempts; completed stage records and valid exported trace retained.']}
    write(OUT/'profile.json',result)
    print('Stages:',json.dumps(stage_summary,indent=2))
    print('Device coverage:',result['forward_batch16']['own_device_activity_coverage_percent'])
    print('Scalar parents:',scalar_parents)
    print('Verified live production PID',after['pid'],'batch',after.get('batch'),'state',after['state'])

if __name__=='__main__':main()
