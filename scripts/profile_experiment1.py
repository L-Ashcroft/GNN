"""Isolated real-batch profiling; never writes to the running experiment."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from collections import Counter
from datetime import datetime,timezone
import gc
import hashlib
import ctypes
import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read

def sha(path):
    with open(path,'rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(temporary,path)

OUT=ROOT/'reports/experiment1_performance'
PRODUCTION=ROOT/'artifacts/experiment1_fusion_id_v1'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    before=read(PRODUCTION/'status.json');recipe=read(PRODUCTION/'recipe.json')
    protected={p:sha(ROOT/p) for p in recipe['implementation_sha256']}
    protected['config/experiment1_fusion.json']=sha(ROOT/'config/experiment1_fusion.json')
    protected['artifacts/experiment1_fusion_id_v1/recipe.json']=sha(PRODUCTION/'recipe.json')
    atomic_json(OUT/'control_before.json',{'status':before,'protected_sha256':protected})
    if psutil.virtual_memory().available<1.1*2**30:raise RuntimeError('Insufficient RAM for a nonintrusive separate CUDA process')
    stopping=threading.Event();resource=[];phase={'name':'importing'};own_trims=[]
    def trim_own_working_set():
        # Windows releases only this diagnostic process's pageable working set.
        # Never opens, suspends, trims, or signals the production process.
        result=ctypes.windll.kernel32.SetProcessWorkingSetSize(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.c_size_t(-1),ctypes.c_size_t(-1))
        own_trims.append({'t':time.time(),'phase':phase['name'],'success':bool(result)})
    def guard():
        while not stopping.wait(.25):
            available=psutil.virtual_memory().available
            resource.append({'t':time.time(),'available_RAM_bytes':available,'phase':phase['name']})
            if available<640*2**20:
                if not own_trims or time.time()-own_trims[-1]['t']>5:
                    trim_own_working_set();available=psutil.virtual_memory().available
                if available>=640*2**20:continue
                atomic_json(OUT/'resource_abort.json',{'reason':'Diagnostic process exited to protect live training',
                    'available_RAM_bytes':available,'phase':phase['name'],'production_pid':before['pid'],
                    'diagnostic_RSS_bytes':psutil.Process().memory_info().rss})
                os._exit(86)
    watcher=threading.Thread(target=guard,daemon=True);watcher.start()
    import torch
    from torch.profiler import profile,ProfilerActivity,record_function
    # Same attributes and .to behavior as CircuitBatch, without importing the
    # dataset/parser dependency tree into this memory-limited CUDA process.
    class CircuitBatch(SimpleNamespace):
        @property
        def num_graphs(self):return len(self.group_ids)
        def to(self,device):
            return CircuitBatch(**{k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in vars(self).items()})
    from src.model.network import EdgeCentricRGCN
    from src.train.losses import supervision_loss
    from src.train.loop import TrainingCoverage,LossAccumulator
    torch.set_num_threads(2);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(.25);torch.backends.cuda.matmul.allow_tf32=False
    config=recipe['config'];cache=ROOT/config['cache'];manifest=read(cache/'manifest.json')
    schema=read(cache/'schema.json');tasks=read(cache/'tasks.json')
    own=psutil.Process();gpu_samples=[]
    def gpu_watch():
        while not stopping.is_set():
            proc=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
            gpu_samples.append({'t':time.time(),'phase':phase['name'],'raw':proc.stdout.strip()})
            stopping.wait(.5)
    gpu_thread=threading.Thread(target=gpu_watch,daemon=True);gpu_thread.start()
    phase['name']='baseline_before';time.sleep(5)
    open_started=time.perf_counter();probe=SimpleNamespace()
    probe.config=config;probe.path=cache;probe.manifest=manifest;probe.schema=schema;probe.tasks=tasks;probe.arrays={};probe.static={}
    fixture_manifest=read(OUT/'fixtures/manifest.json')
    assert fixture_manifest['config_sha256']==sha(ROOT/'config/experiment1_fusion.json')
    assert fixture_manifest['cache_manifest_sha256']==sha(cache/'manifest.json')
    selected={16:[],256:[]};fixtures={}
    for item in fixture_manifest['selections']:
        rows=[tuple(pair) for pair in item['references']];selected[item['size']].append(rows)
        fixtures[tuple(rows)]=item
    needed={n for batches in selected.values() for batch in batches for n,_ in batch}
    for name in needed:
        folder=ROOT/manifest['topologies'][name]['folder']
        probe.arrays[name]={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in
            ['values','known','zero','target','mask','group','domain','sample_ids']}
        probe.static[name]=read(folder/'structure.json')
    open_seconds=time.perf_counter()-open_started
    samples=[];cuda_stage={'h2d','forward','loss','backward','gradient_clip','optimizer_step','coverage_statistics','loss_statistics'}
    def timed(name,fn):
        phase['name']=name
        if name in cuda_stage:
            torch.cuda.synchronize();start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True)
            start_event.record()
        start=time.perf_counter();cpu_start=time.process_time()
        with record_function('stage::'+name):value=fn()
        if name in cuda_stage:end_event.record();end_event.synchronize()
        result={'wall_ms':(time.perf_counter()-start)*1000,'process_cpu_ms':(time.process_time()-cpu_start)*1000}
        if name in cuda_stage:result['cuda_stream_span_ms']=start_event.elapsed_time(end_event)
        atomic_json(OUT/'last_stage.json',{'phase':name,'timing':result,'available_RAM_bytes':psutil.virtual_memory().available})
        return value,result
    def load(rows):
        loaded=[]
        for name,index in rows:
            a=probe.arrays[name]
            loaded.append((name,{k:np.array(a[k][index],copy=True) for k in
                ['values','known','zero','target','mask','group','domain']},a['sample_ids'][index].decode('utf-8')))
        return loaded
    def build(loaded):
        x=[];src=[];dst=[];attrs=[];types=[];gi=[];targets=[];masks=[];scales=[];groups=[];semantics=[];domains=[];ids=[];offset=0
        for i,(name,a,sid) in enumerate(loaded):
            s=probe.static[name];d=np.concatenate([a['values'],a['known'],a['zero']],axis=1)
            x.extend(s['x']);gi.extend([i]*len(s['x']));src.extend(n+offset for n in s['edge_index'][0]);dst.extend(n+offset for n in s['edge_index'][1])
            attrs.append(d[np.asarray(s['device_index'])]);types.extend(s['edge_type']);offset+=len(s['x'])
            group=manifest['group_names'][int(a['group'])];task=tasks[group];groups.append(group)
            targets.append(a['target']);masks.append(a['mask']);domains.append(int(a['domain']));ids.append(sid)
            scales.append([task['slots'].get(str(j),{}).get('scale',1.) for j in range(16)])
            semantics.append([task['slots'].get(str(j),{}).get('semantic_id') for j in range(16)])
        return CircuitBatch(x=torch.tensor(x,dtype=torch.float32),edge_index=torch.tensor([src,dst]),
            edge_attr=torch.from_numpy(np.concatenate(attrs).astype(np.float32)),edge_type=torch.tensor(types),graph_index=torch.tensor(gi),
            targets=torch.from_numpy(np.stack(targets)),mask=torch.from_numpy(np.stack(masks)),scales=torch.tensor(scales),
            group_ids=groups,semantic_ids=semantics,sample_ids=ids,domain_ids=torch.tensor(domains),split_role='train')
    model=EdgeCentricRGCN(schema,tasks,**{k:config[k] for k in
        ['hidden_dim','num_layers','dropout','update_edges','use_domain_emb','use_film','collapse_relations']}).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    trim_own_working_set()
    coverage=TrainingCoverage();accumulator=LossAccumulator()
    def run(rows,save=True):
        result={};started=time.perf_counter()
        loaded,result['data_loading']=timed('data_loading',lambda:load(rows))
        batch,result['graph_tensor_build']=timed('graph_tensor_build',lambda:build(loaded))
        if save:
            fixture=fixtures[tuple(rows)];path=OUT/'fixtures'/fixture['file'];assert sha(path)==fixture['sha256']
            original=torch.load(path,map_location='cpu',weights_only=True)
            for key in ['x','edge_index','edge_type','edge_attr','graph_index','targets','mask','scales','domain_ids']:
                torch.testing.assert_close(getattr(batch,key),original[key],rtol=0,atol=0)
            assert batch.sample_ids==original['sample_ids'];del original
        gpu,result['h2d']=timed('h2d',lambda:batch.to('cuda'))
        _,result['zero_grad']=timed('zero_grad',lambda:optimizer.zero_grad(set_to_none=True))
        prediction,result['forward']=timed('forward',lambda:model(gpu))
        pair,result['loss']=timed('loss',lambda:supervision_loss(prediction,gpu));loss=pair[0]
        _,result['backward']=timed('backward',loss.backward)
        norm,result['gradient_clip']=timed('gradient_clip',lambda:torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True))
        _,result['optimizer_step']=timed('optimizer_step',optimizer.step)
        _,result['coverage_statistics']=timed('coverage_statistics',lambda:coverage.observe(gpu))
        _,result['loss_statistics']=timed('loss_statistics',lambda:accumulator.update(prediction.detach(),gpu))
        start=time.perf_counter();float(norm);scalar_ms=(time.perf_counter()-start)*1000
        if save:
            counts=Counter(batch.edge_type.tolist());active=len(counts);b=batch.num_graphs
            sample={'batch_size':b,'nodes':len(batch.x),'message_edges':len(batch.edge_type),'unique_tasks':len(set(batch.group_ids)),
                'active_relations':active,'edges_per_relation':dict(counts),'stages':result,'scalar_read_ms':scalar_ms,
                'staged_wall_ms':sum(v['wall_ms'] for v in result.values()),'instrumented_total_wall_ms':(time.perf_counter()-started)*1000,
                'layer_relation_nonzero_calls':4*48,'pool_nonzero_calls':2*b,'coverage_nonzero_calls':b,
                'relation_index_add_forward_calls':4*active,'sample_ids':batch.sample_ids}
            samples.append(sample);atomic_json(OUT/'stage_measurements_partial.json',samples)
            print('Measured batch',b,'sum seconds',sample['staged_wall_ms']/1000,flush=True)
        del prediction,pair,loss,norm,gpu,batch,loaded;gc.collect();torch.cuda.empty_cache()
    phase['name']='warmup';run(selected[16][0],save=False)
    if not args.profile_only:
        for rows in selected[16]+selected[256]:run(rows)
    else:samples=read(OUT/'stage_measurements_partial.json')
    # Profile a small batch only; no instrumentation is attached to the live process.
    phase['name']='operator_profile_16';activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]
    hook_handles=[];scope_stack={}
    def attach_scope(module,name):
        def enter(_module,_args):
            scope=record_function('module::'+name);scope.__enter__();scope_stack[name]=scope
        def leave(_module,_args,_output):scope_stack.pop(name).__exit__(None,None,None)
        hook_handles.extend([module.register_forward_pre_hook(enter),module.register_forward_hook(leave)])
    for i,layer in enumerate(model.layers):attach_scope(layer,f'RGCN_layer_{i+1}')
    attach_scope(model.pool,'node_attention_pool');attach_scope(model.edge_pool,'edge_attention_pool');attach_scope(model.head,'semantic_head')
    # Profile forward only to bound Kineto event-buffer RAM. All training stages
    # were already measured with end-event synchronization in separate batches.
    prof_batch=build(load(selected[16][0])).to('cuda');torch.cuda.synchronize()
    with profile(activities=activities,record_shapes=False,profile_memory=False,with_stack=False) as prof:
        with record_function('stage::forward_profile_only'):
            prof_prediction=model(prof_batch)
        torch.cuda.synchronize()
    del prof_prediction,prof_batch
    for handle in hook_handles:handle.remove()
    prof.export_chrome_trace(str(OUT/'trace_batch16.json'))
    # Kineto key_averages/events materialization can allocate hundreds of MiB.
    # Parse the exported trace in a later CPU-only process instead.
    del prof;gc.collect()
    phase['name']='baseline_after';time.sleep(4);stopping.set();watcher.join(1);gpu_thread.join(2)
    after=read(PRODUCTION/'status.json')
    for path,value in protected.items():assert sha(ROOT/path)==value,'Protected source/config changed'
    assert before['pid']==after['pid'] and psutil.pid_exists(before['pid'])
    report={'created_at_UTC':datetime.now(timezone.utc).isoformat(),'torch':torch.__version__,
        'GPU':torch.cuda.get_device_name(0),'cpu_threads':2,'deterministic':torch.are_deterministic_algorithms_enabled(),
        'fresh_independent_model':True,'formal_checkpoint_loaded':False,'shared_GPU_with_live_training':True,
        'one_time_cache_index_and_open_seconds':open_seconds,'measurements':samples,
        'profiled_batch_size':16,'operator_statistics':'CPU-only summary of trace_batch16.json',
        'operator_profile_scope':'Forward only; separate synchronized stage measurements include full backward/optimizer/statistics',
        'GPU_samples':gpu_samples,'memory_samples':resource,'diagnostic_peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
        'diagnostic_only_working_set_trims':own_trims,
        'production_before':before,'production_after':after,'protected_source_sha256':protected,
        'production_same_pid_alive':True,'production_configuration_unchanged':True,'no_production_files_written':True,
        'timing_policy':'CPU perf_counter; every GPU stage uses CUDA Events and synchronizes its end event. Stream span includes CPU launch gaps and shared-device scheduling; not summed kernel busy time.',
        'comparison_policy':'Stage synchronization perturbs overlap; independent process contends briefly with live training. No claim of exclusive-device throughput.'}
    atomic_json(OUT/'profile_run.json',report)
    print('Profile complete; production remains alive:',after['pid'],'batch',after['batch'],flush=True)


if __name__=='__main__':main()
