"""Isolated, memory-guarded, same-real-Batch benchmark; CUDA refuses live training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import ast
import gc
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace,MethodType
import psutil
ROOT=Path(__file__).resolve().parents[1];WORKSPACE=ROOT.parents[1];sys.path.insert(0,str(ROOT))
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['reference','optimized'],required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--repeats',type=int,default=3);parser.add_argument('--profile-forward',action='store_true')
    parser.add_argument('--profile-size',type=int,choices=[16,256],default=16)
    parser.add_argument('--profile-full-step',action='store_true')
    parser.add_argument('--timing',choices=['staged','end_to_end'],default='staged')
    parser.add_argument('--cpu-threads',type=int,default=1);parser.add_argument('--output')
    args=parser.parse_args()
    output=Path(args.output).resolve() if args.output else ROOT/'results'/f'benchmark_{args.device}_{args.variant}_{args.batch_size}.json'
    assert output.is_relative_to(ROOT/'results');output.parent.mkdir(parents=True,exist_ok=True)
    status=read(WORKSPACE/'artifacts/experiment1_fusion_id_v1/status.json');available=psutil.virtual_memory().available
    required_RAM=(6 if args.profile_forward and args.profile_size==256 else 4)*2**30
    if args.device=='cuda':
        from gpu_preflight import check_environment
        environment=check_environment(required_RAM/2**30)
    if args.device=='cuda' and not environment['eligible_for_CUDA']:
        write(output,{'status':'deferred','reason':'Production PID alive or insufficient host RAM (4 GiB; 6 GiB for large trace)',
                      'production':status,'available_RAM_bytes':available,'GPU_speedup':None,'environment':environment})
        print('CUDA deferred; no torch/CUDA imported and no training process touched.');raise SystemExit(75)
    stop=threading.Event();resources=[];phase={'value':'import'};process=psutil.Process()
    def guard():
        while not stop.wait(.02):
            vm=psutil.virtual_memory();resources.append((process.memory_info().rss,vm.available,phase['value']))
            if vm.available<640*2**20:
                write(output,{'status':'aborted_for_memory','phase':phase['value'],'variant':args.variant,
                              'reason':'Only diagnostic exits below 640 MiB available RAM','production_pid':status['pid']})
                os._exit(86)
    threading.Thread(target=guard,daemon=True).start()
    import torch
    import numpy as np
    from reference_src.graph.builder import CircuitBatch as FrozenBatch
    from src.graph.builder import CircuitBatch
    if args.variant=='reference':
        from reference_src.model.network import EdgeCentricRGCN
        from reference_src.train.losses import supervision_loss
        from reference_src.train.loop import TrainingCoverage,LossAccumulator
    else:
        from src.model.network import EdgeCentricRGCN
        from src.train.losses import supervision_loss
        from src.train.loop import TrainingCoverage,LossAccumulator
    torch.set_num_threads(args.cpu_threads);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    if args.device=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.4)
    config=read(WORKSPACE/'config/experiment1_fusion.json');folder=WORKSPACE/config['cache'];manifest=read(folder/'manifest.json')
    # Execute only the frozen, unchanged batch method. Importing its whole module
    # pulls unrelated historical simulation scripts into this isolated process.
    tree=ast.parse((ROOT/'reference_src/train/fusion_cache.py').read_text(encoding='utf-8'))
    cache_class=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CachedFusion')
    method=next(n for n in cache_class.body if isinstance(n,ast.FunctionDef) and n.name=='batch')
    namespace={'np':np,'torch':torch,'CircuitBatch':FrozenBatch}
    exec(compile(ast.Module(body=[method],type_ignores=[]),'frozen_CachedFusion_batch','exec'),namespace)
    probe=SimpleNamespace();probe.batch=MethodType(namespace['batch'],probe)
    probe.manifest=manifest;probe.schema=read(folder/'schema.json');probe.tasks=read(folder/'tasks.json')
    probe.arrays={};probe.static={}
    fixtures=read(WORKSPACE/'reports/experiment1_performance/fixtures/manifest.json')['selections']
    fixture=next(f for f in fixtures if f['size']==args.batch_size)
    references=[tuple(pair) for pair in fixture['references']]
    for name,_ in references:
        if name in probe.arrays:continue
        import numpy as np
        path=WORKSPACE/manifest['topologies'][name]['folder']
        probe.arrays[name]={k:np.load(path/(k+'.npy'),mmap_mode='r') for k in ['values','known','zero','target','mask','group','domain','sample_ids']}
        probe.static[name]=read(path/'structure.json')
    kwargs={k:config[k] for k in ['hidden_dim','num_layers','dropout','update_edges','use_domain_emb','use_film','collapse_relations']}
    model=EdgeCentricRGCN(probe.schema,probe.tasks,**kwargs).to(args.device);model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    coverage=TrainingCoverage();accumulator=LossAccumulator();measurements=[];pending_events={}
    def timed(name,fn):
        phase['value']=name
        gpu=args.device=='cuda' and name not in {'data_and_graph_build'}
        if gpu:
            if args.timing=='staged':torch.cuda.synchronize()
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
        tick=time.perf_counter();value=fn()
        if gpu:
            end.record()
            if args.timing=='staged':end.synchronize()
        result={'wall_ms':1000*(time.perf_counter()-tick)}
        if gpu:
            if args.timing=='staged':result['cuda_event_span_ms']=start.elapsed_time(end)
            else:pending_events[name]=(start,end);result['host_dispatch_ms']=result.pop('wall_ms')
        return value,result
    for iteration in range(args.repeats+1):
        gc.collect()
        if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
        stages={};pending_events.clear()
        if args.device=='cuda':torch.cuda.synchronize()
        tick=time.perf_counter()
        batch,stages['data_and_graph_build']=timed('data_and_graph_build',lambda:probe.batch(references,'train'))
        if args.variant=='optimized':batch=CircuitBatch(**vars(batch))
        batch,stages['prepare_and_transfer']=timed('prepare_and_transfer',lambda:batch.to(args.device))
        _,stages['zero_grad']=timed('zero_grad',lambda:optimizer.zero_grad(set_to_none=True))
        prediction,stages['forward']=timed('forward',lambda:model(batch))
        losses,stages['loss']=timed('loss',lambda:supervision_loss(prediction,batch));loss=losses[0]
        _,stages['backward']=timed('backward',loss.backward)
        norm,stages['gradient_clip']=timed('gradient_clip',lambda:torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True))
        _,stages['optimizer_step']=timed('optimizer_step',optimizer.step)
        _,stages['coverage_statistics']=timed('coverage_statistics',lambda:coverage.observe(batch))
        _,stages['loss_statistics']=timed('loss_statistics',lambda:accumulator.update(prediction.detach(),batch))
        float(norm)
        if args.device=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-tick
        for name,(start,end) in pending_events.items():stages[name]['cuda_event_span_ms']=start.elapsed_time(end)
        if iteration:
            measurements.append({'iteration':iteration,'wall_seconds':elapsed,'samples_per_second':args.batch_size/elapsed,'stages':stages,
                'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda' else None})
            write(output,{'status':'partial','measurements':measurements})
            print(args.variant,args.device,args.batch_size,iteration,round(elapsed,4),'s',flush=True)
        del batch,prediction,losses,loss,norm;gc.collect()
    cpu_ops=None;trace_file=None
    if args.profile_forward:
        small=next(f for f in fixtures if f['size']==args.profile_size)
        small_fields=torch.load(WORKSPACE/'reports/experiment1_performance/fixtures'/small['file'],map_location='cpu',weights_only=True)
        if args.variant=='reference':
            from reference_src.graph.builder import CircuitBatch as Batch
        else:Batch=CircuitBatch
        small_batch=Batch(**small_fields).to(args.device)
        if args.device=='cuda':torch.cuda.synchronize()
        phase['value']='operator_profile_'+str(args.profile_size)
        activities=[torch.profiler.ProfilerActivity.CPU]
        if args.device=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities,record_shapes=False,profile_memory=False) as profiler:
            if args.profile_full_step:optimizer.zero_grad(set_to_none=True)
            prediction=model(small_batch)
            if args.profile_full_step:
                pair=supervision_loss(prediction,small_batch);pair[0].backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                optimizer.step();coverage.observe(small_batch);accumulator.update(prediction.detach(),small_batch);float(norm)
                del pair,norm
            if args.device=='cuda':torch.cuda.synchronize()
        if args.device=='cpu':
            cpu_ops={item.key:{'calls':item.count,'inclusive_cpu_ms':item.cpu_time_total/1000} for item in profiler.key_averages()}
        else:
            trace_file=str(ROOT/'results'/f'trace_cuda_{args.variant}_{args.profile_size}.json');profiler.export_chrome_trace(trace_file)
        del profiler,prediction,small_batch;gc.collect()
    stop.set()
    result={'status':'complete','variant':args.variant,'device':args.device,'torch':torch.__version__,'cpu_threads':args.cpu_threads,
        'timing_mode':args.timing,'end_to_end_sync_policy':'Whole-step synchronization before/after; per-stage Events do not synchronize in end_to_end mode. Event spans include launch gaps.',
        'batch_size':args.batch_size,'warmup_steps':1,'repeats':args.repeats,'seed':42,'deterministic':True,
        'model_parameters':sum(p.numel() for p in model.parameters()),'measurements':measurements,
        'sample_ids':[probe.arrays[n]['sample_ids'][r].decode('utf-8') for n,r in references],
        'CPU_RSS_peak_bytes':max(r[0] for r in resources),'minimum_available_RAM_bytes':min(r[1] for r in resources),
        'CPU_forward_ops_batch16':cpu_ops,'CUDA_trace_batch16':trace_file,
        'operator_profile_size':args.profile_size,'operator_profile_scope':'full_training_step' if args.profile_full_step else 'forward',
        'timing_policy':'Same real batch; build + execution-plan preparation + transfer + complete optimizer/statistics step. CUDA stages synchronize Events. Shared machine, warm-cache data.',
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/('reference_src' if args.variant=='reference' else 'src')).rglob('*.py')},
        'production_before':status,'production_after':read(WORKSPACE/'artifacts/experiment1_fusion_id_v1/status.json')}
    write(output,result);print('Benchmark complete:',output)

if __name__=='__main__':main()
