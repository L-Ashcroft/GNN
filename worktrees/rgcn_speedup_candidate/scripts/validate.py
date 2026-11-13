"""Real-data CPU equivalence checks against the frozen production snapshot."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import psutil

ROOT=Path(__file__).resolve().parents[1];WORKSPACE=ROOT.parents[1];sys.path.insert(0,str(ROOT))

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def guard_memory(stop):
    while not stop.wait(.25):
        if psutil.virtual_memory().available<640*2**20:
            write(ROOT/'results/memory_abort.json',{'pid':os.getpid(),'action':'only diagnostic exits','time':time.time()})
            os._exit(86)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--batch-size',type=int,choices=[16,256],default=16)
    parser.add_argument('--checkpoint',help='Read-only parent checkpoint for migration equivalence')
    args=parser.parse_args()
    if args.device=='cuda':
        from gpu_preflight import check_environment
        environment=check_environment(4)
        if not environment['eligible_for_CUDA']:
            write(ROOT/'results'/('correctness_'+args.stage+'.json'),{'status':'deferred','device':'cuda',
                'reason':'Process-creation-time or RAM gate; no torch/CUDA imported','environment':environment})
            print('CUDA correctness deferred safely; not an acceptance pass.');raise SystemExit(75)
    stop=threading.Event();threading.Thread(target=guard_memory,args=(stop,),daemon=True).start()
    import torch
    from reference_src.model.network import EdgeCentricRGCN as Reference
    from src.model.network import EdgeCentricRGCN as Optimized
    from reference_src.train.losses import supervision_loss as ref_loss
    from src.train.losses import supervision_loss as opt_loss
    from reference_src.train.loop import TrainingCoverage as RefCoverage,LossAccumulator as RefAccumulator
    from src.train.loop import TrainingCoverage as OptCoverage,LossAccumulator as OptAccumulator
    from src.model.execution import move_batch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if args.device=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.8)
    class Batch(SimpleNamespace):
        @property
        def num_graphs(self):return len(self.group_ids)
        def to(self,device):return move_batch(self,device)
    config=read(WORKSPACE/'config/experiment1_fusion.json');cache=WORKSPACE/config['cache']
    schema=read(cache/'schema.json');tasks=read(cache/'tasks.json')
    parent=None;parent_sha=None
    if args.checkpoint:
        parent_path=Path(args.checkpoint).resolve()
        assert parent_path.is_relative_to(WORKSPACE)
        with parent_path.open('rb') as stream:parent_sha=hashlib.file_digest(stream,'sha256').hexdigest()
        parent=torch.load(parent_path,map_location='cpu',weights_only=True)
        recipe=read(WORKSPACE/'artifacts/experiment1_fusion_id_v1/recipe.json')
        expected=hashlib.sha256(json.dumps(recipe,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        assert parent['recipe_digest']==expected,'Parent checkpoint recipe mismatch'
    fixture_name='batch_16_7.pt' if args.batch_size==16 else 'batch_256_11.pt'
    fixture=WORKSPACE/'reports/experiment1_performance/fixtures'/fixture_name
    batch=Batch(**torch.load(fixture,map_location='cpu',weights_only=True))
    pairs=list(zip(*batch.edge_index.tolist()));triples=list(zip(*batch.edge_index.tolist(),batch.edge_type.tolist()))
    coverage={'relations':len(set(batch.edge_type.tolist())),'self_loops':sum(a==b for a,b in pairs),
        'parallel_endpoint_edges':len(pairs)-len(set(pairs)),'duplicate_typed_edges':len(triples)-len(set(triples)),
        'missing_labels':int((~batch.mask).sum()),'supervision_contracts':len(set(batch.group_ids))}
    assert coverage['relations']==48 and coverage['self_loops'] and coverage['parallel_endpoint_edges'] and coverage['missing_labels'] and coverage['supervision_contracts']>1
    batch=batch.to(args.device)
    limits={'states':(5e-6,5e-5),'loss':(2e-6,2e-5),'gradient':(2e-5,2e-3),'adamw':(2e-5,2e-4),'statistics':(2e-5,2e-5)}
    results=[]
    def compare(name,a,b,kind):
        a=a.detach().double();b=b.detach().double();assert a.shape==b.shape,name
        assert torch.isfinite(a).all() and torch.isfinite(b).all(),name
        absolute=(a-b).abs();denom=torch.maximum(a.abs(),b.abs()).clamp_min(1e-8)
        atol,rtol=limits[kind]
        item={'name':name,'kind':kind,'max_abs':absolute.max().item() if a.numel() else 0.,
              'max_relative_floor_1e8':(absolute/denom).max().item() if a.numel() else 0.,
              'acceptance_ratio':(absolute/(atol+rtol*a.abs())).max().item() if a.numel() else 0.}
        results.append(item)
        if item['acceptance_ratio']>1:
            write(ROOT/'results'/('failure_'+args.stage+'.json'),{'failed':item,'comparisons':results,'limits':limits})
            raise AssertionError(item)
    cases=[]
    for mode in ['eval','train']:
        torch.manual_seed(20261030)
        kwargs={k:config[k] for k in ['hidden_dim','num_layers','dropout','update_edges','use_domain_emb','use_film','collapse_relations']}
        left=Reference(schema,tasks,**kwargs).to(args.device);right=Optimized(schema,tasks,**kwargs).to(args.device)
        assert list(left.state_dict())==list(right.state_dict())
        right.load_state_dict(left.state_dict(),strict=True);left.load_state_dict(right.state_dict(),strict=True)
        if parent is not None:
            left.load_state_dict(parent['model'],strict=True);right.load_state_dict(parent['model'],strict=True)
        left.train(mode=='train');right.train(mode=='train')
        left_opt=torch.optim.AdamW(left.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        right_opt=torch.optim.AdamW(right.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        if parent is not None:
            left_opt.load_state_dict(copy.deepcopy(parent['optimizer']));right_opt.load_state_dict(copy.deepcopy(parent['optimizer']))
            torch.set_rng_state(parent['rng_cpu'].cpu())
            if args.device=='cuda':torch.cuda.set_rng_state_all([value.cpu() for value in parent['rng_cuda']])
        left_capture={};right_capture={};handles=[]
        def hook(destination,name):
            def capture(_module,_args,output):
                destination[name]=tuple(v.detach().clone() for v in output)
            return capture
        for model,destination in [(left,left_capture),(right,right_capture)]:
            for i,layer in enumerate(model.layers):handles.append(layer.register_forward_hook(hook(destination,'layer'+str(i))))
            handles.append(model.pool.register_forward_hook(hook(destination,'node_pool')))
            handles.append(model.edge_pool.register_forward_hook(hook(destination,'edge_pool')))
        rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if args.device=='cuda' else None
        a=left(batch,True);left_rng=torch.get_rng_state();left_cuda_rng=torch.cuda.get_rng_state_all() if args.device=='cuda' else None
        torch.set_rng_state(rng)
        if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
        b=right(batch,True);right_rng=torch.get_rng_state()
        assert torch.equal(left_rng,right_rng),'Dropout RNG consumption changed'
        if cuda_rng is not None:assert all(torch.equal(a,b) for a,b in zip(left_cuda_rng,torch.cuda.get_rng_state_all()))
        for key in a:compare(mode+'/'+key,a[key],b[key],'states')
        for key in left_capture:
            for i,(u,v) in enumerate(zip(left_capture[key],right_capture[key])):compare(mode+'/'+key+'/'+str(i),u,v,'states')
        la,ga=ref_loss(a['prediction'],batch);lb,gb=opt_loss(b['prediction'],batch)
        assert list(ga)==list(gb)
        compare(mode+'/total_loss',la,lb,'loss')
        for group in ga:compare(mode+'/group_loss/'+group,ga[group],gb[group],'loss')
        la.backward();lb.backward();none=0
        for (name,p),(other,q) in zip(left.named_parameters(),right.named_parameters()):
            assert name==other and (p.grad is None)==(q.grad is None),name
            if p.grad is None:none+=1
            else:compare(mode+'/gradient/'+name,p.grad,q.grad,'gradient')
        n1=torch.nn.utils.clip_grad_norm_(left.parameters(),5.,error_if_nonfinite=True)
        n2=torch.nn.utils.clip_grad_norm_(right.parameters(),5.,error_if_nonfinite=True)
        compare(mode+'/gradient_norm',n1,n2,'gradient')
        left_opt.step();right_opt.step()
        for (name,p),(other,q) in zip(left.named_parameters(),right.named_parameters()):compare(mode+'/adamw/'+name,p,q,'adamw')
        c1=RefCoverage();c2=OptCoverage();c1.observe(batch);c2.observe(batch)
        assert c1.slots==c2.slots and c1.domains==c2.domains
        c1.check(batch,True);c2.check(batch,True)
        s1=RefAccumulator();s2=OptAccumulator();s1.update(a['prediction'].detach(),batch);s2.update(b['prediction'].detach(),batch)
        assert set(s1.values)==set(s2.values)
        for group in s1.values:
            assert s1.values[group][1]==s2.values[group][1]
            compare(mode+'/statistics/'+group,torch.tensor(s1.values[group][0],dtype=torch.float64),torch.tensor(s2.values[group][0],dtype=torch.float64),'statistics')
        compare(mode+'/statistics_mean',torch.tensor(s1.mean()),torch.tensor(s2.mean()),'statistics')
        cases.append({'mode':mode,'none_gradients':none,'dropout_rng_identical':True,'checkpoint_strict_bidirectional':True})
        for handle in handles:handle.remove()
        del left,right,left_opt,right_opt,a,b,la,lb,ga,gb,left_capture,right_capture;gc.collect()
    maxima={kind:{'max_abs':max(v['max_abs'] for v in results if v['kind']==kind),
                  'max_relative_floor_1e8':max(v['max_relative_floor_1e8'] for v in results if v['kind']==kind),
                  'max_acceptance_ratio':max(v['acceptance_ratio'] for v in results if v['kind']==kind)} for kind in limits}
    snapshot=read(ROOT/'snapshot.json')
    for path,expected in snapshot['protected'].items():assert hashlib.sha256((WORKSPACE/path).read_bytes()).hexdigest()==expected,path
    result={'stage':args.stage,'status':'passed','device':args.device,'batch_size':args.batch_size,'torch':torch.__version__,'cpu_threads':1,
        'parent_checkpoint_sha256':parent_sha,'parent_optimizer_loaded':parent is not None,
        'deterministic':True,'fixture':str(fixture),'coverage':coverage,'thresholds_atol_rtol':limits,
        'relative_error_definition':'abs(a-b)/max(abs(a),abs(b),1e-8); acceptance uses atol+rtol*abs(reference)',
        'cases':cases,'maxima':maxima,'comparisons':results,'production_hashes_unchanged':True,
        'optimized_source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'src').rglob('*.py')}}
    write(ROOT/'results'/('correctness_'+args.stage+'.json'),result);stop.set()
    print(args.stage,'PASS',json.dumps(maxima),flush=True)

if __name__=='__main__':main()
