"""Small real TRAIN tests only. Diagnostic weights never enter experiments."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
import json,copy,time,hashlib
import psutil
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
def main():
    assert psutil.virtual_memory().available>=1.5*2**30,'Defer small CPU test: memory gate'
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    import runtime,torch,numpy as np
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    from data import ScopedData,scope_receipt,sha,epoch_indices
    from model import make_model,loss_by_contract,gcn_propagate
    from fit import fit_stats
    from src.train.stage import build_optimizer
    from assessment import metrics
    from src.train.metrics import MetricAccumulator
    suite=json.loads((HERE/'suite.json').read_text());out=ROOT/'artifacts/controlled_evidence_v1';out.mkdir(parents=True,exist_ok=True)
    rows=[];scopes={};test_calls=0;common=None
    for job in suite['jobs']:
        cfg=json.loads((ROOT/job['config']).read_text());reader=ScopedData(cfg,verify=False)
        scopes[cfg['id']]=scope_receipt(reader)
        if common is None:common={r:reader.refs[r].copy() for r in ('train','validation','test')}
        else:
            for role in ('validation','test'):assert np.array_equal(common[role],reader.refs[role])
        # Greedy relation coverage, then ensure each permitted source is represented.
        chosen=[];covered=set()
        options=[]
        for top,indices in reader.group_references('train'):
            i=next(i for i,r in enumerate(reader.refs['train']) if reader.topologies[int(r[0])]==top)
            options.append((i,set(reader.fetch(top)['static']['edge_type']),top.split('__')[0]))
        while set(range(48))-covered:
            candidate=max(options,key=lambda x:len(x[1]-covered))
            if not candidate[1]-covered:break
            chosen.append(candidate[0]);covered|=candidate[1]
        for source in cfg['train_sources']:
            candidate=next(o for o in options if o[2]==source)
            if candidate[0] not in chosen:chosen.append(candidate[0])
        # Several varying designs per selected topology, rather than one value
        # per head (which would make a mean-initialized smoke loss trivial).
        expanded=[]
        for i in chosen:
            topology_number=reader.refs['train'][i,0]
            expanded.extend(np.flatnonzero(reader.refs['train'][:,0]==topology_number)[:4].tolist())
        chosen=sorted(set(expanded));reader.refs['train']=reader.refs['train'][chosen]
        reader.fold['counts']['train']=len(chosen)
        reader.fold['sha256']=hashlib.sha256(reader.refs['train'].tobytes()).hexdigest()
        path=out/'diagnostics'/(cfg['id']+'_v2_TRAIN_subset_stats.json')
        stats=fit_stats(reader,path)
        b=reader.batch('train',np.arange(len(chosen))).to('cpu')
        torch.manual_seed(42);model=make_model(reader,stats,cfg)
        initial_hash=hashlib.sha256()
        for name,p in model.named_parameters():initial_hash.update(name.encode());initial_hash.update(p.detach().numpy().tobytes())
        params=sum(p.numel() for p in model.parameters());optimizer=build_optimizer(model,'finetune_all',3e-4,2e-4,.01)
        tick=time.perf_counter();losses=[]
        for step in range(2):
            optimizer.zero_grad(set_to_none=True);pred=model(b);loss,_=loss_by_contract(pred,b,model.head)
            assert torch.isfinite(loss);loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step();losses.append(float(loss.detach()))
        # Deterministic eval and state_dict roundtrip, including condition adapter.
        model.eval();state=copy.deepcopy(model.state_dict());p=model(b).detach();model.load_state_dict(state,strict=True)
        torch.testing.assert_close(p,model(b),rtol=0,atol=0)
        if cfg['model_kind']=='gcn':
            h=torch.randn(4,3);edges=torch.tensor([[0,0,1,2,3,3],[1,1,0,2,3,2]])
            a=torch.zeros(4,4)
            for u,v in edges.t().tolist():a[v,u]+=1
            a+=torch.eye(4);d=a.sum(1).rsqrt()
            torch.testing.assert_close(gcn_propagate(h,edges),(d[:,None]*a*d[None,:])@h,atol=2e-7,rtol=2e-6)
            bb=copy.copy(b);bb.edge_type=b.edge_type.flip(0).clone();bb.execution_plan=None
            torch.testing.assert_close(model(b,True)['embedding'],model(bb,True)['embedding'],rtol=0,atol=0)
        if cfg['model_kind']=='mlp':
            bb=copy.copy(b);bb.x=torch.zeros_like(b.x);bb.edge_attr=torch.zeros_like(b.edge_attr)
            torch.testing.assert_close(model(b,True)['embedding'],model(bb,True)['embedding'],rtol=0,atol=0)
            # Exact interrupted/resumed optimizer+dropout continuation.
            model.train();saved=(copy.deepcopy(model.state_dict()),copy.deepcopy(optimizer.state_dict()),torch.get_rng_state())
            def step_once():
                optimizer.zero_grad(set_to_none=True);loss=loss_by_contract(model(b),b,model.head)[0];loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
            step_once();expected=copy.deepcopy(model.state_dict())
            model.load_state_dict(saved[0]);optimizer.load_state_dict(saved[1]);torch.set_rng_state(saved[2]);step_once()
            for k in expected:torch.testing.assert_close(expected[k],model.state_dict()[k],rtol=0,atol=0)
        # Evaluation parity with production MetricAccumulator on real TRAIN rows.
        model.eval();pred=model(b).detach();metric=MetricAccumulator(reader.tasks);metric.update(pred,b);baseline=metric.compute()
        physical=(pred.sign()*torch.expm1(pred.abs())*b.scales.to(torch.float64)).numpy()
        records=[dict(sample_id=sid,source_dataset=top.split('__')[0],topology_id=top,supervision_group=g,prediction=physical[i].tolist(),target=b.targets[i].tolist(),mask=b.mask[i].tolist()) for i,(sid,top,g) in enumerate(zip(b.sample_ids,b.topology_ids,b.group_ids))]
        new=metrics(records,reader.tasks)[0]['task_macro']
        for key in ('macro_hit5','macro_hit10','macro_r2'):
            if baseline[key] is not None:assert np.isclose(baseline[key],new[key],rtol=1e-8,atol=1e-9),(key,baseline[key],new[key])
        rows.append(dict(id=cfg['id'],model_parameters=params,initial_parameters_sha256=initial_hash.hexdigest(),real_train_rows=len(chosen),relations=len(set(b.edge_type.tolist())),finite_steps=2,losses=losses,seconds=time.perf_counter()-tick,evaluation_parity=True))
        del model,optimizer,b,state,reader;import gc;gc.collect()
    single=scopes['controlled_single_rgcn_lb_v1'];multi=scopes['controlled_multi_rgcn_lb_v1']
    assert rows[0]['initial_parameters_sha256']==rows[1]['initial_parameters_sha256'],'Unpaired source-control initialization'
    assert single['sources']['train']['LOADBench']==multi['sources']['train']['LOADBench']
    for n in (5,160003):
        order=epoch_indices(n,400083,42,0);assert len(order)==400083 and len(set(order.tolist()))==n
        assert np.array_equal(order,epoch_indices(n,400083,42,0))
    pins={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(HERE.glob('*.py'))+sorted((HERE/'configs').glob('*.json'))+[HERE/'config.json',HERE/'suite.json']}
    result=dict(passed=True,device='cpu',diagnostic_only=True,test_batches_constructed=test_calls,validation_labels_used=False,official_weights_saved=False,
                jobs=rows,source_scopes=scopes,shared_test_roles_verified=True,shared_validation_roles_verified=True,
                schedule_reproducible=True,GCN_dense_formula_verified=True,MLP_no_graph_dependency_verified=True,MLP_exact_optimizer_RNG_resume_verified=True,
                RSS_MiB=psutil.Process().memory_info().rss/2**20,source_sha256=pins)
    (out/'preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
