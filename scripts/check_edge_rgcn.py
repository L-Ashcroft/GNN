"""Small real-data construction/gradient benchmark, not an accuracy experiment."""
import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import pandas as pd
import pyarrow.parquet as pq
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read,dump
from src.dataset_fusion import FusionDataset
from src.gnn.data import task_registry,input_schema,tensorize
from src.gnn.model import EdgeCentricRGCN
from src.gnn.losses import supervision_loss,inverse_symlog

TOPOLOGIES=['FALCON__CGVA','FALCON__CSVA','FALCON__SBPMixer','AICircuit__CVA','AICircuit__LNA','AICircuit__PA',
    'LOADBench__SimpleOpAmp1','LOADBench__SimpleThreeStageOpAmp_SimpleOpAmp1_1_3','LOADBench__SymmetricalOpAmp2']


def selected_rows(dataset,topologies,role='train',count=4):
    split=pd.read_parquet(dataset.path/'splits/id.parquet',columns=['sample_id','role','topology_id'])
    rows=[]
    for key in topologies:
        frame=pq.read_table(dataset.path/'sampled'/(key+'.parquet')).to_pylist()
        allowed=set(split.loc[(split.role==role)&(split.topology_id==key),'sample_id'])
        candidates=[r for r in frame if r['sample_id'] in allowed]
        if len(candidates)<count:raise ValueError('Not enough genuine '+role+' rows: '+key)
        rows.extend(candidates[:count])
    assert len({r['sample_id'] for r in rows})==len(rows)
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=12)
    parser.add_argument('--threads',type=int,default=4);args=parser.parse_args()
    if not 0<args.steps<=100:raise ValueError('Smoke check supports 1..100 steps, not full training')
    torch.set_num_threads(args.threads);torch.manual_seed(0)
    config=read(ROOT/'config/edge_rgcn.json');dataset=FusionDataset();tasks=task_registry(dataset);schema=input_schema(dataset)
    rows=selected_rows(dataset,TOPOLOGIES);batch=tensorize(rows,dataset,tasks,schema)
    model=EdgeCentricRGCN(schema,tasks,**{k:config[k] for k in ['hidden_dim','num_layers','dropout','update_edges']})
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    model.eval()
    with torch.no_grad():initial=supervision_loss(model(batch),batch)[0].item()
    started=time.perf_counter();losses=[];gradient_norms=[]
    for _ in range(args.steps):
        model.train();optimizer.zero_grad(set_to_none=True)
        result=model(batch,True);loss,_=supervision_loss(result['prediction'],batch)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip_norm'],error_if_nonfinite=True)
        for layer in model.layers:
            assert layer.edge_update[0].weight.grad is not None
            assert layer.edge_update[0].weight.grad.abs().sum()>0
        optimizer.step();losses.append(loss.item());gradient_norms.append(norm.item())
    seconds=time.perf_counter()-started;model.eval()
    with torch.no_grad():
        result=model(batch,True);final=supervision_loss(result['prediction'],batch)[0].item()
        assert torch.isfinite(result['prediction']).all()
        for i in range(batch.num_graphs):
            assert torch.allclose(result['node_attention'][batch.graph_index==i].sum(),torch.tensor(1.),atol=1e-6)
        # Permute node IDs on real multi-source batched graphs, without changing labels.
        perm=torch.randperm(len(batch.x));reverse=torch.argsort(perm);changed=copy.copy(batch)
        changed.x=batch.x[perm];changed.graph_index=batch.graph_index[perm];changed.edge_index=reverse[batch.edge_index]
        torch.testing.assert_close(result['prediction'],model(changed),atol=2e-5,rtol=2e-5)
    out=ROOT/'artifacts/edge_rgcn_smoke';out.mkdir(parents=True,exist_ok=True)
    checkpoint=out/'model.pt'
    trained_slots={g:[slot for slot in range(16) if any(batch.mask[i,slot].item() for i,key in enumerate(batch.group_ids) if key==g)]
                   for g in sorted(set(batch.group_ids))}
    torch.save({'model':model.state_dict(),'schema':schema,'tasks':tasks,'config':config,
        'trained_sample_ids':batch.sample_ids,'trained_group_ids':sorted(set(batch.group_ids)),
        'trained_target_slots':trained_slots,
        'status':'construction_smoke_only_not_benchmark','dataset_manifest_sha256':hashlib.sha256((dataset.path/'manifest.json').read_bytes()).hexdigest()},checkpoint)
    restored=EdgeCentricRGCN(schema,tasks,**{k:config[k] for k in ['hidden_dim','num_layers','dropout','update_edges']})
    restored.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model']);restored.eval()
    with torch.no_grad():torch.testing.assert_close(result['prediction'],restored(batch),rtol=0,atol=0)
    ablation=EdgeCentricRGCN(schema,tasks,config['hidden_dim'],config['num_layers'],dropout=0.,update_edges=False).eval()
    with torch.no_grad():
        a=ablation(batch,True);assert torch.equal(a['initial_edge_states'],a['edge_states'])
    predictions=inverse_symlog(result['prediction'])*batch.scales.to(torch.float64)
    pq.write_table(__import__('pyarrow').Table.from_pylist([
        {'sample_id':r['sample_id'],'source_dataset':r['source_dataset'],'topology_id':r['topology_id'],
         'supervision_group':batch.group_ids[i],'performance':r['performance'],'mask':r['mask'],
         'prediction_smoke_only':[float(v) if slot in trained_slots[batch.group_ids[i]] else None for slot,v in enumerate(predictions[i])],
         'trained_target_slots':trained_slots[batch.group_ids[i]],
         'node_attention':result['node_attention'][batch.graph_index==i].tolist()}
        for i,r in enumerate(rows)]),out/'predictions.parquet',compression='zstd')
    report={'status':'passed','scope':'construction_and_gradient_smoke_not_generalization_evaluation',
        'torch':torch.__version__,'python':platform.python_version(),'os':platform.platform(),'processor':platform.processor(),
        'cpu_threads':args.threads,'cuda_available':torch.cuda.is_available(),'source_datasets':sorted({r['source_dataset'] for r in rows}),
        'real_training_samples':len(rows),'source_topologies':len(TOPOLOGIES),'optimizer_steps':args.steps,
        'initial_eval_loss':initial,'final_eval_loss':final,'loss_decreased_on_smoke_batch':final<initial,
        'optimization_losses':losses,'gradient_norm_before_clipping':gradient_norms,
        'total_parameters':sum(p.numel() for p in model.parameters()),'fixed_edge_ablation_parameters':sum(p.numel() for p in ablation.parameters()),
        'node_input_features':len(schema['node_features']),'edge_input_features':batch.edge_attr.shape[1],
        'relations':len(schema['relations']),'physical_relations':schema['physical_relations'],
        'auxiliary_coupling_relations':len(schema['relations'])-schema['physical_relations'],
        'batch_nodes':len(batch.x),'batch_message_edges':len(batch.edge_type),
        'batch_coupling_messages':int((batch.edge_type>=schema['physical_relations']).sum()),
        'supervision_heads_registered':len(tasks),'heads_exercised':len(set(batch.group_ids)),
        'trained_target_slots':trained_slots,
        'seconds_for_optimizer_steps':seconds,'seconds_per_step':seconds/args.steps,
        'state_reload_exact':True,'node_relabeling_invariant':True,'all_edge_update_layers_have_gradients':True,
        'checkpoint':checkpoint.relative_to(ROOT).as_posix(),'normalization_fitted':False,
        'dataset_files_modified':False,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'implementation_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ['src/gnn/data.py','src/gnn/model.py','src/gnn/losses.py','src/gnn/inference.py',
             'src/graph/builder.py','src/model/layers.py','src/model/heads.py','src/model/network.py',
             'src/train/losses.py','scripts/check_edge_rgcn.py','config/edge_rgcn.json','config/gnn_targets.json']}}
    dump(ROOT/'reports/edge_rgcn_build.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['implementation_sha256','optimization_losses','gradient_norm_before_clipping','trained_target_slots']},indent=2),flush=True)


if __name__=='__main__':main()
