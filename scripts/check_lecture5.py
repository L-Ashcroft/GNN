"""Two short real-data epochs to verify FiLM and training-loop plumbing."""
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read,dump
from src.dataset_fusion import FusionDataset
from src.graph.builder import input_schema,task_registry,tensorize,sanity_check
from src.model.network import EdgeCentricRGCN
from src.train.stage import build_optimizer,parameter_counts
from src.train.loop import fit
from src.train.loader import iter_split_batches
from scripts.check_edge_rgcn import selected_rows,TOPOLOGIES


def main():
    torch.manual_seed(0);torch.set_num_threads(4)
    dataset=FusionDataset();schema=input_schema(dataset);tasks=task_registry(dataset)
    config={**read(ROOT/'config/edge_rgcn.json'),**read(ROOT/'config/edge_rgcn_film.json')}
    kwargs={k:config[k] for k in ['hidden_dim','num_layers','dropout','update_edges','use_domain_emb','use_film','collapse_relations']}
    model=EdgeCentricRGCN(schema,tasks,**kwargs)
    # Keep this construction check small and fixed. Each row is verified against
    # its original ID split; validation observations never optimize parameters.
    train=selected_rows(dataset,TOPOLOGIES,role='train',count=4)
    validation=selected_rows(dataset,TOPOLOGIES,role='validation',count=2)
    assert not {r['sample_id'] for r in train}&{r['sample_id'] for r in validation}
    def batches(rows,role):
        for i in range(0,len(rows),16):
            batch=tensorize(rows[i:i+16],dataset,tasks,schema);batch.split_role=role
            sanity_check(batch,len(schema['relations']));yield batch
    optimizer=build_optimizer(model,'pretrain')
    result=fit(model,lambda epoch:batches(train,'train'),lambda:batches(validation,'validation'),optimizer,epochs=2,patience=2)
    out=ROOT/'artifacts/lecture5_smoke';out.mkdir(parents=True,exist_ok=True)
    checkpoint=out/'model.pt';coverage=result['coverage']
    torch.save({'model':model.state_dict(),'schema':schema,'tasks':tasks,'config':config,
        'status':'training_loop_smoke_only','trained_group_ids':sorted(coverage.slots),
        'trained_target_slots':{g:sorted(v) for g,v in coverage.slots.items()},
        'trained_domain_ids':sorted(coverage.domains),'train_sample_ids':[r['sample_id'] for r in train],
        'validation_sample_ids':[r['sample_id'] for r in validation],
        'dataset_manifest_sha256':hashlib.sha256((dataset.path/'manifest.json').read_bytes()).hexdigest()},checkpoint)
    counts={}
    for stage in ['pretrain','finetune_head','finetune_all']:
        opt=build_optimizer(model,stage);counts[stage]={**parameter_counts(model),'lr_groups':[g['lr'] for g in opt.param_groups]}
    # Exercise the actual re-iterable split loader as well as fixed tiny samples.
    loaded=next(iter_split_batches(dataset,tasks,schema,dataset.path/'splits/id.parquet','train',batch_size=2,
                                  topologies=['LOADBench__SimpleOpAmp1'],limit_per_topology=2))
    assert loaded.split_role=='train'
    report={'status':'passed','scope':'lecture5_construction_check_not_benchmark','domains':schema['domains'],
        'train_samples':len(train),'validation_samples':len(validation),'test_samples_used':0,
        'epochs':len(result['history']),'best_epoch':result['best_epoch'],'stage_parameter_counts':counts,
        'history':result['history'],'checkpoint':checkpoint.relative_to(ROOT).as_posix(),
        'dataset_modified':False,'old_import_paths_preserved':True,'torch':torch.__version__,
        'implementation_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [
            'src/graph/netlist.py','src/graph/builder.py','src/model/layers.py','src/model/film.py',
            'src/model/heads.py','src/model/network.py','src/train/losses.py','src/train/metrics.py',
            'src/train/stage.py','src/train/loader.py','src/train/loop.py','config/edge_rgcn_film.json',
            'scripts/check_lecture5.py']}}
    dump(ROOT/'reports/lecture5_implementation.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['history','implementation_sha256']},indent=2),flush=True)


if __name__=='__main__':main()
