"""Prepare, smoke-check, train or evaluate the personal-corpus RGCN explicitly."""
import argparse
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read,digest
from src.analoggym_campaign import sha
from src.windows_campaign_io import atomic_json
from src.personal_gnn_data import prepare,PersonalDataset
from src.model.personal_network import PersonalEdgeRGCN,initialize_backbone
from src.train.stage import build_optimizer,parameter_counts
from src.train.loop import fit,TrainingCoverage,evaluate
from src.train.losses import supervision_loss,inverse_symlog

CONFIG=ROOT/'config/personal_rgcn.json'
MODEL_KEYS=['hidden_dim','num_layers','dropout','update_edges','use_domain_emb','use_film','collapse_relations']
SOURCES=['src/personal_gnn_data.py','src/model/personal_network.py','src/model/network.py','src/model/layers.py',
    'src/model/heads.py','src/model/film.py','src/graph_enrichment.py','src/graph/builder.py',
    'src/train/losses.py','src/train/metrics.py','src/train/loop.py','src/train/stage.py']


def model_for(dataset,config):return PersonalEdgeRGCN(dataset.schema,dataset.tasks,**{k:config[k] for k in MODEL_KEYS})


def save_checkpoint(path,model,data,config,coverage,status):
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'status':status,'model':model.state_dict(),'schema':data.schema,'tasks':data.tasks,'config':config,
        'dataset_fingerprint':digest(data.manifest),'source_implementation_sha256':{p:sha(ROOT/p) for p in SOURCES},
        'trained_target_slots':{g:sorted(v) for g,v in coverage.slots.items()},'trained_domain_ids':sorted(coverage.domains)},path)


def load_checkpoint(path,data,allow_smoke=False):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint['status']!='trained' and not allow_smoke:raise ValueError('Smoke checkpoint is not a trained predictor')
    if checkpoint['dataset_fingerprint']!=digest(data.manifest):raise ValueError('Checkpoint dataset changed')
    for p,h in checkpoint['source_implementation_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Model implementation changed: '+p)
    model=model_for(data,checkpoint['config']);model.load_state_dict(checkpoint['model']);model.eval()
    coverage=TrainingCoverage();coverage.slots={g:set(v) for g,v in checkpoint['trained_target_slots'].items()}
    coverage.domains=set(checkpoint['trained_domain_ids']);return model,coverage


@torch.no_grad()
def predict_rows(path,data,rows,allow_smoke=False):
    model,coverage=load_checkpoint(path,data,allow_smoke)
    batch=data.tensorize(rows)
    for group in batch.group_ids:
        if not {int(s) for s in data.tasks[group]['slots']}<=coverage.slots.get(group,set()):
            raise ValueError('Prediction head/slots never optimized')
    if not set(batch.domain_ids.tolist())<=coverage.domains:raise ValueError('Domain never optimized')
    physical=inverse_symlog(model(batch))*batch.scales.to(torch.float64)
    if not torch.isfinite(physical).all():raise ValueError('Nonfinite physical prediction')
    results=[]
    for i,group in enumerate(batch.group_ids):
        slots=data.tasks[group]['slots']
        results.append({'sample_id':batch.sample_ids[i],'prediction':[
            float(physical[i,j]) if str(j) in slots else None for j in range(16)],
            'semantic_ids':[slots.get(str(j),{}).get('semantic_id') for j in range(16)],
            'prediction_available':[str(j) in slots for j in range(16)]})
    return results


def smoke(data,config):
    rows=list(data.rows('train',limit_per_topology=2));batch=data.tensorize(rows,'train')
    validation=list(data.rows('validation',limit_per_topology=1))
    assert len(rows)==20 and len(validation)==10
    assert not set(batch.sample_ids)&{r['sample_id'] for r in validation}
    model=model_for(data,config);optimizer=build_optimizer(model);coverage=TrainingCoverage()
    with torch.no_grad():before=float(supervision_loss(model.eval()(batch),batch)[0])
    for _ in range(8):
        model.train();optimizer.zero_grad(set_to_none=True);loss,_=supervision_loss(model(batch),batch)
        loss.backward();assert model.condition_encoder[0].weight.grad.abs().sum()>0
        for layer in model.layers:assert layer.edge_update[0].weight.grad.abs().sum()>0
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step();coverage.observe(batch)
    model.eval()
    with torch.no_grad():after=float(supervision_loss(model(batch),batch)[0])
    metrics=evaluate(model,[data.tensorize(validation,'validation')],coverage)
    artifact=ROOT/'artifacts/personal_rgcn_smoke/model.pt'
    save_checkpoint(artifact,model,data,config,coverage,'construction_smoke_only')
    restored,_=load_checkpoint(artifact,data,allow_smoke=True)
    with torch.no_grad():torch.testing.assert_close(restored(batch),model(batch),rtol=0,atol=0)
    roles={}
    for name,g in data.graphs.items():
        marked={a['net']:a['roles'] for a in g['node_annotations'] if a['roles']}
        assert any('ground' in r for r in marked.values()) and any('positive_supply' in r for r in marked.values())
        roles[name]=marked
    report={'status':'passed','train_samples':20,'validation_samples':10,'test_samples_used':0,'steps':8,
        'initial_training_batch_eval_loss':before,'final_training_batch_eval_loss':after,'validation':metrics,
        'node_input_dim':len(data.schema['node_features']),'edge_input_dim':3*len(data.schema['device_features']),
        'condition_input_dim':data.schema['condition_input_dim'],'relations':len(data.schema['relations']),
        'supervision_groups':len(data.tasks),'parameters':parameter_counts(model),'roles':roles,
        'checkpoint':artifact.relative_to(ROOT).as_posix(),'checkpoint_reload_exact':True,
        'original_dataset_modified':False,'full_training_started':False,'normalization_fitted':False,
        'scope':'Engineering gradient/validation smoke; not a benchmark or a usable predictor'}
    atomic_json(ROOT/'reports/personal_rgcn_build.json',report)
    print({k:v for k,v in report.items() if k not in ['validation','roles']})


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','check','train','evaluate'])
    p.add_argument('--epochs',type=int);p.add_argument('--stage',default='pretrain',choices=['pretrain','finetune_head','finetune_all'])
    p.add_argument('--initialize-from',type=Path);p.add_argument('--allow-smoke-initialization',action='store_true')
    budgets=p.add_mutually_exclusive_group();budgets.add_argument('--budget',type=int);budgets.add_argument('--budget-per-topology',type=int)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',default='artifacts/personal_rgcn_training');args=p.parse_args()
    config=read(CONFIG);torch.set_num_threads(config['cpu_threads']);torch.manual_seed(config['seed'])
    folder=(ROOT/args.output).resolve()
    if not folder.is_relative_to(ROOT):raise ValueError('Output must remain inside workspace')
    if args.command=='prepare':print(prepare(config));return
    data=PersonalDataset(config)
    if args.command=='check':smoke(data,config);return
    if args.command=='evaluate':
        if args.checkpoint is None:raise ValueError('--checkpoint required')
        model,coverage=load_checkpoint(args.checkpoint,data)
        metrics=evaluate(model,data.batches('test',config['batch_size']),coverage,role='test')
        atomic_json(folder/'test_metrics.json',metrics);print('Test evaluated:',metrics['samples']);return
    if args.epochs is not None and args.epochs<1:raise ValueError('epochs must be positive')
    if args.stage!='pretrain' and args.initialize_from is None:raise ValueError('Fine tuning requires explicit pretrained weights')
    if args.initialize_from and args.initialize_from.resolve()==folder/'model.pt':raise ValueError('Do not overwrite initialization checkpoint')
    model=model_for(data,config);transfer=None
    if args.initialize_from:
        checkpoint=torch.load(args.initialize_from,map_location='cpu',weights_only=True)
        if checkpoint['status']!='trained' and not args.allow_smoke_initialization:
            raise ValueError('Refusing smoke weights without --allow-smoke-initialization')
        transfer=initialize_backbone(model,checkpoint)
        if args.stage=='finetune_head' and (transfer['added_relations'] or transfer['added_node_features']):
            raise ValueError('New substrate relations/features require finetune_all, not a frozen random backbone')
    allowed=None
    if args.budget is not None:
        budgets=read(data.source/config['split_directory']/'few_shot_global.json')['budgets']
        choice=next((b for b in budgets if b['budget']==args.budget),None)
        if choice is None or args.budget<10:raise ValueError('Both-contract training requires a declared global budget >=10')
        allowed=set(choice['sample_ids'])
    if args.budget_per_topology is not None:
        budgets=read(data.source/config['split_directory']/'few_shot_per_topology.json')['budgets']
        choice=next((b for b in budgets if b['budget_per_topology']==args.budget_per_topology),None)
        if choice is None:raise ValueError('Undeclared per-topology budget')
        allowed=set(choice['sample_ids'])
    optimizer=build_optimizer(model,args.stage,config['learning_rate'],config['backbone_learning_rate'],config['weight_decay'])
    result=fit(model,lambda epoch:data.batches('train',config['batch_size'],config['seed']+epoch,allowed_ids=allowed),
        lambda:data.batches('validation',config['batch_size']),optimizer,
        epochs=args.epochs or config['epochs'],patience=config['patience'])
    save_checkpoint(folder/'model.pt',model,data,config,result['coverage'],'trained')
    atomic_json(folder/'history.json',{'history':result['history'],'best_epoch':result['best_epoch'],
        'train_label_budget':len(allowed) if allowed is not None else 16000,
        'budget_scope':'per_topology' if args.budget_per_topology is not None else 'global',
        'validation_label_budget':2000,'test_used':False,
        'stage':args.stage,'transfer':transfer,'source_dataset_fingerprint':digest(data.manifest)})
    print('Training complete; fixed test set has not been evaluated.')


if __name__=='__main__':main()
