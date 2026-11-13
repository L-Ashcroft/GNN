"""Prepare isolated P3/P4 jobs without changing a pinned active worker."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
def write(p,v):
    if p.exists():assert json.loads(p.read_text(encoding='utf-8'))==v,('Existing config changed',p)
    else:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2),encoding='utf-8')
def main():
    source=ROOT/'worktrees/rgcn_memory_safe_id_control_v2'
    cfg=json.loads((source/'config.json').read_text(encoding='utf-8'))
    jobs=[]
    for name,kind,sources in [
        ('single_rgcn','rgcn',['LOADBench']),('multi_rgcn','rgcn',['FALCON','AICircuit','LOADBench']),
        ('single_mlp','mlp',['LOADBench']),('single_gcn','gcn',['LOADBench'])]:
        c=dict(cfg);c.update(id='controlled_'+name+'_lb_v1',fold_id='controlled_'+name+'_lb_v1',
            output='artifacts/controlled_'+name+'_lb_v1',model_kind=kind,train_sources=sources,
            experiment_kind='ID_LOADBench_control',selection_metric='LOADBench_validation_macro_hit5',
            epoch_presentations=400083,validation_sources=['LOADBench'],test_sources=['LOADBench'],
            conditions='Shared8: fixed symlog temperature/100, supply/1, load/1e-12, bias/1e-5 plus known flags',
            condition_unknown_policy='Unknown stays zero with known=false; no inferred values',
            budget='Same LOADBench train IDs, same400083 presentations per epoch and1563 maximum updates; single-source cycles TRAIN only',
            CPU_only=True)
        c['cpu_threads']=2
        path=HERE/'configs'/(name+'.json');write(path,c)
        jobs.append({'id':c['id'],'config':str(path.relative_to(ROOT)).replace('\\','/'),'output':c['output']})
    write(HERE/'suite.json',{'jobs':jobs,'requires_previous_queue':'worktrees/public_evidence_queue_v1/queue.json',
           'serial_only':True,'new_test_tuning':False,'extra_FewShot_or_LOTO_runs':False})
    code=(source/'train.py').read_text(encoding='utf-8')
    def change(a,b):
        nonlocal code
        assert a in code,a[:70]
        code=code.replace(a,b)
    change("names=['runtime.py','data.py','fit.py','model.py','train.py','qualify.py','supervisor.py','config.json']", "names=[p.name for p in HERE.glob('*.py')]+['config.json','suite.json']")
    change("files=[HERE/n for n in names]+", "files=[HERE/n for n in names]+[ROOT/cfg['config_path']]+")
    change("'private_AnalogGym_used':False}", "'private_AnalogGym_used':False,'scope':reader.fold,'parameter_input_dim':reader.parameter_dim}")
    a=code.index('def build_model(');b=code.index('\ndef main():',a)
    code=code[:a]+"def build_model(reader,stats,cfg):\n    from model import make_model\n    return make_model(reader,stats,cfg)\n"+code[b:]
    change("parser.add_argument('--backend',choices=['cpu','cuda'],default='cpu');", "parser.add_argument('--config',type=Path,required=True);parser.add_argument('--backend',choices=['cpu'],default='cpu');")
    change("cfg=read(HERE/'config.json');output=ROOT/cfg['output'];", "cfg=read(args.config);cfg['config_path']=str(args.config.resolve().relative_to(ROOT)).replace('\\\\','/');output=ROOT/cfg['output'];")
    change('from data import FoldData','from data import ScopedData,epoch_indices,scope_receipt')
    change("reader=FoldData(cfg['fold_id'],cfg['lru_topologies'],verify=True);stats=fit_stats(reader,output/'normalization.json')", "reader=ScopedData(cfg,verify=True);atomic(output/'scope.json',scope_receipt(reader));stats=fit_stats(reader,output/'normalization.json')")
    change("model=build_model(reader,stats,cfg).to(args.backend);optimizer=", "model=build_model(reader,stats,cfg).to(args.backend);atomic(output/'model_summary.json',{'parameters':sum(p.numel() for p in model.parameters()),'model_kind':cfg['model_kind'],'max_devices_train':reader.max_devices,'parameter_input_dim':reader.parameter_dim});optimizer=")
    change("'mask':b.mask[i].cpu().tolist()}", "'mask':b.mask[i].cpu().tolist(),'topology_id':b.topology_ids[i],'source_dataset':b.topology_ids[i].split('__')[0]}")
    change("np.random.default_rng(cfg['seed']+epoch).permutation(len(reader.refs['train']))", "epoch_indices(len(reader.refs['train']),cfg['epoch_presentations'],cfg['seed'],epoch)")
    change("'loss_sum':0.,'seconds':0.}", "'loss_sum':0.,'seconds':0.,'source_presentations':{s:0 for s in ['FALCON','AICircuit','LOADBench']}}")
    change("stats_epoch['samples']+=len(chosen);", "\n                for source_name in stats_epoch['source_presentations']:\n                    stats_epoch['source_presentations'][source_name]+=sum(reader.topologies[int(t)].startswith(source_name+'__') for t in reader.refs['train'][chosen,0])\n                stats_epoch['samples']+=len(chosen);")
    change("assert stats_epoch['samples']==reader.fold['counts']['train']", "assert stats_epoch['samples']==cfg['epoch_presentations']")
    change("batches=int(np.ceil(len(reader.refs['train'])/cfg['effective_batch_size']))", "batches=int(np.ceil(cfg['epoch_presentations']/cfg['effective_batch_size']))") if "batches=int(np.ceil(len(reader.refs['train'])/cfg['effective_batch_size']))" in code else None
    # Epoch-complete resume cursor is derived from the generated schedule above.
    path=HERE/'train.py'
    if path.exists():assert path.read_text(encoding='utf-8')==code,'Trainer generator drift'
    else:path.write_text(code,encoding='utf-8')
    print(json.dumps({'jobs':[j['id'] for j in jobs],'active_sources_modified':False}))
if __name__=='__main__':main()
