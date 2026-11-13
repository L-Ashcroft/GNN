"""Verify documented model settings without importing torch or running training."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def main():
    config=read('config/edge_rgcn.json');film_config=read('config/edge_rgcn_film.json')
    base=read('reports/edge_rgcn_build.json');film=read('reports/lecture5_implementation.json')
    checked={}
    for report in [base,film]:
        for path,expected in report['implementation_sha256'].items():
            assert sha(path)==expected,'Historical self-check no longer describes source: '+path
            checked[path]=expected
    manifest=read('data/fusion_v2/manifest.json')
    for path,expected in manifest['metadata_sha256'].items():assert sha(path)==expected
    catalog={s['topology_id']:s for s in read('data/fusion_v2/catalog.json')}
    group_audit=read('data/fusion_v2/supervision_group_details.json')
    assert group_audit['index_sha256']==manifest['index_sha256']
    graph=read(next(iter(catalog.values()))['graph'])
    groups=group_audit['current_groups'];domains=sorted({(s['source_dataset'],s.get('process') or 'unknown') for s in catalog.values()})
    targets=read('config/gnn_targets.json');families={n:f for f,names in targets['families'].items() for n in names}
    active_families=set();slots=0
    for group in groups:
        rules=catalog[group['topology_ids'][0]]['metric_mapping'];slots+=len(rules)
        active_families.update(families[r['original_name']] for r in rules)
    d=config['hidden_dim'];r=len(graph['feature_vocabulary']['relations'])+8;layers=config['num_layers']
    linear=lambda a,b:a*b+b
    components={'node_encoder':linear(len(graph['node_feature_names']),d),
        'edge_encoder':linear(3*len(graph['feature_names']),d),'relation_embedding':r*d,
        'relation_matrices_all_layers':layers*r*2*d*d,'root_transforms_all_layers':layers*linear(d,d),
        'layer_norms_all_layers':layers*4*d,
        'edge_update_MLPs_all_layers':layers*(linear(3*d,d)+linear(d,d)),
        'two_attention_pools':2*(linear(d,d//2)+linear(d//2,1)),
        'pool_fusion':linear(2*d,d)+2*d,'head_shared':linear(d,d),
        'head_family_MLPs':len(active_families)*linear(d,d//2),'contract_readouts':slots*(d//2+1)}
    total=sum(components.values());film_added=len(domains)*d+layers*linear(d,2*d)
    assert total==base['total_parameters']
    assert total+film_added==film['stage_parameter_counts']['pretrain']['total']
    report={'checked_at_UTC':datetime.now(timezone.utc).isoformat(),'config':config,'film_config':film_config,
        'node_features':graph['node_feature_names'],'device_features':graph['feature_names'],
        'node_dimensions':len(graph['node_feature_names']),'edge_dimensions':3*len(graph['feature_names']),
        'physical_relations':r-8,'auxiliary_relations':8,'all_relations':r,'domains':domains,
        'supervision_contracts':len(groups),'registered_scalar_slots':slots,'families':sorted(active_families),
        'analytical_parameter_counts':components,'base_total':total,'film_added':film_added,'film_total':total+film_added,
        'historical_smoke_source_hashes_verified':checked,'fusion_metadata_hashes_verified':len(manifest['metadata_sha256']),
        'dataset_summary':{k:manifest[k] for k in ['sample_count','source_counts','sampled_topology_entries','full_port_structure_classes']},
        'observed_description_mismatch':{'config_pooling':config['pooling'],'actual_model_pooling':'node attention + edge attention + fusion'},
        'training_run':False,'torch_imported':False,'simulation_files_modified':False}
    (ROOT/'reports/gnn_settings_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['node_dimensions','edge_dimensions','all_relations','supervision_contracts','registered_scalar_slots','base_total','film_total']},indent=2))


if __name__=='__main__':main()
