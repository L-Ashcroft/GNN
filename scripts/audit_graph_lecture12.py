"""Read-only audit of graph templates against lecture 3 chapter 12."""
from collections import Counter
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read,dump
from src.netlist import make_graph


def main():
    base=ROOT/'data/fusion_v2';catalog=read(base/'catalog.json');manifest=read(base/'manifest.json')
    statistics={s['topology_id']:s for s in read(base/'sampling_statistics.json')}
    for name,expected in manifest['metadata_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected
    models=Counter();rows=[]
    for spec in catalog:
        g=read(ROOT/spec['graph']);rebuilt=make_graph(g['devices'])
        for field,value in rebuilt.items():assert g[field]==value
        for d in g['devices']:
            models[d['model']]+=1
            if d['model'] in {'nmos','pmos'}:assert d['ports']==['D','G','S','B']
        rows.append({'topology_id':spec['topology_id'],'sample_count':statistics[spec['topology_id']]['sampled'],
            'directed':g['directed'],'has_node_features':'node_features' in g,
            'has_node_roles':any(k in g for k in ['node_annotations','node_roles','net_roles','node_types','is_supply','is_ground']),
            'has_structured_hierarchy_depth':'device_hierarchy_depth' in g,
            'has_bidirectional_message_edges':g.get('num_message_edges')==2*g['num_edges'] and 'message_passing' in g,
            'has_hierarchical_instance_names':any('/' in d['name'] for d in g['devices']),
            'feature_names':g['feature_names'],'graph':spec['graph']})
    summary={'graph_templates':len(rows),'templates_with_samples':sum(r['sample_count']>0 for r in rows),
        'templates_with_node_features':sum(r['has_node_features'] for r in rows),
        'templates_with_node_role_fields':sum(r['has_node_roles'] for r in rows),
        'templates_marked_directed':sum(r['directed'] for r in rows),
        'templates_with_structured_hierarchy_depth':sum(r['has_structured_hierarchy_depth'] for r in rows),
        'templates_with_bidirectional_message_edges':sum(r['has_bidirectional_message_edges'] for r in rows),
        'all_MOS_four_terminal_roles_preserved':True,'all_existing_connectivity_checks_passed':True,
        'models':dict(models),'dataset_modified':False}
    dump(ROOT/'reports/graph_lecture12_audit.json',{'summary':summary,'graphs':rows,
        'lecture':'background/讲义3-1.md chapter 12','index_sha256':manifest['index_sha256']})
    print(summary)


if __name__=='__main__':main()
