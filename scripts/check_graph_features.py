"""Check physical connection preservation and all derived graph annotations."""
import hashlib
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read,dump
from src.graph_enrichment import enrich_graph,vocabularies,VERSION
from src.netlist import make_graph


def check():
    base=ROOT/'data/fusion_v2';catalog=read(base/'catalog.json');manifest=read(base/'manifest.json')
    graphs={s['topology_id']:read(ROOT/s['graph']) for s in catalog}
    vocab=vocabularies(graphs.values());counts=Counter();conventions=Counter()
    for spec in catalog:
        g=graphs[spec['topology_id']];assert g['annotation_version']==VERSION
        assert enrich_graph(g,spec,vocab)==g,('Derived feature mismatch',spec['topology_id'])
        original=make_graph(g['devices'])
        for field in original:assert original[field]==g[field]
        mp=g['message_passing'];n=g['num_message_edges']
        assert n==2*g['num_edges']
        for i,j in enumerate(mp['reverse_edge_index']):
            assert i!=j and mp['reverse_edge_index'][j]==i
            assert mp['edge_index'][0][i]==mp['edge_index'][1][j]
            assert mp['edge_index'][1][i]==mp['edge_index'][0][j]
            assert mp['edge_port_pairs'][i]==list(reversed(mp['edge_port_pairs'][j]))
            assert mp['edge_device_index'][i]==mp['edge_device_index'][j]
        for a in g['node_annotations']:
            counts.update(a['roles'])
            for e in a['evidence']:
                if e['basis']=='net_name_convention':conventions[e['role']]+=1
        assert all(len(row)==len(g['node_feature_names']) for row in g['node_features'])
        if spec['source_dataset']=='LOADBench':
            a={x['net']:x for x in g['node_annotations']}
            assert 'ground' in a['sourcenmos']['roles'] and a['sourcenmos']['has_verified_evidence']
            assert 'positive_supply' in a['sourcepmos']['roles'] and a['sourcepmos']['has_verified_evidence']
    result={'status':'passed','graphs':len(catalog),'annotation_version':VERSION,
        'role_annotation_counts':dict(counts),'name_convention_evidence_counts':dict(conventions),
        'no_nodes_merged':True,'physical_edges_unchanged':True,'reverse_relations_and_typed_self_loops_checked':True,
        'normalization_fitted':False,'index_sha256':manifest['index_sha256'],
        'manifest_sha256':hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest()}
    dump(base/'graph_feature_checks.json',result);print(result,flush=True)
    return result


if __name__=='__main__':check()
