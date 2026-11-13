"""Recoverable annotation migration; samples, labels and physical edges unchanged."""
import hashlib
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read,dump
from src.graph_enrichment import enrich_graph,vocabularies,VERSION


def main():
    base=ROOT/'data/fusion_v2';catalog=read(base/'catalog.json');manifest=read(base/'manifest.json')
    original={s['topology_id']:read(ROOT/s['graph']) for s in catalog}
    vocab=vocabularies(original.values())
    # Prepare and validate every annotation before updating any graph.
    converted={s['topology_id']:enrich_graph(original[s['topology_id']],s,vocab) for s in catalog}
    stable_fields=['nets','devices','edges','edge_index','edge_device_index','device_terminals','device_references',
                   'num_nodes','num_nets','num_devices','num_edges','num_terminals']
    for key,g in converted.items():
        for field in stable_fields:assert g[field]==original[key][field]
        assert g['num_message_edges']==2*g['num_edges']
        assert len(g['node_features'])==g['num_nodes']
    folder=base/'migrations';folder.mkdir(exist_ok=True)
    archive=folder/'graph_features_v1_before.zip'
    if not archive.exists():
        files=[ROOT/s['graph'] for s in catalog]+[base/'manifest.json',base/'quality_checks.json']
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in files:z.write(p,p.relative_to(ROOT).as_posix())
    for spec in catalog:dump(ROOT/spec['graph'],converted[spec['topology_id']])
    for name in manifest['metadata_sha256']:
        manifest['metadata_sha256'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    manifest['graph_annotation_version']=VERSION
    manifest['message_passing_policy']='Bidirectional typed port-pair edges; base physical graph remains unchanged'
    dump(base/'manifest.json',manifest)
    dump(base/'graph_feature_migration.json',{'version':VERSION,'graphs':len(catalog),
        'samples_unchanged':True,'labels_unchanged':True,'splits_unchanged':True,
        'physical_connectivity_unchanged':True,'normalization_fitted':False,
        'rollback_archive':archive.relative_to(ROOT).as_posix(),
        'index_sha256':manifest['index_sha256'],'relations':len(vocab['relations']),
        'terminal_types':len(vocab['terminal_types'])})
    print('Enriched',len(catalog),'graphs;',len(vocab['relations']),'directed relation types; samples unchanged')


if __name__=='__main__':main()
