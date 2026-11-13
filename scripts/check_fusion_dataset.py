"""Independent checks against original candidate tables, graph ports and splits."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read,dump
from src.netlist import make_graph,parse_spectre,evaluate
from src.load_loadbench import graph_from_combined
from src.dataset_fusion import FusionDataset


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(preview=False):
    base=ROOT/'data/fusion_v2';manifest=read(base/'manifest.json')
    index=pd.read_parquet(base/'index.parquet')
    assert sha(base/'index.parquet')==manifest['index_sha256']
    for name,expected_sha in manifest.get('metadata_sha256',{}).items():assert sha(ROOT/name)==expected_sha
    expected={'FALCON':200000,'AICircuit':100000}
    if not preview:expected['LOADBench']=200000
    assert index.source_dataset.value_counts().to_dict()==expected
    assert index.sample_id.is_unique and len(index)==sum(expected.values())
    catalog=read(base/'catalog.json');stats={s['topology_id']:s for s in read(base/'sampling_statistics.json')}
    api=FusionDataset(require_validated=False);mask_counts=Counter();rows=0;graphs=0;materialized=0
    source_hashes={}
    for spec in catalog:
        key=spec['topology_id'];path=base/'sampled'/(key+'.parquet')
        assert sha(path)==manifest['outputs_sha256'][path.relative_to(ROOT).as_posix()]
        graph=read(ROOT/spec['graph']);rebuilt=make_graph(graph['devices'])
        for field in rebuilt:assert graph[field]==rebuilt[field],(key,field)
        if spec['source_dataset']=='LOADBench':
            original,_=graph_from_combined((ROOT/graph['source_netlist']).read_text(encoding='utf-8'))
            for field in rebuilt:assert original[field]==graph[field],(key,field)
        else:
            original=make_graph(parse_spectre((ROOT/graph['source_netlist']).read_text(encoding='utf-8')))
            for field in rebuilt:assert original[field]==graph[field],(key,field)
        for d in graph['devices']:
            if d['model'] in {'nmos','pmos'}:assert d['ports']==['D','G','S','B']
        graphs+=1
        frame=pd.read_parquet(path);count=len(frame);rows+=count
        for source,expected_sha in frame[['source_path','source_sha256']].drop_duplicates().itertuples(index=False,name=None):
            if source not in source_hashes:source_hashes[source]=sha(ROOT/source)
            assert source_hashes[source]==expected_sha,('Raw source changed',source)
        if spec['source_dataset']=='LOADBench':
            acquisition=read(ROOT/spec['acquisition'])
            raw_prefix=(ROOT/spec['acquisition']).parent/'intermediate_prefix.deflate'
            assert sha(raw_prefix)==acquisition['compressed_prefix_sha256']
        assert count==stats[key]['sampled']
        assert frame.record_hash.is_unique and frame.design_group.is_unique
        assert frame.sample_id.tolist()==index.loc[index.topology_id==key,'sample_id'].tolist()
        candidates=pd.read_parquet(ROOT/spec['candidate_path']).drop_duplicates('record_hash').set_index('sample_id')
        assert frame.sample_id.isin(candidates.index).all()
        original=candidates.loc[frame.sample_id]
        for field in ['parameters','raw_performance','performance','mask','source_row','source_sha256']:
            a=frame[field].tolist();b=original[field].tolist()
            for left,right in zip(a,b):
                if isinstance(left,(list,np.ndarray)):
                    assert np.array_equal(np.asarray(left),np.asarray(right),equal_nan=True),(key,field)
                else:assert left==right,(key,field)
        for row in pq.read_table(path).to_pylist():
            assert len(row['performance'])==len(row['mask'])==len(row['semantic_ids'])==16
            assert len(row['parameters'])==len(row['parameter_names'])==len(row['parameter_units'])
            assert np.isfinite(row['parameters']).all()
            for value,known,semantic in zip(row['performance'],row['mask'],row['semantic_ids']):
                assert known==(value is not None)==(semantic is not None)
                if known:assert np.isfinite(value)
            assert np.isfinite(row['raw_performance']).all()
            assert sum(row['mask'])==len(row['metric_names'])
            mask_counts[(row['source_dataset'],sum(row['mask']))]+=1
        for row in pq.read_table(path).slice(0,min(count,3)).to_pylist():
            sample=api.materialize(row);materialized+=1
            assert len(sample['edge_features'])==graph.get('num_message_edges',graph['num_edges'])
            for device,feature in zip(graph['devices'],sample['device_features']):
                if device['model'] in {'nmos','pmos'}:
                    assert feature[0] is not None and feature[0]>0
                    assert feature[1] is not None and feature[1]>0
            if spec['source_dataset']=='LOADBench':
                area=sum(d['attributes']['w']*d['attributes']['l']*d['attributes'].get('m',1.)
                         for d in sample['resolved_device_parameters'] if d['model'] in {'nmos','pmos'})*1e12
                published=dict(zip(row['metric_names'],row['raw_performance']))['area']
                assert abs(area-published)<=max(.1,.002*abs(published)),('W/L/area unit inconsistency',key,area,published)
        for cell in stats[key]['parameter_strata']:
            assert int((frame.parameter_stratum==cell['stratum']).sum())==cell['selected']
            assert cell['selected']<=cell['candidates']
        print('CHECKED',key,count,flush=True)
    split_checks={}
    for path in sorted((base/'splits').glob('*.parquet')):
        if path.name.startswith('cross_dataset'):continue
        data=pd.read_parquet(path)
        assert len(data)==len(index) and data.sample_id.is_unique
        assert set(data.sample_id)==set(index.sample_id)
        assert data.groupby('design_group').role.nunique().max()==1
        if path.name=='id.parquet':
            sizes=data.groupby('topology_id').design_group.nunique()
            roles=data.groupby('topology_id').role.nunique()
            assert (roles.loc[sizes>=3]==3).all()
        if path.name.startswith('loto'):
            assert data.groupby('connectivity_group').role.apply(lambda x:len(set(x)&{'test'})==0 or set(x)=={'test'}).all()
            assert data.groupby('holdout_group').role.apply(lambda x:'test' not in set(x) or set(x)=={'test'}).all()
        split_checks[path.name]=data.role.value_counts().to_dict()
    preserved=read(ROOT/'reports/reproducibility.json')['sha256']
    for name,expected_sha in preserved.items():assert sha(ROOT/name)==expected_sha,('Historical file modified',name)
    result={'status':'passed','scope':'legacy_preview' if preview else 'complete',
        'rows':rows,'source_counts':expected,'graphs':graphs,'materialized_examples':materialized,
        'mask_dimensions':{str(k):v for k,v in mask_counts.items()},'splits':split_checks,
        'historical_hashes_unchanged':len(preserved),'index_sha256':manifest['index_sha256'],
        'manifest_sha256':sha(base/'manifest.json'),'raw_sources_hash_verified':len(source_hashes),
        'limitations':['Feature materialization spot checked per topology; every stored parameter and label checked against candidate tables.',
                      'LOADBench candidate construction is separately tested and retains immutable raw CSV provenance.',
                      'Cross-dataset comparable-label evaluation remains blocked without verified shared target semantics.']}
    dump(base/('quality_preview.json' if preview else 'quality_checks.json'),result)
    if manifest.get('graph_annotation_version'):
        from scripts.check_graph_features import check as check_graph_features
        check_graph_features()
    print('PASS',rows,expected,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');check(p.parse_args().preview)
