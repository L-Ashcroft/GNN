"""Build independent 200k FALCON + 100k AICircuit + 200k LOADBench dataset."""
import argparse
import ast
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read,dump,digest,rel
from src.convert import SCHEMA
from src.netlist import topology_graph,topology_hash,evaluate
from src.load_loadbench import METRICS,quality_mask,definitions,resolve,csv_to_si,graph_from_combined
from src.stratified_sampling import hierarchical_quotas,sample_parameters

QUOTAS={'FALCON':200000,'AICircuit':100000,'LOADBench':200000}
OUT=ROOT/'data/fusion_v2'
EXTENDED=pa.schema(list(SCHEMA)+[
    pa.field('semantic_ids',pa.list_(pa.string(),16)),pa.field('parameter_units',pa.list_(pa.string())),
    pa.field('sample_component',pa.string()),pa.field('source_generation',pa.int32()),
    pa.field('source_point_index',pa.int64()),pa.field('source_feasible',pa.bool_()),
    pa.field('parameter_stratum',pa.int16())])


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mapping_for_lb(group):
    descriptions={
        'dc_gain':'Low-frequency gain from the official 1Hz AC start; not RF voltage gain',
        'ft':'Official zero-dB interpolation/extrapolation from closest positive/negative sweep values; not automatically GBW or guaranteed first crossing',
        'p':'Official DC main-supply power under the included bias/testbench',
        'phasemargin':'Official phase-margin extraction at unity-gain crossing',
        'sr_neg':'Full DC output span divided by signed falling (t90 - t10); negative; no 0.8 swing factor',
        'sr_pos':'Full DC output span divided by rising (t90 - t10); no 0.8 swing factor',
        'voffset':'Official input offset magnitude from operating-point testbench',
        'area':'Source analytic active-area estimate, not actual layout or PEX area'}
    return [{'slot':i,'original_name':name,'unit':unit,'source_unit':unit,'factor':1.,
             'semantic_id':'LOADBench::'+group+'::'+name,
             'definition':descriptions.get(name,'Official LOADBench '+name+' extraction; see frozen testbench and standalone extractor'),
             'cross_dataset_compatible':False,'evidence':['https://zenodo.org/records/21759759','https://arxiv.org/html/2610.21591v1#S2.T2']}
            for i,(name,unit) in enumerate(METRICS)]


def load_lb_spec(folder):
    evidence=read(folder/'acquisition.json')
    for item in evidence['small_member_proofs']:assert sha(ROOT/item['path'])==item['sha256']
    text=(folder/'CombinedTestBench.cir').read_text(encoding='utf-8')
    defs=definitions((folder/'params.cir').read_text(encoding='utf-8'))
    graph,fixture=graph_from_combined(text)
    tree=ast.parse((folder/'standalone_tb_extract.py').read_text(encoding='utf-8'))
    functions={'parse_raw','parse_device_models','read_param','op_value','correct_phase_series',
               'calc_dc','calc_ac','calc_tran','calc_area','load_raw'}
    semantic_nodes=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in functions:
            semantic_nodes.append(ast.dump(node,include_attributes=False))
        elif isinstance(node,ast.Assign) and not any(isinstance(t,ast.Name) and t.id in {'OPAMP_NAME','CONSTRAINTS'} for t in node.targets):
            semantic_nodes.append(ast.dump(node,include_attributes=False))
    extractor_semantics=digest(semantic_nodes)
    semantic_group=digest([fixture,extractor_semantics])
    defaults=resolve(defs,{},allow_unresolved=True)
    for expressions in graph['feature_expressions']:
        for expression in expressions:
            if expression is not None:
                try:evaluate(expression,defaults)
                except KeyError:pass  # Some official templates use ERROR placeholders; CSV must supply these.
    name=evidence['topology'];key='LOADBench__'+name
    graph.update(source_netlist=rel(folder/'CombinedTestBench.cir'),netlist_sha256=sha(folder/'CombinedTestBench.cir'))
    global_parameters=definitions(text[:text.lower().index('.subckt')])
    common={k:evaluate(v,{}) for k,v in global_parameters.items()}
    conditions={'corner':'tt','PDK':'LOADBench reduced SkyWater130 LVT',
                'supply_voltage_V':common.get('vdd'),'load_capacitance_F':common.get('cload'),
                'bias_current_A':common.get('ibias'),'temperature_C':27,
                'temperature_evidence':'ngspice default; no explicit .temp in supplied template',
                'fixture_signature':fixture,'testbench_sha256':sha(folder/'CombinedTestBench.cir'),
                'performance_extractor_signature':extractor_semantics,
                'extractor_sha256':sha(folder/'standalone_tb_extract.py'),
                'source_record':'10.5281/zenodo.21759759'}
    return {'source_dataset':'LOADBench','source_topology':name,'topology_id':key,'circuit_family':'OpAmp',
            'architecture_family':evidence['source_archive'].removesuffix('.zip'),
            'graph':rel(OUT/'graphs'/(key+'.json')),'defaults':defaults,'definitions':defs,
            'process':'SKY130','temperature':27.,'supply_voltage_default':common.get('vdd'),
            'testbench_id':'LOADBench:'+fixture,'compatibility_group':'LOADBench:'+semantic_group,
            'metric_mapping':mapping_for_lb(semantic_group),'conditions':conditions,
            'parameter_units':{k:'m' if k.startswith(('param_w','param_l')) else 'F' if k.startswith('param_c') else '1' if k.startswith('param_m') else 'source_native' for k in defs},
            'unresolved_template_parameters':sorted(set(defs)-set(defaults)),
            'acquisition':rel(folder/'acquisition.json'),'input_complete':True,
            'candidate_path':rel(OUT/'candidates'/(key+'.parquet'))},graph


def build_lb_candidates(spec):
    dest=ROOT/spec['candidate_path'];folder=(ROOT/spec['acquisition']).parent
    fingerprint=digest([sha(folder/'acquisition.json'),spec['definitions'],spec['topology_group'],spec['connectivity_group'],spec['compatibility_group'],'candidate-v4'])
    stats_path=dest.with_suffix('.json')
    if dest.exists() and stats_path.exists() and read(stats_path)['fingerprint']==fingerprint:return read(stats_path)
    import gzip
    data=gzip.decompress((folder/'intermediate_prefix.csv.gz').read_bytes())
    evidence=read(folder/'acquisition.json');assert hashlib.sha256(data).hexdigest()==evidence['prefix_csv_sha256']
    sources=[('intermediate_prefix',pd.read_csv(folder/'intermediate_prefix.csv.gz'),folder/'intermediate_prefix.csv.gz')]
    if (folder/'FinalPOFfull.csv').exists():sources.insert(0,('final_pof',pd.read_csv(folder/'FinalPOFfull.csv'),folder/'FinalPOFfull.csv'))
    allparams=sorted(spec['definitions']);frames=[];raw_count=0;invalid=0
    for component,frame,path in sources:
        raw_count+=len(frame);ok=quality_mask(frame);invalid+=int((~ok).sum())
        frame=frame.loc[ok].copy();rows=frame.index.to_numpy()+1
        params=[k for k in frame if k.lower().startswith('param_')]
        native=frame[params].to_numpy(dtype=float)
        resolved=[]
        for row in native:
            values=resolve(spec['definitions'],{k.lower():csv_to_si(k,v) for k,v in zip(params,row)})
            resolved.append([values[k] for k in allparams])
        frame['_parameters']=resolved;frame['_row']=rows;frame['_component']=component
        frame['_source']=rel(path);frame['_sha']=sha(path)
        present=[k for k,u in METRICS if k in frame]
        frame['_metric_names']=[present]*len(frame)
        frame['_design']=[digest([spec['topology_id'],[format(v,'.12g') for v in p]]) for p in resolved]
        frames.append(frame)
    combined=pd.concat(frames,ignore_index=True);conflicts=set()
    duplicate=combined.duplicated('_design',keep=False)
    repeated=combined.loc[duplicate]
    if len(repeated):
        for name,unit in METRICS:
            if name not in repeated:continue
            g=repeated.groupby('_design')[name]
            lo=g.min();hi=g.max()
            floor={'degree':.001,'V':1e-6,'W':1e-10,'Hz':1.,'dB':.001,'V/s':1.,'um2_estimated':.1}[unit]
            bad=(hi-lo)>(.002*np.maximum(lo.abs(),hi.abs())+floor)
            conflicts.update(bad[bad].index.tolist())
    rejected_conflict=int(combined['_design'].isin(conflicts).sum())
    clean=combined.loc[~combined['_design'].isin(conflicts)].drop_duplicates('_design',keep='first')
    records=[]
    for row in clean.to_dict('records'):
        names=row['_metric_names'];raw=[float(row[k]) for k in names]
        perf=[None]*16;mask=[False]*16;sem=[None]*16
        for rule in spec['metric_mapping']:
            if rule['original_name'] in names:
                i=rule['slot'];perf[i]=float(row[rule['original_name']]);mask[i]=True;sem[i]=rule['semantic_id']
        generation=None if row['_component']=='final_pof' else int(row['generation'])
        point=None if row['_component']=='final_pof' else int(row['point_index'])
        feasible=None if row['_component']=='final_pof' else str(row['feasible']).lower()=='true'
        identity='LOADBench:21759759:'+spec['source_topology']+':'+row['_component']+':'+str(row['_row'])
        records.append({'sample_id':identity,'source_dataset':'LOADBench','topology_id':spec['topology_id'],
            'topology_group':spec['topology_group'],'connectivity_group':spec['connectivity_group'],
            'circuit_family':'OpAmp','process':'SKY130','PDK':'LOADBench_reduced_SKY130_LVT',
            'temperature':27.,'supply_voltage':spec['supply_voltage_default'],'testbench_id':spec['testbench_id'],
            'compatibility_group':spec['compatibility_group'],'graph':spec['graph'],
            'device_features':spec['topology_id']+':SI_feature_recipe','input_complete':True,
            'source_path':row['_source'],'source_sha256':row['_sha'],'source_row':row['_row'],
            'parameter_names':allparams,'parameters':row['_parameters'],'metric_names':names,'raw_performance':raw,
            'performance':perf,'mask':mask,'semantic_ids':sem,
            'design_group':row['_design'],'record_hash':digest([spec['topology_id'],row['_parameters'],raw]),
            'parameter_units':[spec['parameter_units'][k] for k in allparams],
            'sample_component':row['_component'],'source_generation':generation,'source_point_index':point,
            'source_feasible':feasible,'parameter_stratum':None})
    dest.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records,schema=EXTENDED),dest,compression='zstd')
    stats={'fingerprint':fingerprint,'topology_id':spec['topology_id'],'raw_rows_in_downloaded_pool':raw_count,
           'invalid_rows':invalid,'conflicting_rows_quarantined':rejected_conflict,
           'duplicate_design_occurrences_removed':len(combined)-rejected_conflict-len(records),
           'unique_accepted_candidates':len(records),
           'policy':'Source sentinels/nonfinite/invalid ranges quarantine whole rows; optimization feasible=False alone is not a failed simulation. POF full precision preferred to rounded history for duplicate designs within tolerance; larger conflicts quarantined.'}
    dump(stats_path,stats);print('CANDIDATES',spec['source_topology'],len(records),flush=True)
    return stats


def structures(specs,graphs):
    buckets=defaultdict(list);representatives={};counts=Counter()
    for s in specs:
        graph=graphs[s['topology_id']];h=topology_hash(graph);g=topology_graph(graph)
        group=None
        for candidate in buckets[h]:
            if nx.is_isomorphic(g,representatives[candidate],node_match=nx.algorithms.isomorphism.categorical_node_match('color','')):
                group=candidate;break
        if group is None:
            group='full_ports_v2_'+str(len(representatives)).zfill(4)
            representatives[group]=g;buckets[h].append(group)
        s['connectivity_group']=group
        if s['source_dataset']=='LOADBench':s['topology_group']=group
        counts[group]+=1
        dump(ROOT/s['graph'],graph)
    return dict(counts)


def add_legacy_fields(frame,spec):
    frame=frame.copy();frame['graph']=spec['graph'];frame['connectivity_group']=spec['connectivity_group']
    sem=[None]*16
    for rule in spec['metric_mapping']:sem[rule['slot']]=rule['semantic_id']
    frame['semantic_ids']=[[v if k else None for v,k in zip(sem,mask)] for mask in frame['mask']]
    frame['parameter_units']=[spec['parameter_units']]*len(frame)
    frame['sample_component']='published_simulation_table'
    for c in ['source_generation','source_point_index','source_feasible','parameter_stratum']:frame[c]=None
    return frame


def legacy_units(spec,graph):
    units=[]
    for key in spec['parameters']:
        found=set()
        import re
        for d in graph['devices']:
            for attr,expr in d['attributes'].items():
                if re.search(r'(?<![\w])'+re.escape(key)+r'(?![\w])',expr):
                    unit={'w':'m','l':'H' if d['model']=='inductor' else 'm','r':'ohm','c':'F','dc':'A' if d['model']=='isource' else 'V','m':'1','freq':'Hz','fq':'Hz','coupling':'1'}.get(attr)
                    if unit:found.add(unit)
        units.append(next(iter(found)) if len(found)==1 else 'source_native_see_evidence')
    return units


def build(legacy_only=False):
    OUT.mkdir(parents=True,exist_ok=True)
    dump(OUT/'build_status.json',{'state':'building','expected_source_quotas':QUOTAS})
    specs=[];graphs={};old_mapping=read(ROOT/'config/performance_mapping.json')['groups']
    for original in read(ROOT/'data/processed/catalog.json'):
        if original['source_dataset'] not in {'FALCON','AICircuit'}:continue
        s=dict(original);key=s['topology_id'];graph=read(ROOT/s['graph'])
        s['graph']=rel(OUT/'graphs'/(key+'.json'));s['candidate_path']='data/processed/converted/'+key+'.parquet'
        s['metric_mapping']=old_mapping[key];s['parameter_units']=legacy_units(s,graph)
        graphs[key]=graph;specs.append(s)
    if not legacy_only:
        folders=sorted((ROOT/'data/raw/LOADBench/subset').glob('*/acquisition.json'))
        if len(folders)!=101:raise ValueError('Need completed acquisition checkpoints for all 101 topologies; currently '+str(len(folders)))
        for path in folders:
            s,g=load_lb_spec(path.parent);specs.append(s);graphs[s['topology_id']]=g
    specs.sort(key=lambda s:({'FALCON':0,'AICircuit':1,'LOADBench':2}[s['source_dataset']],s['topology_id']))
    groups=structures(specs,graphs)
    from src.graph_enrichment import enrich_graph,vocabularies,VERSION
    feature_vocab=vocabularies(graphs.values())
    for spec in specs:
        key=spec['topology_id'];graphs[key]=enrich_graph(graphs[key],spec,feature_vocab)
        dump(ROOT/spec['graph'],graphs[key])
    counts={};candidate_stats=[]
    for s in specs:
        key=s['topology_id']
        if s['source_dataset']=='LOADBench':
            stats=build_lb_candidates(s);candidate_stats.append(stats);counts[key]=stats['unique_accepted_candidates']
        else:
            candidates=pd.read_parquet(ROOT/s['candidate_path'],columns=['record_hash'])
            counts[key]=int((~candidates.duplicated('record_hash')).sum())
    dump(OUT/'catalog.json',specs)
    dump(OUT/'performance_mapping.json',{'max_dimensions':16,'mask_semantics':'only source-absent metrics; failed measurements quarantine entire row',
         'sharing_rule':'Only equal semantic_id and verified conditions permit common targets; equal slots do not imply equality',
         'groups':{s['topology_id']:s['metric_mapping'] for s in specs},'cross_dataset_approved_pairs':[]})
    (OUT/'sampled').mkdir(exist_ok=True);index=[];statistics=[]
    for source,total in QUOTAS.items():
        source_specs=[s for s in specs if s['source_dataset']==source]
        if not source_specs:continue
        quotas=hierarchical_quotas(source_specs,counts,total)
        if sum(quotas.values())<total:raise ValueError(source+' unique valid candidates insufficient: '+str(sum(quotas.values())))
        for s in source_specs:
            key=s['topology_id'];quota=quotas[key]
            frame=pd.read_parquet(ROOT/s['candidate_path'])
            frame=frame.drop_duplicates('record_hash')
            if source!='LOADBench':frame=add_legacy_fields(frame,s)
            if quota:
                frame,strata=sample_parameters(frame,quota,key)
            else:strata=[];frame=frame.iloc[:0]
            shard=OUT/'sampled'/(key+'.parquet')
            table=pa.Table.from_pandas(frame,schema=EXTENDED,preserve_index=False)
            pq.write_table(table,shard,compression='zstd')
            ix=frame[['sample_id','source_dataset','topology_id','circuit_family','connectivity_group','topology_group',
                      'design_group','record_hash','compatibility_group','parameter_stratum','source_path','source_row']].copy()
            ix['partition']=rel(shard);ix['partition_row']=np.arange(len(frame))
            index.append(ix)
            statistics.append({'source_dataset':source,'topology_id':key,'circuit_family':s['circuit_family'],
                'connectivity_group':s['connectivity_group'],'unique_candidates':counts[key],'sampled':len(frame),
                'components':frame.sample_component.value_counts().to_dict(),'parameter_strata':strata})
        print('SOURCE COMPLETE',source,total,flush=True)
    merged=pd.concat(index,ignore_index=True)
    merged.to_parquet(OUT/'index.parquet',index=False,compression='zstd')
    dump(OUT/'sampling_statistics.json',statistics)
    pd.DataFrame([{k:v for k,v in r.items() if k not in {'components','parameter_strata'}} for r in statistics]).to_csv(OUT/'sampling_statistics.csv',index=False)
    dump(OUT/'loadbench_candidate_quality.json',candidate_stats)
    summary={'state':'legacy_preview' if legacy_only else 'complete','sample_count':len(merged),'seed':42,
        'source_counts':{k:int(v) for k,v in merged.source_dataset.value_counts().items()},
        'source_topology_entries':len(specs),'sampled_topology_entries':int(merged.topology_id.nunique()),
        'full_port_structure_classes':int(merged.connectivity_group.nunique()),
        'LOADBench_topologies_with_samples':int(merged.loc[merged.source_dataset=='LOADBench','topology_id'].nunique()),
        'zero_sample_topologies':[s['topology_id'] for s in specs if not counts[s['topology_id']]],
        'graph_representation':'net_device_multigraph_v2','normalization_fitted':False,'historical_datasets_overwritten':False,
        'graph_annotation_version':VERSION,
        'message_passing_policy':'Bidirectional typed port-pair edges; base physical graph remains unchanged',
        'sampling_population':'All accepted historical FALCON/AICircuit rows; downloaded LOADBench final fronts plus initial-history prefixes, not the complete 1e8 optimization history',
        'outputs_sha256':{rel(p):sha(p) for p in sorted((OUT/'sampled').glob('*.parquet'))},
        'metadata_sha256':{rel(p):sha(p) for p in [OUT/'catalog.json',OUT/'performance_mapping.json',*sorted((OUT/'graphs').glob('*.json'))]},
        'index_sha256':sha(OUT/'index.parquet')}
    dump(OUT/'manifest.json',summary);dump(OUT/'build_status.json',{'state':summary['state'],'rows':len(merged)})
    print(summary['source_counts'],summary['sample_count'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--legacy-only',action='store_true')
    args=parser.parse_args();build(args.legacy_only)
