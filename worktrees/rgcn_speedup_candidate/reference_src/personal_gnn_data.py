"""Model adapter for the immutable, measured personal 20k corpus."""
from collections import Counter
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import pyarrow.parquet as pq
import torch
from .common import ROOT,read
from .analoggym_campaign import evaluate,sha
from .graph_enrichment import enrich_graph,ROLES,transform_features,vocabularies
from .graph.builder import CircuitBatch,feature_vector,COUPLING_RELATIONS,sanity_check
from .windows_campaign_io import atomic_json

# Explicit inputs from the frozen measurement fixture. No observed response is used.
CONDITIONS=[('supply_voltage_V',1.),('temperature_C',100.),('power_offset_temperature_C',100.),
 ('dc_temperature_min_C',100.),('dc_temperature_max_C',100.),('load_capacitance_F',1e-9),
 ('common_mode_V',1.),('reference_voltage_V',1.),('target_output_V',1.),
 ('load_light_A',1e-3),('load_heavy_A',1e-3),('ac_min_Hz',1.),('ac_max_Hz',1e6),
 ('transient_maxstep_s',1e-8),('transient_stop_s',1e-4),('transient_input_transition_s',1e-9),
 ('load_step_transition_s',1e-7),('input_low_V',1.),('input_high_V',1.),
 ('vb',1.),('vb1',1.),('vb2',1.),('current_0_bias',1e-6)]
UNIT_SCALES={'V':1.,'W':1e-3,'dB':1.,'Hz':1e6,'degree':45.,'V/s':1e6,'s':1e-6,'1/K':1e-5,'V/A':1.,'V/V':1.}


def context_values(row):
    c=json.loads(row['conditions_json']);values=dict(c)
    for key,low,high in [('dc_temperature_sweep_C','dc_temperature_min_C','dc_temperature_max_C'),
            ('load_current_A','load_light_A','load_heavy_A'),('ac_frequency_Hz','ac_min_Hz','ac_max_Hz'),
            ('transient_levels_V','input_low_V','input_high_V')]:
        if key in c:values[low],values[high]=c[key][:2]
    values.update({k:v for k,v in zip(row['parameter_names'],row['parameters']) if k in {'vb','vb1','vb2','current_0_bias'}})
    transformed=[];known=[]
    for name,scale in CONDITIONS:
        value=values.get(name);known.append(float(value is not None))
        if value is not None and not math.isfinite(value):raise ValueError('Nonfinite condition '+name)
        transformed.append(0. if value is None else math.copysign(math.log1p(abs(value/scale)),value))
    return transformed+known


def annotate_fixture(graph,entry):
    """Roles follow actual Xop/Xdut positional bindings, never response values."""
    formal=graph['external_ports']
    if entry['family']=='Amplifier':
        assert formal==['gnda','vdda','vinn','vinp','vout']
        bindings=dict(zip(formal,['0','vdd','opout_dc','opin','opout']))
        roles={'gnda':['ground'],'vdda':['positive_supply'],'vinn':['input'],'vinp':['input'],'vout':['output']}
    else:
        actual=['vdd','out','ref','vb','0','out'] if entry['name']=='ldo_simple' else ['vdd','out','vb2','ref','vb1','0','out']
        expected=['vdd','vinp','vinn','vb','vss','vreg'] if entry['name']=='ldo_simple' else ['vdd','vinp','vb2','vinn','vb1','vss','vreg']
        assert formal==expected
        bindings=dict(zip(formal,actual))
        roles={'vdd':['positive_supply'],'vss':['ground'],'vinp':['input'],'vinn':['input','bias'],'vreg':['output']}
        for name in ['vb','vb1','vb2']:
            if name in formal:roles[name]=['bias']
    vocab=graph['feature_vocabulary'];features=[];annotations=[]
    for net in graph['nets']:
        attached=[d for d in graph['devices'] if net in d['nets']]
        counts=Counter(d['model']+':'+p for d in attached for p,n in zip(d['ports'],d['nets']) if n==net)
        r=set(roles.get(net,[]));verified=bool(r)
        flags=[int(bool(r&{'positive_supply','negative_supply','ground'}))]+[int(role in r) for role in ROLES]
        features.append([len(attached)]+flags+[int(not r),int(verified),0]+[counts[t] for t in vocab['terminal_types']])
        annotations.append({'net':net,'roles':sorted(r),'has_verified_evidence':verified,
            'evidence':[] if not r else [{'basis':'frozen_testbench_positional_binding','actual_net':bindings[net]}]})
    graph.update(node_features=features,node_annotations=annotations,fixture_port_bindings=bindings,
        graph_scope='DUT primitives; external fixture/load/feedback bindings recorded explicitly, not merged into DUT nets')
    return graph


def prepare(config):
    source=ROOT/config['dataset'];output=ROOT/config['derived'];output.mkdir(parents=True,exist_ok=True)
    state=read(source/'status.json');assert state['complete'] and state['accepted']==20000
    recipe=read(source/'recipe.json');snapshot=read(source/'dataset.json')
    master=read(ROOT/'data/fusion_v2/graphs/AICircuit__CVA.json')
    normalized={}
    for entry in recipe['entries']:
        raw=read(source/'topologies'/entry['name']/'graph.json');graph=copy.deepcopy(raw)
        for device in graph['devices']:
            original=device['model']
            if original=='sky130_fd_pr__cap_mim_m3_1':device['model']='capacitor'
            elif original=='sky130_fd_pr__res_high_po_0p35':device['model']='resistor'
            device['source_device_model']=original
        normalized[entry['name']]=graph
    extra_vocab=vocabularies(list(normalized.values()))
    vocab={key:sorted(set(master['feature_vocabulary'][key])|set(extra_vocab[key])) for key in extra_vocab}
    catalog={};tasks={};inputs={}
    for entry in recipe['entries']:
        name=entry['name'];path=source/'topologies'/name/'graph.json';mapping=read(path.parent/'performance_mapping.json')
        raw=read(path);g=annotate_fixture(enrich_graph(normalized[name],{'source_dataset':'AnalogGym_generated'},vocab),entry)
        assert g['num_edges']==raw['num_edges']
        for a,b in zip(g['devices'],raw['devices']):
            assert all(a[k]==b[k] for k in ['name','nets','ports','attributes','pdk_model'])
        assert g['message_passing']['reverse_edge_index']==[i^1 for i in range(g['num_message_edges'])]
        graph_path=output/'graphs'/(name+'.json');atomic_json(graph_path,g)
        family='Amplifier' if entry['family']=='Amplifier' else 'LDO'
        metric_families=(['temperature','power','swing_offset','gain_frequency','rf_quality','rf_quality','rf_quality',
            'gain_frequency','stability','speed','speed','speed','speed','swing_offset','swing_offset'] if family=='Amplifier'
            else ['swing_offset','swing_offset','power','power','regulation','regulation','rf_quality','rf_quality','rf_quality',
                  'swing_offset','swing_offset','swing_offset','swing_offset'])
        group=mapping['group_id'];slots={}
        for metric,f in zip(mapping['metrics'],metric_families):
            scale=UNIT_SCALES[metric['unit']]
            if 'offset' in metric['name'] or 'tracking_error' in metric['name']:scale=1e-3
            slots[str(metric['slot'])]={'semantic_id':metric['semantic_id'],'unit':metric['unit'],
                'original_name':metric['name'],'definition':metric['definition'],'family':f,'scale':scale}
        task={'contract':{'source':'AnalogGym_generated','conditions':mapping['conditions'],'family':family},'slots':slots}
        if group in tasks:assert task==tasks[group],'Unverified within-source target sharing'
        tasks[group]=task;catalog[name]={'graph':graph_path.relative_to(ROOT).as_posix(),'group':group,'family':family,
            'mapping':mapping,'source_graph':path.relative_to(ROOT).as_posix()}
        inputs[path.relative_to(ROOT).as_posix()]=sha(path)
        for filename in ['performance_mapping.json','conditions.json']:inputs[(path.parent/filename).relative_to(ROOT).as_posix()]=sha(path.parent/filename)
    schema={'node_features':g['node_feature_names'],'device_features':g['feature_names'],
        'relations':vocab['relations']+COUPLING_RELATIONS,'physical_relations':len(vocab['relations']),
        'domains':[{'id':0,'source_dataset':'AnalogGym_generated','process':'SKY130'}],
        'conditions':[{'name':n,'scale':s} for n,s in CONDITIONS],'condition_input_dim':2*len(CONDITIONS)}
    for path in ['dataset.json','recipe.json','manifest.json','quality_checks.json','quality_gate_checks.json',
            config['split_directory']+'/id.parquet',config['split_directory']+'/few_shot_global.json',
            config['split_directory']+'/few_shot_per_topology.json']:
        inputs[(source/path).relative_to(ROOT).as_posix()]=sha(source/path)
    for shard in snapshot['shards']:inputs[shard['path']]=sha(ROOT/shard['path'])
    atomic_json(output/'catalog.json',catalog);atomic_json(output/'tasks.json',tasks);atomic_json(output/'schema.json',schema)
    atomic_json(output/'manifest.json',{'source':config['dataset'],'samples':20000,'topologies':10,'source_inputs_sha256':inputs,
        'model_metadata_sha256':{(output/n).relative_to(ROOT).as_posix():sha(output/n) for n in ['catalog.json','tasks.json','schema.json']},
        'split':(source/config['split_directory']/'id.parquet').relative_to(ROOT).as_posix(),
        'derived_graph_sha256':{c['graph']:sha(ROOT/c['graph']) for c in catalog.values()},
        'normalization_fitted':False,'original_data_modified':False,'cross_source_target_equivalence_approved':False})
    return output


@dataclass
class PersonalBatch(CircuitBatch):
    conditions:torch.Tensor|None=None

    def to(self,device):
        return PersonalBatch(**{k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in vars(self).items()})


class PersonalDataset:
    def __init__(self,config,verify=True):
        self.config=config;self.path=ROOT/config['derived'];self.source=ROOT/config['dataset']
        self.manifest=read(self.path/'manifest.json');self.catalog=read(self.path/'catalog.json')
        self.schema=read(self.path/'schema.json');self.tasks=read(self.path/'tasks.json');self.graphs={}
        if verify:
            for path,expected in {**self.manifest['source_inputs_sha256'],**self.manifest['derived_graph_sha256'],
                    **self.manifest['model_metadata_sha256']}.items():
                if sha(ROOT/path)!=expected:raise ValueError('Model dataset version changed: '+path)
        self.split=pq.read_table(ROOT/self.manifest['split']).to_pylist()
        self.roles={r['sample_id']:r['role'] for r in self.split}
        assert len(self.roles)==20000 and len({r['design_group'] for r in self.split})==20000

    def materialize(self,row):
        spec=self.catalog[row['topology_id']];mapping=spec['mapping']
        if row['source_dataset']!='AnalogGym_generated' or row['testbench_id']!=spec['group']:raise ValueError('Wrong source/contract')
        if json.loads(row['conditions_json'])!=mapping['conditions']:raise ValueError('Test conditions differ from registered contract')
        expected=[m['semantic_id'] for m in mapping['metrics']]+[None]*(16-len(mapping['metrics']))
        if not all(len(row[k])==16 for k in ['performance','mask','semantic_ids']):raise ValueError('Expected sixteen target slots')
        for value,known,semantic,wanted in zip(row['performance'],row['mask'],row['semantic_ids'],expected):
            if bool(known)!=(value is not None) or bool(known)!=(semantic is not None):raise ValueError('Invalid target mask')
            if known and semantic!=wanted:raise ValueError('Incompatible target semantics')
            if known and not math.isfinite(value):raise ValueError('Invalid observed target')
        if row['topology_id'] not in self.graphs:self.graphs[row['topology_id']]=read(ROOT/spec['graph'])
        graph=self.graphs[row['topology_id']];params=dict(zip(row['parameter_names'],row['parameters']))
        features=[[None if x is None else evaluate(x,params) for x in f] for f in graph['feature_expressions']]
        return graph,features

    def tensorize(self,rows,role=None):
        xs=[];src=[];dst=[];attrs=[];relations=[];indices=[];targets=[];masks=[];scales=[];groups=[];semantics=[];contexts=[]
        offset=0
        for i,row in enumerate(rows):
            if role and self.roles.get(row['sample_id'])!=role:raise ValueError('Split role mismatch')
            graph,features=self.materialize(row);mp=graph['message_passing'];names=graph['feature_names']
            values,known,zero=transform_features(features,names)
            device=[feature_vector(v,k,z,names) for v,k,z in zip(values,known,zero)]
            xs.extend([[math.log1p(v) if n=='connected_device_count' or n.startswith('terminal_count:') else float(v)
                for n,v in zip(graph['node_feature_names'],f)] for f in graph['node_features']])
            indices.extend([i]*graph['num_nodes'])
            src.extend(n+offset for n in mp['edge_index'][0]);dst.extend(n+offset for n in mp['edge_index'][1])
            attrs.extend(device[d] for d in mp['edge_device_index']);relations.extend(mp['edge_type'])
            group=self.catalog[row['topology_id']]['group'];task=self.tasks[group]
            targets.append([0. if v is None else v for v in row['performance']]);masks.append(row['mask'])
            scales.append([task['slots'].get(str(j),{}).get('scale',1.) for j in range(16)])
            groups.append(group);semantics.append(row['semantic_ids']);contexts.append(context_values(row));offset+=graph['num_nodes']
        batch=PersonalBatch(x=torch.tensor(xs,dtype=torch.float32),edge_index=torch.tensor([src,dst]),
            edge_attr=torch.tensor(attrs,dtype=torch.float32),edge_type=torch.tensor(relations),graph_index=torch.tensor(indices),
            targets=torch.tensor(targets,dtype=torch.float32),mask=torch.tensor(masks,dtype=torch.bool),
            scales=torch.tensor(scales,dtype=torch.float32),group_ids=groups,semantic_ids=semantics,
            sample_ids=[r['sample_id'] for r in rows],domain_ids=torch.zeros(len(rows),dtype=torch.long),
            split_role=role,conditions=torch.tensor(contexts,dtype=torch.float32))
        sanity_check(batch,len(self.schema['relations']))
        if not all(torch.isfinite(x).all() for x in [batch.x,batch.edge_attr,batch.conditions,batch.targets]):raise ValueError('Nonfinite input')
        return batch

    def rows(self,role,limit_per_topology=None,allowed_ids=None):
        if role not in {'train','validation','test'}:raise ValueError('Invalid split')
        count=Counter()
        for shard in read(self.source/'dataset.json')['shards']:
            for batch in pq.ParquetFile(ROOT/shard['path']).iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    name=row['topology_id']
                    if self.roles[row['sample_id']]!=role or (allowed_ids is not None and row['sample_id'] not in allowed_ids):continue
                    if limit_per_topology is not None and count[name]>=limit_per_topology:continue
                    row.pop('device_features',None)
                    count[name]+=1;yield row

    def batches(self,role,batch_size=16,seed=42,limit_per_topology=None,allowed_ids=None):
        # Balanced topology round-robin; only compact row dictionaries are cached.
        groups={n:[] for n in self.catalog}
        for row in self.rows(role,limit_per_topology,allowed_ids):groups[row['topology_id']].append(row)
        rng=random.Random(seed);names=sorted(groups)
        if role=='train':
            rng.shuffle(names)
            for part in groups.values():rng.shuffle(part)
        pending=[]
        for i in range(max(map(len,groups.values()),default=0)):
            for name in names:
                if i<len(groups[name]):pending.append(groups[name][i])
                if len(pending)==batch_size:yield self.tensorize(pending,role);pending=[]
        if pending:yield self.tensorize(pending,role)
