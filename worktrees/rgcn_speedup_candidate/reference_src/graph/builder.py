"""Label-independent graph tensorization and semantic-contract target routing."""
from dataclasses import dataclass
import math
import torch
from ..common import ROOT,read
from ..graph_enrichment import transform_features,LOG10,LOG2
from ..supervision_fusion import SupervisionRouter

COUPLING_RELATIONS=[f'mutual_reference:L1.{a}>L2.{b}' for a in ['P','N'] for b in ['P','N']]+[
    f'mutual_reference:L2.{b}>L1.{a}' for a in ['P','N'] for b in ['P','N']]


def task_registry(dataset):
    evidence=read(dataset.path/'supervision_group_details.json')
    if evidence['index_sha256']!=dataset.manifest['index_sha256']:
        raise ValueError('Supervision audit refers to another dataset')
    config=read(ROOT/'config/gnn_targets.json')
    families={name:family for family,names in config['families'].items() for name in names}
    tasks={}
    for group in evidence['current_groups']:
        spec=dataset.catalog[group['topology_ids'][0]];contract=group['contract']
        slots={}
        for rule in spec['metric_mapping']:
            name=rule['original_name'];unit=rule['unit']
            if name not in families or unit not in config['unit_reference_scales']:
                raise ValueError('Unreviewed target '+name+' '+unit)
            slots[str(rule['slot'])]={'semantic_id':rule['semantic_id'],'unit':unit,'original_name':name,
                'family':families[name],'scale':config.get('metric_reference_scales',{}).get(name,config['unit_reference_scales'][unit])}
        tasks[group['group_id']]={'contract':contract,'slots':slots}
    return tasks


def input_schema(dataset):
    spec=next(iter(dataset.catalog.values()));graph=read(ROOT/spec['graph'])
    domains=sorted({(s['source_dataset'],s.get('process') or 'unknown') for s in dataset.catalog.values()})
    return {'node_features':graph['node_feature_names'],'device_features':graph['feature_names'],
            'domains':[{'id':i,'source_dataset':source,'process':process} for i,(source,process) in enumerate(domains)],
            'relations':graph['feature_vocabulary']['relations']+COUPLING_RELATIONS,
            'edge_input_components':['deterministically_transformed_values','known_mask','zero_mask'],
            'node_input_transform':'log1p structural counts; other flags unchanged',
            'edge_input_transform':'graph log10/log2 features; symlog remaining dimensional scalars; no fitted standardization',
            'physical_relations':len(graph['feature_vocabulary']['relations'])}


@dataclass
class CircuitBatch:
    x:torch.Tensor
    edge_index:torch.Tensor
    edge_attr:torch.Tensor
    edge_type:torch.Tensor
    graph_index:torch.Tensor
    targets:torch.Tensor
    mask:torch.Tensor
    scales:torch.Tensor
    group_ids:list
    semantic_ids:list
    sample_ids:list
    domain_ids:torch.Tensor|None=None
    split_role:str|None=None

    @property
    def num_graphs(self):return len(self.group_ids)

    def to(self,device):
        return CircuitBatch(**{k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in vars(self).items()})


def feature_vector(values,known,zero,names):
    numeric=[]
    for v,k,name in zip(values,known,names):
        if not k:numeric.append(0.);continue
        # SI frequency/area/phase/source amplitudes are not covered by the
        # graph's W/L/R/C/L logs. Compress these without fitting test statistics.
        if name not in LOG10|LOG2 and name!='hierarchy_depth' and not name.startswith('device_type:'):
            v=math.copysign(math.log1p(abs(v)),v)
        numeric.append(float(v))
    return numeric+[float(k) for k in known]+[float(z) for z in zero]


def tensorize(samples,dataset,tasks=None,schema=None):
    samples=list(samples)
    if not samples:raise ValueError('Empty graph batch')
    tasks=tasks or task_registry(dataset);schema=schema or input_schema(dataset)
    router=SupervisionRouter(dataset.catalog)
    xs=[];src=[];dst=[];attributes=[];relations=[];graph_ids=[]
    targets=[];masks=[];scales=[];groups=[];semantics=[];ids=[];offset=0;domain_ids=[]
    domain_table={(d['source_dataset'],d['process']):d['id'] for d in schema.get('domains',[])}
    for gi,raw in enumerate(samples):
        # Check semantics before graph materialization (graph is replaced by a dict).
        description=router.describe(raw);group=description['supervision_group']
        if domain_table:
            key=(raw['source_dataset'],raw.get('process') or 'unknown')
            if key not in domain_table:raise ValueError('Unregistered source/process domain: '+str(key))
            domain_ids.append(domain_table[key])
        if group not in tasks:raise ValueError('Unregistered supervision group; held-out target head cannot be silently created')
        row=raw if isinstance(raw['graph'],dict) else dataset.materialize(raw)
        graph=row['graph'];names=graph['feature_names']
        if names!=schema['device_features'] or row['node_feature_names']!=schema['node_features']:
            raise ValueError('Input feature schema mismatch')
        if graph['feature_vocabulary']['relations']!=schema['relations'][:schema['physical_relations']]:
            raise ValueError('Relation vocabulary mismatch')
        values,known,zero=transform_features(row['device_features'],names)
        device_vectors=[feature_vector(v,k,z,names) for v,k,z in zip(values,known,zero)]
        x=[]
        for feature in row['node_features']:
            x.append([math.log1p(v) if name=='connected_device_count' or name.startswith('terminal_count:') else float(v)
                      for name,v in zip(schema['node_features'],feature)])
        xs.extend(x);graph_ids.extend([gi]*len(x))
        for u,v,d,r in zip(row['edge_index'][0],row['edge_index'][1],row['edge_device_index'],row['edge_types']):
            src.append(u+offset);dst.append(v+offset);attributes.append(device_vectors[d]);relations.append(r)
        # Mutual inductors have no physical terminals. Preserve their explicit
        # inductor references as typed auxiliary messages (not added wires).
        refs={}
        for ref in graph['device_references']:refs.setdefault(ref['source'],{})[ref['role']]=ref['target']
        ni={n:i for i,n in enumerate(graph['nets'])};ri={n:i for i,n in enumerate(schema['relations'])}
        for device,mapping in refs.items():
            if graph['devices'][device]['model']!='mutual_inductor' or set(mapping)!={'ind1','ind2'}:
                raise ValueError('Unsupported device-reference semantics')
            a=graph['devices'][mapping['ind1']];b=graph['devices'][mapping['ind2']]
            if a['model']!='inductor' or b['model']!='inductor':raise ValueError('Coupling references must target inductors')
            for pa,na in zip(a['ports'],a['nets']):
                for pb,nb in zip(b['ports'],b['nets']):
                    for u,v,label in [(na,nb,f'mutual_reference:L1.{pa}>L2.{pb}'),(nb,na,f'mutual_reference:L2.{pb}>L1.{pa}')]:
                        src.append(ni[u]+offset);dst.append(ni[v]+offset)
                        attributes.append(device_vectors[device]);relations.append(ri[label])
        target=[0.]*16;scale=[1.]*16;semantic=[None]*16
        for slot,rule in tasks[group]['slots'].items():
            i=int(slot);scale[i]=rule['scale'];semantic[i]=rule['semantic_id']
            if row['mask'][i]:target[i]=float(row['performance'][i])
        targets.append(target);masks.append(row['mask']);scales.append(scale)
        groups.append(group);semantics.append(semantic);ids.append(row['sample_id']);offset+=len(x)
    batch=CircuitBatch(torch.tensor(xs,dtype=torch.float32),torch.tensor([src,dst],dtype=torch.long),
        torch.tensor(attributes,dtype=torch.float32),torch.tensor(relations,dtype=torch.long),
        torch.tensor(graph_ids,dtype=torch.long),torch.tensor(targets,dtype=torch.float32),
        torch.tensor(masks,dtype=torch.bool),torch.tensor(scales,dtype=torch.float32),groups,semantics,ids,
        torch.tensor(domain_ids,dtype=torch.long) if domain_table else None)
    if not all(torch.isfinite(x).all() for x in [batch.x,batch.edge_attr,batch.targets]):raise ValueError('Nonfinite model input or target')
    return batch


def sanity_check(batch,num_relations):
    """Lecture-5 shape/range checks; keep rather than discard typed self-loops."""
    n=len(batch.x);e=batch.edge_index.shape[1]
    if batch.edge_index.shape!=(2,e) or e==0:raise ValueError('Empty/invalid edge tensor')
    if batch.edge_index.min()<0 or batch.edge_index.max()>=n:raise ValueError('Node index out of range')
    if len(batch.edge_type)!=e or batch.edge_type.min()<0 or batch.edge_type.max()>=num_relations:
        raise ValueError('Relation index out of range')
    if not torch.equal(batch.graph_index[batch.edge_index[0]],batch.graph_index[batch.edge_index[1]]):
        raise ValueError('Message edge crosses two batched circuits')
    degree=torch.bincount(batch.edge_index[1],minlength=n)
    if (degree==0).any():raise ValueError('Isolated net in tensorized graph')
    return {'nodes':n,'message_edges':e,'typed_self_loops':int((batch.edge_index[0]==batch.edge_index[1]).sum())}
