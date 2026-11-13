"""Disk-backed, deterministic graph inputs; checked against original tensorize."""
import ast
from collections import Counter
import math
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from ..common import ROOT,read
from ..netlist import expression,OPS
from ..dataset_fusion import FusionDataset
from ..supervision_fusion import SupervisionRouter
from ..graph.builder import input_schema,task_registry,tensorize,CircuitBatch,COUPLING_RELATIONS
from ..graph_enrichment import LOG10,LOG2
from ..analoggym_campaign import sha
from ..windows_campaign_io import atomic_json


def vector_evaluate(text,env):
    def walk(node):
        if isinstance(node,ast.Constant):return float(node.value)
        if isinstance(node,ast.Name):return env[node.id]
        if isinstance(node,ast.BinOp) and type(node.op) in OPS:return OPS[type(node.op)](walk(node.left),walk(node.right))
        if isinstance(node,ast.UnaryOp):return (-1 if isinstance(node.op,ast.USub) else 1)*walk(node.operand)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='sqrt':return np.sqrt(walk(node.args[0]))
        raise ValueError('Unsupported vector expression')
    return walk(expression(str(text)))


def structure(graph,schema):
    mp=graph['message_passing'];src=list(mp['edge_index'][0]);dst=list(mp['edge_index'][1])
    devices=list(mp['edge_device_index']);relations=list(mp['edge_type'])
    refs={}
    for ref in graph['device_references']:refs.setdefault(ref['source'],{})[ref['role']]=ref['target']
    ni={n:i for i,n in enumerate(graph['nets'])};ri={n:i for i,n in enumerate(schema['relations'])}
    for device,mapping in refs.items():
        assert set(mapping)=={'ind1','ind2'}
        a=graph['devices'][mapping['ind1']];b=graph['devices'][mapping['ind2']]
        for pa,na in zip(a['ports'],a['nets']):
            for pb,nb in zip(b['ports'],b['nets']):
                for u,v,label in [(na,nb,f'mutual_reference:L1.{pa}>L2.{pb}'),(nb,na,f'mutual_reference:L2.{pb}>L1.{pa}')]:
                    src.append(ni[u]);dst.append(ni[v]);devices.append(device);relations.append(ri[label])
    x=[[math.log1p(v) if n=='connected_device_count' or n.startswith('terminal_count:') else float(v)
        for n,v in zip(schema['node_features'],f)] for f in graph['node_features']]
    return {'x':x,'edge_index':[src,dst],'device_index':devices,'edge_type':relations}


def prepare_cache(config):
    dataset=FusionDataset(config['dataset']);schema=input_schema(dataset);tasks=task_registry(dataset)
    output=ROOT/config['cache'];output.mkdir(parents=True,exist_ok=True)
    split=pq.read_table(ROOT/config['split'],columns=['sample_id','role','design_group']).to_pylist()
    roles={r['sample_id']:{'train':0,'validation':1,'test':2}[r['role']] for r in split}
    assert len(roles)==500000
    router=SupervisionRouter(dataset.catalog);group_names=sorted(tasks);group_ids={g:i for i,g in enumerate(group_names)}
    domains={(d['source_dataset'],d['process']):d['id'] for d in schema['domains']}
    atomic_json(output/'schema.json',schema);atomic_json(output/'tasks.json',tasks)
    specs={};role_coverage={0:{},1:{},2:{}};estimated=0
    for name,spec in dataset.catalog.items():
        path=dataset.path/'sampled'/(name+'.parquet');n=pq.ParquetFile(path).metadata.num_rows
        if n:
            graph=read(ROOT/spec['graph']);estimated+=n*(len(graph['devices'])*len(schema['device_features'])*6+100)
    print('Estimated tensor-cache disk GiB:',round(estimated/2**30,2),flush=True)
    for number,(name,spec) in enumerate(dataset.catalog.items(),1):
        path=dataset.path/'sampled'/(name+'.parquet');n=pq.ParquetFile(path).metadata.num_rows
        if not n:continue
        graph=read(ROOT/spec['graph']);folder=output/name;folder.mkdir(exist_ok=True)
        signature={'source':sha(path),'graph':sha(ROOT/spec['graph']),'split':sha(ROOT/config['split']),'producer':sha(Path(__file__))}
        receipt=folder/'receipt.json'
        if receipt.exists() and read(receipt)['inputs']==signature:
            old=read(receipt);specs[name]=old
            for role,groups in old['coverage'].items():
                for group,slots in groups.items():role_coverage[int(role)].setdefault(group,set()).update(slots)
            print('Cache reuse',name,n,flush=True);continue
        rows=pq.read_table(path).to_pylist();params=rows[0]['parameter_names']
        assert all(r['parameter_names']==params for r in rows)
        matrix=np.asarray([r['parameters'] for r in rows],dtype=np.float64)
        env={**spec['defaults'],**{p:matrix[:,i] for i,p in enumerate(params)}}
        d=len(graph['devices']);f=len(schema['device_features'])
        values=np.lib.format.open_memmap(folder/'values.npy',mode='w+',dtype='float32',shape=(n,d,f))
        known=np.lib.format.open_memmap(folder/'known.npy',mode='w+',dtype='bool',shape=(n,d,f))
        zero=np.lib.format.open_memmap(folder/'zero.npy',mode='w+',dtype='bool',shape=(n,d,f))
        for device,expressions in enumerate(graph['feature_expressions']):
            for j,(text,feature) in enumerate(zip(expressions,schema['device_features'])):
                try:raw=np.broadcast_to(np.asarray(vector_evaluate(text,env) if text is not None else np.nan,dtype=float),(n,))
                except KeyError:raw=np.full(n,np.nan)
                valid=np.isfinite(raw);iszero=np.zeros(n,dtype=bool)
                if feature in LOG10|LOG2:
                    valid&=raw>=0;iszero=valid&(raw==0);safe=np.where(valid&(raw>0),raw,1.)
                    converted=np.log10(safe) if feature in LOG10 else np.log2(safe)
                elif feature=='hierarchy_depth' or feature.startswith('device_type:'):converted=raw
                else:converted=np.sign(raw)*np.log1p(np.abs(raw))
                values[:,device,j]=np.where(valid,converted,0.);known[:,device,j]=valid;zero[:,device,j]=iszero
        values.flush();known.flush();zero.flush()
        target=np.array([[0. if v is None else v for v in r['performance']] for r in rows],dtype=np.float32)
        mask=np.array([r['mask'] for r in rows],dtype=bool);role=np.array([roles[r['sample_id']] for r in rows],dtype=np.uint8)
        groups=[];coverage={0:{},1:{},2:{}}
        for i,row in enumerate(rows):
            group=router.describe(row)['supervision_group'];groups.append(group_ids[group])
            slots=np.flatnonzero(mask[i]).tolist();coverage[int(role[i])].setdefault(group,set()).update(slots)
        domain=np.array([domains[(r['source_dataset'],r.get('process') or 'unknown')] for r in rows],dtype=np.uint8)
        assert np.isfinite(target[mask]).all()
        for filename,array in [('target',target),('mask',mask),('role',role),('group',np.asarray(groups,dtype=np.int16)),('domain',domain)]:
            np.save(folder/(filename+'.npy'),array,allow_pickle=False)
        np.save(folder/'sample_ids.npy',np.array([r['sample_id'].encode('utf-8') for r in rows]),allow_pickle=False)
        static=structure(graph,schema);atomic_json(folder/'structure.json',static)
        # Independent reference tensorization on first/middle/last real designs.
        for i in sorted({0,n//2,n-1}):
            reference=tensorize([rows[i]],dataset,tasks,schema)
            dev=np.concatenate([np.asarray(values[i]),np.asarray(known[i],dtype=np.float32),np.asarray(zero[i],dtype=np.float32)],axis=1)
            torch.testing.assert_close(reference.edge_attr,torch.from_numpy(dev[np.asarray(static['device_index'])]),rtol=2e-6,atol=2e-6)
            torch.testing.assert_close(reference.x,torch.tensor(static['x'],dtype=torch.float32),rtol=0,atol=0)
            assert reference.edge_index.tolist()==static['edge_index'] and reference.edge_type.tolist()==static['edge_type']
            assert reference.group_ids==[group_names[groups[i]]]
        item={'rows':n,'folder':folder.relative_to(ROOT).as_posix(),'inputs':signature,
            'cache_sha256':{p.name:sha(p) for p in folder.glob('*.npy')},
            'coverage':{str(r):{g:sorted(s) for g,s in gs.items()} for r,gs in coverage.items()},'reference_checks':len({0,n//2,n-1})}
        atomic_json(receipt,item);specs[name]=item
        for r,gs in coverage.items():
            for g,s in gs.items():role_coverage[r].setdefault(g,set()).update(s)
        print('Cached',number,'/',len(dataset.catalog),name,n,flush=True)
        del values,known,zero,rows,matrix
    for role in [1,2]:
        for group,slots in role_coverage[role].items():
            assert set(slots)<=set(role_coverage[0].get(group,set())),'Held-out target slot has no training observations'
    manifest={'dataset':config['dataset'],'dataset_manifest_sha256':sha(dataset.path/'manifest.json'),
        'index_sha256':dataset.manifest['index_sha256'],'split_sha256':sha(ROOT/config['split']),
        'schema_sha256':sha(output/'schema.json'),'tasks_sha256':sha(output/'tasks.json'),
        'topologies':specs,'group_names':group_names,'normalization_fitted':False,
        'role_counts':dict(Counter(r['role'] for r in split)),'coverage_verified':True}
    atomic_json(output/'manifest.json',manifest);return manifest


class CachedFusion:
    def __init__(self,config):
        self.config=config;self.path=ROOT/config['cache'];self.manifest=read(self.path/'manifest.json')
        assert sha(ROOT/config['split'])==self.manifest['split_sha256']
        assert sha(ROOT/config['dataset']/'manifest.json')==self.manifest['dataset_manifest_sha256']
        self.schema=read(self.path/'schema.json');self.tasks=read(self.path/'tasks.json');self.arrays={};self.static={}
        assert sha(self.path/'schema.json')==self.manifest['schema_sha256'] and sha(self.path/'tasks.json')==self.manifest['tasks_sha256']
        for name,spec in self.manifest['topologies'].items():
            assert spec['inputs']['producer']==sha(Path(__file__))
            for filename,expected in spec['cache_sha256'].items():assert sha(ROOT/spec['folder']/filename)==expected
            folder=ROOT/spec['folder'];self.arrays[name]={k:np.load(folder/(k+'.npy'),mmap_mode='r')
                for k in ['values','known','zero','target','mask','role','group','domain','sample_ids']}
            self.static[name]=read(folder/'structure.json')

    def references(self,role,seed=42):
        code={'train':0,'validation':1,'test':2}[role]
        refs=[(name,int(i)) for name,a in self.arrays.items() for i in np.flatnonzero(a['role']==code)]
        if role=='train':np.random.default_rng(seed).shuffle(refs)
        return refs

    def batch(self,refs,role):
        x=[];src=[];dst=[];attrs=[];types=[];gi=[];targets=[];masks=[];scales=[];groups=[];semantics=[];domains=[];offset=0
        for i,(name,row) in enumerate(refs):
            a=self.arrays[name];s=self.static[name];d=np.concatenate([a['values'][row],a['known'][row],a['zero'][row]],axis=1)
            x.extend(s['x']);gi.extend([i]*len(s['x']));src.extend(n+offset for n in s['edge_index'][0]);dst.extend(n+offset for n in s['edge_index'][1])
            attrs.append(d[np.asarray(s['device_index'])]);types.extend(s['edge_type']);offset+=len(s['x'])
            group=self.manifest['group_names'][int(a['group'][row])];task=self.tasks[group];groups.append(group)
            targets.append(a['target'][row]);masks.append(a['mask'][row]);domains.append(int(a['domain'][row]))
            scales.append([task['slots'].get(str(j),{}).get('scale',1.) for j in range(16)])
            semantics.append([task['slots'].get(str(j),{}).get('semantic_id') for j in range(16)])
        return CircuitBatch(x=torch.tensor(x,dtype=torch.float32),edge_index=torch.tensor([src,dst]),
            edge_attr=torch.from_numpy(np.concatenate(attrs).astype(np.float32)),edge_type=torch.tensor(types),graph_index=torch.tensor(gi),
            targets=torch.from_numpy(np.stack(targets)),mask=torch.from_numpy(np.stack(masks)),scales=torch.tensor(scales),
            group_ids=groups,semantic_ids=semantics,sample_ids=[self.arrays[n]['sample_ids'][i].decode('utf-8') for n,i in refs],
            domain_ids=torch.tensor(domains),split_role=role)

    def batches(self,role,batch_size,seed=42,start_batch=0):
        refs=self.references(role,seed)
        for i in range(start_batch*batch_size,len(refs),batch_size):yield self.batch(refs[i:i+batch_size],role)
