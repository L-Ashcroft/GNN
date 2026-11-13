"""Fold-aware bounded raw circuit loader; never opens an ID feature cache."""
from pathlib import Path
from types import SimpleNamespace,MethodType
from collections import OrderedDict
import ast,json,hashlib,math
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
class OwnedRows:
    def __init__(self,p):self.array=np.load(p,mmap_mode='r',allow_pickle=False)
    def __getitem__(self,index):
        value=self.array[index];return value.copy() if isinstance(value,np.ndarray) else value.item() if isinstance(value,np.generic) else value
    def close(self):self.array._mmap.close()
class Proxy:
    def __init__(self,reader,key):self.reader=reader;self.key=key
    def __getitem__(self,name):return self.reader.fetch(name)[self.key]
class FoldData:
    def __init__(self,fold_id,capacity=4,verify=True):
        import torch
        from src.graph.builder import CircuitBatch
        self.path=ROOT/'data/model_ready/fusion_v2_training_v1';self.manifest=read(self.path/'manifest.json');self.schema=read(self.path/'schema.json');original=read(self.path/'tasks.json')
        self.protocol_path=ROOT/read(Path(__file__).with_name('config.json'))['protocol'];self.protocol=read(self.protocol_path)
        self.fold=next(f for f in self.protocol['folds'] if f['fold_id']==fold_id);path=ROOT/self.fold['path']
        assert sha(path)==self.fold['sha256'] and sha(ROOT/self.protocol['source_split'])==self.protocol['source_split_sha256']
        assert sha(self.path/'schema.json')==self.manifest['schema_sha256'] and sha(self.path/'tasks.json')==self.manifest['tasks_sha256']
        self.alias=self.protocol['head_alias_map'];self.topologies=self.protocol['topologies'];self.groups=self.manifest['group_names']
        self.tasks={};self.semantic_map={}
        for g,t in original.items():
            name=self.alias.get(g,g);copy=json.loads(json.dumps(t))
            if name!=g:
                merge=self.protocol['shared_contracts'][name];mapping=merge['slot_aliases'][g]
                copy['contract']['topology_id']='verified_multi_topology';copy['contract']['compatibility_group']=name
                copy['contract']['testbench_id']='verified_protocol:'+merge['protocol_signature']
                copy['contract']['target_semantic_ids']=[mapping.get(str(j),{}).get('shared_semantic_id') for j in range(16)]
                for slot,rule in copy['slots'].items():rule['semantic_id']=mapping[slot]['shared_semantic_id']
                self.semantic_map[g]=copy['contract']['target_semantic_ids']
            if name in self.tasks:
                a=self.tasks[name]['slots'];b=copy['slots'];assert set(a)==set(b)
                for slot in a:assert all(a[slot][k]==b[slot][k] for k in ['semantic_id','unit','scale','original_name'])
            else:self.tasks[name]=copy
        self.refs={}
        with np.load(path,allow_pickle=False) as f:
            for role in ['train','validation','test']:self.refs[role]=f[role+'_references'].copy()
        assert len(self.refs['train'])==self.fold['counts']['train']
        self.held_topologies={n for n in self.topologies if n in self.fold['test_topologies']+self.fold['excluded_structural_aliases']}
        for role in ['train','validation']:
            assert not {self.topologies[int(i)] for i in np.unique(self.refs[role][:,0])}&self.held_topologies
        producer=ROOT/'versions/rgcn_calibrated_head_v1/src/train/fusion_cache.py';tree=ast.parse(producer.read_text(encoding='utf-8'))
        assert all(s['inputs']['producer']==sha(producer) for s in self.manifest['topologies'].values())
        function=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CachedFusion');function=next(n for n in function.body if isinstance(n,ast.FunctionDef) and n.name=='batch')
        namespace={'torch':torch,'np':np,'CircuitBatch':CircuitBatch};exec(compile(ast.Module(body=[function],type_ignores=[]),'unchanged_raw_batch_constructor','exec'),namespace)
        self.capacity=capacity;self.entries=OrderedDict();self.probe=SimpleNamespace(manifest=self.manifest,tasks=original,arrays=Proxy(self,'arrays'),static=Proxy(self,'static'))
        self.probe.batch=MethodType(namespace['batch'],self.probe)
        if verify:
            for name,spec in self.manifest['topologies'].items():
                for filename,value in spec['cache_sha256'].items():assert sha(ROOT/spec['folder']/filename)==value,('raw data changed',name,filename)
                assert sha(ROOT/'data/fusion_v2/graphs'/(name+'.json'))==spec['inputs']['graph']
    def fetch(self,name):
        if name in self.entries:self.entries.move_to_end(name);return self.entries[name]
        if len(self.entries)>=self.capacity:
            _,old=self.entries.popitem(last=False)
            for a in old['arrays'].values():a.close()
        folder=ROOT/self.manifest['topologies'][name]['folder']
        value={'arrays':{k:OwnedRows(folder/(k+'.npy')) for k in ['values','known','zero','target','mask','group','domain','sample_ids']},'static':read(folder/'structure.json')}
        self.entries[name]=value;return value
    def batch(self,role,indices):
        if role not in self.refs:raise ValueError('Unknown fold role')
        refs=[(self.topologies[int(a)],int(b)) for a,b in self.refs[role][indices]]
        if role in {'train','validation'} and {n for n,_ in refs}&self.held_topologies:raise ValueError('Heldout topology entered fitting')
        refs.sort(key=lambda item:item[0])
        batch=self.probe.batch(refs,role)
        batch.group_ids=[self.alias.get(g,g) for g in batch.group_ids]
        batch.semantic_ids=[[self.tasks[g]['slots'].get(str(j),{}).get('semantic_id') for j in range(16)] for g in batch.group_ids]
        return batch
    def counts(self,role,indices):
        result={}
        for topology,rows in self.group_references(role,indices):
            a=self.fetch(topology)['arrays'];group=np.asarray(a['group'][rows]);mask=np.asarray(a['mask'][rows])
            for number in np.unique(group):
                g=self.alias.get(self.groups[int(number)],self.groups[int(number)]);result[g]=result.get(g,0)+int(mask[group==number].sum())
        return {g:n for g,n in result.items() if n}
    def group_references(self,role,indices=None):
        refs=self.refs[role] if indices is None else self.refs[role][indices]
        for number in np.unique(refs[:,0]):yield self.topologies[int(number)],refs[refs[:,0]==number,1]
    def close(self):
        for entry in self.entries.values():
            for a in entry['arrays'].values():a.close()
        self.entries.clear()
