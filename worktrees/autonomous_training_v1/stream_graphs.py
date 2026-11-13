"""Bounded-memory raw-graph reader, exact frozen CachedFusion.batch method.

LRU eviction explicitly closes Windows memmaps. Every indexed ndarray is
copied first: a batch must never retain a view into an evicted mapping.
Cached embeddings are never used here, only sample references/label metadata.
"""
from pathlib import Path
from collections import OrderedDict
from types import SimpleNamespace,MethodType
import ast
import json
import hashlib
import math
import numpy as np

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(2**20),b''):value.update(part)
    return value.hexdigest()

class OwnedRows:
    def __init__(self,path):self.array=np.load(path,mmap_mode='r',allow_pickle=False)
    def __getitem__(self,index):
        value=self.array[index]
        return value.copy() if isinstance(value,np.ndarray) else value.item() if isinstance(value,np.generic) else value
    def close(self):self.array._mmap.close()

class StoreProxy:
    def __init__(self,store,field):self.store=store;self.field=field
    def __getitem__(self,key):return self.store.fetch(key)[self.field]

class StreamGraphs:
    def __init__(self,root,raw_cache,metadata_cache,version,capacity=8,verify=False):
        if capacity<1:raise ValueError('Positive LRU capacity required')
        self.root=Path(root);self.path=self.root/raw_cache;self.meta_path=self.root/metadata_cache
        self.manifest=read(self.path/'manifest.json');self.feature_manifest=read(self.meta_path/'manifest.json')
        self.schema=read(self.path/'schema.json');self.tasks=read(self.path/'tasks.json')
        if sha(self.path/'manifest.json')!=self.feature_manifest['source_cache_manifest_sha256']:raise ValueError('Metadata/raw cache mismatch')
        if sha(self.path/'schema.json')!=self.manifest['schema_sha256'] or sha(self.path/'tasks.json')!=self.manifest['tasks_sha256']:
            raise ValueError('Graph schema/task registry mismatch')
        producer=self.root/version/'src/train/fusion_cache.py'
        if any(s['inputs']['producer']!=sha(producer) for s in self.manifest['topologies'].values()):raise ValueError('Batch producer source mismatch')
        if verify:
            source_tree=ast.parse(producer.read_text(encoding='utf-8'))
            structure_fn=next(n for n in source_tree.body if isinstance(n,ast.FunctionDef) and n.name=='structure')
            namespace={'math':math}
            exec(compile(ast.Module(body=[structure_fn],type_ignores=[]),'unchanged_graph_structure','exec'),namespace)
            for spec in self.manifest['topologies'].values():
                for name,value in spec['cache_sha256'].items():
                    if sha(self.root/spec['folder']/name)!=value:raise ValueError('Raw cache changed: '+name)
            graph_root=self.root/self.manifest['dataset']/'graphs'
            for name,spec in self.manifest['topologies'].items():
                original=graph_root/(name+'.json')
                if sha(original)!=spec['inputs']['graph']:raise ValueError('Original netlist graph changed: '+name)
                expected=namespace['structure'](read(original),self.schema)
                if expected!=read(self.root/spec['folder']/'structure.json'):raise ValueError('Cached graph connections/features mismatch: '+name)
            for role in ['train','validation']:
                for name,value in self.feature_manifest['roles'][role]['files'].items():
                    if name=='embedding.npy':continue # Embeddings intentionally unused.
                    if sha(self.meta_path/role/name)!=value:raise ValueError('Reference/label metadata changed: '+name)
        self.capacity=capacity;self.entries=OrderedDict();self.evictions=0;self.hits=0;self.misses=0
        self.topologies=self.feature_manifest['topologies'];self.names=self.feature_manifest['groups']
        self.metadata={r:{k:np.load(self.meta_path/r/(k+'.npy'),mmap_mode='r') for k in ['refs','mask','group']} for r in ['train','validation']}
        import torch
        from src.graph.builder import CircuitBatch
        tree=ast.parse(producer.read_text(encoding='utf-8'));cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CachedFusion')
        function=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='batch')
        namespace={'np':np,'torch':torch,'CircuitBatch':CircuitBatch}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'unchanged_CachedFusion_batch','exec'),namespace)
        self.probe=SimpleNamespace(manifest=self.manifest,schema=self.schema,tasks=self.tasks,arrays=StoreProxy(self,'arrays'),static=StoreProxy(self,'static'))
        self.probe.batch=MethodType(namespace['batch'],self.probe)

    def fetch(self,name):
        if name in self.entries:
            self.hits+=1;self.entries.move_to_end(name);return self.entries[name]
        self.misses+=1
        while len(self.entries)>=self.capacity:
            _,entry=self.entries.popitem(last=False)
            for value in entry['arrays'].values():value.close()
            self.evictions+=1
        folder=self.root/self.manifest['topologies'][name]['folder']
        arrays={k:OwnedRows(folder/(k+'.npy')) for k in ['values','known','zero','target','mask','group','domain','sample_ids']}
        entry={'arrays':arrays,'static':read(folder/'structure.json')};self.entries[name]=entry;return entry

    def batch_refs(self,references,role,sort_for_io=False):
        refs=[(str(name),int(row)) for name,row in references]
        if sort_for_io:refs.sort(key=lambda r:r[0])
        return self.probe.batch(refs,role)

    def batch(self,role,indices,sort_for_io=True):
        refs=self.metadata[role]['refs'][indices]
        return self.batch_refs([(self.topologies[int(a)],int(b)) for a,b in refs],role,sort_for_io)

    def counts(self,role,indices):
        meta=self.metadata[role];numbers=np.asarray(meta['group'][indices]);observed=np.asarray(meta['mask'][indices]).sum(1)
        result={}
        for number in np.unique(numbers):
            count=int(observed[numbers==number].sum())
            if count:result[self.names[int(number)]]=count
        return result

    def indices(self,role,seed=42):
        rows=len(self.metadata[role]['refs'])
        return np.random.default_rng(seed).permutation(rows) if role=='train' else np.arange(rows)

    def stratified_indices(self,role,count,seed=42):
        """Pilot only: source quotas 40/20/40 and even contract coverage."""
        numbers=np.asarray(self.metadata[role]['group']);rng=np.random.default_rng(seed);selected=[]
        for source,ratio in [('FALCON',.4),('AICircuit',.2),('LOADBench',.4)]:
            groups=[i for i,g in enumerate(self.names) if self.tasks[g]['contract']['source_dataset']==source]
            pools={g:rng.permutation(np.flatnonzero(numbers==g)) for g in groups};quota=int(round(count*ratio));used=0
            while used<quota and pools:
                for group in list(pools):
                    if used>=quota:break
                    pool=pools[group]
                    if not len(pool):del pools[group];continue
                    selected.append(int(pool[0]));pools[group]=pool[1:];used+=1
        result=np.array(selected,dtype=np.int64);rng.shuffle(result)
        if len(np.unique(result))!=len(result):raise ValueError('Duplicate pilot samples')
        return result

    def close(self):
        for entry in self.entries.values():
            for value in entry['arrays'].values():value.close()
        self.entries.clear()
        for meta in self.metadata.values():
            for value in meta.values():value._mmap.close()

def weighted_micro_loss(group_losses,slices,full_counts):
    """Exact equal-contract effective-batch MSE weighting, before clipping."""
    if not full_counts:raise ValueError('No observed labels in effective batch')
    sizes={g:end-start for g,start,end in slices};loss=None
    for group,value in group_losses.items():
        if group not in full_counts or sizes[group]>full_counts[group]:raise ValueError('Micro/effective label counts mismatch')
        term=value*(sizes[group]/full_counts[group])/len(full_counts)
        loss=term if loss is None else loss+term
    if loss is None:raise ValueError('Empty supervised microbatch')
    return loss
