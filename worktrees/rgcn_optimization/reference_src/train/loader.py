"""Replayable, split-aware batches; stochastic order is controlled by a seed."""
import random
import pandas as pd
import pyarrow.parquet as pq
from ..graph.builder import tensorize,sanity_check


def iter_split_batches(dataset,tasks,schema,split_file,role,batch_size=16,seed=42,topologies=None,limit_per_topology=None):
    if role not in {'train','validation','test'} or batch_size<1:raise ValueError('Invalid loader role/batch size')
    split=pd.read_parquet(split_file,columns=['sample_id','topology_id','role'])
    if split.sample_id.duplicated().any():raise ValueError('Split has duplicate IDs')
    split=split.loc[split.role==role]
    rng=random.Random(seed);keys=sorted(split.topology_id.unique())
    if topologies is not None:keys=[k for k in keys if k in set(topologies)]
    if role=='train':rng.shuffle(keys)
    pending=[]
    for key in keys:
        allowed=set(split.loc[split.topology_id==key,'sample_id'])
        table=pq.read_table(dataset.path/'sampled'/(key+'.parquet'))
        indices=[i for i,sid in enumerate(table['sample_id'].to_pylist()) if sid in allowed]
        if role=='train':rng.shuffle(indices)
        if limit_per_topology is not None:indices=indices[:limit_per_topology]
        for row in table.take(indices).to_pylist():
            pending.append(row)
            if len(pending)==batch_size:
                batch=tensorize(pending,dataset,tasks,schema);batch.split_role=role
                sanity_check(batch,len(schema['relations']));yield batch;pending=[]
    if pending:
        batch=tensorize(pending,dataset,tasks,schema);batch.split_role=role
        sanity_check(batch,len(schema['relations']));yield batch
