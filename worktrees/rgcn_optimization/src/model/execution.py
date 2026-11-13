"""Per-batch structural plans: prepared on CPU, transferred once, no parameters."""
import copy
import torch


def _stamp(tensor):
    if tensor is None:return None
    try:version=tensor._version
    except RuntimeError:version=None
    return (tensor.data_ptr(),tuple(tensor.shape),str(tensor.device),version)


def signature(batch):
    return (len(batch.x),tuple(batch.group_ids),*[_stamp(getattr(batch,k,None)) for k in
        ['edge_index','edge_type','graph_index','mask','domain_ids']])


class ExecutionPlan:
    def __init__(self):
        self.tensors={};self.relation_slices={};self.pool_buckets={};self.group_slices=();self.signature=None
        self.loss_groups=();self.coverage={};self.domains=None;self.row_counts=()

    def to(self,device):
        result=copy.copy(self)
        result.tensors={k:v.to(device) for k,v in self.tensors.items()}
        return result

    def relations(self,collapsed=False):
        prefix='collapsed' if collapsed else 'relation'
        return (self.relation_slices[prefix],self.tensors[prefix+'_order'],
                self.tensors[prefix+'_dst'],self.tensors[prefix+'_count'])

    def pooling(self,name):
        return (self.pool_buckets[name],self.tensors[name+'_bucket_order'],self.tensors[name+'_bucket_graphs'])

    def grouping(self):return self.group_slices,self.tensors['group_order']


def segment_plan(graph_index,num_graphs):
    graph_index=graph_index.detach().cpu()
    if graph_index.numel() and (int(graph_index.min())<0 or int(graph_index.max())>=num_graphs):
        raise ValueError('Invalid graph membership')
    order=torch.argsort(graph_index,stable=True)
    return (order,torch.argsort(order),graph_index[order],torch.bincount(graph_index,minlength=num_graphs))


def prepare_batch(batch):
    existing=getattr(batch,'execution_plan',None)
    if existing is not None and existing.signature==signature(batch):return existing
    plan=ExecutionPlan();edge_index=batch.edge_index.detach().cpu();edge_type=batch.edge_type.detach().cpu()
    if edge_type.numel() and int(edge_type.min())<0:raise ValueError('Negative relation id')
    if edge_index.shape!=(2,len(edge_type)):raise ValueError('Invalid edge shape')
    if edge_index.numel() and (int(edge_index.min())<0 or int(edge_index.max())>=len(batch.x)):
        raise ValueError('Invalid endpoint')
    n=len(batch.x);destination=edge_index[1]
    for prefix,types in [('relation',edge_type),('collapsed',torch.zeros_like(edge_type))]:
        order=[];counts=[];slices=[];offset=0
        for relation in sorted(set(types.tolist())):
            indices=torch.nonzero(types==relation,as_tuple=False).flatten()
            norm=torch.bincount(destination[indices],minlength=n).clamp_min(1)[destination[indices]]
            order.append(indices);counts.append(norm);slices.append((relation,offset,offset+len(indices)));offset+=len(indices)
        packed=torch.cat(order) if order else torch.empty(0,dtype=torch.long)
        plan.tensors[prefix+'_order']=packed
        plan.tensors[prefix+'_dst']=destination[packed]
        normalizer=torch.cat(counts) if counts else torch.empty(0,dtype=torch.long)
        plan.tensors[prefix+'_count']=normalizer.to(batch.x.dtype)
        plan.relation_slices[prefix]=tuple(slices)
    graph_index=batch.graph_index.detach().cpu()
    if len(graph_index)!=n:raise ValueError('Invalid graph membership shape')
    if not torch.equal(graph_index[edge_index[0]],graph_index[edge_index[1]]):raise ValueError('Cross-graph edge')
    for name,indices in [('node',graph_index),('edge',graph_index[edge_index[0]])]:
        order,inverse,sorted_graph,lengths=segment_plan(indices,len(batch.group_ids))
        starts=torch.cat([torch.zeros(1,dtype=torch.long),lengths.cumsum(0)])
        graph_order=sorted(range(len(batch.group_ids)),key=lambda i:(int(lengths[i]),i))
        packed=[];groups=[];buckets=[];item_offset=0;group_offset=0
        for length in sorted(set(lengths.tolist())):
            if not length:continue
            selected=[i for i in graph_order if int(lengths[i])==length]
            packed.extend(order[int(starts[i]):int(starts[i+1])] for i in selected);groups.extend(selected)
            count=len(selected);buckets.append((length,count,item_offset,item_offset+length*count,group_offset,group_offset+count))
            item_offset+=length*count;group_offset+=count
        plan.tensors[name+'_bucket_order']=torch.cat(packed) if packed else torch.empty(0,dtype=torch.long)
        plan.tensors[name+'_bucket_graphs']=torch.tensor(groups,dtype=torch.long)
        plan.pool_buckets[name]=tuple(buckets)
    groups={}
    for i,group in enumerate(batch.group_ids):groups.setdefault(group,[]).append(i)
    group_order=[];group_slices=[]
    for group,rows in sorted(groups.items()):
        start=len(group_order);group_order.extend(rows);group_slices.append((group,start,len(group_order)))
    plan.tensors['group_order']=torch.tensor(group_order,dtype=torch.long);plan.group_slices=tuple(group_slices)
    if getattr(batch,'mask',None) is not None:
        mask=batch.mask.detach().cpu().bool()
        if mask.shape!=(len(batch.group_ids),16):raise ValueError('Invalid target mask')
        valid=torch.nonzero(mask.reshape(-1),as_tuple=False).flatten()
        plan.tensors['statistics_order']=valid
        plan.row_counts=tuple(mask.sum(1).tolist())
        plan.tensors['statistics_lengths']=mask.sum(1)
        loss_order=[];loss_lengths=[];loss_groups=[]
        for group,rows in sorted(groups.items()):
            row_tensor=torch.tensor(rows,dtype=torch.long)
            observed=mask[row_tensor]
            plan.coverage[group]=tuple(torch.nonzero(observed.any(0),as_tuple=False).flatten().tolist())
            locations=row_tensor[:,None]*16+torch.arange(16)[None,:]
            locations=locations[observed]
            if len(locations):loss_order.append(locations);loss_lengths.append(len(locations));loss_groups.append(group)
        plan.tensors['loss_order']=torch.cat(loss_order) if loss_order else torch.empty(0,dtype=torch.long)
        plan.tensors['loss_lengths']=torch.tensor(loss_lengths,dtype=torch.long);plan.loss_groups=tuple(loss_groups)
    if getattr(batch,'domain_ids',None) is not None:plan.domains=frozenset(batch.domain_ids.detach().cpu().tolist())
    # A GPU-only caller remains supported, with one explicit CPU fallback.
    # Normal training calls CircuitBatch.to(), preparing this before transfer.
    if batch.x.device.type!='cpu':plan=plan.to(batch.x.device)
    plan.signature=signature(batch);batch.execution_plan=plan
    return plan


def move_batch(batch,device):
    plan=prepare_batch(batch)
    fields={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in vars(batch).items()}
    fields['execution_plan']=plan.to(device)
    result=type(batch)(**fields);result.execution_plan.signature=signature(result)
    return result
