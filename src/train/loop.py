"""Train/validation isolation, diagnostics, early stopping and best-state restore."""
import math
import copy
import time
import torch
from .losses import supervision_loss,symlog
from .metrics import MetricAccumulator


class TrainingCoverage:
    def __init__(self):self.slots={};self.domains=set()

    def observe(self,batch):
        for i,group in enumerate(batch.group_ids):
            self.slots.setdefault(group,set()).update(torch.nonzero(batch.mask[i],as_tuple=False).flatten().tolist())
        if batch.domain_ids is not None:self.domains.update(batch.domain_ids.detach().cpu().tolist())

    def check(self,batch,domain_aware=False):
        for i,group in enumerate(batch.group_ids):
            known=set(torch.nonzero(batch.mask[i],as_tuple=False).flatten().tolist())
            if not known<=self.slots.get(group,set()):raise ValueError('Evaluation target head/slot never optimized: '+group)
        if domain_aware and (batch.domain_ids is None or not set(batch.domain_ids.cpu().tolist())<=self.domains):
            raise ValueError('Evaluation domain embedding never optimized')


class LossAccumulator:
    def __init__(self):self.values={}

    @torch.no_grad()
    def update(self,prediction,batch):
        target=symlog(batch.targets/batch.scales)
        for i,group in enumerate(batch.group_ids):
            mask=batch.mask[i];count=int(mask.sum())
            if not count:continue
            error=(prediction[i,mask]-target[i,mask]).square().sum().item()
            value=self.values.setdefault(group,[0.,0]);value[0]+=error;value[1]+=count

    def mean(self):
        values=[num/count for num,count in self.values.values() if count]
        return sum(values)/len(values) if values else None


@torch.no_grad()
def smoothness(h,edge_index):
    src,dst=edge_index;valid=src!=dst
    # Typed self-loops are physically retained; omit trivial cosine(h,h)=1
    # from this diagnostic, not from message passing.
    if not valid.any():return None
    return torch.nn.functional.cosine_similarity(h[src[valid]],h[dst[valid]],dim=-1).mean().item()


def train_epoch(model,loader,optimizer,coverage,device='cpu',clip_norm=5.):
    model.train();total=LossAccumulator();norms=[];smooth=[];count=0;start=time.perf_counter()
    for batch in loader:
        if batch.split_role!='train':raise ValueError('Only train split batches may optimize parameters')
        batch=batch.to(device);optimizer.zero_grad(set_to_none=True)
        result=model(batch,return_states=True);loss,_=supervision_loss(result['prediction'],batch)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],clip_norm,error_if_nonfinite=True)
        optimizer.step();coverage.observe(batch);total.update(result['prediction'].detach(),batch)
        norms.append(float(norm));value=smoothness(result['node_states'],batch.edge_index)
        if value is not None:smooth.append(value)
        count+=batch.num_graphs
    if not count:raise ValueError('Empty training loader')
    return {'loss':total.mean(),'samples':count,'grad_norm_mean':sum(norms)/len(norms),'grad_norm_max':max(norms),
            'smoothness':sum(smooth)/len(smooth) if smooth else None,'seconds':time.perf_counter()-start}


@torch.no_grad()
def evaluate(model,loader,coverage,device='cpu',role='validation'):
    model.eval();metrics=MetricAccumulator(model.tasks);loss=LossAccumulator();count=0
    for batch in loader:
        if batch.split_role!=role or role not in {'validation','test'}:raise ValueError('Evaluation split role mismatch')
        batch=batch.to(device);coverage.check(batch,getattr(model,'use_domain_emb',False))
        prediction=model(batch);metrics.update(prediction,batch);loss.update(prediction,batch);count+=batch.num_graphs
    if not count:raise ValueError('Empty evaluation loader')
    return {**metrics.compute(),'loss':loss.mean(),'samples':count}


class EarlyStopping:
    def __init__(self,patience=20,min_delta=1e-4):
        if patience<1:raise ValueError('patience must be positive')
        self.patience=patience;self.min_delta=min_delta;self.best=-math.inf;self.bad=0;self.best_state=None;self.best_epoch=None

    def step(self,score,model,epoch):
        if score is None or not math.isfinite(score):raise ValueError('No valid validation early-stopping score')
        if score>self.best+self.min_delta:
            self.best=score;self.bad=0;self.best_epoch=epoch
            self.best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:self.bad+=1
        return self.bad>=self.patience

    def restore(self,model):
        if self.best_state is None:raise ValueError('No validation-selected model state')
        model.load_state_dict(self.best_state)


def fit(model,train_loader_factory,validation_loader_factory,optimizer,epochs,patience=20,device='cpu'):
    if epochs<1:raise ValueError('epochs must be positive')
    model.to(device);coverage=TrainingCoverage();stop=EarlyStopping(patience);best_coverage=None
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs);history=[]
    for epoch in range(epochs):
        training=train_epoch(model,train_loader_factory(epoch),optimizer,coverage,device)
        validation=evaluate(model,validation_loader_factory(),coverage,device)
        score=validation['macro_hit5']
        metric='macro_hit5'
        if score is None:score=-validation['loss'];metric='negative_validation_loss_no_ratio_targets'
        history.append({'epoch':epoch,'training':training,'validation':validation,
                        'early_stopping_metric':metric,'learning_rates':[g['lr'] for g in optimizer.param_groups]})
        should_stop=stop.step(score,model,epoch);scheduler.step()
        if stop.best_epoch==epoch:best_coverage=copy.deepcopy(coverage)
        if should_stop:break
    stop.restore(model)
    return {'history':history,'coverage':best_coverage,'best_epoch':stop.best_epoch,'best_score':stop.best}
