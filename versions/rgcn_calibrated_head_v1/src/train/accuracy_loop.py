"""Same physical-unit metrics, explicitly changed standardized loss reporting."""
import torch
from .losses import symlog
from .loop import TrainingCoverage
from .metrics import MetricAccumulator
from ..model.execution import prepare_batch


class LossAccumulator:
    def __init__(self,head):self.head=head;self.values={}

    @torch.no_grad()
    def update(self,prediction,batch):
        plan=prepare_batch(batch);self.head.check_observed(plan.coverage)
        order=plan.tensors['statistics_order'];target=symlog(batch.targets/batch.scales)
        _,std=self.head.affine(batch.group_ids)
        error=((prediction.reshape(-1)[order]-target.reshape(-1)[order])/std.reshape(-1)[order]).square()
        errors=torch.segment_reduce(error,'sum',lengths=plan.tensors['statistics_lengths'],unsafe=True).cpu().tolist()
        for i,group in enumerate(batch.group_ids):
            count=plan.row_counts[i]
            if count:
                item=self.values.setdefault(group,[0.,0]);item[0]+=errors[i];item[1]+=count

    def mean(self):
        values=[total/count for total,count in self.values.values() if count]
        return sum(values)/len(values) if values else None


@torch.no_grad()
def evaluate(model,loader,coverage,device='cpu',role='validation'):
    model.eval();metrics=MetricAccumulator(model.tasks);loss=LossAccumulator(model.head);count=0
    for batch in loader:
        if batch.split_role!=role or role not in {'validation','test'}:raise ValueError('Evaluation split role mismatch')
        batch=batch.to(device);coverage.check(batch,model.use_domain_emb)
        prediction=model(batch);metrics.update(prediction,batch);loss.update(prediction,batch);count+=batch.num_graphs
    if not count:raise ValueError('Empty evaluation loader')
    return {**metrics.compute(),'loss':loss.mean(),'samples':count,
            'loss_space':'train-standardized symlog' if model.head.calibrated_targets else 'legacy symlog'}
