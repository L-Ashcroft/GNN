"""Explicit new objective; retains missing-only mask and equal contract weights."""
import torch
from .losses import symlog,supervision_loss as legacy_supervision_loss
from ..model.execution import prepare_batch


def supervision_loss(prediction,batch,head):
    plan=prepare_batch(batch);head.check_observed(plan.coverage)
    if not head.calibrated_targets:return legacy_supervision_loss(prediction,batch)
    target=symlog(batch.targets/batch.scales);_,std=head.affine(batch.group_ids)
    result={};finite=[]
    for group,start,end in plan.loss_slices:
        positions=plan.tensors['loss_order'][start:end]
        p=prediction.reshape(-1)[positions];y=target.reshape(-1)[positions];s=std.reshape(-1)[positions]
        finite.append(torch.isfinite(p).all() & torch.isfinite(y).all() & torch.isfinite(s).all())
        result[group]=((p-y)/s).square().mean()
    if not result:raise ValueError('No observed targets in batch')
    if not torch.stack(finite).all():raise ValueError('Nonfinite observed calibrated prediction/target')
    return torch.stack(list(result.values())).mean(),result
