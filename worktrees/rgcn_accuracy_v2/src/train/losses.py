"""Missing-label-safe losses. Each loss is evaluated within a semantic contract."""
import torch


def symlog(value):return value.sign()*torch.log1p(value.abs())


def inverse_symlog(value):
    value=value.to(torch.float64)
    return value.sign()*torch.expm1(value.abs())


def masked_mse(prediction,target,mask,weight=None):
    if prediction.shape!=target.shape or mask.shape!=target.shape:raise ValueError('Loss shape mismatch')
    valid=mask.bool();p=prediction[valid];y=target[valid]
    if not torch.isfinite(p).all() or not torch.isfinite(y).all():raise ValueError('Nonfinite observed prediction/target')
    w=torch.ones_like(target) if weight is None else torch.broadcast_to(weight,target.shape)
    w=w[valid]
    if not torch.isfinite(w).all() or (w<0).any():raise ValueError('Invalid loss weights')
    denominator=w.sum()
    return ((p-y).square()*w).sum()/torch.where(denominator>0,denominator,torch.ones_like(denominator))


def supervision_loss(prediction,batch):
    from ..model.execution import prepare_batch
    plan=prepare_batch(batch);transformed=symlog(batch.targets/batch.scales);losses={};finite=[]
    for group,start,end in plan.loss_slices:
        indices=plan.tensors['loss_order'][start:end]
        p=prediction.reshape(-1)[indices];y=transformed.reshape(-1)[indices]
        finite.append(torch.isfinite(p).all() & torch.isfinite(y).all())
        # Keep the original per-group reduction and multiplication order.
        # Only constant-one weight construction and known mask positions differ.
        w=torch.ones_like(y);denominator=w.sum()
        losses[group]=((p-y).square()*w).sum()/torch.where(denominator>0,denominator,torch.ones_like(denominator))
    if not losses:raise ValueError('No observed targets in batch')
    if not torch.stack(finite).all():raise ValueError('Nonfinite observed prediction/target')
    # Equal group weighting is explicit; samples/units do not silently dominate.
    return torch.stack(list(losses.values())).mean(),losses
