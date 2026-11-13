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
    transformed=symlog(batch.targets/batch.scales);losses={}
    for group in sorted(set(batch.group_ids)):
        indices=torch.tensor([i for i,g in enumerate(batch.group_ids) if g==group],device=prediction.device)
        if batch.mask[indices].any():
            losses[group]=masked_mse(prediction[indices],transformed[indices],batch.mask[indices])
    if not losses:raise ValueError('No observed targets in batch')
    # Equal group weighting is explicit; samples/units do not silently dominate.
    return torch.stack(list(losses.values())).mean(),losses
