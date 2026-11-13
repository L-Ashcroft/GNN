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
    plan=prepare_batch(batch)
    if not plan.loss_groups:raise ValueError('No observed targets in batch')
    order=plan.tensors['loss_order'];lengths=plan.tensors['loss_lengths']
    p=prediction.reshape(-1)[order]
    transformed=symlog(batch.targets/batch.scales);y=transformed.reshape(-1)[order]
    # One combined finite check; only observed entries participate. No NaN*0.
    if not (torch.isfinite(p).all() & torch.isfinite(y).all()):
        raise ValueError('Nonfinite observed prediction/target')
    sums=torch.segment_reduce((p-y).square(),'sum',lengths=lengths,unsafe=True)
    group_losses=sums/lengths.to(sums.dtype)
    losses={group:group_losses[i] for i,group in enumerate(plan.loss_groups)}
    # Equal group weighting is explicit; samples/units do not silently dominate.
    return group_losses.mean(),losses
