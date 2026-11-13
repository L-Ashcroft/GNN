"""Real-data analytic gradients, unit/mask guards and exact AdamW resume checks."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'1')
import copy
import numpy as np
from common import HERE, OUTPUT, read, atomic, now
from features import Inputs, ridge, denormalize, metric_rows, summarize
from neural import Readout


def main():
    inputs=Inputs(5);name=sorted(inputs.structures)[0]
    block=inputs.block(name,mode='physical');w,b=ridge(block)
    rng=np.random.default_rng(42);head=Readout(block,w,b,rng,dropout=0.)
    head.parameters['output_w'][:]=rng.normal(0,.01,head.parameters['output_w'].shape)
    rows=np.arange(5);x=block['train_x'][rows];objective,grads=head.backward(x,rows)
    errors=[]
    for key in head.parameters:
        p=head.parameters[key]
        for flat in [0,p.size//2,p.size-1]:
            old=p.flat[flat];eps=1e-5
            p.flat[flat]=old+eps;plus=head.backward(x,rows)[0]
            p.flat[flat]=old-eps;minus=head.backward(x,rows)[0]
            p.flat[flat]=old
            finite=(plus-minus)/(2*eps);analytical=grads[key].flat[flat]
            np.testing.assert_allclose(analytical,finite,atol=2e-6,rtol=2e-4)
            errors.append(abs(finite-analytical))
    # A missing target contributes neither loss nor gradient.
    head.mask[0,0]=False
    reference,reference_g=head.backward(x,rows)
    old=head.target[0,0];head.target[0,0]=1e30
    changed,changed_g=head.backward(x,rows)
    assert reference==changed
    for key in reference_g:np.testing.assert_array_equal(reference_g[key],changed_g[key])
    head.target[0,0]=old
    head.dropout=.05
    _,g=head.backward(x,rows,rng);head.update(g,1e-4,1e-3)
    state={k:v.copy() for k,v in head.state().items()};rng_state=copy.deepcopy(rng.bit_generator.state)
    restored=Readout(block,w,b,np.random.default_rng(99),dropout=.05);restored.restore(state)
    restored.mask=head.mask.copy();restored.target=head.target.copy()
    r2=np.random.default_rng();r2.bit_generator.state=rng_state
    _,g=head.backward(x,rows,rng);head.update(g,1e-4,1e-3)
    _,g=restored.backward(x,rows,r2);restored.update(g,1e-4,1e-3)
    for key in head.parameters:np.testing.assert_array_equal(head.parameters[key],restored.parameters[key])
    zero=np.zeros((len(block['validation_x']),len(head.slots)))
    pred=denormalize(zero,block);metrics=metric_rows(name,pred,block)
    assert len(metrics)==len(head.slots) and all(np.isfinite(r['rmse']) for r in metrics)
    assert block['normalization']['validation_or_test_statistics_used'] is False
    for topology in inputs.structures:
        assert len(inputs.roles[topology]['train'])==5
        assert not {r['sample_id'] for r in inputs.roles[topology]['train']}&{r['sample_id'] for r in inputs.roles[topology]['validation']}
    # Reject a checkpoint whose backbone has seen a larger target budget.
    try: inputs.parent_check('pretrained_k100_seed42');raise RuntimeError('Budget guard failed')
    except AssertionError:pass
    result={'passed':True,'at':now(),'real_train_rows':5,'test_samples_used':0,
            'gradient_max_abs_error':float(max(errors)),'gradient_tolerance':{'atol':2e-6,'rtol':2e-4},
            'missing_label_loss_and_gradient_unchanged':True,'dropout_AdamW_resume_exact':True,
            'budget_cross_contamination_rejected':True,'normalization_train_only':True,
            'original_labels_masks_and_roles_modified':False,'GPU_tested':False}
    atomic(HERE/'checks.json',result);print(result)


if __name__=='__main__':main()
