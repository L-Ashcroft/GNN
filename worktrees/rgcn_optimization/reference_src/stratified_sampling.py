"""Reproducible source/function/structure/parameter sampling without labels."""
import hashlib
import numpy as np
import pandas as pd
from .merge_sample import balanced_budgets


def derived_seed(key,seed=42):return seed+int(hashlib.sha256(key.encode()).hexdigest()[:8],16)


def hierarchical_quotas(specs,counts,total):
    """Source budget is fixed externally; balance function, then actual graph."""
    functions={}
    for s in specs:functions.setdefault(s['circuit_family'],{}).setdefault(s['connectivity_group'],[]).append(s['topology_id'])
    family_counts={f:sum(counts[k] for variants in gs.values() for k in variants) for f,gs in functions.items()}
    family_budgets=balanced_budgets(family_counts,total);quotas={}
    for family,groups in sorted(functions.items()):
        budgets=balanced_budgets({g:sum(counts[k] for k in ks) for g,ks in groups.items()},family_budgets[family])
        for group,keys in sorted(groups.items()):
            quotas.update(balanced_budgets({k:counts[k] for k in keys},budgets[group]))
    return quotas


def parameter_strata(values,key):
    """Joint quantile strata using only design inputs, not performance labels.

    Per-column percentile ranks make units irrelevant. Three fixed-seed random
    projections encode joint coverage; quartiles give <=64 nonempty cells.
    These are sampling descriptors, never fitted training normalization.
    """
    x=np.asarray(values,dtype=float)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Finite design parameter matrix required')
    varying=np.ptp(x,axis=0)>0
    if not varying.any():return np.zeros(len(x),dtype=np.int16)
    ranks=pd.DataFrame(x[:,varying]).rank(method='average',pct=True).to_numpy()
    rng=np.random.default_rng(derived_seed(key));projection=ranks@rng.standard_normal((ranks.shape[1],3))
    codes=np.zeros(len(x),dtype=np.int16)
    for j in range(3):
        edges=np.unique(np.quantile(projection[:,j],[.25,.5,.75]))
        codes+=np.searchsorted(edges,projection[:,j],side='right').astype(np.int16)*(4**j)
    return codes


def sample_parameters(frame,quota,key):
    if not 0<=quota<=len(frame):raise ValueError('Quota exceeds unique candidate pool')
    matrix=np.stack(frame['parameters'].map(lambda x:np.asarray(x,dtype=float)))
    codes=parameter_strata(matrix,key)
    groups={int(c):np.flatnonzero(codes==c) for c in np.unique(codes)}
    # Cover nonempty cells first, then preserve candidate-pool mass as closely
    # as possible. No global labels/test-set statistics enter this selection.
    budgets={c:0 for c in groups};left=quota
    if quota>=len(groups):
        budgets={c:1 for c in groups};left-=len(groups)
    capacity={c:len(ix)-budgets[c] for c,ix in groups.items()}
    while left:
        active=[c for c in groups if capacity[c]>0]
        mass=sum(capacity[c] for c in active)
        ideal={c:left*capacity[c]/mass for c in active}
        allocated={c:min(capacity[c],int(np.floor(ideal[c]))) for c in active}
        if not sum(allocated.values()):
            allocated={c:0 for c in active}
            for c in sorted(active,key=lambda c:(-ideal[c],c))[:left]:allocated[c]=1
        for c,n in allocated.items():budgets[c]+=n;capacity[c]-=n;left-=n
    rng=np.random.default_rng(derived_seed(key));selected=[];stats=[]
    for c,indices in sorted(groups.items()):
        chosen=rng.choice(indices,budgets[c],replace=False)
        selected.extend(chosen.tolist());stats.append({'stratum':c,'candidates':len(indices),'selected':budgets[c]})
    selected=sorted(selected)
    result=frame.iloc[selected].copy();result['parameter_stratum']=codes[selected]
    return result,stats
