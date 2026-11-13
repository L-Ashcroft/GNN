"""Streaming head/scale audit, validation mean baselines; never reads test labels."""
from pathlib import Path
import json
import collections
import math
import sys
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parents[3];PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from src.train.target_calibration import read,sha,combine

def main():
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    stats=read(ROOT/'data/model_ready/fusion_v2_accuracy_v2/target_calibration.json')
    cache=ROOT/'data/model_ready/fusion_v2_training_v1';manifest=read(cache/'manifest.json');tasks=read(cache/'tasks.json')
    # Only completed validation records; never load a model/checkpoint/GPU.
    history_path=ROOT/'artifacts/experiment1_fusion_id_fast_v1/history.json'
    history=read(history_path);latest=history[-1];models={(r['supervision_group'],r['semantic_id']):r for r in latest['validation']['per_target']}
    epoch=latest['epoch'];del history
    acc={};validation_groups=collections.Counter()
    for name,spec in manifest['topologies'].items():
        folder=ROOT/spec['folder'];role=np.load(folder/'role.npy',mmap_mode='r');groups=np.load(folder/'group.npy',mmap_mode='r')
        target=np.load(folder/'target.npy',mmap_mode='r');mask=np.load(folder/'mask.npy',mmap_mode='r')
        rows=np.flatnonzero(role==1)
        for number in np.unique(groups[rows]):
            group=manifest['group_names'][int(number)];take=rows[groups[rows]==number];validation_groups[group]+=len(take)
            for slot,rule in tasks[group]['slots'].items():
                column=int(slot);valid=mask[take,column];values=np.asarray(target[take[valid],column],dtype=np.float64)
                if not len(values):continue
                item=stats['groups'][group]['slots'][slot]
                if not item['count']:raise ValueError('Observed validation slot without training observations')
                center=math.copysign(math.expm1(abs(item['center'])),item['center'])*rule['scale']
                a=acc.setdefault((group,slot),{'moments':(0,0.,0.),'mean_sse':0.,'mean_sae':0.,'calibrated_sse':0.,'calibrated_sae':0.})
                a['moments']=combine(*a['moments'],values)
                for prefix,p in [('mean',item['training_raw_mean']),('calibrated',center)]:
                    errors=values-p;a[prefix+'_sse']+=float(np.square(errors).sum());a[prefix+'_sae']+=float(np.abs(errors).sum())
    rows=[]
    for (group,slot),a in sorted(acc.items()):
        rule=tasks[group]['slots'][slot];item=stats['groups'][group]['slots'][slot];n,mean,m2=a['moments'];m=models[(group,rule['semantic_id'])]
        rows.append({'group':group,'slot':slot,'source_dataset':stats['groups'][group]['source_dataset'],
            'semantic_id':rule['semantic_id'],'train_count':item['count'],'validation_count':n,
            'training_symlog_std':item['empirical_std'],'effective_std':item['std'],'validation_raw_std':math.sqrt(max(0.,m2/n)),
            'model_mae':m['mae'],'model_r2':m['r2'],
            'train_mean_baseline_mae':a['mean_sae']/n,'train_mean_baseline_r2':1-a['mean_sse']/m2 if n>=2 and m2>0 else None,
            'calibrated_mean_baseline_mae':a['calibrated_sae']/n,
            'calibrated_mean_baseline_r2':1-a['calibrated_sse']/m2 if n>=2 and m2>0 else None})
    by_source={}
    for source in sorted({r['source_dataset'] for r in stats['groups'].values()}):
        groups=[(g,r) for g,r in stats['groups'].items() if r['source_dataset']==source];counts=[r['training_rows'] for _,r in groups]
        targets=[r for r in rows if r['source_dataset']==source]
        by_source[source]={'contracts':len(groups),'train_rows_per_contract_min_median_max':[min(counts),float(np.median(counts)),max(counts)],
            'groups_below_adapter_threshold':sum(n<256 for n in counts),'target_combinations':len(targets),
            'model_worse_than_train_mean_mae':sum(r['model_mae']>r['train_mean_baseline_mae'] for r in targets),
            'model_worse_than_calibrated_mean_mae':sum(r['model_mae']>r['calibrated_mean_baseline_mae'] for r in targets),
            'median_model_r2':float(np.median([r['model_r2'] for r in targets if r['model_r2'] is not None]))}
    prior=read(PACKAGE/'baseline_snapshot.json')
    assert all(sha(ROOT/name)==value for name,value in prior['protected'].items())
    report={'status':'audited','analyzed_epoch':epoch,'test_labels_used':False,'training_modified':False,
        'calibration_fit_validation_labels_used':False,'calibration_fit_test_labels_used':False,
        'validation_usage':'diagnostic comparison only; not used for fitting mean/std or adapter eligibility',
        'existing_architecture':'independent final readouts for 142 contracts; shared nonlinear family blocks across sources',
        'by_source':by_source,'total_adapters_eligible':sum(r['training_rows']>=256 for r in stats['groups'].values()),
        'unobserved_declared_slots':sum(not s['count'] for r in stats['groups'].values() for s in r['slots'].values()),
        'worst_r2':sorted([r for r in rows if r['model_r2'] is not None],key=lambda x:x['model_r2'])[:12],
        'per_target':rows,'baseline_implementation_unchanged':True}
    (PACKAGE/'results/head_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'epoch':epoch,'by_source':by_source,'adapters':report['total_adapters_eligible'],'declared_but_unobserved_slots':report['unobserved_declared_slots']},indent=2))

if __name__=='__main__':main()
