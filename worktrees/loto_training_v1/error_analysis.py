"""Read-only RMSE sidecar, no trainer changes and no premature outer-test access.

Recorded validation R2 plus its exact target population recovers SSE/RMSE for
nonconstant targets. UndefinedR2 is never treated as missing simulation data.
After frozen test predictions exist, compute exactRMSE and absolute-error tails.
"""
from pathlib import Path
from datetime import datetime,timezone
import os,json,time,hashlib,argparse,statistics,math,msvcrt
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def atomic(p,v):
    q=p.with_suffix('.analysis.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
    for attempt in range(50):
        try:os.replace(q,p);return
        except PermissionError:time.sleep(.1)
    raise RuntimeError('Cannot publish sidecar')
def moments(reader_manifest,protocol,refs):
    result={};alias=protocol['head_alias_map'];names=reader_manifest['group_names']
    for number in np.unique(refs[:,0]):
        topology=protocol['topologies'][int(number)];rows=refs[refs[:,0]==number,1];folder=ROOT/reader_manifest['topologies'][topology]['folder']
        target=np.load(folder/'target.npy',mmap_mode='r');mask=np.load(folder/'mask.npy',mmap_mode='r');groups=np.load(folder/'group.npy',mmap_mode='r')
        for code in np.unique(groups[rows]):
            g=alias.get(names[int(code)],names[int(code)]);selected=rows[groups[rows]==code]
            y=target[selected].astype('float64');valid=mask[selected]
            if not np.isfinite(y[valid]).all():raise ValueError('Invalid observed reference label')
            for slot in range(16):
                values=y[valid[:,slot],slot]
                if not len(values):continue
                key=g+'::slot'+str(slot);n=len(values);mean=float(values.mean());m2=float(np.square(values-mean).sum())
                if key not in result:result[key]={'count':n,'mean':mean,'m2':m2}
                else:
                    old=result[key];total=old['count']+n;delta=mean-old['mean'];old['m2']+=m2+delta*delta*old['count']*n/total;old['mean']+=delta*n/total;old['count']=total
        for a in [target,mask,groups]:a._mmap.close()
    return result
def reconstruct(rows,tasks,train,val):
    out=[]
    for row in rows:
        g=row['supervision_group'];slot=next(j for j,rule in tasks[g]['slots'].items() if rule['semantic_id']==row['semantic_id']);key=g+'::slot'+slot
        v=val[key];tr=train[key];assert v['count']==row['count']
        if row['r2']==1. and row['mae']>0:
            rmse=None;method='RecordedR2 rounded to1 despite positiveMAE; exact predictions required'
        elif row['r2'] is not None:
            difference=1.-row['r2']
            if difference<-1e-12:raise ValueError('R2 exceeds numerical upper bound')
            rmse=math.sqrt(max(0.,difference)*v['m2']/v['count']);method='Recovered from recordedR2 and same-label validationM2; floating-order rounding may differ slightly'
        elif row['count']==1:rmse=row['mae'];method='Single observation:RMSE equalsMAE'
        elif row['mae']==0:rmse=0.;method='All absolute errors zero'
        else:rmse=None;method='Cannot recoverSSE from undefinedR2+MAE; requires predictions'
        std=math.sqrt(tr['m2']/tr['count']) if tr['count'] else 0.
        if rmse is not None and rmse+1e-8*max(1.,row['mae'])<row['mae']:raise ValueError('Recovered RMSE below MAE; inconsistent population or rounding')
        out.append(dict(row,rmse=rmse,mse=rmse*rmse if rmse is not None else None,rmse_method=method,
            reference_train_physical_std=std,nrmse_train_std=rmse/std if rmse is not None and std>0 else None,
            nrmse_policy='Divide by this fold observed TRAIN physical standard deviation; undefined when training variance is zero',
            physical_unit=row['unit']))
    return out
def exact_rows(path,tasks,train):
    import pyarrow.parquet as pq
    rows=pq.read_table(path).to_pylist();values={}
    for row in rows:
        g=row['supervision_group']
        for slot,valid in enumerate(row['mask']):
            if not valid:continue
            rule=tasks[g]['slots'][str(slot)];key=(g,slot)
            error=abs(float(row['prediction'][slot])-float(row['target'][slot]));values.setdefault(key,[]).append(error)
    out=[]
    for (g,slot),errors in values.items():
        e=np.asarray(errors,dtype='float64');rule=tasks[g]['slots'][str(slot)];tr=train[g+'::slot'+str(slot)];std=math.sqrt(tr['m2']/tr['count']) if tr['count'] else 0.
        out.append({'supervision_group':g,'semantic_id':rule['semantic_id'],'original_name':rule['original_name'],'unit':rule['unit'],'count':len(e),
            'mae':float(e.mean()),'rmse':float(np.sqrt(np.square(e).mean())),'mse':float(np.square(e).mean()),'absolute_error_p50':float(np.quantile(e,.5)),
            'absolute_error_p90':float(np.quantile(e,.9)),'absolute_error_p95':float(np.quantile(e,.95)),'absolute_error_max':float(e.max()),
            'reference_train_physical_std':std,'nrmse_train_std':float(np.sqrt(np.square(e).mean())/std) if std>0 else None,
            'rmse_method':'Directly computed from frozen outer-test predictions after model selection'})
    return out
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args();cfg=read(HERE/'config.json');output=ROOT/cfg['output'];dest=output/'error_analysis';dest.mkdir(parents=True,exist_ok=True)
    control=ROOT/'monitoring/experiment2_loto_error_analysis_v1';control.mkdir(parents=True,exist_ok=True)
    lock=(control/'watcher.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    try:
        protocol=read(ROOT/cfg['protocol']);fold=next(f for f in protocol['folds'] if f['fold_id']==cfg['fold_id']);raw=read(ROOT/'data/model_ready/fusion_v2_training_v1/manifest.json');tasks=read(output/'task_registry.json')
        with np.load(ROOT/fold['path'],allow_pickle=False) as f:tr=f['train_references'];va=f['validation_references']
        proof={'fold_sha256':fold['sha256'],'raw_manifest_sha256':sha(ROOT/'data/model_ready/fusion_v2_training_v1/manifest.json'),
            'metric_source_sha256':sha(ROOT/'versions/rgcn_calibrated_head_v1/src/train/metrics.py'),'training_rows':len(tr),'validation_rows':len(va),'outer_test_access_before_predictions':False}
        if (dest/'reference_moments.json').exists():
            reference=read(dest/'reference_moments.json');assert reference['proof']==proof
        else:
            reference={'proof':proof,'train':moments(raw,protocol,tr),'validation':moments(raw,protocol,va)};atomic(dest/'reference_moments.json',reference)
        last_epoch=-1;outer_done=False;history_mtime=None;h=[]
        while True:
            if (control/'STOP').exists():return
            history_path=output/'history.json'
            if history_path.exists() and history_path.stat().st_mtime_ns!=history_mtime:
                history_mtime=history_path.stat().st_mtime_ns;h=read(history_path)
            if h and h[-1]['epoch']!=last_epoch:
                # Original validation target-population proof, independent of
                # model loading or any current mutable optimizer/checkpoint.
                for record in h:
                    path=dest/('validation_epoch_'+str(record['epoch'])+'.json')
                    if not path.exists() or read(path).get('analysis_source_sha256')!=sha(Path(__file__)):
                        atomic(path,{'epoch':record['epoch'],'role':'validation','analysis_source_sha256':sha(Path(__file__)),'normalization_reference':'train_only_physical_std',
                            'per_target':reconstruct(record['validation']['per_target'],tasks,reference['train'],reference['validation'])})
                latest=read(dest/('validation_epoch_'+str(h[-1]['epoch'])+'.json'));last_epoch=h[-1]['epoch'];atomic(dest/'latest_validation.json',latest)
                compatible=[r for r in latest['per_target'] if r['supervision_group']==fold['head_alias']]
                report=['# 实验二误差分析：MAE、RMSE和训练尺度归一化误差','',f'已完成验证轮次：{last_epoch}。本分析为独立只读侧车，不修改运行源码、Loss、选择规则或checkpoint。', '',
                    '原训练已记录MAE/MRE/R²/hit@5/10。新增RMSE定义为sqrt(mean((prediction-target)^2))，在原始物理/报告单位内逐契约逐指标计算。MSE的单位是原单位平方，RMSE与原指标相同。不能将Hz、W、V、dB的RMSE直接混成一个总体均值，也不能对归一化训练Loss开根号冒充物理RMSE。','',
                    '训练历史未导出SSE，因此非恒定验证目标用保存的R²和同一批真实标签M2重建：RMSE=sqrt((1-R²)*M2/n)。标签方差计算顺序与原Welford可能有微小舍入差。R²未定义且不能由单样本/零误差精确推断的项保留null，明确需要预测文件；这不会改任何样本Mask。','',
                    'NRMSE=RMSE/训练集原单位标准差，标准差仅由本折train拟合；常量训练目标的NRMSE无定义。dB指标报告绝对dB误差，不作物理相对百分比。最终选择完成且外部逐样本预测文件出现后，将计算精确RMSE及绝对误差P50/P90/P95/最大值。', '',
                    '下面是目标兼容头在已知其他拓扑验证集上的误差，**不是留出SimpleOpAmp4的测试成绩**。','',
                    '| 指标 | 单位 | 验证数 | MAE | RMSE | NRMSE(trainσ) |','| --- | --- | ---: | ---: | ---: | ---: |']
                fmt=lambda v:'未定义/待预测' if v is None else f'{v:.6g}'
                for r in compatible:report.append(f"| {r['original_name']} | {r['unit']} | {r['count']} | {fmt(r['mae'])} | {fmt(r['rmse'])} | {fmt(r['nrmse_train_std'])} |")
                computed=sum(r['rmse'] is not None for r in latest['per_target']);report += ['',f"当前全部{len(latest['per_target'])}个验证目标中，{computed}个可恢复RMSE，其余如实标记待预测。详见 `artifacts/experiment2_loto_v1/{cfg['fold_id']}/error_analysis/latest_validation.json`。",'',
                    'RMSE更敏感于少量大误差；应与MAE、R²、命中率和样本数联合判断，不能只挑一个较好的指标。']
                (ROOT/'reports/experiment2_error_analysis.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
            prediction_path=output/'heldout_predictions.parquet'
            if prediction_path.exists() and (output/'selection_frozen.json').exists() and not outer_done:
                exact=exact_rows(prediction_path,tasks,reference['train']);atomic(dest/'heldout_test_exact_errors.json',{'source_prediction_sha256':sha(prediction_path),'selection_sha256':sha(output/'selection_frozen.json'),'per_target':exact});outer_done=True
                final=['','## 留出拓扑最终测试误差（选择已冻结）','','以下由最终逐样本预测直接计算，不再依赖R²重建。','',
                    '| 指标 | 单位 | 测试数 | MAE | RMSE | 绝对误差P95 |','| --- | --- | ---: | ---: | ---: | ---: |']
                for r in exact:final.append(f"| {r['original_name']} | {r['unit']} | {r['count']} | {r['mae']:.6g} | {r['rmse']:.6g} | {r['absolute_error_p95']:.6g} |")
                with (ROOT/'reports/experiment2_error_analysis.md').open('a',encoding='utf-8') as f:f.write('\n'.join(final)+'\n')
            s=read(output/'status.json');atomic(control/'status.json',{'state':'watching' if args.watch and s['state']!='complete' else 'analysis_complete',
                'pid':os.getpid(),'create_time':psutil.Process().create_time(),'updated_at':now(),'last_analyzed_validation_epoch':last_epoch,'outer_test_exact_errors_available':outer_done,
                'role':'read_only_metrics_sidecar','training_source_modified':False})
            if not args.watch or s['state']=='complete':return
            time.sleep(30)
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
if __name__=='__main__':main()
