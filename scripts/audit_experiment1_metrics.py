"""CPU-only audit of completed validation epochs and train-mean baselines."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from collections import Counter
import gc
import math
from pathlib import Path
import threading
import psutil
import numpy as np
from experiment1_health import ROOT,read,atomic,now,digest,history_summary

OUTPUT=ROOT/'monitoring/experiment1_fusion_id_v1/audit'

def add_moments(acc,values):
    if not len(values):return
    x=np.asarray(values,dtype=np.float64);n=len(x);mean=float(x.mean());m2=float(np.square(x-mean).sum())
    old=acc['count'];total=old+n;delta=mean-acc['mean']
    acc['m2']+=m2+delta*delta*old*n/total;acc['mean']+=delta*n/total;acc['count']=total
    acc['min']=min(acc['min'],float(x.min()));acc['max']=max(acc['max'],float(x.max()))

def moments():return {'count':0,'mean':0.,'m2':0.,'min':float('inf'),'max':float('-inf')}
def finalize(acc):
    if not acc['count']:return {'count':0,'mean':None,'std':None,'min':None,'max':None}
    return dict(acc,std=math.sqrt(max(0.,acc['m2']/acc['count'])))

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True);stop=threading.Event()
    def guard():
        while not stop.wait(.1):
            if psutil.virtual_memory().available<768*2**20:
                atomic(OUTPUT/'memory_abort.json',{'at_UTC':now(),'action':'only audit exits; no training signals'});os._exit(86)
    threading.Thread(target=guard,daemon=True).start()
    production=ROOT/'artifacts/experiment1_fusion_id_v1';history=read(production/'history.json');recipe=read(production/'recipe.json')
    before=read(production/'status.json');summaries=history_summary(history);latest=history[-1]
    config=recipe['config'];cache=ROOT/config['cache'];manifest=read(cache/'manifest.json');tasks=read(cache/'tasks.json')
    protected={p:digest(ROOT/p)==expected for p,expected in recipe['implementation_sha256'].items()}
    protected['cache_manifest']=digest(cache/'manifest.json')==recipe['cache_manifest_sha256']
    protected['split']=digest(ROOT/config['split'])==recipe['split_sha256']
    observed={};errors=[];known_label_rows=0
    for name,spec in manifest['topologies'].items():
        folder=ROOT/spec['folder'];arrays={key:np.load(folder/(key+'.npy'),mmap_mode='r') for key in ['target','mask','role','group']}
        role=arrays['role'];group_index=arrays['group'];target=arrays['target'];mask=arrays['mask']
        selected_role=(role==0)|(role==1)
        for group_number in np.unique(group_index[selected_role]):
            group=manifest['group_names'][int(group_number)];task=tasks[group];member=(group_index==group_number)&selected_role
            declared={int(slot) for slot in task['slots']}
            undeclared=[i for i in range(16) if i not in declared and np.any(mask[member,i])]
            if undeclared:errors.append({'code':'observed_undeclared_slots','topology':name,'slots':undeclared})
            for slot_text,rule in task['slots'].items():
                slot=int(slot_text);key=(group,rule['semantic_id'])
                record=observed.setdefault(key,{'group':group,'semantic':rule['semantic_id'],'name':rule['original_name'],
                    'unit':rule['unit'],'scale':rule['scale'],'topologies':set(),'train':moments(),'validation':moments(),'val_arrays':[]})
                record['topologies'].add(name)
                for code,label in [(0,'train'),(1,'validation')]:
                    indices=np.flatnonzero(member&(role==code)&mask[:,slot]);values=np.asarray(target[indices,slot],dtype=np.float64)
                    if not np.isfinite(values).all():errors.append({'code':'observed_nonfinite_label','topology':name,'slot':slot,'role':label});continue
                    add_moments(record[label],values);known_label_rows+=len(values)
                    if code==1 and len(values):record['val_arrays'].append(values)
        del arrays;gc.collect()
    model_rows={(r['supervision_group'],r['semantic_id']):r for r in latest['validation']['per_target']}
    rows=[];logs={'dB','dBm','dBc/Hz','source_label_dBm_compression_function'}
    for key,record in observed.items():
        train=finalize(record['train']);validation=finalize(record['validation'])
        if key not in model_rows:continue
        model=model_rows[key]
        if model['count']!=validation['count']:errors.append({'code':'validation_count_mismatch','semantic':record['semantic']})
        values=np.concatenate(record['val_arrays']) if record['val_arrays'] else np.empty(0)
        baseline_mae=baseline_r2=baseline_hit5=baseline_hit10=None
        if train['count'] and len(values):
            error=np.abs(values-train['mean']);baseline_mae=float(error.mean())
            baseline_r2=1.-float(np.square(error).sum())/validation['m2'] if validation['m2']>0 and len(values)>=2 else None
            relative=np.abs(values)>1e-8*record['scale']
            if record['unit'] not in logs and np.any(relative):
                ratio=error[relative]/np.abs(values[relative]);baseline_hit5=float((ratio<=.05).mean());baseline_hit10=float((ratio<=.1).mean())
        row={k:v for k,v in record.items() if k not in {'train','validation','val_arrays','topologies'}}
        row.update(train=train,validation=validation,topologies=sorted(record['topologies']),model_r2=model['r2'],model_mae=model['mae'],
            model_hit5=model['hit5'],model_hit10=model['hit10'],model_mre=model['mre'],
            train_mean_baseline_mae=baseline_mae,train_mean_baseline_r2=baseline_r2,
            train_mean_baseline_hit5=baseline_hit5,train_mean_baseline_hit10=baseline_hit10,
            train_std_over_scale=train['std']/record['scale'] if train['std'] is not None else None)
        rows.append(row)
    def mean(field):
        values=[row[field] for row in rows if row[field] is not None and math.isfinite(row[field])]
        return sum(values)/len(values) if values else None
    defined=[r for r in rows if r['model_r2'] is not None];worst=sorted(defined,key=lambda r:r['model_r2'])
    summary={'created_at_UTC':now(),'analyzed_through_epoch':latest['epoch'],'history':summaries,
        'source_pipeline_errors':errors,'fingerprints_match':all(protected.values()),'fingerprints':protected,
        'training_and_validation_observed_entries_checked':known_label_rows,'test_labels_used':False,
        'target_combinations':len(rows),'model_better_than_train_mean_MAE':sum(r['model_mae']<r['train_mean_baseline_mae'] for r in rows if r['train_mean_baseline_mae'] is not None),
        'model_macro_hit5':mean('model_hit5'),'train_mean_baseline_macro_hit5':mean('train_mean_baseline_hit5'),
        'model_macro_hit10':mean('model_hit10'),'train_mean_baseline_macro_hit10':mean('train_mean_baseline_hit10'),
        'small_scaled_variation_targets':sum(r['train_std_over_scale'] is not None and r['train_std_over_scale']<1e-3 for r in rows),
        'worst_targets':worst[:20],'all_targets_file':str((OUTPUT/'per_target.json').relative_to(ROOT)),
        'production_before':before,'production_after':read(production/'status.json')}
    atomic(OUTPUT/'per_target.json',rows);atomic(OUTPUT/'summary.json',summary)
    table='\n'.join(f"| {e['epoch']} | {e['train_loss']:.4f} | {e['validation_loss']:.4f} | {100*e['hit5']:.2f}% | {e['median_r2']:.4f} | {e['positive_r2']}/{e['defined_r2']} | {e['macro_r2']:.2f} |" for e in summaries)
    target_table='\n'.join(f"| {r['topologies'][0]} / {r['name']} | {r['validation']['count']} | {r['validation']['std']:.3g} | {r['model_mae']:.3g} | {r['train_mean_baseline_mae']:.3g} | {r['model_r2']:.3g} |" for r in worst[:10])
    source_table='\n'.join(f"| {source} | {m['target_combinations']} | {100*m['hit5']:.2f}% | {m['median_r2']:.4f} |" for source,m in summaries[-1]['by_source'].items())
    text=f'''# 实验一训练指标审阅与夜间监控

生成时间（UTC）：{summary['created_at_UTC']}。指标审阅覆盖已完成的第 1–{latest['epoch']} 轮验证；新的验证结果由后台监控自动更新 `monitoring/experiment1_fusion_id_v1/history_summary.json`。

**运行稳定性与模型效果分开判断。当前原进程持续训练，没有确认的 NaN/Inf、标签槽位错配或源码变更；但模型效果存在明显弱项，不能称为预测质量正常。** 本次不修改训练口径、不裁剪负 R²、不写测试集结果，也不热改正在运行的模型。

| 轮次 | 训练 Loss | 验证 Loss | 验证 hit@5% | R²中位数 | 正 R²组合数 | 原始 macro R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

最新完整验证按来源划分（均在各自监督契约内计算）：

| 来源 | 契约—指标组合数 | macro hit@5% | R²中位数 |
| --- | ---: | ---: | ---: |
{source_table}

## 1. 已完成的只读检查

- 原训练实现、缓存元数据和划分指纹匹配：{summary['fingerprints_match']}。
- 扫描训练/验证已观测标签条目：{known_label_rows}；发现的非有限观测值、未声明槽位和验证数量错配问题数：{len(errors)}。测试标签未参与。
- 每轮验证 Loss、hit@5/10、R²中位数、正 R²数量、极差组合、梯度统计和逐来源表现已审阅。原始日志保持原样。
- 用每个契约—指标的**训练集均值**建立独立诊断基线，评价同一验证集；没有用验证标签拟合基线，也没有加载当前模型或增加 GPU 测试。

## 2. 已确认的质量问题

R² 为负表示该目标的预测平方误差大于直接使用验证目标均值时的误差。它没有负数下界，因此算术平均可能被个别组合拉低；R² 的不稳定与目标方差有关，不是由目标均值接近零直接造成。

本次最差组合的标签分布与可部署的训练均值基线如下，单位沿用原指标；std 是验证标签标准差：

| 来源拓扑 / 指标 | 验证数量 | 标签 std | 模型 MAE | 训练均值基线 MAE | 模型 R² |
| --- | ---: | ---: | ---: | ---: | ---: |
{target_table}

最差目标的真实分布很窄，而模型误差显著更大，**不能仅用“低方差导致 R² 不好看”掩盖其预测不如常量基线的事实**。极少样本（例如仅 2 条验证记录）的 R² 也不宜孤立作强结论；这些值仍原样保留并标记。

模型比训练均值基线 MAE 更低的组合数为 {summary['model_better_than_train_mean_MAE']}/{len(rows)}。适用相对误差指标的 macro hit@5%：模型 {100*summary['model_macro_hit5']:.2f}%，训练均值基线 {100*summary['train_mean_baseline_macro_hit5']:.2f}%。这是同一验证集的只读诊断，不能替代独立测试结论，也不能证明所有指标都应改成常量预测。

## 3. 需要后续隔离验证的原因

已有 {summary['small_scaled_variation_targets']} 个组合的训练标签标准差相对当前工程参考尺度小于 1e-3。固定单位尺度不等于逐目标方差归一化；毫伏量级摆幅按 1 V 参考尺度参与 symlog Loss 时，相对误差很大也可能只贡献较小 Loss。这是目标尺度/损失权重不均衡的**候选原因**，不是已证明的唯一根因。

当前训练与验证总体仍在改善，不符合仅凭一次负 macro R² 就确定过拟合或程序崩溃的条件。下一版应在隔离实验中比较训练集拟合的逐契约目标变换、物理合理的逐指标尺度及常量/简单模型基线，并重新检查槽位、符号、原始单位与极端标签。拟合只能使用训练集，跨来源监督契约仍不能随意合并。

**这些更改会改变目标变换、梯度或 recipe，不能在同一训练过程中静默切换；本次未修改当前模型、Mask、标签、划分、学习率或归一化。** 确认的监控程序问题则在独立脚本中修复，不影响主训练。

## 4. 夜间监控

独立脚本 `scripts/monitor_experiment1.py` 每 30 秒核对正式进程 PID/创建时间、状态更新时间、内存/磁盘、检查点时效和错误日志；每次新验证写入时重算指标摘要，每 5 分钟核验实现指纹并只读查询显卡状态。监控不导入 Torch，不建立 CUDA 工作，不计算额外验证或测试集预测。

- 当前心跳：[监控状态](../monitoring/experiment1_fusion_id_v1/status.json)。
- 自动更新的逐轮/逐来源摘要：[history_summary.json](../monitoring/experiment1_fusion_id_v1/history_summary.json)。
- 异常事件：[alerts.jsonl](../monitoring/experiment1_fusion_id_v1/alerts.jsonl)。
- 本次完整数据诊断：[summary.json](../monitoring/experiment1_fusion_id_v1/audit/summary.json)、[逐指标明细](../monitoring/experiment1_fusion_id_v1/audit/per_target.json)。

运行故障与模型质量警告分别显示，不会因为负 R² 删除标签、伪造指标或自动修改 Loss。当前恢复策略为仅监控记录、不自动重启；自动断点续训需得到用户对既有“不重启”限制的明确变更。

训练存活时，监控申请 `ES_SYSTEM_REQUIRED | ES_CONTINUOUS` 防止系统因空闲自动休眠，不要求屏幕常亮，也不修改系统电源计划；训练结束时释放。它无法阻止用户主动睡眠、合盖策略、断电或系统故障。依据：[Microsoft SetThreadExecutionState 文档](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate)。监控自身退出后也不能继续观测，因此不承诺故障绝不发生。
'''
    (ROOT/'reports/experiment1_metrics_audit.md').write_text(text,encoding='utf-8');stop.set()
    print('AUDIT',summary['analyzed_through_epoch'],'pipeline errors',len(errors),'fingerprints',summary['fingerprints_match'])
    print('MODEL_HIT5',summary['model_macro_hit5'],'TRAIN_MEAN_BASELINE_HIT5',summary['train_mean_baseline_macro_hit5'])
    print('Model beats train-mean MAE',summary['model_better_than_train_mean_MAE'],'/',len(rows))

if __name__=='__main__':main()
