"""Finalize recorded experiment results without retraining or re-evaluating test."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json
import hashlib
import statistics
ROOT=Path(__file__).resolve().parents[3];PACKAGE=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/experiment1_accuracy_v2_cached_head';CONTROL=ROOT/'monitoring/accuracy_acceleration_v1'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for v in iter(lambda:stream.read(2**20),b''):h.update(v)
    return h.hexdigest()
def main():
    status=read(OUT/'status.json');promotion=read(OUT/'promotion.json');control=read(CONTROL/'status.json')
    if status['state']!='complete' or control['state']!='complete_promoted':raise RuntimeError('Experiment is not yet complete/promoted')
    assert promotion['accepted'];test=read(OUT/'test_metrics.json');history=read(OUT/'history.json');recipe=read(OUT/'recipe.json');gpu=read(OUT/'GPU_acceptance.json')
    parent=read(OUT/'parent_validation.json');selected=history[promotion['head_best_epoch']-1]['validation'];plan=read(CONTROL/'plan.json')
    assert not promotion['test_used_for_selection'] and (OUT/'test_metrics.json').stat().st_mtime>=(OUT/'promotion.json').stat().st_mtime
    for name,value in recipe['source_sha256'].items():assert sha(ROOT/name)==value,name
    assert sha(ROOT/plan['target_calibration'])==recipe['target_calibration_sha256']
    assert sha(ROOT/plan['feature_cache']/'manifest.json')==recipe['feature_cache_manifest_sha256']
    # These are the original metric formulas, not a changed display convention.
    assert sha(PACKAGE/'src/train/metrics.py')==sha(ROOT/'versions/rgcn_cached_fast_v1/src/train/metrics.py')
    paused=read(CONTROL/'parent_paused.json');old=ROOT/'artifacts/experiment1_fusion_id_fast_v1'
    for name,key in [('last.pt','last_checkpoint_sha256'),('best.pt','best_checkpoint_sha256'),('recipe.json','recipe_sha256')]:assert sha(old/name)==paused[key]
    official=read(old/'history.json')[plan['parent_best_epoch']-1]['validation']
    assert abs(parent['macro_hit5']-official['macro_hit5'])<1e-12
    cache=ROOT/plan['feature_cache'];cache_manifest=read(cache/'manifest.json')
    assert cache_manifest['roles']['train']['rows']==400083 and cache_manifest['roles']['validation']['rows']==50861
    assert not cache_manifest['test_features_or_labels_cached']
    for role,spec in cache_manifest['roles'].items():
        for name,value in spec['files'].items():assert sha(cache/role/name)==value
    import torch
    torch.set_num_threads(1)
    parent_weights=torch.load(CONTROL/'parent_best.pt',map_location='cpu',weights_only=True)['model']
    saved=torch.load(OUT/'best.pt',map_location='cpu',weights_only=True)
    for name,value in parent_weights.items():
        if not name.startswith('head.'):assert torch.equal(value,saved['model'][name]),name
    assert saved['epoch']==promotion['head_best_epoch'] and saved['parent_pretraining_epoch']==plan['parent_best_epoch']
    del parent_weights,saved
    source_rows=[]
    for source in ['FALCON','AICircuit','LOADBench']:
        a=parent['by_source'][source];b=selected['by_source'][source];c=test['by_source'][source]
        source_rows.append(f"| {source} | {a['macro_hit5']*100:.2f}% / {a['median_r2']:.3f} | {b['macro_hit5']*100:.2f}% / {b['median_r2']:.3f} | {c['macro_hit5']*100:.2f}% / {c['median_r2']:.3f} |")
    defined=sorted([r for r in test['per_target'] if r['r2'] is not None],key=lambda r:r['r2'])
    validation_lookup={(r['supervision_group'],r['semantic_id']):r for r in selected['per_target']}
    tasks=read(ROOT/'data/model_ready/fusion_v2_training_v1/tasks.json');weak_rows=[];diagnostics=[]
    for row in defined[:6]:
        v=validation_lookup[(row['supervision_group'],row['semantic_id'])];task=tasks[row['supervision_group']]
        topology=task['contract']['topology_id'];unit=row['unit']
        weak_rows.append(f"| {topology} / {row['original_name']} ({unit}) | {row['count']} | {v['mae']:.6g} / {row['mae']:.6g} | {v['r2']:.6g} / {row['r2']:.6g} |")
        diagnostics.append({'topology_id':topology,'metric':row['original_name'],'unit':unit,'test':row,'validation':v})
    diagnostic={'purpose':'post-selection reporting only; no refitting or further test inference','test_labels_used_for_model_selection':False,
        'defined_r2_combinations':len(defined),'positive_r2_combinations':sum(r['r2']>0 for r in defined),
        'r2_below_minus_100':sum(r['r2']<-100 for r in defined),'worst_targets':diagnostics}
    (OUT/'test_diagnostics.json').write_text(json.dumps(diagnostic,indent=2),encoding='utf-8')
    time_mean=statistics.mean(h['training_seconds'] for h in history)
    old_epoch=read(old/'history.json')[plan['parent_best_epoch']-1]['training_seconds']
    cache_bytes=sum(p.stat().st_size for p in cache.rglob('*.npy'))
    categories={}
    for kind,prefix in [('states',None),('loss','GPU_vs_CPU_loss'),('gradient','gradient/'),('adamw','adamw/')]:
        items=[c for c in gpu['comparisons'] if (c['name'].startswith(prefix) if prefix else 'prediction' in c['name'] or 'forward' in c['name'])]
        categories[kind]={'max_abs':max(c['max_abs'] for c in items),'max_acceptance_ratio':max(c['acceptance_ratio'] for c in items)}
    assert gpu['status']=='passed' and all(r['max_acceptance_ratio']<=1 for r in categories.values())
    summary={'at_Asia_Shanghai':datetime.now(timezone(timedelta(hours=8))).isoformat(),'status':'accepted_complete',
        'protocol':'public-fusion ID, 19-epoch graph pretraining plus frozen-backbone calibrated-head stage',
        'parent_pretraining_epoch':plan['parent_best_epoch'],'completed_head_epochs':len(history),'selected_head_epoch':promotion['head_best_epoch'],
        'selected_checkpoint':str((OUT/'best.pt').relative_to(ROOT)),'selected_checkpoint_sha256':sha(OUT/'best.pt'),
        'parent_validation':{k:v for k,v in parent.items() if k!='per_target'},'selected_validation':{k:v for k,v in selected.items() if k!='per_target'},
        'test':{k:v for k,v in test.items() if k!='per_target'},'head_training_epoch_mean_seconds':time_mean,
        'parent_training_epoch_seconds':old_epoch,'phase_training_time_ratio':old_epoch/time_mean,
        'time_ratio_is_changed_frozen_backbone_protocol_not_equivalent_algorithm':True,
        'cache_build_seconds':status['cache_build_seconds'],'feature_cache_bytes':cache_bytes,
        'metric_implementation_unchanged':True,'frozen_graph_weights_exactly_equal_to_parent':True,
        'original_checkpoint_and_recipe_unchanged':True,'cache_source_checks_passed':True,'GPU_checks':categories,
        'test_evaluated_once_after_validation_selection':True}
    (OUT/'final_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    report=f'''# 实验一结果：公开融合数据集ID预测

更新时间：{summary['at_Asia_Shanghai']}。**本轮模型已通过验证门槛，完成固定测试集评价。** 选定模型为原RGCN第{plan['parent_best_epoch']}轮主干加第{promotion['head_best_epoch']}轮校准输出头；原训练检查点完整保留。

## 实际结果

| 指标 | 父模型完整验证集 | 选定模型完整验证集 | 独立测试集 |
| --- | ---: | ---: | ---: |
| macro hit@5% | {parent['macro_hit5']*100:.3f}% | {selected['macro_hit5']*100:.3f}% | {test['macro_hit5']*100:.3f}% |
| macro hit@10% | {parent['macro_hit10']*100:.3f}% | {selected['macro_hit10']*100:.3f}% | {test['macro_hit10']*100:.3f}% |
| R²中位数 | {parent['median_r2']:.6f} | {selected['median_r2']:.6f} | {test['median_r2']:.6f} |
| 原始macro R² | {parent['macro_r2']:.6f} | {selected['macro_r2']:.6f} | {test['macro_r2']:.6f} |

各来源的“hit@5% / R²中位数”：

| 来源 | 父模型验证 | 选定模型验证 | 独立测试 |
| --- | --- | --- | --- |
{chr(10).join(source_rows)}

所有指标先在契约/指标内计算，再按原规则汇总；对数单位不强行计算物理相对误差，近零目标仍按原工程阈值处理。R²公式及负值保留规则未修改，实际源码哈希与父模型一致。验证与测试分区的指标支持数量可能不同，不能把两者差异全部归于模型泛化。

### 测试集仍存在的明显弱项

测试集1479个R²有定义的组合中，1392个为正，10个低于-100；其余未定义项仍保留MAE和真实观测Mask。少数极端项使原始macro R²仍为大负数，不能只展示较好的中位数。

| 目标 | 测试样本数 | 验证MAE / 测试MAE | 验证R² / 测试R² |
| --- | ---: | --- | --- |
{chr(10).join(weak_rows)}

小样本和测试标签窄方差会放大负R²，但部分目标也有较大的物理单位误差，不能据此说它们预测正常。例如同一vout_max目标在验证与测试的MAE相近，R²却差异巨大，说明必须同时看MAE、样本数和R²。本轮模型与指标保持冻结，以上分析仅作报告，不利用测试结果继续拟合或改变本轮选择。明细为 `test_diagnostics.json`。

## 数据与训练口径

主集合500000条，FALCON200000、AICircuit100000、LOADBench200000。原分层划分不变：train400083、validation50861、test49056。图、器件参数、标签定义、监督契约、单位和缺失Mask保持原样。

本轮是两阶段训练，不能写成“端到端训练{plan['parent_best_epoch']+len(history)}轮”：

1. 原图编码器、4层128维48关系RGCN、FiLM和Pooling在原训练分区完成{plan['parent_best_epoch']}轮预训练。
2. 固定全部图级embedding生成参数，eval模式提取训练/验证特征一次。原始网表仍用于推理，不用标签代替图特征。
3. 只用训练标签拟合逐契约/指标的symlog校准；用完整训练分区判断哪些读出行不如固定训练均值，显式初始化这些行，其余做仿射重参数化。
4. 保留原独立监督头，不增加137个非线性分支。输出头Batch1024、AdamW学习率2e-4，最多6轮、patience2；实际完成{len(history)}轮，验证选择第{promotion['head_best_epoch']}轮。

原端到端训练停在第20轮第7批，保存的last/best/recipe哈希与暂停时一致。新阶段使用新优化器、recipe和独立结果目录，未绕过旧recipe直接续训。原有参数预算及训练历史归属写入recipe与父快照。

## 效率与正确性

- 一次性特征缓存构建：{status['cache_build_seconds']/60:.2f}分钟，实际Numpy文件约{cache_bytes/2**20:.1f} MiB，未缓存测试特征/标签。
- 输出头训练每轮平均：{time_mean:.2f}秒（{time_mean/60:.2f}分钟）；父模型端到端训练最近一轮：{old_epoch:.2f}秒（{old_epoch/60:.2f}分钟）。阶段训练时间比约{old_epoch/time_mean:.2f}倍，**这来自冻结主干，是改变训练阶段后的成本对比，不是数学等价实现提速**，也不含一次性缓存/验证/测试时间。
- 真实Batch256 GPU检查通过：新头CPU/GPU预测、Loss、梯度、AdamW，以及缓存读出与完整图Forward。可选分组执行：{gpu['grouped_head_accepted']}；无容差放宽。原报告中的48关系与4层主干保持不变，本阶段只计算输出头梯度。
- 每个非head参数逐张量比较，最终选定模型与父检查点完全相同。缓存文件逐文件SHA256核对通过；源样本、Mask、监督组与原数据核对通过。
- 采用条件预先写入计划：完整验证macro hit@5%至少提升0.5个百分点、R²中位数下降不超过0.01。实际已满足。测试集只在验证选择完成后评价一次，未用于选择结构、轮数或归一化。

GPU最大绝对误差及验收比值详见 `GPU_acceptance.json`；完整明细保留，不能把小误差当作验证精度指标。

## 文件与使用

- 模型：[best.pt](../artifacts/experiment1_accuracy_v2_cached_head/best.pt)
- 测试指标：[test_metrics.json](../artifacts/experiment1_accuracy_v2_cached_head/test_metrics.json)
- 最终摘要：[final_summary.json](../artifacts/experiment1_accuracy_v2_cached_head/final_summary.json)
- 逐轮验证：[history.json](../artifacts/experiment1_accuracy_v2_cached_head/history.json)
- 配方与父来源：[recipe.json](../artifacts/experiment1_accuracy_v2_cached_head/recipe.json)
- 采用判定：[promotion.json](../artifacts/experiment1_accuracy_v2_cached_head/promotion.json)

在独立实现 `worktrees/rgcn_accuracy_v2` 下使用 `src.gnn.accuracy_inference.load_model` 加载本模型，再用 `predict_physical(model, batch)` 推理；它恢复校准参数并输出原单位预测和原Mask。旧模型加载器不兼容新增的校准state_dict格式，不能直接拿新Checkpoint去旧入口 --resume。

当前实验得到可复现的ID结果，不代表已解决未知拓扑/未知PDK的零样本泛化；LOTO和跨数据集实验仍需各自划分、训练统计与可训练输出契约。测试结果也不意味着所有目标都达到工程可用精度，尤其是稀疏、近常数指标需逐项查看原始MAE/R²。
'''
    (ROOT/'reports/experiment1_id_results.md').write_text(report,encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k in ['status','completed_head_epochs','selected_head_epoch','selected_validation','test','phase_training_time_ratio']},indent=2))

if __name__=='__main__':main()
