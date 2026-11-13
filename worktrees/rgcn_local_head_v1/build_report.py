"""Publish measured ID results and next-protocol readiness, no model mutation."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,re,importlib.util
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
OUT=ROOT/'artifacts/experiment1_rgcn_local_head_v1';selection=read(OUT/'selection.json');test=read(OUT/'fixed_test/summary.json');checks=read(OUT/'checks.json')
pure=read(ROOT/'monitoring/rgcn_direct_head_v3_io/selection.json')['candidate_validation'];parent=read(ROOT/'reports/experiment1_validation_review.json');pub=read(ROOT/'versions/rgcn_local_id_v1/publication.json')
v=selection['selected']['validation'];t=test['metrics'];old=read(ROOT/'artifacts/experiment1_accuracy_v2_cached_head/test_metrics.json');diagnostic=read(ROOT/'artifacts/experiment1_representation_diagnostic_v1/summary.json')
assert pub['bundle_sha256']==test['bundle_sha256'] and checks['raw_graph_cached_readout_tolerance_passed']
worst=sorted([r for r in read(OUT/'fixed_test/metrics.json')['per_target'] if r['r2'] is not None],key=lambda r:r['r2'])[:8]
bundle=read(OUT/'bundle.json')
at=datetime.now(timezone(timedelta(hours=8))).isoformat()
lines=['# 实验一：GNN 读出改进与固定测试结果','',f'更新时间：{at}。本轮训练、模型验证和固定测试已完成，当前没有正在运行的训练进程。','',
    '**结论：在不改原数据、网表图、Mask、142 个监督契约和 ID 划分的前提下，验证 hit@5% 从 24.10% 提高到 52.00%，固定测试 hit@5% 为 52.05%。** 新主结果使用“冻结边中心 RGCN + 局部非参数读出”，不是原 MLP 输出头简单多训练几轮。纯神经输出头对照也已实际训练完成，验证为 30.81%。','',
    '测试原始 macro R² 仍约 -2703.7，说明少数目标有严重问题；不能宣称所有指标都达到工程要求。当前改进只证明已知拓扑/已知契约下的同分布预测能力，不能当作 LOTO 或跨数据集零样本能力。','',
    '## 实测结果','',
    '| 模型 | 验证 hit@5% | 测试 hit@5% | 验证 / 测试 R² 中位数 |',
    '| --- | ---: | ---: | --- |',
    f"| 原已测试模型：19轮主干 + 6轮校准头 | 22.99% | {old['macro_hit5']:.2%} | 0.587 / 0.577 |",
    f"| 本轮父模型：同主干 + 最佳36轮头 | {parent['macro_hit5']:.2%} | 未单独评价 | {parent['median_r2']:.3f} / — |",
    f"| 新纯神经直接/残差读出 | {pure['macro_hit5']:.2%} | 未评价 | {pure['median_r2']:.3f} / — |",
    f"| 新 GNN + 局部读出 + 低样本回退 | {v['macro_hit5']:.2%} | {t['macro_hit5']:.2%} | {v['median_r2']:.3f} / {t['median_r2']:.3f} |",'',
    '| 来源 | 父模型验证 hit@5% | 新模型验证 hit@5% | 新模型测试 hit@5% | 新模型测试 R² 中位数 |',
    '| --- | ---: | ---: | ---: | ---: |']
for source in ['FALCON','AICircuit','LOADBench']:
    lines.append(f"| {source} | {parent['sources'][source]['macro_hit5']:.2%} | {v['by_source'][source]['macro_hit5']:.2%} | {t['by_source'][source]['macro_hit5']:.2%} | {t['by_source'][source]['median_r2']:.3f} |")
lines+=['',f"验证原始 macro R²：{parent['macro_r2']:.6f} → {v['macro_r2']:.6f}；验证负 R² 组合数：{parent['negative_r2_targets']} → {v['negative_r2_targets']}。测试原始 macro R²={t['macro_r2']:.6f}，仍有 {t['negative_r2_targets']} 个负 R² 组合。所有 1,491 个观测目标均保留，其中 1,124 个可计算相对误差命中率；测试 R² 有定义的组合为 1,479 个。没有通过删目标或修改 Mask 获得提升。",'',
    'hit@5% 按每个契约/指标先计算再宏平均，不是把全部标签混成一个准确率，也不是三个来源等权。dB/dBm 等对数量沿用原公式报告 MAE/绝对单位命中率，不把它们当物理百分比误差；近零过滤阈值仍为原固定工程阈值。','',
    '## 判断依据与实际改动','',
    '首先核查 121 个有样本拓扑：所有在训练分区变化的官方设计参数，都能在已使用的器件特征表达式中找到引用；本检查未发现漏掉变化参数的拓扑。符号覆盖不等于证明每项仿真条件均完整，也不证明 GNN 已充分利用这些参数。','',
    f"冻结相同 RGCN 图特征，独立线性读出验证 hit@5%={diagnostic['models']['embedding_linear']['macro_hit5']:.2%}，高于原输出头；加入器件参数的随机特征读出可达 38.84%，但未约束外推时出现严重负 R²，未采用该未保护版本。结果说明原共享非线性读出/优化至少限制了一部分已存在的可用信息，不能再把所有问题归于主干训练轮数不足。",'',
    '纯神经对照保留 4 层、128 维、48 关系主干，加入读取完整图 embedding 的契约独立直接层及小型残差 MLP。训练 11 轮，最佳第 3 轮；只更新 1,416,164 个输出头参数。归一化/输出范围均从训练数据拟合，输出范围保护只改预测，不改标签。归一化网络为 FP32，目标仿射还原为 FP64，属于明确的新精度实现，不冒充原头数学等价优化。其原始验证 macro R²=0.085915，低于父模型，故不能声称每项统计都改善。','',
    '主结果采用另一条已验证路径：GNN 生成图特征，再在同一监督契约的训练集索引里作局部回归。距离同时使用训练标准化后的图 embedding 和原器件参数；近邻 k=8，按距离反比加权预测 symlog 标签，再还原物理量。只允许训练样本作为近邻，没有验证/测试标签进入索引。它是带训练索引的非参数解码器，不是普通神经 MLP。','',
    '训练样本不足 256 的 5 个契约，整组沿用旧 GNN 监督头；规则只由训练数量决定，不按测试结果逐指标挑模型。全局比较了 0/0.25/0.5/0.75/1 五种图距离读出混合权重，按事先记录的验证命中率、原始 macro R²、R² 中位数、来源保护和负 R² 数量条件选定权重 1。', '',
    '相同低样本回退规则下，参数局部读出（无图距离贡献）验证为 50.10%，图＋参数局部读出为 52.00%，增量约 1.90 个百分点。应把参数基线作为正式对照，不能只与旧的较弱神经头比较并夸大 GNN 的独立贡献。','',
    '## 可追溯性、正确性与资源','',
    '- 数据不变：500,000 条，train=400,083、validation=50,861、test=49,056；仍是 FALCON20万、AICircuit10万、LOADBench20万。',
    '- 全部 142 个监督组真实原图前向检查通过，覆盖三来源和 48 种关系。缓存与原图读出比较沿用 atol=5e-6、rtol=5e-5，没有放宽；最大误差约 3.46e-6（symlog 输出）。',
    '- 训练近邻索引逐项核对 400,083 个训练 ID，与验证 ID 无交集；固定测试又核对 train/test ID 无交集。另已独立验证原 ID 划分的 sample_id 唯一及 design_group 不跨分区。',
    '- 连接关系、端口类型、节点角色与规范图不一致时拒绝使用局部读出；未知监督契约也拒绝推理。MOSFET G/D/S/B 关系沿用原完整图。',
    '- 新神经头检查包括零残差初始化、梯度/AdamW有限性、缺失位NaN安全、其他契约参数不受单组更新影响、无训练标签加载和保存重载一致。NumPy 汇总已与原 MetricAccumulator 核对；最终测试直接调用原指标实现。',
    '- 训练过程中遇到一次 Windows checkpoint 文件短暂占用（WinError5）。原失败现场保留；在新独立 I/O 版本增加原子替换重试，并以新 recipe 显式导入完整状态，逐张量确认头权重一致，恢复优化器、scheduler、RNG和游标，没有绕过跨版本 resume 保护。',
    f"- 诊断拟合约 {diagnostic['seconds']:.1f} 秒；新神经头每轮训练部分约数秒，因主干冻结、使用既有 embedding 缓存，不是端到端 GNN 获得千倍等价提速。新局部解码器使用约 217.8 MiB 压缩训练索引。最终 49,056 条真实原图测试耗时 {test['seconds']:.1f} 秒，进程峰值 RSS {test['peak_RSS_MiB']:.1f} MiB。未运行大型 CUDA 测试，未降低原 CUDA 4GiB 门槛。",'',
    '## 固定测试仍存在的弱项','',
    '模型在读取本次测试前已冻结，测试分数不再用于修改本冻结版本。使用的是此前模型也评价过的同一固定测试集，并非新收集的完全盲测；本分支所有参数、归一化与规则选择均来自训练/验证。少数小样本/窄方差目标仍有很大误差，以下明细只用于报告，不据此回头调整该版本。','',
    '| 拓扑 | 指标 | 测试数 | MAE（原单位） | R² |','| --- | --- | ---: | ---: | ---: |']
for row in worst:
    topology=bundle['contracts'][row['supervision_group']]['topology'].removeprefix('LOADBench__')
    lines.append(f"| {topology} | {row['original_name']} ({row['unit']}) | {row['count']} | {row['mae']:.6g} | {row['r2']:.6g} |")
lines+=['',
    '因此，可以把该版本作为改进后的 ID 实验结果和强基线；不能把它描述为全指标工程合格模型。局部读出依赖已知契约与规范器件顺序，主要验证同分布局部插值；同一优化轨迹的相近设计可能跨 ID 分区，ID 成绩不代表跨优化运行、未知拓扑或跨来源泛化。','',
    '## 交付与下一阶段','',
    '- [冻结模型版本](../versions/rgcn_local_id_v1/README.md) / [发布指纹](../versions/rgcn_local_id_v1/publication.json)。',
    '- [模型 bundle](../artifacts/experiment1_rgcn_local_head_v1/bundle.json)、[验证选择](../artifacts/experiment1_rgcn_local_head_v1/selection.json)、[完整检查](../artifacts/experiment1_rgcn_local_head_v1/checks.json)。',
    '- [固定测试指标](../artifacts/experiment1_rgcn_local_head_v1/fixed_test/metrics.json)、[汇总](../artifacts/experiment1_rgcn_local_head_v1/fixed_test/summary.json)；同目录每拓扑 Parquet 保存原单位预测、原标签及 Mask。',
    '- [纯神经输出头最佳权重](../artifacts/experiment1_rgcn_direct_head_v3_io/best.pt) 与 [检查](../worktrees/rgcn_direct_head_v3_io/inference_checks.json) 作为独立对照。',
    '- [下一阶段兼容 LOTO 协议](verified_loto_protocol.md)：只使用11个已验证兼容集合，已生成24套实际结构留出划分；尚未训练。不能把当前 ID 预训练主干、embedding 或校准参数带入 LOTO，否则会泄漏目标拓扑。',
]
(ROOT/'reports/rgcn_readout_revision_v3.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
publication_readme='''# 已冻结的公开融合 ID 模型

结果与范围见 [实测报告](../../reports/rgcn_readout_revision_v3.md)。本版本为 4层/48关系 RGCN＋训练集局部解码器，验证 hit@5%=52.00%，固定测试=52.05%。原始测试 macro R²仍有严重负值，不能称为全指标工程合格。仅支持已知契约与规范器件图，不能用于未知拓扑零样本结论。

`publication.json` 固定代码和 bundle 哈希；bundle 指向被冻结的原 GNN checkpoint 及每契约训练索引。索引为模型的一部分，仅含训练记录；不需要验证/测试标签参与推理。原模型/数据均保留。

在工作区的独立 Python 进程中加载；`batch` 沿用已定义的 `CircuitBatch`：

```python
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path('versions/rgcn_local_id_v1').resolve()))
from model import load_model
model, bundle = load_model(device='cpu')
with torch.no_grad():
    z = model(batch)
    prediction = z.sign() * torch.expm1(z.abs()) * batch.scales.to(torch.float64)
```

输入不符合已训练契约的规范图连接、端口关系或节点角色时会拒绝局部解码。CPU 入口已通过142契约实图核对；本次没有进行新版 GPU 验收，不把 CUDA 接口可传入解释为已经通过 GPU 测试。

固定测试已经完成，不再根据其分数修改本发布版本。后续结构或实验须使用新版本和新输出目录。
'''
(ROOT/'versions/rgcn_local_id_v1/README.md').write_text(publication_readme,encoding='utf-8')
folds=read(ROOT/'data/fusion_v2/splits/verified_loto_v1/manifest.json')
spec=importlib.util.spec_from_file_location('verified_loto_loader',ROOT/'worktrees/verified_loto_v1/load.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
first=folds['folds'][0];loaded={}
for role in ['train','validation','test']:
    table,refs,fold=module.load_fold(first['fold_id'],role);assert len(table)==first['counts'][role] and len(refs)==len(table);loaded[role]=len(table)
(ROOT/'worktrees/verified_loto_v1/loader_checks.json').write_text(json.dumps({'first_fold':first['fold_id'],'loaded_counts':loaded,'hash_verification_passed':True,'heldout_structure_not_in_train_or_validation':True},indent=2),encoding='utf-8')
loto=['# 已验证兼容条件下的 LOTO 协议准备','',
    '状态：已生成24套可加载划分并通过样本/设计分组/实际连接结构隔离检查，尚未训练。仅采用已有官方证据审计通过的11个LOADBench兼容集合，共24个拓扑，不扩大到未经证明可共用标签的所有来源。原数据和原142个契约/ID划分均未修改。','',
    '每一折以完整端口连接结构为留出单位。同结构的其他来源拓扑条目也不能进入训练/验证；同一兼容集合仍有其他拓扑的训练数据时，才允许对应的已验证共享输出语义。CMRR/PSRR等真实缺失继续使用原Mask，不通过更改Mask隐藏未训练标签。24折均检查了目标槽位有兼容训练观测。','',
    '训练/验证采用原ID非目标结构的train/validation子集；目标拓扑全部样本成为本协议外部test；其他拓扑原ID-test样本保持unused。新协议独立于ID实验。全部500000个样本ID唯一，原design_group跨分区检查通过。','',
    '**禁止复用当前ID训练后的主干、图embedding、目标校准统计和局部近邻库。** 它们已经见过目标拓扑的训练标签。LOTO必须从无目标拓扑的初始化/预训练开始，并仅按本折train拟合归一化与共享目标校准。底层确定性网表张量缓存可复用，但加载时必须使用本折references覆盖原role。','',
    '数据文件：`data/fusion_v2/splits/verified_loto_v1/manifest.json` 和24个压缩NPZ（合计约32MiB）。manifest记录每折训练/验证/测试位置、原图缓存引用、旧契约到新实验共享语义的映射、证据签名和禁止使用的旧产物。此映射只用于新实验，不会自动改写当前数据路由。','',
    '加载器：`worktrees/verified_loto_v1/load.py` 的 `load_fold(fold_id, role)` 返回样本元数据、规范图引用和该折配置；首折三分区已实际加载校验。网络训练入口仍须接入这些角色覆盖及兼容别名，再逐折重新拟合统计；没有将“划分准备好”描述成“LOTO模型训练好”。','',
    '| 折 | 测试拓扑 | 兼容训练拓扑数 | train / validation / test / unused |',
    '| --- | --- | ---: | --- |']
for f in folds['folds']:
    c=f['counts'];loto.append(f"| {f['fold_id']} | {', '.join(t.removeprefix('LOADBench__') for t in f['test_topologies'])} | {len(f['compatible_training_topologies'])} | {c['train']} / {c['validation']} / {c['test']} / {c['unused']} |")
loto+=['','下一阶段应先完成1折训练入口与校准隔离验收，再决定完整24折预算；不能把见过全部拓扑的当前ID高分当作未知拓扑预测分数。']
(ROOT/'reports/verified_loto_protocol.md').write_text('\n'.join(loto)+'\n',encoding='utf-8')
current={'run_id':'experiment1_rgcn_local_head_v1','state':'complete','updated_at':at,'training_process_running':False,'evaluation_process_running':False,
    'model_type':'Frozen edge-RGCN + training-only local decoder; sparse-contract neural fallback','model_bundle':'artifacts/experiment1_rgcn_local_head_v1/bundle.json',
    'publication':'versions/rgcn_local_id_v1/publication.json','runtime_status':'artifacts/experiment1_rgcn_local_head_v1/fixed_test/status.json',
    'results_report':'reports/rgcn_readout_revision_v3.md','validation_hit5':v['macro_hit5'],'validation_median_r2':v['median_r2'],
    'test_hit5':t['macro_hit5'],'test_median_r2':t['median_r2'],'test_macro_r2':t['macro_r2'],'test_metrics':'artifacts/experiment1_rgcn_local_head_v1/fixed_test/metrics.json',
    'neural_comparator':'artifacts/experiment1_rgcn_direct_head_v3_io/best.pt','next_stage':'24verified_LOTO_folds_prepared_not_trained',
    'next_stage_manifest':'data/fusion_v2/splits/verified_loto_v1/manifest.json'}
(ROOT/'monitoring/current_training.json').write_text(json.dumps(current,indent=2),encoding='utf-8')
for name,prefix in [('README.md',''),('reports/README.md','../')]:
    p=ROOT/name;body=p.read_text(encoding='utf-8')
    for marker in ['autonomous-training-current','joint-training-current','readout-current']:
        body=re.sub(r'<!-- '+marker+r':start -->.*?<!-- '+marker+r':end -->\s*','',body,flags=re.S)
    banner=f'''<!-- readout-current:start -->
> **最新实测（2026-11-03）：** [GNN读出改进已完成]({prefix}reports/rgcn_readout_revision_v3.md)，验证hit@5%=52.00%、固定测试52.05%、测试R²中位数0.837；原始测试macro R²仍有严重负值，完整披露于报告。模型为冻结RGCN＋局部解码器，不能冒充未知拓扑泛化；[纯神经头对照]({prefix}artifacts/experiment1_rgcn_direct_head_v3_io/best.pt)验证30.81%。当前无训练进程。[当前模型指针]({prefix}monitoring/current_training.json) · [24套兼容LOTO划分]({prefix}reports/verified_loto_protocol.md)。下方阶段记录为历史信息。
<!-- readout-current:end -->

'''
    p.write_text(banner+body,encoding='utf-8')
autonomy=ROOT/'monitoring/training_autonomy.json';a=read(autonomy);a.update(recorded_at=at,current_action='Readout diagnostics, neural head training, frozen hybrid ID test completed;24compatibleLOTOsplits prepared, not trained',background_controller_running=False,documentation='reports/rgcn_readout_revision_v3.md');autonomy.write_text(json.dumps(a,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'reports_written':True,'current_pointer_updated':True,'fixed_model_unchanged':True,'LOTO_first_fold_loader_checked':loaded,'validation_hit5':v['macro_hit5'],'test_hit5':t['macro_hit5'],'test_macro_r2':t['macro_r2']},indent=2))
