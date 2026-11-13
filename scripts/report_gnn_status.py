"""Current GNN milestone report, grounded in measured construction artifacts."""
import argparse
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read,dump


def main(as_of):
    baseline=read(ROOT/'reports/edge_rgcn_build.json')
    film=read(ROOT/'reports/lecture5_implementation.json')
    dataset=read(ROOT/'data/fusion_v2/manifest.json')
    verified={}
    for report in [baseline,film]:
        for name,expected in report['implementation_sha256'].items():
            actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            if actual!=expected:raise ValueError('Construction report is stale: '+name)
            verified[name]=actual
    for name,expected in dataset['metadata_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError('Dataset metadata changed: '+name)
    chosen=next(h for h in film['history'] if h['epoch']==film['best_epoch'])
    rows=['# GNN 阶段完成情况报告',
          '',f'核对日期：{as_of}（Asia/Shanghai）。',
          '', '**结论：边中心 RGCN 的第一版网络、数据接口和训练流程已经做好，并通过真实小样本检查；正式训练、可靠的性能预测模型和泛化实验尚未完成。**',
          '', '目前可以说“已有一套能够训练和验证的模型工程”，还不能说“已有能够替代仿真、可靠预测电路性能的模型”。现有 checkpoint 均为自检产物。',
          '', '## 1. 哪些已经完成',
          '', '| 部分 | 当前状态 | 实现位置 |',
          '| --- | --- | --- |',
          '| 数据接入 | 对接三来源统一数据、图、参数、Mask 和语义契约 | `src/graph/builder.py` |',
          '| 图输入 | 34 维节点特征、90 维边输入，保留电源/地及未知值标志 | `src/graph/builder.py` |',
          '| 主干 | 默认 4 层、隐藏维 128；节点和边状态逐层更新，关系内归一化、残差、LayerNorm | `src/model/layers.py` |',
          '| 关系 | 40 种原图有向端口关系，另有 8 种互感引用消息；不将引用当作物理导线 | `src/graph/builder.py` |',
          '| 图级读出 | 节点与边注意力池化，保留注意力供后续分析 | `src/model/layers.py`、`network.py` |',
          '| 域感知 | 来源 × 实际工艺域表；每层独立、恒等初始化的 FiLM | `src/model/film.py` |',
          '| 输出与监督 | 最多 16 槽；物理指标族共享中间特征，最终读出按监督契约隔离 | `src/model/heads.py` |',
          '| 损失 | 保号变换、缺失 Mask、已知标签数分母、分组损失；不把 NaN×0 当作屏蔽 | `src/train/losses.py` |',
          '| 优化阶段 | 预训练、冻结主干适配、全参分层学习率；检查真实冻结参数 | `src/train/stage.py` |',
          '| 训练与验证 | 角色隔离、梯度裁剪、余弦调度、验证早停、最优权重及对应训练覆盖恢复 | `src/train/loop.py` |',
          '| 指标 | 还原计量单位，逐契约/语义统计；不混算 W、Hz 和不同 Gain | `src/train/metrics.py` |',
          '| 预测约束 | 默认拒绝自检 checkpoint；检查未训练头、指标槽及域 | `src/gnn/inference.py` |',
          '', '网络装配位于 [`src/model/network.py`](../src/model/network.py)。你之前打开的 `src/gnn/model.py` 现在是兼容入口，不再放完整网络；旧导入方式仍可使用。',
          '', '```text',
          '图与器件参数 → 确定性特征变换 → 节点/边编码',
          '                                 ↓',
          '                    4 层边中心 RGCN + 可选 FiLM',
          '                                 ↓',
          '                    节点/边注意力池化与融合',
          '                                 ↓',
          '              物理指标族特征层 + 独立监督契约输出头',
          '                                 ↓',
          '                   symlog 预测 → 还原计量单位',
          '```',
          '', '## 2. 模型规模与数据范围',
          '', '| 项目 | 数值 |',
          '| --- | ---: |',
          f"| 基础版总参数 | {baseline['total_parameters']:,} |",
          f"| FiLM 版总参数 | {film['stage_parameter_counts']['pretrain']['total']:,} |",
          f"| FiLM 版冻结主干适配时可训练参数 | {film['stage_parameter_counts']['finetune_head']['trainable']:,} |",
          f"| 可接入的数据总量 | {dataset['sample_count']:,} |",
          f"| 有样本的来源拓扑条目 / 连接结构类 | {dataset['sampled_topology_entries']} / {dataset['full_port_structure_classes']} |",
          f"| 注册的保守监督契约 | {baseline['supervision_heads_registered']} |",
          '', '“可接入 50 万条”不表示“已用 50 万条训练”。142 个输出契约也不表示 142 个头都已优化；基础自检只覆盖其中 9 组，并按指标槽位记录实际训练覆盖。',
          '', '## 3. 已做的验证及其证明范围',
          '', '### 3.1 自动测试',
          '', '本次重新运行 `python -B -m unittest discover -s tests`，**68 项通过**。覆盖以下工程性质：',
          '', '- 图批次隔离、节点重编号和边排序不变性、关系归一化、互感参数参与梯度。',
          '- 每一层边更新收到梯度；固定边状态、FiLM 和关系折叠开关可运行。',
          '- FiLM 恒等初始化、域参数展开、真实模块冻结、阶段切换解冻。',
          '- 缺失标签不贡献误差或梯度；负值保号变换可还原；损失分母正确。',
          '- 验证样本不能用于训练；验证不修改模型；最优权重使用独立副本。',
          '- 基础和 FiLM 自检 checkpoint 可加载，未训练的预测输出有独立可用性标记。',
          '', '**通过测试证明的是代码接口和已检查性质正确，不证明预测精度、物理机理解释或泛化能力已经成立。**',
          '', '### 3.2 基础 RGCN 自检',
          '', f"使用三来源、{baseline['source_topologies']} 个拓扑的 {baseline['real_training_samples']} 条 train 样本，执行 {baseline['optimizer_steps']} 步优化。",
          f"同一诊断批次的 eval 模式损失从 {baseline['initial_eval_loss']:.6f} 降至 {baseline['final_eval_loss']:.6f}；记录的平均耗时为 {baseline['seconds_per_step']:.3f} 秒/步（CPU 4 线程，36 条诊断批次）。",
          '这说明优化和反向传播能够工作。这是同一批样本上的结果，没有独立测试精度含义，也不能用于推算整套 50 万条的训练耗时。',
          '', '### 3.3 FiLM 训练流程自检',
          '', f"使用 {film['train_samples']} 条 train、{film['validation_samples']} 条 validation 样本运行 {film['epochs']} 轮，未使用 test 样本。恢复的是 epoch={film['best_epoch']} 的最优状态（编号从 0 开始）。",
          '', '| epoch | 训练损失 | 验证损失 | 验证 macro hit@5% | 验证 macro R² |',
          '| ---: | ---: | ---: | ---: | ---: |']
    for h in film['history']:
        v=h['validation'];hit=v['macro_hit5']
        rows.append(f"| {h['epoch']} | {h['training']['loss']:.6f} | {v['loss']:.6f} | {hit:.1%} | {v['macro_r2']:.3f} |")
    rows += ['', '**这些验证结果没有达到可用预测器的水平。** 每个参与任务只有极少量验证观测，目标方差很小时 R² 会出现非常大的负数；它不能作为正式泛化结论。两轮自检也没有证实 FiLM 有益。基础版和 FiLM 版的批次安排、优化步数不同，不能据此比较两个版本优劣。',
        'macro hit@5% 只汇总适合相对误差且远离零的目标；dB 等对数量不按百分比误差评价。宏 R² 是先按契约/指标计算再平均，未将不同单位原值混在一起计算。',
        '', '### 3.4 现有权重的用途',
        '', '| 权重目录 | 用途 |',
        '| --- | --- |',
        '| `artifacts/edge_rgcn_smoke/` | 基础版自检权重、预测和注意力；不是正式模型 |',
        '| `artifacts/lecture5_smoke/` | FiLM 训练/验证流程自检权重；不是正式模型 |',
        '', '正式预测接口默认拒绝上述权重。显式允许自检模式也只用于检查代码路径，不代表输出已具备工程可靠性。',
        '', '## 4. 尚未完成及主要限制',
        '', '| 待完成工作 | 原因与验收依据 |',
        '| --- | --- |',
        '| 正式训练运行 | 尚未在既定训练划分上完成长期训练、验证选择和独立测试 |',
        '| 长训练入口与恢复 | 需完善运行配置、预算/日志，以及优化器、调度器、随机状态的持久化和断点续训；目前检查脚本不等同于完整训练任务入口 |',
        '| 标签共享任务审核 | 当前逐拓扑契约较保守，未启用 11 个已审核 LOADBench 合并集合；FiLM 不解决标签定义不一致 |',
        '| 跨拓扑有效评估 | 输出头必须在训练中有对应可比任务覆盖；不能拿未训练头做未知拓扑测试并声称泛化 |',
        '| 跨来源有效评估 | 尚无已批准的跨来源共享性能标签，不能直接把同名 Gain/Bandwidth 混成一个目标 |',
        '| 公平基线与消融 | 尚无正式 GCN/MLP/节点中心等基线、等预算对照或 FiLM 有效性证据 |',
        '| 多种子与逐折结果 | 尚未完成重复训练、逐折报告、方差与负结果分析 |',
        '| 资源与稳定性 | 小批次可运行，仍需有代表性批次的内存、吞吐和长时稳定性实测 |',
        '', '现有模型仅用公开网表与设计输入，不含版图；当前结果不能说明后仿寄生效应已被准确预测，也不能声称替代 SPICE。',
        '', '## 5. 接下来建议的执行顺序',
        '', '1. **固定首个训练任务**：选定义和测试条件可比的 LOADBench 指标集合，明确是否按白名单共享目标头；保留其他来源的独立任务。',
        '2. **补齐正式运行配置和断点恢复**：固定协议、种子、模型开关、目标尺度、训练预算和输出目录；核查训练/验证/测试及头、域的覆盖。',
        '3. **先做中等规模试训**：验证学习曲线、内存与吞吐，再确定全量训练预算；所有拟合统计量仅用训练集。',
        '4. **完成 ID 正式训练和独立测试**：报告逐语义指标，不只看总损失下降。',
        '5. **再做可比目标下的跨拓扑/少样本实验和消融**：统一数据、预算及随机种子，保留每折结果；跨来源可比标签未获确认前不作相应精度结论。',
        '', '这项任务的下一阶段重点是“定义可训练的共享任务并完成训练实验”，而不是继续无依据地增加网络复杂度。',
        '', '## 6. 入口和复现',
        '', '- [完整网络](../src/model/network.py)、[FiLM](../src/model/film.py)、[训练循环](../src/train/loop.py)。',
        '- [基础版构建与实测](edge_rgcn_build.md)、[讲义 5 落地与代码评审](lecture5_implementation.md)。',
        '- [监督组兼容性](fusion_supervision_groups.md)、[图特征检查](graph_lecture12_audit.md)。',
        '- 机器记录：`reports/edge_rgcn_build.json`、`reports/lecture5_implementation.json`。本次生成报告前已核对其代码哈希与当前实现一致。',
        '', '```powershell',
        r'.\.venv\Scripts\python.exe -B -m unittest discover -s tests',
        '# 以下两条会重新运行并覆盖各自的自检产物，不启动全量训练',
        r'.\.venv\Scripts\python.exe -B scripts/check_edge_rgcn.py --steps 12 --threads 4',
        r'.\.venv\Scripts\python.exe -B scripts/check_lecture5.py',
        r'.\.venv\Scripts\python.exe -B scripts/report_gnn_status.py --as-of 2026-10-28',
        '```',
        '', '本次工作只核对状态并生成报告，未重新训练、未改动模型结构或数据集。']
    (ROOT/'reports/gnn_status.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    dump(ROOT/'reports/gnn_status.json',{'as_of':as_of,'engineering_status':'first_version_implemented_and_smoke_tested',
        'predictor_status':'not_formally_trained_or_validated','full_training_completed':False,
        'baseline_parameters':baseline['total_parameters'],'film_parameters':film['stage_parameter_counts']['pretrain']['total'],
        'selected_validation_epoch':film['best_epoch'],'selected_validation_hit5':chosen['validation']['macro_hit5'],
        'selected_validation_macro_r2':chosen['validation']['macro_r2'],
        'verified_implementation_sha256':verified,'dataset_index_sha256':dataset['index_sha256']})
    print('Wrote reports/gnn_status.md and gnn_status.json; no training performed')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--as-of',default='2026-10-28')
    main(parser.parse_args().as_of)
