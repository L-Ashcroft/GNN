# 第六版后续受控实验与投稿压缩安排

核查日期：2026-11-12。本文是可执行安排，不是已完成实验的结果表。当前正式队列和配方保持不变，不启动第二个并行训练器。

## 1. 已有产物与证据边界

| 项目 | 现有状态 | 已核实入口 |
| --- | --- | --- |
| 同神经头冷启动ID | 已完成11轮；第12轮断点保留，M02后自动恢复，尚无最终测试 | `worktrees/rgcn_memory_safe_id_control_v2/` |
| 仅移除FiLM | 已有配置、归一化及小规模运行检查；未完成训练测试 | `worktrees/rgcn_memory_safe_no_film_v2/` |
| 关闭动态边更新 | 同上 | `worktrees/rgcn_memory_safe_static_edges_v2/` |
| 将关系合并为一种 | 同上 | `worktrees/rgcn_memory_safe_collapsed_relations_v2/` |
| 第二折LOTO | M02的SimpleThreeStageOpAmp1_1_10正在训练；尚无测试分数 | `worktrees/rgcn_memory_safe_loto_v2/` |
| 已完成LOTO | 仅SimpleOpAmp4首折，2,404条固定测试 | `artifacts/experiment2_loto_v1/LB_M04__hold_SimpleOpAmp4/` |
| 参数MLP／GCN基线 | 新入口及真实TRAIN小批验证已完成，等待P1/P2结束后串行训练；无测试成绩。GAT尚未实现 | `worktrees/controlled_evidence_v1/` |

`artifacts/public_evidence_queue_v1/preflight.json`只记录真实训练小批上的两步运行检查。`artifacts/repair_id_no_film_v1/preflight.json`和`repair_id_static_edges_v1/preflight.json`中的`state=complete`指预检完成，其`test_batches_constructed=0`，不能作为消融结果。

## 2. 优先级与控制变量

**P1：完成同神经头ID参考，并建立同目标的ID—LOTO比较。** 新ID保留公开数据400,083／50,861／49,056划分、129组兼容路由、4层128维、FP32、有效批256／微批16、AdamW主干2×10⁻⁴／适配3×10⁻⁴、权重衰减0.01、最多20轮及相同早停和验证选模规则。仅在选择冻结后评价固定测试。分别报告全体ID和目标拓扑的ID测试交集；与LOTO比较时使用相同sample_id、指标语义和Mask，不能拿49,056条ID总体与2,404条LOTO总体直接相减。

现有SimpleOpAmp4首折采用相同神经头形式，但其训练微批、历史实现及测试范围与新ID不完全相同。严格受控结论应在新入口下为M04再建独立LOTO运行，而不覆盖原首折。LOTO训练条数因结构留出而变化，须报告，不声称它与ID训练集大小相同。

**P2：完成已排队的三项结构消融。** 与冷启动ID保持同一划分、监督路由、损失、最高训练预算、种子42和验证选择规则。实际停止轮数按各自同一早停规则确定并公开。

| 消融 | 只改变的开关 | 仍保留的输入与组件 | 参数量 |
| --- | --- | --- | ---: |
| 完整模型 | 无 | 48关系、动态边、FiLM、域嵌入 | 8,146,624 |
| 去FiLM | `use_film=false` | 域嵌入仍保留；不是完全去域 | 8,014,528 |
| 静态边 | `update_edges=false` | 器件参数及初始边编码、边池化、关系仍保留 | 7,883,456 |
| 合并关系 | `collapse_relations=true` | 连接、自环、并行边、器件参数及节点端口统计仍保留 | 1,980,224 |

合并关系不是标准GCN，也不是删除全部端口信息；参数量减少须一并解释。相同种子不自动保证结构不同模型的所有共有层初值逐项相同。补强的配对初始化或多种子实验须采用新版本配方，不更改当前运行。至少对主要结论追加42、43、44三个种子，保留单次结果、均值和样本标准差；不把三次重复当作统计显著性证据。

**P3：增加域嵌入对照与标准GCN／RGCN／GAT基线。** 域对照同时关闭`use_film`与`use_domain_emb`，其余配置不变；监督契约路由与训练统计仍需保留，不能称为完全来源无关。标准基线需独立实现骨干，并统一输入适配器，确保器件参数没有被基线无意丢弃；固定数据、监督头、损失、最高轮数和验证选择规则。发布输入适配、参数量及计算开销。关系合并开关不能冒充该基线，也不能把局部读出的收益当作骨干组件收益。

**P4：扩展LOTO。** 先完成已登记M02折，再覆盖不同结构家族及不同兼容监督支持程度的折。全部按完整连接结构排除等价拓扑，冷启动，不复用见过目标拓扑的ID权重、图嵌入或邻居库。逐折报告有效样本、有效任务数及误差，完成几折就报告几折；跨折汇总采用折等权并给出每折结果。

共同评价：完整样本MRE、中位误差、P95、原定义hit@5%／hit@10%、逐任务R²（含负值及未定义数）、逐指标原单位MAE／RMSE。所有变换与归一化只用本实验训练分区拟合。尾部敏感性作为描述性诊断独立列示，不能用于删除测试记录或修改主结果。

## 3. Windows原生执行入口

先确认队列PID、创建时间与命令行。已有实例时只查看状态，不重复运行下列监督器：

```powershell
Set-Location -LiteralPath 'D:\EDA'
Get-Content -LiteralPath '.\monitoring\public_evidence_queue_v1\status.json' -Encoding UTF8
Get-Content -LiteralPath '.\artifacts\rgcn_memory_safe_id_control_v2\status.json' -Encoding UTF8

# 仅在确认原队列已退出后使用；续训仍由原recipe保护。
.\.venv\Scripts\python.exe -X utf8 -B -u .\worktrees\public_evidence_queue_v1\supervise.py
```

某任务具有选择冻结、对应权重哈希与完整测试预测后，可以单独运行评价；材料未齐时脚本保持待测，不生成假分数：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B .\scripts\evaluate_falcon_reference.py --job rgcn_memory_safe_id_control_v2
.\.venv\Scripts\python.exe -X utf8 -B .\scripts\evaluate_falcon_reference.py --job rgcn_memory_safe_no_film_v2
.\.venv\Scripts\python.exe -X utf8 -B .\scripts\evaluate_falcon_reference.py --job rgcn_memory_safe_static_edges_v2
.\.venv\Scripts\python.exe -X utf8 -B .\scripts\evaluate_falcon_reference.py --job rgcn_memory_safe_collapsed_relations_v2
.\.venv\Scripts\python.exe -X utf8 -B .\scripts\evaluate_falcon_reference.py --job rgcn_memory_safe_loto_v2
```

独立M04与去域候选可按以下方式准备。此代码尚未执行，且不修改正在运行的队列；目标已存在时直接停止，避免覆盖。CPU训练入口保留数据、训练统计与recipe校验，仅在原队列空闲且资源充足时串行运行。现有`qualify.py`专用于CPU／CUDA数值对照，不能将它称为CPU预检；只有准备使用CUDA并满足资源门槛时才执行。

```powershell
function New-ControlledRun {
    param([string]$TemplateName, [string]$RunName, [hashtable]$Patch)
    $edaRoot = (Resolve-Path -LiteralPath 'D:\EDA').Path
    $templatePath = Join-Path $edaRoot ('worktrees\' + $TemplateName)
    $candidatePath = [IO.Path]::GetFullPath((Join-Path $edaRoot ('worktrees\' + $RunName)))
    if (-not $candidatePath.StartsWith($edaRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw '目标不在EDA内' }
    if (Test-Path -LiteralPath $candidatePath) { throw '候选目录已存在，请先核对其配方' }
    Copy-Item -LiteralPath $templatePath -Destination $candidatePath -Recurse
    $configPath = Join-Path $candidatePath 'config.json'
    $candidateConfig = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $candidateConfig.output = 'artifacts/' + $RunName
    foreach ($key in $Patch.Keys) { $candidateConfig.$key = $Patch[$key] }
    [IO.File]::WriteAllText($configPath, ($candidateConfig | ConvertTo-Json -Depth 30), [Text.UTF8Encoding]::new($false))
}

New-ControlledRun -TemplateName 'rgcn_memory_safe_id_control_v2' -RunName 'rgcn_no_domain_candidate_v1' -Patch @{use_domain_emb=$false;use_film=$false}
New-ControlledRun -TemplateName 'rgcn_memory_safe_loto_v2' -RunName 'rgcn_paired_M04_candidate_v1' -Patch @{fold_id='LB_M04__hold_SimpleOpAmp4'}

# 在队列空闲后逐项运行；新的运行首次启动不带 --resume。
.\.venv\Scripts\python.exe -X utf8 -B -u .\worktrees\rgcn_no_domain_candidate_v1\train.py --backend cpu
.\.venv\Scripts\python.exe -X utf8 -B -u .\worktrees\rgcn_paired_M04_candidate_v1\train.py --backend cpu
```

新候选的评价配置应独立登记其任务、划分、选择冻结和权重哈希，不能拿已有任务ID绕过评价凭据检查。标准GCN／GAT目前缺少已验收入口，须先完成上述统一输入与骨干实现，不能提供一个实际不存在的训练命令。

资源采用现有CPU启动门槛与运行保护；CUDA须有至少4 GiB实际可用主机内存并通过对应实现的数值验收。当前主机内存紧张，采用串行而非并行训练。实际训练耗时从各run日志汇总，不以小批预检耗时外推为正式总耗时。

## 4. 约6页正文与补充材料的分工

在目标会议模板、参考文献是否计页及补充材料规则明确前，不将当前单栏页数直接换算为6页。建议正文保留研究问题、数据来源与语义兼容原则、端口图定义、核心传播／边更新／损失、关键训练设置，以及四类协议的实施状态。

正文表格可合并为：数据来源与划分简表；含来源完整MRE、中位误差、P95与hit@5%的主结果表；单折LOTO代表指标；少样本三种子和全量读出对照。保留极端ft案例及98.85%贡献、删除单条后仍很大的描述性MRE，并以表注区分主结果与诊断。

详细工程尺度、版本哈希、所有文件路径、全部14项LOTO物理误差、各预算次要读出、完整超参数与敏感性逐指标归因适合放入补充材料。正文仍必须给出层数、维度、关系数、参数量、优化器、学习率、批大小、选择准则、随机种子和监督预算，并指向实际可获取的补充材料。现有本地路径仅证明内部归档，尚不代表材料已公开。

摘要只保留问题、方法和少量核心发现；讨论集中说明测试重复访问、单折LOTO、读出差异、单位尺度与尾部风险，避免在各段重复同一限制。未完成的结构消融和基线继续列为证据缺口，不在缩稿时改写成已验证贡献。


## 2026-11-12优先级与核心论证补充

用户要求优先取得LOTO证据。已安全保存ID第12轮第593批断点，队列调整为LOTO、冷启动ID、去FiLM、静态边、合并关系；模型原recipe及权重保留。此调度变更与旧策略、断点哈希记入monitoring/public_evidence_queue_v1/events.jsonl。新LOTO首次创建recipe前将CPU线程设为4；真实TRAIN微批16、三来源图、每设置三次计时，2线程平均0.100278秒，4线程0.075255秒，8线程0.078211秒。该结果不是全轮吞吐或精度结果，未降低4 GiB CUDA验收门槛。现有独立supervisor.py具有不适用的旧脚本身份匹配，因此本次使用已核验的公共串行队列，不使用该旧监督入口。

论文必须围绕两个可检验的问题组织后续对照，不预设复杂模型一定优于简单模型。

**问题一：多源融合是否有价值？** 对每个目标来源分别比较仅该来源监督与三来源联合训练，固定目标测试ID、监督语义、原Mask、头结构、初始化策略和验证准则。先选LOADBench作为主要目标，再以FALCON与AICircuit检查结果是否一致。目标来源训练记录及验证标签预算保持相同，加入的其他来源只通过共享编码器起作用，不能按同名指标强行共用标签。同步给出固定优化步数的计算预算对照和训练标签数量；必要时增加总标签预算匹配的补充对照，避免把更多标签、更多更新或不同头解释为融合收益。报告逐来源、逐功能和逐拓扑的完整MRE、中位误差、P95、hit@5%、R²和物理单位误差，包含负迁移。

该来源消融还没有完成的训练入口或冻结结果，需要在新的独立运行中生成来源受限的train/validation/test引用和训练统计，再校验唯一ID、分区与语义覆盖；不能直接使用含其他来源训练统计的归一化文件。测试范围固定为目标来源，避免对未训练输出头评价。现有冷启动多源ID可作参照，但其原验证准则须明确；严格来源对照应采用相同的目标来源验证准则并分别冻结选择。

**问题二：端口感知RGCN是否值得其复杂度？** 先完成已可执行的关系合并、静态边和去FiLM对照；另实现不做消息传递的参数模型以及标准GCN基线，必要时加入标准RGCN/GAT。所有方法须获得同一器件参数、已知测试条件与目标语义，使用相同划分、监督头与最高更新预算，并报告参数量和实际时间。变长图的参数基线需预先规定输入组织，不能让MLP因缺少关键器件参数而处于不公平条件。合并关系的现有模型不是标准GCN，且参数量下降，因此也需报告容量差异。

主指标优先观察未见拓扑上的同一目标测试交集，而非只比较局部插值结果。若简单模型在相同任务上相当或更好，应据实收窄端口RGCN的适用范围；若完整模型更好，也须用受控消融区分端口关系、动态边和域调制的贡献。多源覆盖更多电路是数据接口的事实性价值，预测收益则必须由上述对照证明，两者不混写。


## 2026-11-13已落实的P1—P4执行入口

本轮以P1—P4为准：正常M02保持运行，ID使用已有断点；原队列随后执行三个从零结构消融。已经实现并验证的来源与简单模型对照位于 `worktrees/controlled_evidence_v1/`，由独立 `followup.py` 在原五项完成、原监督器退出且资源满足后串行执行。此处之前标为“尚未实现”的历史描述已由本段替代，不能当作目前状态。

新对照固定LOADBench训练160,046、验证20,653、测试19,301条；多源另加FALCON 160,051及AICircuit 79,986条。每轮同为400,083次呈现／1,563次更新；单源循环TRAIN池，独特目标标签量相同而目标呈现次数不同，实际选中轮与更新成本分别记录。新的RGCN／MLP／GCN共用已知条件8维适配分支，不与原P2版本混写。参数量、输入限制、检查记录和自动执行方式详见该目录README。

统一结果文件为[受控实验结果](controlled_experiment_results.md)，由冻结权重、recipe和测试预测核验后自动更新；所有未完成项仍标为待测。测试只在验证选择冻结后推理一次，后续MRE与共同测试交集统计是对冻结预测的分析，不用于再选模。不会在本轮增加Few-shot重复、复杂模型或大规模LOTO扩展。
