# P1—P4受控实验续行实现

只在EDA内运行。原P1/P2队列、源码、训练配方、ID断点及M02进程保持不变。本目录是独立实现和独立输出，不会接管或停止原监督器。

## 实际执行次序

1. 原队列继续M02 LOTO → 从ID第12轮断点恢复 → No-FiLM → Static-Edge → Merged-Relation。
2. `followup.py`只观察，核验已冻结测试并补齐统一指标。原五项全部完成、原监督器退出且不存在其他训练器后，才运行本目录四项：LOADBench单源RGCN → 三源RGCN → 参数MLP → GCN。
3. 每个任务支持原recipe断点恢复。无数字／指纹异常时，内存不足会保存并等待；其他错误进入`requires_review`，不自动改标签或训练规则。

本次不增加Few-shot重复和大规模LOTO折数。已有单折M04单独列示；ID—LOTO只比较完全相同sample_id的测试交集，核对观察标签、Mask、语义、单位及尺度后再输出。

## P3/P4公平比较

LOADBench原训练160,046、验证20,653、测试19,301条全部固定。多源额外使用FALCON训练160,051及AICircuit训练79,986条。监督头按原129组语义独立路由，不强行共享跨来源标签。

两条来源路线的所有可训练参数初值相同（已校验哈希），每轮400,083次呈现、1,563次优化，最多20轮；单源循环原训练池，不复制新数据。独特目标标签量相同，但单源目标呈现次数更多；实际停止轮数和所选epoch也可能不同。因此归档实际更新数、来源呈现次数及耗时，并保留这些限制。尚未添加总独特标签量相同的第二组实验。

所有P3/P4模型共用8维条件向量：已知温度、供电、负载、偏流的固定尺度symlog及known标记，不拟合测试统计、不补造未知值。RGCN增加相同的8→128适配器，参数8,147,776；这是新版本，不与P2的8,146,624参数模型混算差值。

参数MLP限定于LOADBench已知拓扑，按原缓存固定器件顺序，训练池最大器件数填充，保留90维器件值／known／zero和present标记，无邻接、端口边或消息传递。节点34维、边90维的GCN使用相同图；器件参数经初始节点注入及静态边池化保留。传播严格为对原多重邻接加I后的对称度归一化图卷积，没有关系权重或动态边更新；残差、LayerNorm、FiLM及输出框架保留。MLP为1,834,686、GCN为1,585,984参数。这些参数量含保留的全部129组头；单源实际只监督LOADBench头，其他来源头不评价。

## 已运行验证

见`artifacts/controlled_evidence_v1/preflight.json`：真实TRAIN中8条LOADBench设计及20条三源设计，覆盖公共48种关系；各模型两次有限的前向／反向／AdamW更新，评估与生产MetricAccumulator一致。GCN与稠密矩阵公式一致、关系编号置换不影响输出；MLP图依赖检查和优化器／RNG恢复检查通过。未构建测试批、未用验证标签、诊断权重未保存为正式权重。

首轮检查每个头仅选一条训练设计，导致均值初始化的Loss接近零，作为不足的学习检查已被替换；原记录保留于`preflight_superseded_trivial_targets.json`。新检查每拓扑取4种真实参数设计，确认非平凡Loss和实际参数更新。初次导入评估模块时的版本路径冲突也已修复，不影响旧训练。

新任务的完整训练统计在正式启动时从自身TRAIN池重新拟合，不复用含被排除来源统计的归一化；诊断统计不作为正式统计。

## 查看和恢复

状态：`monitoring/controlled_evidence_v1/status.json`。统一表：`reports/controlled_experiment_results.md`。原训练仍看`monitoring/public_evidence_queue_v1/status.json`。

确认没有本续行实例后才能运行：

```powershell
Set-Location -LiteralPath 'D:\EDA'
.\.venv\Scripts\python.exe -X utf8 -B -u .\worktrees\controlled_evidence_v1\followup.py
```

`--once`只核查与更新报告，不启动训练。CPU启动门槛1.4 GiB，连续资源退出升至1.8／2.2 GiB；CUDA未启用，4 GiB门槛未放宽。所有训练串行且低优先级，不承诺固定完成时间。预计本目录新增权重、预测与日志约1—2 GiB，不含原队列；没有新增下载或仿真。

安全停止本续行队列：创建`monitoring/controlled_evidence_v1/STOP`。如本队列已启动新训练，将在批边界保存断点；不会向原P1/P2任务发送停止指令。该进程依赖电脑和用户会话持续运行，不是开机服务。

`assessment.py`只在选择冻结、权重及recipe哈希核对通过后读取测试预测，不作模型推理或调参。保留任务宏平均hit@5/10%、含负值的R²、未定义项数、物理单位MAE/RMSE，以及使用既有FALCON参考实现的完整样本MRE、中位误差、P95。汇总结果及逐任务/逐样本Parquet另存于`artifacts/controlled_evidence_v1/evaluation/`，不覆盖原始实验产物。
