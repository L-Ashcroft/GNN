# 多源模拟电路性能预测：端口感知边中心 RGCN

本仓库归档多源模拟电路前版图性能预测的代码、固定数据划分及真实实验成果。公开融合集与自建集保持独立。代码基于 Windows 原生 Python 3.12，提供 ID、留一拓扑（LOTO）、结构消融、单源／多源对照及简单模型基线的统一入口。

**发布范围为 2026-11-13 的研究快照。存在代码不代表相应实验已经完成。** 分布内局部读出、独立神经头和冻结特征混合读出的结果须分别解释，不能直接归因于 RGCN 的某一结构组件。

## 已有成果与待完成研究

| 任务 | 真实进展 |
| --- | --- |
| 公开 ID | 已冻结 49,056 条测试预测；任务宏平均 hit@5%=54.32%、hit@10%=66.38%，R²中位数 0.850。包含已知拓扑的局部校准读出 |
| LOTO | SimpleOpAmp4 首折已完成，2,404 条测试记录；hit@5%=31.28%、hit@10%=42.81%，R²中位数 0.915。M02 第二折在原工作站训练，未发布其未冻结权重为正式成果 |
| 跨数据集零样本 | 已审查监督兼容性；尚无经过核验的共同监督目标，不报告预测准确率 |
| 自建集少样本 | 已有端到端预训练／随机初始化、冻结特征与全量监督对照。k=100 的三种子端到端 hit@5% 均值约25.11%／24.45%；总目标域监督预算为3,000条，包括2,000条验证标签 |
| 自建集全量混合读出 | hit@5%=67.69%。读出收益与公开预训练收益分别分析 |
| 受控证据 | 同神经头 ID、No-FiLM、Static-Edge、Merged-Relation、单／多源、MLP、GCN 已配置，完成状态见带时间戳的[成果表](reports/controlled_experiment_results.md)，不作为已完成消融结论 |

现有冻结预测、逐指标原单位误差、完整 MRE、负 R² 和极端误差均保留。MRE 对近零标签、对数单位和固定分母尺度敏感；任务宏平均命中率与样本等权 MRE 使用不同口径。模型结构独立贡献、多源融合收益仍需受控实验验证。

## 数据与下载

| 组成 | 内容 | 获取方式 |
| --- | --- | --- |
| `public-data` | 500,000 条：FALCON 200,000、AICircuit 100,000、LOADBench 200,000；121个来源拓扑条目，118类连接结构；图、参数、标签、Mask及既有划分 | Release分块附件 |
| `personal-data` | AnalogGym派生独立自建集，10种拓扑各2,000条；图、条件、参数、标签及16,000／2,000／2,000划分 | Release分块附件 |
| `model-inputs` | 当前训练直接依赖的原始图张量与既有冻结编码器特征；不计作新增样本 | Release分块附件 |
| `experiment-results` | 正式选定权重、必要历史父权重／读出索引、固定测试预测、指标、选模记录、已暂停ID恢复状态及历史源码附件 | Release分块附件 |
| `upstream-evidence`（可选） | 实际下载的LOADBench子集、数据审计与AC诊断、个人生成账本 | Release分块附件 |

大型二进制不进入 Git 历史。每个压缩包按 **64 MiB** 分块；[清单](distribution/manifest.json)记录各块、完整压缩包及每个解压文件的 SHA256。恢复后保持原路径与原字节，不重新划分或改写标签。小型图、定义及评价 JSON 可直接在仓库浏览。

```powershell
git clone --recurse-submodules https://github.com/L-Ashcroft/GNN.git
Set-Location -LiteralPath GNN

# 下载和校验不需要安装第三方Python包
python scripts/download_release.py --list
# 默认下载两套最终数据、训练缓存和成果（不含可选来源证据）
python scripts/download_release.py
# 需要核查实际下载来源、生成账本和AC诊断时
python scripts/download_release.py --components upstream-evidence
```

下载器只从本仓库的固定 Release 获取附件，校验分块、合并包与成员文件；已存在且不同的文件会拒绝覆盖。可以重复运行继续下载。离线时先取得同一组件的全部 `.partNNN` 文件，再用 `--from-dir 'D:\release-parts'` 恢复。

**没有随本次快照发布**：约8.38 GiB的个人逐次仿真原始ZIP、探索性仿真、正在写入的M02权重、机器进程锁／旧日志、虚拟环境、维护备份、论文草稿和讲义。正式样本的原始路径与哈希仍保留；完整原始波形审计需要另外提供这些ZIP，不能将本发布包声称为已附齐所有仿真波形。原有 `verify_workspace.py --stage data` 会要求这些原始包；发布内容验证使用下述 `check_release.py`。

## 环境和验证

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-gnn.txt -r requirements-evaluation.txt
.\.venv\Scripts\python.exe -X utf8 -B scripts/check_release.py
# 完整读取已发布的50万／2万最终样本，检查字段、Mask、计数及图接口
.\.venv\Scripts\python.exe -X utf8 -B scripts/check_release.py --data
# 仅重算已冻结预测；不解封待完成任务
.\.venv\Scripts\python.exe -X utf8 -B scripts/verify_workspace.py --stage metrics
```

公开图输入为34维节点／90维边／48关系，自建图为35／90／52。MOS端口角色、并行边、自环、电源／地标记均保留。16个性能槽位是存储接口，监督按物理定义与测试条件路由；Mask仅表示缺失。归一化仅在训练集拟合。

原训练配方校验运行库文件指纹。`experiment-results`附原CUDA版本凭据及只含 `runtime_receipts` 的小型SQLite，**不包含CUDA运行环境，也不是完整维护账本**。CPU环境须满足原Torch版本和指纹。不得改写recipe绕过版本保护或跨版本直接续训；换环境／方法必须使用独立输出与新配方。可选GPU安装见 `scripts/setup_cuda.ps1`，使用前仍需内存和数值验收。

## 运行入口

```powershell
.\.venv\Scripts\python.exe -X utf8 -B scripts/research.py list
# 先解析真实配置与命令，不开始训练
.\.venv\Scripts\python.exe -X utf8 -B scripts/research.py train loto --holdout SimpleThreeStageOpAmp_SimpleOpAmp1_1_10 --prepare-only
.\.venv\Scripts\python.exe -X utf8 -B scripts/research.py train no-film --prepare-only
.\.venv\Scripts\python.exe -X utf8 -B scripts/research.py train static-edge --prepare-only
.\.venv\Scripts\python.exe -X utf8 -B scripts/research.py train merged-relation --prepare-only
.\.venv\Scripts\python.exe -X utf8 -B scripts/research.py train single --prepare-only
.\.venv\Scripts\python.exe -X utf8 -B scripts/research.py train multi --prepare-only
.\.venv\Scripts\python.exe -X utf8 -B scripts/research.py train mlp --prepare-only
.\.venv\Scripts\python.exe -X utf8 -B scripts/research.py train gcn --prepare-only
# 原神经头ID配方有暂停断点，用同配方恢复
.\.venv\Scripts\python.exe -X utf8 -B scripts/research.py train id --resume --prepare-only
```

确认配置、资源和输出目录后去掉 `--prepare-only` 即训练。当前RGCN实现为 `versions/rgcn_calibrated_head_v1/src/`，五项P1/P2适配层在 `worktrees/rgcn_memory_safe_*/`；P3/P4统一实现为 `worktrees/controlled_evidence_v1/`。ID、LOTO和消融独立初始化，监督契约及固定划分不变；Merged-Relation不等同标准GCN。

训练按队列串行执行：`scripts/research.py queue`，后续对照用 `scripts/research.py queue --which followup`，状态用 `scripts/research.py status`。这些命令会真实启动进程；仓库中的 `reproducibility/current_training_snapshot.json`只是原工作站的带时间戳快照，不能当作新机器运行状态。

历史Few-shot／读出复现所需的归档源码已恢复到本仓库原路径；旧配置仍在 `artifacts/inactive_json_records_20261113.zip` 附件内，按对应recipe按需提取，禁止覆盖当前运行配置。历史绝对溯源路径不作为可移植新路径的承诺；接收方需先运行清单验证和解析检查。

历史维护ZIP的发布副本移除了论文编辑和讲义审阅记录，其公开SHA256以本发布清单为准；科研配置与已有实验评价记录保留。该筛选不涉及数据标签、样本划分或冻结预测。

## 目录与来源

`src/`数据与评价；`versions/`冻结模型实现；`worktrees/`各实验适配层；`config/`统一配置；`scripts/`入口与验证；`artifacts/`成果及模型；`data/`两套数据和缓存；`distribution/`分块和文件清单；`reproducibility/`来源提交、源码指纹和运行库凭据。

上游仓库作为固定提交的Git子模块引用，不复制其Git历史。第三方代码／数据许可和署名见 [DATA_SOURCES.md](DATA_SOURCES.md) 与 [LICENSES](LICENSES)。本仓库公开可见不等同于为全部原创代码另行授予MIT许可；上游材料沿用其原许可。FALCON与AICircuit保留MIT声明，AnalogGym保留BSD-3-Clause声明，LOADBench保留CC BY 4.0来源记录。
