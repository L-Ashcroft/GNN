# 基于GNN的模拟电路性能预测

本项目面向模拟电路前版图性能预测，以电路连接关系、器件参数及已知测试条件为输入，预测增益、带宽、功耗等性能指标，探索减少设计过程中重复仿真的可能性。

项目整合FALCON、AICircuit和LOADBench的数据，采用端口感知的边中心关系图卷积网络（RGCN）。研究以分布内预测（ID）和留一拓扑泛化（LOTO）为核心，通过结构消融、单源／多源训练及简单模型对照，评估模型结构、多源监督和拓扑变化对预测性能的影响。

## 目录结构

| 目录 | 主要内容 |
| --- | --- |
| `data/` | 电路数据，包括图结构、器件参数、性能标签和训练／验证／测试划分 |
| `artifacts/` | 实验产物，包括模型权重、预测值、真实标签、评价指标和训练记录，是查阅实验成果的主要目录 |
| `config/` | 数据与实验配置，包括性能指标映射、监督分组及仿真条件 |
| `src/` | 数据读取、网表转图、性能评价等基础代码 |
| `versions/` | 与已有模型配套的固定版本代码。当前RGCN主要在`versions/rgcn_calibrated_head_v1/src/` |
| `worktrees/` | 各项实验的训练程序和配置，包括ID、LOTO、消融、单／多源以及MLP、GCN |
| `scripts/` | 数据下载、完整性检查、实验启动与状态查询脚本 |
| `reports/` | 实验进度、结果分析及适用范围说明 |
| `distribution/` | 发布文件清单与校验值，用于检查下载完整性 |
| `reproducibility/` | 数据来源版本、代码指纹和环境记录，用于核对复现条件 |
| `third_party/` | 对官方项目的引用，需要时可通过Git子模块下载 |
| `LICENSES/` | 第三方许可说明；数据来源和作者信息见[DATA_SOURCES.md](DATA_SOURCES.md) |

`versions/`和`worktrees/`保留了不同实验实际使用的实现版本。复现已有结果时，应使用对应版本的代码与配置。

## 数据集组成

**公开融合数据集包含500,000条样本**：FALCON 200,000条、AICircuit 100,000条、LOADBench 200,000条。按来源标识计有121个拓扑条目，经连接结构核验后对应118类不同结构。

**自建数据集包含20,000条样本**，基于AnalogGym的10种拓扑，每种2,000条，通过调整器件参数并执行仿真获得。该数据集独立保存，用于少样本适配及补充迁移研究。

每条样本包含电路图、器件参数、测试条件和性能标签，并以Mask标记缺失指标。同名指标可能具有不同的物理定义或提取方式，因此按经过核验的监督分组处理。

数据恢复后的主要目录如下：

- `data/fusion_v2/`：公开50万条数据。
- `data/generated/analoggym_personal_20k_v1/`：自建2万条数据。
- `data/model_ready/`：供现有模型直接读取的张量和特征缓存。这些缓存由已有样本派生，不计作新增样本。

## 数据下载与恢复

Git仓库保留代码、小型图文件、配置和结果索引。完整数据、模型及预测文件通过[Release附件](https://github.com/L-Ashcroft/GNN/releases/tag/research-snapshot-20261113)提供，总计约3.62 GiB，分为61个不超过64 MiB的分块，另附文件清单。

下载脚本自动完成分块合并、校验和目录恢复。以下命令使用Python 3.12：

```powershell
git clone https://github.com/L-Ashcroft/GNN.git
Set-Location GNN

# 列出可下载的组件
python scripts/download_release.py --list

# 仅下载两套最终数据集
python scripts/download_release.py --components public-data personal-data

# 下载数据、配套缓存和实验成果
python scripts/download_release.py
```

各下载组件的用途如下：

| 组件 | 用途 |
| --- | --- |
| `public-data` | 使用公开50万条数据 |
| `personal-data` | 使用自建2万条数据 |
| `model-inputs` | 为现有训练代码和模型提供配套缓存 |
| `experiment-results` | 查看完整预测、评价结果、模型权重和已有实验记录 |
| `upstream-evidence` | 提供原始下载来源、生成账本和AC诊断等补充证据，可按需下载 |

仅查阅已有实验成果时，可下载：

```powershell
python scripts/download_release.py --components experiment-results
```

本次发布不包含约8.38 GiB的逐次仿真原始ZIP、论文与讲义、虚拟环境、旧日志及维护备份。最终样本已提供；逐条核查原始仿真波形仍需另附原始ZIP。

## 实验成果与文件索引

| 内容 | 文件或目录 |
| --- | --- |
| 公开ID总体结果 | `artifacts/loadbench_remediation_v1/combined_fixed_test_summary.json` |
| 已完成的SimpleOpAmp4留一拓扑实验 | `artifacts/experiment2_loto_v1/LB_M04__hold_SimpleOpAmp4/` |
| 少样本实验及不同随机种子 | `artifacts/experiment4_personal_transfer_v1/`、`artifacts/research_repair_personal_repeats_v1/` |
| 自建集全量混合读出结果 | `artifacts/research_repair_v1/personal_test/full_joint/metrics.json` |
| MRE和逐指标误差 | `artifacts/falcon_reference_eval_v1/` |
| 受控实验完成情况 | [受控实验结果说明](reports/controlled_experiment_results.md) |

主要文件格式：

- `.json`：保存配置、评价指标或说明，可直接阅读。
- `.parquet`：保存样本和逐条预测等表格数据，相比CSV更适合大规模数据的存储与读取。
- `.pt`：PyTorch模型或训练断点。
- `.npy`、`.npz`：数值数组，常用于特征、缓存和读出参数。

已有ID主结果的hit@5%为54.32%，其中包含局部校准读出；SimpleOpAmp4首折LOTO为31.28%。这两项的读出和训练条件不同，不能直接相减来衡量拓扑泛化损失。新神经头ID、结构消融、来源与简单基线对照仍需按各自完成状态报告。

仓库与Release反映发布时的实验状态，不随原工作站训练自动更新。已有少样本适配（Few-shot）结果作为补充保留；跨数据集零样本评价目前仅完成兼容性审查，尚无可报告的预测分数。

## 环境配置与运行

运行环境为Windows原生Python 3.12，依赖安装与基本检查命令如下：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-gnn.txt -r requirements-evaluation.txt

# 下载默认组件后检查文件完整性
.\.venv\Scripts\python.exe scripts/check_release.py

# 查看已有实验入口
.\.venv\Scripts\python.exe scripts/research.py list

# 查看本机训练状态
.\.venv\Scripts\python.exe scripts/research.py status
```

完整文件检查需要默认组合中的数据、缓存和实验成果。仅下载部分组件时，可使用下载脚本的组件校验。

实验通过`scripts/research.py`统一启动。已有结果对应固定的代码、数据划分和环境，断点恢复应遵循原配置。完整训练、恢复及环境要求见[发布版本使用说明](https://github.com/L-Ashcroft/GNN/blob/research-snapshot-20261113/README.md)中。

第三方材料分别沿用原许可，具体见[数据来源](DATA_SOURCES.md)和[许可文件](LICENSES)。
