# 新的三来源数据集

本目录独立保存 FALCON 20 万、AICircuit 10 万、LOADBench 20 万的目标抽样结果。实际构建状态以 `manifest.json` 为准，质量检查以 `quality_checks.json` 为准。旧双源转换／抽样副本及其划分已清理，官方来源表保留在third_party/，本目录为唯一公开训练数据集。

- `sampled/*.parquet`：逐样本参数、性能、Mask、语义 ID、原始标签和来源。
- `graphs/*.json`：固定拓扑图。网为节点，器件按端口对展开为边；MOS 的四个端口全部保留。
- `graphs/*.json` 中的 `node_annotations` / `node_features`：电源、地、输入、输出、偏置角色及证据，连接器件数和端口类型计数。`has_verified_evidence` 区分网表/测试平台依据与纯命名约定；未分配角色不等于已证明它是内部节点。
- `catalog.json`：图、参数表达式、默认值证据、工艺和测试条件。
- `performance_mapping.json`：最多 16 维指标映射；同槽位不代表相同物理标签。
- `index.parquet`：样本索引。
- `sampling_statistics.csv` / `.json`：各拓扑、参数分层的候选及抽样数量。
- `candidates/`：LOADBench 本次局部下载经过清洗的候选池，不是完整发布数据。
- `splits/`：ID、五折拓扑留出及未获可比标签批准的跨来源候选划分。

从项目根目录运行：

```python
from src.dataset_fusion import FusionDataset

dataset = FusionDataset()
sample = next(dataset.samples(source="LOADBench"))
print(sample["sample_id"])
print(sample["edge_index"], sample["edge_features"])
print(sample["performance"], sample["mask"])
print(sample["metrics_by_semantic_id"])
print(sample["node_feature_names"], sample["node_annotations"])
print(sample["edge_types"], sample["edge_features_transformed"])

# 流式读取原始字段，省去逐样本器件特征计算
rows = dataset.samples(materialize=False)

# 直接加载训练划分
train = dataset.samples(split="data/fusion_v2/splits/id.parquet", role="train")

# 推荐的监督接口：每批只有一个语义及测试条件分组
batches = dataset.supervision_batches(
    batch_size=128, split="data/fusion_v2/splits/id.parquet", role="train")
batch = next(batches)
print(batch["supervision_group"], batch["target_semantic_ids"])
```

`samples()` 只统一存储访问，不能把它返回的不同样本按 16 个槽位直接堆叠计算统一损失。`supervision_batches()` 会按来源、拓扑、指标语义及测试条件保守分组；不同组使用各自的输出头、标签尺度和损失，再按明确权重汇总组损失。Mask 仅排除组内缺失值，不能解决物理定义或测试条件差异。当前没有获批准的跨来源共享性能标签。

GNN 骨干可以共享，但这不等于跨来源同名指标已经可以共同监督。归一化参数也必须按目标语义/监督组仅用训练集拟合。每组损失先按该组已知标签数量归约，再明确设置组权重；不要让量纲或样本数决定损失权重。跨来源可比标签评估仍未就绪。

当前分组有意保守。若给每个拓扑单独设置输出头，不能直接声称完成未知拓扑的性能预测；已有 LOTO 文件只保证数据留出，不代表预测目标已完成跨拓扑对齐。开展此实验前，还需根据增益类型、开闭环、测试频率、差分定义、偏置、负载、工艺及提取方法等证据审核可共享的目标，再建立相应共享任务头。单位都为 dB 不足以批准共享。

全部 142 个现有组的拓扑覆盖及已验证相同发布协议的 LOADBench 合并集合，见 [历史监督组审计归档](../../artifacts/report_archive_20261113.zip) 和 `supervision_group_details.json`。这些合并集合尚未改变默认路由；现有“单拓扑组”来自保守规则，不表示没有可共享的任务。

Mask 只表示源表未提供某指标；失败测量对应的样本已排除。模型归一化尚未拟合，训练时必须只用训练集拟合。LOADBench 抽样仅代表下载的最终表和早期历史前缀，不代表其完整百万次优化历史。

完整结果、覆盖缺口、下载方式与许可见 [历史构建审计归档](../../artifacts/report_archive_20261113.zip)。

图特征版本为 `lecture12_features_v1`。JSON 中 `num_edges` / `edge_index` 仍指原始物理端口对边；`message_passing` 是双向展开，包含有向关系类型、反向索引、可选结构归一化权重。加载 API 的 `edge_index`、`edge_features`、`edge_types` 已对齐双向消息边，条数为 `num_message_edges`；要访问原始边使用 `physical_edge_index`。不要再对 API 返回的边重复执行无向展开，也不要让框架重复添加已有端口自环。

原始 `device_features` / `edge_features` 保留 SI 数值，末尾补充 nf、W/L、层次深度和器件类型 one-hot。`edge_features_transformed` 仅做规定的 log10/log2 变换，未标准化；`edge_transformed_known` 表示适用且已知，`edge_feature_zero` 区分零元件值与非零值的对数零点。缺少原始 nf 的器件保持未知。供电标记不会把 `GND` 与 `0` 合并。具体证据规则和剩余边界见 [历史图特征审计归档](../../artifacts/report_archive_20261113.zip)。
