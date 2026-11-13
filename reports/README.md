# 当前报告入口

此目录仅保留5份当前文档和3个代码直接引用的JSON，共8个文件，不再保留历史报告子目录。

| 当前文档 | 内容 |
| --- | --- |
| [实验进度与成果](public_experiment_completion_status.md) | 四项实验已有成绩、当前进度和证据边界 |
| [受控实验结果](controlled_experiment_results.md) | P1—P4统一结果表，后台根据冻结产物更新 |
| [受控实验计划](gnn_controlled_experiment_plan_v6.md) | 控制变量、任务顺序和比较限制 |
| [FALCON参考评价](falcon_reference_results.md) | 完整MRE、中位误差和P95汇总 |
| 本README | 文件入口与归档说明 |

3个保留JSON：`source_inventory.json`用于来源转换，`analoggym_replay_audit.json`用于回放审计接口，`analoggym_audit.json`用于来源审计接口。为避免破坏现有代码路径，保持原位。

历史审计、旧模型说明、性能分析及清理记录等70个文件，已经逐文件SHA256核验后收进一个[历史证据归档](../artifacts/report_archive_20261113.zip)，并从本目录移除。归档约4.18 MiB，保留原始`reports/...`路径和清单；需要复跑旧审计、历史性能验收或核查旧凭据时，将包内的`reports/`恢复到EDA原位置。当前训练和后续P1—P4不需要恢复这些历史报告。

实时训练看[原队列](../monitoring/public_evidence_queue_v1/status.json)与[续行队列](../monitoring/controlled_evidence_v1/status.json)。论文仍只编辑[现有副本](../background/模拟电路论文新_编辑副本.docx)。
