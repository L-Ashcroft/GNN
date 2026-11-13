# AnalogGym 个人独立数据集

目标：10 个已验证拓扑，各 2,000 条不同有效设计，共 20,000 条。使用最多 8 个独立 ngspice 进程。**目标不代表已完成，实际进度读取 `status.json`。**

- `status.json`：账本实时进度、状态和 worker PID。
- `dataset.json`：已导出 Parquet 的快照，可能略滞后于仿真进度。
- `samples/`：已接受样本；参数、性能、缺失 Mask、语义和原始来源。
- `topologies/`：DUT 图、测试条件、指标定义与采样参数。
- `raw/`：每次仿真的网表、参数、日志和波形归档；失败点也保留。
- `ledger.sqlite`：持久接受/失败记录和唯一设计约束。
- `parameter_anchor_audit.json`：40 组真实成功参数中心与出处；历史标签不作为新样本标签。
- `pilot_quality_checks.json`：首批每拓扑 3 条有效样本的原始波形重提取检查。
- `production_stdout.log` / `production_stderr.log`：后台运行日志。
- `supervisor_status.json`：内存保护监督状态；仅对自身触发的低内存暂停，在可用内存持续 30 秒高于 1.5GiB 后自动恢复。用户 STOP、磁盘预算和错误不会自动清除。
- 全量结束后生成 `personal_index.parquet` 和 `splits/`，包括固定测试集及嵌套少样本预算。

数据独立保存，不合并到 `data/fusion_v2/`。它是新仿真的目标域，不直接复用 LOADBench/FALCON/AICircuit 的指标槽位含义。跨来源零样本比较仍需要公共标签和测试条件审核；微调需明确新目标头、域和实际测试条件。

**2026-10-30 最终验收：已完成20,000条有效样本，10个拓扑各2,000条，最终全量重扫无AC连续性复核标记，必需字段无意外缺失。** 详见 [最终检查与实验可用性评估](../../../reports/analoggym_final_20k_assessment.md)。这表示仿真数据交付通过，不表示当前RGCN输入和目标域适配已经完成。

实验建议使用新增的 `splits/anchor_safe_v1/id.parquet` 及同目录两个少样本预算文件，勿混用原始预算。此版本将三个自适应采样中心全部放入train，保留原始测试集全部ID不变，各拓扑仍为1600/200/200；原始划分未改。差异和检查结果见 `splits/anchor_safe_v1/audit.json`。独立测试集不得用于拟合；完整2,000条validation若用于调参，是k条适配标签之外的额外预算。

LDO 外部偏置电压、电流、负载等已知条件保存在参数和条件记录中，模型适配必须带入这些输入，不能只取 DUT 器件边特征，也不能用仿真节点电压/电流补输入。少样本预算中的验证集标签属于额外目标域预算，需单独披露。

查看进度、停止或恢复的方法见 [生成报告](../../../reports/analoggym_personal_20k.md)。

需要持续监督资源恢复时，从项目根目录运行 `python -B -u scripts/supervise_analoggym_personal.py`；它调用相同的 8 并发生成器和完成后的划分流程。不要与已有生成器/监督进程同时启动。

2026-10-28 采样修订：folded LDO 从候选 83 起围绕三个可工作参数中心进行约 ±5% 尺寸、±2% 偏置扰动，其余九个拓扑保持原采样。详见 `sampling_policy_v2.json` 和 [优化报告](../../../reports/analoggym_folded_optimization.md)。原 293 条有效记录保留；新增两列采样版本/哈希，配方哈希和既有标签不变。对照试验在独立目录，不计入本生产集合。当前后台日志为 `supervisor_v2_stdout.log` / `supervisor_v2_stderr.log`，进度以 `status.json` 为准。

## 每五分钟自动巡检

独立后台进程 `scripts/monitor_analoggym_personal.py` 每 300 秒检查一次，不额外启动仿真，不增加 8 并发上限。最近结果看 `monitor_status.json`：`health=ok` 表示本轮未发现异常，`attention` 表示需查看 alerts，`critical` 表示已发现完整性问题，`check_failed` 表示巡检自身失败，不能视为数据正常。检查时间 `checked_at_UTC` 应持续更新；超过约十分钟不更新时，检查巡检进程和 `monitor_stderr.log`。

检查内容：每拓扑最近 20 个候选的参数/标签有限性、完整维度、同采样版本的有效率和连续失败；每拓扑最新一条有效归档的 CRC/SHA 及原始波形重算；冻结代码及采样策略哈希；进程存活、进度更新时间、剩余磁盘与内存。新策略不足 20 条时不套用 20 条窗口的低通过率判断，避免混入旧策略失败。连续失败 ≥10 次、或至少 20 条窗口有效率低于 50% 时告警。巡检是滚动抽查，不代替最终全量检查。

`monitor_events.jsonl` 每轮记录摘要、新增告警及已解除告警。低通过率只告警；标签或归档等完整性错误会新建 STOP 请求生成器排空保存，不清除已有 STOP、不删除或修改样本，也不自动放宽验收规则。原有内存保护与恢复逻辑继续独立运行。

已在当前 Windows 会话启动。重启电脑后需在项目根目录手动恢复（文件锁防止重复巡检进程）：

```powershell
.\.venv\Scripts\python.exe -B -u scripts/monitor_analoggym_personal.py
```

巡检把结果写入本地文件，不会主动向聊天窗口推送消息。仿真完成后巡检自动退出；用户暂停时保留巡检以记录暂停状态。

## Windows 状态写入修复（2026-10-28）

任务曾在 1,354 条有效样本处因 `status.json` 原子替换报 WinError 5 而退出。新增 `src/windows_campaign_io.py`：唯一临时文件名，写入后关闭并刷新，只对 Win32 5/32/33 错误最多重试 15 秒；持续失败仍报错并保留临时文件。运行器、归档、监督和巡检采用此逻辑，七个冻结模块和第三方代码未修改。采样策略内容和性能定义不变。

修复核对了已有 1,589 条尝试记录，内容全部一致；另外从 7 个已完成归档恢复台账，其中 6 条有效，恢复后共 1,360 条有效样本。81 项测试通过，包括真实 Windows 读锁重试测试；生产样本质量检查通过。修复前快照与核验记录在 `revisions/windows_io_retry_v1/`。

本次恢复后的日志为 `supervisor_io_stdout.log` / `supervisor_io_stderr.log` 和 `monitor_io_stdout.log` / `monitor_io_stderr.log`；启动信息在 `launch_io_repair.json`。`last_error.json` 中旧错误已标记 resolved，判断当前运行状态应查看持续更新的 `status.json` 与 `monitor_status.json`。

## AC 波形质量升级

采用 `quality_policy_v1.json` 的 `ac_continuity_v1` 策略：对放大器 AC 波形检查大量相邻增益跳变及异常多重单位增益交越。触发后整条设计隔离，不用 Mask 遮盖错误；归档保留，继续仿真其他参数点补足每拓扑 2,000 条。该规则不是高性能器件产品规格筛选，真实但性能差的可测量设计仍保留。

已回查当时全部 2,739 条已接受记录，将 Song 候选 101、252 两条移出 accepted，剩余 2,737 条，其他台账记录未变。原始归档未改写，SQLite 的 `quality_reviews` 表保留原始结果及复核决定，维护前指纹在 `revisions/ac_quality_v1/`。加载请使用 `dataset.json` 列出的 shards 或 `iter_campaign_samples`，不要直接递归拼接目录内所有旧 Parquet 文件。

所有新候选在入库前执行此检查，包括崩溃后恢复的旧归档。每五分钟巡检也会复查每拓扑最新有效样本的 AC 连续性。完成 2 万条时，先对所有接受样本重算波形标签并再次执行全量 AC 质量门控，通过后才进入 complete 并生成正式划分。`quality_gate_checks.json` 保存最近一次门控检查结果；它是时间快照，不是运行中全体新数据的持续证明。

本次升级后的日志见 `supervisor_quality_stdout.log` / `supervisor_quality_stderr.log`、`monitor_quality_stdout.log` / `monitor_quality_stderr.log`，启动信息在 `launch_quality.json`。主数据集 `data/fusion_v2/` 不受影响。
