# 实验二：公开50万条的真实留一拓扑训练

已在Windows原生后台启动首折，入口与结果说明见 [训练报告](../../reports/experiment2_loto_training.md)。当前保留4层/128维/48关系RGCN，所有参数从seed42随机初始化，完整训练主干与可跨已验证语义共用的直接/残差输出头。没有ID权重、IDembedding或私有2万条数据。

首折 `LB_M04__hold_SimpleOpAmp4`：train446984、validation50612、test2404，恰好覆盖公开500000条。目标整类连接结构从训练/验证中排除；模型只在同协议其他拓扑的验证指标上选择，目标标签在选择冻结后才评价。

| 文件 | 职责 |
| --- | --- |
| `data.py` | 原始图有界加载、折角色覆盖、已审计语义别名、目标训练注入拒绝 |
| `fit.py` | 按本折train稳定拟合源域输入和语义目标统计 |
| `model.py` | 可迁移边中心RGCN、直接/残差监督头与掩码分组损失 |
| `train.py` | 整轮训练、微批累积、验证早停、严格recipe恢复、最后外部测试 |
| `supervisor.py` | 身份/指纹监督、内存恢复、先验收再CPU/GPU安全迁移 |
| `qualify.py` | 在实际>=4GiB主机可用内存后进行真实训练图CUDA数值验收 |
| `checks.py` / `checks.json` | 已运行CPU正确性、非零主干梯度、有效Batch256性能探针 |
| `prepare_splits.py` | 独立协议版本生成；已发布目录禁止覆盖 |
| `audit_and_report.py` | 只读运行审计与报告；不改运行源码、校准或checkpoint |
| `prepare_paired_id.py` | 同架构ID参考模板准备；不自动启动，不覆盖已发布控制实验 |

当前监督器与训练器已在运行，不要重复启动。查看：

```powershell
Get-Content -Encoding utf8 artifacts\experiment2_loto_v1\LB_M04__hold_SimpleOpAmp4\status.json
Get-Content -Encoding utf8 monitoring\experiment2_loto_v1\LB_M04__hold_SimpleOpAmp4\status.json
```

只有确认原监督/训练均退出且recipe完整时，才可恢复同一入口：

```powershell
Start-Process -FilePath D:\EDA\.venv\Scripts\python.exe -ArgumentList '-X','utf8','-B','-u','D:\EDA\worktrees\loto_training_v1\supervisor.py' -WorkingDirectory D:\EDA -WindowStyle Hidden -RedirectStandardOutput D:\EDA\monitoring\experiment2_loto_manual_stdout.log -RedirectStandardError D:\EDA\monitoring\experiment2_loto_manual_stderr.log
```

通过监控目录的 `STOP` 请求安全停止。电脑需开机、未睡眠；控制器未安装成开机服务。模型或划分发生改变必须新建版本和新recipe，不能跨版本强制resume。

原始runtime协议保留其指纹；人读元数据的历史覆盖数/旧unused文字说明已通过 `data/fusion_v2/splits/verified_loto_full_v2/metadata_clarification.json` 和 `manifest_documented.json` 澄清。实际训练校准和引用计数已独立核对。
