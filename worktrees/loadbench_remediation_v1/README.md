# LOADBench独立修正版

本目录不修改原训练或数据。详见[完整排查报告](../../reports/loadbench_remediation.md)。

运行（在D:\EDA的PowerShell中）：

```powershell
.\.venv\Scripts\python.exe -B worktrees/loadbench_remediation_v1/test_readout.py
.\.venv\Scripts\python.exe -B worktrees/loadbench_remediation_v1/run.py --probe
.\.venv\Scripts\python.exe -B worktrees/loadbench_remediation_v1/run.py
.\.venv\Scripts\python.exe -B worktrees/loadbench_remediation_v1/calibrate.py
.\.venv\Scripts\python.exe -B worktrees/loadbench_remediation_v1/publish.py
```

均可恢复；recipe不一致时拒绝复用。索引只含原TRAIN标签，不读取TEST标签。

接口：将本目录加入Python路径，`from model import CorrectedReadout`，调用`CorrectedReadout().predict(group, parameters, embedding, published_prediction, parameter_names=names)`。parameters为二维SI原始设计参数，必须遵循bundle中记录的参数顺序；embedding与published_prediction由原冻结GNN模型提供，预测输出仍为原symlog空间。92组真实验证接口已回放，最大误差0。其他已知来源逐位透传；未知契约拒绝。不支持LOTO，不直接替换旧训练入口。

状态：验证接受，冻结后固定测试完成。LOADBench验证49.32%→51.83%，负R²目标仍31；固定测试49.40%→51.92%，负R²仍38，宏平均R²仍很负。原标签/Mask/分区、在训LOTO均保留。敏感性质量筛查不是正式选优策略，也不是已认证的新标签。

冻结测试入口（只评估，完成后不会重复覆盖）：`python -B scripts/test_loadbench_correction.py --probe`及`python -B scripts/test_loadbench_correction.py`。要求2GiB可用主机内存，CPU单线程。后续不得根据该测试分数重选本版本；详见evaluation_receipt.json。
