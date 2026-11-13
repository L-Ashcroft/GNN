# 独立目标域实验三、四

本目录使用已冻结的 `versions/rgcn_calibrated_head_v1` 图层实现，不改写公开实验或个人原始样本。

- `audit.py`：实验三的跨来源监督兼容准入，不能将未验证目标强行拼接。
- `data.py`：20,000条个人样本的磁盘映射缓存；原anchor-safe分区16,000/2,000/2,000不变。
- `model.py`：35维节点、90维边、52种关系、46维已知条件；按名称迁移公共48种关系及图编码器，新增衬底连接保留。目标输出头、域调制及条件编码器不复用公开输出语义。
- `checks.py`：真实20条训练记录的缓存、梯度、微批次权重、迁移、预算与检查点自检，不用测试记录。
- `suite.json`：首次测试前固定11项任务。每拓扑k=5/10/20/50/100，两种初始化同架构、同优化设置比较；另有全量目标域参考模型。所有任务全参数训练。
- `train.py`：单任务CPU训练、断点、recipe保护、训练集统计及选择冻结后独立测试。
- `supervise.py`：只运行一个训练任务；核对PID创建时间，内存不足暂停并按相同recipe恢复，非资源错误停止队列。

少样本预算为每拓扑k条，总10k条；另用2,000条固定验证标签，必须报告总监督预算10k+2,000。测试集2,000条不拟合、不调参。

目标symlog标准差采用固定下限0.1，避免极小预算近常数指标支配梯度。这是独立的新recipe，不宣称与先前公开Loss完全相同，不能用旧recipe跨版本resume。

该队列是seed42的初步对照，并非多随机种子验证。全量目标域模型只作为经验参考线，不保证是理论上界。无有效零样本结果时不计算适配效率η。

PowerShell（工作目录D:\EDA）：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B worktrees/personal_transfer_v1/audit.py
.\.venv\Scripts\python.exe -X utf8 -B worktrees/personal_transfer_v1/checks.py
.\.venv\Scripts\python.exe -X utf8 -B -u worktrees/personal_transfer_v1/supervise.py
```

监控入口：`monitoring/experiment34_personal_v1/status.json`；自动报告：`reports/experiment34_progress.md`。

数据采集曾使用校准仿真，既有anchor-safe划分已隔离其测试角色。下游10k+2,000是模型适配的标签预算，不是生成整套数据的总仿真成本；局部采样区域内仍可能存在近邻相似性。
