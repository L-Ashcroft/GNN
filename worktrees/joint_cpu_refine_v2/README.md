# 完整 GNN 的低学习率 CPU 联合训练

当前已由隐藏后台监督器启动；不要重复启动。运行细节与实测见 [训练报告](../../reports/joint_cpu_training.md)。

本副本继承原 4 层/128 维/48 关系校准 RGCN，保留全部数据、划分、契约和监督损失。`train.py` 更新完整主干与输出头，CPU 4 线程、有效 Batch256、微批128；主干 LR2e-6、适配 LR1e-5、100 Step 预热，使用全部训练/验证分区。它是新 recipe 的权重暖启动，不能跨版本 --resume。

| 文件 | 作用 |
| --- | --- |
| `train.py` | 带预热、完整验证与独立断点的联合训练 |
| `stream_graphs.py` | 有界原始图/参数加载，不使用冻结图嵌入训练主干 |
| `config.json` | 当前固定配置 |
| `supervisor.py` | 身份/哈希检查、资源暂停与有限恢复 |
| `origin.json` | 源码来源、被拒绝试训与本次变更说明 |
| `prepare.py` | 一次性生成副本，已有副本时拒绝覆盖 |

实际产物在 `artifacts/experiment1_joint_cpu_refine_v2/`，实时监督在 `monitoring/joint_cpu_refine_v2/status.json`。父模型及短试训产物完整保留。此副本不创建 CUDA 上下文，原 CUDA 资源门槛仍保留。
