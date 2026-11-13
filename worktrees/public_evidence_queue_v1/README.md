# 公开数据对照与消融的继续执行

2026-11-11建立。仅使用公开融合50万条数据；保留原结果、划分、标签和模型权重，不修改论文。

已有ID、SimpleOpAmp4首折LOTO、目标域少样本和全量监督结果；跨数据集零样本目前只有监督兼容性审查，没有有效准确率。此次补充的是同神经头ID参考、结构消融和第二折LOTO。不能将运行检查当作新增实验成绩。

## 任务顺序

| 顺序 | 独立输出目录（位于artifacts） | 目的 |
| --- | --- | --- |
| 1 | `rgcn_memory_safe_loto_v2` | 正在运行的M02三阶放大器LOTO折 |
| 2 | `rgcn_memory_safe_id_control_v2` | 恢复已有冷启动ID断点，129组兼容映射 |
| 3 | `rgcn_memory_safe_no_film_v2` | 仅关闭FiLM；保留域嵌入 |
| 4 | `rgcn_memory_safe_static_edges_v2` | 保留初始边状态，关闭动态边状态更新 |
| 5 | `rgcn_memory_safe_collapsed_relations_v2` | 合并关系类型为一种；保留连接、器件参数和节点端口统计 |

四个ID任务使用同一400,083／50,861／49,056划分、随机种子42、4层128维、有效批256／微批16、相同优化与验证选模规则，最多20轮。三项消融各只改变一个声明的开关。合并关系会降低参数量，它不等于标准GCN，也不表示删除全部端口信息；应连同参数量解释。第二折LOTO保持其原有结构留出划分。

已有暖启动微调`rgcn_memory_safe_id_v2`保留，不插入这组从零对照。新ID与旧首折LOTO即使神经头一致，训练条数和测试群体仍有差别；完成后应先比较同一目标的ID测试交集，不能把两个总体分数相减称为纯结构泛化损失。

本五项全部完成且监督器退出后，独立的[受控实验续行实现](../controlled_evidence_v1/README.md)再执行LOADBench单源／多源和MLP／GCN。续行监控目前只观察与核验已冻结结果，不并行启动其他大型训练器。

## 已完成运行检查

`artifacts/public_evidence_queue_v1/preflight.json`：5张真实TRAIN图，覆盖三个来源、48种关系，四个模型各完成两次CPU训练Step，Loss和梯度范数有限。未访问测试标签，未保存为正式训练权重。进程结束时RSS约351MiB，该值不是正式大批训练峰值。

各模型参数量：完整8,146,624；去FiLM8,014,528；静态边7,883,456；合并关系1,980,224。FP32、Dropout 0.1及确定性设置保留。

## 自动执行与状态

监控文件：`monitoring/public_evidence_queue_v1/status.json`。全局入口：`monitoring/current_training.json`。当前训练自己的状态路径在监控的`runtime_status`字段中（启动后出现）。

`training_active=true`须同时满足训练阶段与PID、创建时间、脚本路径相符；`waiting_resources`表示没有正式训练在计算，不能根据旧状态推断正在训练。

启动门槛为CPU可用主机内存1.4GiB，需连续稳定20秒。使用单个低优先级进程；内存压力下由训练器在批边界保存断点，编排器冷却后恢复。两次资源退出后门槛升至1.8GiB，四次后升至2.2GiB，避免反复崩溃。源码、配置及队列均固定哈希，变化时请求安全停止；续训使用该任务自己的原配方，不跨版本绕过recipe保护。

此编排器当前仅执行CPU，未使用未验收的CUDA，也未放宽4GiB GPU主机内存门槛。按过去低内存吞吐约19.5条/秒，40万训练记录每轮约5.7小时，另加验证；五项完整串行任务可能超过数天。预计新增产物约2–3GiB，磁盘空间已核查充足。资源等待不计作实际训练吞吐。

已有后台实例时不要重复启动。没有实例时的Windows原生入口：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -u .\worktrees\public_evidence_queue_v1\supervise.py
```

安全停止：在`monitoring/public_evidence_queue_v1/STOP`创建文本文件。它会请求当前任务保存断点，并停止后续队列；不会杀死进程。未知的代码或数值错误进入`requires_review`，不擅自换模型或读取测试标签调参。监控仅在电脑与后台进程仍运行时有效，不是开机自启服务。
