# 自动监督训练：独立运行副本

实际策略与结果见 [监督报告](../../reports/autonomous_training.md)，实时运行状态见 [controller status](../../monitoring/autonomous_training_v1/status.json)。所有路径基于 `D:\EDA`，仅 Windows 原生环境。

| 文件 | 用途 |
| --- | --- |
| `controller.py` | 后台协调、身份核对、资源门槛、模型选择与有限资源恢复 |
| `head_train.py` / `head_supervisor.py` | 已执行的 CPU 输出头断点续训及监督 |
| `config_head.json` | 第 26 轮继承、独立 recipe、验证早停和滚动改善规则 |
| `stream_graphs.py` | 有界 mmap LRU、原图批构建、有效批 Mask 损失加权 |
| `joint_train.py` | 新 recipe 的完整主干微调，分层小规模 pilot 后再完整训练 |
| `check_stream.py` / `StreamGraphs(..., verify=True)` | CPU 数学等价与真实缓存/图结构检查 |
| `stream_checks.json` / `stream_integrity.json` | 已实际运行的检查证据 |

输出头在累计第 42 轮正常早停，按验证 hit@5% 保存第 36 轮。联合阶段等待至少 4 GiB 可用主机内存，不读取测试集。GPU 检查和实际收益尚未发生时不能视为通过。

控制器在运行时固定源码和配置哈希，不要直接编辑这些文件。父产物均保留；输出位于 `artifacts/experiment1_autonomous_head_v1/`，联合阶段使用另一个独立目录。

仅在确认控制器与任何联合阶段均未运行、没有已有联合阶段待恢复产物时，从工作区启动：

```powershell
Set-Location D:\EDA
Start-Process -FilePath .\.venv\Scripts\python.exe -ArgumentList '-X','utf8','-B','-u','worktrees/autonomous_training_v1/controller.py' -WorkingDirectory D:\EDA -WindowStyle Hidden -RedirectStandardOutput monitoring\autonomous_controller_manual_stdout.log -RedirectStandardError monitoring\autonomous_controller_manual_stderr.log
```

当前控制器已经启动，不要再次运行此命令。重启操作系统后不会自动成为开机服务；有联合产物时先审核断点和原进程身份再恢复。

轻量读取进度，不载入 Torch 或模型：

```powershell
Get-Content -Encoding utf8 monitoring\autonomous_training_v1\status.json
Get-Content -Encoding utf8 monitoring\current_training.json
```
