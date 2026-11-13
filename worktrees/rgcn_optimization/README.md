# 边中心 RGCN 优化隔离副本

> 2026-10-31 更新：[GPU 验收报告](../../reports/rgcn_gpu_acceptance_report.md)。本目录的激进分组版未通过 GPU AdamW 容差，未投入训练；另行实现的[保守缓存版](../../versions/rgcn_cached_fast_v1/README.md)通过真实 B256 验收，实测约 1.29×，已在独立目录从原断点继续训练。下文待测状态为历史记录。

这是隔离副本，不是 Git Worktree：原工作区没有 Git 元数据。所有模型修改均位于本目录，正式 `D:\EDA\src`、`config` 和训练产物没有修改。

- `src/`：优化后的副本；核心变化为 `model/execution.py`、`model/layers.py`、`model/heads.py`、`model/network.py`、`graph/builder.py`、`train/losses.py`、`train/loop.py`。
- `reference_src/`：冻结原版，仅供对照，不应修改。
- `snapshot.json`：原始文件哈希、正式进程 ID/创建时间和保护文件清单。
- `scripts/validate.py`：真实小批次 Forward、层状态、池化、Loss、梯度、AdamW 和随机状态的对照。
- `scripts/test.py`、`tests/test_equivalence.py`：边界、真实 Batch=256 组件和消融兼容检查。
- `scripts/benchmark.py`：独立计时，包含真实缓存读取/构图、执行计划准备、传输、Forward、Loss、Backward、裁剪、AdamW 与统计。
- `scripts/summarize.py`、`results/summary.json`：汇总结果及正式源码未变的证据。

完整报告：[rgcn_optimization_report.md](../../reports/rgcn_optimization_report.md)。

在 `D:\EDA` 的 PowerShell 中执行 CPU 检查：

```powershell
.\.venv\Scripts\python.exe -B worktrees/rgcn_optimization/scripts/validate.py --stage 4_final
.\.venv\Scripts\python.exe -B worktrees/rgcn_optimization/scripts/test.py
.\.venv\Scripts\python.exe -B worktrees/rgcn_optimization/scripts/summarize.py
```

这些脚本显式读取原工作区的只读真实缓存/训练分区批次，结果只写到本副本 `results/`。性能测试使用冻结原版 `CachedFusion.batch` 的未改动 AST 函数体，避免其模块导入链加载无关仿真脚本。不是另造一套图张量或标签。

CPU Batch=16 的配对测速（分别运行，避免多个诊断进程同时占内存）：

```powershell
.\.venv\Scripts\python.exe -B worktrees/rgcn_optimization/scripts/benchmark.py --variant reference --batch-size 16 --profile-forward
.\.venv\Scripts\python.exe -B worktrees/rgcn_optimization/scripts/benchmark.py --variant optimized --batch-size 16 --profile-forward
```

正式训练结束、主机内存充足后，先验证 CUDA 正确性，再测速。**当前 GPU 测试被主动延后，不要为运行这些命令停止正式训练。** CUDA 脚本现在共同检查原 PID 的创建时间、实际工作区训练命令和可用内存；不能仅凭旧 PID 判定进程身份。资源不满足时，在导入 Torch/CUDA 前写入 deferred 记录并以退出码 75 返回；延后不算验收通过。普通测试要求至少 4 GiB 可用主机内存，大 trace 至少 6 GiB；运行时低于 640 MiB 只终止诊断进程。

可先运行不加载 Torch 的环境/指纹/真实批次检查（不自动开始 GPU 测试或训练）：

```powershell
.\.venv\Scripts\python.exe -B worktrees/rgcn_optimization/scripts/gpu_preflight.py
.\.venv\Scripts\python.exe -B worktrees/rgcn_optimization/scripts/check_gpu_gate.py
```

`preflight_passed` 仅表示可以开始后续测试。GPU 正确性必须在原容差下实际通过，性能也必须实际测得收益，才能准备独立正式版本；不能根据环境检查或 CPU 加速结果发布。

```powershell
.\.venv-gpu\Scripts\python.exe -B worktrees/rgcn_optimization/scripts/validate.py --stage cuda_256 --device cuda --batch-size 256
.\.venv-gpu\Scripts\python.exe -B worktrees/rgcn_optimization/scripts/benchmark.py --variant reference --device cuda --batch-size 256 --profile-forward --profile-size 256 --profile-full-step
.\.venv-gpu\Scripts\python.exe -B worktrees/rgcn_optimization/scripts/benchmark.py --variant optimized --device cuda --batch-size 256 --profile-forward --profile-size 256 --profile-full-step
```

大 trace 消耗较多内存；`--profile-size 16` 可先用于较小算子检查。GPU 阶段用 CUDA Events 且同步完成；Event 流跨度不能当纯内核计算时间。CUDA trace 可在随后 CPU 进程中统计 `kernel` 和 `cuda_runtime` 同步事件，避免在 CUDA 进程内再次物化大型 profiler 事件树。原有 JSON 键 `CPU_forward_ops_batch16` / `CUDA_trace_batch16` 是历史兼容命名，实际范围以 `operator_profile_size` / `operator_profile_scope` 为准。

这里没有启动训练的入口，复制的配置也不是可直接覆盖正式实验的指令。不要直接把 `src/` 覆盖回正式目录；CUDA 验收通过后，后续实验需要独立输出目录、更新实现指纹（包含新增 `execution.py`）和明确的数据路径。模型权重/优化器参数顺序兼容，不代表旧 runner 的配方哈希允许跨版本直接 `--resume`。
