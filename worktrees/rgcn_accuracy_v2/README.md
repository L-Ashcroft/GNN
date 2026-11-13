# RGCN 精度改进候选：契约残差输出头与训练集校准

> ????????????????????????????19?????6??GPU?????????????????[?????????](../../reports/experiment1_id_results.md)??????? `versions/rgcn_calibrated_head_v1`???GPU?????combined???????????????????GPU???

完整说明：[诊断与改动报告](../../reports/rgcn_accuracy_revision.md)。实现已通过11项CPU检查，包括完整4层/128维/48关系模型在真实小批次上的反向传播与AdamW更新。GPU及完整验证精度对照仍待执行，本候选尚未替换当前训练。

主要文件：

- `src/model/contract_head.py`：原共享读出＋契约专属非线性残差分支，137个契约启用，5个训练样本不足256的契约保持原读出容量。
- `src/model/accuracy_network.py`：原缓存RGCN主干，接入可配置新头。
- `src/train/target_calibration.py`：仅使用训练集拟合逐契约、逐指标的symlog中心/标准差，绑定数据与划分哈希。
- `src/train/accuracy_loss.py`、`accuracy_loop.py`：明确的新标准化Loss，保持缺失Mask与契约隔离，评价还原至原物理单位。
- `src/gnn/accuracy_inference.py`：新格式Checkpoint严格加载与原单位预测。
- `scripts/train_accuracy_v2.py`：独立recipe、输出命名空间、断点和日志；原训练存活或CUDA启动时RAM不足4 GiB即拒绝启动。
- `config/`：baseline、head_only、calibration_only、combined四组公平对照。数据、划分及其他超参数一致。
- `results/`：实际来源审计、CPU检查、工程拟合和实现指纹。

训练统计文件为 `data/model_ready/fusion_v2_accuracy_v2/target_calibration.json`。核对已有统计（不会覆盖）：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B worktrees/rgcn_accuracy_v2/scripts/fit_targets.py --verify-existing
.\.venv\Scripts\python.exe -X utf8 -B worktrees/rgcn_accuracy_v2/scripts/test_accuracy.py
```

从原缓存重新拟合应使用一个新的、明确的 `--output` 路径；更换训练划分后还需要相应的新缓存/role信息和新统计，不能沿用旧划分的校准。

当前正式训练继续作为基线。待它结束、资源满足并完成GPU小批验收后，独立运行新实验，例如：

```powershell
.\.venv-gpu\Scripts\python.exe -B -u worktrees/rgcn_accuracy_v2/scripts/train_accuracy_v2.py --config worktrees/rgcn_accuracy_v2/config/combined.json
```

上述命令现在会被进程门禁阻止，不会启动第二个GPU训练。新模型有不同的Loss和state_dict格式，不允许直接拿旧实验Checkpoint跨recipe续训。旧原始数据、Mask、标签语义、图、划分和Checkpoint保留。

小批Loss下降只证明实现可优化，不证明验证精度优于当前模型。最终应比较四组的原单位验证指标后决定采用哪种修改。未训练监督契约/未知域的零样本预测问题仍需单独处理。
