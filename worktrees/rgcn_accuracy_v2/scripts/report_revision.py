"""Record actual audit/tests and freeze candidate fingerprints, CPU only."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json
import hashlib
import psutil

PACKAGE=Path(__file__).resolve().parents[1];ROOT=PACKAGE.parents[1]
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    audit=read(PACKAGE/'results/head_audit.json');tests=read(PACKAGE/'results/cpu_tests.json');tiny=read(PACKAGE/'results/cpu_tiny_fit.json')
    assert tests['status']=='passed' and not tests['skipped']
    snapshot=read(PACKAGE/'baseline_snapshot.json');assert all(sha(ROOT/p)==v for p,v in snapshot['protected'].items())
    baseline=ROOT/'artifacts/experiment1_fusion_id_fast_v1';status=read(baseline/'status.json');launch=read(baseline/'launch.json')
    assert psutil.Process(status['pid']).create_time()==launch['training_create_time']
    monitor=read(ROOT/'monitoring/experiment1_fusion_id_fast_v1/status.json')
    assert monitor['training_identity_matches'] and monitor['protected_fingerprints_match']
    assert not (baseline/'training_stderr.log').stat().st_size
    calibration_path=ROOT/'data/model_ready/fusion_v2_accuracy_v2/target_calibration.json'
    calibration=read(calibration_path)
    for p in PACKAGE.rglob('*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')
    summary={'state':'implemented_CPU_verified_GPU_and_validation_comparison_pending',
        'at_Asia_Shanghai':datetime.now(timezone(timedelta(hours=8))).isoformat(),'baseline_training_running':True,
        'baseline_pid':status['pid'],'baseline_epoch':status['epoch'],'baseline_batch':status['batch'],
        'baseline_code_and_recipe_unchanged':sha(baseline/'recipe.json')==snapshot['baseline_recipe_sha256'],
        'CPU_tests':tests,'GPU_tests_run':False,'new_formal_training_started':False,
        'CUDA_start_gate':'Live invocation refused another training before torch import; no new run directory created',
        'normalization_fit_role':'train','normalization_validation_labels_used':False,'normalization_test_labels_used':False,
        'target_calibration_sha256':sha(calibration_path),
        'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted((PACKAGE/'src').rglob('*.py'))},
        'entry_sha256':sha(PACKAGE/'scripts/train_accuracy_v2.py'),
        'config_sha256':{p.name:sha(p) for p in sorted((PACKAGE/'config').glob('*.json'))}}
    assert summary['baseline_code_and_recipe_unchanged']
    assert not any((ROOT/'artifacts'/('experiment1_accuracy_v2_'+name)).exists() for name in ['baseline','head_only','calibration_only','combined'])
    (PACKAGE/'results/revision_status.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    source_rows=[]
    for source,r in audit['by_source'].items():
        n=r['train_rows_per_contract_min_median_max']
        source_rows.append(f"| {source} | {r['contracts']} | {n[0]} / {n[1]:g} / {n[2]} | {r['groups_below_adapter_threshold']} | {r['model_worse_than_train_mean_mae']} / {r['target_combinations']} |")
    worst_rows=[]
    for r in audit['worst_r2'][:5]:
        worst_rows.append(f"| {r['semantic_id']} | {r['model_mae']:.6g} | {r['train_mean_baseline_mae']:.6g} | {r['calibrated_mean_baseline_mae']:.6g} |")
    report=f'''# RGCN 精度改进：输出头诊断与独立候选版

生成时间：{summary['at_Asia_Shanghai']}（北京时间）。状态：**修改已经实现，11 项 CPU 检查通过；GPU 与完整验证集的精度对照尚未完成。当前基线训练保持运行。**

## 结论与依据

来源间效果差异不能直接归因于输出头接错。实际实现已有 **142 个独立监督契约的最终输出层**，不同物理定义、测试条件的标签没有并入一个输出头；但最终输出层前面的 shared/family 非线性层跨来源共享。该结构可能限制不同契约的非线性表达，尚无隔离消融证明它是主要根因。

同时，原目标用固定工程单位尺度做 symlog，未按每个契约/指标的训练分布中心化和缩放。不同目标的误差贡献差异明显；共享特征是否足够、目标尺度是否失衡应分别验证。不能从负 macro R² 直接推断整套网络失效。

本次审阅第 {audit['analyzed_epoch']} 轮完整验证。FALCON / AICircuit / LOADBench 的 R² 中位数分别约 0.786 / 0.609 / 0.383。对同一验证集使用**只在训练集拟合的原单位均值**作基线：

| 来源 | 监督契约数 | 每契约训练样本 最小 / 中位 / 最大 | 训练数不足256的契约 | 当前模型MAE比训练均值差的指标组合 |
| --- | ---: | --- | ---: | ---: |
{chr(10).join(source_rows)}

LOADBench 大多数契约并不比 FALCON 的契约样本更少；不能把来源差异全部解释为小样本。其指标组合更多（1286），任务分布也不同，来源间 R² 不是同一任务的公平难度比较。上述常量基线未拟合验证标签，验证标签只用于评价。

几个差项的原单位 MAE 如下。第三列是训练原单位均值基线，第四列是训练 symlog 均值逆变换后的基线；二者不保证相同，也不等于训练后新模型的效果。

| 指标 | 第16轮模型MAE | 训练原单位均值MAE | 校准中心逆变换MAE |
| --- | ---: | ---: | ---: |
{chr(10).join(worst_rows)}

这些目标的低方差会放大 R² 的负值，但模型确实比简单基线更差，需要改善，不能仅把负值解释成统计显示问题。

## 已实施的修改

### 1. 独立的非线性残差输出分支

保留原有共享层、指标族层及全部独立最终读出。对有足够训练样本的契约增加 `128 → 32 → SiLU → 该契约槽位数` 的小型分支，直接读取原图级 embedding，输出契约专属修正。最后一层初始为零；在关闭校准的 head-only 模式下，初始预测与原输出头相同。

分支路由按原监督契约，不按指标名称粗暴合并。修改契约 A 的专属分支不会影响契约 B 的输出；共享主干/共享基座训练时仍会发生多任务梯度交互，本次并未宣称彻底消除负迁移。

仅依据训练样本数量，阈值设为256，实际 **137 个契约启用分支、5 个小样本契约沿用共享基座与独立线性读出**。阈值是工程先验，不是用验证集调出的最优值。新增参数约612,495，总参数 **{tiny['parameters']:,}**，主干保持4层、128维和48种关系。

### 2. 训练集拟合的逐契约、逐指标校准

沿用原始物理单位与工程变换：`z = symlog(y / engineering_scale)`。仅使用原8:1:1划分中的 **400,083 条训练记录**，分别拟合每个契约/指标的中心 `mu` 和标准差 `sigma`。验证/测试标签不参与拟合或分支启用判断。

新头输出：

```text
z_prediction[g,k] = mu[g,k] + sigma[g,k] * (shared_readout[g,k](h) + contract_residual[g,k](h))
y_prediction[g,k] = inverse_symlog(z_prediction[g,k]) * engineering_scale[g,k]
```

标准差下限为 `max(1e-6, 32 * float32_eps * max(1, abs(mu)))`，仅防止常量/近常量目标导致除零或放大舍入噪声。常量目标仍保留为观测标签，不通过 Mask 隐藏。1491 个有训练观测的目标组合全部保留；另有2个声明槽位无训练观测，只记录“未拟合”的恒等回退，不虚构均值/方差，如被当作观测监督会报错。

校准文件独立保存于 `data/model_ready/fusion_v2_accuracy_v2/target_calibration.json`，绑定原任务定义、缓存 manifest、划分、源数组及训练样本ID哈希。原始数据、标签、Mask、图和划分保持原样。新的统计参数不会写回旧缓存。

### 3. 明确的新 Loss，与原实验分开

```text
L_group = mean_over_observed_labels( ((z_prediction - z_target) / sigma[group,slot]) ** 2 )
L = mean_over_present_groups(L_group)
```

Mask 仍只标记真实缺失，组间等权、有效标签计数与原方案保持相同口径；**改变的是目标误差尺度，因此梯度和优化目标明确不同于旧版**。标准化 Loss 不能与原实验的 symlog Loss 数值直接比较。验证仍逆变换到原物理单位计算各契约/指标的 MAE、R²、相对误差与 hit@5/10，未截断负R²或修改评价标签。

## 对照实验接口

四组配置保持同一数据、划分、seed42、4层/128维/48关系、Batch256、AdamW、学习率、scheduler 和早停规则。每组为新的独立 recipe/output，不从旧 recipe 直接 --resume：

| 配置 | 契约残差分支 | 训练集校准及标准化Loss | 输出目录 |
| --- | --- | --- | --- |
| `baseline.json` | 关 | 关 | `artifacts/experiment1_accuracy_v2_baseline` |
| `head_only.json` | 开 | 关 | `artifacts/experiment1_accuracy_v2_head_only` |
| `calibration_only.json` | 关 | 开 | `artifacts/experiment1_accuracy_v2_calibration_only` |
| `combined.json` | 开 | 开 | `artifacts/experiment1_accuracy_v2_combined` |

配置位于 `worktrees/rgcn_accuracy_v2/config/`。baseline是为本轮比较准备的重新训练对照，不冒充原训练已经产生的实验结果。必须以原单位验证指标决定是否采用 combined；单次小批次下降不能作为更高精度的证明。正式测试集仍仅在验证选定后使用，不能用四组测试结果挑选结构。

## 实际验证结果

- **{tests['tests']} 项 CPU 检查全部通过，无跳过。** 覆盖 train-only 拟合（任意修改人工单元测试中的验证/测试标签，拟合参数不变）、契约/槽位语义绑定、零残差初始一致性、契约分支隔离、缺失位置NaN、观测非有限值拒绝、无标签拒绝、未训练槽位拒绝、Loss公式/梯度、物理单位恢复、Checkpoint及评估接口。
- 完整4层/128维/48关系模型使用4条真实训练电路，CPU完成6步固定批次工程拟合：Loss **{tiny['losses'][0]:.6f} → {tiny['losses'][-1]:.6f}**。该检查关闭Dropout以检查可优化性；另在 train 模式下以原Dropout完成反向、裁剪和AdamW一步，有限性检查通过。它不是验证精度或泛化实验。
- 新格式推理加载器保存/加载校准 buffer，严格恢复同一预测；原模型不能直接通过新格式加载器，校准版也不能把旧头权重静默解释为标准化头权重。
- CPU测试进程峰值RSS约 {tests['peak_RSS_MiB']:.1f} MiB，无CUDA工作；按内存门槛及低CPU优先级运行。
- 实际调用新训练入口时，在导入Torch和创建输出前因现有正式训练存活而拒绝启动；四个新正式输出目录均未创建。原训练PID {status['pid']}（创建时间已核对）继续第 {status['epoch']} 轮，第 {status['batch']} 批，原实现/配置recipe指纹一致，监控正常。

证据位于 `worktrees/rgcn_accuracy_v2/results/`：`head_audit.json`、`cpu_tests.json`、`cpu_tests.log`、`cpu_tiny_fit.json`、`revision_status.json`。

## 使用与剩余限制

模型入口：`src/model/accuracy_network.py`；新头：`src/model/contract_head.py`；Loss与统计：`src/train/accuracy_loss.py`、`accuracy_loop.py`；目标拟合：`src/train/target_calibration.py`；推理：`src/gnn/accuracy_inference.py`。这些路径均位于独立副本 `worktrees/rgcn_accuracy_v2/`。

**GPU数值/吞吐/显存测试以及完整验证精度对照尚未运行。** 当前正式训练继续作基线，不替换或暂停；旧版约1.29×的提速结论不能套用到增加分支的新结构。新结构是否值得采用，应由后续对照结果决定。当前源码是候选版，并非已验收的精度升级正式版。

本改动优先针对已知契约的ID实验。不同契约仍有独立头；LOTO或Cross-Dataset遇到未训练契约/域时依旧拒绝无依据预测，不会把不兼容标签混用。更换划分必须重新仅用新训练分区拟合统计。现有数据的条件缺失、图级读出是否充分、主干负迁移与少数契约样本过少，仍是后续检查项，没有被本次头部修改自动解决。

当前训练结束、主机至少4 GiB可用且GPU资源充足后，可先做GPU小批验收，再独立运行对照。例如 combined 的未来正式命令（当前会被门禁拒绝，不能重复启动当前训练）：

```powershell
Set-Location D:\\EDA
.\\.venv-gpu\\Scripts\\python.exe -B -u worktrees/rgcn_accuracy_v2/scripts/train_accuracy_v2.py --config worktrees/rgcn_accuracy_v2/config/combined.json
```

CPU复现检查：

```powershell
.\\.venv\\Scripts\\python.exe -X utf8 -B worktrees/rgcn_accuracy_v2/scripts/test_accuracy.py
```

目标统计文件不可覆盖；重新拟合需使用新的明确输出路径，并生成新的recipe。所有新增运行都使用独立 `experiment1_accuracy_v2_` 命名空间，原Checkpoint不被覆盖。
'''
    (ROOT/'reports/rgcn_accuracy_revision.md').write_text(report,encoding='utf-8')
    print(json.dumps({'status':summary['state'],'tests':tests['tests'],'baseline_alive':True,'report':'reports/rgcn_accuracy_revision.md'},indent=2))

if __name__=='__main__':main()
