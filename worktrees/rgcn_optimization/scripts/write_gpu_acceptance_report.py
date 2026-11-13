"""Render current CPU-only preflight evidence without implying GPU acceptance."""
from datetime import datetime,timezone
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];WORKSPACE=ROOT.parents[1];RESULTS=ROOT/'results/gpu_acceptance'
def read(path):return json.loads(path.read_text(encoding='utf-8'))

def main():
    pointer=read(RESULTS/'latest_preflight.json');preflight=read(Path(pointer['path']))
    if preflight['status']!='deferred':raise RuntimeError('Preflight is now eligible; execute GPU acceptance rather than issue a deferred report')
    assert not preflight['torch_imported'] and not preflight['CUDA_work_started']
    environment=preflight['environment_after'];status=environment['production_status'];identity=environment['original_process']
    audit=preflight['real_batch256_audit'];fingerprints=preflight['fingerprints'];gate=read(RESULTS/'gate_tests.json')
    limits=fingerprints['thresholds_atol_rtol'];receipt=Path(pointer['path']).relative_to(WORKSPACE).as_posix()
    rows='\n'.join(f'| {name} | {values[0]:g} | {values[1]:g} | 待 GPU 验证 |' for name,values in limits.items())
    acceptance={'created_at_UTC':datetime.now(timezone.utc).isoformat(),'status':'deferred',
        'GPU_acceptance_passed':False,'formal_release_allowed':False,'blocking_reasons':preflight['blocking_reasons'],
        'preflight_evidence':receipt,'GPU_correctness':preflight['GPU_correctness'],'GPU_performance':preflight['GPU_performance'],
        'model_source_fingerprints_verified':fingerprints['all_passed'],'thresholds_unchanged':limits,
        'optimized_execution_sha256':fingerprints['optimized_execution_sha256'],
        'original_recipe_sha256':fingerprints['production_recipe_sha256'],
        'formal_version_created':False,'formal_output_directory_created':False,'formal_training_command':None,
        'CPU_only_preflight_passed':fingerprints['all_passed'] and audit['passed'] and gate['passed']}
    (RESULTS/'acceptance.json').write_text(json.dumps(acceptance,indent=2,ensure_ascii=False),encoding='utf-8')
    text=f'''# 边中心 RGCN 优化版 GPU 验收报告

检查时刻：{environment['checked_at_beijing']}（北京时间）。状态：**DEFERRED / 尚未完成 GPU 验收**。

**本次没有执行 CUDA 正确性测试或 GPU 性能测试，也没有创建优化版正式运行版本。** 原训练仍在运行，且主机可用内存不足 4 GiB；按本次任务的限制保留待测状态。这不表示数值测试失败，也不能视为验收通过。

## 1. 实际运行环境检查

| 项目 | 实测结果 |
| --- | --- |
| 原正式训练 PID | {identity['pid']} |
| 快照记录的创建时间戳 | {identity['expected_create_time']} |
| 当前进程创建时间戳 | {identity['create_time']} |
| 是否为同一个原始进程 | 是，PID 与创建时间精确相同；不是仅凭旧 PID 判断 |
| 创建时间（北京时间） | {identity['created_at_beijing']} |
| 命令与工作目录 | `scripts/train_experiment1.py --resume`，`D:\\EDA` |
| 检查时正式训练状态 | {status['state']}；第 {status['epoch']} 轮，第 {status['batch']} / {status['batches_per_epoch']} 批 |
| 主机可用内存 | **{environment['available_RAM_GiB']:.3f} GiB** |
| 大型 CUDA 测试最低要求 | 原训练不运行，且至少 4 GiB 可用主机内存 |
| 完整大批次 profiler 的额外门限 | 至少 6 GiB 可用主机内存 |
| 显卡 / 驱动 / 总显存 | RTX 4050 Laptop / 610.60 / 6141 MiB（`nvidia-smi` 只读查询） |
| GPU 环境安装记录 | 静态读取 `torch/version.py`：2.9.1+cu128，CUDA 12.8；未导入 Torch，未建立测试 CUDA 工作 |

同时扫描了工作区中的实际训练命令，识别到原启动器和训练工作进程；检查了原进程创建时间，避免 PID 复用误判。两项独立阻止条件目前都成立：**原训练存活、RAM 不足**。即使仅满足其中一项，也不得开始大型 CUDA 测试。

没有停止、暂停、重启或修改原训练。没有读取/覆盖检查点内容，没有向原正式产物目录写入文件，也没有创建新的后台训练或等待任务。原训练自行更新状态和检查点属于正常行为。

## 2. 已实际完成的安全检查

- **{fingerprints['checks_count']} 项指纹检查全部通过**：原工作区源码及正式配置/recipe、冻结原版、优化版与既有 CPU 验收记录的源码、缓存元数据、性能任务定义、数据划分与批次文件。
- 新增 `execution.py` 的当前 SHA256：`{fingerprints['optimized_execution_sha256']}`。已记录供未来正式版本使用；没有改写当前正式 recipe。
- 原正式 recipe SHA256：`{fingerprints['production_recipe_sha256']}`，保持原值。其他实现源码哈希与 CPU 验收时一致。
- 本次真实 Batch=256 只读数据检查通过：样本 ID 唯一，全部为训练分区，来自 {audit['source_topology_ids']} 个来源拓扑 ID，覆盖全部 {audit['relations']} 种关系、{audit['supervision_contracts']} 个监督契约。
- 来源：FALCON {audit['sources']['FALCON']}、AICircuit {audit['sources']['AICircuit']}、LOADBench {audit['sources']['LOADBench']}。
- 批次含 {audit['nodes']} 个节点、{audit['message_edges']} 条消息边、{audit['self_loops']} 条自环、{audit['duplicate_typed_edges']} 条同类型重复消息边；未做边去重或图结构变更。
- 已观测标签 {audit['observed_labels']} 项，全部有限；缺失槽位 {audit['missing_labels']} 项。没有生成或替换标签。
- **{gate['tests_run']} 项 CPU 门禁单元测试通过**：同 PID/创建时间核对、PID 复用、识别新启动的训练、身份无法核实时拒绝、4/6 GiB 阈值和准确命令匹配。测试没有导入 Torch。

数据与哈希检查是 CPU 只读检查，不是 Forward、Backward、实际 Checkpoint 加载或 CUDA 可执行性测试。此前已有的 CPU 权重/优化器状态兼容检查见原优化报告，本次核实其对应模型源码仍一致，没有把旧 CPU 结果当作新 GPU 结果。

## 3. GPU 正确性：尚未执行

将继续使用真实训练 Batch=256、4 层 RGCN、48 种关系、全部独立监督头、FP32、原 Loss 和确定性设定。以下既有阈值没有放宽：

| 对比类别 | atol | rtol | 本次结果 |
| --- | ---: | ---: | --- |
{rows}

逐元素条件仍为 `abs(new-ref) <= atol + rtol*abs(ref)`；最大相对误差仍以 `max(abs(new), abs(ref), 1e-8)` 为分母报告。

| CUDA 验收内容 | 状态 |
| --- | --- |
| Forward、四层节点/边状态、最终预测 | 待测 |
| 节点/边 Attention Pooling 输出与注意力 | 待测 |
| Masked Loss、各监督组 Loss | 待测 |
| 全部参数梯度、裁剪范数 | 待测 |
| 一步 AdamW 后参数 | 待测 |
| Dropout、CPU/CUDA 随机状态一致性 | 待测 |

上述项的 GPU 最大绝对/相对误差均为未知，不填零。没有实际执行数值比较，因此也不作 GPU 数值不通过的判断。

## 4. GPU 性能：尚未执行

| Batch=256、FP32 指标 | 原版 | 优化版 |
| --- | --- | --- |
| 多次完整训练 Step 的均值/范围 | 待测 | 待测 |
| Forward / Backward | 待测 | 待测 |
| GPU 内核启动数 | 待测 | 待测 |
| CPU/GPU 同步次数 | 待测 | 待测 |
| GPU 显存峰值 / 主机 RSS | 待测 | 待测 |
| Samples/s | 待测 | 待测 |

**实际 GPU 加速倍数：未知。** 先前 CPU Batch=16 的 1.26 倍加速不是 RTX 4050 的验收成绩。`nvidia-smi` 的全卡瞬时利用率和显存占用，也不是独立优化版的峰值或性能基准。

后续普通计时与 profiler 必须分开运行，不能将插桩步骤计入正常速度均值；完整 Step 必须包含数据构图、执行计划准备、传输、Forward、Loss、Backward、裁剪、AdamW 和统计。CUDA Event 流跨度用于正确同步计时，不能当作纯内核执行时长。当前没有采集新 trace。

## 5. 正式投入条件与后续命令

**目前不符合正式投入条件。** 未创建正式优化版目录、输出目录、新 recipe 或训练入口；未自动替换原版，也不提供未经验收的正式训练启动命令。

只有完整 GPU 正确性在原阈值下通过、配对性能测试证明有实际收益后，才能建立独立正式版本，记录包括 `execution.py` 在内的实现指纹与 recipe 哈希，选择新输出目录，并再次验证配置/数据加载/Checkpoint 加载及入口。旧检查点和原实验结果必须保留，不能绕过原 runner 的 recipe 哈希保护直接跨版本 `--resume`。

ID、LOTO 和消融的数据划分与监督契约保持原样。CPU 消融检查通过不等于 GPU 全协议验收完成；LOTO 中未训练契约/槽位仍须遵守原 Coverage 限制，不能为运行实验而取消保护。

原训练结束且内存充足后，可先重跑以下**环境检查**（不启动 GPU 测试或训练）：

```powershell
Set-Location D:\\EDA
.\\.venv\\Scripts\\python.exe -B worktrees/rgcn_optimization/scripts/gpu_preflight.py
```

条件满足后，先执行 GPU 正确性验证；此命令也不是正式训练：

```powershell
.\\.venv-gpu\\Scripts\\python.exe -B worktrees/rgcn_optimization/scripts/validate.py --stage cuda_256 --device cuda --batch-size 256
```

门禁现在由原进程创建时间、实际训练命令和当前可用内存共同决定。CUDA 验证/测速因资源延后时退出码为 **75**，数值断言失败也不能算通过；必须检查结果中的实际 `status=passed`。环境检查的 `preflight_passed` 仅表示资源可测试，不等于 GPU 正确性或正式发布已经通过。

## 6. 本次证据

- [本次环境、进程、指纹与真实批次记录](../{receipt})。
- [CPU 门禁测试日志](../worktrees/rgcn_optimization/results/gpu_acceptance/gate_tests.log)。
- [结构化验收状态](../worktrees/rgcn_optimization/results/gpu_acceptance/acceptance.json)：`GPU_acceptance_passed=false`、`formal_release_allowed=false`；GPU 数值和加速倍数保持空值。
- [先前优化实现报告](rgcn_optimization_report.md)与[隔离副本说明](../worktrees/rgcn_optimization/README.md)。
'''
    path=WORKSPACE/'reports/rgcn_gpu_acceptance_report.md';path.write_text(text,encoding='utf-8')
    print('Deferred GPU acceptance report written:',path)

if __name__=='__main__':main()
