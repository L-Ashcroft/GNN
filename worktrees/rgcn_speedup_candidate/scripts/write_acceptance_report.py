"""Summarize recorded evidence; no CUDA, no training/validation invocation."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import statistics
import psutil

ROOT=Path(__file__).resolve().parents[3]
CANDIDATE=ROOT/'worktrees/rgcn_speedup_candidate'
OUTPUT=ROOT/'artifacts/experiment1_fusion_id_fast_v1'
MONITOR=ROOT/'monitoring/experiment1_fusion_id_fast_v1'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for value in iter(lambda:f.read(2**20),b''):h.update(value)
    return h.hexdigest()
def stats(data):
    times=[v['wall_seconds'] for v in data['measurements']]
    return {'seconds_mean':statistics.mean(times),'seconds_std':statistics.stdev(times),'seconds_min':min(times),'seconds_max':max(times),
        'samples_per_second':data['batch_size']/statistics.mean(times),
        'GPU_peak_allocated_MiB':max(v['GPU_peak_allocated_bytes'] for v in data['measurements'])/2**20,
        'CPU_peak_RSS_MiB':data['CPU_RSS_peak_bytes']/2**20,
        'stages':{name:{k:statistics.mean(v['stages'][name][k] for v in data['measurements']) for k in row}
            for name,row in data['measurements'][0]['stages'].items()}}
def main():
    bench=CANDIDATE/'results/speedup_20261031'
    a=stats(read(bench/'reference_v2_e2e.json'));b=stats(read(bench/'optimized_v2_e2e.json'))
    speed=a['seconds_mean']/b['seconds_mean'];receipt=read(OUTPUT/'migration_receipt.json')
    fresh=read(CANDIDATE/'results/correctness_cuda_cached_v2_fresh.json');parent=read(CANDIDATE/'results/correctness_cuda_cached_v2_parent.json')
    recipe=read(OUTPUT/'recipe.json');status=read(OUTPUT/'status.json');monitor=read(MONITOR/'status.json')
    source_equal=all(sha(ROOT/name)==value for name,value in recipe['implementation_sha256'].items())
    for name,value in parent['optimized_source_sha256'].items():
        assert sha(CANDIDATE/name)==value
        assert sha(ROOT/'versions/rgcn_cached_fast_v1'/name)==value
    old=ROOT/'artifacts/experiment1_fusion_id_v1'
    snapshot=read(ROOT/'monitoring/experiment1_speedup_migration/parent_snapshot.json')
    old_unchanged=all(sha(old/name)==snapshot[key] for name,key in [('last.pt','parent_checkpoint_sha256'),('best.pt','parent_best_sha256'),('recipe.json','parent_recipe_sha256')])
    identity=psutil.Process(status['pid']);launch=read(OUTPUT/'launch.json')
    assert identity.create_time()==launch['training_create_time']
    assert monitor['training_identity_matches'] and monitor['protected_fingerprints_match']
    assert source_equal and old_unchanged
    summary={'at_UTC':datetime.now(timezone.utc).isoformat(),'status':'GPU_correctness_and_end_to_end_speed_passed',
        'reference':a,'optimized':b,'speedup':speed,'wall_time_reduction_fraction':1-1/speed,
        'profiler_B256':'deferred; 6 GiB host RAM gate not met; no measured kernel/synchronization counts',
        'profiler_B16':'deferred by 4 GiB gate on attempted run; no CUDA profiler started',
        'published_source_matches_GPU_tested_source':True,'parent_outputs_unchanged':True,
        'runtime_status_at_report':status,'monitor_at_report':{'monitor_pid':monitor['monitor_pid'],'runtime_health':monitor['runtime_health'],
            'protected_fingerprints_match':True,'recovery':monitor['recovery']},'new_recipe_digest':receipt['new_recipe_digest']}
    (CANDIDATE/'results/gpu_acceptance_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    stage_rows=[]
    for name in a['stages']:
        x,y=a['stages'][name],b['stages'][name]
        field='cuda_event_span_ms' if 'cuda_event_span_ms' in x else 'wall_ms'
        stage_rows.append(f"| {name} | {x[field]:.2f} | {y[field]:.2f} | {'CUDA Event span' if field.startswith('cuda') else 'CPU wall'} |")
    max_rows=[]
    for kind in fresh['maxima']:
        x,y=fresh['maxima'][kind],parent['maxima'][kind];atol,rtol=fresh['thresholds_atol_rtol'][kind]
        max_rows.append(f"| {kind} | {atol:g} / {rtol:g} | {x['max_abs']:.9g} / {y['max_abs']:.9g} | {max(x['max_relative_floor_1e8'],y['max_relative_floor_1e8']):.9g} | {max(x['max_acceptance_ratio'],y['max_acceptance_ratio']):.9g} |")
    def perf(label,key,unit=''):
        return f"| {label} | {a[key]:.4f}{unit} | {b[key]:.4f}{unit} |"
    report=f'''# 边中心 RGCN GPU 验收与提速迁移报告

更新时间：{summary['at_UTC']}。**最终选用的保守缓存版通过 GPU 正确性与完整 Step 测速，已迁移到独立目录继续训练。** 原 `worktrees/rgcn_optimization` 的激进分组版没有通过 GPU AdamW 验收，未投入正式训练。不得把两种实现混为同一验收结论。

## 授权、环境与隔离

用户随后明确授权“安全暂停、验收后迁移；不通过则恢复原版”，替代此前禁止暂停的约束。原 PID 27996 / 创建时间 1793332674.8457127 经身份核对，收到 STOP 后自行保存到第 16 轮已完成 1024 批并退出。没有强杀训练。原 `last.pt`、`best.pt`、recipe 和历史结果保留；本次重新核对其 SHA256 全部一致。

GPU：RTX 4050 Laptop，Torch 2.9.1+cu128，Batch=256、CPU 线程=2，FP32、TF32 关闭、确定性算法开启、CUBLAS `:4096:8`。每个 CUDA 测试在导入 Torch 前执行进程身份扫描及至少 4 GiB 可用主机内存门禁；不足时直接延期。大型 profiler 要求 6 GiB，未满足，未强行运行。

隔离实现位于 `worktrees/rgcn_speedup_candidate/src`，冻结原版位于该目录 `reference_src`。发布到 `versions/rgcn_cached_fast_v1` 后逐文件验证，模型源码与 GPU 验收时相同。原 `D:/EDA/src` 和原正式配置未被覆盖。

## 正确性验收

真实固定批次 `reports/experiment1_performance/fixtures/batch_256_11.pt`：3770 节点、32840 消息边，覆盖 48 种关系、95 个监督契约、84 个来源拓扑，含并行边、自环及缺失标签。使用两种初始状态：固定种子的全新权重，以及原第 16 轮断点的真实权重、AdamW 状态和 CPU/CUDA RNG。

每种状态均检查 eval 与 train/Dropout，逐层节点/边状态、初始边状态、预测、节点/边 Pooling、注意力、总 Loss、各监督契约 Loss、全部参数梯度、裁剪范数及一步 AdamW 更新。随机状态消耗完全相同，state_dict 双向严格加载通过。**原有容差没有放宽**：

| 类别 | atol / rtol | 最大绝对误差：新初始化 / 原断点 | 最大相对误差 | 最大验收比值 |
| --- | --- | --- | --- | --- |
{chr(10).join(max_rows)}

相对误差分母为 `max(abs(reference), abs(new), 1e-8)`；验收比值为 `abs(new-reference)/(atol+rtol*abs(reference))`，必须 ≤1。统计累加的微小误差来自 detached 分段求和，不影响优化 Loss、梯度与模型更新。上述模型计算与更新本次实测绝对误差均为 0，不等于证明所有未来批次均逐位一致。

另有 18 项边界/门禁测试通过，包括无标签、缺失位置 NaN、观测位置非有限值拒绝、并行关系/自环、空图池化、图成员重排、缓存失效，以及关系合并/不更新边/关闭域与 FiLM 三类消融兼容。发布副本的 19 项监控与恢复检查、CPU 检查点验证和真实续训游标加载检查通过。

证据：`worktrees/rgcn_speedup_candidate/results/correctness_cuda_cached_v2_fresh.json`、`correctness_cuda_cached_v2_parent.json`、`edge_case_tests.json`，以及新监控目录内 `migration_checkpoint_validation.json`、`release_integration_checks.json`。

## 性能实测

正常速度基准与 profiler 分开。每个版本单独进程、同一个真实 Batch=256、相同初始种子、1 次预热后 8 次完整 Step。完整时间包括缓存读入/构图、执行计划、CPU→GPU、Forward、Loss、Backward、裁剪、AdamW、Coverage、LossAccumulator；CUDA 在 Step 前后同步，阶段使用 Events 且不在各阶段强制等待。机器为共享 Windows 环境，数值只代表这组实测。

| 指标 | 原版 | 保守缓存版 |
| --- | ---: | ---: |
{perf('完整 Batch 平均秒数','seconds_mean')}
{perf('标准差（秒）','seconds_std')}
{perf('最短（秒）','seconds_min')}
{perf('最长（秒）','seconds_max')}
{perf('吞吐量 samples/s','samples_per_second')}
{perf('GPU 峰值 allocated MiB','GPU_peak_allocated_MiB')}
{perf('主机进程峰值 RSS MiB','CPU_peak_RSS_MiB')}

**实测加速 {speed:.3f}×，吞吐量提高 {(speed-1)*100:.1f}%，每批时间减少 {(1-1/speed)*100:.1f}%。** 不是将旧训练在内存紧张时约 3.3 秒与当前独占测试直接相比；这里原版和新版在同一阶段重新实测。不能据此保证所有拓扑批次与整轮验证都有相同加速。

| 阶段 | 原版 ms | 保守缓存版 ms | 口径 |
| --- | ---: | ---: | --- |
{chr(10).join(stage_rows)}

CUDA Event span 包含 CPU 发射间隙，并非纯 GPU 内核时间；异步阶段可能重叠，不能用相加结果代替完整 Step 墙钟时间。GPU 指标是 PyTorch allocated 峰值，非 nvidia-smi 总占用；RSS 含映射文件驻留，非整机提交内存。

**内核启动/同步次数仍未完成 B256 profiler 复测。** 6 GiB 主机内存门槛未满足；尝试较小 B16 profiler 时也被 4 GiB 前置门禁延期，未创建该诊断 CUDA 工作。没有用旧版/B16 次数冒充本版 B256 结果。此项属于分析完整性的待办，不影响已经实际完成的 GPU 数值验收与普通完整 Step 提速证据。正式训练恢复后不再叠加 GPU 测试。

记录：`results/speedup_20261031/reference_v2_e2e.json` 与 `optimized_v2_e2e.json`；汇总为 `results/gpu_acceptance_summary.json`。

## 实际采用的优化与放弃项

- `model/execution.py`、`graph/builder.py`：CPU 预计算 48 种关系边索引、目标节点和度归一化，传输一次供四层复用；紧凑执行缓冲的分片起点对齐。填充只存在于索引存储，实际运算切片排除填充，图和消息数未变。缓存随结构/Mask/域张量版本变化失效。
- `model/layers.py`、`network.py`：传入缓存计划；池化缓存每图成员索引，保留原每图 Softmax、求和顺序，避免重复 CUDA nonzero。四层、隐藏维度、48 套独立权重全部保留。
- `train/losses.py`：CPU 确定监督组有效位置，复用扁平索引；保持原逐组平方误差/分母/等组权重与规约顺序，将多次有限性检查集中成一次同步。缺失槽位与原始目标语义不变。
- `train/loop.py`：Coverage 使用 CPU 已知 Mask/域；LossAccumulator 批量转回 detached 统计，保持行/组累加次序。
- `model/heads.py`：本次采用原逐样本输出头计算。此前激进分组 GEMM/分桶池化版改变浮点求和路径，一步 AdamW 对 `pool.score.2.bias` 的绝对误差约 6.58e-5 超过既有容差，因此没有发布。第一版保守实现虽通过数值检查，测速 1.57s 对原版 1.54s 无收益，也未发布；最终版本在消除多余计划准备、缓存 Loss 索引后重新验收并测速通过。

未启用混合精度、降低维度/层数、修改图、监督契约、样本划分、标签变换、Loss 定义、学习率或早停规则。

## 正式迁移与兼容

新代码：`versions/rgcn_cached_fast_v1`。新配置：`config/experiment1_fusion_id_fast_v1.json`。新结果：`artifacts/experiment1_fusion_id_fast_v1`。新监控：`monitoring/experiment1_fusion_id_fast_v1`。

迁移是显式生成新 recipe 与派生检查点，**没有绕过旧 recipe 保护跨版本直接 --resume**。新 recipe 包含全部 52 个 src Python 文件及新入口、包括 `execution.py`，以及配置、缓存 manifest、划分与不可变迁移来源记录的哈希。新 recipe digest：`{receipt['new_recipe_digest']}`。

迁移前后对模型、优化器、scheduler、CPU/CUDA RNG、history、coverage、早停计数、游标和 partial 各项独立散列，全部相同；仅派生 recipe_digest 改变。原断点是 zero-based next_epoch=15、next_batch=1024、partial.samples=262144。15 个已完成轮次保留，第 16 轮前 1024 批来自原版，后续来自缓存版，因此该轮属于混合执行历史，不能作为新版完整整轮速度。新状态中的 `active_segment_samples_per_second` 单独记录恢复后的吞吐。

state_dict 键、形状、参数顺序与优化器分组兼容。后续 ID、LOTO、消融可复用这套模型接口，但应建立各自新 recipe/输出，保持各自原定数据划分与监督契约；兼容不表示所有泛化协议已训练完成或能接受未知域/未训练输出头。

当前训练 PID {status['pid']}、创建时间 {identity.create_time()}，状态 {status['state']}，第 {status.get('epoch')} 轮、第 {status.get('batch')}/1563 批。新监控身份核对与指纹正常。原版 STOP 和禁用恢复标记保留，防止旧监控同时启动原训练。新监控继承用户授权的有界恢复规则：进程退出且指纹不变、RAM≥3 GiB 稳定 120 秒、磁盘≥10 GiB、GPU 空闲≥2048 MiB，经 CPU 检查点验证才可恢复；数值/完整性错误不自动重试。

已经运行，无需再次启动。只有确认当前进程结束且资源满足时，才在 PowerShell 续训同一新版：

```powershell
Set-Location D:\\EDA
.\\.venv-gpu\\Scripts\\python.exe -B -u versions/rgcn_cached_fast_v1/scripts/train_experiment1_fast.py --config config/experiment1_fusion_id_fast_v1.json --resume
```

新建其他实验时使用新的 config/output，不能拿旧原版目录做 --resume。提速不保证立即提高 R²；原有少数近常数标签的极负 R² 警报继续保留，未隐藏指标或更改目标尺度。
'''
    report_path=ROOT/'reports/rgcn_gpu_acceptance_report.md'
    archive=CANDIDATE/'results/previous_gpu_acceptance_deferred.md'
    if not archive.exists():archive.write_bytes(report_path.read_bytes())
    report_path.write_text(report,encoding='utf-8')
    (ROOT/'reports/experiment1_speedup_migration.md').write_text(f'''# 实验一提速迁移

GPU 数值验收通过，真实 B256 完整 Step：原版 {a['seconds_mean']:.3f}s，新版 {b['seconds_mean']:.3f}s，加速 {speed:.3f}×。完整证据见 [GPU 验收报告](rgcn_gpu_acceptance_report.md)。

已从第 16 轮已完成 1024 批继续训练，保留模型/AdamW/scheduler/RNG/早停/历史，不重新开始。原源码、检查点与结果保留。

- 最新训练状态：[status.json](../artifacts/experiment1_fusion_id_fast_v1/status.json)
- 最新监控：[status.json](../monitoring/experiment1_fusion_id_fast_v1/status.json)
- 新历史：[history.json](../artifacts/experiment1_fusion_id_fast_v1/history.json)
- 状态迁移核对：[migration_receipt.json](../artifacts/experiment1_fusion_id_fast_v1/migration_receipt.json)

旧目录 `experiment1_fusion_id_v1` 保持暂停，是保留的原版记录。不要手动同时恢复旧版。第 16 轮包含两版计算时间，观察新版的 `active_segment_samples_per_second`；后续完整轮次才适合比较整轮耗时。
''',encoding='utf-8')
    print(json.dumps({'speedup':speed,'formal_state':status,'original_files_unchanged':old_unchanged},indent=2))

if __name__=='__main__':main()
