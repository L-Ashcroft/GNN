"""Compare four and eight independent ngspice processes on all 18 topologies."""
from datetime import datetime,timezone
from pathlib import Path
import sys
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.benchmark_analoggym import run_phase
from src.analoggym_campaign import read,sha,atomic_json,parameter_plan


if __name__=='__main__':
    output=ROOT/'data/generated/analoggym_eight_worker_benchmark'
    recipe=read(ROOT/'data/generated/analoggym_campaign_v1/recipe.json')
    env=recipe['environment']
    for path,h in recipe['code_sha256'].items(): assert sha(ROOT/path)==h
    assert sha(ROOT/env['ngspice_executable'])==recipe['executable_sha256']
    assert sha(ROOT/env['pdk_manifest'])==recipe['model_manifest_sha256']
    for path,h in read(ROOT/env['pdk_manifest'])['files_sha256'].items(): assert sha(ROOT/path)==h
    qualified=read(ROOT/'data/generated/analoggym_benchmark_20261027/parameter_probe/summary.json')['results']
    jobs=[]
    for e in recipe['entries']:
        assert sha(ROOT/e['dut'])==e['dut_sha256']
        good=sorted((r for r in qualified if r['topology']==e['name'] and r['accepted']),key=lambda r:r['candidate'])
        assert good, 'No measured valid design: '+e['name']
        r=good[0]
        assert parameter_plan(e,r['candidate'])==r['parameters_native']
        jobs.append((e,r['candidate']))
    assert len(jobs)==18
    atomic_json(output/'plan.json',{'created_at':datetime.now(timezone.utc).isoformat(),
                'jobs':[(e['name'],c) for e,c in jobs],'comparison':[4,8],
                'RAM_available_bytes_before':psutil.virtual_memory().available,
                'scope':'Identical previously measured designs, fresh paired simulations; repeats excluded from formal dataset'})
    phases=[run_phase(output/('workers_'+str(w)),jobs,w,env) for w in (4,8)]
    reference={(r['topology'],r['candidate']):r for r in phases[0]['results']}
    for r in phases[1]['results']:
        previous=reference[(r['topology'],r['candidate'])]
        assert r['parameters_native']==previous['parameters_native']
        assert r['accepted']==previous['accepted']
        assert r['accepted'], 'Previously qualified design failed in concurrency test'
        np.testing.assert_allclose(r['performance'],previous['performance'],rtol=1e-7,atol=1e-12)
    speed=phases[0]['wall_seconds']/phases[1]['wall_seconds']
    baseline=read(ROOT/'reports/analoggym_20000_benchmark.json')['balanced_estimate_seconds_at_probe_workers']
    report={'timestamp_UTC':datetime.now(timezone.utc).isoformat(),'topologies':18,
            'identical_parameters_and_consistent_targets':True,
            'phases':[{k:v for k,v in p.items() if k!='results'} for p in phases],
            'speedup_8_vs_4':speed,'extrapolated_20000_seconds_at_8_workers':baseline/speed,
            'scope':'One valid design per topology; timing affected by background load and uneven final batch. Scaling a prior 20k estimate is provisional, not a new failure-rate measurement.',
            'recommendation':'8' if speed>1.1 else '4 (8 showed no material advantage in this trial)'}
    atomic_json(output/'comparison.json',report)
    atomic_json(ROOT/'reports/analoggym_eight_workers.json',report)
    lines=['# AnalogGym：4 与 8 并发实测','',f'更新时间：{report["timestamp_UTC"]}。', '',
           '相同18个拓扑、相同器件参数，分别重跑DC/AC/瞬态。参数选自此前真实验证的有效设计；全部标签重新提取核对。测速复算不计为新增独立样本。', '',
           '| 并发 ngspice 进程 | 仿真数 | 总秒数 | 接受数 | 完成时采样的最少可用内存 GiB |',
           '| ---: | ---: | ---: | ---: | ---: |']
    for p in phases:
        lines.append(f'| {p["workers"]} | {p["attempts"]} | {p["wall_seconds"]:.1f} | {p["accepted"]} | {p["min_available_RAM_bytes_at_completions"]/2**30:.2f} |')
    lines +=['',f'8并发相对4并发加速 **{speed:.2f}x**。两组参数和标签一致性检查通过。', '',
             f'将此前按拓扑均匀、包含失败重试的2万条估算按本次加速比缩放，约 {baseline/speed/3600:.1f} 小时；这个外推仍受失败率、电脑负载、温度/功耗限制及末批任务不齐影响，不保证翻倍加速。', '',
             '这里的并发是Python线程调度多个独立ngspice进程，各进程内部仍限制为单线程。可用内存只在任务完成时采样，不代表连续测得的峰值内存。', '',
             '正式数据生成任务未启动；未修改第三方仓库、物理测试条件或已有历史数据。']
    (ROOT/'reports/analoggym_eight_workers.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({k:report[k] for k in ('speedup_8_vs_4','extrapolated_20000_seconds_at_8_workers','recommendation')},flush=True)
