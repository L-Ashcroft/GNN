"""Bounded 8/12/16/20-worker ngspice scaling experiment, with an 8-worker repeat.

Only measured working parameter vectors are rerun. These repeated simulations
measure scheduling/resource costs and never augment the supervised dataset.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import sys
import time
import zipfile

import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.analoggym_campaign import inside,read,sha,atomic_json,simulate,parameter_plan,extract


def now(): return datetime.now(timezone.utc).isoformat()


def resources():
    children=[]
    for p in psutil.Process().children(recursive=True):
        try:
            if p.name().lower()=='ngspice_con.exe': children.append(p.memory_info().rss)
        except (psutil.NoSuchProcess,psutil.AccessDenied): pass
    return {'time_UTC':now(),'available_RAM_bytes':psutil.virtual_memory().available,
            'system_CPU_percent':psutil.cpu_percent(None),'active_ngspice':len(children),
            'ngspice_RSS_bytes':sum(children)}


def phase(output,name,workers,jobs,environment):
    folder=inside(output/name)
    if (folder/'summary.json').exists(): return read(folder/'summary.json')
    if (folder/'raw').exists():
        folder.rename(folder.with_name(folder.name+'_interrupted_'+str(time.time_ns())))
    started=time.perf_counter(); done_results=[]; samples=[]; next_job=0; stopping=None
    # Only this process's descendants are observed; unrelated ngspice runs are
    # neither counted as workers nor killed. Memory guard drains active work.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending={}
        while pending or next_job<len(jobs):
            state=resources(); samples.append(state)
            if state['available_RAM_bytes']<768*2**20: stopping='stopped_low_available_RAM'
            if (output/'STOP').exists(): stopping='stopped_by_request'
            while not stopping and len(pending)<workers and next_job<len(jobs):
                e,c=jobs[next_job]; next_job+=1
                pending[pool.submit(simulate,e,c,folder,environment,120)]=(e,c)
            if not pending: break
            done,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
            for future in done:
                e,c=pending.pop(future); r=future.result()
                archive=ROOT/r['archive']
                with zipfile.ZipFile(archive) as z:
                    assert z.testzip() is None
                    for member,h in r['artifacts_sha256'].items(): assert hashlib.sha256(z.read(member)).hexdigest()==h
                    if r['accepted']:
                        measured=extract(e,{s:z.read(s) for s in z.namelist() if s.endswith('_wave.dat')})
                        np.testing.assert_array_equal(measured,r['performance'])
                done_results.append(r)
                print(now(),name,len(done_results),'/',len(jobs),e['name'],c,
                      'OK' if r['accepted'] else 'REJECT',round(r['elapsed_seconds'],2),flush=True)
                atomic_json(folder/'progress.json',{'workers':workers,'finished':len(done_results),
                            'planned':len(jobs),'stopping':stopping,'results':done_results,'latest_resources':state})
    wall=time.perf_counter()-started
    summary={'phase':name,'workers':workers,'planned':len(jobs),'attempts':len(done_results),
             'accepted':sum(r['accepted'] for r in done_results),'wall_seconds':wall,
             'state':stopping or 'complete','throughput_attempts_per_hour':len(done_results)/wall*3600,
             'min_available_RAM_bytes':min(s['available_RAM_bytes'] for s in samples),
             'peak_sampled_ngspice_RSS_bytes':max(s['ngspice_RSS_bytes'] for s in samples),
             'max_sampled_active_ngspice':max(s['active_ngspice'] for s in samples),
             'mean_sampled_system_CPU_percent':float(np.mean([s['system_CPU_percent'] for s in samples])),
             'resource_sampling_scope':'approximately once per second plus completions; sampled RSS, not exact peak private committed memory; whole-system CPU includes other applications',
             'results':done_results}
    atomic_json(folder/'resource_samples.json',samples)
    atomic_json(folder/'summary.json',summary)
    return summary


def report(output,phases,plan):
    complete=[p for p in phases if p['state']=='complete' and p['attempts']==p['planned']]
    baseline=complete[0]
    oracle={(r['topology'],r['candidate']):r for r in baseline['results']}
    assert len(oracle)==54
    for p in complete:
        assert p['accepted']==54, 'Qualified design failed at concurrency '+str(p['workers'])
        for r in p['results']:
            ref=oracle[(r['topology'],r['candidate'])]
            assert ref['parameters_native']==r['parameters_native']
            np.testing.assert_allclose(r['performance'],ref['performance'],rtol=1e-7,atol=1e-12)
    complete_by_workers={}
    for p in complete: complete_by_workers.setdefault(p['workers'],[]).append(p)
    times={w:float(np.mean([p['wall_seconds'] for p in items])) for w,items in complete_by_workers.items()}
    fastest=min(times,key=times.get)
    # Prefer the smaller concurrency when apparent gains are within a 5% band;
    # this is a practical rule, not a significance test on one timing repeat.
    recommended=min(w for w,t in times.items() if t<=times[fastest]*1.05)
    base20=read(ROOT/'reports/analoggym_20000_benchmark.json')['balanced_estimate_seconds_at_probe_workers']
    old8=read(ROOT/'reports/analoggym_eight_workers.json')['extrapolated_20000_seconds_at_8_workers']
    rows=[]
    for w,t in sorted(times.items()):
        group=complete_by_workers[w]
        rows.append({'workers':w,'repetitions':len(group),'mean_wall_seconds':t,
                     'attempts_each':54,'throughput_attempts_per_hour':54/t*3600,
                     'speedup_vs_8':times[8]/t,
                     'extrapolated_20000_hours':old8/(times[8]/t)/3600,
                     'min_available_RAM_GiB':min(p['min_available_RAM_bytes'] for p in group)/2**30,
                     'peak_sampled_ngspice_RSS_GiB':max(p['peak_sampled_ngspice_RSS_bytes'] for p in group)/2**30,
                     'max_sampled_active_ngspice':max(p['max_sampled_active_ngspice'] for p in group),
                     'mean_sampled_system_CPU_percent':float(np.mean([p['mean_sampled_system_CPU_percent'] for p in group]))})
    repeats=complete_by_workers.get(8,[])
    drift=(repeats[-1]['wall_seconds']/repeats[0]['wall_seconds']-1) if len(repeats)>1 else None
    result={'timestamp_UTC':now(),'plan':plan,'script_sha256':sha(Path(__file__)),
            'topologies':18,'designs_per_topology':3,'all_completed_targets_consistent':True,
            'phase_summaries':[{k:v for k,v in p.items() if k!='results'} for p in phases],
            'comparison':rows,'fastest_tested_workers':fastest,'recommended_tested_workers':recommended,
            'recommendation_rule':'Smallest tested concurrency within 5% of fastest observed mean time; engineering threshold, not statistical significance.',
            'end_vs_start_8_worker_time_change_fraction':drift,
            'earlier_4_worker_20000_estimate_hours':base20/3600,
            'scope':'Valid-design timing only. The 20k projections scale the prior retry-inclusive 8-worker estimate. No new failure-rate estimate, no exhaustive maximum-process claim, no production job launched.'}
    atomic_json(output/'comparison.json',result)
    atomic_json(ROOT/'reports/analoggym_concurrency_scaling.json',result)
    lines=['# AnalogGym：8/12/16/20 并发效率','',f'更新时间：{result["timestamp_UTC"]}。', '',
           '使用相同54个已验证设计，18拓扑各3个，实际重跑DC/AC/瞬态。测试顺序为8、12、16、20、8；最后的8并发用于观察负载漂移，表内8并发耗时取两次平均。复算不计入正式数据集。', '',
           '| 并发进程 | 重复轮数 | 每轮设计数 | 平均耗时秒 | 相对8并发加速 | 2万有效样本粗估小时 | 最少可用内存GiB | ngspice采样峰值RSS GiB |',
           '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f'| {r["workers"]} | {r["repetitions"]} | 54 | {r["mean_wall_seconds"]:.1f} | {r["speedup_vs_8"]:.2f}x | {r["extrapolated_20000_hours"]:.1f} | {r["min_available_RAM_GiB"]:.2f} | {r["peak_sampled_ngspice_RSS_GiB"]:.2f} |')
    lines+=['',f'本轮最快档位为 **{fastest} 并发**；按“与最快耗时相差不超过5%时选较低并发”的规则，建议 **{recommended} 并发**。这不是操作系统的硬性上限。', '',
            f'首尾8并发耗时变化为 {drift:+.1%}。' if drift is not None else '未完成第二次8并发复测，负载漂移尚未测得。', '',
            '已完成轮次的54个结果全部通过质量检查，各档位参数相同、标签在rtol=1e-7/atol=1e-12内一致；归档CRC、SHA与从原始波形重提标签均已核对。', '',
            '## 估计及资源边界','',
            '- 2万条时长以此前包含失败重试、均匀分拓扑的8并发估计为锚点，再按本轮加速比缩放，不是假定线程数与速度成正比。',
            '- 本轮每拓扑3个有效设计，长时间失败仿真、温度/功耗限制和其他应用负载仍可能改变并发收益；高并发档位仅测一次，微小差异不作为强结论。',
            '- 54个任务会有不满末批，尤其高并发时，末批损失可能低估大批量持续满载的效率。',
            '- 每秒左右采样内存与CPU；RSS不等于独占或提交内存，采样值不保证捕获瞬时峰值。只统计本测速进程的ngspice后代进程。',
            '- 可用内存低于768 MiB或出现STOP文件时停止提交新任务，等待在途仿真完成并保存，未完成档位不参与最快档位比较。',
            '- 器件参数、供电负载和求解设置不随并发数改变；仍使用已冻结的生产配方、工艺模型和仿真器。',
            '- Windows原生运行；上游仓库、现有图/标签及正式生成任务配置未修改，全量生成未启动。', '',
            '## 复现','',
            '资源监控依赖见requirements-benchmark.txt。新测速使用新输出目录；完整阶段可恢复读取，中断阶段保留证据后重测。', '',
            '```powershell',
            '.\\.venv\\Scripts\\python.exe scripts/benchmark_analoggym_scaling.py --output data/generated/analoggym_scaling_new',
            '```']
    (ROOT/'reports/analoggym_concurrency_scaling.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({'fastest_tested_workers':fastest,'recommended_tested_workers':recommended,'rows':rows,'drift':drift},flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='data/generated/analoggym_scaling_20261027')
    a=p.parse_args(); output=inside(ROOT/a.output)
    if platform.system()!='Windows': raise RuntimeError('Native Windows required')
    recipe=read(ROOT/'data/generated/analoggym_campaign_v1/recipe.json'); env=recipe['environment']
    for path,h in recipe['code_sha256'].items(): assert sha(ROOT/path)==h
    assert sha(ROOT/env['ngspice_executable'])==recipe['executable_sha256']
    assert sha(ROOT/env['pdk_manifest'])==recipe['model_manifest_sha256']
    for path,h in read(ROOT/env['pdk_manifest'])['files_sha256'].items(): assert sha(ROOT/path)==h
    # This exact frozen recipe was used by the earlier finite campaign.
    smoke=ROOT/'data/generated/analoggym_campaign_smoke_v5'
    assert read(smoke/'manifest.json')['recipe_hash']==read(ROOT/'data/generated/analoggym_campaign_v1/manifest.json')['recipe_hash']
    with sqlite3.connect(f'file:{(smoke/"ledger.sqlite").as_posix()}?mode=ro',uri=True) as db:
        candidates=[json.loads(x[0]) for x in db.execute('select result_json from attempts where accepted=1')]
    candidates+=read(ROOT/'data/generated/analoggym_benchmark_20261027/parameter_probe/summary.json')['results']
    jobs=[]
    for e in recipe['entries']:
        assert sha(ROOT/e['dut'])==e['dut_sha256']
        assert sha(ROOT/e['parameter_source'])==e['parameter_source_sha256']
        valid={r['candidate']:r for r in candidates if r['topology']==e['name'] and r['accepted'] and r['candidate']>0}
        selected=sorted(valid)[:3]
        assert len(selected)==3
        for c in selected:
            assert parameter_plan(e,c)==valid[c]['parameters_native']
            jobs.append((e,c))
    assert len(jobs)==54 and len({(e['name'],c) for e,c in jobs})==54
    # Round robin by topology so the last portion is not always just the LDOs.
    jobs=[jobs[i*3+j] for j in range(3) for i in range(18)]
    plan={'created_at':now(),'jobs':[(e['name'],c) for e,c in jobs],
          'phases':[8,12,16,20,8],'CPU_logical':psutil.cpu_count(),
          'RAM_total_bytes':psutil.virtual_memory().total,'RAM_available_bytes_before':psutil.virtual_memory().available,
          'recipe_file_sha256':sha(ROOT/'data/generated/analoggym_campaign_v1/recipe.json'),
          'minimum_available_RAM_bytes':768*2**20,'OMP_NUM_THREADS_per_process':1}
    if (output/'plan.json').exists():
        previous=read(output/'plan.json')
        assert previous['jobs']==plan['jobs'] and previous['recipe_file_sha256']==plan['recipe_file_sha256']
        plan=previous
    else: atomic_json(output/'plan.json',plan)
    psutil.cpu_percent(None)
    phases=[]
    for name,w in [('workers_8_start',8),('workers_12',12),('workers_16',16),('workers_20',20),('workers_8_end',8)]:
        result=phase(output,name,w,jobs,env); phases.append(result)
        if result['state']!='complete': break
    report(output,phases,plan)


if __name__=='__main__': main()
