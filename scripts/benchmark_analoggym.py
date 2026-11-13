"""Fresh native-Windows simulations to time balanced 20k-design generation.

Comparison reruns of identical designs measure concurrency, not extra samples.
The separate unselected parameter probe measures failures and topology costs.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
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
from src.analoggym_campaign import read, inside, sha, atomic_json, simulate, parameter_plan, extract


def now(): return datetime.now(timezone.utc).isoformat()


def run_phase(folder, jobs, workers, environment):
    if (folder/'summary.json').exists():
        return read(folder/'summary.json')
    # Cached individual ZIPs would invalidate wall-clock throughput measurements.
    # Preserve interrupted evidence; resume only a fully completed timing phase.
    if (folder/'raw').exists():
        folder.rename(folder.with_name(folder.name+'_interrupted_'+str(time.time_ns())))
    started=time.perf_counter(); results=[]
    resources=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending=[pool.submit(simulate,e,c,folder,environment,120) for e,c in jobs]
        for job in as_completed(pending):
            r=job.result()
            with zipfile.ZipFile(ROOT/r['archive']) as z:
                assert z.testzip() is None
                for member,expected in r['artifacts_sha256'].items():
                    import hashlib
                    assert hashlib.sha256(z.read(member)).hexdigest()==expected
                if r['accepted']:
                    entry=next(e for e,c in jobs if e['name']==r['topology'] and c==r['candidate'])
                    measured=extract(entry,{s:z.read(s) for s in z.namelist() if s.endswith('_wave.dat')})
                    np.testing.assert_array_equal(measured,r['performance'])
            results.append(r)
            vm=psutil.virtual_memory()
            resources.append({'available_RAM_bytes':vm.available,'system_CPU_percent':psutil.cpu_percent(None)})
            print(now(),folder.name,len(results),'/',len(jobs),r['topology'],r['candidate'],
                  'OK' if r['accepted'] else 'REJECT',round(r['elapsed_seconds'],2),r['failure'] or '',flush=True)
            atomic_json(folder/'progress.json',{'finished':len(results),'planned':len(jobs),'results':results})
    summary={'workers':workers,'attempts':len(results),'accepted':sum(r['accepted'] for r in results),
             'wall_seconds':time.perf_counter()-started,
             'raw_bytes':sum((ROOT/r['archive']).stat().st_size for r in results),
             'min_available_RAM_bytes_at_completions':min(r['available_RAM_bytes'] for r in resources),
             'resources_scope':'sampled at completions; not peak RSS or continuous CPU profiling',
             'results':results}
    atomic_json(folder/'summary.json',summary)
    return summary


def finish(output,recipe,phases):
    # Repeat serial after model files are warm, rather than attributing the first
    # cold-cache / changing desktop-load delay to parallel speedup.
    comparison=[phases['concurrency_1_warm'],phases['concurrency_2'],phases['concurrency_4']]
    reference={(r['topology'],r['candidate']):r for r in comparison[0]['results']}
    max_difference=0.
    for phase in comparison[1:]:
        for r in phase['results']:
            old=reference[(r['topology'],r['candidate'])]
            assert r['parameters_native']==old['parameters_native']
            assert r['accepted']==old['accepted'], 'Concurrency changed simulation acceptance'
            if r['accepted']:
                a=np.asarray(old['performance']); b=np.asarray(r['performance'])
                np.testing.assert_allclose(a,b,rtol=1e-7,atol=1e-12)
                max_difference=max(max_difference,float(np.max(np.abs(a-b))))
    serial=comparison[0]['wall_seconds']
    for s in comparison:
        s['speedup_vs_serial']=serial/s['wall_seconds']
    recommended=max(comparison,key=lambda s:s['attempts']/s['wall_seconds'])['workers']
    probe=phases['parameter_probe']; grouped={}
    smoke=ROOT/'data/generated/analoggym_campaign_smoke_v5'
    with sqlite3.connect(f'file:{(smoke/"ledger.sqlite").as_posix()}?mode=ro',uri=True) as db:
        historic=[json.loads(x[0]) for x in db.execute('select result_json from attempts')]
    for e in recipe['entries']:
        name=e['name']; fresh=[r for r in probe['results'] if r['topology']==name]
        old=[r for r in historic if r['topology']==name]
        ok=sum(r['accepted'] for r in fresh); n=len(fresh)
        old_ok=sum(r['accepted'] for r in old)
        # Baseline candidate zero is a previously qualified design: omit it when
        # using historical successes to estimate perturbed-design acceptance.
        perturbed=[r for r in old if r['candidate']!=0]+fresh
        p=sum(r['accepted'] for r in perturbed)/len(perturbed)
        mean=float(np.mean([r['elapsed_seconds'] for r in fresh]))
        grouped[name]={'fresh_attempts':n,'fresh_accepted':ok,'fresh_mean_seconds':mean,
                       'fresh_success_rate':ok/n,'historical_attempts':len(old),'historical_accepted':old_ok,
                       'combined_perturbed_attempts':len(perturbed),'combined_perturbed_success_rate':p}
    # Balanced quotas need per-topology costs/rates, not only pooled throughput.
    quota={e['name']:20000//len(grouped)+(i<20000%len(grouped)) for i,e in enumerate(recipe['entries'])}
    worker_saturation=probe['wall_seconds']/sum(r['elapsed_seconds'] for r in probe['results'])
    unresolved=[name for name,g in grouped.items() if not g['combined_perturbed_success_rate']]
    estimate=None if unresolved else sum(quota[name]*g['fresh_mean_seconds']/g['combined_perturbed_success_rate']
                                      for name,g in grouped.items())*worker_saturation
    combined_ok=sum(r['accepted'] for r in historic+probe['results'])
    storage=(sum((ROOT/r['archive']).stat().st_size for r in historic)+probe['raw_bytes'])/combined_ok*20000
    result={'date_UTC':now(),'target_valid_unique_designs':20000,'topologies':len(grouped),
            'benchmark_script_sha256':sha(Path(__file__)),
            'environment':read(output/'environment.json'),
            'new_timing_and_probe_simulations':sum(s['attempts'] for s in phases.values()),
            'initial_serial_wall_seconds':phases['concurrency_1']['wall_seconds'],
            'serial_comparison_basis':'Warm repeat; initial serial retained separately because caches/desktop load confound timing',
            'comparison_identical_parameters_and_consistent_labels':True,
            'comparison_max_absolute_label_difference_mixed_units':max_difference,
            'quotas':quota,'comparison':[{k:v for k,v in s.items() if k!='results'} for s in comparison],
            'recommended_workers_from_small_comparison':recommended,'probe_workers':probe['workers'],
            'fresh_parameter_probe':{k:v for k,v in probe.items() if k!='results'},
            'topology_results':grouped,'balanced_estimate_seconds_at_probe_workers':estimate,
            'unresolved_topologies':unresolved,'estimated_raw_bytes_for_20000':storage,
            'estimate_method':'Per-topology fresh mean attempt runtime / observed perturbed success rate, balanced quotas; empirical probe wall/sum(runtime) factor. Historical plus fresh outcomes for sparse rates, excluding qualified baseline. No confidence interval claimed.',
            'cautions':['Small samples; candidate ordering and cold/warm caches affect timing.',
                        'Hardware load, failing-transient timeouts and adaptive retry workload may change estimates.',
                        'Comparison repeats and diagnostic runs are not added to the formal dataset.',
                        'The 20k production job is not launched by this benchmark.'],
            'failure_counts':dict(Counter(r['failure'] for r in probe['results'] if not r['accepted']))}
    atomic_json(output/'benchmark.json',result)
    atomic_json(ROOT/'reports/analoggym_20000_benchmark.json',result)
    lines=['# AnalogGym 20,000 条有效样本：本机仿真测速','',
           f'更新时间：{result["date_UTC"]}。Windows 原生 ngspice，多进程独立仿真。', '',
           f'本机：{result["environment"].get("CPU_name", "CPU见environment.json")}；逻辑CPU {result["environment"]["logical_CPUs"]}，物理内存 {result["environment"]["RAM_bytes"]/2**30:.1f} GiB。', '',
           f'本次合计 {result["new_timing_and_probe_simulations"]} 次真实仿真：并发比较含同设计复算，不计作不同新样本。', '',
           '## 实测并行能力','', '| 并发进程 | 同组设计数 | 耗时（秒） | 相对串行加速 |', '| ---: | ---: | ---: | ---: |']
    for s in comparison:
        lines.append(f'| {s["workers"]} | {s["attempts"]} | {s["wall_seconds"]:.1f} | {s["speedup_vs_serial"]:.2f}x |')
    lines +=['',f'表中串行使用预热后的复测；首次串行为 {phases["concurrency_1"]["wall_seconds"]:.1f} 秒，单独保留，避免把缓存差异全部算成并行加速。各模式参数完全相同，接受状态及性能标签一致性已核对。']
    lines+=['',f'未预筛参数试跑：{len(grouped)} 个拓扑，共 {probe["attempts"]} 次，接受 {probe["accepted"]} 次，耗时 {probe["wall_seconds"]:.1f} 秒。每条包括实际 DC/AC/瞬态及原始波形归档；接受标签从归档波形重新提取核对。', '',
            '## 2 万条均匀样本的估计','',
            f'每拓扑 {20000//len(grouped)} 或 {20000//len(grouped)+1} 条，总计 20,000 条不同有效设计。']
    if estimate:
        lines.append(f'按逐拓扑成本及扰动参数通过率估算，{probe["workers"]} 并发约 **{estimate/3600:.1f} 小时（{estimate/86400:.2f} 天）**；粗略预留 ±40% 的工程余量为 {estimate*.6/3600:.1f}–{estimate*1.4/3600:.1f} 小时。这是小样本外推，不是统计置信区间；低通过率拓扑可能更久。')
    else:
        lines.append('部分拓扑的扰动点仍没有成功记录，无法可靠给出全覆盖 2 万条所需时长：'+', '.join(unresolved))
    lines +=[f'原始归档按现有失败点一并保留估计约 {storage/2**30:.1f} GiB，建议另外预留输出和失败长波形的空间。', '',
             '## 各拓扑试跑','', '| 拓扑 | 新试跑接受/尝试 | 新试跑平均秒/次 | 历史+新扰动点通过率 | 正式配额 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name,g in grouped.items():
        lines.append(f'| {name} | {g["fresh_accepted"]}/{g["fresh_attempts"]} | {g["fresh_mean_seconds"]:.2f} | {g["combined_perturbed_success_rate"]:.1%} | {quota[name]} |')
    lines +=['','## 参数、条件和估计边界','',
              '- 参数中心来自最终资格与精度验证通过的设计；保持别名、电流镜/差分对共享尺寸，种子42。',
              '- 20%窄扰动、60%中等扰动、20%较宽扰动的采样规则不变；偏置电压约±10%，补偿电容 x1/1.35..1.35，倍数离散取整，几何边界和测试平台见 campaign_method。',
              '- 电气稳定性须靠实际仿真判断；DC偏离、无法稳定、无有效交越或求解超时均隔离，不填假标签。',
              '- 18拓扑是16个放大器和2个LDO；本次不宣称覆盖五类。',
              '- 并发比较重跑相同设计，只用于测速；新参数探测每拓扑数量少，通过率存在很大不确定性。',
              '- 历史小批按每拓扑成功3条后停止，通过率有停止规则偏差；这里只作工程预算参考。',
              '- 所有模型、源代码和仿真器指纹已核验；上游仓库不修改。',
              '- 本次没有启动20,000条正式生成，也没有合并到历史FALCON/AICircuit数据。']
    lines +=['', '## 复现','',
              '附加资源监控依赖见 `requirements-benchmark.txt`。运行：', '',
              '```powershell',
              '.\\.venv\\Scripts\\python.exe scripts/benchmark_analoggym.py --output data/generated/analoggym_benchmark_new_run',
              '```', '',
              '输出路径必须位于当前工作区。重新测速使用新目录；已有完整阶段会复用保存的测速结果，不把缓存读取时间当作重新仿真的速度。中断且未完成的阶段会另存原始证据后重新测量。']
    (ROOT/'reports/analoggym_20000_benchmark.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('topologies','balanced_estimate_seconds_at_probe_workers','estimated_raw_bytes_for_20000','recommended_workers_from_small_comparison')},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='data/generated/analoggym_benchmark_20261027')
    a=p.parse_args(); output=inside(ROOT/a.output)
    if platform.system()!='Windows': raise RuntimeError('Native Windows required')
    recipe=read(ROOT/'data/generated/analoggym_campaign_v1/recipe.json'); env=recipe['environment']
    for name,expected in recipe['code_sha256'].items():
        assert sha(ROOT/name)==expected, 'Pinned code changed: '+name
    assert sha(ROOT/env['ngspice_executable'])==recipe['executable_sha256']
    assert sha(ROOT/env['pdk_manifest'])==recipe['model_manifest_sha256']
    for name,expected in read(ROOT/env['pdk_manifest'])['files_sha256'].items(): assert sha(ROOT/name)==expected
    by_name={e['name']:e for e in recipe['entries']}
    for e in by_name.values():
        assert sha(ROOT/e['dut'])==e['dut_sha256']
        assert sha(ROOT/e['parameter_source'])==e['parameter_source_sha256']
    names=['Fan_SMC_Pin_3','Leung_NMCF_Pin_3','Qu2017_AZC_Pin_3','Sau_CFCC_Pin_3','ldo_simple','ldo_folded_cascode']
    with sqlite3.connect(f'file:{(ROOT/"data/generated/analoggym_campaign_smoke_v5/ledger.sqlite").as_posix()}?mode=ro',uri=True) as db:
        jobs=[]
        for name in names:
            candidate,stored=db.execute('select candidate,result_json from attempts where topology=? and accepted=1 and candidate>0 order by candidate limit 1',(name,)).fetchone()
            assert parameter_plan(by_name[name],candidate)==json.loads(stored)['parameters_native']
            jobs.append((by_name[name],candidate))
    import winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
        cpu_name=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    existing=read(output/'environment.json') if (output/'environment.json').exists() else {}
    atomic_json(output/'environment.json',{'started_at':existing.get('started_at',now()),'CPU_name':cpu_name,'platform':platform.platform(),'logical_CPUs':os.cpu_count(),
                'physical_CPUs':psutil.cpu_count(logical=False),'RAM_bytes':psutil.virtual_memory().total,
                'available_RAM_bytes':psutil.virtual_memory().available,'production_recipe_hash':sha(ROOT/'data/generated/analoggym_campaign_v1/recipe.json'),
                'sampling_policy':'Same frozen qualified centers and campaign parameter_plan; fresh ngspice runs for all timing comparisons'})
    phases={}
    for w in (1,2,4): phases['concurrency_'+str(w)]=run_phase(output/('concurrency_'+str(w)),jobs,w,env)
    phases['concurrency_1_warm']=run_phase(output/'concurrency_1_warm',jobs,1,env)
    # Unselected new candidate IDs, three per topology, spread over the fixed
    # ten-index strata rather than artificially counting only qualified centers.
    fresh=[(e,10+(3*i+j)%10) for j in range(3) for i,e in enumerate(recipe['entries'])]
    phases['parameter_probe']=run_phase(output/'parameter_probe',fresh,4,env)
    finish(output,recipe,phases)


if __name__=='__main__': main()
