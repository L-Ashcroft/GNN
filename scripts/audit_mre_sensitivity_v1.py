"""Read-only audit of frozen predictions and LOADBench source provenance.

No fitting, relabeling, test selection, downloads, or training-process control.
Optional AC replay reuses bounded diagnostic functions without their main/pause.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
import argparse, ast, csv, gzip, hashlib, importlib.util, json, math, sys
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np
import pyarrow.parquet as pq
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.falcon_reference_metrics import relative_errors
OUT=ROOT/'artifacts/falcon_reference_sensitivity_v1'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main(replay=False):
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    OUT.mkdir(parents=True,exist_ok=True)
    samples_path=ROOT/'artifacts/falcon_reference_eval_v1/public_id_published/per_sample_errors.parquet'
    samples=pq.read_table(samples_path).to_pylist()
    assert len(samples)==49056 and len({r['sample_id'] for r in samples})==49056
    assert all(r['sample_mre_pct'] is not None and np.isfinite(r['sample_mre_pct']) for r in samples)
    ordered=sorted(samples,key=lambda r:(-r['sample_mre_pct'],r['sample_id']))
    values=[r['sample_mre_pct'] for r in ordered]; total=math.fsum(values); n=len(values)
    cohorts={'top1':1,'top5':5,'top1percent':math.ceil(.01*n),'top5percent':math.ceil(.05*n)}
    members={k:{r['sample_id'] for r in ordered[:count]} for k,count in cohorts.items()}
    contributions={k:{'sample_count':count,'count_rule':'ceil(fraction*N) for percentage cohorts',
                      'sum_sample_errors_pct':math.fsum(values[:count]),
                      'contribution_pct':100*math.fsum(values[:count])/total} for k,count in cohorts.items()}
    summary_path=samples_path.with_name('summary.json')
    assert math.isclose(total/n,read(summary_path)['overall']['untrimmed_sample_mre']['mean_pct'],rel_tol=1e-14)
    lb=[r['sample_mre_pct'] for r in ordered if r['source_dataset']=='LOADBench']
    result={'analysis':'descriptive sensitivity on fixed test; no new model improvement',
            'created_at':datetime.now(timezone.utc).isoformat(),'full_samples':n,
            'full_sample_mre_pct':total/n,'full_sample_error_sum_pct':total,
            'without_single_largest':{'samples':n-1,'mean_pct':math.fsum(values[1:])/(n-1),
                                     'excluded_for_diagnostic_only':ordered[0]['sample_id']},
            'LOADBench_without_single_largest':{'samples':len(lb)-1,'mean_pct':math.fsum(lb[1:])/(len(lb)-1)},
            'tail_cohorts':contributions,'top5_samples':ordered[:5],
            'no_mutation_to_original_mask_labels_predictions_splits':True}
    config=read(ROOT/'config/falcon_reference_eval_v1.json')
    job=next(j for j in config['jobs'] if j['id']=='public_id_published')
    tasks=read(ROOT/job['tasks']); mapping=read(ROOT/'data/fusion_v2/performance_mapping.json')['groups']
    by_top=defaultdict(list)
    for group,t in tasks.items():
        if t.get('contract',{}).get('topology_id'):by_top[t['contract']['topology_id']].append(group)
    metric_contrib={k:defaultdict(float) for k in cohorts}
    details={}; recomputed={}; pins={str(samples_path.relative_to(ROOT)).replace('\\','/'):sha(samples_path)}
    # Compare every sample with the actual predictions; masks and source-unit
    # restoration mirror the audited evaluator, without changing its code.
    for spec in job['inputs']:
        for p in sorted(ROOT.glob(spec['glob'])):
            pins[p.relative_to(ROOT).as_posix()]=sha(p)
            for batch in pq.ParquetFile(p).iter_batches(batch_size=256):
                routed=defaultdict(list)
                for row in batch.to_pylist():
                    top=row.get('topology_id') or p.stem
                    group=row.get('supervision_group')
                    if not group:
                        assert len(by_top[top])==1
                        group=by_top[top][0]
                    routed[(group,top)].append(row)
                for (group,top),rows in routed.items():
                    t=tasks[group]; source=t['contract'].get('source_dataset') or t['contract'].get('source')
                    factors=np.ones(16); rules=t['slots']; native={}
                    for j,rule in rules.items():
                        candidates=[v for v in mapping.get(top,[]) if v['original_name']==rule['original_name'] and v['unit']==rule['unit']]
                        native[j]=rule['unit']
                        if candidates:
                            choices={(v['source_unit'],float(v['factor'])) for v in candidates}; assert len(choices)==1
                            native[j],factor=next(iter(choices)); factors[int(j)]=1/factor
                    yp=np.asarray([r['prediction'] for r in rows],dtype=np.float64)*factors
                    yt=np.asarray([r['target'] for r in rows],dtype=np.float64)*factors
                    mask=np.asarray([r['mask'] for r in rows],dtype=bool)
                    rel,valid,means=relative_errors(yp,yt,mask); counts=valid.sum(1)
                    for i,row in enumerate(rows):
                        sid=row['sample_id']; assert sid not in recomputed
                        recomputed[sid]=float(means[i]*100)
                        memberships=[k for k,ids in members.items() if sid in ids]
                        if not memberships:continue
                        scalars=[]
                        for j in np.flatnonzero(valid[i]):
                            contribution=float(rel[i,j]*100/counts[i])
                            key=(source,rules[str(j)]['original_name'],native[str(j)])
                            for cohort in memberships:metric_contrib[cohort][key]+=contribution
                            if sid in members['top5']:
                                scalars.append({'source':source,'metric':key[1],'native_unit':key[2],
                                                'slot':int(j),'target':float(yt[i,j]),'prediction':float(yp[i,j]),
                                                'absolute_error_native':float(abs(yp[i,j]-yt[i,j])),
                                                'relative_error_pct':float(100*rel[i,j]),
                                                'sample_weighted_error_pct':contribution,
                                                'native_multiplier':float(factors[j]),'mask':bool(mask[i,j])})
                        if scalars:
                            details[sid]={'file':p.relative_to(ROOT).as_posix(),'mask':row['mask'],
                                          'valid_count':int(counts[i]),'sample_mre_pct':recomputed[sid],
                                          'scalars':sorted(scalars,key=lambda r:-r['sample_weighted_error_pct'])}
    assert len(recomputed)==n
    errors=[abs(recomputed[r['sample_id']]-r['sample_mre_pct']) for r in samples]
    assert max(errors)==0.,max(errors)
    for cohort,rows in metric_contrib.items():
        ranked=sorted(rows.items(),key=lambda x:-x[1])
        target=contributions[cohort]['sum_sample_errors_pct']
        assert math.isclose(math.fsum(rows.values()),target,rel_tol=1e-12)
        contributions[cohort]['metric_attribution']=[{'source':key[0],'metric':key[1],'unit':key[2],
                                                   'sum_sample_weighted_error_pct':v,
                                                   'within_cohort_contribution_pct':100*v/target,
                                                   'global_contribution_pct':100*v/total} for key,v in ranked]
    worst=ordered[0]; trace=details[worst['sample_id']]
    top=worst['topology_id']; rawdir=ROOT/'data/raw/LOADBench/subset'/top.removeprefix('LOADBench__')
    rawfile=rawdir/'intermediate_prefix.csv.gz'; rawrow=int(worst['sample_id'].rsplit(':',1)[1])
    with gzip.open(rawfile,'rt',encoding='utf-8-sig',newline='') as f:
        original=next(row for i,row in enumerate(csv.DictReader(f),1) if i==rawrow)
    part=ROOT/'data/fusion_v2/sampled'/(top+'.parquet')
    record=pq.read_table(part,filters=[('sample_id','=',worst['sample_id'])]).to_pylist()
    assert len(record)==1; record=record[0]
    assert record['source_row']==rawrow and record['source_sha256']==sha(rawfile)
    assert record['mask']==trace['mask']
    assert all(record['mask'][i] for i in range(14)) and not any(record['mask'][14:])
    for scalar in trace['scalars']:
        j=scalar['slot']; name=scalar['metric']; raw_value=float(original[name]); stored=float(record['performance'][j])
        assert math.isclose(raw_value,stored,rel_tol=1e-14,abs_tol=0)
        assert float(np.float32(stored))==scalar['target']
        assert scalar['native_multiplier']==1.
    acquisition=read(rawdir/'acquisition.json')
    h=hashlib.sha256()
    with gzip.open(rawfile,'rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    assert h.hexdigest()==acquisition['prefix_csv_sha256']
    result['worst_record_trace']={'sample_id':worst['sample_id'],'raw_csv':rawfile.relative_to(ROOT).as_posix(),
        'one_based_data_row':rawrow,'generation':original['generation'],'point_index':original['point_index'],
        'source_feasible':original['feasible'],'raw_ft_Hz':float(original['ft']),
        'frozen_target_ft_Hz':trace['scalars'][0]['target'],'predicted_ft_Hz':trace['scalars'][0]['prediction'],
        'float32_cache_rounding_verified':True,'source_hash_and_downloaded_prefix_hash_verified':True,
        'mask_true_slots':list(range(14)),'absent_slots':[14,15],'ft_native_multiplier':1.,
        'raw_row':original,'prediction_trace':trace}
    for p in [summary_path,ROOT/job['tasks'],ROOT/'data/fusion_v2/performance_mapping.json',
              ROOT/'config/falcon_reference_eval_v1.json',ROOT/'src/falcon_reference_metrics.py',
              ROOT/'scripts/evaluate_falcon_reference.py',rawfile,rawdir/'CombinedTestBench.cir',
              rawdir/'standalone_tb_extract.py',rawdir/'acquisition.json',part,Path(__file__),
              ROOT/'third_party/FALCON/utils/compute_metrics.py']:
        pins[p.relative_to(ROOT).as_posix()]=sha(p)
    # AST-extracted official metric function: all real rows were independently
    # recomputed above; the existing eight golden tests exercise the function.
    result['all_49056_samples_recomputed_max_abs_error_pct']=max(errors)
    save(OUT/'sensitivity.json',result)
    save(OUT/'top5_trace.json',details)
    if replay:
        module=load_module('bounded_ac_replay',ROOT/'worktrees/loadbench_physical_v1/replay.py')
        module.BASE=OUT/'ac_diagnostics'
        models=ROOT/'data/audit/loadbench_physical_v1/models/reduced_sky130_combined/sky130.lib.spice'
        assert models.exists()
        cases=[('worst_22567',module.source_row(rawdir,'intermediate_prefix.csv.gz',rawrow)),
               ('normal_same_topology',module.source_row(rawdir,'intermediate_prefix.csv.gz',normal=True))]
        replay_results=[]
        for label,source in cases:
            if psutil.virtual_memory().available<1.1*2**30:
                replay_results.append({'label':label,'state':'not_run_memory_guard'});break
            replay_results.append(module.run_case(label,source,models))
        result['AC_replay']={'never_called_pause_or_resume':True,'new_training_labels_created':False,
                             'results':replay_results}
        save(OUT/'sensitivity.json',result)
    assert all(sha(ROOT/p)==v for p,v in pins.items())
    save(OUT/'receipt.json',{'input_sha256':pins,'all_pinned_inputs_unchanged':True,
                            'max_rescoring_difference_pct':max(errors),
                            'no_training_process_control':True,'RSS_MiB':psutil.Process().memory_info().rss/2**20})
    rows=['# MRE敏感性分析','','全部统计使用既有冻结测试预测，不重新选模，不修改原Mask、标签或主结果。',
          f'完整公开ID：N={n:,}，MRE={total/n:.10f}%。',
          f'仅在描述性分析中移除最大单条：N={n-1:,}，MRE={result["without_single_largest"]["mean_pct"]:.10f}%。',
          f'LOADBench对应移除单条后的描述性MRE={result["LOADBench_without_single_largest"]["mean_pct"]:.10f}%。','',
          '| 高误差记录 | 条数 | 对原完整相对误差和的贡献 | 主要指标（各队列内加权份额） |',
          '| --- | ---: | ---: | --- |']
    for k,c in contributions.items():
        labels='；'.join(f'{r["source"]} {r["metric"]} {r["within_cohort_contribution_pct"]:.6f}%' for r in c['metric_attribution'][:3])
        rows.append(f'| {k} | {c["sample_count"]} | {c["contribution_pct"]:.9f}% | {labels} |')
    rows += ['', '百分位队列取ceil(pN)条，按样本MRE降序、同分按sample_id排序。贡献分母为49,056条样本误差之和；指标归因为每个有效指标相对误差先除以该样本有效指标数后累加。它不是未加权的逐标量误差和。',
             '', '最大记录来自原始CSV第22,567条数据行（不含表头），generation=11、point_index=38；ft单位为Hz、换算因子为1，14个标签Mask为真。原始0.008057经模型缓存float32舍入为0.008057000115513802，已逐项核对。',
             '', '未将敏感性诊断视作模型改进，也未据此删除记录。原始历史AC波形与完整历史仿真环境未随CSV提供；本机回放即使吻合，也不等于逐位复现原始波形。详见sensitivity.json和AC回放产物。']
    if result.get('AC_replay'):
        rows += ['', '## 本机诊断回放','']
        for r in result['AC_replay']['results']:
            a=r.get('analysis',{})
            rows.append(f'- {r["label"]}：{r["state"]}；无窗口内交越={a.get("no_measured_unity_crossing")}；增益范围={a.get("gain_min_dB")}至{a.get("gain_max_dB")} dB；提取={a.get("official_function_on_replayed_waveform")}。')
    (ROOT/'reports/mre_sensitivity_v1.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('full_samples','full_sample_mre_pct','without_single_largest','tail_cohorts','all_49056_samples_recomputed_max_abs_error_pct')},ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');args=parser.parse_args();main(args.replay)
