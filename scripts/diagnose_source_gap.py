"""Read-only LOADBench provenance/label/cache audit and frozen-error diagnosis.

No model inference, fitting, simulation, data filtering, or recipe changes.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
from collections import Counter,defaultdict
import csv,gzip,json,hashlib,statistics,time
import psutil
import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    started=time.perf_counter();catalog={s['topology_id']:s for s in read(ROOT/'data/fusion_v2/catalog.json')}
    cache=read(ROOT/'data/model_ready/fusion_v2_training_v1/manifest.json')
    sample_counts=Counter();nonempty=Counter();training_counts=Counter();support=defaultdict(list)
    for top,spec in cache['topologies'].items():
        source=top.split('__')[0];sample_counts[source]+=spec['rows'];nonempty[source]+=1
        a=np.load(ROOT/spec['folder']/'role.npy',mmap_mode='r',allow_pickle=False)
        # Role code0 is train in the pinned tensor cache.
        training_counts[source]+=int((a==0).sum());support[source].append(int((a==0).sum()));a._mmap.close()
    frozen=read(ROOT/'artifacts/experiment1_rgcn_local_head_v1/fixed_test/metrics.json')
    groups=defaultdict(list)
    for r in frozen['per_target']:
        source=r['semantic_id'].split('::')[0].split('__')[0]
        groups[source].append(r)
    summaries={};per_metric={}
    for source,rows in groups.items():
        r2=[r['r2'] for r in rows if r['r2'] is not None];hit=[r['hit5'] for r in rows if r['hit5'] is not None]
        neg=[r for r in rows if r['r2'] is not None and r['r2']<0]
        summaries[source]={'sampled_records':sample_counts[source],'topologies':nonempty[source],
            'mean_records_per_topology':sample_counts[source]/nonempty[source],
            'training_rows':training_counts[source],'min_train_rows_per_topology':min(support[source]),
            'median_train_rows_per_topology':statistics.median(support[source]),
            'contract_metric_pairs':len(rows),'relative_hit_targets':len(hit),'macro_hit5':statistics.mean(hit),
            'r2_defined':len(r2),'negative_r2_targets':len(neg),'macro_r2':statistics.mean(r2),'median_r2':statistics.median(r2),
            'test_targets_under_30_observations':sum(r['count']<30 for r in rows),
            'negative_targets_under_30_observations':sum(r['count']<30 for r in neg)}
        if source=='LOADBench':
            for metric in sorted({r['original_name'] for r in rows}):
                part=[r for r in rows if r['original_name']==metric]
                rr=[r['r2'] for r in part if r['r2'] is not None];hh=[r['hit5'] for r in part if r['hit5'] is not None]
                per_metric[metric]={'targets':len(part),'unit':part[0]['unit'],'median_r2':statistics.median(rr) if rr else None,
                    'macro_hit5':statistics.mean(hh) if hh else None,'negative_targets':sum(v<0 for v in rr)}
    ordered=sorted([r for r in groups['LOADBench'] if r['r2'] is not None],key=lambda r:r['r2'])
    load_values=defaultdict(list);components=Counter();feasible=Counter();source_requests=defaultdict(dict)
    checked_rows=0;cache_rows=0;slot_checks=0;max_cache_rounding=Counter();raw_records=0;raw_label_checks=0;raw_parameter_checks=0
    param_names_seen=set();hashes={};plan=[]
    columns=['sample_id','source_path','source_row','source_sha256','metric_names','raw_performance','performance','mask',
             'semantic_ids','parameter_names','parameters','parameter_units','sample_component','source_feasible']
    for top,spec in sorted(catalog.items()):
        if spec['source_dataset']!='LOADBench':continue
        file=ROOT/'data/fusion_v2/sampled'/(top+'.parquet');pf=pq.ParquetFile(file)
        if not pf.metadata.num_rows:continue
        rows=pq.read_table(file,columns=columns).to_pylist();arrays=cache['topologies'][top]
        target=np.load(ROOT/arrays['folder']/'target.npy',mmap_mode='r');mask=np.load(ROOT/arrays['folder']/'mask.npy',mmap_mode='r');ids=np.load(ROOT/arrays['folder']/'sample_ids.npy',mmap_mode='r')
        assert len(target)==len(rows)
        source_requests.clear()
        for i,row in enumerate(rows):
            raw=dict(zip(row['metric_names'],row['raw_performance']));observed=np.asarray(row['mask'],dtype=bool)
            sid=ids[i].decode() if isinstance(ids[i],bytes) else str(ids[i]);assert sid==row['sample_id']
            assert np.array_equal(mask[i],observed)
            for rule in spec['metric_mapping']:
                j=rule['slot'];name=rule['original_name'];present=name in raw
                assert present==row['mask'][j]
                if present:
                    assert row['performance'][j]==raw[name]*rule.get('factor',1.)
                    assert row['semantic_ids'][j]==rule['semantic_id'];slot_checks+=1
                    load_values[name].append(row['performance'][j])
            y=np.asarray([v if v is not None else 0. for v in row['performance']],dtype='float64')
            assert np.array_equal(target[i][observed],y.astype(target.dtype)[observed])
            for name,val in raw.items():assert np.isfinite(val) and abs(val)<1e12
            for name,val in raw.items():
                j=next(r['slot'] for r in spec['metric_mapping'] if r['original_name']==name)
                max_cache_rounding[name]=max(max_cache_rounding[name],abs(float(target[i,j])-val))
            source_requests[row['source_path']][int(row['source_row'])]=row
            hashes[row['source_path']]=row['source_sha256'];components[row['sample_component']]+=1
            feasible[str(row['source_feasible'])]+=1;checked_rows+=1;cache_rows+=1
        for source_path,requests in source_requests.items():
            file=ROOT/source_path;assert sha(file)==hashes[source_path]
            opener=gzip.open if file.suffix=='.gz' else open;remaining=set(requests)
            with opener(file,'rt',encoding='utf-8-sig',newline='') as stream:
                for line,csv_row in enumerate(csv.DictReader(stream),start=1):
                    if line not in requests:continue
                    row=requests[line];native={k.lower():v for k,v in csv_row.items()}
                    for name,val in zip(row['metric_names'],row['raw_performance']):
                        original=float(native[name.lower()]);assert np.isclose(original,val,rtol=1e-12,atol=1e-15),(top,line,name,original,val)
                        raw_label_checks+=1
                    for name,val,unit in zip(row['parameter_names'],row['parameters'],row['parameter_units']):
                        if name.lower() not in native:continue # derived matching aliases have separate graph checks
                        original=float(native[name.lower()]);expected=original*1e-6 if name.lower().startswith(('param_w','param_l')) else original
                        assert np.isclose(expected,val,rtol=1e-12,atol=1e-20),(top,line,name,expected,val)
                        raw_parameter_checks+=1;param_names_seen.add(name)
                    raw_records+=1;remaining.remove(line)
                    if not remaining:break
            assert not remaining,(source_path,len(remaining))
        for a in [target,mask,ids]:a._mmap.close()
        if len(plan)%15==0:print('checked',top,'total_rows',checked_rows,flush=True)
        plan.append(top)
    distributions={}
    for name,values in load_values.items():
        a=np.asarray(values);distributions[name]={'count':len(a),'min':float(a.min()),'p01':float(np.quantile(a,.01)),
            'median':float(np.median(a)),'p99':float(np.quantile(a,.99)),'max':float(a.max()),'zeros':int((a==0).sum())}
        if name=='phasemargin':distributions[name].update(outside_minus180_plus180=int((abs(a)>180).sum()),abs_over360=int((abs(a)>360).sum()))
    quality=read(ROOT/'data/fusion_v2/loadbench_candidate_quality.json');quality_totals={k:sum(r[k] for r in quality) for k in ['raw_rows_in_downloaded_pool','invalid_rows','conflicting_rows_quarantined','duplicate_design_occurrences_removed','unique_accepted_candidates']}
    data={'source_summaries':summaries,'LOADBench_per_metric':per_metric,'worst_LOADBench_targets':ordered[:12],
          'LOADBench_raw_to_sample_to_cache':{'passed':True,'rows':checked_rows,'nonempty_topologies':len(plan),'metric_slot_checks':slot_checks,
            'original_CSV_rows_verified':raw_records,'original_CSV_scalar_labels_verified':raw_label_checks,
            'original_CSV_direct_parameters_verified':raw_parameter_checks,'cache_rows_checked':cache_rows,
            'source_files_hash_verified':len(hashes),'provenance_row_convention':'1-based data row excluding CSV header',
            'label_cache_dtype':'float32; bit-exact to original labels after explicit cast; not lossless float64',
            'max_float32_cache_absolute_rounding_by_metric':dict(max_cache_rounding)},
          'LOADBench_component_counts':dict(components),'LOADBench_feasibility_counts':dict(feasible),
          'LOADBench_sampled_label_distributions':distributions,'LOADBench_candidate_quality_totals':quality_totals,
          'limits':['Exact reconstruction of stored/source labels does not prove physical validity of extraction.',
            'No raw AC/transient waveforms re-simulated. Official extrapolation/dewrapping semantics require physical audit.',
            'Frozen test metrics are diagnostic reporting only; no refitting or test-driven model selection.',
            'Relative thresholds and metric populations differ across sources; this is not a controlled causal comparison.'],
          'seconds':time.perf_counter()-started,'process_RSS_MiB_at_completion':psutil.Process().memory_info().rss/2**20}
    out=ROOT/'reports/source_gap_diagnosis.json';out.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# 三来源效果差异与LOADBench标签复核','',
      '本次仅只读检查现有样本、原始CSV、训练缓存及冻结测试指标；没有训练、重新仿真、删标签或修改Mask。',
      '', '## 已验证的转换一致性','',
      f'LOADBench全部{checked_rows:,}条、{len(plan)}个有样本拓扑，原始CSV→统一指标槽位→训练缓存检查通过；逐项核对{raw_label_checks:,}个原始性能标量与{raw_parameter_checks:,}个直接参数。W/L的微米→米及负压摆率符号保持一致，缺失槽位与缓存Mask一致。缓存为float32，已按其实际舍入核对，不能称为无损float64。',
      '', '本结论只排查转录/映射/单位/缓存错位，不证明每个原始指标的提取均物理正确。',
      '', '## 数据支持与冻结测试表现','',
      '| 来源 | 样本/拓扑 | 每拓扑平均样本 | 5%命中率 | R²中位数 | 负R²目标 |',
      '| --- | ---: | ---: | ---: | ---: | ---: |']
    for source in ['FALCON','AICircuit','LOADBench']:
        s=summaries[source];lines.append(f"| {source} | {s['sampled_records']:,}/{s['topologies']} | {s['mean_records_per_topology']:.0f} | {s['macro_hit5']:.2%} | {s['median_r2']:.3f} | {s['negative_r2_targets']}/{s['r2_defined']} |")
    lines += ['', '## LOADBench逐指标分布','', '| 指标 | R²中位数 | 5%命中率 | 负R²目标数 |','| --- | ---: | ---: | ---: |']
    for name,s in per_metric.items():
        hit='不适用' if s['macro_hit5'] is None else f"{s['macro_hit5']:.2%}"
        lines.append(f"| {name} | {s['median_r2']:.3f} | {hit} | {s['negative_targets']} |")
    s=summaries['LOADBench']
    lines += ['', '## 判断边界','',
      f"LOADBench有{s['test_targets_under_30_observations']}个契约—指标测试项不足30条，其中{s['negative_targets_under_30_observations']}个为负R²。少样本及窄方差会放大R²波动，但不能掩盖真实预测误差。",
      '', '确定事实：LOADBench拓扑更多、每拓扑记录更少、指标更多，且下载池混合最终Pareto表与早期优化历史。各来源宏命中率使用的目标集合不同，不能将来源差值全部解释为模型优劣。',
      '', '模型侧已知限制：ID最佳版本依赖同契约训练近邻；训练稀疏和优化区域切换会影响这种局部预测。参数近邻验证50.10%、加图后52.00%，提示仍应检查表示与读出能力，不能把误差全归因数据集。',
      '', f"物理质量风险：20万条LOADBench的原始phasemargin最小{distributions['phasemargin']['min']:.8g}度、最大{distributions['phasemargin']['max']:.8g}度，其中{distributions['phasemargin']['abs_over360']:,}条绝对值超过360度。它们与原始CSV逐值相符，属于原始指标定义或仿真工作状态的审计问题，不能归咎于转换时误换单位。",
      '', '官方calc_ac包含0dB频率插值/外推与相位去缠绕；有限值与去哨兵不足以保证常规相位裕度定义。超过固定角度并非统一判废规则，但如此极端的值不能直接当作正常稳定电路的相位裕度解释。对异常值应回查波形或独立复算，不能直接裁剪、取绝对值或改为Mask=False。此前物理质量筛查不充分，应补充独立审计标记，不改写已冻结实验。',
      '', '下一步应在独立版本审计异常指标提取与条件，再用相同划分运行强参数基线/单来源模型来区分数据难度、负迁移与GNN表示限制。未经控制实验，不能认定“LOADBench有问题”或“数据处理已完全没有问题”。',
      '', '完整计数、逐指标分位数、最差目标与检查范围见source_gap_diagnosis.json。']
    (ROOT/'reports/source_gap_diagnosis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'audit_passed':True,'rows':checked_rows,'scalar_labels':raw_label_checks,'seconds':data['seconds'],'source_summaries':summaries,'LOADBench_per_metric':per_metric,'phase_margin_distribution':distributions['phasemargin']},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
