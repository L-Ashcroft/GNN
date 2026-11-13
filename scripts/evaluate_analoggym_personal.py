"""Audit a stable exported snapshot without training or starting simulations."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3
import sys
import time
import zipfile
import numpy as np
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read,digest
from src.analoggym_campaign import sha,extract,evaluate,conditions
from src.analoggym_campaign_runner import metadata,design_fingerprint,unit
from src.analoggym_personal_engine import parameter_plan,sampling_trace
from src.windows_campaign_io import atomic_json
from src.analoggym_personal_quality import ac_diagnostics

OUT=ROOT/'data/generated/analoggym_personal_20k_v1'
REPORT=ROOT/'reports/analoggym_result_evaluation.json'


def stable_snapshot():
    for attempt in range(5):
        snapshot=read(OUT/'dataset.json');rows=[]
        for shard in snapshot['shards']:rows.extend(pq.ParquetFile(ROOT/shard['path']).read().to_pylist())
        if digest(snapshot)==digest(read(OUT/'dataset.json')) and len(rows)==snapshot['sample_count']:
            return snapshot,rows
        time.sleep(.5)
    raise RuntimeError('Export changed during snapshot capture; rerun after next export')


def main(report_path=None,review_index_path=None,render_report=True):
    captured=datetime.now(timezone.utc).isoformat();snapshot,rows=stable_snapshot()
    recipe=read(OUT/'recipe.json');manifest=read(OUT/'manifest.json');revision=read(OUT/'sampling_policy_v2.json')
    entries={e['name']:dict(e) for e in recipe['entries']}
    entries['ldo_folded_cascode']['_sampling_policy']=revision['sampling_policy']
    for path,expected in recipe['code_sha256'].items():assert sha(ROOT/path)==expected
    assert digest(recipe)==manifest['recipe_hash']
    db=sqlite3.connect(f'file:{(OUT/"ledger.sqlite").as_posix()}?mode=ro',uri=True)
    db.execute('begin')
    attempts={(n,c):(bool(ok),json.loads(raw),archive_sha) for n,c,ok,raw,archive_sha in db.execute(
        'select topology,candidate,accepted,result_json,archive_sha256 from attempts')};db.close()
    by_topology=defaultdict(list)
    for row in rows:by_topology[row['topology_id']].append(row)
    graphs={};graph_summary={};issues=[];warnings=[];metric_stats={};parameter_stats={}
    for name,entry in entries.items():
        graph=read(OUT/'topologies'/name/'graph.json');graphs[name]=graph
        assert sha(ROOT/entry['dut'])==entry['dut_sha256']==graph['netlist_sha256']
        text=(ROOT/entry['dut']).read_text().lower();expected=[]
        assert graph['num_devices']==len(graph['devices'])
        assert graph['num_nodes']==len(graph['nets'])==len({n for d in graph['devices'] for n in d['nets']})
        for i,device in enumerate(graph['devices']):
            match=re.search(r'^'+re.escape(device['name'])+r'\s+(.+)$',text,re.M)
            assert match and match[1].split()[:len(device['nets'])]==device['nets']
            if device['model'] in {'nmos','pmos'}:assert device['ports']==['D','G','S','B']
            for a in range(len(device['ports'])):
                for b in range(a+1,len(device['ports'])):
                    expected.append((i,device['ports'][a],device['ports'][b],device['nets'][a],device['nets'][b]))
        actual=[(e['device'],*e['ports'],graph['nets'][e['source']],graph['nets'][e['target']]) for e in graph['edges']]
        assert expected==actual and len(actual)==graph['num_edges']
        assert graph['edge_index']==[[e['source'] for e in graph['edges']],[e['target'] for e in graph['edges']]]
        assert graph['edge_device_index']==[e['device'] for e in graph['edges']]
        missing=[k for k in ['node_feature_names','feature_vocabulary'] if k not in graph]
        graph_summary[name]={'nodes':graph['num_nodes'],'devices':graph['num_devices'],'edges':graph['num_edges'],
            'edges_per_device':graph['num_edges']/graph['num_devices'],'topology_hash':graph['topology_hash'],
            'stored_feature_dimensions':len(graph['feature_names']),'missing_gnn_graph_fields':missing,
            'explicit_supply_ground_annotations':bool(graph.get('node_annotations'))}
        part=by_topology[name];mapping=metadata(entry)
        metric_stats[name]={}
        for metric in mapping['metrics']:
            values=np.array([r['performance'][metric['slot']] for r in part])
            metric_stats[name][metric['name']]={'unit':metric['unit'],'definition':metric['definition'],
                'min':float(values.min()),'p05':float(np.quantile(values,.05)),'median':float(np.median(values)),
                'p95':float(np.quantile(values,.95)),'max':float(values.max()),'unique_values':len(set(values))}
        values=np.array([r['parameters'] for r in part]);names=part[0]['parameter_names']
        assert all(r['parameter_names']==names for r in part)
        varying=(np.ptp(values,axis=0)>0)
        parameter_stats[name]={'total_native_parameters':len(names),'varying_in_snapshot':int(sum(varying)),
            'constant_names':[n for n,v in zip(names,varying) if not v],
            'per_parameter':{n:{'min':float(values[:,i].min()),'median':float(np.median(values[:,i])),
                'max':float(values[:,i].max()),'unit':part[0]['parameter_units'][i]} for i,n in enumerate(names)}}
    ids=set();designs=set();deep_counts=Counter();versions=Counter();flags=[];margins=defaultdict(Counter)
    selected={}
    for name,part in by_topology.items():
        ordered=sorted(part,key=lambda r:r['candidate'])
        selected[name]={ordered[int(i)]['sample_id'] for i in np.linspace(0,len(ordered)-1,min(10,len(ordered)),dtype=int)}
    for index,row in enumerate(rows):
        name=row['topology_id'];entry=entries[name];graph=graphs[name];mapping=metadata(entry)
        assert row['sample_id'] not in ids and row['design_group'] not in designs
        ids.add(row['sample_id']);designs.add(row['design_group'])
        assert len(row['performance'])==len(row['mask'])==len(row['semantic_ids'])==16
        for v,m,s in zip(row['performance'],row['mask'],row['semantic_ids']):
            assert bool(m)==(v is not None)==(s is not None)
            if m:assert np.isfinite(v)
        params=dict(zip(row['parameter_names'],row['parameters']))
        assert len(params)==len(row['parameter_names'])==len(row['parameter_units'])
        assert row['parameter_units']==[unit(k) for k in row['parameter_names']]
        assert params==parameter_plan(entry,row['candidate'])
        assert row['design_group']==design_fingerprint(entry,params)
        assert row['sample_id']=='AnalogGym_generated:'+manifest['recipe_hash'][:16]+':'+row['design_group']
        trace=sampling_trace(entry,row['candidate'])
        assert row['sampling_policy_version']==trace['sampling_policy_version']
        assert row['sampling_policy_sha256']==(trace['sampling_policy_sha256'] or manifest['recipe_hash'])
        versions[(name,row['sampling_policy_version'])]+=1
        assert json.loads(row['conditions_json'])==conditions(entry)
        assert row['testbench_id']==mapping['group_id'] and row['recipe_hash']==manifest['recipe_hash']
        assert row['semantic_ids']==[m['semantic_id'] for m in mapping['metrics']]+[None]*(16-len(mapping['metrics']))
        assert row['temperature']==(27. if entry['family']=='LDO' else None)
        expected=[[None if x is None else evaluate(x,params) for x in f] for f in graph['feature_expressions']]
        assert expected==row['device_features']
        for device,features in zip(graph['devices'],row['device_features']):
            assert len(features)==len(graph['feature_names'])
            if device['model'] in {'nmos','pmos'}:
                w,l,m=features[:3]
                assert .42e-6*(1-1e-10)<=w<=100e-6*(1+1e-10)
                assert .15e-6*(1-1e-10)<=l<=30e-6*(1+1e-10)
                assert m>=1 and abs(m-round(m))<1e-9
        accepted,result,archive_sha=attempts[(name,row['candidate'])]
        assert accepted and result['parameters_native']==params
        assert result['performance']==[v for v,m in zip(row['performance'],row['mask']) if m]
        archive=(ROOT/row['source_path']).resolve();assert archive.is_relative_to(OUT.resolve())
        assert sha(archive)==row['source_sha256']==archive_sha
        with zipfile.ZipFile(archive) as z:
            ac_bytes=z.read('ac_wave.dat')
            assert hashlib.sha256(ac_bytes).hexdigest()==result['artifacts_sha256']['ac_wave.dat']
            ac=np.loadtxt(io.BytesIO(ac_bytes),ndmin=2)
            assert np.isfinite(ac).all() and np.all(np.diff(ac[:,0])>0)
            diagnostic=ac_diagnostics(ac) if entry['family']=='Amplifier' else None
            if diagnostic and diagnostic['review_required']:
                flags.append({'sample_id':row['sample_id'],'topology_id':name,'candidate':row['candidate'],
                    'source_path':row['source_path'],'reason':'AC numerical-continuity review',**diagnostic})
            if row['sample_id'] in selected[name] or (diagnostic and diagnostic['review_required']):
                assert z.testzip() is None
                for member,expected_sha in result['artifacts_sha256'].items():
                    assert hashlib.sha256(z.read(member)).hexdigest()==expected_sha
                wave={n:z.read(n) for n in z.namelist() if n.endswith('_wave.dat')}
                np.testing.assert_array_equal(extract(entry,wave),result['performance'])
                native={k:float(v) for k,v in re.findall(r'\.param (\w+)=(\S+)',z.read('parameters.spice').decode())}
                assert native==params
                # Independent spot formulas, rather than calling the label extractor again.
                if entry['family']=='Amplifier':
                    dc=np.loadtxt(io.BytesIO(wave['dc_wave.dat']));at25=dc[dc[:,0]==25][0]
                    np.testing.assert_allclose([20*np.log10(np.hypot(ac[0,1],ac[0,2])),-1.8*at25[3],at25[1]-.45],
                        [row['performance'][3],row['performance'][1],row['performance'][2]],rtol=1e-12,atol=1e-14)
                else:
                    dc=np.loadtxt(io.BytesIO(wave['load_wave.dat']))
                    np.testing.assert_allclose([*dc[:,1],*(-1.8*dc[:,3]),(dc[1,1]-dc[0,1])/.009],
                        row['performance'][:5],rtol=1e-12,atol=1e-14)
                deep_counts[name]+=1
        p=row['performance']
        if entry['family']=='Amplifier':
            assert p[1]>0 and .1<=p[7]<=1e9 and min(p[9:13])>0
            if abs(p[13])>.01 or abs(p[14])>.01:margins[name]['tracking_error_over_10mV']+=1
            if abs(p[8])>360:margins[name]['absolute_unwrapped_phase_over_360deg']+=1
        else:
            assert .96<=p[0]<=1.44 and .96<=p[1]<=1.44 and min(p[2:4])>0
            assert .96<=p[11]<=1.44 and .96<=p[12]<=1.44
            if max(abs(p[0]-1.2),abs(p[1]-1.2))>.06:margins[name]['dc_regulation_error_over_5pct']+=1
            if abs(p[5])>1:margins[name]['absolute_line_slope_over_1']+=1
            if p[6]>0:margins[name]['supply_transfer_10Hz_above_0dB']+=1
        if index%250==0:print('checked',index,'/',len(rows),flush=True)
    policy_yields=defaultdict(lambda:Counter(attempts=0,accepted=0))
    failures=defaultdict(Counter)
    for (name,candidate),(accepted,result,_) in attempts.items():
        version=result.get('sampling_policy_version','personal_sampler_v1')
        group=policy_yields[(name,version)];group['attempts']+=1;group['accepted']+=accepted
        if not accepted:failures[name][result.get('failure')]+=1
    group_coverage=defaultdict(list)
    for name,entry in entries.items():group_coverage[metadata(entry)['group_id']].append(name)
    flagged_ids={r['sample_id'] for r in flags}
    result={'captured_at_UTC':captured,'export_snapshot_updated_at':snapshot['updated_at'],
        'snapshot_samples':len(rows),'snapshot_topologies':len(by_topology),
        'unique_structural_hashes':len({g['topology_hash'] for g in graphs.values()}),
        'ledger_attempts_at_audit':len(attempts),'ledger_accepted_at_audit':sum(v[0] for v in attempts.values()),
        'counts':dict(Counter(r['topology_id'] for r in rows)),
        'format_and_provenance_checks_passed':True,'archive_sha_checked':len(rows),'ac_waveforms_scanned':len(rows),
        'deep_crc_sha_and_labels_recomputed':dict(deep_counts),'independent_formula_spot_checks_passed':True,
        'graph_summary':graph_summary,'metric_statistics':metric_stats,'parameter_statistics':parameter_stats,
        'engineering_warnings':{n:dict(v) for n,v in margins.items()},
        'sampling_version_counts':[{'topology':n,'version':v,'rows':c} for (n,v),c in versions.items()],
        'policy_yields':[{'topology':n,'version':v,**dict(c)} for (n,v),c in policy_yields.items()],
        'failure_reasons':{n:dict(c) for n,c in failures.items()},'supervision_group_coverage':dict(group_coverage),
        'ac_review_flags':flags,'ac_review_flag_count':len(flags),'remaining_without_this_flag':len(rows)-len(flags),
        'review_rule':'At least 10 adjacent gain jumps >20 dB or at least 20 descending unity crossings, on the existing 100-points/decade grid. Review flag, not a proof of physical impossibility; not a changed production acceptance criterion.',
        'gnn_adapter_ready':False,'formal_splits_ready':(OUT/'splits/id.parquet').exists(),
        'normalization_fitted':False,'simulation_or_production_files_modified':False,
        'experiment_decision':'Conditional use after AC review/exclusion, graph/domain/head adapter, and frozen leak-checked splits. Cross-source zero-shot not approved.'}
    from src.load_analoggym_generated import iter_campaign_samples
    loaded=set();loaded_topologies=set()
    for row in iter_campaign_samples(OUT):
        assert len(row['edge_features'])==row['graph']['num_edges']==len(row['edge_port_pairs'])
        assert row['sample_id'] not in loaded
        loaded.add(row['sample_id']);loaded_topologies.add(row['topology_id'])
    result['loader_check']={'passed':True,'rows':len(loaded),'topologies':len(loaded_topologies),
        'scope':'Additional current export read after snapshot audit; may include newly exported rows.'}
    atomic_json(report_path or REPORT,result)
    # Snapshot-scoped review decisions; do not change mask or production accepted status.
    import pyarrow as pa
    decisions=[{'sample_id':r['sample_id'],'topology_id':r['topology_id'],'candidate':r['candidate'],
        'requires_ac_review':r['sample_id'] in flagged_ids,'snapshot_captured_at_UTC':captured} for r in rows]
    pq.write_table(pa.Table.from_pylist(decisions),review_index_path or ROOT/'reports/analoggym_result_review_index.parquet',compression='zstd')
    if render_report:render(result)
    print(json.dumps({k:result[k] for k in ['snapshot_samples','ac_review_flag_count','remaining_without_this_flag','deep_crc_sha_and_labels_recomputed']},indent=2))


def render(r):
    lines=['# AnalogGym 个人仿真数据质量与实验可用性评估','',
        f"审计时间：{r['captured_at_UTC']}。导出快照时间：{r['export_snapshot_updated_at']}。后台仍在运行，本报告不代表后续新样本的检查结果。",'',
        '**结论：真实数据的格式和追溯检查通过，但当前不能直接作为最终实验集。存在需隔离复核的 AC 曲线异常；图输入、目标域输出头及固定划分尚未就绪。**','',
        f"本轮覆盖 {r['snapshot_samples']:,} 条导出样本、{r['snapshot_topologies']} 个拓扑、{r['unique_structural_hashes']} 个结构哈希。",
        f"发现 {r['ac_review_flag_count']} 条 AC 连续性复核标记；剩余 {r['remaining_without_this_flag']:,} 条仅表示未触发本项标记，不能宣称全部达到产品规格。",'',
        '## 1. 核验覆盖与数据形式','',
        f"全部样本检查唯一 ID/设计、参数重现、源台账、归档 SHA、16 槽 Mask、语义契约、单位及器件特征；扫描全部 {r['ac_waveforms_scanned']} 个 AC 波形。",
        f"分拓扑均匀选择首尾及中间候选，另加入触发 AC 复核标记的样本，深查 {sum(r['deep_crc_sha_and_labels_recomputed'].values())} 个归档：CRC、文件 SHA、原始参数和全部标签重算通过。独立公式核对增益、供电功耗、偏差或 LDO 负载稳压指标通过。",
        '这些检查证明记录与既定提取算法一致，不能单独证明测量平台没有数值问题，也不代替硅后验证。','',
        f"独立加载接口 iter_campaign_samples 已额外实测：{r.get('loader_check',{}).get('rows','待记录')} 条记录的图与边特征能够正常组装。这只证明数据加载正常，不等于现有 GNN tensorize 接口已适配。额外加载时间晚于审计快照，数量可略多。",'',
        '每条 Parquet 记录包含网表图 JSON 的路径、参数名/值/单位、器件特征、performance[16]、mask[16]、semantic_ids、条件与归档来源。原生参数以 parameter_units 指定的单位保存（尺寸通常 µm），device_features 中宽长为 m、电容为 F。不能把原生参数误当全部 SI。',
        '放大器使用 15 个真实指标，LDO 使用 13 个；剩余槽位为 None / Mask=False。不同电路族的槽位含义不同，必须按 semantic_ids 和测试条件路由。缺失 Mask 不用于遮盖坏数据。',
        '原始图为网络节点、器件端口对边；MOS 六个端口对以及 G/D/S/B 保留，自环和并行边保留。图文件与原网表逐端口及逐边核对通过。','',
        '| 拓扑 | 导出样本 | 器件 | 网络节点 | 端口对边 | AC 复核标记 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    counts=Counter(f['topology_id'] for f in r['ac_review_flags'])
    for n,c in r['counts'].items():
        g=r['graph_summary'][n];lines.append(f"| {n} | {c} | {g['devices']} | {g['nodes']} | {g['edges']} | {counts[n]} |")
    lines+=['','## 2. 数值异常与工程质量','',
        'AC 复核规则：同一既有 100 点/十倍频网格上，至少 10 次相邻增益跳变超过 20dB，或至少 20 次下降方向的 0dB 交越。多重交越本身不一定错误，该规则筛选的是应人工检查和重仿的极端曲线，不能直接解释为物理不可能。',
        'Song 候选 101 的原始曲线具有低频相邻点约 50dB 跳变和大量交越，导致展开相位累积到数千度；候选 252 也有同类表现。这类数据即使 NaN/Inf 检查和原标签重算均通过，也不能直接监督性能预测。当前提取器和五分钟巡检未包含这项连续性检查，因此巡检 health=ok 不表示本项物理/数值合理性已通过。',
        '复核清单位于同名 JSON 的 ac_review_flags；reports/analoggym_result_review_index.parquet 按 sample_id 标记本次快照。建议在实验视图中暂时排除触发者，保留原归档和生产状态，之后通过独立数值设置复仿确认；不要把这些标签改成缺失 Mask，也不要只剪裁异常相位。未在本轮改变正在运行的生产配方。','',
        '放大器的闭环终值容许约 20% 偏差，且 observed settling 是相对观察终值的收敛时间，不是相对理想目标的精密建立时间。因此“成功样本”包含性能差但可测量的设计；不能将其全部描述为高性能合格设计。相位不是自动的 phase margin，电源传递也不是通用正号 PSRR。','',
        '| 拓扑 | 终值跟踪偏差 >10mV | LDO DC 误差 >5% | 低频电源传递 >0dB |',
        '| --- | ---: | ---: | ---: |']
    for n in r['counts']:
        w=r['engineering_warnings'].get(n,{})
        lines.append(f"| {n} | {w.get('tracking_error_over_10mV',0)} | {w.get('dc_regulation_error_over_5pct',0)} | {w.get('supply_transfer_10Hz_above_0dB',0)} |")
    lines+=['', 'folded LDO 部分点虽然通过 1.2V±20% 窗口，但低频电源扰动抑制、线性调整率较差。负的调整率是带符号斜率，不是负功耗；正的电源传递 dB 表示放大电源扰动，可能是真实差设计，需要按实验规格分层说明，不自动删值。没有做所有晶体管工作区、每管端电压应力或版图可制造性验证。','',
        '本节阈值是本次评估采用的工程观察线，不是此前已有的验收标准；表中不适用的电路族显示 0，不表示该指标被测为零。','',
        '## 3. 参数、覆盖与采样偏差','',
        '所有样本参数均能按候选编号、种子和对应采样版本重现；MOS 尺寸边界、倍乘整数、原始别名和 SI 器件特征一致。不同参数值与设计哈希不重复，但数值唯一不等于统计独立；邻近设计可能高度相似。逐参数范围和标签的 min/p05/median/p95/max 在 JSON 中。',
        '当前仅覆盖八种放大器与两种 LDO，并未覆盖参考源、传感前端、PLL 等全部五类。大部分参数围绕成功中心局部抽样；folded v2 更窄且中心来自前期成功设计。随机设计划分衡量局部插值，不等于宽设计空间或未知拓扑泛化。若增加按中心/参数空间分组的压力测试，必须先定义可追踪分组，不能将近邻随机划分结果冒充域外能力。','',
        '## 4. 能否用于讲义中的实验','',
        '| 用途 | 当前判断 | 必须补齐 |','| --- | --- | --- |',
        '| 数据接口、图与标签工程检查 | 可以；本次已执行，无训练 | 无法据此声称预测精度 |',
        '| 个人数据集监督基线/上界 P1 | 有条件可用，尚未正式启动 | AC 复核筛除、固定版本、划分、输入适配 |',
        '| 公开预训练后少样本适配 | 有条件可用 | 新域嵌入、目标头、真实偏置和测试条件输入；声明 k 与验证标签预算 |',
        '| 公开数据直接零样本预测个人数据 P2 | 当前不可直接成立 | 共同指标物理定义与条件验证，并证明头和域已受训练 |',
        '| 正式跨拓扑泛化结论 | 尚不能给出 | 明确 LOTO、结构去重、按拓扑完整留出及训练覆盖 |','',
        '个人放大器条件主要为 500pF、0.45V 共模、0.1Hz 起始 AC，功耗/偏差在25°C而 AC/瞬态在27°C，另含 -40..125°C 扫温。LOADBench 为不同负载/激励/频率条件，FALCON/AICircuit 部分平台未公开。不能把同名 Gain、功耗或相位直接混合计算损失。',
        '个人数据图仅有 17 维基本器件特征，尚无主集合所用的节点角色/34维节点输入、完整30维器件特征、关系词表及有向 edge_types；当前 GNN builder 依赖 fusion catalog/监督路由。VDD/GND 虽在端口和网名中保留，个人图没有显式角色特征。因此此数据已有图并不等于已经能直接喂给现有 RGCN。偏置电压在参数记录中，但未作为 DUT 器件边自动进入网络，适配时必须补入条件输入，禁止加入仿真输出作为特征。',
        '完整 2 万条后的自动协议已实现，但目前最终划分尚未生成，不应将全量目标写成已完成。协议预期每拓扑1600/200/200，并生成全域与每拓扑两种嵌套少样本预算。正式结果需先固定样本及划分，测试集不得参与训练/调参，归一化只在所声明训练或适配样本拟合。AC 复核若改变最终可用数量，需要重新审计配额与划分，不能继续沿用“每拓扑2000”的断言。','',
        '## 5. 交付与下一步优先级','',
        '1. 优先为 AC 连续性复核标记做测量平台数值验证；确认后版本化修复验收规则，避免只看 finite/收敛日志就接受。',
        '2. 保留原始生产数据，建立审阅后的训练视图；本次索引只覆盖本次快照，新样本需要重新检查。',
        '3. 对齐个人数据图与模型接口，加入可追溯的电源/地、端口关系、已知偏置条件和独立目标域头。',
        '4. 达到目标后冻结并审计划分，再进行 P1 与少样本实验；零样本兼容性另行证明。','',
        '复现：` .\\.venv\\Scripts\\python.exe -B scripts/evaluate_analoggym_personal.py `。只读生产数据，不增加仿真，不训练、不拟合归一化。报告 JSON 与 review_index 会更新为新审计快照。']
    if (OUT/'quality_policy_v1.json').exists():
        lines.insert(2,'> 后续修订：已回查升级时的全部 2,739 条已接受样本，隔离 Song 101、252 两条并保留原始归档；后续新候选加入 AC 连续性入库门控。以下是此前审计快照，图与模型接口限制仍适用。当前策略见生产目录 quality_policy_v1.json。')
        lines.insert(3,'')
    (ROOT/'reports/analoggym_result_evaluation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
