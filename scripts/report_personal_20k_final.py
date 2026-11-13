"""Combine final audit evidence into an experiment-readiness report."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import sys
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read
from src.analoggym_campaign import sha
from src.windows_campaign_io import atomic_json


def main():
    audit=read(ROOT/'reports/analoggym_final_delivery_audit.json')
    measured=read(ROOT/'reports/analoggym_final_measurements.json')
    output=ROOT/'data/generated/analoggym_personal_20k_v1'
    assert audit['samples']==measured['snapshot_samples']==20000
    assert measured['ac_review_flag_count']==0
    assert measured['format_and_provenance_checks_passed']
    for path,expected in {**audit['input_artifact_sha256'],**audit['parquet_shard_sha256']}.items():
        assert sha(ROOT/path)==expected,'Dataset changed after audit: '+path
    safe_folder=output/'splits/anchor_safe_v1';safe_audit=read(safe_folder/'audit.json')
    safe=pq.read_table(safe_folder/'id.parquet').to_pylist();original=pq.read_table(output/'splits/id.parquet').to_pylist()
    assert sha(safe_folder/'id.parquet')==safe_audit['new_split_sha256']
    assert sha(output/'splits/id.parquet')==safe_audit['original_split_sha256']
    assert len(safe)==len({r['sample_id'] for r in safe})==len({r['design_group'] for r in safe})==20000
    assert set(r['sample_id'] for r in safe)==set(r['sample_id'] for r in original)
    assert {r['sample_id'] for r in safe if r['role']=='test'}=={r['sample_id'] for r in original if r['role']=='test'}
    role_counts=Counter((r['topology_id'],r['role']) for r in safe)
    for name in audit['counts']:
        assert [role_counts[(name,role)] for role in ['train','validation','test']]==[1600,200,200]
    for row in safe:
        if row['topology_id']=='ldo_folded_cascode' and row['candidate'] in {14,35,38}:assert row['role']=='train'
    train={r['sample_id'] for r in safe if r['role']=='train'};topology={r['sample_id']:r['topology_id'] for r in safe}
    for filename,kind in [('few_shot_global.json','global'),('few_shot_per_topology.json','per_topology')]:
        previous=set()
        for budget in read(safe_folder/filename)['budgets']:
            selected=set(budget['sample_ids']);assert len(selected)==len(budget['sample_ids'])
            assert previous<=selected<=train;previous=selected
            if kind=='global':assert len(selected)==budget['budget']
            else:
                assert len(selected)==budget['total_budget']==10*budget['budget_per_topology']
                assert set(Counter(topology[s] for s in selected).values())=={budget['budget_per_topology']}
    completed=datetime.fromisoformat(audit['completed_at_UTC']).astimezone(__import__('zoneinfo').ZoneInfo('Asia/Shanghai'))
    lines=['# AnalogGym 独立 2 万条数据集：最终检查与实验可用性评估','',
        '评估日期：2026-10-30。结论基于最终完整数据，替代此前运行中快照的数量与就绪状态；原始来源与数值定义不变。','',
        '**数据生成与存储交付通过：20,000 条有效设计，10 个拓扑各 2,000 条，未发现必需字段意外缺失。完成了图、参数、标签、Mask、来源、去重和划分核验。**',
        '**实验接入有条件通过：监督基线与少样本实验的数据和划分已经具备，但仍需对齐当前 GNN 的图特征、条件输入、新域与输出头；跨来源零样本目前不能直接成立。**','',
        '## 1. 最终规模与检查范围','',
        f"生成完成时间（北京时间）：{completed.strftime('%Y-%m-%d %H:%M:%S')}。实际提交 {audit['attempts']:,} 次尝试，接受20,000条，隔离 {audit['quarantined']:,} 次，有效率 {20000/audit['attempts']*100:.2f}%。其中 {audit['quality_review_exclusions']} 次因AC连续性规则隔离，未用于填充有效样本配额。",'',
        '| 拓扑 | 有效样本 | 器件 | 网络节点 | 物理端口对边 | train / val / test |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for name,count in audit['counts'].items():
        g=measured['graph_summary'][name]
        lines.append(f"| {name} | {count} | {g['devices']} | {g['nodes']} | {g['edges']} | 1600 / 200 / 200 |")
    lines+=['',
        '拓扑覆盖为8种放大器、2种LDO，并未覆盖参考源、传感前端或PLL；符合当前已确认的10拓扑方案，不能写成覆盖全部AnalogGym五类电路。10个原始端口结构哈希均不同；这不等于已经穷尽忽略网名重编号后的全部图同构分析。',
        '完成时运行器已经对20,000条原始波形重算标签，完成记录逐拓扑均为2,000条。本次重新核对完成代码指纹及检查记录，另重新检查全部20,000条导出数据、每条归档SHA和AC波形；均未触发现有AC连续性规则。再按拓扑首尾及中间候选深查100个归档，CRC、逐文件SHA、参数、标签重算及独立公式抽查通过。',
        '本轮不是再次对全部瞬态文本重算20,000遍；全量重算证据来自完成阶段的已核验记录，本轮覆盖范围与之明确区分。既定检查通过不构成所有工作区/应力/版图或硅后性能的保证。','',
        '## 2. 数据形式与文件位置','',
        '根目录：`data/generated/analoggym_personal_20k_v1/`。数据仍独立，未合并到fusion_v2。','',
        '| 文件或目录 | 内容及使用方法 |','| --- | --- |',
        '| dataset.json | 有效Parquet分片清单；数据加载以此为准 |',
        '| samples/<topology>/part-*.parquet | 参数、器件特征、16槽性能/Mask、语义和原始来源 |',
        '| topologies/<topology>/graph.json | 共用网表图，网络节点与器件端口对多重边 |',
        '| topologies/<topology>/performance_mapping.json | 指标名称、单位、定义、语义ID与槽位 |',
        '| topologies/<topology>/conditions.json | PDK、供电、温度、负载及测量条件 |',
        '| raw/ | 每次尝试的网表、参数、日志、波形；包括失败/隔离尝试 |',
        '| ledger.sqlite | 接受、失败、去重与质量复核台账 |',
        '| personal_index.parquet | 20,000条唯一样本/设计的索引 |',
        '| splits/id.parquet | 原始16,000/2,000/2,000划分，保留未改 |',
        '| splits/anchor_safe_v1/ | 本次补充、建议实验采用的中心隔离版划分及配套少样本预算 |','',
        '必需字段包含 sample_id、source_dataset、topology_id、circuit_family、process/PDK、temperature、supply_voltage、testbench_id、graph、device_features、performance、mask，另有原生参数名/值/单位、条件JSON、语义ID、候选编号、采样版本、设计组与来源SHA。图在JSON中单独存储，样本记录图路径，加载器可物化为图字典和边特征。','',
        '原生参数根据 parameter_units 解释，尺寸常为µm；器件特征宽长采用m、电容采用F。二者不是单位错误，不能丢掉单位列后混用。MOS G/D/S/B、六个端口对、并行边及端口短接自环完整保留；图与源网表逐端口检查通过。','',
        '## 3. 缺失项是否有问题','',
        '| 检查项 | 结果 | 判定 |','| --- | --- | --- |',
        '| 必需字段缺列、空ID、源路径等意外缺失 | 0 | 通过 |',
        '| 样本ID、设计组重复 | 0 | 通过 |',
        '| 已知标签NaN/Inf或与原台账不符 | 0 | 通过 |',
        '| 已知器件必需W/L/m/R/C/L缺失 | 0 | 通过 |',
        '| 放大器temperature顶层为None | 16,000条 | 有意留空：不同指标采用扫温、25°C或27°C，完整条件另存 |',
        '| performance未使用槽位 | 28,000个标量槽 | 有意留空，Mask=False |',
        '| 器件类型不适用的可选特征 | 存在None | 正常，例如电阻没有MOS宽长；不得伪造填值 |','',
        '16,000条放大器每条有15个已定义指标和1个空槽；4,000条LDO每条有13个已定义指标和3个空槽。总计292,000个真实性能标量与28,000个未使用槽位，Mask、数值存在性与语义ID逐项一致。Mask只表示缺失，不负责屏蔽失败仿真或指标定义差异。','',
        '没有确认过的指数量、面积等可选特征不能臆造；应在模型中保留known mask。具体按器件类型的缺失计数见机器审计JSON。本次未发现为了正常读取和监督必需的器件参数意外缺失。','',
        '## 4. 结果合理性与工程边界','',
        '此前Song候选101、252已隔离，不在20,000条有效数据或划分中。后续AC规则也隔离了其他触发者，最终全量重新扫描触发数为0。该规则检查大量相邻频点增益跳变/异常多重交越，不保证能检测全部数值不稳定，不能写成“电路物理正确性已被完全证明”。',
        '性能差和标签错误须分开。既定验收主要保证可收敛、可提取、有限、稳态/终值处于较宽工作窗口；它不是高性能放大器或高精度LDO的产品规格。下列阈值只是本次工程观察线，不改变有效样本定义。','',
        '| 拓扑 | 终值跟踪误差>10mV | DC稳压误差>5% | 10Hz电源传递>0dB |',
        '| --- | ---: | ---: | ---: |']
    for name in audit['counts']:
        w=measured['engineering_warnings'].get(name,{})
        amp='ldo_' not in name
        lines.append(f"| {name} | {w.get('tracking_error_over_10mV',0) if amp else '不适用'} | {w.get('dc_regulation_error_over_5pct',0) if not amp else '不适用'} | {w.get('supply_transfer_10Hz_above_0dB',0) if not amp else '不适用'} |")
    lines+=['',
        'folded LDO 有1,415/2,000条超过5% DC稳压误差，1,266条在10Hz放大电源扰动；这些是当前固定平台下的差性能设计，不应冒充精密稳压器。Alfio 有1,987/2,000条终值跟踪误差超过10mV，说明本采样域明显包含较差跟踪表现。性能回归可保留这些点；若实验只研究满足某规格的设计，必须先声明规格并报告筛除率，样本量可能显著下降，不能继续声称全部都合格。',
        '带符号偏差/调整率允许为负。供电传递不是通用正号PSRR，展开相位不是自动的相位裕度，observed settling 是围绕实测终值的建立时间。这些名称与定义必须沿用，不得为了与主集合合并而改名。','',
        '## 5. 划分、少样本预算与本次修正','',
        '原始划分的样本与设计组交集均为0；各拓扑1600/200/200，索引、参数记录、图引用与台账逐项一致。全域预算k=0/5/10/20/50/100/200/500/1000；每拓扑预算k=5/10/20/50/100。两种预算均嵌套、无重复，且只从train选取；前者k<10不能覆盖全部拓扑，后者总预算为10k。',
        '另发现：用于自适应选择folded采样中心的候选14、38在原验证集，候选35在训练集。不存在这些中心进入测试集的问题，也不存在相同样本跨集合重复，但验证集不宜声称完全未参与数据收集设计。',
        '**本次保留原文件，新增 `splits/anchor_safe_v1/`，建议后续实验统一使用该版本。** 将候选14、38移到train，并按固定哈希从同拓扑原train交换两条非中心设计到validation，共4条角色变化；样本不增减，所有test ID保持不变，仍是每拓扑1600/200/200。两套少样本预算已根据新训练池重新生成并从文件独立读回检查通过。',
        '此修正避免把已用于中心选择的记录当成独立验证观察，但不消除局部参数近邻的统计相似性。中心附近采样导致随机划分主要衡量局部插值；若要声称未知参数区域或未知拓扑泛化，需要额外按中心/参数空间或拓扑完整留出的协议。当前个人集没有独立LOTO文件，不影响当前ID/少样本用途，但不能声称所有泛化协议都已完成。',
        '如用完整2,000条目标域validation调参，它们是k之外的额外目标标签预算；不能只报告“用了k条”。数据收集阶段用于资格、精度和中心选择的仿真成本也应与模型适配预算分开披露。归一化尚未拟合，后续只能用声明的训练或适配数据拟合。','',
        '## 6. 满不满足当前GNN实验输入','',
        '| 要求/实验 | 当前状态 | 还需要做什么 |','| --- | --- | --- |',
        '| 2万真实有效设计、10拓扑均衡、独立保存 | 已满足 | 无需补仿来凑数 |',
        '| 图+器件参数+性能+Mask+来源 | 已满足存储层形式 | 通过正式加载器读取 |',
        '| 原始图电路连接与MOS端口 | 已满足 | 继续保留并行边、自环和bulk |',
        '| 主GNN相同的电源/地角色、特征和关系张量 | 尚未完成 | 个人图17维基本器件特征，需增强为30维并生成34维节点特征、双向关系等 |',
        '| 个人目标域/指标头与已知条件输入 | 尚未完成 | 加入新域、新目标语义/单位尺度，显式带入偏置、负载等 |',
        '| 个人监督基线P1 | 数据与划分具备 | 接口适配后可开展 |',
        '| 公开预训练后少样本适配 | 数据与预算具备 | 同上，使用一致的预算和独立目标头 |',
        '| 公开直接零样本预测个人数据P2 | 不可直接批准 | 证明指标定义/条件兼容及读出/域训练覆盖 |',
        '| 正式跨拓扑LOTO | 尚无个人LOTO协议 | 单独设计完整拓扑留出和已训练读出共享规则 |','',
        '当前 GNN builder 依赖fusion catalog、增强图词表和监督路由；个人数据已有图不等于可以不改接口直接交给该builder。个人图虽保留电源/地网名与端口，却没有同主集合一致的显式角色特征。需要从已经保存的网表与测试平台生成派生输入，不应重新定义或改写原始图。',
        'AnalogGym个人平台的500pF负载、0.45V共模和0.1Hz起始AC，与LOADBench等来源条件不同。Mask不能解决指标物理定义不兼容。当前无通用连续PVT向量编码器，外部偏置即使已存在参数列中，也必须明确接入模型，不能仅靠域嵌入代替。',
        '**因此，不需要因为上述接口差异再做2万条仿真；需要补的是后处理/模型适配。** 如果改用新的测试频率、负载或新的物理指标，则需要评估是否能从现有波形严谨重提取，不能直接伪造新标签。','',
        '## 7. 交付判定与证据','',
        '数据生成验收：通过。必需内容缺失：未发现。存储形式：满足网表图、器件参数、统一标签容器与Mask、完整来源追溯。原始ID和少样本划分通过检查，并补充中心隔离版本。',
        '直接启动现有RGCN正式实验：尚不满足，输入/目标域适配与实验入口仍需完成。跨来源零样本：指标条件未批准，仍阻塞。全部样本高性能合格：不成立，也不是当前“有效样本”的含义。','',
        '- [全量参数、指标分布与AC重扫证据](analoggym_final_measurements.json)',
        '- [必需项、缺失、来源、划分与预算核验](analoggym_final_delivery_audit.json)',
        '- [原始结果评估报告（历史阶段）](analoggym_result_evaluation.md)',
        '- [当前GNN设置详解](gnn_settings_detailed.md)','',
        '复现本次审计：','',
        '```powershell',
        r'.\.venv\Scripts\python.exe -B scripts/assess_personal_20k_final.py',
        r'.\.venv\Scripts\python.exe -B scripts/build_personal_anchor_safe_splits.py',
        r'.\.venv\Scripts\python.exe -B scripts/report_personal_20k_final.py',
        '```','',
        '全量波形分布重扫由 scripts/evaluate_analoggym_personal.py 的 main(report_path=..., review_index_path=..., render_report=False) 生成本报告引用的独立最终快照，未覆盖历史阶段报告。本轮没有训练模型、拟合归一化或修改仿真标签。']
    path=ROOT/'reports/analoggym_final_20k_assessment.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    atomic_json(ROOT/'reports/analoggym_final_assessment_receipt.json',{
        'assessed_at_UTC':datetime.now(timezone.utc).isoformat(),'data_generation_passed':True,'missing_required':0,
        'samples':20000,'ac_review_flags':0,'direct_current_model_ready':False,
        'recommended_split':'data/generated/analoggym_personal_20k_v1/splits/anchor_safe_v1/id.parquet',
        'original_split_preserved':True,'fixed_test_ids_preserved':True,'few_shot_budget_files_reverified':True,
        'training_performed':False,'final_report_sha256':sha(path)})
    print('Final assessment report ready. 20,000 records passed; model adapter still required.')


if __name__=='__main__':main()
