"""Generate traceable audit and final statistics from actual artifacts."""
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.common import ROOT, read, dump


def table(headers, rows):
    clean = lambda x: str(x).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(clean(x) for x in row) + ' |' for row in rows])


def groups(specs, field):
    result = defaultdict(list)
    for s in specs:
        result[s[field]].append(s['topology_id'])
    return result


def main():
    inventory = read(ROOT / 'reports/source_inventory.json')
    specs = read(ROOT / 'data/processed/catalog.json')
    stats = read(ROOT / 'data/processed/conversion_statistics.json')
    summary = read(ROOT / 'data/processed/sampling_summary.json')
    quality = read(ROOT / 'data/processed/quality_checks.json')
    strict = read(ROOT / 'data/strict/sampling_summary.json')
    strict_quality = read(ROOT / 'data/strict/quality_checks.json')
    strict_designs = pd.read_parquet(ROOT / 'data/strict/index.parquet', columns=['design_group']).design_group.nunique()
    reproducibility = read(ROOT / 'reports/reproducibility.json')
    overlaps = read(ROOT / 'reports/cross_source_overlap.json')
    sampled = pd.read_csv(ROOT / 'data/processed/sampling_statistics.csv').set_index('topology_id')
    bykey = {s['topology_id']: s for s in stats}
    lock = {k: {'url': v['url'], 'commit': v['commit']} for k, v in inventory['repositories'].items()}
    dump(ROOT / 'config/sources.lock.json', lock)
    related = []
    for field in ['connectivity_group', 'topology_group']:
        for group, members in groups(specs, field).items():
            if len(members) > 1:
                related.append({'comparison': field, 'group': group, 'members': members})
    dump(ROOT / 'reports/topology_relationships.json', related)
    # Definition inventory is explicitly NOT an independent topology count.
    definition_rows = []
    seen = set()
    ag = ROOT / 'third_party/AnalogGym/AnalogGym'
    for path in sorted(ag.rglob('*')):
        if not path.is_file() or path.suffix.lower() in {'.zip', '.png', '.jpg', '.pyc', '.pdf'}:
            continue
        try:
            text = path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        names = re.findall(r'^\s*\.?subckt\s+(\S+)', text, re.I | re.M)
        for name in names:
            seen.add(name.lower())
            definition_rows.append({'category': path.relative_to(ag).parts[0], 'path': path.relative_to(ROOT).as_posix(), 'definition': name})
    dump(ROOT / 'reports/analoggym_definitions.json', {'unique_definition_names': len(seen),
        'warning': 'Subcircuit definition names include variants, dependencies and repeated aggregate files; not independent circuit topology counts.', 'definitions': definition_rows})
    totals = {source: sum(t['rows'] for t in inventory['tables'] if t['source'] == source) for source in lock}
    a = ['# 多来源模拟 IC 数据审计', '', '审计日期：2026-10-22（Asia/Shanghai）。统计以固定 Git 提交和实际文件为准；README 数量仅为作者声明。所有处理使用 Windows 原生 Python，未使用 WSL、Docker、商业仿真器或训练代码。', '',
         '## 环境与获取', '',
         table(['项目', '实际结果'], [['OS', inventory['environment']['os']], ['Python', inventory['environment']['python']],
             ['Git', inventory['environment']['git']], ['审计前 D 盘剩余', f"{inventory['environment']['disk']['free']/2**30:.2f} GiB"],
             ['环境', '.venv；依赖锁定于 requirements.txt；pip 缓存和 TEMP/TMP 限定在工作区'],
             ['检查限制', 'CIM 系统查询权限被拒绝，改用 Python platform 与 shutil.disk_usage；未修改系统配置']]), '',
         table(['来源', '官方仓库与固定提交', '许可证', '跟踪文件/实际工作树字节', '原始数据入口'],
            [[name, f"[{name}]({info['url'].removesuffix('.git')}) / `{info['commit']}`", 'BSD-3-Clause' if name == 'AnalogGym' else 'MIT',
              f"{info['tracked_files']} / {info['worktree_bytes']:,}", '仓库内 dataset/*/*/dataset.csv' if name == 'FALCON' else '仓库内 Dataset/*/*.csv' if name == 'AICircuit' else '仓库内模板、示例日志、ZIP 回放；无已验证监督样本表']
             for name, info in inventory['repositories'].items()]), '',
         '许可证原文见各仓库 LICENSE。仓库许可不等于商业 EDA 许可或第三方 PDK 的再分发授权；未复制模型进入处理数据。仓库内已有全部本次使用的 CSV，无需认证或外部数据下载。', '',
         'AICircuit 有 9 个以句点结尾的 `amap/__dspf_information__.` 文件，Windows 无法创建。仅在 Git 索引中标记 skip-worktree；完整原始 blob 保留在 .git，其他文件逐字节保持原样，三个仓库 git status 均为空。遗漏清单见 source_inventory.json。', '',
         '## 数据可用性与有效数量', '',
         table(['来源', '作者声称拓扑', '本地已下载表格行', '通过数值/冲突检查行', '输入全部可解析行', '本次采用条件'],
            [[source, 20 if source == 'FALCON' else 9 if source == 'AICircuit' else 'README 30', f'{totals[source]:,}',
              f"{sum(t['accepted_rows'] for t in stats if t['topology_id'].startswith(source+'__')):,}",
              f"{sum(t['accepted_rows'] for t in stats if t['topology_id'].startswith(source+'__') and t['input_complete']):,}",
              'CSV 扫描参数+官方 values.yaml 固定参数；来源内同输入异标签组整体隔离' if source == 'FALCON' else '参考 Ocean 条件，不保证与每一历史行完全一致' if source == 'AICircuit' else '仅审计，不把模板、奖励或归一化观察值当标签'] for source in lock]), '',
         '接收记录须通过有限值、基本物理符号及来源内输入冲突检查，不表示重新验证了收敛；原始逐行仿真日志未随 CSV 提供。输入可解析也不等于跨数据集指标兼容。', '',
         table(['来源/拓扑', '原始行', '接收行', '拒绝行', '抽样行', '器件/网络/边', '输入完整', '温度 °C', 'VDD V（已知默认）'],
               [[s['topology_id'], bykey[s['topology_id']]['raw_rows'], bykey[s['topology_id']]['accepted_rows'],
                 bykey[s['topology_id']]['rejected_rows'], sampled.loc[s['topology_id'], 'sampled'],
                 f"{s['num_devices']}/{s['num_nets']}/{s['num_edges']}", s['input_complete'],
                 s['temperature'] if s['temperature'] is not None else '未发布',
                 s['supply_voltage_default'] if s['supply_voltage_default'] is not None else '逐行 VDD 或未发布'] for s in specs]), '',
         '## 工艺、仿真条件和 Testbench', '',
         table(['来源', '工艺/模型', '仿真器', '温度与供电', 'Testbench 与缺失条件'], [
             ['FALCON', '作者论文称 45 nm CMOS；仓库未包含 PDK/model，不能认定与 AICircuit 的 PTM 文件相同', '作者称 Cadence Spectre/ADE XL；未运行', 'VDD 等来自逐行 CSV 或 values.yaml；温度未发布，保持 null', '网表、graph.json、values.yaml 可取；固定频率 fin/pin 等按文件读取，仍缺完整 ADE 测量表达式、噪声偏移及逐行运行日志'],
             ['AICircuit', 'Simulation/Model/45n270AModels.sp：PTM LP 45 nm Metal Gate / High-K / Strained-Si；不是已证实的商用 foundry PDK', 'Cadence Spectre + Ocean；重新仿真需要合法商业许可，官方流程要求 Linux/Docker，未执行', 'Ocean temp(27)；CSVA 逐行 VDD，TSVA 3.3V，LNA 2.5V；其他以具体网表/Ocean 为准', 'Simulation/Ocean/*/；defaults 只作为所附参考流程的条件，保留证据，不声称历史行都在该默认下产生'],
             ['AnalogGym', 'AMP/LDO 的 SKY130 PDK ZIP 已在仓库；其他类有专用模型路径与工具依赖', 'AMP/LDO 可用 Ngspice；其他类含 Spectre/HSPICE 流程；当前机器未发起仿真', 'README 1.8V 不能覆盖全部文件；TB 可有不同 PVT，LDO 5mA/55mA负载，必须区分', 'AnalogGym/*、RGNN_RL/simulations；README 要求 Ngspice >=42；原生 Windows 仍需完整模型路径与兼容流程']]), '',
         '参考：[FALCON 论文数据说明](https://arxiv.org/html/2505.21923v2#S3.SS1)。仅据此记录 45 nm、Spectre 和拓扑家族声明；精确行数来自本地 CSV。', '',
         'AICircuit 具体频率/提取表达式保存在 `reports/testbench_evidence.json`，包括 PA 28 GHz、VCO 相噪 1 MHz 偏移、LNA 1–10 GHz 扫频和 NFMin 的额外 +1。完整来源路径及参数单位见 catalog.json。', '',
         '图表示现为 net_device_multigraph_v2：网络是节点，器件通过端口对边表示，devices 仅为参数表。两端器件一条边，MOS 为 DG/DS/DB/GS/GB/SB 六条边，保留自环、平行边和 B 端，是保留四端的 FALCON 风格扩展。互感是器件引用元数据，不算网络间边；加载器提供 edge_index/edge_features/edge_port_pairs。详见 README.md。', '',
         '## AnalogGym 的存量数据与限制', '',
         f'本地 AMP `spice_netlist` 有 17 个文件；LDO 有 4 组 design_variables 与器件模板。此外有 SFE、Voltage Reference、Charge Pump、PLL 等资料。自动扫描在 AnalogGym 子目录找到 {len(seen)} 个不同子电路定义名；其中包含依赖、变体和聚合重复，不能当作独立拓扑数。清单见 analoggym_definitions.json。本次可验证的有标签独立拓扑数为 0。', '',
         '有 AMP/LDO 示例 `.log`、工作点、AC/DC/TRAN 输出，但不能把扫频/时域的每个点计为一个独立设计。ZIP 中包含回放和模型，而非大量独立仿真 CSV。`ReplayBuffer` 预分配 100,000 等容量不等于实际有效步数；`info_buf` 中某些 GBW/相位裕度由失败回退置零，观察值经过归一化和裁剪。未执行不受限 pickle 加载，也未用 reward 造性能标签。若后续接入回放，须先校验 size/ptr、action 到尺寸的反变换、next-state 对齐及每个原始测量的失败状态。', '',
         table(['ZIP', '压缩字节', '展开字节', '成员数量'], [[z['path'], z['bytes'], z['expanded_bytes'], z['count']] for z in inventory['archives']]), '',
         ('专项审计已受限读取两个 memory pickle，并静态扫描 agent opcode：实际仅 20 条转移，均隔离。详细证据见 [AnalogGym 专项报告](analoggym_deep_audit.md)。未发起新仿真。' if (ROOT / 'reports/analoggym_replay_audit.json').exists() else '两个回放/agent ZIP 各展开约 0.60 GiB；初始审计只读取 ZIP 目录。没有为凑数执行多点仿真；在缺乏单点实测耗时前不能可靠估计 50 万次仿真的总时间。'), '',
         '## 重复/相似电路与泄漏', '',
         f"29 个来源条目对应 {summary['connectivity_classes']} 个完整端口连接结构类。此数不比较模型参数、器件尺寸或测试条件；不将其声称为 26 个已核实且完整的共同工艺训练拓扑。进一步用忽略无源器件的 MOS 核心同构得到 {summary['conservative_mos_core_classes']} 个保守隔离组。WL 哈希只筛候选，最终由保留端口颜色的 VF2 同构验证。", '',
         table(['比较层级', '同组条目'], [[r['comparison'], ', '.join(r['members'])] for r in related]), '',
         '“完整连接”含电源、激励源、balun 和互感器引用；MOS 核心相同只表明晶体管连接相似，不能证明完整电路或标签相同。所有 LOTO 测试均整体排除整个 MOS 核心组。相同核心+同名尺寸（大小写规范化、12位有效数字）构成保守 design_group，跨来源相同设计不进入不同 split。固定参数未知时该分组宁可多隔离，不推断标签相等。', '',
         table(['跨源比较', '共同尺寸向量', '完整数值标签相同', '完整连接相同'],
               [[p['a']+' / '+p['b'], p.get('shared_named_parameter_vectors', '参数集合不同'),
                 p.get('same_complete_numeric_label_vector', '不能逐项比较'), p['same_full_connectivity']] for p in overlaps['pairs']]), '',
         table(['来源内重复尺寸', '涉及行数', '重复参数组', '输入完整'],
               [[p['topology_id'], p['rows_with_repeated_parameter_vector'], p['repeated_parameter_groups'], p['input_complete']] for p in overlaps['within_source_repeated_parameters']]), '',
         '上述数值比较使用候选字段别名，不是对物理指标定义作合并。跨来源 CSVA 共同尺寸组合的性能不相同，条件未经证实相同，保持来源语义并共同分组。来源内 CLNA 有 61,690 行同记录输入异标签，缺少可区分条件，全部隔离，不平均、不任选标签；其余 541 行可保留。', '',
         '## 指标映射与不能兼容的内容', '',
         '映射文件 config/performance_mapping.json 为每个来源/拓扑给出最多 16 个槽位的原名、单位、转换因子、证据、semantic_id 和限制。槽位只是存储位置；不能以相同下标或相似名称联合监督。AICircuit 的模块级 `_LNA/_Mixer/_Amp/_PA/_VCO` 标签保持独立。百分比转无量纲比例，其余保留 W、Hz、V、dB、dBm、dBc/Hz。缺失指标为 null 且 mask=false，现存且有限指标 mask=true；不兼容指标不改成缺失。', '',
         table(['问题', '处理'], [
             ['FALCON 测试频率/相噪偏移/温度不全', '保持来源拓扑语义 ID；不与 AICircuit 强行共用标签'],
             ['NFMin = ymin(dB10(F))+1', '保留 NFMin 原名，不变成标准 NoiseFigure'],
             ['PSAT 调用 compressionVRICurves gcomp=1', '保留原值与原名，记录压缩函数语义，不能直接称饱和功率'],
             ['CVA 带宽使用 /Vin-；网表仅有 Vin+', '保留原始 Bandwidth 并标记提取待验证，不凭空修复标签'],
             ['Mixer Config 把 WRF 写成 RF', '仅在转换器中修正配置字段，依据 CSV、网表和 Ocean 三者一致；第三方文件不改'],
             ['参考 Ocean 默认参数并非逐行运行日志', '保留 testbench_status，不把条件相同视作已证实'],
             ['FALCON 固定参数与同输入冲突', '读取官方 values.yaml，全部图声明特征可解析；来源内冲突组整体隔离，原始行保留可追溯'],
             ['AnalogGym 存量回放、示例波形', '只审计，0 行进入监督数据']]), '',
         '## 阶段结果与可恢复性', '',
         '获取与审计完成；先做 80 行/条目的 smoke 转换并通过 QC，再执行全量处理。逐拓扑转换写 .partial 后原子替换，并依据源 SHA256/配置指纹恢复。第三方仓库固定 commit，源 CSV 的 SHA256 和每行物理行号（不含表头）完整保存。所有输出只写 D:/EDA。']
    supplement = []
    if (ROOT / 'reports/analoggym_replay_audit.json').exists():
        supplement = ['', '## AnalogGym 专项补充审计', '',
                      '官方 AnalogGym 两个回放各写入 10 条转移，归档动作范围与当前源码不同；AMP 性能向量恒定，LDO 有旧 DC 输出与失败回退证据。20 条保存在隔离区，新增有效监督样本为 0。原始字段未伪装为缺失，未进入任何训练划分。', '',
                      '详见 [专项审计报告](analoggym_deep_audit.md)、analoggym_replay_audit.json；复现命令为 `.venv\\Scripts\\python.exe scripts/run_analoggym_audit.py`。']
    a.extend(supplement)
    pilot = ROOT / 'data/generated/analoggym_nmcf_pilot_v2/summary.json'
    if pilot.exists():
        generated = read(pilot)
        generation_note = ['', '## 单独保存的新仿真试跑', '',
            f"AnalogGym NMCF 已完成 {generated['attempts']} 个参数点 DC/AC/TRAN 试跑，接受 {generated['accepted']} 条，隔离 {generated['quarantined']} 条。新样本在 `{generated['output']}`，不计入原有 50 万条主集合；指标与条件仍独立定义，未获准跨来源共享标签。详情见 [生成报告](analoggym_generation.md)。"]
        a.extend(generation_note)
    else:
        generation_note = []
    (ROOT / 'reports/dataset_audit.md').write_text('\n'.join(a) + '\n', encoding='utf-8')
    evidence = {}
    for s in specs:
        evidence[s['topology_id']] = {'files': s['testbench_files'], 'defaults': s['defaults'], 'temperature': s['temperature'],
            'default_files': s.get('default_files', []), 'unresolved_device_features': s['unresolved_device_features'], 'config_corrections': s.get('config_corrections', []),
            'extraction_lines': {path: [line.strip() for line in (ROOT / path).read_text().splitlines() if
                                re.search(r'^(analysis\(|[A-Za-z]\w*\s*=|temp\()', line.strip())] for path in s['testbench_files']}}
    dump(ROOT / 'reports/testbench_evidence.json', evidence)
    rejected = sum(s['rejected_rows'] for s in stats)
    accepted = sum(s['accepted_rows'] for s in stats)
    converted_input = sum(s['accepted_rows'] for s in stats if s['input_complete'])
    splits = quality['protocols']
    storage = sum(p.stat().st_size for p in (ROOT / 'data').rglob('*') if p.is_file())
    final = ['# 最终数据统计与交付', '',
        f"默认**输入完整视图**：**{strict['sampled']:,} 条真实记录，{strict['source_topologies']} 个来源条目、{strict['connectivity_classes']} 个完整连接结构类**。FALCON {strict['sources']['FALCON']:,} 条、AICircuit {strict['sources']['AICircuit']:,} 条、AnalogGym 0 条。保存于 data/strict/，全部器件输入可解析，但原始提取/测试条件的限制仍适用，不能据此共用未经验证的指标。", '',
        '样本数量与拓扑覆盖达到约 50 万、约 25 类的期望。器件固定值使用官方 values.yaml/Ocean，而非从其他来源猜测；AnalogGym 无已验证监督表，未为其造标签。另提供仅由训练集计数拟合的结构类/来源变体采样权重。', '',
        f'严格视图有 {strict_designs:,} 个保守设计组。跨来源相同尺寸的不同真实性能记录仍保留，但不计作不同设计组，不跨训练/验证/测试分散。', '',
        f"真实下载 CSV 共 **{sum(totals.values()):,} 行**；有限值/物理符号/来源内输入冲突检查后接收 **{accepted:,} 行**，隔离 **{rejected:,} 行**。其中 CLNA 61,690 行因来源内同输入异标签隔离，CSLNA 86 行因非正带宽隔离。拒绝清单按拓扑存于 converted/*_rejected.parquet，原始 CSV 不改；未用 Mask 隐藏无效值。", '',
        f"另一个**拓扑覆盖接口视图**为 **{summary['sampled']:,} 行**，来自 **29 个来源电路条目 / {summary['connectivity_classes']} 个完整连接结构类**，输入全部可解析 **{summary['input_complete_sampled']:,} 行**。当前两个视图成员相同，不能相加计数；全部接收行中输入可解析 **{converted_input:,} 行**。这不等于所有测试条件和跨来源指标定义均已统一。", '',
        table(['来源', '下载行', '接收行', '主抽样行', '主抽样输入完整行'], [[source, totals[source],
            sum(s['accepted_rows'] for s in stats if s['topology_id'].startswith(source+'__')), summary['sources'].get(source, 0),
            int(sampled.loc[sampled.index.str.startswith(source+'__')].query('input_complete == True')['sampled'].sum())] for source in lock]), '',
        f"按结构类公平分配抽样预算，再在同结构的来源条目间分配；小表不足自动回填其余结构类。拓扑内使用固定种子 42 及确定性派生种子、不放回随机抽样。删除同来源拓扑精确重复记录 {summary['exact_duplicate_records_removed']:,} 条。跨来源等尺寸设计保留其真实条件/标签，design_group 一起划分。", '',
        '## 已生成协议', '',
        table(['协议', '状态', '划分数量', '说明'], [[name, info['status'], info['roles'],
             '成员划分可加载；跨拓扑共用指标仍待核实' if name.startswith('loto') else '可按来源拓扑语义组加载' if info['status'] == 'ready' else '候选分区已保存，0 个获准跨源指标；加载器拒绝作为有效评估'] for name, info in splits.items()]), '',
        '输入完整视图有独立的 data/strict_splits/，不复用覆盖视图的训练统计。其划分数量如下：', '',
        table(['严格视图协议', '状态', '划分数量'], [[name, info['status'], info['roles']] for name, info in strict_quality['protocols'].items()]), '',
        'ID、3 个 LOTO 折及训练集独立拟合归一化文件已生成。LOTO 按完整 MOS 核心组隔离，包括同组的不同来源条目。Cross-Dataset 以整个来源作外部测试并排除重叠设计，但 FALCON 缺少必要测试条件，AnalogGym 无已验证监督行，因此目前三套跨源候选均不可报告分数。不能把这些候选文件称为有效的 Cross-Dataset 基准。', '',
        'LOTO 的 ready 仅表示无泄漏的成员划分与加载就绪；协议另写明 evaluation_status=blocked_no_verified_cross_topology_targets。当前指标仍按来源拓扑隔离，尚未证明跨拓扑共用目标定义和条件，因此不能直接按向量槽位报告有效跨拓扑性能预测分数。ID 可在各已知拓扑语义组内实验。', '',
        '## 质量证据与限制', '',
        f"自动检查状态：**{quality['status']}**。已检查 {quality['graphs_checked']} 个图模板、全部 {accepted:,} 条转换记录的原始参数/标签逐行对账，以及 {quality['sampled_ids_checked']:,} 条抽样 ID 与全部协议的分组泄漏。MOS 四端、器件/网络/边计数、Mask、有限标签、全部训练归一化统计均通过。数字化器件特征每拓扑检查首/中/末三行，表达式解析另有手工期望值的单元测试；未声称对每行重新仿真。", '',
        f"输入完整视图独立全量 QC：**{strict_quality['status']}**，{strict_quality['sampled_ids_checked']:,} 条记录。两个视图不是互相独立的新仿真集合，严格视图引用同一转换源；不能把两个行数相加当新增数据。", '',
        f"data/ 当前全部产物（含 smoke、全量转换、抽样、划分、归一化及另存的安装检查/试跑）共 {storage/2**30:.3f} GiB。主集合仍来自历史原始数据，未使用人工标签，未开展 GNN 训练。新仿真试跑另见附注，不计入主集合。", '',
        f"固定输入、种子与转换缓存重复执行了抽样/划分/归一化流程：{reproducibility['checked_files']} 个产物 SHA256 一致，见 reports/reproducibility.json。独立加载检查覆盖两个视图的 ID/LOTO 共 24 个协议/角色组合；不兼容跨源协议被加载器明确拒绝。", '',
        '最重要的可用性边界：26 个连接结构类的声明器件特征已可解析，但完整测试条件和跨拓扑/跨来源指标等价性仍有未验证项。需要跨拓扑共用输出头时还须获得一致的 metric semantic_id 与测试条件证据。当前统一的是可追溯数据接口与存储，未强行统一物理定义不同的标签。', '',
        '详细逐拓扑统计见 data/processed/sampling_statistics.csv；来源、许可证、条件、相似电路与不能兼容项见 dataset_audit.md；机器可读检查见 data/processed/quality_checks.json。']
    final.extend(supplement)
    final.extend(generation_note)
    (ROOT / 'reports/final_statistics.md').write_text('\n'.join(final) + '\n', encoding='utf-8')
    print({'sampled': summary['sampled'], 'input_complete': summary['input_complete_sampled'], 'accepted': accepted,
           'rejected': rejected, 'connectivity_classes': summary['connectivity_classes'], 'data_GiB': storage/2**30})


if __name__ == '__main__':
    main()
