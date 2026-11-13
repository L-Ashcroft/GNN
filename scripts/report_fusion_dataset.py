"""Generate one consolidated report from actual build and audit artifacts."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read,dump


def build():
    base=ROOT/'data/fusion_v2';m=read(base/'manifest.json');q=read(base/'quality_checks.json')
    assert m['state']=='complete' and q['status']=='passed'
    acquisition=read(ROOT/'data/raw/LOADBench/acquisition_summary.json')
    catalog=read(base/'catalog.json');stats=read(base/'sampling_statistics.json')
    lb=[s for s in catalog if s['source_dataset']=='LOADBench']
    raw=[read(ROOT/s['acquisition']) for s in lb]
    checkpoints=ROOT/'data/raw/LOADBench'
    record=read(checkpoints/'record_21759759.json')
    lock={k:v for k,v in read(ROOT/'config/sources.lock.json').items() if k in {'FALCON','AICircuit'}}
    code_path=ROOT/'third_party/LOADBench-Scripts'
    git=['git','-c','safe.directory='+code_path.as_posix(),'-C',str(code_path)]
    lock['LOADBench']={'record':'https://zenodo.org/records/21759759','doi':record['doi'],
        'record_json_sha256':hashlib.sha256((checkpoints/'record_21759759.json').read_bytes()).hexdigest(),
        'license':'CC-BY-4.0','files':record['files'],
        'official_code':'https://github.com/LOADBench/LOADBench-Scripts',
        'official_code_commit':subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()}
    # Record the actual configured origin rather than guessing organization names.
    lock['LOADBench']['official_code']=subprocess.check_output(git+['remote','get-url','origin'],text=True).strip()
    dump(ROOT/'config/fusion_v2_sources.lock.json',lock)
    lines=['# FALCON / AICircuit / LOADBench 新数据集',
        '',f"构建与质量检查已完成：**{m['sample_count']:,} 条真实来源记录**，随机种子 42。输出 `data/fusion_v2/`。",
        '原有 `data/processed/`、`data/strict/` 及 AnalogGym 数据均保留；这是一套新的抽样结果，不能与旧版本相加当作独立样本。','',
        '| 来源 | 样本数 | 有样本的来源拓扑条目 | 完整端口连接结构类别（来源内） |',
        '| --- | ---: | ---: | ---: |']
    for source,count in m['source_counts'].items():
        ss=[s for s in stats if s['source_dataset']==source and s['sampled']]
        lines.append(f"| {source} | {count:,} | {len(ss)} | {len({s['connectivity_group'] for s in ss})} |")
    lines+=['',f"总计 {m['sampled_topology_entries']} 个有样本的来源拓扑条目，{m['full_port_structure_classes']} 个完整端口连接结构类。条目数与去重后的结构数不同。",
        '结构分类使用模型类型、所有端口角色及连接关系的 WL 候选分桶，再以 VF2 精确同构确认；不根据拓扑名称计数。',
        'FALCON/AICircuit 延续原已审计的网表范围；LOADBench 仅取 DUT，测试电路作为条件保留。结构同构不等于器件尺寸、功能或测试条件相同。','',
        '## 下载与覆盖',
        '',f"官方记录提供 101 个拓扑：单级 31、两级 44、三级 18、对称 8。全部 101 个拓扑的网表、参数约束/匹配、提取脚本及图模板已收集。",
        f"本次 LOADBench 成功采集的字节数（含每拓扑元数据开销）约 **{acquisition['network_bytes']/2**20:.1f} MiB**；另有约 1 MiB 的模型/测试平台完整小文件及最初目录探测。中断或重试的网络开销未全部计入。",
        '完整发布包约 36 GiB；没有下载完整包，也没有重复下载额外的 15 拓扑 benchmark 子集。',
        f"{sum(r['pof_rows']==0 for r in raw)} 个拓扑未提供可读取的最终 POF 表（空表也计入）；有表但指标失败的记录仍会剔除。",
        '每拓扑采用完整 `FinalPOFfull.csv` 加自适应 256 KiB–8 MiB 压缩历史前缀。候选量达标便停止，因此不同拓扑的下载量不同。',
        '**抽样总体是已下载的最终结果及早期历史，不是完整优化过程。无法声称覆盖整个百万次优化记录的参数分布。**',
        'HTTP Range 必须返回准确的 206、Content-Range 与字节数，否则拒绝下载；完整小成员验证 CRC32，保存 SHA256。历史前缀无法验证完整 ZIP CRC，保留原压缩字节、偏移及 SHA256 供追溯。','',
        '## 分层抽样与质量规则','',
        '先固定来源配额 20万/10万/20万，再按电路功能均衡分配，功能内按实际完整端口结构分配，结构内再分配来源拓扑条目；不足的层按真实容量重新分配。',
        '每拓扑用器件设计参数的百分位秩、三组固定种子投影及四分位单元描述联合分布，覆盖非空单元后尽量按候选池比例无放回抽样。性能标签不参与抽样。',
        'LOADBench 的 W/L 从原 CSV 的微米转为米，电容保持 F，匹配别名按 params.cir 解析，保存原始 CSV 和表达式。原有两个来源的 SI 参数延续已有审计；字段单位逐样本保存，未确认单位明确标记。',
        '部分官方参数模板包含 ERROR 占位符，必须由对应真实 CSV 的参数完整覆盖后才接收记录，未生成替代尺寸；零补偿电容允许表示关闭该补偿支路，图仍保留其零值器件。',
        '去除非有限参数/标签、±9999999999999 及其舍入值等失败哨兵、无效区间。相同拓扑及设计参数去重；重复设计的公共指标差异超过 0.2% 加指标绝对容差则整组隔离。轻微历史舍入差异优先保留最终 POF 原精度，不平均或拼接标签。',
        '`feasible=False` 表示未满足优化约束，不单独作为仿真失败依据；真实数值有效但性能不达标的设计也保留该状态。','',
        '## 标签与统一接口','',
        '统一 Parquet 保存参数、最多 16 维性能向量、Mask、语义 ID、原始标签和行号；固定图按拓扑单独保存 JSON。网为节点、器件按端口对展开为边，MOS 保留 G/D/S/B 六对连接和短接端口自环。',
        '图已增加电源/地等角色及证据、节点器件连接度/端口计数、双向消息传递边、40 种有向端口关系、W/L、显式 nf（未知不填）、器件类型与层次深度。物理边保持原样，API 的 edge_index 返回双向消息边；详见 [讲义第 12 章图审计](graph_lecture12_audit.md)。原始 SI 值和无拟合的对数变换同时提供，零值有单独标志，未拟合全数据标准化。',
        'LOADBench 的 14 个指标包括低频增益、单位增益交越频率、功耗、相位裕度、正负压摆率、失调、电压范围、解析面积估计、CMRR、PSRR。负压摆率保持原有负号，解析面积不冒充实际版图面积。',
        '最终 POF 未提供 CMRR/PSRR，故对应 Mask=False；历史有这些指标才标 True。失败指标对应的整条记录已剔除，Mask 只表示未提供。',
        '**相同槽位不等于相同物理标签。** `performance_mapping.json` 的语义 ID 按来源、测试平台和定义区分。尚未批准跨来源共享标签；可使用统一接口和模型骨干加语义分组输出头，不能把所有第 0 维直接当成同一种增益。',
        'LOADBench 为 reduced SKY130 LVT、TT、1.8 V；每拓扑实际测试条件、负载、偏置、测试平台/提取器哈希见 catalog。27°C 是未设 .temp 时的仿真器默认值，不能当作逐条历史温度日志。',
        '未拟合任何模型归一化参数，后续仅用训练集拟合。','',
        '推荐使用 `FusionDataset.supervision_batches()`，按来源、拓扑、完整指标语义和测试条件分组加载；混合组调用 `SupervisionRouter.batch()` 会报错。普通 `samples()` 只统一访问格式，不能直接按槽位堆叠计算公共损失。',
        '当前没有获批准的跨来源共享标签。可共享 GNN 骨干，但使用分组输出头、按训练集拟合的分组标签尺度和独立组损失，再明确设置组权重。Mask 只能排除组内未提供的值，不能把不同定义变成同一标签。',
        '全量监督契约检查：`scripts/check_fusion_supervision.py`，结果 `data/fusion_v2/supervision_audit.json`。','',
        '142 个保守监督组的逐组拓扑数、样本数与可合并集合，见 [监督组兼容性审计](fusion_supervision_groups.md)；可运行 `scripts/audit_supervision_groups.py` 复现。142 是当前包含拓扑 ID 的隔离规则产生的数量，不能直接解释为不可兼容任务的数量。','',
        '当前监督分组有意保守。逐拓扑输出头不能直接完成未知拓扑性能预测；LOTO 文件仅保证数据留出，目标对齐仍需另行审核。批准共享目标前，必须核对增益类型、开闭环、频率、差分定义、偏置、负载、工艺和提取方法等证据，单位同为 dB 不足以证明兼容。','',
        '```python',
        'from src.dataset_fusion import FusionDataset',
        'dataset = FusionDataset()',
        'sample = next(dataset.samples(source="LOADBench"))',
        'print(sample["edge_index"], sample["device_features"])',
        'print(sample["metrics_by_semantic_id"])',
        '```','',
        '## 划分与验证','',
        'ID 按 design_group 固定哈希进行约 80/10/10 划分；同设计不跨集合。LOTO 提供 5 个独立折，将完整端口结构及保守 MOS 核心相似组整体留出。',
        'Cross-Dataset 仅提供来源留出候选文件，协议明确标为不可进行可比标签评估：目前缺少已确认的跨来源同定义、同测试条件标签。',
        f"质量检查通过：{q['rows']:,} 条参数与标签逐条比对候选原表、唯一 ID、图与网表端口连接、Mask、来源数、抽样单元及划分泄漏；{q['historical_hashes_unchanged']} 个历史数据文件哈希保持一致。设备特征实体化每拓扑抽查，非逐条重跑仿真。",
        '完整机器检查结果：`data/fusion_v2/quality_checks.json`；逐拓扑计数：`sampling_statistics.csv`；候选清洗统计：`loadbench_candidate_quality.json`。','',
        '## 零有效样本的拓扑','']
    zero=m['zero_sample_topologies']
    lines+=['这些拓扑图已保留，但本次前缀和最终表没有通过质量检查的样本；没有复制其他拓扑或制造标签。'] if zero else ['全部来源拓扑均有有效样本。']
    lines += ['- `'+k+'`' for k in zero]
    lines+=['','## 复现','',
        '```powershell',
        r'.\.venv\Scripts\python.exe scripts/inspect_loadbench_remote.py',
        r'.\.venv\Scripts\python.exe scripts/acquire_loadbench_subset.py',
        r'.\.venv\Scripts\python.exe scripts/build_fusion_dataset.py',
        r'.\.venv\Scripts\python.exe scripts/split_fusion_dataset.py',
        r'.\.venv\Scripts\python.exe scripts/check_fusion_dataset.py',
        r'.\.venv\Scripts\python.exe scripts/report_fusion_dataset.py',
        '```','',
        '下载按 acquisition.json 断点恢复。已有 FALCON/AICircuit 原始与转换数据的复现方法见原 README。下载脚本需要联网，不需要商业仿真软件，也未新增仿真标签。',
        '官方来源与许可证：[LOADBench Zenodo 数据记录（CC BY 4.0）](https://zenodo.org/records/21759759)、[论文](https://arxiv.org/html/2610.21591v1)。另两个来源沿用原审计许可；合并不改变各来源许可，来源锁见 `config/fusion_v2_sources.lock.json`。']
    (ROOT/'reports/fusion_v2_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (base/'SOURCES.md').write_text('Source records and license evidence: see reports/fusion_v2_report.md and config/fusion_v2_sources.lock.json.\nLOADBench DOI: 10.5281/zenodo.21759759, CC BY 4.0. Preserve original record metadata (including creators), source CSV provenance and this attribution when redistributing.\nFALCON/AICircuit: original notices in third_party and reports/dataset_audit.md apply.\n',encoding='utf-8')


if __name__=='__main__':build()
