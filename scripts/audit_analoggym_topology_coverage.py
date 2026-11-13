"""Extract source assets for review; never count these as simulated samples."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analoggym_campaign import UPSTREAM, catalog, read, sha, rel, atomic_json


def main():
    destination = ROOT / 'data/audit/analoggym_topology_assets'
    destination.mkdir(parents=True, exist_ok=True)
    source = UPSTREAM / 'AnalogGym/Voltage Reference/Description of  Voltage References  in AnalogGym.md'
    references = []
    title = None
    opened = None
    blocks = []
    lines = source.read_text(encoding='utf-8').splitlines(keepends=True)
    for number, line in enumerate(lines, 1):
        if line.startswith('### '):
            title = line[4:].strip()
        if line.lstrip().startswith('```'):
            if opened is None:
                opened = number
            else:
                text = ''.join(lines[opened:number-1])
                target = destination / f'reference_block_{len(blocks)+1:02d}.spice'
                target.write_text(text, encoding='utf-8')
                blocks.append({'title':title, 'source':rel(source), 'start_line':opened+1,
                               'end_line':number-1, 'extracted':rel(target), 'sha256':sha(target),
                               'model_import_lines':[s.strip() for s in text.splitlines()
                                                     if s.lstrip().lower().startswith(('.lib ', '.inc '))]})
                opened = None
    assert opened is None
    for title in dict.fromkeys(b['title'] for b in blocks):
        references.append({'name':title, 'blocks':[b for b in blocks if b['title']==title],
                           'status':'requires_missing_PDK_or_validated_process_port'})

    sensing_source = UPSTREAM / 'AnalogGym/Sensing Front End/topology.txt'
    text = sensing_source.read_text(encoding='utf-8')
    sensing = []
    for i, match in enumerate(re.finditer(r'^subckt\s+(\S+)[^\n]*\n.*?^ends[^\n]*(?:\n|$)', text, re.M|re.S), 1):
        target = destination / f'sensing_block_{i:02d}.spectre'
        target.write_text(match[0], encoding='utf-8')
        sensing.append({'name':match[1], 'source':rel(sensing_source),
                        'start_line':text.count('\n',0,match.start())+1,
                        'extracted':rel(target), 'sha256':sha(target),
                        'status':'Spectre_syntax_and_missing_process_models'})

    entries, excluded = catalog()
    passed = {r['topology'] for r in read(ROOT/'data/generated/analoggym_standard_v4/summary.json')['results'] if r['accepted']}
    inventory = [{'name':e['name'], 'family':e['family'], 'source':e['dut'],
                  'source_sha256':e['dut_sha256'], 'topology_hash':e['graph']['topology_hash'],
                  'qualified':e['name'] in passed} for e in entries]
    result = {'source_commit':'0a9d1390ade361e2b4a2d33181e22367edbb8afc',
              'official_suite_topology_claim':30, 'official_family_claim':5,
              'open_model_candidates':len(entries), 'qualified':len(passed),
              'qualified_families':sorted({e['family'] for e in entries if e['name'] in passed}),
              'inventory':inventory, 'excluded':excluded,
              'voltage_references':references, 'sensing_subcircuits':sensing,
              'sensing_unique_subcircuit_names':len({s['name'] for s in sensing}),
              'caution':'Extracted blocks, subcircuit names, official benchmark topologies and validated topologies are different counts. Extracted assets are not runnable dataset samples.',
              'source_sha256':{rel(source):sha(source),rel(sensing_source):sha(sensing_source)}}
    atomic_json(ROOT/'reports/analoggym_topology_coverage.json',result)
    lines = ['# AnalogGym 拓扑及五类覆盖核对', '',
             '官方 README 声明套件有 30 个拓扑、五大类，同时明确放大器和 LDO 支持 ngspice/SKY130。这个声明不等于当前环境中已有 30 个可运行、可提取有效标签的电路。', '',
             '| 类别 | 本地实际材料 | 本批进入批量仿真 | 其余工作的缺口 |',
             '| --- | --- | ---: | --- |',
             '| 放大器 | 16 个非空 SKY130 网表；Qu_LEC 网表为空 | 16 | 空网表不能仿真 |',
             '| LDO | 5 个 SKY130 候选结构 | 2 | Basic_LDO、ldo_1、ldo_2 的试跑仍有振荡或调节异常 |',
             f'| 电压参考 | Markdown 内实际包含 {len(references)} 组电路、{len(blocks)} 个代码块 | 0 | 作者明确省略保密 PDK；需合法取得模型或经过验证的工艺移植 |',
             f'| 传感前端 | topology.txt 含 {len(sensing)} 个子电路定义、{len({s["name"] for s in sensing})} 个不同名称，另有测试平台 | 0 | Spectre 语法，nch_mac/pch_mac 等模型；测试平台引用 TSMC180HV；部分器件涉及特殊阈值 |',
             '| PLL | Cadence 工程归档、OCEAN 脚本及 VCO 测试 | 0 | 依赖 Spectre、GPDK045 和电感模型，当前未形成可用 ngspice 电路 |', '',
             'Charge Pump 目录另有 HSPICE 网表，引用 SMIC40 模型。不能把一个电荷泵模块算成完整 PLL 来满足类别覆盖。', '',
             '传感子电路包含变体和重复名称；没有把这些定义数相加冒充官方的 30 个独立基准。官方套件口径、源文件资产数、端口连接结构去重数、成功仿真数应分开报告。', '',
             '## 当前 21 个开源模型候选', '',
             '| 拓扑 | 类别 | 当前资格验证 |', '| --- | --- | --- |']
    for e in inventory:
        lines.append(f'| {e["name"]} | {e["family"]} | {"通过" if e["qualified"] else "未纳入，保留调试证据"} |')
    lines += ['', '## 电压参考的原始资产', '',
              '此前将电压参考描述成“只有说明文档”不准确。文档内确有网表与测试代码，真正缺的是相应工艺模型和可验证的运行条件。已逐块提取为审计副本，不修改上游，也不计入有效样本。', '']
    for ref in references:
        ranges = ', '.join(f'{b["start_line"]}–{b["end_line"]}' for b in ref['blocks'])
        lines.append(f'- {ref["name"]}：原文行 {ranges}。')
    lines += ['', '逐块路径、哈希和行号见 `reports/analoggym_topology_coverage.json`；副本位于 `data/audit/analoggym_topology_assets/`。', '',
              '本批优先用已通过验证的 18 个拓扑生成独立数据，覆盖两类。补齐五类需要另行完成工艺模型与测试平台移植；本批不宣称已实现五类覆盖。', '',
              '## 原始证据', '',
              '- `third_party/AnalogGym/README.md`：30 拓扑、五类及开源支持范围。',
              '- `third_party/AnalogGym/AnalogGym/Voltage Reference/Description of  Voltage References  in AnalogGym.md` 第 9 行：保密 PDK 路径省略说明。',
              '- `third_party/AnalogGym/AnalogGym/Sensing Front End/TB_2T_sensor_core.sp`：TSMC180HV 模型引用。',
              '- `third_party/AnalogGym/AnalogGym/Phase-Locked Loop/circuit/pll_vco.ocn`：Spectre/GPDK045/电感模型引用。',
              '- `third_party/AnalogGym/AnalogGym/Charge Pump/circuit/tran_27corner.sp`：SMIC40 模型引用。']
    (ROOT/'reports/analoggym_topology_coverage.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('open_model_candidates','qualified','qualified_families','sensing_unique_subcircuit_names')},ensure_ascii=False))
    print({'reference_circuits':len(references),'code_blocks':len(blocks),'sensing_blocks':len(sensing)})


if __name__ == '__main__':
    main()
