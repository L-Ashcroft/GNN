"""Render actual personal-campaign progress; never infer completion from target."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read
from scripts.generate_analoggym_personal import settings


def report():
    config,output=settings();manifest=read(output/'manifest.json');status=read(output/'status.json')
    anchors=read(output/'parameter_anchor_audit.json')['anchor_counts']
    snapshot=read(output/'dataset.json');recipe=read(output/'recipe.json')
    lines=['# AnalogGym 个人目标域数据集：约 2 万条、10 拓扑','',
        f"当前状态：**{status['state']}**，已提交真实尝试 **{status['attempts']}** 次，通过 **{status['accepted']}** 条；已导出 **{snapshot['sample_count']}** 条。",
        '目标 20,000 条有效且设计唯一的样本，每拓扑 2,000 条，最多 8 个原生 ngspice 并行进程。启动不等于完成，实时状态以输出目录的 status.json 为准，Parquet 导出快照最多约滞后一轮导出间隔。',
        '',f"独立目录：`{config['output']}/`。不合并到 fusion_v2，不复用历史性能值作为新标签。",'',
        '| 拓扑 | 类别 | 实测成功参数中心数 | 已接受 | 最终目标 |',
        '| --- | --- | ---: | ---: | ---: |']
    for entry in recipe['entries']:
        name=entry['name'];s=status['topologies'][name]
        lines.append(f"| {name} | {entry['family']} | {anchors[name]} | {s['accepted']} | 2000 |")
    lines+=['','## 对应的实验用途','',
        '讲义 4 §25.3 的后两种协议是跨数据集与少样本微调；讲义 5 §38–41 给出域调制、两阶段优化和少样本曲线分析。本数据集提供独立目标域，后续可比较仅公开数据、仅个人数据、公开预训练后微调及相关域适应对照。',
        '**零样本跨来源指标目前未获兼容批准。** 个人数据沿用经精度验证的 AnalogGym 衍生平台：放大器一般为 500pF/0.45V 共模/0.1Hz 起始 AC，LOADBench 已发布条件为 1pF/不同激励/1Hz 起始，FALCON/AICircuit 部分历史测试平台未发布。Gain、UGF 或功耗不能因同名、同单位就直接共用监督头。若需要可比零样本结果，必须先验证公共目标与测试条件，或另行重新仿真一致测试平台。',
        '这不妨碍生成目标域实测样本和研究共享主干的少样本迁移，但目标域输出头、实际测试条件、域表和必要的新 PDK 器件类型需要明确适配。已有 fusion_v2 预测器不会自动支持新的 AnalogGym 目标契约。','',
        '## 参数为何有依据','',
        f"共有 {sum(anchors.values())} 组不同的真实成功参数作为中心，来自已核验资格/精度检查和其他参数试跑；完整参数、源文件、别名表达式及归档路径见 parameter_anchor_audit.json。",
        '中心只决定尺寸采样范围，所有个人样本标签重新运行 ngspice 并从本次原始波形提取。中心的历史标签不计入样本数量。',
        '继续使用冻结的采样规则：20% 窄、60% 中等、20% 较宽局部对数扰动；W 为 0.42–100µm、L 为 0.15–30µm；补偿电容约中心的 ×1/1.35–1.35，偏置电压约 ±10%，m 离散取整。参数别名和共享尺寸保持共享，不独立打乱同一个匹配参数。种子 42 按拓扑和候选编号确定，可恢复。',
        '几何界限不等于所有组合都通过模型分箱和电路稳定性检查。无法收敛、无有效交越、跟踪/稳定窗口异常等记录均隔离并保留失败证据。接受集代表成功设计邻域，不代表整个设计空间或制造良率。',
        '先在同一生产目录生成每拓扑 3 条有效设计并深度重算波形标签；通过后只提高目标配额继续采样，首批真实样本保留且不重复计数。','',
        '## 仿真条件、存储与资源','',
        '原生 Windows ngspice 47、SKY130 TT，1.8V；放大器使用相同的 AC/DC/瞬态衍生平台，LDO 使用独立 1.2V 输出、1/10mA 和 100nF 测试条件。放大器 15 项指标、LDO 13 项指标放入 16 槽容器，未使用槽位 Mask=False；已定义指标失败时整条隔离。',
        '放大器温扫与 AC/TRAN 温度不同，顶层温度不伪装成单一值；具体条件随指标/拓扑保存。展开相位不是自动的相位裕度，电源传递也不冒充正向 PSRR。',
        '固定求解精度、归档 CRC/SHA 和 SQLite 唯一设计约束沿用已经验证的冻结代码。原仓库及旧 10 万条配方不改。',
        '预计一天量级，失败率和其他应用负载会改变实际时长；历史 8 并发 18 拓扑外推约 31.8 小时，当前 10 拓扑应以新进度估算，不能把小批次时间当作完成承诺。原始数据预计十几 GiB，设 40GiB 本批上限、20GiB 最低剩余磁盘。可用内存低于 512MiB 时停止提交并排空保存；不杀其他应用。',
        'samples/ 是真实接受样本 Parquet，topologies/ 是网表图/条件/指标映射，raw/ 保存每次输入、波形、日志及结果，ledger.sqlite 保存接受/失败及去重账本。全部完成后自动生成 personal_index.parquet 和 splits/。','',
        '## 评估划分','',
        '完成 20,000 条后按设计分组，每拓扑 1600/200/200 划分 train/validation/test；固定测试集不进入参数拟合或微调。',
        '生成两套明确区分的嵌套少样本预算：全目标域总 k=0/5/10/20/50/100/200/500/1000；以及每拓扑 k=5/10/20/50/100。后者总预算为 10k，不能冒称总共只用了 k 条。k<10 时总预算不能覆盖所有拓扑，文件记录实际覆盖。验证集所用目标域标签是额外预算，必须披露，不能声称完全零标签或只用了 k 条。',
        '不预先拟合任何归一化参数，不训练 GNN。','',
        '## 状态、停止和恢复','',
        '```powershell',
        r'.\.venv\Scripts\python.exe scripts/generate_analoggym_personal.py status',
        r'.\.venv\Scripts\python.exe scripts/generate_analoggym_personal.py stop',
        '# 资源恢复或明确要继续时，只移除此独立目录中的 STOP 文件',
        r"Remove-Item -LiteralPath 'D:\EDA\data\generated\analoggym_personal_20k_v1\STOP'",
        r'.\.venv\Scripts\python.exe -B -u scripts/run_analoggym_personal_pipeline.py',
        r'.\.venv\Scripts\python.exe scripts/report_analoggym_personal.py','```','',
        '同一目录使用原生文件锁，禁止两个生成器同时写入。锁随进程退出自动释放，不按旧 PID 擅自杀进程；关闭电脑/休眠会停止计算，已提交结果可继续。无新结果时 status 只是已有检查点，不代表仿真仍在运行。']
    if (output/'sampling_policy_v2.json').exists():
        revision=read(output/'sampling_policy_v2.json');p=revision['sampling_policy']
        lines+=['','## Folded LDO 采样策略修订 v2','',
            f"ldo_folded_cascode 从候选 {p['start_candidate']} 起使用 {p['version']}。上文的宽范围规则仅适用于修订前的候选及其余九个拓扑。",
            '围绕三个完整通过验收的工作点，W/L 在中心值 ×[1/1.05, 1.05] 内采样，偏置在 ×[1/1.02, 1.02] 内采样；保持匹配器件共享关系，固定各中心的倍乘器、反馈和负载参数。',
            '四策略筛选中原规则 3/10，新规则 10/10；另 20 个参数点独立验证通过 19/20。小样本结果不代表后续保证通过率。',
            '已有有效样本、原配方哈希、仿真条件和质量阈值不变；新样本增加采样版本及哈希。对照试验样本不计入生产数据。',
            '新规则偏向成功工作点的局部区域，不代表全局设计空间。少样本和域外泛化实验应披露这项分布限制。',
            '详见 [优化报告](analoggym_folded_optimization.md) 与输出目录 sampling_policy_v2.json。']
    (ROOT/'reports/analoggym_personal_20k.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Personal report refreshed:',status['state'],status['accepted'],'accepted of 20000',flush=True)


if __name__=='__main__':report()
