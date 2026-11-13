"""Publish facts from completed simulations/cohort checks/validation-only fits."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'data/audit/loadbench_physical_v1'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))

def main():
    m=read(BASE/'manifest.json');check=read(BASE/'checks.json');assert check['passed']
    replay=read(BASE/'native_wrapper_units_v2_summary.json');b=read(ROOT/'artifacts/loadbench_physical_baseline_v1/validation_comparison.json')
    assert b['state']=='validation_diagnostic_complete' and len(b['proof'])==92
    retained=[r for r in m['per_topology'] if r['source']=='LOADBench' and r['rows']>r.get('needs_review',0)]
    empty=[r['topology_id'] for r in m['per_topology'] if r['source']=='LOADBench' and r['rows']==r.get('needs_review',0)]
    stamp=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    lines=['# LOADBench物理标签回放与后续实验准备','',f'北京时间：{stamp}。本次操作已完成，不是待执行计划。',
      '', '## 已确认的异常机制','',
      '使用官方发布的DUT、测试平台、reduced SKY130 TT模型和真实CSV参数，原生Windows单进程完成4次OP/偏置校正/AC回放。扫描仍为1Hz–1GHz、每十倍频20点。未使用这些回放生成或替换训练标签。',
      '', '最极端的SimpleOpAmp130历史样本在整个测量窗口内的增益最大约−4.996dB，始终没有0dB穿越。官方提取函数仍执行外推，得出约564GHz和−4699万度；存档值为410.6GHz和−3421万度。两次外推数值不逐位相同，历史CSV参数舍入会放大这种计算的敏感性，但“没有测到交越点却输出窗口外ft/极端相位”已由实测波形确认。',
      '', '| 回放样本 | 存档低频增益dB | 回放低频增益dB | 扫描内下降0dB穿越数 |',
      '| --- | ---: | ---: | ---: |']
    for r in replay['results']:
        a=r['analysis'];lines.append(f"| {r['label']} | {a['archived']['dc_gain']:.6g} | {a['official_function_on_replayed_waveform']['dc_gain']:.6g} | {len(a['descending_unity_crossings'])} |")
    lines += ['', '三个正常存档值对照经官方函数重提取，与原CSV吻合；历史对照最大增益差约0.00334dB、频率相对差约0.00775%、相位差约0.00394度。正常存档值不等同于已认证稳定性：NumPy独立相位展开在一个历史样本上与原函数相差360度分支，本次没有武断选择其中一个作为新真值。',
      '', '首轮SI数值直接送入微米模型包装器时模型bin匹配失败，失败日志保留在replays/。后续把CSV微米尺寸传给使用scale=1e-6的SKY130包装器，得到上述对照一致性；原图中的米单位、原CSV和第三方仓库均未修改。',
      '', '## 独立质量视图','',
      '新索引覆盖原50万条sample_id，全部唯一、覆盖与来源一致；原参数、标签、Mask、ID/LOTO划分不改。以下条件只触发复核，不自动证明整行失败：非正DC增益、报告ft位于其实际AC扫描窗口外、相位绝对值超过360度及非有限AC值。feasible=False本身不是排除条件。',
      '', f"- 触发复核：{m['counts']['needs_review']:,}条，全部来自LOADBench。",
      f"- 未触发：{m['counts']['no_trigger']:,}条，其中LOADBench为{200000-m['counts']['needs_review']:,}条，其余30万条在本策略中未被物理评估。",
      f"- LOADBench未触发子集仍覆盖{len(retained)}/92个原有拓扑；无剩余记录的拓扑：{', '.join(empty) if empty else '无'}。",
      '- 各触发项有重叠，不能将其出现次数直接相加作为样本数。',
      '', '| 条件 | 触发次数 |','| --- | ---: |']
    for k,v in m['flag_occurrences'].items():lines.append(f'| {k} | {v:,} |')
    lines += ['', '未触发不等于已通过物理认证。尤其不能将34,788条简单删掉后宣称原基准质量已修好：其中仍可能有有用的功耗、面积等标签，或合法的低增益/不稳定设计。这里保留完整复核组及未触发组，原始缺失Mask不承担有效性筛选。',
      '', '文件：data/audit/loadbench_physical_v1/{policy.json,quality_index.parquet,review_references.parquet,no_trigger_references.parquet,manifest.json,checks.json}。仅引用原样本，不复制一套新标签；质量索引及引用约7.50MiB。独立接口见worktrees/loadbench_physical_v1/load.py，默认拒绝测试角色，LOTO必须另按本折引用覆盖ID角色。',
      '', '## 已完成的同验证集对照','',
      '先运行2拓扑探针，再遍历全部92拓扑；其中89个拓扑在新群体中仍有样本，三个拓扑没有未触发记录。固定同一套17,121条未触发验证样本：原训练池160,046条，对照训练池132,169条。比较包含1,228个契约—指标组合，无额外因缺少训练标签而删除的验证目标。每个原监督契约使用相同参数Ridge形式；输入中心/尺度及symlog目标校准均仅由对应训练池拟合。没有请求任何测试标签行，不使用GNN权重，也没有调整旧模型。',
      '', '| Ridge正则系数 | 原训练池hit@5% | 未触发训练池hit@5% | 原/新R²中位数 | 原/新负R²目标数 |',
      '| --- | ---: | ---: | ---: | ---: |']
    for lam in b['configuration_lambdas']:
        a=b['summary'][f'original_train:{lam}'];c=b['summary'][f'no_trigger_train:{lam}']
        lines.append(f"| {lam:g} | {a['macro_hit5']:.2%} | {c['macro_hit5']:.2%} | {a['median_r2']:.3f}/{c['median_r2']:.3f} | {a['negative_r2_targets']}/{c['negative_r2_targets']} |")
    lines += ['', '原始macro R²没有隐藏：λ=0.01时两边仍约−9.90×10³⁷；λ=1时约−121.05/−0.434；λ=100时约−98.03/0.134。简单线性基线仍有严重外推/稀疏目标问题。结果仅说明该筛查下训练池选择对同一验证群体有帮助，不证明新GNN已改善，也不等于原52.05%的测试成绩可以按同样幅度增加。训练样本数量和目标群体均明确报告。',
      '', '基线完整证据：artifacts/loadbench_physical_baseline_v1/{probe.json,validation_comparison.json}。报告包括全部3个预设正则系数、每个目标指标和标签请求ID哈希，未只挑最好配置。',
      '', '## 原训练保护与当前安排','',
      '主机内存不足1GiB，先使用既有STOP接口安全保存并暂停，不强杀训练。首次回放适配失败后恢复；修正单位适配后的第二次短审计也在finally中恢复原监督器。恢复前核对last checkpoint与recipe哈希，恢复后核对PID创建时间及更新游标。两次窗口记录位于pause_history.jsonl和pause_record.json；旧模型、优化器/RNG游标恢复逻辑及源码均不改。',
      '', f"复核时原LOTO已恢复为{check['training_status_at_check']['state']}，第{check['training_status_at_check']['epoch']}轮，第{check['training_status_at_check']['batch']}批；最佳兼容验证命中率{check['training_status_at_check']['best_compatible_validation_hit5']:.2%}来自第{check['training_status_at_check']['best_epoch']}轮，仍不是留出拓扑测试分数。原始标签实验作为可追溯基线继续，未强行套用新的质量筛查。",
      '', '当前不启动新的大型GNN训练。原因不是等待审批，而是这套规则只定位可疑样本、尚未认证所有剩余标签；应先确定工程任务的可定义性与相位分支规则，再在独立recipe下建立质量修订实验。已有轻量同群体对照已完成，后续GNN须保持同一验证群体、统计变化的训练量，并单列旧原始标签基准。',
      '', '## 可复现检查','',
      '已运行checks.py、2拓扑probe、92拓扑baseline以及verify_release.py。检查通过：50万唯一ID与原索引一致、所有原分区哈希不变、两引用集合数量互补、六个真实样本通过接口原样读取Mask/标签、四个AC回放完成、原训练同recipe继续。无需安装依赖、下载大型数据或启动GPU测试。']
    (ROOT/'reports/loadbench_physical_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'report':'reports/loadbench_physical_audit.md','LOADBench_no_trigger_topologies':len(retained),'empty_topologies':empty,'paired_baseline_complete':True},ensure_ascii=False))

if __name__=='__main__':main()
