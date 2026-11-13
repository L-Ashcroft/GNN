"""Review only recorded validation metrics with bounded JSON parsing memory."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import statistics

ROOT = Path(__file__).resolve().parents[2]


def array_records(path):
    decoder = json.JSONDecoder()
    buffer = ''
    with path.open(encoding='utf-8') as stream:
        ended = False
        while True:
            buffer = buffer.lstrip(' \r\n\t[,')
            if buffer.startswith(']'):
                return
            try:
                record, offset = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                block = stream.read(65536)
                if not block:
                    if buffer.strip():
                        raise ValueError('Incomplete training history')
                    return
                buffer += block
                continue
            yield record
            buffer = buffer[offset:]


def main():
    output = ROOT / 'artifacts/experiment1_autonomous_head_v1'
    status = json.loads((output / 'status.json').read_text(encoding='utf-8'))
    epoch = status['best_total_head_epoch']
    selected = next(r for r in array_records(output / 'history.json') if r['head_epoch_total'] == epoch)
    validation = selected['validation']
    assert abs(validation['macro_hit5'] - status['best_validation_score']) < 1e-12
    tasks = json.loads((ROOT / 'data/model_ready/fusion_v2_training_v1/tasks.json').read_text(encoding='utf-8'))
    rows = validation['per_target']
    eligible = [r for r in rows if r['hit5'] is not None]
    sources = {}
    for source in ['FALCON', 'AICircuit', 'LOADBench']:
        part = [r for r in rows if tasks[r['supervision_group']]['contract']['source_dataset'] == source]
        hit = [r['hit5'] for r in part if r['hit5'] is not None]
        r2 = [r['r2'] for r in part if r['r2'] is not None]
        sources[source] = {
            'metric_targets': len(part), 'hit_eligible_targets': len(hit),
            'share_of_macro_hit5': len(hit) / len(eligible),
            'macro_hit5': statistics.mean(hit), 'median_r2': statistics.median(r2),
            'negative_r2_targets': sum(v < 0 for v in r2),
            'r2_at_least_0_9_targets': sum(v >= .9 for v in r2),
            'validation_count_below_20_targets': sum(r['count'] < 20 for r in part),
        }
        assert abs(sources[source]['macro_hit5'] - validation['by_source'][source]['macro_hit5']) < 1e-12
    count_buckets = {}
    for label, lower, upper in [('below20', 0, 20), ('20to99', 20, 100), ('atleast100', 100, float('inf'))]:
        part = [r for r in eligible if lower <= r['relative_count'] < upper]
        count_buckets[label] = {'targets': len(part), 'macro_hit5': statistics.mean(r['hit5'] for r in part) if part else None}
    negative = [r for r in rows if r['r2'] is not None and r['r2'] < 0]
    result = {
        'review_at_Beijing': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'recorded_validation_only': True, 'test_read': False, 'selected_head_epoch': epoch,
        'selection_rule_unchanged': 'macro_hit5', 'macro_hit5': validation['macro_hit5'],
        'median_r2': validation['median_r2'], 'macro_r2': validation['macro_r2'],
        'metric_targets': len(rows), 'hit_eligible_targets': len(eligible),
        'sources': sources, 'relative_support_buckets': count_buckets,
        'negative_r2_targets': len(negative),
        'worst_validation_r2': sorted(negative, key=lambda r: r['r2'])[:20],
        'conclusion': 'Head stage complete; overall quality not accepted as engineering-ready. Joint backbone refinement remains resource-gated.',
        'limitations': ['No causal attribution from aggregate metrics alone.', 'No new model inference or GPU benchmark.', 'No universal numerical pass mark was found in the lecture sections reviewed.'],
    }
    report = ROOT / 'reports/experiment1_validation_review.json'
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    lines = [
        '# 实验一当前质量复核', '', f"复核时间：{result['review_at_Beijing']}。仅分析已保存的第 {epoch} 轮验证记录，未读取测试集或修改评价公式。", '',
        '**输出头训练阶段完成，不代表模型精度已达到工程使用要求。** 联合主干微调还未启动。当前验证 hit@5% 为 24.10%、R² 中位数为 0.6073，原始 macro R² 为 0.1901。', '',
        '| 来源 | 指标组合数 | 可计算 hit 的组合数 | 对现有 macro hit 的权重 | hit@5% | R² 中位数 | R²<0 的组合数 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for source, r in sources.items():
        lines.append(f"| {source} | {r['metric_targets']} | {r['hit_eligible_targets']} | {r['share_of_macro_hit5']:.2%} | {r['macro_hit5']:.2%} | {r['median_r2']:.4f} | {r['negative_r2_targets']} |")
    lines += ['',
        '汇总按契约/指标组合宏平均，并非按三个来源等权，也不是按样本数平均。LOADBench 的目标组合更多，因此它的表现对当前总分影响更大。这解释了汇总分数的构成，不能据此认定其弱项可以忽略。不同来源任务难度和标签语义不同，来源分数不是严格的同任务对照。', '',
        f"全部 {len(rows)} 个观测目标中，{len(eligible)} 个可计算相对误差命中率；对数单位等不适用项依既有规则保留其他指标。当前仍有 {len(negative)} 个验证 R² 为负的目标，不能仅展示较好的中位数。", '',
        '讲义5 §39.3 强调优先报告 hit@5%，§39.5 要求观察验证命中率是否趋平；讲义4 §23.1 区分泛化边界研究与最低误差目标。上述内容没有给出所有来源、所有指标通用的达标数值。讲义曲线中的示意数字不能当成已经验证的硬门槛。', '',
        '阶段采用门槛只是“比父模型有改进”，不代表“所有工程指标已经达标”。输出头在固定表示上的改善趋平，应检验联合微调是否提升表示能力，再以同划分、同标签的单来源或参数基线检查共享训练是否存在负迁移。当前证据不能确定输出头、图主干、共享任务干扰中哪一个是主要原因。', '',
        '执行顺序：资源门槛满足后运行已准备的联合微调 pilot；完整验证有实际收益再采用；若仍不足，独立对照同预算单来源/参数基线，定位表示不足或任务干扰，再决定结构修改。保留原模型和所有弱项，不通过改汇总口径或筛除难指标制造提升。', '',
        '目前后台控制器仍受至少 4 GiB 可用主机内存门槛限制。此复核未启动大型 CPU/GPU 任务。',
    ]
    report.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'worst_validation_r2'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
