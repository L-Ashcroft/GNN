"""Cross-source zero-shot admission without reading private performance values."""
from common import ROOT, SOURCE, META, read, atomic, sha, now


def main():
    public = read(ROOT / 'data/model_ready/fusion_v2_training_v1/tasks.json')
    target = read(META / 'tasks.json')
    approved = read(META / 'manifest.json')['cross_source_target_equivalence_approved']
    evidence = []
    for group, task in target.items():
        conditions = task['contract']['conditions']
        for slot, rule in task['slots'].items():
            exact = [(g, j) for g, t in public.items() for j, r in t['slots'].items()
                     if r['semantic_id'] == rule['semantic_id']]
            assert not exact, 'An exact semantic match requires individual admission review'
            reasons = ['No approved shared physical-target/test-condition contract',
                       'Source checkpoint has no trained personal target readout or target-domain embedding']
            if task['contract']['family'] == 'Amplifier':
                reasons += ['Private fixture uses 500pF load; public LOADBench default is 1pF',
                            '0.1Hz transfer, signed follower offset and phase-at-unity are not generic gain/offset/PM synonyms']
            else:
                reasons += ['No public LDO task registered in the selected public fusion corpus',
                            'Explicit light/heavy load and external bias conditions require target-specific supervision']
            evidence.append({'target_group': group, 'slot': int(slot), 'name': rule['original_name'],
                             'unit': rule['unit'], 'definition': rule['definition'], 'conditions': conditions,
                             'exact_semantic_matches': exact, 'zero_shot_admitted': False, 'reasons': reasons})
    assert approved is False
    result = {'state': 'not_admissible_under_current_verified_contracts', 'updated_at': now(),
              'public_supervision_groups': len(public), 'target_supervision_groups': len(target),
              'target_defined_slots': len(evidence), 'verified_shared_targets': 0,
              'target_labels_used_for_fitting_or_selection': 0, 'test_metrics_computed': False,
              'explanation': 'This is a documented protocol limitation, not a measured zero-shot accuracy of zero.',
              'evidence': evidence,
              'sha256': {str(p.relative_to(ROOT)): sha(p) for p in [META / 'tasks.json', META / 'manifest.json',
                         ROOT / 'data/model_ready/fusion_v2_training_v1/tasks.json']}}
    folder = ROOT / 'artifacts/experiment3_cross_source_v1'
    atomic(folder / 'compatibility_audit.json', result)
    (folder / 'README.md').write_text(
        '# 实验三：公开域到独立 AnalogGym 目标域\n\n'
        f'已完成{len(evidence)}个目标槽位的监督兼容准入检查，当前没有获得批准的跨来源共享标签。'
        '这不等于零样本准确率为零；当前无法定义可比较的零样本性能测试。\n\n'
        '放大器负载、频率采样、偏差定义与相位定义不同；公开融合任务没有LDO监督头。'
        '不创建随机目标头并将输出记作零样本预测，不改写标签或Mask。\n\n'
        '实验四单独检验表征迁移：只迁移公开模型图编码权重，目标头、域调制及条件编码器使用目标训练标签学习。'
        '若后续需要真正的零样本对照，需先建立经验证的相同指标定义与兼容测试平台，独立版本化评估数据。\n', encoding='utf-8')
    print({k: v for k, v in result.items() if k not in ['evidence', 'sha256']})


if __name__ == '__main__':
    main()
