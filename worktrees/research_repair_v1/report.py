"""Regenerate a concise repair report from measured artifacts, not guesses."""
from core import ROOT, OUT, CONTROL, read, now


def pct(value):
    return '待测' if value is None else f'{value * 100:.2f}%'


def main():
    lines = ['# 实验修订与执行状态', '', '更新：' + now(), '',
             '所有修订均位于独立工作副本和输出目录。原始数据、划分、监督语义、旧代码、Checkpoint与论文文件保持原状。', '',
             '## 已完成的修复与实测', '',
             '1. 修复新ID对照的模型选择范围：由仅M04监督组，改为整个固定验证集。LOTO仍只用可比较的兼容组验证成绩选择模型。',
             '2. 补全LOADBench小监督组的候选实验：11个经核验的兼容组，比较共享岭读出、组内训练范围约束与拓扑自身训练范围约束。输入/输出统计仅使用TRAIN；原缺失Mask不变。',
             '3. 新增小样本候选准入门槛：命中率提高且平均/中位R²均不降低、负R²目标数量不增加，才进入待测试名单。验证不合格则沿用原已选模型。',
             '4. 为全部18个已冻结个人模型建立固定测试评估；提取器在16条真实验证样本上的特征与旧缓存最大绝对误差为0。',
             '5. 增加种子43、44的独立全网络预训练/从零成对重训，补足原端到端迁移配方的三种子研究；不能称为后续混合读出的三种子结果。', '']
    path = OUT / 'loadbench_shared_v2/fixed_test/metrics.json'
    if path.exists():
        r = read(path); a = r['raw_all_original']['baseline']; b = r['raw_all_original']['candidate']
        lines += ['### LOADBench固定测试（历史测试集重复评估）', '',
                  '| 指标 | 上一修正版 | 本次验证预选候选 |', '|---|---:|---:|',
                  f"| 宏平均5%命中率 | {pct(a['macro_hit5'])} | {pct(b['macro_hit5'])} |",
                  f"| 中位R² | {a['median_r2']:.6f} | {b['median_r2']:.6f} |",
                  f"| 平均R²（保留全部小样本目标） | {a['macro_r2']:.3f} | {b['macro_r2']:.3f} |", '',
                  f"测试{r['samples']:,}条、92种来源拓扑；{r['changed_predictions_samples']}条的预测发生变化。实质收益主要来自M01的拓扑训练范围约束，共享线性头未带来通过门槛的实质收益。M03仅有浮点数量级变化，不作为效果提升。", '',
                  '这是一项有限修复：平均R²仍很差，尤其是极少样本、低方差目标；不能宣称LOADBench问题已经解决。完整原始统计、AC质量敏感性统计与n≥30敏感性统计分开记录，后两者不替代原指标，也不改Mask。', '',
                  '此测试先冻结候选，再读取历史测试预测；测试结果不用于选择新模型或决定回滚。由于此前分析已经用过同一测试集，这不是新的盲测。', '']
    p = OUT / 'fewshot_calibration/promotion_guard.json'
    if p.exists():
        g = read(p); accepted = [k for k, v in g['jobs'].items() if v['validation_qualified']]
        lines += ['### 小样本读出校准', '',
                  f"已拟合15个预算/初始化组合，每组比较5个岭系数；{g['eligible_count']}个候选通过综合验证门槛：" + '、'.join(accepted) + '。', '',
                  '这些候选尚未部署或取得新测试成绩。原k100预训练混合模型优于本轮线性校准，继续保留。每种预算实际还使用2000条验证标签，不能写成仅k条标签完成模型选择。', '']
    p = OUT / 'personal_test/summary.json'
    lines += ['### 个人数据集固定测试', '']
    if p.exists():
        result = read(p)
        lines += ['| 模型 | 测试5%命中率 | 测试中位R² |', '|---|---:|---:|']
        for name in ['full_physical', 'full_joint', 'full_gnn', 'k100_pretrained_joint', 'k100_scratch_joint']:
            m = result['jobs'][name]; lines += [f"| {name} | {pct(m['macro_hit5'])} | {m['median_r2']:.6f} |"]
        lines += ['', '全部18个模型按预先冻结的名单评估；不能按测试成绩另选最优模型。', '']
    else:
        s = read(OUT / 'personal_test/status.json') if (OUT / 'personal_test/status.json').exists() else {}
        lines += ['尚未完成；当前：' + str(s.get('state', '排队等待')) + '。新模型原先的约68%成绩属于验证集，不能先写为测试集结果。', '']
    lines += ['## 自动执行队列与公平比较', '',
              '顺序：短时读出校准 → 18个冻结个人模型测试 → 四个独立模型种子重复 → 公共ID同架构对照 → 三阶放大器LOTO折 → 去FiLM、静态边两项真实重训消融。', '',
              '| 公共任务 | train / validation / test | 模型选择 |', '|---|---|---|',
              '| ID对照、两个ID消融 | 400083 / 50861 / 49056 | 整个验证集宏平均hit5 |',
              '| M02三阶放大器LOTO | 446983 / 50613 / 2404 | 可比较共享监督组的验证hit5 |', '',
              '主模型4层、128维、48种关系，8,146,624个参数；去FiLM为8,014,528；静态边为7,883,456。所有任务从随机种子42重新训练，不加载已见目标拓扑的ID权重或嵌入。消融只改变声明的组件，其余公共划分、Loss、训练预算和选模规则相同。', '',
              'CPU四线程，FP32，最大20轮、至少4轮、早停耐心5；AdamW，主干2e-4/适配3e-4、权重衰减0.01，有效批256/微批128。三套公开任务的配方、源文件指纹、训练统计和独立产物均已记录。', '',
              '四个公共任务的小批前向/反向检查通过（各含三种来源），划分按精确引用交集检查无重叠，没有构建测试图张量。不能把这些检查当成正式训练成绩。', '',
              '按既有CPU速度估计，公共队列上限约180小时，加四个小样本重复约4小时；早停可能缩短。新增产物预计约3.5GiB，个人测试特征约0.11GiB。串行运行，低优先级，内存不足会等待或断点暂停。4GiB主机可用内存与当前实现数值验收同时满足后才允许CUDA；不宣称未经测速的GPU加速。', '',
              '## 尚不能自动消除的限制', '',
              '实验三仍无有效零样本指标：公共/个人数据没有已核验的共同监督契约，500pF与1pF负载、模型束和提取器不同，输出头与输入词表也不同。已重新执行契约准入检查；不能通过改标签名、随机新头或用目标标签训练来伪装零样本实验。下一步须建立源域与目标域共同PDK/夹具/提取器的桥接仿真数据，然后只在源域拟合共同头。此次没有擅自启动大规模桥接仿真。', '',
              '现有LOTO只有一折正式结果；新折、同架构ID和消融仍在执行队列，尚不能给出多折均值±标准差或组件有效性结论。小样本多种子结果完成前也不能声称预训练稳定提升。', '',
              '## 跟踪与复现', '',
              '`monitoring/research_repair_v1/status.json`查看当前任务、工作进程PID/创建时间、阶段与内存。各任务日志也在该目录。`artifacts/research_repair_v1/public_queue.json`及`personal_repeats_queue.json`列出独立版本、配置和输出路径。', '',
              '启动/恢复编排器（有互斥锁，保持已有recipe哈希检查）：', '',
              '```powershell', '.\\.venv\\Scripts\\python.exe -X utf8 -B -u .\\worktrees\\research_repair_v1\\supervise.py', '```', '',
              '安全停止：在`monitoring/research_repair_v1/STOP`创建文本文件；当前梯度训练保存断点后退出，编排器不启动下一项。', '']
    if (CONTROL / 'status.json').exists():
        s = read(CONTROL / 'status.json'); lines += ['生成报告时编排状态：' + s.get('state', '') + '；当前任务：' + str(s.get('current_job')), '']
    (ROOT / 'reports/research_repair_v1.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Updated reports/research_repair_v1.md')


if __name__ == '__main__':
    main()
