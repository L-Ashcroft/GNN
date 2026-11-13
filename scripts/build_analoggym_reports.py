"""Generate a report for the official AnalogGym replay only."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import read, dump
from src.load_analoggym import audit


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(x).replace('|', '\\|') for x in row) + ' |' for row in rows])


def main():
    replay = read(ROOT / 'reports/analoggym_replay_audit.json')
    dump(ROOT / 'reports/analoggym_audit.json', audit())
    content = [
        '# 官方 AnalogGym 回放专项审计', '',
        '本报告保留 2026-10-22 的固定版本审计证据，于 2026-10-23 清理无关来源后重整。全部操作位于 D:/EDA，使用原生 Windows Python。未运行仿真或训练。', '',
        f"**实际可恢复 {replay['written_transitions']} 条转移；新增可验证监督样本为 {replay['supervised_accepted']}。** 性能新鲜度、失败状态及历史版本证据不足，全部保存在隔离区。", '',
        table(['环境', '预分配容量', '实际写入', '不同动作', '不同性能向量', '接收监督样本'],
              [[r['topology'], r['allocated_capacity'], r['written_transitions'], r['distinct_actions'],
                r['distinct_performance_vectors'], r['supervised_accepted']] for r in replay['replays']]), '',
        '## 来源与读取方式', '',
        f"官方仓库：[CODA-Team/AnalogGym](https://github.com/CODA-Team/AnalogGym)，提交 `{replay['repository']['commit']}`，BSD-3-Clause；PDK 自身许可单独适用。", '',
        '一个 saved_memories ZIP 含 AMP/LDO 两个 pickle 及对应 reward NPY。每个 buffer 预分配 100000 个位置，但 size=ptr=10，非空 info 也各为 10。容量不是样本数。', '',
        '受限反序列化仅允许 NumPy 重建与本地无行为 ReplayState，不导入上游 ddpg。saved_agents 仅 pickletools 静态扫描：动作 payload、size/ptr 与 memory 一致，不作为额外样本；reward NPY 各 10 项也一致。顺序读取建议预留约 2 GiB 内存，无需安装 PyTorch。', '',
        '## 参数恢复与版本差异', '',
        'ddpg.py 将动作与 env.step 的 next_state/info 一起保存。上下界取自归档 agent 的 float64 NumPy 载荷，记录 offset/hash，不能套用现源码范围。归档动作 float32，原随机动作可能 float64，因此反变换尺寸只标为候选值；整数倍数按 int 截断并检查舍入歧义。', '',
        table(['环境', '低界不同参数', '高界不同参数'],
              [[r['topology'], ', '.join(r['bound_mismatch_parameter_names']['action_space_low']),
                ', '.join(r['bound_mismatch_parameter_names']['action_space_high'])] for r in replay['replays']]), '',
        '参数顺序来自当前环境 AST，历史完整代码提交未知。候选 Ib 与 next_obs 相符；LDO 第 7 条（index=6）的 21 个参数与提交的变量文件一致；AMP 无完整吻合行。参数对应成功不证明性能有效。', '',
        '## AMP：不同动作对应恒定性能', '',
        '10 个不同动作的 11 项性能完全相同，如 Power=261.17946、dcgain=137.6036、GBW=2514078。现源码不检查 os.system 返回码、不清除旧输出、不校验时间，随后读取固定文件名，存在旧结果复用疑点；不能据此断言每一步具体失败原因。', '',
        '当前 testbench 对 Power 乘 1e3，历史缩放未验证，不能自行认作 µW。当前 CMRR/PSR 为 0.1 Hz 传递 dB，不直接等同其他来源的 rejection dB。setting_time 失败时可能被 d_settle 替代，info 不含分支标记；裁剪观测不能当原始工作点。', '',
        '## LDO：旧 DC 文件及失败回退', '',
        '10 条的 LDR=-31.3856287、Power_maxload=0.019677348、Power_minload=0.00086794722、vos_maxload=17658.77、vos_minload=1838.636 与提交的 DC 文件对应列一致。日志显示负载调整率测量失败、no such vector LR，wrdata 无法刷新；日志中的 power/vos 与文件不同。这支持文件过期的判断，但日志不是逐条回放日志。', '',
        'GBW/PM 在非正 loop gain 分支置零，异常分支可能保留旧属性。info 缺 loop gain、每步成功标志及文件哈希，不能将零自动当有效测量，也不能改为 mask=False。部分非零 AC 值仍缺逐步新鲜度证据。CL (pF) 是输入相关电容量，不列为预测目标。', '',
        '## 产物、检查与使用边界', '',
        table(['路径', '内容'], [
            ['data/audit/analoggym_replay/quarantined_transitions.parquet', '20 条动作、候选参数、原始指标、Mask 与隔离原因'],
            ['reports/analoggym_replay_audit.json', '全量回放、范围、参数及源文件哈希核对'],
            ['reports/analoggym_agent_opcode_probe.json', '静态解析证据'],
            ['config/analoggym_replay_performance_mapping.json', '原始指标映射与隔离理由'],
            ['src/load_analoggym.py', '默认拒绝将隔离数据用作训练的访问接口'],
            ['reports/analoggym_audit_validation.json', 'ID、有限性、Mask、隔离与主集合排除检查'],
            ['reports/analoggym_audit_reproducibility.json', '重复执行的产物哈希']]), '',
        '质量与缺失分离：全部原始字段保留，异常/回退值不改成缺失；supervised_ready=False。隔离记录未加入 50 万条主集合或划分，未拟合新增归一化参数。', '',
        '官方网表、参数范围和测试平台仍是拓扑来源及补仿真基础。补数据应先少量点验证成功状态、新鲜度、耗时和失败率，报告资源需求，再批量运行；新结果登记为新仿真数据。', '',
        '复现：', '', '```powershell', r'.\.venv\Scripts\python.exe scripts/run_analoggym_audit.py', '```', '',
        '[固定版本源码](https://github.com/CODA-Team/AnalogGym/tree/'+replay['repository']['commit']+'/RGNN_RL)。',
    ]
    (ROOT / 'reports/analoggym_deep_audit.md').write_text('\n'.join(content)+'\n', encoding='utf-8')
    print('Official AnalogGym replay report updated')


if __name__ == '__main__': main()
