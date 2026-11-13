"""Write a truthful, timestamped progress snapshot, never equating launch with completion."""
import sys
import json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.analoggym_campaign import read, atomic_json, sha


def main():
    output=ROOT/'data/generated/analoggym_campaign_v1'
    status=read(output/'status.json'); recipe=read(output/'recipe.json'); manifest=read(output/'manifest.json')
    dataset=read(output/'dataset.json')
    lines=['# AnalogGym 独立仿真数据集运行快照','',f"更新于 {datetime.now(timezone.utc).isoformat()}。实时数据请看 `data/generated/analoggym_campaign_v1/status.json`。",'',
           f"状态：**{status['state']}**。目标 {status['target_accepted']:,} 条，当前实际接受 **{status['accepted']:,}** 条，尝试 {status['attempts']:,} 次，隔离 {status['quarantined']:,} 次。",'',
           f"Parquet 快照已导出 {dataset['sample_count']:,} 条；运行中每分钟刷新，与账本短暂不同是正常的。只有 status.complete=true 且最终质量检查通过才算完成。",'',
           '| 拓扑 | 有效样本 | 尝试数 | 配额 |','| --- | ---: | ---: | ---: |']
    for name,s in status['topologies'].items(): lines.append(f"| {name} | {s['accepted']} | {s['attempts']} | {s['target']} |")
    lines+=['','## 数据边界','',
            '- 新数据独立保存；没有合并到 FALCON/AICircuit，也没有训练或拟合归一化。',
            '- 16 个放大器、2 个 LDO 进入本次批量候选范围；图、参数、标签及原始仿真归档各自可追溯。',
            '- 未通过的 Basic_LDO、ldo_1、ldo_2 保留调试波形。其他缺模型/网表的电路列在 recipe.json/excluded。',
            '- 早期宽松求解精度、旧负载配置和仅看平均输出的结果均为诊断，不计入本批正式配额。',
            '- 跨来源标签尚未批准统一；这批也按电路族和实际负载条件分语义组。',
            '', '## 运行资源','',
            f"并发 {manifest['workers']}，单个分析超时 {manifest['timeout_seconds']} 秒。原始归档当前占 {status['raw_bytes']/2**30:.3f} GiB，磁盘剩余 {status['free_disk_bytes']/2**30:.1f} GiB。",'',
            '按小批试跑估算，10 万条有效样本约需 8–15 天、约 40–100 GiB；会受不同拓扑成功率及电脑负载影响，低成功率时可能更久。180 GiB 归档预算或剩余空间低于 20 GiB 时平稳停止。电脑须保持开机且不休眠；不自动修改系统电源设置。','',
            '操作、参数范围和指标定义见 [生成说明](analoggym_campaign_method.md)。精度复查见 [数值验证](analoggym_campaign_precision_final.json)。']
    path=ROOT/'reports/analoggym_campaign_status.md'; path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({'state':status['state'],'accepted':status['accepted'],'exported':dataset['sample_count'],'target':status['target_accepted']})


if __name__=='__main__': main()
