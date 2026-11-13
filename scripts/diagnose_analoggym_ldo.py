"""Read-only diagnosis of folded-cascode LDO failures in the live ledger."""
from collections import Counter,defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import io
import json
import sqlite3
import sys
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read,dump


def main():
    folder=ROOT/'data/generated/analoggym_personal_20k_v1';name='ldo_folded_cascode'
    db=sqlite3.connect(f'file:{(folder/"ledger.sqlite").as_posix()}?mode=ro',uri=True)
    records=db.execute('select candidate,accepted,result_json from attempts where topology=? order by candidate',(name,)).fetchall()
    db.close();entry=next(e for e in read(folder/'recipe.json')['entries'] if e['name']==name)
    failures=Counter();bands=defaultdict(lambda:Counter(attempts=0,accepted=0));centres=defaultdict(lambda:Counter(attempts=0,accepted=0))
    excursions=Counter();examples=[];outputs=defaultdict(list)
    for candidate,accepted,payload in records:
        r=json.loads(payload);failures[r['failure'] or 'accepted']+=1
        band=(0,8,2,1,3,4,5,6,7,9)[candidate%10]
        label='baseline' if candidate==0 else 'narrow' if band<2 else 'medium' if band<8 else 'wide'
        bands[label].update(attempts=1,accepted=accepted)
        centres[candidate%len(entry['sampling_centers'])].update(attempts=1,accepted=accepted)
        with zipfile.ZipFile(ROOT/r['archive']) as z:
            if 'load_wave.dat' not in z.namelist():continue
            a=np.loadtxt(io.BytesIO(z.read('load_wave.dat')),ndmin=2)
        if a.shape!=(2,4):raise ValueError('Unexpected LDO load-sweep shape')
        light,heavy=map(float,a[:,1]);outputs['accepted' if accepted else 'rejected'].append([light,heavy])
        if not accepted:
            excursions['light_outside']+=int(abs(light-1.2)>.24)
            excursions['heavy_outside']+=int(abs(heavy-1.2)>.24)
            excursions['negative_output']+=int(min(light,heavy)<0)
            if len(examples)<6:examples.append({'candidate':candidate,'light_1mA_V':light,'heavy_10mA_V':heavy,'reason':r['failure']})
    accepted=sum(ok for _,ok,_ in records);total=len(records)
    data={'snapshot_local':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'topology':name,
        'attempts':total,'accepted':accepted,'acceptance_rate':accepted/total,
        'failures':dict(failures),'load_excursions':dict(excursions),
        'by_band':dict(bands),'by_center':dict(centres),'failed_examples':examples,
        'output_medians_V':{k:np.median(np.asarray(v),axis=0).tolist() for k,v in outputs.items()},
        'live_sampling_recipe_changed':False,'background_generation_stopped':False,
        'inference':'Measured regulation failure, predominantly under 10mA load. Parameter sensitivity/operating-margin hypothesis is not yet a causal identification.'}
    dump(ROOT/'reports/analoggym_ldo_folded_diagnosis.json',data)
    lines=['# Folded-cascode LDO 失败诊断','',f"只读快照：{data['snapshot_local']}。后台计数继续变化。",
        '',f"已提交 {total} 次，接受 {accepted} 次，接受率 **{accepted/total:.1%}**。已接受设计确实存在，因此不是该拓扑完全不能运行。",
        '', '## 实际失败含义','',
        '当前记录的失败是 `LDO outside 20% regulation window`：在 1mA / 10mA DC 负载下，至少一个输出超出 1.2V ±20%，即 0.96–1.44V。不能将这个错误统一解释成仿真器崩溃或不收敛。',
        f"失败记录中，轻载越界 {excursions['light_outside']} 次、重载越界 {excursions['heavy_outside']} 次；两项可能重叠。出现负输出 {excursions['negative_output']} 次。",
        '部分设计输出贴近 1.8V 电源轨，部分在理想恒流负载下掉压甚至成为负值。它们不符合本数据集的正常稳压工作检查，已经隔离，未计入有效样本。',
        '', '| 候选 | 1mA 输出 V | 10mA 输出 V |','| ---: | ---: | ---: |']
    for e in examples:lines.append(f"| {e['candidate']} | {e['light_1mA_V']:.6f} | {e['heavy_10mA_V']:.6f} |")
    lines+=['','## 采样范围与中心','',
        '| 范围 | 尝试 | 接受 | 接受率 |','| --- | ---: | ---: | ---: |']
    for label,c in bands.items():lines.append(f"| {label} | {c['attempts']} | {c['accepted']} | {c['accepted']/c['attempts']:.1%} |")
    lines+=['', '每个中心均来自真实成功仿真。中心成功不保证对全部独立器件参数同时扰动后仍有足够工作余量。即便宽度/长度本身在模型范围内，偏置、电流比例和输出驱动也可能失衡。',
        '当前对这个拓扑的局部盒式采样偏激进，应单独优化它的采样策略；其他拓扑不必据此全部收窄。窄范围结果样本仍少，且没有完成单因素或相关扰动实验，尚不能断言某个具体器件是唯一原因。',
        '', '## 建议的后续处理','',
        '1. 保留当前接受点和失败证据，继续执行原检查，不能把超差值改为缺失 Mask 或放宽条件凑数。',
        '2. 在有资源预算的小批试验中，比较约 ±5%、±10% 的几何扰动，进一步收紧偏置；检查是否需要关联变化保持尾电流/支路电流比例，并维护原有匹配别名。',
        '3. 输出 PMOS M10、误差放大器偏置 Vb1/Vb2 及补偿是需要分别验证的候选因素；不能仅凭观察宣称原因已确认。',
        '4. 若新的合理采样显著提高通过率，再建立有版本和起始编号记录的新采样配方；不要在运行中悄悄改已冻结函数或覆盖旧参数。',
        '', '本次仅诊断并写报告，未停止后台任务，未更换拓扑，未改变冻结采样配方或有效样本定义。']
    (ROOT/'reports/analoggym_ldo_folded_diagnosis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in data.items() if k in ['snapshot_local','attempts','accepted','acceptance_rate','failures','load_excursions']},indent=2))


if __name__=='__main__':main()
