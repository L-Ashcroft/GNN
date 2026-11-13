"""Explain conservative groups and verify exact published LOADBench protocols.

Read-only with respect to datasets/routing; produces a reviewable merge audit.
"""
import ast
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read,dump,digest
from src.dataset_fusion import FusionDataset
from src.supervision_fusion import SupervisionRouter
from scripts.build_fusion_dataset import load_lb_spec


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_proof(spec):
    """Verify artifacts independently, including settings and metric orchestration."""
    folder=(ROOT/spec['acquisition']).parent
    verified,graph=load_lb_spec(folder)
    assert verified['compatibility_group']==spec['compatibility_group']
    assert verified['metric_mapping']==spec['metric_mapping']
    params=(folder/'params.cir').read_text(encoding='utf-8')
    for line in params.splitlines():
        line=line.strip()
        assert not line or line.startswith('*') or line.lower().startswith('.param '),('Unexpected parameter directive',folder,line)
    tree=ast.parse((folder/'standalone_tb_extract.py').read_text(encoding='utf-8'))
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    prefix=[]
    for statement in main.body:
        if (isinstance(statement,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='constraints' for t in statement.targets)):
            break
        prefix.append(ast.dump(statement,include_attributes=False))
    assert prefix and any("id='perfs'" in n for n in prefix)
    conditions={k:v for k,v in spec['conditions'].items() if k not in {'testbench_sha256','extractor_sha256'}}
    mapping=[{k:r[k] for k in ('slot','original_name','semantic_id','unit','source_unit','factor','definition')} for r in spec['metric_mapping']]
    proof={'conditions':conditions,'metric_mapping':mapping,'formal_DUT_ports':graph['external_ports'],
           'spiceinit_sha256':sha(folder/'.spiceinit'),
           'metric_orchestration_signature':digest(prefix),
           'process':spec['process'],'temperature':spec['temperature'],
           'source_record':'10.5281/zenodo.21759759',
           'shared_model_zip_sha256':sha(ROOT/'data/raw/LOADBench/reduced_sky130_combined.zip')}
    return digest(proof),proof


def main():
    ds=FusionDataset();router=SupervisionRouter(ds.catalog)
    columns=['source_dataset','topology_id','compatibility_group','testbench_id','process','PDK','temperature','supply_voltage']
    groups={};topology_groups=defaultdict(list);proofs={}
    for key,spec in ds.catalog.items():
        path=ds.path/'sampled'/(key+'.parquet')
        frame=pq.read_table(path,columns=columns).to_pandas()
        if frame.empty:continue
        grouped=frame.groupby(columns,dropna=False,sort=False).size().reset_index(name='sample_count')
        for row in grouped.to_dict('records'):
            count=int(row.pop('sample_count'))
            row={k:None if pd.isna(v) else v for k,v in row.items()}
            row.update(performance=[None]*16,mask=[False]*16,semantic_ids=[None]*16)
            desc=router.describe(row);group=desc['supervision_group']
            assert group not in groups
            groups[group]={'group_id':group,'source_dataset':spec['source_dataset'],
                'topology_ids':[key],'topology_count':1,'structure_count':1,
                'connectivity_groups':[spec['connectivity_group']],'sample_count':count,
                'contract':desc['contract']}
            topology_groups[key].append(group)
        if spec['source_dataset']=='LOADBench':proofs[key]=protocol_proof(spec)
    old=read(ds.path/'supervision_audit.json')
    assert {k:v['sample_count'] for k,v in groups.items()}==old['group_counts']
    buckets=defaultdict(list)
    for key,(signature,proof) in proofs.items():buckets[signature].append(key)
    merges=[]
    for signature,keys in sorted(buckets.items()):
        if len(keys)<2:continue
        members=[g for k in sorted(keys) for g in topology_groups[k]]
        # Exact row PVT remains required even when the published templates agree.
        pvts={json.dumps({k:groups[g]['contract'][k] for k in ['process','PDK','temperature','supply_voltage']},sort_keys=True) for g in members}
        assert len(pvts)==1
        merges.append({'merge_id':'LB_M'+str(len(merges)+1).zfill(2),'protocol_signature':signature,
            'supervision_group_ids':members,'topology_ids':sorted(keys),'topology_count':len(keys),
            'sample_count':sum(groups[g]['sample_count'] for g in members),
            'status':'verified_same_published_protocol_not_activated',
            'shared_targets':[r['original_name'] for r in ds.catalog[keys[0]]['metric_mapping']],
            'proof':proofs[keys[0]][1],
            'artifact_provenance':{k:{'testbench':ds.catalog[k]['conditions']['testbench_sha256'],
                                      'extractor':ds.catalog[k]['conditions']['extractor_sha256']} for k in keys}})
    summary={'supervision_groups':len(groups),'single_topology_groups':sum(g['topology_count']==1 for g in groups.values()),
        'covered_source_topologies':len(topology_groups),'covered_structure_classes':len({s['connectivity_group'] for k,s in ds.catalog.items() if k in topology_groups}),
        'groups_per_source':dict(Counter(g['source_dataset'] for g in groups.values())),
        'verified_merge_sets':len(merges),'groups_in_merge_sets':sum(len(m['supervision_group_ids']) for m in merges),
        'groups_after_these_merges':len(groups)-sum(len(m['supervision_group_ids'])-1 for m in merges),
        'cross_source_merges_verified':0,'routing_or_data_modified':False}
    result={'summary':summary,'current_groups':list(groups.values()),'verified_merge_sets':merges,
        'limitations':['Compatibility is based on identical published testbench, metric extraction/orchestration, units, simulator settings, model package and PVT; historical per-run simulation logs were not published.',
                      'Only fourteen mapped performance metrics are approved within listed sets, not optimization constraints or feasible flags.',
                      'Other groups are unverified rather than proven incompatible; voltage variants require an explicit conditioned prediction task, not an unconditional merge.',
                      'No automatic changes to routing, metric mappings, normalization, samples or splits.'],
        'index_sha256':ds.manifest['index_sha256']}
    dump(ds.path/'supervision_group_details.json',result)
    lines=['# 监督组拓扑覆盖与兼容性审计','',
        f"当前 {len(groups)} 个监督组全部只覆盖 **1 个来源拓扑条目、1 个实际连接结构**；共覆盖 {summary['covered_source_topologies']} 个来源拓扑条目、{summary['covered_structure_classes']} 个结构类。",
        '**这是当前路由把 topology_id 纳入分组键的结果，不表示这些拓扑在物理上只能各自训练。** 同一拓扑因供电等条件不同又会拆成多个组。',
        '',f"按当前官方文件可验证 {len(merges)} 组合并集合，共涉及 {summary['groups_in_merge_sets']} 个原监督组；若启用这些集合，142 组可缩减到 {summary['groups_after_these_merges']} 组。当前路由和数据没有修改。",
        '跨来源（FALCON/AICircuit/LOADBench）尚无通过这次审核的合并集合。','',
        '## 可共享目标的 LOADBench 集合','',
        '证据要求：移除 DUT 后测试平台一致，参数文件不含额外仿真指令，性能计算函数及常量一致，main 中读入和计算性能的调用流程一致，器件端口约定一致，.spiceinit、模型包、映射单位和测试条件一致。DUT 结构与器件尺寸允许不同。',
        '仅忽略纯来源追踪的整文件哈希差异；约束列表随拓扑变化，不批准共享约束标签或 feasible 含义。27°C 为已发布模板的默认条件；这不是独立逐条运行日志的证明。',
        '全部以下集合来自 LOADBench，可共享其映射的 14 个性能目标；POF 缺少 CMRR/PSRR 时仍使用缺失 Mask。','',
        '| 合并集合 | 拓扑（省略 LOADBench__ 前缀） | 原监督组数 | 样本数 |',
        '| --- | --- | ---: | ---: |']
    for m in merges:
        lines.append('| '+m['merge_id']+' | '+'、'.join('`'+k.removeprefix('LOADBench__')+'`' for k in m['topology_ids'])+f" | {len(m['supervision_group_ids'])} | {m['sample_count']:,} |")
    lines+=['','## 全部现有监督组','',
        '下表编号仅供阅读；完整稳定组 ID、条件、指标语义和合并集合的对应关系见 `data/fusion_v2/supervision_group_details.json`。','',
        '| 编号 | 组 ID 前 12 位 | 来源 | 拓扑 | 拓扑数 | 结构数 | 供电 V | 温度 °C | 样本数 | 可合并集合 |',
        '| --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | --- |']
    merge_by_group={g:m['merge_id'] for m in merges for g in m['supervision_group_ids']}
    for i,(gid,g) in enumerate(sorted(groups.items(),key=lambda kv:(kv[1]['source_dataset'],kv[1]['topology_ids'][0],str(kv[1]['contract']['supply_voltage']))),1):
        c=g['contract']
        lines.append(f"| G{i:03} | `{gid[:12]}` | {g['source_dataset']} | `{g['topology_ids'][0]}` | 1 | 1 | {c['supply_voltage']} | {c['temperature']} | {g['sample_count']:,} | {merge_by_group.get(gid,'未验证')} |")
    lines+=['','## 尚不能直接合并的情况','',
        'FALCON/AICircuit 的历史完整测试平台未发布，不能依据同名 VoltageGain、同单位 dB、结构相似或 PDK 名称就批准共享。',
        '不同供电条件下可以另行定义“显式输入供电电压的条件化任务”，但这需要审核和实现该任务，不能将目前的不同条件直接当作相同条件合并。',
        '本次严格相同协议之外的 LOADBench 拓扑也不等于不兼容；部分差异可能仅为命名、格式或等价的提取实现，仍需逐项证明后才能扩大共享范围。',
        '下一步若采用上述合并集合，应显式启用白名单、复查共享目标的划分和训练集归一化，再进行跨拓扑训练；本报告未训练模型。']
    (ROOT/'reports/fusion_supervision_groups.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
