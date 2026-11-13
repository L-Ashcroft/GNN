"""Audit live LOTO references and publish truthful progress/metadata clarification."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,re,ast
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
cfg=read(HERE/'config.json');OUT=ROOT/cfg['output'];protocol=read(ROOT/cfg['protocol']);raw=read(ROOT/'data/model_ready/fusion_v2_training_v1/manifest.json')
proof=[];correction={}
for fold in protocol['folds']:
    path=ROOT/fold['path'];assert sha(path)==fold['sha256']
    with np.load(path,allow_pickle=False) as f:
        allpositions=np.concatenate([f[r+'_positions'] for r in ['train','validation','test','unused']]);assert len(allpositions)==500000 and len(np.unique(allpositions))==500000
        train=f['train_references'];held={*fold['test_topologies'],*fold['excluded_structural_aliases']}
        assert not {protocol['topologies'][int(n)] for n in np.unique(train[:,0])}&held
        actual=np.zeros(16,dtype='int64')
        for peer in fold['compatible_training_topologies']:
            number=protocol['topologies'].index(peer);rows=train[train[:,0]==number,1]
            mask=np.load(ROOT/raw['topologies'][peer]['folder']/'mask.npy',mmap_mode='r');actual+=mask[rows].sum(0);mask._mmap.close()
    correction[fold['fold_id']]={'actual_expanded_train_target_counts':actual.tolist(),'historical_v1_counts_in_source_manifest':fold['training_target_counts'],'counts_recomputed_from_frozen_references':True}
    proof.append({'fold_id':fold['fold_id'],'positions_cover_public500000_once':True,'heldout_topology_excluded':True})
# Do not rewrite an already running protocol or pretend its fingerprint did
# not change. Publish an explicit, hash-bound metadata correction/view.
clarification={'source_manifest':cfg['protocol'],'source_manifest_sha256':sha(ROOT/cfg['protocol']),
    'role_authority':'Each fold NPZ and fold.role_policy; expanded train includes non-target formerIDtest records',
    'superseded_historical_v1_clause':'Non-target original ID-test samples remain unused',
    'note':'Inherited v1 training_target_counts were lower bounds; runtime fitting never used those values. Actual fold-train counts are below and in normalization.json.',
    'runtime_inputs_or_training_state_modified':False,'folds':correction}
base=ROOT/'data/fusion_v2/splits/verified_loto_full_v2';(base/'metadata_clarification.json').write_text(json.dumps(clarification,indent=2),encoding='utf-8')
view=json.loads(json.dumps(protocol));view['policies']=[p for p in view['policies'] if not p.startswith('Non-target original ID-test samples remain unused')]
for f in view['folds']:f['training_target_counts']=correction[f['fold_id']]['actual_expanded_train_target_counts']
view['view_only_not_runtime_recipe_input']=True;view['runtime_manifest_sha256']=clarification['source_manifest_sha256'];(base/'manifest_documented.json').write_text(json.dumps(view,indent=2),encoding='utf-8')
s=read(OUT/'status.json');monitor=ROOT/'monitoring/experiment2_loto_v1'/cfg['fold_id'];m=read(monitor/'status.json');recipe=read(OUT/'recipe.json');stats=read(OUT/'normalization.json');checks=read(HERE/'checks.json')
assert psutil.Process(s['pid']).create_time()==s['create_time'];assert psutil.Process(m['supervisor_pid']).create_time()==m['supervisor_create_time']
assert all(sha(ROOT/p)==h for p,h in recipe['source_sha256'].items());assert sha(ROOT/cfg['protocol'])==recipe['protocol_sha256'] and sha(OUT/'normalization.json')==recipe['normalization_sha256']
assert stats['targets']['verified_LOADBench_LB_M04']['count']==correction[cfg['fold_id']]['actual_expanded_train_target_counts']
assert stats['proof']['training_rows']==446984 and not stats['proof']['test_or_validation_labels_used'] and not s['heldout_test_used']
for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
audit={'passed':True,'checked_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'all24reference_partitions_verified':True,'folds':proof,
    'live_worker_identity_verified':True,'supervisor_identity_verified':True,'source_recipe_and_normalization_unchanged':True,
    'training_uses_public_only':True,'heldout_test_labels_used':False,'live_state':s['state'],'live_batch':s.get('batch'),'live_train_samples':s.get('samples_done'),
    'checkpoint_exists':(OUT/'last.pt').exists(),'metadata_clarification_without_recipe_mutation':True}
(OUT/'startup_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
lines=['# 实验二：公开50万样本的留一拓扑训练','',f"更新时间：{audit['checked_at']}。首折已实际启动，状态以实时文件为准。",'',
    '第一项ID实验已经冻结，适合作为已知拓扑预测基线；它不是全指标工程达标模型。混合GNN+局部读出的固定测试hit@5%=52.05%、R²中位数0.837，但原始macro R²仍约-2703.7。纯神经头对照验证为30.81%。这些模型见过当前目标拓扑，不能拿来作LOTO预训练。','',
    '## 首折数据与监督定义','',
    '**仅使用FALCON、AICircuit、LOADBench公开融合50万条，不使用AnalogGym个人2万条。** 首折完整留出 `LOADBench__SimpleOpAmp4`，包括它的所有参数设计；具有相同完整端口连接结构的其他条目也不能进入训练。对应已验证同条件的 `SimpleOpAmp6` 参与训练共享语义头。', '',
    '| 分区 | FALCON | AICircuit | LOADBench | 总计 |','| --- | ---: | ---: | ---: | ---: |',
    '| 训练 | 179880 | 89912 | 177192 | **446984** |','| 验证 | 20120 | 10088 | 20404 | **50612** |','| 未知拓扑测试 | 0 | 0 | 2404 | **2404** |','| 未使用 | 0 | 0 | 0 | **0** |','',
    '原ID数据和8:1:1文件未修改。本实验建立独立角色：非目标拓扑的原IDtrain与IDtest并入LOTOtrain，原IDvalidation仍作验证；目标拓扑全部作为外部test。这是独立实验的重新划分，不是把LOTO的测试样本拿去训练。每折新索引约按剩余样本9:1划分训练/验证，保持原设计分组与分层验证分布。','',
    '监督仅在11个已审计同PDK/PVT、测试平台、提取方法及单位的LOADBench集合内共享，142个原路由在本实验视图变为129个。三来源同名Gain等标签没有强行合并。共享后的14个目标及真实缺失Mask保持原定义；当前全局dataset文件和原有路由不改写。','',
    '## 模型、校准与选择','',
    '- 从seed42随机初始化，**没有加载任何ID权重、ID图embedding、ID目标校准或近邻索引**。仅复用确定性的原始网表张量缓存和经过验收的消息传递代码。',
    '- 4层、128维、48关系边中心RGCN，节点与器件状态均更新；保留电源/地角色和MOS端口。域嵌入与FiLM启用，输出为可跨兼容拓扑共用的直接/残差神经头。所有8,146,624个参数可训练，不是冻结主干训练。',
    '- 输入仅按本折训练的source/PDK域拟合已知器件数值均值/尺度；目标按本折已验证语义组拟合symlog中心和尺度。掩码只表示真实缺失。目标仿射和损失使用FP64保持低方差数值稳定，图主干和可训练层为FP32；未启用AMP/TF32，确定性算法启用。',
    '- 验证和早停选择使用目标兼容头在**其他拓扑**上的验证hit@5%；完整非目标验证集指标同时记录。验证不包含SimpleOpAmp4。外部目标测试只在选择冻结之后计算，不用于调学习率、选轮数或修改模型。',
    '- AdamW：主干LR2e-4、适配LR3e-4，100步预热，梯度裁剪5；有效Batch256、微批128累积。20轮是安全上限，至少4轮后允许patience5验证早停。','',
    '**可比性限制：** ID的52.05%来自非参数混合解码器，不能直接与本次纯神经LOTO分数相减后声称全部差距都是拓扑迁移造成。已准备相同模型源码、相同优化规程的独立ID神经参考（未启动），其目标归一化也须重新仅用IDtrain拟合；后续主表应优先采用同架构对照，并说明两个协议训练样本数不同。','',
    '## 已运行检查与当前进度','',
    f"真实CPU有效Batch256探针为{checks['samples_per_second']:.1f}样本/秒，估计单轮纯训练{checks['estimated_training_epoch_minutes']:.1f}分钟；正式早期实测约75样本/秒，首轮含验证预计约1.5～2小时。20轮上限在纯CPU上可能需要约30～36小时，早停会缩短。当前只启动这一折，没有同时启动24折或同架构ID参考。",'',
    '- 24套新全量索引均覆盖500000条位置且分区无重叠；所有目标实际连接结构从训练/验证中排除。原设计分组隔离检查已通过。',
    '- 将目标引用故意注入训练时，加载器会在读取其标签之前拒绝；训练校准计数446984与索引一致，未使用验证/目标标签。',
    f"- 同一真实8图批次，整批与微批损失/梯度核对通过；梯度最大绝对误差{checks['micro_gradient_max_absolute_error']:.3g}，容差仍为atol2e-5、rtol0.002，最大验收比{checks['micro_gradient_max_tolerance_ratio']:.4f}；主干梯度非零。CPU探针覆盖全部48关系。",'',
    f"最近检查：第{s.get('epoch')}轮、第{s.get('batch')}/{s.get('batches_per_epoch')}批，累计处理本轮{s.get('samples_done')}条训练样本，来源/校准/实现指纹均匹配。状态里的 `best_epoch=0` 和约10.94%的值目前是训练均值常量初始化在已知兼容拓扑验证集上的基线，不是未知拓扑测试成绩。当前尚无训练后完整验证或LOTO测试结果。",'',
    '## 监督、GPU条件与恢复','',
    '后台监督器核对PID与创建时间，定期检查源码、配置、协议、校准和两个Torch运行时指纹。每100个更新保存可恢复的模型/AdamW/scheduler/RNG/游标；每10批更新进度，验证后按预定规则选择模型。已保存初始与中途断点。', '',
    'CPU使用4线程和有界原图映射缓存，在低内存时安全暂停或退出受影响子进程；可识别资源故障有限恢复，数值/完整性问题不会放宽标准继续。当前CUDA尚未验收；主机实际可用内存至少4GiB且显存足够才允许测试。监督器可先保存并暂停CPU、确认实际内存释放后执行CPU/CUDA预测、节点/边状态、池化、分组Loss、梯度、AdamW检查；通过才迁移。未通过或资源仍不足则继续CPU。', '',
    '同一提前声明的双后端recipe保存两套Torch构建指纹；迁移不改权重、优化器或数据游标，CPU/CUDA不同Dropout随机流会显式记录派生种子，不能宣称跨设备逐步位相等。没有绕过不同源码/recipe的resume保护。', '',
    '## 协议元数据说明','',
    '原v1协议闲置非目标IDtest；全量v2以新的fold.role_policy和NPZ引用取代该规则。v2初版说明中继承的一条v1历史文字和训练覆盖计数未同步展开；实际训练始终按新索引重新拟合，**没有使用旧覆盖数作为统计参数**。为保持运行中recipe完整，不改写其输入文件，已发布绑定源哈希的 `metadata_clarification.json` 与只供查阅的 `manifest_documented.json`。例如本折兼容头实际校准前12目标各2153个、CMRR/PSRR各1476个观测，与新训练索引完全相符。准备脚本也已加已发布目录禁止覆盖的保护，未来修订使用新版本。','',
    '## 文件入口','',
    '- [实时训练状态](../'+cfg['output']+'/status.json) 与 [后台监督](../monitoring/experiment2_loto_v1/'+cfg['fold_id']+'/status.json)。',
    '- [配置](../worktrees/loto_training_v1/config.json)、[本折校准及训练范围证明](../'+cfg['output']+'/normalization.json)、[启动审计](../'+cfg['output']+'/startup_audit.json)。',
    '- [全量协议清晰视图](../data/fusion_v2/splits/verified_loto_full_v2/manifest_documented.json)、[元数据更正](../data/fusion_v2/splits/verified_loto_full_v2/metadata_clarification.json)；原运行输入的SHA仍记录在recipe中。',
    '- [同架构ID对照模板](../worktrees/experiment2_paired_id_v1/comparison_contract.json)，目前未训练。',
    '- [已冻结的第一项结果](rgcn_readout_revision_v3.md)。',
    '', '电脑需保持开机、未睡眠；后台监督不依赖聊天窗口继续采样，但没有安装开机服务。需要停止时创建对应监控目录的STOP，程序会请求断点保存。']
(ROOT/'reports/experiment2_loto_training.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
for name,prefix in [('README.md',''),('reports/README.md','../')]:
    path=ROOT/name;body=path.read_text(encoding='utf-8');body=re.sub(r'<!-- loto-current:start -->.*?<!-- loto-current:end -->\s*','',body,flags=re.S)
    banner=f'''<!-- loto-current:start -->
> **实验二已启动（2026-11-03）：** [公开50万条的留一拓扑训练]({prefix}reports/experiment2_loto_training.md)，首折留出SimpleOpAmp4，train446984 / validation50612 / test2404；不使用私有2万条、不复用ID权重。查看[实时训练]({prefix}{cfg['output']}/status.json)。第一项ID结果已冻结，后文为此前阶段记录。
<!-- loto-current:end -->

''';path.write_text(banner+body,encoding='utf-8')
old_report=ROOT/'reports/verified_loto_protocol.md';text=old_report.read_text(encoding='utf-8');marker='<!-- full-loto-running -->'
if marker not in text:text=marker+'\n> 已启动[全量公开50万条的首折训练](experiment2_loto_training.md)。下文为保留的v1划分准备记录；当前运行使用全量v2角色覆盖，原输入文件不改写。\n\n'+text;old_report.write_text(text,encoding='utf-8')
autonomy=ROOT/'monitoring/training_autonomy.json';a=read(autonomy);a.update(recorded_at=audit['checked_at'],current_action='First public500kLOTO fold actively trains from random initialization; ID results preserved',background_controller_running=True,background_controller_status=str((monitor/'status.json').relative_to(ROOT)).replace('\\','/'),documentation='reports/experiment2_loto_training.md');autonomy.write_text(json.dumps(a,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in audit.items() if k!='folds'},indent=2))
