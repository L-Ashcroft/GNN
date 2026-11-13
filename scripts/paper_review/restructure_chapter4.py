"""Restructure only chapter IV; retain native equations and move detail to appendix.

No training, simulation, or metric computation. All displayed scores are copied
from the input manuscript and checked against existing archival summaries.
"""
from pathlib import Path
from zipfile import ZipFile
from xml.dom import minidom as D
import hashlib, json, re

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / 'background/模拟电路论文新_编辑副本.docx'
QA = ROOT / 'artifacts/paper_v6_review_20261112'
BACKUP = QA / 'chapter4_input.zip'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
read = lambda p: json.loads((ROOT / p).read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

if not BACKUP.exists(): BACKUP.write_bytes(TARGET.read_bytes())
prior = json.loads((QA/'changes.json').read_text(encoding='utf-8')).get('chapter4_restructure_20261113',{})
assert sha(TARGET) in (sha(BACKUP), prior.get('output_sha256')), 'Do not overwrite subsequent author edits'
with ZipFile(BACKUP) as z: doc = D.parseString(z.read('word/document.xml'))
body = doc.getElementsByTagName('w:body')[0]
nodes = [e for e in body.childNodes if e.nodeType == 1]
def txt(e):
    return ''.join(t.firstChild.data for t in e.getElementsByTagName('*')
                   if t.tagName in ('w:t', 'm:t') and t.firstChild)
assert txt(nodes[64]) == 'IV. 实验设置与结果'
assert txt(nodes[143]) == 'V. 讨论'
assert txt(nodes[153]) == '参考文献'
protected = [e.toxml() for e in nodes[:64] + nodes[143:]]
equations = [e.toxml() for e in doc.getElementsByTagName('m:oMath')]

# Read existing results only. No re-evaluation or inference is performed.
evidence_paths = [
    'background/falcon_2505.21923.pdf',
    'artifacts/experiment1_fusion_id_v1/recipe.json',
    'data/model_ready/fusion_v2_frozen_e19_bf0d71ad/manifest.json',
    'artifacts/experiment1_autonomous_head_v1/status.json',
    'artifacts/experiment1_joint_cpu_refine_v2/recipe.json',
    'artifacts/experiment1_joint_cpu_refine_v2/status.json',
    'artifacts/experiment1_rgcn_local_head_v1/selection.json',
    'artifacts/loadbench_remediation_v1/evaluation_receipt.json',
    'artifacts/experiment2_loto_v1/LB_M04__hold_SimpleOpAmp4/recipe.json',
    'artifacts/experiment2_loto_v1/LB_M04__hold_SimpleOpAmp4/selection_frozen.json',
    'artifacts/experiment4_personal_transfer_v1/pretrained_k100_seed42/recipe.json',
    'artifacts/falcon_reference_eval_v1/public_id_published/summary.json',
    'artifacts/falcon_reference_eval_v1/loto_M04/summary.json',
    'artifacts/falcon_reference_sensitivity_v1/sensitivity.json',
    'config/falcon_reference_eval_v1.json',
    'src/falcon_reference_metrics.py', 'src/train/metrics.py',
]
for p in evidence_paths: assert (ROOT / p).is_file(), p
assert read(evidence_paths[2])['parent_epoch'] == 19
assert read(evidence_paths[9])['best_epoch'] == 20
assert read(evidence_paths[11])['overall']['samples'] == 49056
assert read(evidence_paths[12])['overall']['samples'] == 2404

def el(tag, **attrs):
    ns = M if tag.startswith('m:') else W
    e = doc.createElementNS(ns, tag)
    for k, v in attrs.items(): e.setAttributeNS(ns, tag[:2]+k, str(v))
    return e
def first(e, tag): return next((n for n in e.childNodes if n.nodeType == 1 and n.tagName == tag), None)
def prop(e, tag, **attrs):
    p = first(e, tag)
    if p is None: p = el(tag); e.appendChild(p)
    for k, v in attrs.items(): p.setAttributeNS(W, 'w:'+k, str(v))
    return p
def run(text, size=21, bold=False):
    r = el('w:r'); rp = el('w:rPr'); r.appendChild(rp)
    rp.appendChild(el('w:rFonts', ascii='Times New Roman', hAnsi='Times New Roman', eastAsia='宋体'))
    rp.appendChild(el('w:sz', val=size)); rp.appendChild(el('w:szCs', val=size))
    rp.appendChild(el('w:color', val='000000')); rp.appendChild(el('w:highlight', val='yellow'))
    if bold: rp.appendChild(el('w:b'))
    t = el('w:t'); t.setAttribute('xml:space', 'preserve'); t.appendChild(doc.createTextNode(text)); r.appendChild(t)
    return r
def para(s, template=69, size=21, bold=False):
    p = el('w:p')
    pp = first(nodes[template], 'w:pPr')
    if pp: p.appendChild(pp.cloneNode(True))
    p.appendChild(run(s, size, bold)); return p
def heading(s): return para(s, 65, bold=True)
def caption(s):
    p = para(s, 75); pp = first(p, 'w:pPr')
    if pp is None: pp = el('w:pPr'); p.insertBefore(pp, p.firstChild)
    prop(pp, 'w:keepNext', val=1); return p
def note(s):
    p = para(s, 77, size=19)
    pp = first(p, 'w:pPr'); prop(pp, 'w:ind', firstLine=0, firstLineChars=0)
    return p
def table(rows, widths):
    t = el('w:tbl'); pr = first(nodes[76], 'w:tblPr').cloneNode(True)
    prop(pr, 'w:tblW', w=sum(widths), type='dxa'); t.appendChild(pr)
    grid = el('w:tblGrid'); t.appendChild(grid)
    for w in widths: grid.appendChild(el('w:gridCol', w=w))
    for i, row in enumerate(rows):
        tr = el('w:tr'); trp = el('w:trPr'); trp.appendChild(el('w:cantSplit'))
        if i == 0: trp.appendChild(el('w:tblHeader'))
        tr.appendChild(trp)
        for j, value in enumerate(row):
            tc = el('w:tc'); tcp = el('w:tcPr'); tc.appendChild(tcp)
            tcp.appendChild(el('w:tcW', w=widths[j], type='dxa'))
            tcp.appendChild(el('w:vAlign', val='center'))
            if i == 0: tcp.appendChild(el('w:shd', val='clear', fill='EDEFF2'))
            p = el('w:p'); pp = el('w:pPr'); p.appendChild(pp)
            pp.appendChild(el('w:spacing', before=45, after=45, line=240, lineRule='auto'))
            pp.appendChild(el('w:jc', val='center' if j else 'left'))
            pp.appendChild(el('w:ind', left=0, right=0, firstLine=0, firstLineChars=0))
            if i == 0: pp.appendChild(el('w:keepNext'))
            p.appendChild(run(value, size=19, bold=i == 0)); tc.appendChild(p); tr.appendChild(tc)
        t.appendChild(tr)
    return t
def clone_table(i):
    t = nodes[i].cloneNode(True)
    # Keep rows intact, repeat headers, but allow a table to continue across pages.
    for j, tr in enumerate(t.getElementsByTagName('w:tr')):
        trp = first(tr, 'w:trPr')
        if trp is None: trp = el('w:trPr'); tr.insertBefore(trp, tr.firstChild)
        prop(trp, 'w:cantSplit')
        if j == 0: prop(trp, 'w:tblHeader')
        for p in tr.getElementsByTagName('w:p'):
            pp = first(p, 'w:pPr')
            if pp is not None:
                for kn in list(pp.getElementsByTagName('w:keepNext')):
                    if j: kn.parentNode.removeChild(kn)
    return t

out = [nodes[64].cloneNode(True), nodes[65].cloneNode(True)]
def add(s): out.append(para(s))
add('本文设置分布内预测（in-distribution，ID）、留一拓扑泛化（leave-one-topology-out，LOTO）、跨数据集零样本评价和少样本适配四类协议，见表IV。零样本部分目前完成监督兼容性审查，尚无预测精度结果。')
out += [nodes[67].cloneNode(True), clone_table(68)]
# Keep the shared protocol table; clarify the unexecuted zero-shot row.
protocol_row = out[-1].getElementsByTagName('w:tr')[3]
cells = protocol_row.getElementsByTagName('w:tc')
for cell, value in ((cells[1], '仅公开数据训练；自建标签不用于更新或选模'),
                    (cells[3], '监督兼容性审查；尚未预测')):
    p = cell.getElementsByTagName('w:p')[0]
    for child in list(p.childNodes):
        if child.nodeType == 1 and child.tagName != 'w:pPr': p.removeChild(child)
    p.appendChild(run(value, size=20))
add('各协议固定数据分区，同一设计组不跨训练、验证和测试集；归一化统计仅由相应训练集拟合，模型按验证表现选择后冻结，再评价测试集。除k=100另用种子43、44重复外，随机种子为42；各适配预算共用固定验证集和测试集，验证标签计入监督预算。')
add('评价采用决定系数R²、平均绝对误差（MAE）、均方根误差（RMSE）及5%／10%相对误差命中率（hit@5%／hit@10%）。命中率排除对数单位及低于固定工程阈值的近零标签，先逐监督任务计算，再对有效任务等权汇总；R²也逐任务统计。参照FALCON[5]，另报告样本平均相对误差（MRE），按式（7）先在样本内平均，再对样本等权平均。')
out += [nodes[71].cloneNode(True), nodes[72].cloneNode(True)]
add('其中，Jᵢ为原Mask为真且原单位标签非零的指标集合，I为至少含一项此类指标的样本集合，N=|I|；预测值与标签均先还原至来源原单位。MRE、中位误差和P95取自同一组样本误差eᵢ，主结果不截尾。固定常数10⁻⁸的影响依赖数值尺度；dB、dBm的数值相对误差也不等于线性增益或功率误差。因此，MRE与命中率采用不同有效集和权重，需结合物理单位误差解释，不能单独用来排名不同来源。')

out.append(heading('B. 实验一：分布内性能预测（ID）'))
add('ID任务考察已知拓扑内器件参数变化下的性能预测。公开数据集按来源、拓扑及参数分布分层划分为400,083条训练、50,861条验证和49,056条测试记录，测试设计不参与参数拟合或邻居索引构建。')
add('模型先端到端训练图编码器与分组预测头，初始采用AdamW，学习率2×10⁻⁴、权重衰减0.01、批大小256，最多100轮，早停耐心20轮。后续读出比较固定第19轮的图主干；神经头按验证集选取，联合微调未超过保留模型，因而不替换编码器。冻结后分别拟合神经预测头、器件参数近邻、图表示与参数联合近邻，以及训练内校准的局部回归。各阶段的选模规则与检查点对应关系见附录A。')
out.append(caption('表V 固定ID测试的预测性能与样本误差'))
out.append(table([
    ['来源', '记录数', 'hit@5%', 'hit@10%', 'R²中位数', '完整MRE（%）', '中位误差（%）'],
    ['FALCON', '19,829', '78.00%', '90.07%', '0.987', '6.36', '2.29'],
    ['AICircuit', '9,926', '71.85%', '85.33%', '0.956', '12.56', '2.34'],
    ['LOADBench', '19,301', '51.92%', '63.93%', '0.813', '8.9837×10⁷', '12.45'],
    ['总体', '49,056', '54.32%', '66.38%', '0.850', '3.5346×10⁷', '4.27'],
], [1350, 850, 1050, 1050, 1050, 1800, 1350]))
out.append(note('注：命中率按监督任务宏平均，MRE与中位误差按样本等权。FALCON、AICircuit沿用冻结的图—参数近邻预测，LOADBench采用训练内校准局部回归；三者任务及输出配置不同，不构成严格同条件比较。完整P95、R²均值与负值项数见表A-I、A-II。'))
add('表V中FALCON和AICircuit的典型样本误差较小，LOADBench则呈明显长尾：其中位误差为12.45%，P95达1349.15%。总体R²中位数为0.850，均值却为−2703.721，38项负R²任务均来自LOADBench。较好的典型拟合与少数严重失预测同时存在，来源间差距还混合了任务、样本分布和读出设置的影响。')
out.append(caption('表VI 冻结编码器后的读出方式比较'))
out.append(clone_table(86)); out.append(note(txt(nodes[87])))
add('表VI中，参数近邻的验证命中率为50.10%，加入图表示后为52.00%，训练内校准局部回归的固定测试结果为54.32%。读出方式对已知拓扑的参数插值有较大影响，但各方案的拟合预算不同，全部增益不能归于GNN。局部回归使用32个已观测邻居、岭系数0.01和验证选定的全局混合权重0.5；更完整的距离与校准设置见附录A。')
add('在同一LOADBench测试集上，监督邻域混合专家相较训练范围约束读出将hit@5%由52.07%提高至55.83%，R²中位数由0.813升至0.826；负R²项却由38增至43，R²均值由−2643.654降至−2645.134，中位样本误差降至11.27%而P95增至2016.49%。该方法改善了严格容差内的预测，但未控制尾部风险；与表V所用校准局部回归分别报告，详细任务统计见表A-III。')
add('极端误差还与指标提取支持有关。SimpleOpAmp47_2_pfc2一条历史记录的存档ft为0.008057 Hz，预测约1.93 GHz，独占总体样本误差和的98.85%，最高5条累计占99.25%。完整MRE保留全部样本；仅用于敏感性分析时，去除最大单条后的描述性MRE仍为407,682.95%。对该设计在1 Hz–1 GHz内回放AC分析未发现单位增益交越，官方提取函数可在无交越时外推；本机重提取值又与历史存档不一致，故只能确认测量支持不足，不能据此判定存档标签错误。正常对照与存档基本吻合。逐指标误差、误差集中度与波形复核细节见附录A，原标签及Mask均保留。')

out.append(heading('C. 实验二：留一拓扑泛化（LOTO）'))
add('本实验完整留出LOADBench的SimpleOpAmp4及其等价连接结构，训练446,984条、验证50,612条，目标拓扑全部2,404条仅用于测试；结构同构检查未发现须额外排除的跨来源别名。其余拓扑中，SimpleOpAmp6具有已核验兼容的测试平台及提取定义，可为14项共同指标提供监督。兼容契约仅在本协议中归并，模型共使用129个监督组。')
add('LOTO模型以种子42独立初始化，不继承ID编码器或预测头。采用第三部分式（6）的标准化掩码损失，AdamW主干／适配参数学习率为2×10⁻⁴／3×10⁻⁴，权重衰减0.01，有效批大小256；最多20轮、至少4轮，早停耐心5，以其余拓扑兼容头的验证hit@5%选模。该折完成20轮并选择第20轮，训练与统计拟合均未使用留出拓扑标签。')
add('固定测试hit@5%和hit@10%分别为31.28%和42.81%，R²中位数为0.915、均值为0.448；完整MRE为295.56%，中位误差26.90%，P95为403.66%。表VII显示指标间差异：功耗的R²达0.990，严格命中率仍仅17.72%，负向压摆率则出现负R²。趋势拟合与严格容差控制并不等价。')
out.append(caption('表VII SimpleOpAmp4单折LOTO的代表性指标'))
oldrows = nodes[115].getElementsByTagName('w:tr')
def rowtext(row): return [txt(c) for c in row.getElementsByTagName('w:tc')]
selected = [rowtext(oldrows[i]) for i in (0,1,2,3,4,5)]
selected = [[row[j] for j in (0,2,3,4,5)] for row in selected]
out.append(table(selected, [2600, 1400, 1600, 1350, 1550]))
out.append(note('注：各行均有2,404条有效观测；MAE、RMSE单位同指标。存档ft沿用原提取值，对数单位不计算相对命中率。完整14项结果与缺失观测数见表A-VI。'))
add('目前24套预设留出方案仅完成这一折，结果限于SimpleOpAmp4。ID使用已知拓扑局部读出，LOTO使用独立训练的神经头，训练规模也不同；两者差距同时包含拓扑留出和输出配置变化，尚不能作为纯粹的跨拓扑泛化损失。')

out.append(heading('D. 实验三：跨数据集零样本评价与监督兼容性'))
add('零样本评价要求仅用公开数据集训练，不使用自建数据集标签更新模型、拟合输出映射或选择模型。为此，须先核验共同性能目标，并建立公开模型已受监督输出与自建指标之间的映射。工艺、偏置、负载或温度变化可作为迁移所研究的分布差异，并不自动阻止评价；同名指标也不自动满足共享预测头的条件。')
add('现有审查覆盖142个公开监督契约、自建数据集2个契约及28项指标定义，尚无确认可共享的监督目标。例如，LOADBench的1 Hz起始AC提取与自建放大器0.1 Hz响应仍需核验低频含义；压摆率在幅度阈值及符号、电源响应在比值方向及频点上也有差异。两者同用SKY130但模型束、负载不同属于域差异，与上述语义问题分别记录（表A-VII）。当前尚未建立经核验的输出映射，因此没有可报告的零样本预测精度；后续应先确定共同目标和未见输入的处理规则，再开展不使用自建标签选模的测试。')

out.append(heading('E. 实验四：少样本适配（Few-shot）'))
add('自建数据集含10种拓扑、共20,000条记录，每拓扑按1,600／200／200划分训练、验证与测试。在固定训练池内嵌套选取每拓扑k=5、10、20、50、100条用于参数更新；各预算共用2,000条验证和2,000条测试记录。实际监督预算为10k+2,000，因而k=100对应3,000条标签，而非仅1,000条。')
add('端到端对照仅改变初始化方式：公开预训练微调按第三部分的名称映射迁移图主干，另一条路线随机初始化；自建预测头、域嵌入、FiLM和条件编码器均重新初始化。两者采用相同样本、标准化掩码损失与AdamW配置，主干／适配学习率为2×10⁻⁴／3×10⁻⁴、权重衰减0.01、有效批大小64，最多20轮、早停耐心5。各预算只用相应训练标签拟合统计量，并按验证hit@5%选模。另将各预算已选编码器冻结，加入器件参数与已知条件，拟合拓扑局部混合读出；其训练配置及独立的标签表示见附录A。')
out.append(caption('表VIII 少样本监督预算与固定测试hit@5%'))
out.append(clone_table(131)); out.append(note(txt(nodes[132]).replace('目标域', '自建数据集')))
add('表VIII中，端到端预训练模型随k增加总体改善，但并非每个预算均优于随机初始化。k=100的冻结混合读出分别达到47.29%和47.75%，两条路线均高于各自端到端结果，改善主要体现输出映射的作用，而非公开预训练优势。')
add('在k=100下，种子42、43、44的端到端预训练与随机初始化命中率均值±样本标准差分别为25.11%±0.60个百分点和24.45%±0.23个百分点，平均相差0.66个百分点。完整MRE分别为104.93%±5.09个百分点和106.98%±1.67个百分点，但种子43的预训练MRE反而更高。三次重复仅描述固定划分下的初始化波动，预训练收益有限且随评价指标变化；逐种子结果见表A-VIII。')
out.append(caption('表IX 自建数据集全量监督与输入分支比较'))
out.append(clone_table(140))
out.append(note('注：均使用16,000条训练、2,000条验证及同一2,000条测试记录，总监督预算18,000条。命中率按原始值保留三位小数；“—”为未按新增MRE口径统计，不表示误差为零。'))
add('全量监督下，图特征与参数、条件混合读出的命中率由66.03%提高至67.69%，中位误差由5.61%降至4.87%，完整MRE却由23.28%升至26.71%，R²中位数也略降。图表示补充了部分典型样本的信息，但未改善平均相对误差。这一输入分支对照尚不能分离端口关系、动态边更新或FiLM的独立贡献；与公开预训练的收益也应分别评价。固定测试在先前模型迭代中的重复使用及其他研究限制集中在第五部分讨论。')

# A same-file appendix retains information removed from the main experiment narrative.
app = [heading('附录A 实验配置与补充结果')]
def ap(s): app.append(para(s))
def orig(i): app.append(nodes[i].cloneNode(True))
ap('本附录列出训练、读出与误差复核的补充设置。以下相对路径仅说明本地复现归档位置，不代表材料已经公开发布。工程尺度与任务映射分别记录于config/performance_mapping.json、data/model_ready/fusion_v2_training_v1/tasks.json；MRE单位还原及零值规则见config/falcon_reference_eval_v1.json和src/falcon_reference_metrics.py。hit@5%与hit@10%的近零界限为固定工程尺度的10⁻⁸倍，对数单位不计算相对命中率；这与MRE仅排除零标签并在分母加10⁻⁸的规则不同。')
app.append(heading('A. 训练与读出配置'))
ap('公开ID初始端到端训练使用AdamW（学习率2×10⁻⁴、权重衰减0.01、批大小256、最多100轮、早停耐心20），带符号对数空间的分组均方误差不作逐指标标准差归一化。冻结特征缓存对应第19轮主干；冻结神经头续训的最佳总轮数为36。随后联合微调采用主干2×10⁻⁶、预测头10⁻⁵的学习率，有效批大小256，耐心3轮；三轮未超过保留模型，故继续使用原主干。配置和选择记录分别见artifacts/experiment1_fusion_id_v1/recipe.json、data/model_ready/fusion_v2_frozen_e19_bf0d71ad/manifest.json及artifacts/experiment1_joint_cpu_refine_v2/status.json。')
for i in (88,89,90,92):
    s=txt(nodes[i]).replace('表VIII','表V').replace('表X','表A-III')
    ap(s)
ap('LOTO采用随机初始化，主干与适配参数学习率2×10⁻⁴和3×10⁻⁴，权重衰减0.01，前100次更新预热，最小学习率2×10⁻⁶；有效批大小256，以128条微批累积。最多20轮、至少4轮，早停耐心5，最小验证改进10⁻⁴，以兼容头验证命中率选择第20轮。兼容目标头前12项各有2,153条训练观测，CMRR与PSRR各1,476条；CPU训练循环累计约39.2小时。配置与选择记录见artifacts/experiment2_loto_v1/LB_M04__hold_SimpleOpAmp4/recipe.json及同目录selection_frozen.json。')
ap('少样本端到端训练使用有效批大小64、微批16，主干／适配学习率2×10⁻⁴／3×10⁻⁴，权重衰减0.01，最多20轮、至少4轮，耐心5，验证改进阈值10⁻⁴。标签采用本预算训练集的带符号对数中心与标准差，标准差下限0.1；按拓扑×契约×指标等权验证。k=100、种子42的预训练配置与选模记录位于artifacts/experiment4_personal_transfer_v1/pretrained_k100_seed42/recipe.json及同目录selection_frozen.json，所选为第15轮；其他预算与种子各自独立选模。')
for i in (128,129):
    s=txt(nodes[i]).replace('目标域','自建数据集').replace('表XVI','表VIII')
    ap(s)
ap('三随机种子的样本标准差描述固定划分下的训练随机性；成对拓扑聚类Bootstrap区间描述同一已冻结预测在测试拓扑重采样下的不确定性，两者均不单独构成统计显著性的证据。')
app.append(heading('B. 分来源结果、误差及标签复核'))
for cap, idx, nidx in [
    ('表A-I 固定ID测试的样本误差分布',76,77),
    ('表A-II 固定ID测试的任务宏平均结果',80,81),
    ('表A-III LOADBench训练范围约束读出与监督邻域混合专家',94,None),
    ('表A-IV LOADBench代表性指标的物理单位误差',101,102),
    ('表A-V 高误差记录对完整样本相对误差和的贡献',106,107),
]:
    app += [caption(cap), clone_table(idx)]
    if nidx: app.append(note(txt(nodes[nidx]).replace('表VII','表A-I').replace('表VIII','表A-II')))
    if idx==94:
        app.append(note('任务按拓扑、槽位和原指标名配对，原Mask下至少有1条有限标签与有限预测；不以R²有定义筛除RMSE。92×14个可能组合中，SimpleTwoStageOpAmp_SimpleOpAmp39_7的CMRR与PSRR无测试观测，余1,286项可计算RMSE，其中1,274项R²有定义、1,012项适用相对命中率。按未舍入RMSE比较，827项下降、325项上升、134项不变；此池对应表A-II的LOADBench部分。'))
        ap(txt(nodes[97]).replace('表VII','表A-I').replace('表X','表A-III'))
    if idx==101:
        ap(txt(nodes[103])); ap(txt(nodes[104]))
    if idx==106: ap(txt(nodes[108]))
app.append(heading('C. LOTO、兼容性与少样本补充结果'))
for cap, idx, nidx in [
    ('表A-VI SimpleOpAmp4单折LOTO的完整逐指标误差',115,116),
    ('表A-VII 公开数据集与自建数据集的监督兼容性核验',122,None),
    ('表A-VIII k=100端到端适配的三随机种子结果',136,137),
]:
    app += [caption(cap),clone_table(idx)]
    if nidx: app.append(note(txt(nodes[nidx])))
ap('冻结ID预测及读出选择见artifacts/experiment1_rgcn_local_head_v1/selection.json、artifacts/loadbench_remediation_v1/evaluation_receipt.json；MRE完整统计见artifacts/falcon_reference_eval_v1/，极端记录追溯、单位还原和两次AC回放见artifacts/falcon_reference_sensitivity_v1/sensitivity.json。')

# Small result tables stay together. Larger appendix tables may continue with
# repeated headers; no fixed row heights or forced font reduction are used.
for t in (e for e in out+app if e.tagName=='w:tbl'):
    rows=t.getElementsByTagName('w:tr')
    if len(rows)<=7:
        for i,tr in enumerate(rows):
            for p in tr.getElementsByTagName('w:p'):
                pp=first(p,'w:pPr')
                if pp is None:pp=el('w:pPr');p.insertBefore(pp,p.firstChild)
                prop(pp,'w:keepNext',val=1 if i<len(rows)-1 else 0)

anchor=nodes[143]
for n in nodes[64:143]: body.removeChild(n)
for n in out: body.insertBefore(n,anchor)
for n in app: body.insertBefore(n,nodes[153])

# Structural and archival correctness gates before writing.
for original in protected: assert original in [e.toxml() for e in body.childNodes if e.nodeType==1]
assert equations == [e.toxml() for e in doc.getElementsByTagName('m:oMath')]
alltext=txt(body)
captions=re.findall(r'表(?:A-[IVX]+|[IVX]+) ',alltext)
main_captions=[txt(e).split(' ')[0] for e in body.childNodes if e.nodeType==1 and re.match(r'^表(?:A-)?[IVX]+ ',txt(e)) and e.tagName=='w:p']
assert main_captions==['表I','表II','表III','表IV','表V','表VI','表VII','表VIII','表IX']+['表A-'+s for s in ('I','II','III','IV','V','VI','VII','VIII')],main_captions
assert not re.search(r'表(?:X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII)(?![A-Z])',alltext)
paths=set(re.findall(r'(?:config|data|artifacts|src)/[A-Za-z0-9_./\-]+', ''.join(txt(e) for e in app)))
assert all((ROOT/p).exists() for p in paths),[p for p in paths if not (ROOT/p).exists()]
oldtext=''.join(txt(e) for e in nodes[64:143])
newtext=''.join(txt(e) for e in out)
archive=QA/'changes.json';record=json.loads(archive.read_text(encoding='utf-8'))
with ZipFile(BACKUP) as z:
    contents={i.filename:(i,z.read(i.filename)) for i in z.infolist()}
contents['word/document.xml']=(contents['word/document.xml'][0],doc.toxml(encoding='utf-8'))
with ZipFile(TARGET,'w') as z:
    for name,(info,data) in contents.items(): z.writestr(info,data)
with ZipFile(TARGET) as z: assert z.testzip() is None
result={
    'input_sha256':sha(BACKUP),'output_sha256':sha(TARGET),
    'backup':str(BACKUP.relative_to(ROOT)), 'scope':'chapter IV; same-file appendix added before references',
    'protected_chapters_unchanged':True,'all_native_equations_unchanged':len(equations),
    'before_chapter4_characters_including_tables':len(oldtext),
    'after_chapter4_characters_including_tables':len(newtext),
    'main_tables_chapter4':6,'appendix_tables':8,'table_captions':main_captions,
    'all_appendix_paths_exist':sorted(paths),
    'evidence_sha256':{p:sha(ROOT/p) for p in evidence_paths},
    'changes':[
        '保留A节协议总表和原生式（7），补全MRE与命中率口径区别；特有配置移回各实验。',
        '原B/C/D合并为ID，保留两张核心表；原E/F/G依次改为C/D/E。',
        'ID分来源容差与样本误差合表；保留负R²、MRE尾部及AC诊断边界。',
        'LOTO保留单折全部概要和5项代表指标，14项明细移表A-VI。',
        '零样本改为简短兼容性评价，不报告尚无依据的预测分数。',
        '少样本保留预算表与全量对照表，三种子均值留正文、明细移表A-VIII。',
        '原有详细读出、物理误差、敏感性和兼容性转入同文件附录；主表I–IX连续编号。',
        '第一至第三章、摘要、讨论、结论、参考文献逐块XML保持不变；没有新增实验。',
    ],
    'not_resolved_by_restructuring':[
        '仅一折LOTO有冻结测试结果；其余进行中结果未纳入。',
        '尚缺同条件单源/多源、简单模型及结构消融的完成结果。',
        '未确定会议模板，正文及附录尚未强制压缩至6页。',
        '前三章原文中的术语及个别标点不在此次修改范围内。',
    ]}
record['chapter4_restructure_20261113']=result
record['canonical_manuscript_sha256']=sha(TARGET)
archive.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('evidence_sha256','all_appendix_paths_exist')},ensure_ascii=False,indent=2))
