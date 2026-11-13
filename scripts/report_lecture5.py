"""Build the lecture-5 review and implementation map from real test output."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read


def main():
    r=read(ROOT/'reports/lecture5_implementation.json')
    lines=['# 讲义 5 代码评审与工程落地','',
        '已阅读第 35–41 章，采用目录分层、低显存关系分组消息、FiLM 恒等初始化、分阶段优化、验证早停及诊断的思路。示例代码存在维度、数值和评估口径问题，因此在现有已审计数据接口上修正实现，没有覆盖原始仓库、图或标签。','',
        '## 当前目录','',
        '```text','src/','├─ graph/',
        '│  ├─ netlist.py       # 严格方言适配、续行和 SPICE 数值辅助；无 torch',
        '│  └─ builder.py       # 图张量化、域表、批次及断言；不依赖 torch.nn',
        '├─ model/','│  ├─ layers.py        # 关系分组 RGCN 与注意力池化',
        '│  ├─ film.py          # 域嵌入与零初始化 FiLM',
        '│  ├─ heads.py         # 物理指标族 + 独立监督契约读出',
        '│  └─ network.py       # 完整模型装配与消融开关',
        '├─ train/','│  ├─ losses.py        # 保号变换、掩码与分组损失',
        '│  ├─ metrics.py       # 还原单位后的逐契约/逐指标结果',
        '│  ├─ stage.py         # 预训练、冻结和分层学习率',
        '│  ├─ loader.py        # 可重复的 train/validation/test 加载',
        '│  └─ loop.py          # 训练/验证、余弦调度、早停、最优状态恢复',
        '└─ gnn/               # 原有导入兼容层及受训练覆盖限制的预测接口','```','',
        '原来 `from src.gnn.model import EdgeCentricRGCN` 等调用继续可用；新代码建议直接从 `src.model.network`、`src.graph.builder`、`src.train.*` 导入。没有同时保留两套网络实现。',
        '被 AnalogGym 配方固定哈希的 `src/netlist.py` 等历史模块不移动、不修改，新的 graph/netlist.py 显式调用原有严格方言解析器。仍使用原生 PyTorch 张量批次，不为目录示例额外安装 PyG。','',
        '## 采用与修正','',
        '| 讲义部分 | 处理 |','| --- | --- |',
        '| 35.3 文件结构 | 采用 graph/model/train 分层，并保留老路径导入 |',
        '| 36 解析、关系及断言 | 采用续行、单位辅助和索引/关系/孤立节点/跨图连接检查；保留已有全局词表和严格解析，不静默丢未知器件 |',
        '| 37.2 关系分组计算 | 保留低内存实现。示例 msg_in 是 M×2d，weight 却为 d×d，无法相乘；本实现为每关系 2d×d |',
        '| 37.3 注意力 | 保留逐图稳定 softmax 和注意力返回值，不混合不同图；额外边池化使最后一层边更新参与损失 |',
        '| 37.4 输出顺序 | 不采用示例全局 METRIC_ORDER；继续使用实际语义映射和显式槽位 index_copy，避免混淆 GBW/ft/不同 Gain |',
        '| 38 FiLM | 新增小方差域嵌入、每层独立 FiLM，gamma=1+delta、beta=0 初始化，放在节点 LayerNorm 之后、边更新之前 |',
        '| 38.4 阶段优化 | 按真实模块对象识别主干，不靠名称前缀猜冻结；换阶段重建优化器并检查可训练参数数量，重返 pretrain 会重新解冻 |',
        '| 39 变换与损失 | 保留严格互逆 symlog/symexp；不采用示例的错误 epsilon 逆变换；先索引已知标签再作算术，防止 NaN×0 污染 |',
        '| 39.3 评估 | 还原原有计量单位后逐监督组、逐语义计算，不把 W、Hz、dB 或不同 Gain 混成一个原始 R²；常量/单点 R² 为 None |',
        '| 39.4 训练 | 新增 split_role 训练隔离、梯度裁剪、余弦调度、验证早停及深拷贝最优权重恢复；最优状态对应的训练覆盖也同步恢复 |',
        '| 39.5 平滑度 | 记录余弦相似度；只在诊断中排除 trivial self-loop，不删图上的端口自环 |',
        '| 40 装配与消融 | 新增域、FiLM、关系折叠开关，保持原有边状态冻结开关和旧 checkpoint 基线结构 |',
        '| 41 结果分析 | 作为后续实验报告规范；讲义表格及示例数值不当作本项目实测，负面结果与各折结果必须如实保留 |','',
        '没有采用的示例细节：按端口表前半推断输入、直接把 VSS 当作零电位、缺失 W/L 默认填 1、静默忽略未知器件、每图动态注册关系、直接删除端口自环。这些做法会削弱当前图审计和可追溯性。示例的关系注册说明要求在删除自环之前调用 rid，但代码实际在 continue 之后调用，不能照搬。',
        '数学修正：log1p(W)-log1p(L) 不等于 log(W/L)；symlog 中若使用 log1p(abs(y)+eps)，逆变换也不是 expm1(abs(z)-eps)。项目继续使用不加该 epsilon 的严格保号变换及明确零值标记。示例 head 声明 2d 输入但给 d 维图表示，也未采用。',
        '科学表述修正：预期某种域适应方法可能失败不等于可以预设负面实验结果；不采用“曲线应该低”这样的结论约束。R² 是否可定义取决于目标方差，不取决于目标均值是否接近零。','',
        '## 域与监督的界限','',
        '域表从真实 catalog 的“数据来源 × process”生成，不复制讲义中占位的 65nm、自建 n28 等示例。未知工艺保留 unknown，不凭名称补 PDK。域信息用于调制表示，不批准跨来源同名标签共享。','',
        '| domain_id | 来源 | 实际工艺记录 |','| ---: | --- | --- |']
    for d in r['domains']:lines.append(f"| {d['id']} | {d['source_dataset']} | `{d['process']}` |")
    lines+=['',
        '当前仍是 142 个保守监督契约，最终读出独立。FiLM 不会把不同增益定义或测试条件变成可比标签。未参与优化的输出槽位、域嵌入和监督头不可用于正式评估。LOTO/Cross-Dataset 仍需先审核共享目标和读出覆盖。','',
        '## 训练阶段与真实参数量','',
        '| 阶段 | 总参数 | 可训练参数 | 学习率 |','| --- | ---: | ---: | --- |']
    for stage,c in r['stage_parameter_counts'].items():lines.append(f"| {stage} | {c['total']:,} | {c['trainable']:,} | {c['lr_groups']} |")
    lines+=['',
        'finetune_head 冻结节点/边/关系编码器和 RGCN 层，训练池化、共享/分组头、域嵌入和 FiLM。这里的“head”是适配阶段名，不表示只训练最后一个 Linear。finetune_all 对上述主干和适配侧采用不同学习率。',
        '可用消融：use_domain_emb=False 且 use_film=False；use_domain_emb=True/use_film=False；update_edges=False；collapse_relations=True。最后一项只去掉关系专属矩阵和关系嵌入，并未移除节点端口计数，不能声称彻底抹除了全部端口信息。','',
        '## 评估口径','',
        '误差先从 symlog 空间按固定工程参考尺度还原。R²/MAE 按监督契约与语义分开累计；单点和零方差 R² 不定义。MRE、hit@5%、hit@10% 只用于非对数单位且目标远离零的条目，近零排除数逐指标记录。dB/dBm 等用 MAE 和绝对 1 个原单位的命中率，不用其数值的百分比误差冒充电压/功率百分比误差。',
        '早停优先用可定义目标的 macro_hit5；若任务全为 dB 等不适用相对误差的指标，显式退回负验证损失，日志记录实际准则。指标只表示现有定义下的数值性能，不声明已验证其工程有效性。',
        '训练时采用逐批契约损失平均；epoch 日志再按契约累计误差分子/已知标签数，避免批次大小或缺失比例导致日志偏置。没有将不同量纲的 MAE 平均成一个数，也没有拟合全体数据的标准化参数。','',
        '## 实际运行','',
        f"已使用 {r['train_samples']} 条 train 样本、{r['validation_samples']} 条 validation 样本运行 {r['epochs']} 轮，test 样本使用数为 {r['test_samples_used']}。最优权重来自 epoch={r['best_epoch']}（从 0 计数）并已恢复。",
        '这是模块接线、训练/验证隔离和早停流程检查，不是正式泛化实验，也没有执行完整 50 万条训练。只检查了三阶段优化器配置；没有把三阶段都训练成性能基准。',
        f"Checkpoint：`{r['checkpoint']}`，状态显式为 training_loop_smoke_only；未经显式 allow_smoke=True，预测接口拒绝将其当作正式模型。逐轮 loss、梯度范数、平滑度、时间、验证逐指标结果和学习率见 `reports/lecture5_implementation.json`。",'','```powershell',
        r'.\.venv\Scripts\python.exe -B scripts/check_lecture5.py',
        r'.\.venv\Scripts\python.exe -B scripts/report_lecture5.py',
        r'.\.venv\Scripts\python.exe -B -m unittest discover -s tests','```','',
        'FiLM 开关配置：`config/edge_rgcn_film.json`，继承基础 `config/edge_rgcn.json`。基础无 FiLM 检查仍用 scripts/check_edge_rgcn.py，原有构建报告作为对照。','',
        '## 新调用入口示例','',
        '```python','from src.model.network import EdgeCentricRGCN',
        'from src.graph.builder import input_schema, task_registry',
        'from src.train.stage import build_optimizer','',
        'schema, tasks = input_schema(dataset), task_registry(dataset)',
        'model = EdgeCentricRGCN(schema, tasks, use_domain_emb=True, use_film=True)',
        'optimizer = build_optimizer(model, stage="pretrain")',
        '# 切换阶段须新建优化器；不能继续使用旧 optimizer',
        'optimizer = build_optimizer(model, stage="finetune_head")','```']
    (ROOT/'reports/lecture5_implementation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
