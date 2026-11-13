# 已冻结的公开融合 ID 模型

结果与范围见 [实测报告](../../reports/rgcn_readout_revision_v3.md)。本版本为 4层/48关系 RGCN＋训练集局部解码器，验证 hit@5%=52.00%，固定测试=52.05%。原始测试 macro R²仍有严重负值，不能称为全指标工程合格。仅支持已知契约与规范器件图，不能用于未知拓扑零样本结论。

`publication.json` 固定代码和 bundle 哈希；bundle 指向被冻结的原 GNN checkpoint 及每契约训练索引。索引为模型的一部分，仅含训练记录；不需要验证/测试标签参与推理。原模型/数据均保留。

在工作区的独立 Python 进程中加载；`batch` 沿用已定义的 `CircuitBatch`：

```python
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path('versions/rgcn_local_id_v1').resolve()))
from model import load_model
model, bundle = load_model(device='cpu')
with torch.no_grad():
    z = model(batch)
    prediction = z.sign() * torch.expm1(z.abs()) * batch.scales.to(torch.float64)
```

输入不符合已训练契约的规范图连接、端口关系或节点角色时会拒绝局部解码。CPU 入口已通过142契约实图核对；本次没有进行新版 GPU 验收，不把 CUDA 接口可传入解释为已经通过 GPU 测试。

固定测试已经完成，不再根据其分数修改本发布版本。后续结构或实验须使用新版本和新输出目录。
