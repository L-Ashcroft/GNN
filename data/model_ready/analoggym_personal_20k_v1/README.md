# 个人2万条数据的GNN派生视图

原始数据不复制、不改写，仍在 `data/generated/analoggym_personal_20k_v1/`。本目录保存10个增强图、catalog、任务与schema、原始/派生文件SHA清单。默认采用原始目录的 `splits/anchor_safe_v1/`。

节点35维、边90维、条件46维、52种消息关系。电源/地等角色按冻结测试平台位置绑定标记；LDO三端PDK电阻的bulk保留。PDK等效R/C未知时不臆造，使用几何、倍乘与类型特征。原始DUT图、性能和Mask未变。

模型与使用方式见 [构建报告](../../../reports/personal_rgcn_build.md)。

```python
from src.common import read
from src.personal_gnn_data import PersonalDataset
from scripts.personal_gnn import model_for

config = read('config/personal_rgcn.json')
dataset = PersonalDataset(config)
batch = next(dataset.batches('train', batch_size=16))
model = model_for(dataset, config)
prediction = model(batch)  # [16, 16]，未经训练的变换域预测，仅供接口检查
```

完整训练入口是 `scripts/personal_gnn.py train`，不会自动运行。自检权重有construction_smoke_only标记，默认不能用于正式测试。模型输入只包含设计与已知测试条件，禁止加入原始仿真响应作为特征。
