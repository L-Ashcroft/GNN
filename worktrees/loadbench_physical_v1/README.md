# LOADBench物理标签审计与独立质量视图

这里只新增审计代码与引用，不修改原始数据、缺失Mask、旧划分或运行模型。

- `replay.py`：原生Windows、4个真实CSV参数点、官方reduced SKY130、OP/偏置校正及1Hz–1GHz AC扫描。安全断点暂停原训练后执行，finally按同一recipe恢复监督器。当前训练完成或不存在时不要盲目重跑暂停流程。
- `audit.py`：根据已发布扫描窗口、非正DC增益、极端相位生成独立复核标记。阈值只是复核条件，不证明整行物理无效，`feasible=False`本身不是排除条件。
- `baseline.py`：单线程参数Ridge诊断；原训练池和无触发训练池在完全相同的无触发验证样本上比较。先用`--probe`运行2拓扑，再运行92拓扑。需要至少950MiB空闲内存，不用GPU，不请求测试标签行。
- `checks.py` / `verify_release.py`：边界条件、50万ID覆盖、源文件哈希、引用数、原Mask与载入接口、实测AC及旧训练恢复检查。

输出：`data/audit/loadbench_physical_v1/`、`artifacts/loadbench_physical_baseline_v1/`。质量视图是原样本的引用，可这样加载：

```python
import sys
sys.path.insert(0, 'worktrees/loadbench_physical_v1')
from load import QualityView
view = QualityView()
sample = next(view.samples(cohort='no_trigger', role='train', source='LOADBench'))
print(sample['performance'], sample['mask'], sample['quality_review'])
view.close()
```

`no_trigger`不等于物理认证合格。FALCON/AICircuit在此策略中未被评估，scope明确标为`not_assessed_by_this_policy`。测试访问默认拒绝。LOTO应按该折引用覆盖角色，禁止误用上述ID角色训练。

首次SI尺寸直接传给微米模型接口时发生模型bin匹配失败，证据保存在`replays/`；修正的回放把CSV微米尺寸传给带scale=1e-6的SKY130包装器，保存在`replays_native_units/`。这只是回放适配，原图中的米单位不变。3个正常值对照与存档高度一致，极端点的DC增益也一致；但历史参数已舍入，极端外推结果不能声称逐位复现。

独立NumPy相位展开与原函数可能相差360度分支；不得把审计输出当作已经认证的新相位裕度标签，也不得仅靠截断或取模生成替代标签。新GNN训练须另设数据协议与recipe，不继承已见目标拓扑的ID权重。
