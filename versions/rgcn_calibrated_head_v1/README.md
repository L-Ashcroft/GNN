# 已采用的校准输出头 RGCN

公开融合数据集实验一已完成，选定模型为19轮RGCN主干预训练＋6轮冻结主干输出头微调。

模型与原单位测试指标在 `artifacts/experiment1_accuracy_v2_cached_head/`。完整结果：[实验一报告](../../reports/experiment1_id_results.md)。原训练代码、检查点与实验结果保留。

本版本源文件与GPU验收后的训练实现逐文件一致，记录在 `publication.json`。训练来源仍保留在 `worktrees/rgcn_accuracy_v2`；这里用于稳定加载模型，不提供跨recipe续训。实际采用的是校准读出，137个额外非线性分支没有启用。

在新的Python进程中进行推理，例如使用现有真实训练小批次做工程检查：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B versions/rgcn_calibrated_head_v1/scripts/predict.py --batch reports/experiment1_performance/fixtures/batch_16_7.pt --output artifacts/experiment1_accuracy_v2_cached_head/inference_smoke.pt
```

输出是原物理单位预测和原缺失Mask。正式测试指标已评价一次，不能用再次测试来调参。后续LOTO或跨数据集实验需要新的数据划分、训练统计、recipe和可监督契约。
