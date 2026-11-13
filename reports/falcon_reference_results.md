# FALCON参考口径的独立评价

依据FALCON Stage 2：逐指标R²／MAE／RMSE／MRE，以及逐样本平均相对误差。
下表均为完整、未经截尾的样本平均相对误差；不是5%命中率，也不能用100减该值称为准确率。
已有测试预测的重新统计不等于新的盲测。不同数据来源与训练协议的数值不构成同条件模型排名。

| 实验 | 样本数 | 完整MRE (%) | 样本中位误差 (%) | P95 (%) | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| Public ID: calibrated local readout | 49056 | 3.53464e+07 | 4.27043 | 156.458 | 已统计 |
| LOADBench: supervised-neighborhood expert | 19301 | 9.02829e+07 | 11.2661 | 2016.49 | 已统计 |
| LOTO: SimpleOpAmp4 | 2404 | 295.563 | 26.8991 | 403.66 | 已统计 |
| Personal full supervision: full_joint | 2000 | 26.7132 | 4.86931 | 43.4725 | 已统计 |
| Personal full supervision: full_physical | 2000 | 23.2848 | 5.6071 | 48.7847 | 已统计 |
| Personal full supervision: full_gnn | 2000 | 27.9024 | 11.4046 | 63.7743 | 已统计 |
| Personal e2e: pretrained, k=5, seed=42 | 2000 | 214.198 | 56.0321 | 329.474 | 已统计 |
| Personal e2e: pretrained, k=10, seed=42 | 2000 | 213.587 | 56.3586 | 309.301 | 已统计 |
| Personal e2e: pretrained, k=20, seed=42 | 2000 | 105.364 | 33.931 | 135.783 | 已统计 |
| Personal e2e: pretrained, k=50, seed=42 | 2000 | 116.1 | 26.4999 | 112.385 | 已统计 |
| Personal e2e: pretrained, k=100, seed=42 | 2000 | 101.806 | 24.6091 | 105.676 | 已统计 |
| Personal e2e: pretrained, k=100, seed=43 | 2000 | 110.811 | 25.1253 | 107.493 | 已统计 |
| Personal e2e: pretrained, k=100, seed=44 | 2000 | 102.179 | 25.1156 | 107.973 | 已统计 |
| Personal e2e: scratch, k=5, seed=42 | 2000 | 360.02 | 69.3159 | 600.208 | 已统计 |
| Personal e2e: scratch, k=10, seed=42 | 2000 | 303.39 | 56.8935 | 511.438 | 已统计 |
| Personal e2e: scratch, k=20, seed=42 | 2000 | 103.761 | 39.9358 | 138.461 | 已统计 |
| Personal e2e: scratch, k=50, seed=42 | 2000 | 106.514 | 27.0309 | 100.359 | 已统计 |
| Personal e2e: scratch, k=100, seed=42 | 2000 | 106.616 | 26.3501 | 98.4342 | 已统计 |
| Personal e2e: scratch, k=100, seed=43 | 2000 | 108.794 | 26.249 | 109.584 | 已统计 |
| Personal e2e: scratch, k=100, seed=44 | 2000 | 105.516 | 26.9785 | 104.681 | 已统计 |
| Pending neural experiment: rgcn_memory_safe_id_control_v2 | — | — | — | — | 等待选模冻结与测试预测 |
| Pending neural experiment: rgcn_memory_safe_no_film_v2 | — | — | — | — | 等待选模冻结与测试预测 |
| Pending neural experiment: rgcn_memory_safe_static_edges_v2 | — | — | — | — | 等待选模冻结与测试预测 |
| Pending neural experiment: rgcn_memory_safe_collapsed_relations_v2 | — | — | — | — | 等待选模冻结与测试预测 |
| Pending neural experiment: rgcn_memory_safe_loto_v2 | — | — | — | — | 等待选模冻结与测试预测 |

每个实验的逐任务表、原始样本误差和分布图位于`artifacts/falcon_reference_eval_v1/<job>/`。
dB等标签的相对误差按FALCON原代码的数值约定计算，物理解释仍依赖MAE／RMSE。
当前模型选择和数据划分保持原配方；新增统计不回用于已冻结模型的选择。
