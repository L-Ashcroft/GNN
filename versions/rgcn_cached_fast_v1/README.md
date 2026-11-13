# 已验收的边中心 RGCN 缓存执行版

此版本已通过 RTX 4050、真实 Batch=256 的 FP32 正确性检查与完整训练 Step 测速。模型源码与验收副本逐文件相同。实测原版 1.676 秒/批、本版 1.299 秒/批，约 1.29×；这不是所有批次或整轮耗时的保证。

完整报告：[GPU 验收](../../reports/rgcn_gpu_acceptance_report.md)。源代码在本目录 `src/`；原始 `D:/EDA/src` 保留。

当前实验已运行，查看：

- [训练进度](../../artifacts/experiment1_fusion_id_fast_v1/status.json)
- [监控与恢复状态](../../monitoring/experiment1_fusion_id_fast_v1/status.json)
- [迁移状态核对](../../artifacts/experiment1_fusion_id_fast_v1/migration_receipt.json)

第 16 轮前 1024 批属于原版，余下批次由此版本继续。权重、AdamW、scheduler、CPU/CUDA RNG、早停计数与累计统计原样迁移，未重新初始化。旧实验目录保留为暂停记录，不能同时续训旧版。

只有确认当前进程结束、指纹与资源满足时才手动续训：

```powershell
Set-Location D:\EDA
.\.venv-gpu\Scripts\python.exe -B -u versions/rgcn_cached_fast_v1/scripts/train_experiment1_fast.py --config config/experiment1_fusion_id_fast_v1.json --resume
```

后台监控已启用用户授权的有界自动恢复，通常无需手动操作。可关闭屏幕；系统睡眠会暂停运算。监控在训练活跃时请求系统保持唤醒，不能阻止手动睡眠、合盖策略或断电。

后续新 ID、LOTO 或消融实验应使用新的配置及输出目录。入口严格核对 recipe；不要修改已运行版本的模型源码、配置或迁移来源文件，也不能把旧版检查点直接放入其他 recipe 续训。

保持 4 层、128 维、48 种关系、独立监督头、原 Loss 和确定性设置。选用缓存关系/池化成员、Loss 索引与 CPU 统计优化，保留原池化与输出头浮点计算顺序。激进分组 GEMM 版本未通过 GPU AdamW 容差，未用于此版本。

大型 profiler 内核/同步计数因内存门槛仍待补测；训练运行时不会叠加该测试。
