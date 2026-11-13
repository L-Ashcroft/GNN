# 低内存缓存输出头续训

实际运行：`train_cpu.py`，由 `supervise_cpu.py` 监督。源`train.py`是先前GPU尝试，已因资源不足保存暂停；不要同时启动两者。

续训从原6轮状态接上，保留权重、AdamW矩、RNG和第7轮第1批进度。学习率重启5e-5，最多再训练20轮、验证早停5轮，后续不评价测试集。图主干固定，不重新提取特征。

状态与完整设置见 [续训报告](../../reports/cached_head_continuation.md)。使用 `monitoring/current_training.json` 找到当前运行；旧测试分数仅对应先前冻结模型。

模型和Loss使用已验收的 `versions/rgcn_calibrated_head_v1/src`，原版文件不改。每个阶段有独立recipe，只有本阶段的 --resume 才可恢复本阶段检查点。运行中不要修改对应源码或配置。
