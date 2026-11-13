# 实验一：公开50万条的分层ID训练

这是正式GPU训练目录，不是AnalogGym仿真进度目录。

实时看 `status.json`；每轮完成后看 `history.json`；错误看 `training_stderr.log`。`last.pt` 约每250批及每轮保存，`best.pt` 按验证集选优，训练结束后才生成 `test_metrics.json`。前250批之前可能还没有检查点，不能把检查点尚未出现理解为程序没有训练。

当前配方为四层128维边中心RGCN，batch256，RTX4050，float32、种子42、确定性算法；来源为FALCON20万、AICircuit10万、LOADBench20万。采用 `data/fusion_v2/splits/id_stratified_v1.parquet`：400083/50861/49056。设计组不跨集合，个人2万条不参与。

完整说明和时长预算见 [训练报告](../../reports/experiment1_fusion_training.md)。

恢复命令：

```powershell
.\.venv-gpu\Scripts\python.exe -B -u scripts/train_experiment1.py --resume
```

不要同时启动多个训练器，文件锁会拒绝第二个写入者。`STOP`文件可请求保存并暂停，恢复前应先明确移除此文件。已经complete时不会再次运行固定测试。
