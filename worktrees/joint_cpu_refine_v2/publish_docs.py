"""Update live pointers in docs without modifying pinned training sources."""
from pathlib import Path
from datetime import datetime,timezone
import json
import re
ROOT=Path(__file__).resolve().parents[2]
for name,prefix in [('README.md',''),('reports/README.md','../')]:
    path=ROOT/name;body=path.read_text(encoding='utf-8')
    body=re.sub(r'<!-- autonomous-training-current:start -->.*?<!-- autonomous-training-current:end -->\s*','',body,flags=re.S)
    banner=f'''<!-- joint-training-current:start -->
> **当前训练（2026-11-02）：** 已启动[完整 GNN 的低学习率联合训练]({prefix}reports/joint_cpu_training.md)，CPU 4 线程、有效 Batch256；主干与输出头共同更新，不再因 GPU 内存门槛空等。查看[实时训练]({prefix}artifacts/experiment1_joint_cpu_refine_v2/status.json)和[监督状态]({prefix}monitoring/joint_cpu_refine_v2/status.json)。新模型精度尚待完整验证；此前权重与结果保留，下方阶段进度为历史记录。
<!-- joint-training-current:end -->

'''
    body=re.sub(r'<!-- joint-training-current:start -->.*?<!-- joint-training-current:end -->\s*','',body,flags=re.S)
    path.write_text(banner+body,encoding='utf-8')
legacy=ROOT/'monitoring/autonomous_training_v1/status.json'
record=json.loads(legacy.read_text(encoding='utf-8'));record.update(state='superseded_by_joint_cpu_refine_v2',updated_at=datetime.now(timezone.utc).isoformat(),successor='monitoring/joint_cpu_refine_v2/status.json',runtime_status='artifacts/experiment1_joint_cpu_refine_v2/status.json',reason='CPU bounded-memory full joint training is now active; old head phase remains complete')
legacy.write_text(json.dumps(record,indent=2),encoding='utf-8')
path=ROOT/'monitoring/training_autonomy.json';record=json.loads(path.read_text(encoding='utf-8'))
record.update(recorded_at=datetime.now(timezone.utc).isoformat(),latest_user_authorization='那你自己想办法继续训练模型，我把任务全权委托给你了',
    current_action='Bounded CPU full RGCN joint refinement with lower learning rates/warmup; prior short pilots rejected and retained',
    background_controller_status='monitoring/joint_cpu_refine_v2/status.json',controller_policy='monitoring/joint_cpu_refine_v2/policy.json',documentation='reports/joint_cpu_training.md')
path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print('Published current joint-training pointers; runtime source/config unchanged.')
