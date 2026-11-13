"""Publish completed-stage selection, without changing training artifacts."""
from pathlib import Path
from datetime import datetime, timezone
import json
import re
ROOT = Path(__file__).resolve().parents[2]
selection = json.loads((ROOT/'monitoring/joint_cpu_refine_v2/selection.json').read_text(encoding='utf-8'))
pointer = ROOT/'monitoring/current_training.json'
record = json.loads(pointer.read_text(encoding='utf-8'))
assert record['run_id'] == 'experiment1_joint_cpu_refine_v2'
record.update(updated_at=datetime.now(timezone.utc).isoformat(),
              candidate_checkpoint=record['model'], model=selection['selected_checkpoint'],
              candidate_accepted=selection['accepted'],
              model_selection_reason='Three complete joint epochs did not improve validation macro_hit5; parent remains selected',
              training_process_running=False, supervisor_process_running=False)
pointer.write_text(json.dumps(record, indent=2), encoding='utf-8')
for name, prefix in [('README.md',''),('reports/README.md','../')]:
    path = ROOT/name
    body = path.read_text(encoding='utf-8')
    banner = f'''<!-- joint-training-current:start -->
> **联合训练阶段已结束（2026-11-02 05:50，北京时间）：** 完成 3 轮后正常早停，未超过父模型验证 hit@5%=24.10%，保留原最佳权重；当前没有该阶段的训练或监督进程。[结果与历史记录]({prefix}reports/joint_cpu_training.md) · [当前模型指针]({prefix}monitoring/current_training.json)。
<!-- joint-training-current:end -->'''
    body = re.sub(r'<!-- joint-training-current:start -->.*?<!-- joint-training-current:end -->', lambda _: banner, body, flags=re.S)
    path.write_text(body, encoding='utf-8')
autonomy = ROOT/'monitoring/training_autonomy.json'
record = json.loads(autonomy.read_text(encoding='utf-8'))
record.update(recorded_at=datetime.now(timezone.utc).isoformat(),
              current_action='Joint CPU refinement completed3epochs by validation plateau; candidate not adopted; no active worker/supervisor',
              background_controller_running=False)
autonomy.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
print('Updated completed-stage documentation and selected model pointer. Checkpoints/logs unchanged.')
