"""Refresh documentation pointers without touching pinned training files."""
from pathlib import Path
from datetime import datetime
import json
import re

ROOT = Path(__file__).resolve().parents[2]
marker_start = '<!-- autonomous-training-current:start -->'
marker_end = '<!-- autonomous-training-current:end -->'
for name, prefix in [('README.md', ''), ('reports/README.md', '../')]:
    path = ROOT / name
    body = path.read_text(encoding='utf-8')
    body = re.sub(re.escape(marker_start) + r'.*?' + re.escape(marker_end) + r'\s*', '', body, flags=re.S)
    report = prefix + 'reports/autonomous_training.md'
    banner = f'''{marker_start}
> **当前训练入口（2026-11-01）：** 已启用[自动监督与联合微调队列]({report})。输出头累计 42 轮正常早停，最佳第 36 轮验证 hit@5%=24.10%；后台控制器等待内存门槛后自动执行联合阶段。查看[实时控制器状态]({prefix}monitoring/autonomous_training_v1/status.json)和[当前模型指针]({prefix}monitoring/current_training.json)。新候选尚未评估测试集，旧测试成绩只属于原权重。下方早期阶段进度为历史记录。
{marker_end}

'''
    path.write_text(banner + body, encoding='utf-8')

path = ROOT / 'monitoring/training_autonomy.json'
record = json.loads(path.read_text(encoding='utf-8'))
record.update({
    'recorded_at': datetime.now().astimezone().isoformat(),
    'latest_user_authorization': '你后面主动监督训练，可以自己调整，不必获取我的批准，你在这个项目上（EDA文件夹内）有最高权限。尽快使训练达到最佳效果',
    'current_action': 'Autonomous head refinement completed by validation early stopping; persistent controller queues resource-gated joint RGCN refinement.',
    'background_auto_optimization_service_installed': False,
    'background_controller_running': True,
    'background_controller_status': 'monitoring/autonomous_training_v1/status.json',
    'controller_policy': 'monitoring/autonomous_training_v1/policy.json',
    'restart_policy': 'No OS-startup service. Check existing joint artifacts/process before manual restart.',
    'documentation': 'reports/autonomous_training.md',
})
path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('Updated README pointers and training autonomy record; pinned runtime files unchanged.')
