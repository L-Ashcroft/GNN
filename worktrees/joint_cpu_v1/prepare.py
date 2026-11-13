"""Create an isolated CPU joint runner from the preserved GPU implementation."""
from pathlib import Path
import json
import hashlib
import shutil
ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
source=ROOT/'worktrees/autonomous_training_v1/joint_train.py'
text=source.read_text(encoding='utf-8')
changes=[
    ('Validation-only joint RGCN refinement with streaming graphs and microbatches.', 'CPU joint RGCN refinement with streaming graphs; original CUDA admission gate stays unchanged.'),
    ('if psutil.virtual_memory().available<4*2**30:raise SystemExit(75)', "if psutil.virtual_memory().available<cfg['minimum_start_RAM_GiB']*2**30:raise SystemExit(75)"),
    ("'cached_head_continue_v1/train_cpu.py'", "'cached_head_continue_v1/train_cpu.py','joint_cpu_v1/train.py'"),
    ("'device':'cuda'", "'device':'cpu'"),
    ("torch.set_num_threads(2)", "torch.set_num_threads(cfg['cpu_threads']);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)"),
    ('torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.8)', '# CPU-only run: no CUDA context or CUDA memory allocation.'),
    (".to('cuda')", ".to('cpu')"),
    ("map_location='cuda'", "map_location='cpu'"),
    ("'streamed_joint_refinement_v1'", "'streamed_joint_CPU_v1'"),
    ("torch.cuda.set_rng_state_all([x.cpu() for x in saved['rng_cuda']])", "assert saved['rng_cuda']==[]"),
    ("'rng_cuda':torch.cuda.get_rng_state_all()", "'rng_cuda':[]"),
    ("telemetry=[]", "telemetry={'peak_RSS':0,'minimum_available':psutil.virtual_memory().available}"),
    ("telemetry.append((psutil.Process().memory_info().rss,available))", "telemetry.update(peak_RSS=max(telemetry['peak_RSS'],psutil.Process().memory_info().rss),minimum_available=min(telemetry['minimum_available'],available))"),
    ("stop.set()\n        if reader", "stop.set()\n        atomic(output/'resource_summary.json',{'peak_RSS_MiB':telemetry['peak_RSS']/2**20,'minimum_available_RAM_MiB':telemetry['minimum_available']/2**20,'CUDA_used':False})\n        if reader"),
]
for before,after in changes:
    assert before in text, before
    text=text.replace(before,after)
assert 'torch.cuda.' not in text
assert not (HERE/'train.py').exists(),'Never regenerate a running recipe'
(HERE/'train.py').write_text(text,encoding='utf-8')
shutil.copy2(ROOT/'worktrees/autonomous_training_v1/stream_graphs.py',HERE/'stream_graphs.py')
parent=ROOT/'artifacts/experiment1_autonomous_head_v1/best.pt'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg={'parent_checkpoint':parent.relative_to(ROOT).as_posix(),'parent_checkpoint_sha256':sha(parent),
    'metadata_cache':'data/model_ready/fusion_v2_frozen_e19_bf0d71ad','output':'artifacts/experiment1_joint_cpu_v1',
    'effective_batch_size':256,'training_microbatch':128,'validation_microbatch':128,'lru_topologies':2,
    'cpu_threads':2,'minimum_start_RAM_GiB':1.5,
    'head_lr':2.5e-5,'backbone_lr':1e-5,'minimum_lr':1e-6,'max_epochs':48,'patience':6,'checkpoint_every_updates':25,
    'window_epochs':4,'minimum_window_gain':.002,'window_stop_backbone_lr':2.5e-6,
    'pilot_train_samples':5120,'pilot_validation_samples':5120,'pilot_lr_scales':[1.,.5],
    'pilot_min_hit_gain':1e-4,'pilot_min_loss_gain':.005,'pilot_max_median_drop':.01,'test_policy':'No test evaluation'}
(HERE/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
(HERE/'origin.json').write_text(json.dumps({'parent_source':str(source.relative_to(ROOT)),'parent_source_SHA256':sha(source),
    'changes':changes,'CPU_probe_evidence':['probe_result.json','probe_t1_m128.json','probe_t2_m128.json'],
    'policy':'Bounded CPU fallback only; original CUDA gate remains4GiB; no shared production file edits'},indent=2),encoding='utf-8')
print('Prepared isolated CPU joint runner/config. No original source or checkpoint changed.')
