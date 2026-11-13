"""Publish the accepted implementation; preserve training source/checkpoints."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json
import hashlib
import shutil
ROOT=Path(__file__).resolve().parents[3];PACKAGE=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    output=ROOT/'artifacts/experiment1_accuracy_v2_cached_head';summary=read(output/'final_summary.json');assert summary['status']=='accepted_complete'
    recipe=read(output/'recipe.json');version=ROOT/'versions/rgcn_calibrated_head_v1'
    if version.exists():raise FileExistsError('Formal version already exists; no overwrite')
    for name,value in recipe['source_sha256'].items():assert sha(ROOT/name)==value
    shutil.copytree(PACKAGE/'src',version/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    script_dir=version/'scripts';script_dir.mkdir()
    cli='''"""Selected RGCN checkpoint inference. Windows native, fresh Python process."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[3];PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
import torch
import src.common as common
common.ROOT=ROOT
from src.graph.builder import CircuitBatch
from src.gnn.accuracy_inference import load_model,predict_physical

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',default='artifacts/experiment1_accuracy_v2_cached_head/best.pt')
    parser.add_argument('--batch',required=True);parser.add_argument('--output',required=True);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args();target=(ROOT/args.output).resolve()
    if not target.is_relative_to(ROOT):raise ValueError('Output outside EDA')
    if target.exists():raise FileExistsError('Refusing prediction overwrite')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model,saved=load_model(ROOT/args.checkpoint,args.device)
    batch=CircuitBatch(**torch.load(ROOT/args.batch,map_location='cpu',weights_only=True))
    prediction=predict_physical(model,batch)
    prediction={key:value.cpu() if isinstance(value,torch.Tensor) else value for key,value in prediction.items()}
    prediction['checkpoint_recipe_digest']=saved['recipe_digest'];target.parent.mkdir(parents=True,exist_ok=True);torch.save(prediction,target)
    print('Predicted',len(batch.sample_ids),'real samples; physical units; original mask preserved:',target)

if __name__=='__main__':main()
'''
    compile(cli,str(script_dir/'predict.py'),'exec');(script_dir/'predict.py').write_text(cli,encoding='utf-8')
    files={p.relative_to(version).as_posix():sha(p) for p in version.rglob('*.py')}
    for p in (version/'src').rglob('*.py'):assert sha(p)==sha(PACKAGE/p.relative_to(version))
    receipt={'status':'published_inference_compatible','at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'checkpoint':'artifacts/experiment1_accuracy_v2_cached_head/best.pt','checkpoint_sha256':summary['selected_checkpoint_sha256'],
        'training_recipe_digest':read(output/'test_metrics.json')['recipe_digest'],'source_sha256':files,
        'training_source_preserved':'worktrees/rgcn_accuracy_v2','no_cross_recipe_resume':True,
        'GPU_qualified_variant':'calibration_only; contract residual adapters disabled'}
    (version/'publication.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    (version/'README.md').write_text('''# 已采用的校准输出头 RGCN

公开融合数据集实验一已完成，选定模型为19轮RGCN主干预训练＋6轮冻结主干输出头微调。

模型与原单位测试指标在 `artifacts/experiment1_accuracy_v2_cached_head/`。完整结果：[实验一报告](../../reports/experiment1_id_results.md)。原训练代码、检查点与实验结果保留。

本版本源文件与GPU验收后的训练实现逐文件一致，记录在 `publication.json`。训练来源仍保留在 `worktrees/rgcn_accuracy_v2`；这里用于稳定加载模型，不提供跨recipe续训。实际采用的是校准读出，137个额外非线性分支没有启用。

在新的Python进程中进行推理，例如使用现有真实训练小批次做工程检查：

```powershell
.\\.venv\\Scripts\\python.exe -X utf8 -B versions/rgcn_calibrated_head_v1/scripts/predict.py --batch reports/experiment1_performance/fixtures/batch_16_7.pt --output artifacts/experiment1_accuracy_v2_cached_head/inference_smoke.pt
```

输出是原物理单位预测和原缺失Mask。正式测试指标已评价一次，不能用再次测试来调参。后续LOTO或跨数据集实验需要新的数据划分、训练统计、recipe和可监督契约。
''',encoding='utf-8')
    pointer={'state':'complete','at':receipt['at'],'experiment':'public_fusion_ID','model':receipt['checkpoint'],
        'model_sha256':receipt['checkpoint_sha256'],'implementation':'versions/rgcn_calibrated_head_v1',
        'results_report':'reports/experiment1_id_results.md','test_metrics':'artifacts/experiment1_accuracy_v2_cached_head/test_metrics.json',
        'runtime_status':'monitoring/accuracy_acceleration_v1/status.json','parent_training_preserved_paused':True,
        'protocol':'19 graph-pretraining epochs + 6 frozen-backbone calibrated-head epochs'}
    (ROOT/'monitoring/current_training.json').write_text(json.dumps(pointer,indent=2),encoding='utf-8')
    print(json.dumps(pointer,indent=2))

if __name__=='__main__':main()
