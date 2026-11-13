"""Build a guarded, resumable entry and 2x2 head/calibration ablations.

Preparation only. Never starts or modifies the running experiment.
"""
from pathlib import Path
import json

PACKAGE=Path(__file__).resolve().parents[1];ROOT=PACKAGE.parents[1]
def main():
    entry=(ROOT/'versions/rgcn_cached_fast_v1/scripts/train_experiment1_fast.py').read_text(encoding='utf-8')
    preflight='''# Resource/process guard runs before importing torch or writing artifacts.
if __name__=='__main__' and '--help' not in sys.argv and '-h' not in sys.argv:
    probe=argparse.ArgumentParser(add_help=False);probe.add_argument('--config',required=True)
    preliminary,_=probe.parse_known_args()
    trial_root=Path(__file__).resolve().parents[3]
    trial_config=__import__('json').loads((trial_root/preliminary.config).read_text(encoding='utf-8'))
    if trial_config['device']=='cuda':
        for process in psutil.process_iter(['name','cmdline']):
            if process.pid in {os.getpid(),os.getppid()}:continue
            command=process.info['cmdline'] or []
            if (process.info['name'] or '').lower() in {'python.exe','pythonw.exe'} and any(
                Path(a.replace('\\\\','/')).name in {'train_experiment1.py','train_experiment1_fast.py','train_accuracy_v2.py'} for a in command):
                raise SystemExit('CUDA start deferred: another training process is running; existing training untouched')
        if psutil.virtual_memory().available<4*2**30:
            raise SystemExit('CUDA start deferred: at least 4 GiB host RAM required')
'''
    entry=entry.replace('import psutil\nimport torch','import psutil\n'+preflight+'\nimport torch')
    entry=entry.replace('from src.model.network import EdgeCentricRGCN','from src.model.accuracy_network import AccuracyRGCN as EdgeCentricRGCN\nfrom src.train.target_calibration import validate as validate_calibration')
    entry=entry.replace('from src.train.losses import supervision_loss','from src.train.accuracy_loss import supervision_loss')
    entry=entry.replace('from src.train.loop import TrainingCoverage,LossAccumulator,evaluate','from src.train.accuracy_loop import TrainingCoverage,LossAccumulator,evaluate')
    entry=entry.replace("'collapse_relations']","'collapse_relations','contract_residual','calibrated_targets','adapter_hidden','adapter_min_train_samples']")
    entry=entry.replace("model=EdgeCentricRGCN(data.schema,data.tasks,**{k:config[k] for k in KEYS}).to(device)",
        "calibration=read(ROOT/config['target_calibration']);validate_calibration(calibration,data.tasks,data.manifest)\n        if sha(data.path/'manifest.json')!=calibration['cache_manifest_sha256']:raise ValueError('Calibration cache fingerprint mismatch')\n        model=EdgeCentricRGCN(data.schema,data.tasks,calibration=calibration,**{k:config[k] for k in KEYS}).to(device)")
    entry=entry.replace("'execution_variant':'edge_rgcn_cached_fast_v1'","'execution_variant':'edge_rgcn_accuracy_v2'")
    entry=entry.replace("execution_variant='edge_rgcn_cached_fast_v1'","execution_variant='edge_rgcn_accuracy_v2'")
    entry=entry.replace("'config_sha256':sha(CONFIG),","'config_sha256':sha(CONFIG),'target_calibration_sha256':sha(ROOT/config['target_calibration']),")
    entry=entry.replace('accumulated=LossAccumulator()','accumulated=LossAccumulator(model.head)')
    entry=entry.replace("if not output.is_relative_to(ROOT):raise ValueError('Output escapes workspace')",
        "if output.parent!=ROOT/'artifacts' or not output.name.startswith('experiment1_accuracy_v2_'):raise ValueError('Accuracy v2 requires its own experiment1_accuracy_v2_ output namespace')\n    if args.resume and not (output/'last.pt').exists():raise ValueError('Resume requires an existing accuracy-v2 checkpoint')")
    entry=entry.replace('supervision_loss(prediction,batch)','supervision_loss(prediction,batch,model.head)')
    # Inference must have the exact affine statistics that define the model.
    entry=entry.replace("'model':model.state_dict(),'schema':data.schema", "'model':model.state_dict(),'target_calibration':calibration,'schema':data.schema")
    entry=entry.replace("'status':'trained','epoch':best_epoch", "'status':'trained','model_class':'AccuracyRGCN','epoch':best_epoch")
    path=PACKAGE/'scripts/train_accuracy_v2.py'
    if path.exists():raise FileExistsError('Do not silently replace candidate entry')
    compile(entry,str(path),'exec');path.write_text(entry,encoding='utf-8')
    base=json.loads((ROOT/'config/experiment1_fusion.json').read_text())
    for name,residual,calibrated in [('baseline',False,False),('head_only',True,False),('calibration_only',False,True),('combined',True,True)]:
        config=dict(base,output=f'artifacts/experiment1_accuracy_v2_{name}',contract_residual=residual,
            calibrated_targets=calibrated,adapter_hidden=32,adapter_min_train_samples=256,
            target_calibration='data/model_ready/fusion_v2_accuracy_v2/target_calibration.json')
        config['accuracy_variant']=name
        config['precision']='float32; no mixed precision; '+('train-only contract/metric symlog affine calibration' if calibrated else 'legacy fixed engineering target scales')
        (PACKAGE/'config'/f'{name}.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('Prepared fresh-run baseline/head-only/calibration-only/combined configs. No training started.')

if __name__=='__main__':main()
