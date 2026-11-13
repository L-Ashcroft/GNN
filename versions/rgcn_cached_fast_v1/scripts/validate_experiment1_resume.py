"""CPU-only checkpoint validation, invoked only after the old trainer exits."""
import argparse
import gc
import json
import math
from pathlib import Path
import sys
import threading
import os
import zipfile
import psutil
from experiment1_health import ROOT,read,atomic,digest,now,nonfinite_paths

def recipe_digest(recipe):
    import hashlib
    return hashlib.sha256(json.dumps(recipe,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def validate_cursor(checkpoint,recipe,total_train):
    required={'recipe_digest','model','optimizer','scheduler','coverage_slots','coverage_domains','next_epoch','next_batch',
              'partial','history','best','best_epoch','bad','rng_cpu','rng_cuda'}
    if not required<=set(checkpoint):raise ValueError('Checkpoint required fields missing')
    if checkpoint['recipe_digest']!=recipe_digest(recipe):raise ValueError('Checkpoint recipe fingerprint mismatch')
    epoch=checkpoint['next_epoch'];batch=checkpoint['next_batch'];config=recipe['config']
    if not isinstance(epoch,int) or not 0<=epoch<=config['epochs']:raise ValueError('Invalid checkpoint epoch')
    if not isinstance(batch,int) or not 0<=batch<=math.ceil(total_train/config['batch_size']):raise ValueError('Invalid checkpoint batch')
    if len(checkpoint['history'])!=epoch:raise ValueError('History/cursor mismatch')
    partial=checkpoint['partial']
    if batch and partial is None:raise ValueError('Missing in-epoch accumulator')
    if partial is not None:
        if partial['samples']!=min(batch*config['batch_size'],total_train) or partial['grad_count']!=batch:
            raise ValueError('Partial sample/gradient counts mismatch')
        if nonfinite_paths(partial):raise ValueError('Nonfinite checkpoint accumulators')
    if nonfinite_paths(checkpoint['history']):raise ValueError('Nonfinite validation history')
    if checkpoint['best_epoch'] is not None and not math.isfinite(checkpoint['best']):raise ValueError('Nonfinite best score')
    if checkpoint['bad']<0:raise ValueError('Negative early stopping count')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();path=Path(args.checkpoint).resolve();output=Path(args.output).resolve()
    if not path.is_relative_to(ROOT) or not output.is_relative_to(ROOT):raise ValueError('Path outside EDA')
    stop=threading.Event()
    def guard():
        while not stop.wait(.1):
            if psutil.virtual_memory().available<768*2**20:
                atomic(output,{'status':'deferred','reason':'CPU validator RAM guard','at_UTC':now()});os._exit(86)
    threading.Thread(target=guard,daemon=True).start()
    try:
        before=digest(path)
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:raise ValueError('Checkpoint ZIP CRC mismatch')
        import torch
        torch.set_num_threads(1)
        from src.model.network import EdgeCentricRGCN
        from src.train.stage import build_optimizer
        recipe=read(ROOT/'artifacts/experiment1_fusion_id_fast_v1/recipe.json');config=recipe['config'];cache=ROOT/config['cache']
        if config!=read(ROOT/'config/experiment1_fusion_id_fast_v1.json'):raise ValueError('Configuration differs from recipe')
        if torch.__version__.split('+')[0]!=recipe['torch'].split('+')[0]:raise ValueError('CPU validator torch version mismatch')
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        manifest=read(cache/'manifest.json');schema=read(cache/'schema.json');tasks=read(cache/'tasks.json')
        validate_cursor(checkpoint,recipe,manifest['role_counts']['train'])
        def finite_tensors(value):
            if isinstance(value,torch.Tensor):
                if value.is_floating_point() and not torch.isfinite(value).all():raise ValueError('Nonfinite model/optimizer tensor')
            elif isinstance(value,dict):
                for item in value.values():finite_tensors(item)
            elif isinstance(value,(list,tuple)):
                for item in value:finite_tensors(item)
            elif isinstance(value,float) and not math.isfinite(value):raise ValueError('Nonfinite optimizer/scheduler scalar')
        finite_tensors(checkpoint['model']);finite_tensors(checkpoint['optimizer']);finite_tensors(checkpoint['scheduler'])
        model=EdgeCentricRGCN(schema,tasks,**{k:config[k] for k in
            ['hidden_dim','num_layers','dropout','update_edges','use_domain_emb','use_film','collapse_relations']})
        model.load_state_dict(checkpoint['model'],strict=True)
        optimizer=build_optimizer(model,'pretrain',config['learning_rate'],weight_decay=config['weight_decay'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['epochs']);scheduler.load_state_dict(checkpoint['scheduler'])
        torch.Generator().set_state(checkpoint['rng_cpu'].cpu())
        if not checkpoint['rng_cuda'] or any(t.dtype!=torch.uint8 or not t.numel() for t in checkpoint['rng_cuda']):raise ValueError('Invalid CUDA RNG records')
        for group,slots in checkpoint['coverage_slots'].items():
            if group not in tasks or any(str(slot) not in tasks[group]['slots'] for slot in slots):raise ValueError('Invalid trained target coverage')
        if not set(checkpoint['coverage_domains'])<={d['id'] for d in schema['domains']}:raise ValueError('Invalid domain coverage')
        if digest(path)!=before:raise ValueError('Checkpoint changed during validation')
        atomic(output,{'status':'valid','at_UTC':now(),'checkpoint_sha256':before,'recipe_digest':checkpoint['recipe_digest'],
            'next_epoch_zero_based':checkpoint['next_epoch'],'next_batch':checkpoint['next_batch'],
            'required_fields_model_optimizer_scheduler_RNG_coverage_checked':True,'CUDA_context_created':False})
        stop.set();print('Checkpoint valid; CPU only.')
    except Exception as error:
        atomic(output,{'status':'invalid','at_UTC':now(),'error_type':type(error).__name__,'error':str(error),'CUDA_context_created':False})
        stop.set();raise

if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]));main()
