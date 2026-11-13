"""Derive a low-memory CPU fork, preserving the GPU worker's saved cursor."""
from pathlib import Path
import json
import hashlib
ROOT=Path(__file__).resolve().parents[2];PACKAGE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    cfg=json.loads((PACKAGE/'config.json').read_text());source=ROOT/cfg['output']
    cfg.update(output='artifacts/experiment1_cached_head_cpu_continue_v1',device='cpu',minimum_start_RAM_GiB=1.,
        pause_below_RAM_MiB=512,backend_parent_output=str(source.relative_to(ROOT)).replace('\\','/'),
        backend_parent_last_sha256=sha(source/'last.pt'),backend_parent_recipe_sha256=sha(source/'recipe.json'))
    (PACKAGE/'config_cpu.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    code=(PACKAGE/'train.py').read_text()
    code=code.replace("PACKAGE/'config.json'","PACKAGE/'config_cpu.json'")
    code=code.replace("output.name!='experiment1_cached_head_continue_v1'","output.name!='experiment1_cached_head_cpu_continue_v1'")
    code=code.replace("'cached_head_validation_continue_v1'","'cached_head_validation_continue_cpu_v1'")
    code=code.replace("'cuda'","'cpu'")
    code=code.replace("torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.35)","# CPU backend: no CUDA initialization or GPU allocations.")
    code=code.replace("torch.cuda.set_rng_state_all([s.cpu() for s in saved['rng_cuda']])","inherited_cuda_rng=saved['rng_cuda']")
    code=code.replace("torch.cuda.set_rng_state_all([s.cpu() for s in parent['rng_cuda']])","inherited_cuda_rng=parent['rng_cuda']")
    code=code.replace("'rng_cuda':torch.cuda.get_rng_state_all()","'rng_cuda':inherited_cuda_rng")
    code=code.replace("torch.cuda.max_memory_allocated()/2**20","0.")
    code=code.replace('available<384*2**20','available<256*2**20')
    code=code.replace("'memory_scope':'Previously GPU-qualified head only; no GNN/graph tensors loaded to GPU, no large CUDA acceptance test'",
        "'memory_scope':'CPU-only cached head, measured ~304 MiB small-step RSS; no CUDA context; retain graph parameters unchanged'")
    injection="""backend=ROOT/cfg['backend_parent_output']
            assert sha(backend/'recipe.json')==cfg['backend_parent_recipe_sha256']
            assert sha(backend/'last.pt')==cfg['backend_parent_last_sha256']
            inherited=torch.load(backend/'last.pt',map_location='cpu',weights_only=True)
            assert inherited['recipe_digest']==digest(read(backend/'recipe.json'))
            head.load_state_dict(inherited['head'],strict=True);optimizer.load_state_dict(inherited['optimizer']);scheduler.load_state_dict(inherited['scheduler'])
            start_epoch=inherited['next_additional_epoch'];start_batch=inherited['next_batch'];partial=inherited['partial']
            history=inherited['history'];best=inherited['best_score'];best_total=inherited['best_total_head_epoch'];bad=inherited['bad']
            torch.set_rng_state(inherited['rng_cpu'].cpu());inherited_cuda_rng=inherited['rng_cuda']
            assert state_digest(inherited['optimizer']['state'])==state_digest(optimizer.state_dict()['state'])
            assert all(torch.equal(v,head.state_dict()[k]) for k,v in inherited['head'].items())
            atomic_json(output/'backend_migration.json',{'source':cfg['backend_parent_output'],'source_last_sha256':cfg['backend_parent_last_sha256'],
                'weights_and_AdamW_moments_exactly_inherited':True,'next_additional_epoch':start_epoch,'next_batch':start_batch,
                'partial_samples':partial['samples'] if partial else 0,'backend':'CPU FP32','test_data_or_scores_used':False})
            del inherited
            """
    needle="atomic_json(output/'fork_receipt.json',{'at':now()"
    assert needle in code;code=code.replace(needle,injection+needle)
    code=code.replace('if not args.resume:save_cursor(0,0,None)','if not args.resume:save_cursor(start_epoch,start_batch,partial)')
    code=code.replace("'protocol':'frozen_backbone_cached_head_continuation'","'protocol':'frozen_backbone_cached_head_continuation_CPU','device':'cpu'")
    compile(code,str(PACKAGE/'train_cpu.py'),'exec');(PACKAGE/'train_cpu.py').write_text(code,encoding='utf-8')
    watcher=(PACKAGE/'supervise.py').read_text().replace('cached_head_continue_v1','cached_head_cpu_continue_v1')
    watcher=watcher.replace("PACKAGE/'config.json'","PACKAGE/'config_cpu.json'").replace("PACKAGE/'train.py'","PACKAGE/'train_cpu.py'")
    watcher=watcher.replace("ROOT/'.venv-gpu/Scripts/python.exe'","ROOT/'.venv/Scripts/python.exe'")
    watcher=watcher.replace("'memory_scope':'head-only previously GPU-qualified code; no large CUDA test'","'memory_scope':'CPU-only cached head; no CUDA context'")
    watcher=watcher.replace("endswith('cached_head_cpu_continue_v1/train.py')","endswith('cached_head_continue_v1/train_cpu.py')")
    compile(watcher,str(PACKAGE/'supervise_cpu.py'),'exec');(PACKAGE/'supervise_cpu.py').write_text(watcher,encoding='utf-8')
    print('Prepared explicit CPU backend fork with saved optimizer/cursor; no source run altered.')

if __name__=='__main__':main()
