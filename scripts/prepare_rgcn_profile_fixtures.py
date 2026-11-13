"""CPU-only extraction of four real training batches for isolated diagnostics."""
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read
from src.analoggym_campaign import sha
from src.windows_campaign_io import atomic_json
from src.train.fusion_cache import CachedFusion


def main():
    output=ROOT/'reports/experiment1_performance/fixtures';output.mkdir(parents=True,exist_ok=True)
    config=read(ROOT/'config/experiment1_fusion.json');cache=ROOT/config['cache'];manifest=read(cache/'manifest.json')
    probe=CachedFusion.__new__(CachedFusion);probe.manifest=manifest;probe.schema=read(cache/'schema.json');probe.tasks=read(cache/'tasks.json')
    probe.arrays={};probe.static={};indices={};names=[];ends=[];total=0
    for name,spec in manifest['topologies'].items():
        role=np.load(ROOT/spec['folder']/'role.npy',mmap_mode='r');indices[name]=np.flatnonzero(role==0).astype(np.int32)
        names.append(name);total+=len(indices[name]);ends.append(total)
    order=np.arange(total,dtype=np.int32);np.random.default_rng(config['seed']).shuffle(order);starts=[0]+ends[:-1]
    selections=[]
    for size,offset in [(16,7),(256,11),(256,137),(256,503)]:
        refs=[]
        for rank in order[256*offset:256*offset+size]:
            j=int(np.searchsorted(ends,rank,side='right'));name=names[j]
            refs.append((name,int(indices[name][int(rank)-starts[j]])))
        selections.append({'size':size,'offset':offset,'references':refs})
    for name in {n for selection in selections for n,_ in selection['references']}:
        folder=ROOT/manifest['topologies'][name]['folder']
        probe.arrays[name]={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in
            ['values','known','zero','target','mask','group','domain','sample_ids']}
        probe.static[name]=read(folder/'structure.json')
    for selection in selections:
        batch=probe.batch(selection['references'],'train');name=f"batch_{selection['size']}_{selection['offset']}.pt"
        torch.save(vars(batch),output/name);selection['file']=name;selection['sha256']=sha(output/name)
    atomic_json(output/'manifest.json',{'selections':selections,'cache_manifest_sha256':sha(cache/'manifest.json'),
        'config_sha256':sha(ROOT/'config/experiment1_fusion.json'),'source':'original CachedFusion.batch, CPU only',
        'test_samples_used':0,'production_files_modified':False})
    print('Four small real-batch CPU fixtures saved; no GPU or live checkpoint accessed.')


if __name__=='__main__':main()
