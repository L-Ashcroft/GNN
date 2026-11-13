"""Low-memory, below-normal CPU priority; never imports torch."""
from pathlib import Path
import sys
import threading
import os
import argparse
import psutil
ROOT=Path(__file__).resolve().parents[3];PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from src.train.target_calibration import fit,read,sha,validate

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',default='data/model_ready/fusion_v2_training_v1')
    parser.add_argument('--split',default='data/fusion_v2/splits/id_stratified_v1.parquet')
    parser.add_argument('--output',default='data/model_ready/fusion_v2_accuracy_v2/target_calibration.json')
    parser.add_argument('--verify-existing',action='store_true');args=parser.parse_args()
    if psutil.virtual_memory().available<1024*2**20:raise SystemExit('Deferred: need 1 GiB for streaming CPU fit')
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    stop=threading.Event()
    def guard():
        while not stop.wait(.1):
            if psutil.virtual_memory().available<512*2**20:os._exit(86)
    threading.Thread(target=guard,daemon=True).start()
    target=(ROOT/args.output).resolve()
    if not target.is_relative_to(ROOT):raise ValueError('Output outside EDA')
    if args.verify_existing:
        artifact=read(target);cache=ROOT/args.cache;manifest=read(cache/'manifest.json')
        validate(artifact,read(cache/'tasks.json'),manifest)
        if sha(cache/'manifest.json')!=artifact['cache_manifest_sha256'] or sha(ROOT/args.split)!=artifact['split_sha256']:
            raise ValueError('Calibration input fingerprint changed')
        for name,value in artifact['input_sha256'].items():
            if sha(ROOT/name)!=value:raise ValueError('Calibration input changed: '+name)
        stop.set();print('Existing train-only artifact and all source fingerprints verified:',target);return
    artifact=fit(ROOT,args.cache,args.split,target)
    stop.set();print('Fitted train-only calibration:',len(artifact['groups']),'contracts;',artifact['role_counts'],'output',target)

if __name__=='__main__':main()
