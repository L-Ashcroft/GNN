"""10-topology, 20k independent target-domain campaign using pinned simulation code."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import threading
import time
import zipfile
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read,dump,digest
from src.analoggym_campaign import inside,atomic_json,sha,catalog
from src.analoggym_campaign_runner import initialize,run,connect,export,status

CONFIG=ROOT/'config/analoggym_personal_20k.json'


def settings():
    config=read(CONFIG);output=inside(ROOT/config['output'])
    assert len(set(config['topologies']))==10 and config['target_accepted']==20000 and config['workers']==8
    return config,output


def prepare():
    config,output=settings()
    if (output/'manifest.json').exists():return read(output/'manifest.json')
    output.mkdir(parents=True,exist_ok=True)
    # Standard anchors passed a tight-solver precision check. Additional actual
    # benchmark designs are audited by initialize() before becoming centres.
    anchors=[];seen=set()
    for r in read(ROOT/'data/generated/analoggym_standard_v4/summary.json')['results']:
        if r['topology'] in config['topologies'] and r['accepted']:
            anchors.append(r);seen.add(digest([r['topology'],r['parameters_native']]))
    for path in sorted((ROOT/'data/generated/analoggym_benchmark_20261027').rglob('*.zip')):
        with zipfile.ZipFile(path) as z:r=json.loads(z.read('result.json'))
        key=digest([r['topology'],r['parameters_native']])
        if r['topology'] in config['topologies'] and r['accepted'] and key not in seen:
            anchors.append(r);seen.add(key)
    assert set(r['topology'] for r in anchors)==set(config['topologies'])
    qualification=output/'qualification';qualification.mkdir(exist_ok=True)
    atomic_json(qualification/'summary.json',{'results':anchors,
        'purpose':'Measured anchors only; original performance not counted as personal samples'})
    manifest=initialize(output,qualification,target=10*config['pilot_per_topology'],workers=8,
                        max_attempts_factor=config['max_attempts_factor'])
    recipe=read(output/'recipe.json')
    assert len(recipe['entries'])==10
    assert len({e['graph']['topology_hash'] for e in recipe['entries']})==10
    manifest.update(dataset_id=config['dataset_id'],stage='pilot',production_target_accepted=20000,
        production_quotas={n:2000 for n in config['topologies']},
        maximum_campaign_disk_GiB=config['maximum_campaign_disk_GiB'],
        personal_controller_sha256=sha(Path(__file__)),personal_config_sha256=sha(CONFIG),
        resource_policy='8 independent ngspice processes maximum; drain/checkpoint if available RAM below 512 MiB',
        label_reuse=False)
    atomic_json(output/'manifest.json',manifest)
    atomic_json(output/'parameter_anchor_audit.json',{
        'anchor_counts':dict(Counter(r['topology'] for r in anchors)),
        'anchor_label_reuse':False,'parameter_groups':[
            {'topology':e['name'],'family':e['family'],'sampling_centers':e['sampling_centers'],
             'center_provenance':e['center_provenance'],'source_baseline':e['baseline'],
             'definitions':e['definitions']} for e in recipe['entries']]})
    return manifest


def promote():
    config,output=settings();manifest=read(output/'manifest.json')
    if manifest['stage']=='production':return manifest
    quality=read(output/'quality_checks.json');progress=read(output/'status.json')
    assert quality['passed'] and progress['complete']
    assert quality['samples_checked']==10*config['pilot_per_topology']
    assert set(quality['topologies'])==set(config['topologies'])
    atomic_json(output/'pilot_quality_checks.json',quality)
    atomic_json(output/'pilot_status.json',progress)
    manifest.update(stage='production',target_accepted=20000,quotas=manifest['production_quotas'],status='prepared')
    atomic_json(output/'manifest.json',manifest)
    db=connect(output)
    try:status(output,db,manifest,'prepared_production')
    finally:db.close()
    return manifest


def guarded_run():
    config,output=settings();manifest=read(output/'manifest.json')
    if sha(Path(__file__))!=manifest['personal_controller_sha256'] or sha(CONFIG)!=manifest['personal_config_sha256']:
        raise ValueError('Personal controller/config changed; review before resuming')
    stop=threading.Event()
    def monitor():
        while not stop.wait(3):
            if psutil.virtual_memory().available<config['minimum_available_RAM_MiB']*2**20:
                atomic_json(output/'resource_stop.json',{'reason':'low available RAM','available_bytes':psutil.virtual_memory().available,
                    'threshold_MiB':config['minimum_available_RAM_MiB']})
                (output/'STOP').write_text('Resource guard: finish in-flight tasks; RAM below threshold.\n',encoding='utf-8')
                return
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    if (output/'sampling_policy_v2.json').exists():
        from src.analoggym_personal_runner import run as backend
    else:backend=run
    try:backend(output)
    finally:stop.set();watcher.join(timeout=5)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','pilot','promote','run','status','stop'])
    a=p.parse_args();config,output=settings()
    if a.command=='prepare':print(json.dumps(prepare(),indent=2))
    elif a.command in {'pilot','run'}:
        if a.command=='pilot' and read(output/'manifest.json')['stage']!='pilot':raise ValueError('Pilot already promoted')
        guarded_run()
    elif a.command=='promote':print(json.dumps(promote(),indent=2))
    elif a.command=='status':print(json.dumps(read(output/'status.json'),indent=2))
    elif a.command=='stop':(output/'STOP').write_text('User requested graceful stop.\n',encoding='utf-8')


if __name__=='__main__':main()
