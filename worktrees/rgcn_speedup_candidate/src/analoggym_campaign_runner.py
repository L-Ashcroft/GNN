"""Durable native Windows campaign scheduler, SQLite ledger and Parquet shards."""
import json
import os
import platform
import shutil
import sqlite3
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .analoggym_campaign import (ROOT, UPSTREAM, DEFAULT_OUTPUT, AMP_METRICS, LDO_METRICS,
    inside, sha, read, rel, digest, atomic_json, catalog, conditions, testbenches,
    parameter_plan, simulate, evaluate, extract)


def now(): return datetime.now(timezone.utc).isoformat()


def metadata(entry):
    metrics=AMP_METRICS if entry['family']=='Amplifier' else LDO_METRICS
    group=digest([entry['family'],conditions(entry),metrics])
    return {'group_id':group,'max_dimensions':16,'conditions':conditions(entry),
            'mask_policy':'only unused vector slots are missing; measurement failures quarantine entire candidate',
            'metrics':[{'slot':i,'name':name,'unit':unit,'definition':definition,
                        'semantic_id':'AnalogGym_generated::'+group+'::'+name}
                       for i,(name,unit,definition) in enumerate(metrics)]}


def initialize(output, qualification, target=100000, workers=4, max_attempts_factor=8):
    output=inside(output); qualification=inside(qualification)
    if (output/'manifest.json').exists(): raise ValueError('Already initialized; use run/status')
    entries,excluded=catalog(); results=read(qualification/'summary.json')['results']
    selected=[]
    for e in entries:
        successes=sorted((r for r in results if r['topology']==e['name'] and r['accepted']),key=lambda r:r['candidate'])
        if not successes:
            failures=[r for r in results if r['topology']==e['name']]
            excluded.append({'name':e['name'],'reason':'No valid seed in actual qualification simulations',
                             'attempts':len(failures),'failures':dict(Counter(r['failure'] for r in failures))})
            continue
        for r in successes:
            with zipfile.ZipFile(ROOT/r['archive']) as z:
                if z.testzip(): raise ValueError('Corrupt qualification evidence')
                actual=json.loads(z.read('result.json'))
                if actual != r: raise ValueError('Qualification result mismatch')
                for file,expected in actual['artifacts_sha256'].items():
                    if hashlib_bytes(z.read(file))!=expected: raise ValueError('Qualification artifact modified')
                measured=extract(e,{s:z.read(s) for s in z.namelist() if s.endswith('_wave.dat')})
                np.testing.assert_array_equal(measured,r['performance'])
                env_check=read(ROOT/'config/simulation_environment.json')
                for kind,expected in testbenches(e,ROOT/env_check['library']).items():
                    if normalize_deck(z.read(kind+'.cir').decode(),ignore_resolution=True)!=normalize_deck(expected,ignore_resolution=True):
                        raise ValueError('Qualification testbench conditions differ: '+e['name'])
        e['sampling_centers']=[r['parameters_native'] for r in successes]
        e['center_provenance']=[{'candidate':r['candidate'],'archive':r['archive'],'sha256':sha(ROOT/r['archive'])} for r in successes]
        selected.append(e)
    if not selected: raise ValueError('No qualified topologies')
    env=read(ROOT/'config/simulation_environment.json')
    # Verify the actual installed model files, not just an archive filename.
    model=read(ROOT/env['pdk_manifest'])
    for name,expected in model['files_sha256'].items():
        if sha(ROOT/name)!=expected: raise ValueError('Model modified: '+name)
    source_files=['src/analoggym_campaign.py','src/analoggym_campaign_runner.py','src/analoggym_generation.py',
                  'src/netlist.py','src/common.py','src/load_analoggym_generated.py','scripts/check_analoggym_campaign.py']
    recipe={'source_commit': '0a9d1390ade361e2b4a2d33181e22367edbb8afc',
            'code_sha256':{s:sha(ROOT/s) for s in source_files},'environment':env,
            'executable_sha256':sha(ROOT/env['ngspice_executable']),
            'model_manifest_sha256':sha(ROOT/env['pdk_manifest']), 'model_archive_sha256':model['archive_sha256'],
            'platform':platform.platform(),'seed':42,
            'sampling':'deterministic per topology/index; successful measured qualification anchors; 20/60/20 percent local log boxes; caps x1/1.35..1.35; shared parameters/aliases retained',
            'geometry_bounds_um':{'W':[.42,100.],'L':[.15,30.]},
            'source_baseline_adaptations':'Tan_CLIA two 0.253 um NFET widths bounded to >=0.42 um; Yan_AZ load 15 nF from source CLOAD; other amps 500 pF; amp input edges 1 ns; LDO output load 100 nF; fixed explicit fixtures; no source file edits',
            'entries':selected,'excluded':excluded,
            'qualification':rel(qualification),
            'testbench_sha256':{e['name']:{k:hashlib_text(v) for k,v in testbenches(e,ROOT/env['library']).items()} for e in selected},
            'label_mapping':{e['name']:metadata(e) for e in selected}}
    recipe_id=digest(recipe)
    atomic_json(output/'recipe.json',recipe)
    quota={e['name']:target//len(selected)+(i<target%len(selected)) for i,e in enumerate(selected)}
    manifest={'recipe_hash':recipe_id,'created_at':now(),'target_accepted':target,'quotas':quota,
              'workers':workers,'timeout_seconds':120,'max_attempts_factor':max_attempts_factor,
              'minimum_free_disk_GiB':20,'maximum_campaign_disk_GiB':180,
              'status':'prepared','merged_with_historical':False}
    atomic_json(output/'manifest.json',manifest)
    for e in selected:
        folder=output/'topologies'/e['name']
        atomic_json(folder/'graph.json',e['graph']); atomic_json(folder/'conditions.json',conditions(e))
        atomic_json(folder/'performance_mapping.json',metadata(e))
        atomic_json(folder/'parameter_sampling.json',{'baseline':e['baseline'],'definitions':e['definitions'],
                    'sampling_centers':e['sampling_centers'],'center_provenance':e['center_provenance'],
                    'parameter_units':{k:unit(k) for k in e['baseline']}})
    return manifest


def hashlib_text(text):
    import hashlib
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def hashlib_bytes(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def normalize_deck(text,ignore_resolution=False):
    """Ignore whitespace/case and equivalent numeric SPICE literals only."""
    def token(t):
        try: return format(evaluate(t.lower(),{}),'.12g')
        except (ValueError,KeyError,SyntaxError,TypeError): return t.lower()
    lines=[[token(t) for t in line.split()] for line in text.splitlines() if line.strip() and not line.lstrip().startswith('*')]
    if ignore_resolution:
        lines=[line for line in lines if line[0]!='.options']
        for line in lines:
            if line[0]=='tran' and len(line)==5: line[1]=line[4]='resolution'
    return lines


def unit(key):
    import re
    if re.search(r'(^w_|_w_|^l_|_l_)',key): return 'um'
    if key.startswith('current'): return 'A'
    if key.startswith('capacitor') or key=='cload': return 'F'
    if key.startswith('resistor'): return 'ohm'
    if key.startswith('vb') or key=='vcm': return 'V'
    return '1'


def connect(output):
    db=sqlite3.connect(output/'ledger.sqlite',timeout=30)
    db.execute('pragma journal_mode=WAL'); db.execute('pragma synchronous=FULL')
    db.execute('''create table if not exists attempts(
        topology text not null, candidate integer not null, accepted integer not null,
        design_group text, result_json text not null, archive text not null,
        archive_sha256 text not null, archive_bytes integer not null,
        primary key(topology,candidate))''')
    db.execute('create unique index if not exists unique_design on attempts(design_group) where accepted=1')
    db.execute('create index if not exists accepted_by_topology on attempts(topology,accepted,candidate)')
    db.commit(); return db


def design_fingerprint(entry,params):
    attrs=[{k:evaluate(v,params) for k,v in d['attributes'].items()} for d in entry['graph']['devices']]
    external={k:v for k,v in params.items() if k.startswith('vb') or k.startswith('current')}
    return digest([entry['graph']['topology_hash'],attrs,external,conditions(entry)])


def store_result(db,entry,result):
    archive=ROOT/result['archive']; design=design_fingerprint(entry,result['parameters_native'])
    accepted=bool(result['accepted'])
    if accepted and db.execute('select 1 from attempts where design_group=? and accepted=1',(design,)).fetchone():
        result={**result,'accepted':False,'failure':'Duplicate resolved design; original simulation preserved in archive'}; accepted=False
    db.execute('insert into attempts values(?,?,?,?,?,?,?,?)',
        (entry['name'],result['candidate'],int(accepted),design,json.dumps(result,allow_nan=False),rel(archive),sha(archive),archive.stat().st_size))
    db.commit(); return accepted


def record(entry,result,output,recipe_hash):
    params=result['parameters_native']; graph=entry['graph']; mapping=metadata(entry)
    features=[[None if x is None else evaluate(x,params) for x in row] for row in graph['feature_expressions']]
    values=result['performance']; design=design_fingerprint(entry,params)
    return {'sample_id':'AnalogGym_generated:'+recipe_hash[:16]+':'+design,
            'source_dataset':'AnalogGym_generated','topology_id':entry['name'],
            'circuit_family':entry['family'],'process':'SKY130','PDK':mapping['conditions']['PDK'],
            'temperature':27. if entry['family']=='LDO' else None,'supply_voltage':1.8,
            'testbench_id':mapping['group_id'],'graph':rel(output/'topologies'/entry['name']/'graph.json'),
            'conditions_json':json.dumps(mapping['conditions'],sort_keys=True),
            'parameter_names':list(params),'parameters':list(params.values()),'parameter_units':[unit(k) for k in params],
            'device_features':features,'performance':values+[None]*(16-len(values)),
            'mask':[True]*len(values)+[False]*(16-len(values)),
            'semantic_ids':[m['semantic_id'] for m in mapping['metrics']]+[None]*(16-len(values)),
            'source_path':result['archive'],'source_sha256':sha(ROOT/result['archive']),
            'candidate':result['candidate'],'design_group':design,'recipe_hash':recipe_hash,
            'generation_status':'measured_standalone_not_merged'}


def export(output,db,recipe,manifest):
    """Immutable completed chunks plus one atomically replaced tail per topology."""
    entries={e['name']:e for e in recipe['entries']}; chunks=[]
    for name,entry in entries.items():
        rows=db.execute('select candidate,result_json from attempts where topology=? and accepted=1 order by candidate',(name,)).fetchall()
        for start in range(0,len(rows),256):
            batch=rows[start:start+256]; path=output/'samples'/name/f'part-{start//256:05d}.parquet'
            index_path=output/'index'/name/path.name
            size=len(batch)
            # Candidate order can change while workers finish, so compare ordered IDs.
            current=[r[0] for r in batch]
            same=path.exists() and pq.read_table(path,columns=['candidate'])['candidate'].to_pylist()==current
            if not same:
                records=[record(entry,json.loads(r[1]),output,manifest['recipe_hash']) for r in batch]
                table=pa.Table.from_pylist(records)
                # Keep all-null AMP temperature a typed nullable double.
                table=table.set_column(table.schema.get_field_index('temperature'),'temperature',pa.array([r['temperature'] for r in records],type=pa.float64()))
                path.parent.mkdir(parents=True,exist_ok=True); index_path.parent.mkdir(parents=True,exist_ok=True)
                tmp=path.with_suffix('.parquet.tmp'); pq.write_table(table,tmp,compression='zstd'); os.replace(tmp,path)
                idx=table.select(['sample_id','topology_id','circuit_family','graph','design_group','source_path','candidate'])
                tmp=index_path.with_suffix('.parquet.tmp'); pq.write_table(idx,tmp,compression='zstd'); os.replace(tmp,index_path)
            chunks.append({'path':rel(path),'index':rel(index_path),'rows':size})
    atomic_json(output/'dataset.json',{'recipe_hash':manifest['recipe_hash'],'sample_count':sum(c['rows'] for c in chunks),
                                     'topologies':len({c['path'].split('/')[-2] for c in chunks}),
                                     'shards':chunks,'updated_at':now(),'merged_with_historical':False})


def status(output,db,manifest,state):
    grouped={n:{'attempts':0,'accepted':0,'target':q} for n,q in manifest['quotas'].items()}
    for name,count,ok in db.execute('select topology,count(*),sum(accepted) from attempts group by topology'):
        grouped[name].update(attempts=count,accepted=ok)
    attempts=sum(x['attempts'] for x in grouped.values()); accepted=sum(x['accepted'] for x in grouped.values())
    size=db.execute('select coalesce(sum(archive_bytes),0) from attempts').fetchone()[0]
    s={'state':state,'updated_at':now(),'pid':os.getpid(),'target_accepted':manifest['target_accepted'],
       'attempts':attempts,'accepted':accepted,'quarantined':attempts-accepted,'topologies':grouped,
       'raw_bytes':size,'free_disk_bytes':shutil.disk_usage(output).free,'workers':manifest['workers'],
       'merged_with_historical':False,'complete':state=='complete' and accepted>=manifest['target_accepted']}
    atomic_json(output/'status.json',s)
    return s


def run(output):
    if platform.system()!='Windows': raise RuntimeError('Native Windows required')
    output=inside(output); manifest=read(output/'manifest.json'); recipe=read(output/'recipe.json')
    if digest(recipe)!=manifest['recipe_hash']: raise ValueError('Recipe changed')
    for name,expected in recipe['code_sha256'].items():
        if sha(ROOT/name)!=expected: raise ValueError('Code changed; restore pinned code or initialize a new campaign: '+name)
    env=recipe['environment']
    if sha(ROOT/env['ngspice_executable'])!=recipe['executable_sha256']: raise ValueError('Simulator changed')
    if sha(ROOT/env['pdk_manifest'])!=recipe['model_manifest_sha256']: raise ValueError('Model manifest changed')
    for path,expected in read(ROOT/env['pdk_manifest'])['files_sha256'].items():
        if sha(ROOT/path)!=expected: raise ValueError('Model changed: '+path)
    entries={e['name']:e for e in recipe['entries']}
    for e in entries.values():
        if sha(ROOT/e['dut'])!=e['dut_sha256']: raise ValueError('Upstream DUT changed')
    # Windows file lock releases automatically after a crash; stale PID alone is
    # never used to kill a process or authorize concurrent writers.
    import msvcrt
    lock=(output/'runner.lock').open('a+b'); lock.seek(0)
    if lock.read(1)==b'': lock.write(b'0'); lock.flush()
    lock.seek(0)
    try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError: lock.close(); raise RuntimeError('Another runner already holds this campaign lock')
    db=connect(output); atomic_json(output/'worker.json',{'pid':os.getpid(),'started_at':now(),'output':rel(output)})
    # Recover archives committed before an interrupted SQLite transaction.
    for n,e in entries.items():
        known={x[0] for x in db.execute('select candidate from attempts where topology=?',(n,))}
        for p in sorted((output/'raw'/n).glob('*/*.zip')):
            if int(p.stem) in known: continue
            with zipfile.ZipFile(p) as z:
                if z.testzip(): raise ValueError('Corrupt orphan archive')
                r=json.loads(z.read('result.json'))
            if r['parameters_native']!=parameter_plan(e,r['candidate']): raise ValueError('Orphan parameters changed')
            store_result(db,e,r)
    next_id={n:db.execute('select coalesce(max(candidate),-1)+1 from attempts where topology=?',(n,)).fetchone()[0] for n in entries}
    current=status(output,db,manifest,'running'); export(output,db,recipe,manifest)
    inflight=Counter(); futures={}; stopping=None; last_export=time.monotonic(); last_status=last_export
    try:
        with ThreadPoolExecutor(max_workers=manifest['workers']) as pool:
            while True:
                if (output/'STOP').exists(): stopping='stopped_by_request'
                if shutil.disk_usage(output).free < manifest['minimum_free_disk_GiB']*2**30: stopping='stopped_low_disk'
                if current['raw_bytes'] > manifest['maximum_campaign_disk_GiB']*2**30: stopping='stopped_storage_budget'
                while not stopping and len(futures)<manifest['workers']:
                    candidates=[]
                    for name,stats in current['topologies'].items():
                        if stats['accepted']+inflight[name]>=stats['target']: continue
                        if next_id[name]>=stats['target']*manifest['max_attempts_factor']: continue
                        candidates.append(name)
                    if not candidates: break
                    # Give every topology comparable progress and avoid domination.
                    name=min(candidates,key=lambda n:((current['topologies'][n]['accepted']+inflight[n])/manifest['quotas'][n],next_id[n],n))
                    job=pool.submit(simulate,entries[name],next_id[name],output,env,manifest['timeout_seconds'])
                    futures[job]=name; next_id[name]+=1; inflight[name]+=1
                if not futures: break
                done,_=wait(futures,timeout=5,return_when=FIRST_COMPLETED)
                for job in done:
                    name=futures.pop(job); inflight[name]-=1; result=job.result()
                    accepted=store_result(db,entries[name],result)
                    stats=current['topologies'][name]; stats['attempts']+=1; stats['accepted']+=accepted
                    current['raw_bytes']+=(ROOT/result['archive']).stat().st_size
                    print(now(),name,result['candidate'],'accepted' if accepted else 'quarantined',result['failure'] or '',flush=True)
                if time.monotonic()-last_status>=10 or done:
                    current=status(output,db,manifest,'draining' if stopping else 'running'); last_status=time.monotonic()
                if time.monotonic()-last_export>=60:
                    export(output,db,recipe,manifest); last_export=time.monotonic()
            export(output,db,recipe,manifest)
            count=db.execute('select count(*) from attempts where accepted=1').fetchone()[0]
            if count>=manifest['target_accepted']:
                status(output,db,manifest,'final_quality_check')
                from scripts.check_analoggym_campaign import check
                check(output,deep_per_topology=3)
            status(output,db,manifest,'complete' if count>=manifest['target_accepted'] else stopping or 'attempt_budget_exhausted')
    except BaseException as err:
        atomic_json(output/'last_error.json',{'time':now(),'type':type(err).__name__,'message':str(err)})
        status(output,db,manifest,'error'); raise
    finally:
        db.close(); lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1); lock.close()
