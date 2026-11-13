"""Personal runner extension; original campaign engine/runner remain unchanged."""
import json
import os
import platform
import shutil
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import pyarrow as pa
import pyarrow.parquet as pq
from .common import ROOT,read,rel,digest
from .analoggym_campaign import inside,sha
from .analoggym_campaign_runner import (connect,store_result as base_store_result,metadata,now,record as base_record)
from .windows_campaign_io import atomic_json,replace_with_retry
from .analoggym_personal_quality import apply_quality,POLICY
from .analoggym_personal_engine import simulate,parameter_plan,sampling_trace


def record(entry,result,output,recipe_hash):
    row=base_record(entry,result,output,recipe_hash)
    row['sampling_policy_version']=result.get('sampling_policy_version','personal_sampler_v1')
    row['sampling_policy_sha256']=result.get('sampling_policy_sha256') or recipe_hash
    return row


def store_result(db,entry,result):
    return base_store_result(db,entry,apply_quality(entry,result))


def status(output,db,manifest,state):
    """Original status schema, with bounded Windows-safe atomic replacement."""
    grouped={n:{'attempts':0,'accepted':0,'target':q} for n,q in manifest['quotas'].items()}
    for name,count,ok in db.execute('select topology,count(*),sum(accepted) from attempts group by topology'):
        grouped[name].update(attempts=count,accepted=ok)
    attempts=sum(x['attempts'] for x in grouped.values());accepted=sum(x['accepted'] for x in grouped.values())
    size=db.execute('select coalesce(sum(archive_bytes),0) from attempts').fetchone()[0]
    s={'state':state,'updated_at':now(),'pid':os.getpid(),'target_accepted':manifest['target_accepted'],
       'attempts':attempts,'accepted':accepted,'quarantined':attempts-accepted,'topologies':grouped,
       'raw_bytes':size,'free_disk_bytes':shutil.disk_usage(output).free,'workers':manifest['workers'],
       'merged_with_historical':False,'complete':state=='complete' and accepted>=manifest['target_accepted']}
    atomic_json(output/'status.json',s)
    return s

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
            same=path.exists() and 'sampling_policy_version' in pq.read_schema(path).names and pq.read_table(path,columns=['candidate'])['candidate'].to_pylist()==current
            if not same:
                records=[record(entry,json.loads(r[1]),output,manifest['recipe_hash']) for r in batch]
                table=pa.Table.from_pylist(records)
                # Keep all-null AMP temperature a typed nullable double.
                table=table.set_column(table.schema.get_field_index('temperature'),'temperature',pa.array([r['temperature'] for r in records],type=pa.float64()))
                path.parent.mkdir(parents=True,exist_ok=True); index_path.parent.mkdir(parents=True,exist_ok=True)
                tmp=path.with_suffix('.parquet.tmp'); pq.write_table(table,tmp,compression='zstd'); replace_with_retry(tmp,path)
                idx=table.select(['sample_id','topology_id','circuit_family','graph','design_group','source_path','candidate'])
                tmp=index_path.with_suffix('.parquet.tmp'); pq.write_table(idx,tmp,compression='zstd'); replace_with_retry(tmp,index_path)
            chunks.append({'path':rel(path),'index':rel(index_path),'rows':size})
    atomic_json(output/'dataset.json',{'recipe_hash':manifest['recipe_hash'],'sample_count':sum(c['rows'] for c in chunks),
                                     'topologies':len({c['path'].split('/')[-2] for c in chunks}),
                                     'quality_policy_file':manifest.get('quality_policy_file'),
                                     'quality_policy_sha256':manifest.get('quality_policy_file_sha256'),
                                     'shards':chunks,'updated_at':now(),'merged_with_historical':False})


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
    if sha(output/'sampling_policy_v2.json')!=manifest['sampling_policy_file_sha256']:
        raise ValueError('Versioned sampling policy changed')
    policy=read(output/'sampling_policy_v2.json')
    if sha(output/'quality_policy_v1.json')!=manifest['quality_policy_file_sha256']:
        raise ValueError('Quality policy changed')
    if read(output/'quality_policy_v1.json')['policy']!=POLICY:
        raise ValueError('Quality policy implementation mismatch')
    if policy['base_recipe_hash']!=manifest['recipe_hash']:raise ValueError('Sampler base recipe mismatch')
    for name,expected in policy['implementation_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Versioned sampler implementation changed: '+name)
    entries={e['name']:dict(e) for e in recipe['entries']}
    entries['ldo_folded_cascode']['_sampling_policy']=policy['sampling_policy']
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
                    name=futures.pop(job); inflight[name]-=1; result=apply_quality(entries[name],job.result())
                    accepted=base_store_result(db,entries[name],result)
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
                check(output,deep_per_topology=-1)
                from scripts.check_personal_quality import check as check_quality
                check_quality(output)
            status(output,db,manifest,'complete' if count>=manifest['target_accepted'] else stopping or 'attempt_budget_exhausted')
    except BaseException as err:
        atomic_json(output/'last_error.json',{'time':now(),'type':type(err).__name__,'message':str(err)})
        status(output,db,manifest,'error'); raise
    finally:
        db.close(); lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1); lock.close()

