"""Read-only campaign checks every five minutes; integrity faults request a drain.

No simulations are started, no records are modified, and no existing STOP is cleared.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import math
import msvcrt
from pathlib import Path
import sqlite3
import sys
import time
import zipfile
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read
from src.analoggym_campaign import sha
from src.windows_campaign_io import atomic_json

OUTPUT=ROOT/'data/generated/analoggym_personal_20k_v1'


def result_errors(result,dimensions):
    errors=[]
    if result['accepted']:
        values=result.get('performance')
        if not isinstance(values,list) or len(values)!=dimensions:
            errors.append('performance dimensions mismatch')
        elif any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in values):
            errors.append('non-finite performance')
        if result.get('failure'):errors.append('accepted result has a failure')
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in result['parameters_native'].values()):
        errors.append('non-finite parameters')
    return errors


def window_summary(rows):
    # Do not blend old and revised samplers when assessing recent failures.
    version=rows[0].get('sampling_policy_version','personal_sampler_v1') if rows else None
    recent=[r for r in rows if r.get('sampling_policy_version','personal_sampler_v1')==version]
    streak=0
    for r in recent:
        if r['accepted']:break
        streak+=1
    return {'sampling_policy':version,'attempts':len(recent),'accepted':sum(r['accepted'] for r in recent),
        'consecutive_failures':streak,'failures':dict(Counter(r.get('failure') for r in recent if not r['accepted']))}


def check(output=OUTPUT):
    started=datetime.now(timezone.utc);state=read(output/'status.json');recipe=read(output/'recipe.json')
    manifest=read(output/'manifest.json');alerts=[];topologies={};raw_checked=0
    def alert(severity,code,detail):alerts.append({'severity':severity,'code':code,'detail':detail})
    for path,expected in recipe['code_sha256'].items():
        if sha(ROOT/path)!=expected:alert('critical','integrity.code.'+path,'Frozen source hash changed')
    if (output/'sampling_policy_v2.json').exists():
        if sha(output/'sampling_policy_v2.json')!=manifest['sampling_policy_file_sha256']:
            alert('critical','integrity.policy','Sampling policy hash changed')
        for path,expected in read(output/'sampling_policy_v2.json')['implementation_sha256'].items():
            if sha(ROOT/path)!=expected:alert('critical','integrity.implementation.'+path,'Revision implementation changed')
    age=(started-datetime.fromisoformat(state['updated_at'])).total_seconds()
    if state['state'] in {'running','draining'}:
        if not psutil.pid_exists(state['pid']):alert('error','worker_missing','Status claims running but worker PID is absent')
        if age>600:alert('error','progress_stale',f'No status update for {age:.0f} seconds')
    elif not state.get('complete'):
        alert('warning','campaign_not_running',state['state'])
    if (output/'STOP').exists():alert('warning','stop_present',(output/'STOP').read_text(encoding='utf-8'))
    memory=psutil.virtual_memory().available;disk=psutil.disk_usage(str(output)).free
    if memory<512*2**20:alert('warning','low_memory',f'{memory/2**20:.0f} MiB available')
    if disk<manifest['minimum_free_disk_GiB']*2**30:alert('error','low_disk',f'{disk/2**30:.2f} GiB free')
    db=sqlite3.connect(f'file:{(output/"ledger.sqlite").as_posix()}?mode=ro',uri=True,timeout=10)
    try:
        db.execute('begin')  # Stable read snapshot; never change the writer journal mode.
        totals=dict(zip(['attempts','accepted'],db.execute('select count(*),sum(accepted) from attempts').fetchone()))
        for entry in recipe['entries']:
            name=entry['name']
            rows=[json.loads(r[0]) for r in db.execute('select result_json from attempts where topology=? order by candidate desc limit 20',(name,))]
            summary=window_summary(rows);topologies[name]=summary
            if summary['consecutive_failures']>=10:
                alert('warning','failure_streak.'+name,f"{summary['consecutive_failures']} consecutive failures")
            if summary['attempts']>=20 and summary['accepted']/summary['attempts']<.5:
                alert('warning','low_yield.'+name,f"{summary['accepted']}/{summary['attempts']} accepted in latest same-policy window")
            for r in rows:
                for error in result_errors(r,15 if entry['family']=='Amplifier' else 13):
                    alert('critical','integrity.labels.'+name,f"candidate {r['candidate']}: {error}")
            # Recompute one recent accepted result per topology from actual waveforms.
            row=db.execute('select result_json,archive,archive_sha256 from attempts where topology=? and accepted=1 order by candidate desc limit 1',(name,)).fetchone()
            if row:
                r=json.loads(row[0]);path=(ROOT/row[1]).resolve()
                try:
                    if not path.is_relative_to(output.resolve()):raise ValueError('Raw path outside campaign')
                    if sha(path)!=row[2]:raise ValueError('Archive SHA mismatch')
                    with zipfile.ZipFile(path) as z:
                        if z.testzip():raise ValueError('Archive CRC mismatch')
                        for member,expected in r['artifacts_sha256'].items():
                            if hashlib.sha256(z.read(member)).hexdigest()!=expected:raise ValueError('Artifact SHA mismatch')
                        from src.analoggym_campaign import extract
                        values=extract(entry,{n:z.read(n) for n in z.namelist() if n.endswith('_wave.dat')})
                        if values!=r['performance']:raise ValueError('Raw waveform label mismatch')
                        if entry['family']=='Amplifier':
                            import io
                            import numpy as np
                            from src.analoggym_personal_quality import ac_diagnostics
                            diagnostic=ac_diagnostics(np.loadtxt(io.BytesIO(z.read('ac_wave.dat')),ndmin=2))
                            if diagnostic['review_required']:raise ValueError('Accepted AC curve fails numerical-continuity screen')
                    raw_checked+=1
                except (OSError,ValueError,AssertionError,zipfile.BadZipFile,KeyError) as error:
                    alert('critical','integrity.archive.'+name,str(error))
    finally:db.close()
    # Only proven integrity faults request a graceful stop. Low yield merely warns.
    if any(a['severity']=='critical' for a in alerts) and not (output/'STOP').exists():
        try:
            with (output/'STOP').open('x',encoding='utf-8') as f:
                f.write('Monitor integrity guard: finish in-flight tasks and retain all evidence; review monitor_status.json.\n')
        except FileExistsError:pass
    result={'checked_at_UTC':started.isoformat(),'interval_seconds':300,'campaign_state':state['state'],
        **totals,'status_age_seconds':age,'available_memory_bytes':memory,'free_disk_bytes':disk,
        'recent_window_per_topology':20,'topologies':topologies,'latest_raw_labels_recomputed':raw_checked,
        'alerts':alerts,'health':'critical' if any(a['severity']=='critical' for a in alerts) else 'attention' if alerts else 'ok',
        'scope':'Recent 20 attempts per topology and latest accepted raw waveform per topology; not a full-dataset audit.',
        'monitor_pid':__import__('os').getpid()}
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
    lock=(OUTPUT/'monitor.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        previous={}
        while True:
            try:result=check()
            except Exception as error:
                result={'checked_at_UTC':datetime.now(timezone.utc).isoformat(),'health':'check_failed',
                    'alerts':[{'severity':'error','code':'monitor_check_failed','detail':repr(error)}]}
            atomic_json(OUTPUT/'monitor_status.json',result)
            current={a['code']:a for a in result['alerts']}
            # Persist transitions, avoiding an ever-growing duplicate alert every five minutes.
            with (OUTPUT/'monitor_events.jsonl').open('a',encoding='utf-8') as f:
                event={'checked_at_UTC':result['checked_at_UTC'],'health':result['health'],
                    'attempts':result.get('attempts'),'accepted':result.get('accepted'),
                    'new_alerts':[a for k,a in current.items() if k not in previous],
                    'resolved_alerts':[k for k in previous if k not in current]}
                f.write(json.dumps(event,ensure_ascii=False)+'\n')
            previous=current
            print(result['checked_at_UTC'],result['health'],result.get('accepted'),flush=True)
            if args.once or result.get('campaign_state')=='complete':return
            time.sleep(300)
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':main()
