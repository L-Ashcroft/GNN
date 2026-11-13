"""Lightweight, read-only training diagnostics. No torch/CUDA imports."""
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ROOT=Path(__file__).resolve().parents[1]

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def now():return datetime.now(timezone.utc).isoformat()
def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def observer_json(value):
    """Preserve invalid-number evidence in observer output without changing input."""
    if isinstance(value,float) and not math.isfinite(value):return {'__nonfinite__':str(value)}
    if isinstance(value,dict):return {k:observer_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [observer_json(v) for v in value]
    return value

def atomic(path,value):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Output outside EDA')
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(observer_json(value),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    for attempt in range(5):
        try:temporary.replace(path);return
        except PermissionError:
            if attempt==4:raise
            time.sleep(.1*(attempt+1))

def nonfinite_paths(value,path='root'):
    if isinstance(value,float) and not math.isfinite(value):return [path]
    if isinstance(value,dict):return [p for k,v in value.items() for p in nonfinite_paths(v,path+'.'+k)]
    if isinstance(value,list):return [p for i,v in enumerate(value) for p in nonfinite_paths(v,f'{path}[{i}]')]
    return []

def age_seconds(timestamp,clock=None):
    return ((clock or datetime.now(timezone.utc))-datetime.fromisoformat(timestamp)).total_seconds()

def runtime_findings(status,process_matches,available_ram,disk_free,clock=None,validation_seconds=300):
    issues=[];state=status.get('state');terminal=state in {'complete','error','paused'}
    def issue(code,severity,message):issues.append({'code':code,'severity':severity,'message':message})
    if nonfinite_paths(status):issue('nonfinite_live_status','critical','Live status contains NaN/Inf')
    if state in {'error','paused'}:issue('training_'+state,'critical',status.get('reason') or status.get('error') or state)
    if not process_matches and state!='complete':issue('original_process_missing_or_changed','critical','Original PID/creation time no longer matches')
    if available_ram<512*2**20:issue('low_RAM','critical','Available RAM below 512 MiB; formal runner pauses below 256 MiB')
    elif available_ram<1024*2**20:issue('low_RAM','warning','Available RAM below 1 GiB')
    if disk_free<2**30:issue('low_disk','critical','Free disk space below 1 GiB')
    elif disk_free<10*2**30:issue('low_disk','warning','Free disk space below 10 GiB')
    if not terminal:
        age=age_seconds(status['updated_at'],clock)
        slow_phase=state in {'validating','testing_once','initializing'}
        warn=max(900.,3*validation_seconds) if slow_phase else 120.
        critical=warn*2 if slow_phase else 600.
        if age>critical:issue('status_stale','critical',f'No status heartbeat for {age:.0f} seconds ({state})')
        elif age>warn:issue('status_stale','warning',f'No status heartbeat for {age:.0f} seconds ({state})')
    return issues

def history_summary(history):
    summaries=[]
    for epoch in history:
        validation=epoch['validation'];targets=validation['per_target']
        defined=[r for r in targets if r.get('r2') is not None and math.isfinite(r['r2'])]
        r2=[r['r2'] for r in defined];worst=sorted(defined,key=lambda r:r['r2'])[:10]
        grouped={}
        for target in targets:
            source=target['semantic_id'].split('::')[0].split('__')[0]
            grouped.setdefault(source,[]).append(target)
        def metrics(rows):
            def mean(field):
                values=[r[field] for r in rows if r.get(field) is not None and math.isfinite(r[field])]
                return statistics.mean(values) if values else None
            values=[r['r2'] for r in rows if r.get('r2') is not None and math.isfinite(r['r2'])]
            return {'target_combinations':len(rows),'hit5':mean('hit5'),'hit10':mean('hit10'),
                    'median_r2':statistics.median(values) if values else None,'positive_r2':sum(v>0 for v in values)}
        summaries.append({'epoch':epoch['epoch'],'train_loss':epoch['training_loss'],'validation_loss':validation['loss'],
            'hit5':validation['macro_hit5'],'hit10':validation['macro_hit10'],'macro_r2':validation['macro_r2'],
            'median_r2':statistics.median(r2) if r2 else None,'positive_r2':sum(r>0 for r in r2),
            'r2_above_08':sum(r>.8 for r in r2),'defined_r2':len(r2),'r2_below_minus100':sum(r< -100 for r in r2),
            'target_combinations':len(targets),'validation_samples':validation['samples'],
            'training_samples':epoch['training_samples'],'gradient_norm_mean':epoch['gradient_norm_mean'],
            'gradient_norm_max':epoch['gradient_norm_max'],'training_seconds':epoch['training_seconds'],
            'validation_seconds':epoch['validation_seconds'],
            'worst_r2':[{'group':r['supervision_group'],'semantic':r['semantic_id'],'r2':r['r2'],'mae':r['mae'],'count':r['count']} for r in worst],
            'by_source':{source:metrics(rows) for source,rows in grouped.items()}})
    return summaries

def quality_findings(summary):
    if not summary:return []
    current=summary[-1];issues=[]
    if current['r2_below_minus100']:
        issues.append({'code':'extreme_negative_r2','severity':'warning','message':f"{current['r2_below_minus100']} target combinations have R2 < -100; do not suppress or clip original results"})
    if len(summary)>=2:
        previous=summary[-2]
        if current['hit5'] is not None and previous['hit5'] is not None and current['hit5']<previous['hit5']-.02:
            issues.append({'code':'validation_hit5_drop','severity':'warning','message':'Validation macro hit@5 fell by more than 2 percentage points'})
        if current['validation_loss']>previous['validation_loss']*1.2:
            issues.append({'code':'validation_loss_jump','severity':'warning','message':'Validation loss increased more than 20%'})
    if current['gradient_norm_max']>100:
        issues.append({'code':'large_gradient_norm','severity':'warning','message':'Epoch gradient norm exceeded 100 before clipping'})
    return issues
