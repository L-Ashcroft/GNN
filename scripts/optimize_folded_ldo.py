"""Independent real-simulation comparison; never imports trial labels into production."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor,as_completed
import io
import json
from pathlib import Path
import sqlite3
import sys
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read,dump
from src.analoggym_campaign import atomic_json
from src.analoggym_personal_engine import simulate

PRODUCTION=ROOT/'data/generated/analoggym_personal_20k_v1'
OUT=ROOT/'data/generated/analoggym_folded_optimization_v2'


def prepare():
    recipe=read(PRODUCTION/'recipe.json');entry=next(e for e in recipe['entries'] if e['name']=='ldo_folded_cascode')
    db=sqlite3.connect(f'file:{(PRODUCTION/"ledger.sqlite").as_posix()}?mode=ro',uri=True)
    rows=db.execute("select result_json from attempts where topology='ldo_folded_cascode' and accepted=1").fetchall();db.close()
    ranked=[]
    for row, in rows:
        r=json.loads(row)
        with zipfile.ZipFile(ROOT/r['archive']) as z:a=np.loadtxt(io.BytesIO(z.read('load_wave.dat')),ndmin=2)
        score=float(np.max(np.abs(a[:,1]-1.2)))
        ranked.append((score,r))
    ranked.sort(key=lambda x:(x[0],x[1]['candidate']))
    best=[r for score,r in ranked[:3]]
    assert len(best)==3
    centres=[r['parameters_native'] for r in best]
    strategies={
        'control':None,
        'narrow_original':{'mode':'independent','span':.05,'bias_span':.02,'centers':entry['sampling_centers']},
        'narrow_recentered':{'mode':'independent','span':.05,'bias_span':.02,'centers':centres},
        'correlated_recentered':{'mode':'correlated','span':.12,'bias_span':.01,'centers':centres}}
    plan={'entry':entry,'environment':recipe['environment'],'strategies':strategies,
        'recentred_provenance':[{'archive':r['archive'],'candidate':r['candidate'],'performance':r['performance']} for r in best],
        'selection':'Only fully accepted production designs; rank worst absolute 1mA/10mA DC regulation error. No quality threshold changed.',
        'trial_results_not_counted_in_production':True}
    OUT.mkdir(parents=True,exist_ok=True);atomic_json(OUT/'plan.json',plan)
    return plan


def run_trial(phase,count,selected=None):
    plan=read(OUT/'plan.json');jobs=[];results=[]
    strategies=[selected] if selected else list(plan['strategies'])
    for strategy in strategies:
        for i in range(count):
            entry=copy.deepcopy(plan['entry']);policy=plan['strategies'][strategy]
            if policy:
                entry['_sampling_policy']={**policy,'start_candidate':0,'seed_offset':20261028,
                    'version':'folded_v2_trial_'+strategy,'vary_discrete':False}
            candidate=(100000 if phase=='screen' else 200000)+i
            jobs.append((strategy,entry,candidate))
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(simulate,e,c,OUT/phase/strategy,plan['environment'],120):(strategy,c)
                 for strategy,e,c in jobs}
        for future in as_completed(futures):
            strategy,candidate=futures[future];result=future.result()
            results.append({'strategy':strategy,**result})
            atomic_json(OUT/(phase+'_progress.json'),{'completed':len(results),'planned':len(jobs),'results':results})
            print(phase,strategy,candidate,result['accepted'],result['failure'] or '',flush=True)
    summary={s:{'attempts':sum(r['strategy']==s for r in results),
                'accepted':sum(r['strategy']==s and r['accepted'] for r in results)} for s in strategies}
    atomic_json(OUT/(phase+'_summary.json'),{'summary':summary,'results':results})
    print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','screen','validate']);p.add_argument('--strategy');p.add_argument('--count',type=int)
    a=p.parse_args()
    if a.phase=='prepare':prepare()
    else:run_trial(a.phase,a.count or (10 if a.phase=='screen' else 20),a.strategy)
