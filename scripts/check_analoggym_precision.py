"""Compare the same physical design under standard and tighter solver settings."""
import json
import argparse
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.analoggym_campaign import read, atomic_json, AMP_METRICS, LDO_METRICS


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--standard',default='data/generated/analoggym_standard_v4')
    p.add_argument('--tighter',default='data/generated/analoggym_precision_v4')
    p.add_argument('--report',default='reports/analoggym_campaign_precision_final.json')
    a=p.parse_args()
    refs=read(ROOT/a.standard/'summary.json')['results']
    precise=read(ROOT/a.tighter/'summary.json')['results']
    checks=[]
    for result in precise:
        candidates=[r for r in refs if r['accepted'] and r['topology']==result['topology'] and r['parameters_native']==result['parameters_native']]
        if not candidates: raise ValueError('Precision design has no identical standard reference')
        ref=candidates[0]; ok=result['accepted']; details=[]
        if ok:
            definitions=LDO_METRICS if 'ldo' in result['topology'].lower() else AMP_METRICS
            for i,((name,unit,_),v,w) in enumerate(zip(definitions,ref['performance'],result['performance'])):
                absolute={'1/K':1e-8,'V':2e-5,'W':1e-8,'dB':.1,'degree':.25,
                          'Hz':1.,'V/s':10.,'s':20e-9,'V/A':.002,'V/V':2e-4}[unit]
                limit=absolute+.05*abs(v)
                passed=abs(w-v)<=limit
                details.append({'metric':name,'standard':v,'tighter':w,'absolute_difference':abs(w-v),'limit':limit,'passed':bool(passed)})
                ok=ok and passed
        checks.append({'topology':result['topology'],'passed':bool(ok),'standard_archive':ref['archive'],
                       'tighter_archive':result['archive'],'failure':result['failure'],'metrics':details})
    report={'passed':all(c['passed'] for c in checks),'topologies_checked':len(checks),
            'scope':'One identical design per topology; 10x tighter relative tolerance and 2x smaller maximum transient step, with fixed absolute voltage/current tolerances. Not proof of accuracy throughout design space.',
            'checks':checks}
    atomic_json(ROOT/a.report,report)
    print(json.dumps({'passed':report['passed'],'topologies_checked':len(checks),'failed':[c['topology'] for c in checks if not c['passed']]},indent=2))
    if not report['passed']: raise SystemExit(1)


if __name__=='__main__': main()
