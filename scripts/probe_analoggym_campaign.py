"""Finite multi-topology qualification, distinct from the production campaign."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
import time
import itertools
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.analoggym_campaign import catalog, simulate, read, atomic_json, inside, param_definitions, resolve_parameters


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='data/generated/analoggym_qualification_v1')
    p.add_argument('--candidates',type=int,default=1)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--timeout',type=int,default=120)
    p.add_argument('--names',nargs='*')
    p.add_argument('--precision',action='store_true')
    p.add_argument('--recovery',action='store_true',help='Explore documented bias/compensation seeds for failed baseline topologies')
    p.add_argument('--extra-seeds',action='store_true')
    p.add_argument('--qualified',nargs='+',help='Use previous measured parameter points as seeds (labels are always newly simulated)')
    a=p.parse_args(); output=inside(ROOT/a.output)
    entries,excluded=catalog()
    if a.names: entries=[e for e in entries if e['name'] in a.names]
    if a.recovery:
        for e in entries:
            centers=[]
            for bias,cap in itertools.product((1.,.25,.1), (1.,4.,16.,64.)):
                center=e['baseline'].copy()
                for k in center:
                    if k.startswith('current'): center[k]*=bias
                    if k.startswith('capacitor'): center[k]*=cap
                if e['name']=='ldo_folded_cascode':
                    center['m_m10']*=cap; center['vb1']*=max(.5,bias)
                centers.append(center)
            e['sampling_centers']=centers
            e['recovery_method']='bias x[1,.25,.1], compensation x[1,4,16,64]; folded LDO pass multiplicity sweep and tail bias adjustment; then documented local perturbation'
    if a.extra_seeds:
        for e in entries:
            centers=[]
            for cap0,cap1 in itertools.product((.25,1.,4.),(.25,1.,4.)):
                c=e['baseline'].copy()
                if e['name']=='Leung_NMCF_Pin_3':
                    tuned=resolve_parameters(param_definitions(ROOT/'third_party/AnalogGym/RGNN_RL/simulations/AMP_NMCF_vars.spice'))
                    c.update({k:v for k,v in tuned.items() if k in c})
                    c['capacitor_0']=2e-12*cap0; c['capacitor_1']=20e-12*cap1
                elif e['name']=='Yan_AZ_Pin_3':
                    c['capacitor_0']*=cap0; c['capacitor_1']*=cap1
                    for k in c:
                        if k.startswith('current'): c[k]*=.1
                centers.append(c)
            e['sampling_centers']=centers
            e['recovery_method']='NMCF: upstream RGNN device dimensions/bias, independently selected ideal 2p/20p compensation centers; Yan: bias x0.1 and asymmetric compensation; ldo_1: 100nF output fixture; local perturbations'
    if a.qualified:
        qualified=[r for path in a.qualified for r in read(ROOT/path/'summary.json')['results']]
        for e in entries:
            good=[r for r in qualified if r['topology']==e['name'] and r['accepted']]
            e['sampling_centers']=[r['parameters_native'] for r in good] or [e['baseline']]
    atomic_json(output/'catalog.json',{'entries':entries,'excluded':excluded})
    env=read(ROOT/'config/simulation_environment.json')
    started=time.perf_counter(); results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(simulate,e,i,output,env,a.timeout,a.precision)
              for i in range(a.candidates) for e in entries]
        for job in as_completed(jobs):
            r=job.result(); results.append(r)
            print(r['topology'],r['candidate'],'OK' if r['accepted'] else 'REJECT',
                  round(r['elapsed_seconds'],2),r['failure'] or '',flush=True)
            atomic_json(output/'progress.json',{'finished':len(results),'planned':len(jobs),'results':results})
    summary={'attempts':len(results),'accepted':sum(r['accepted'] for r in results),
             'wall_seconds':time.perf_counter()-started,'results':results}
    atomic_json(output/'summary.json',summary)
    print({k:v for k,v in summary.items() if k!='results'},flush=True)


if __name__=='__main__': main()
