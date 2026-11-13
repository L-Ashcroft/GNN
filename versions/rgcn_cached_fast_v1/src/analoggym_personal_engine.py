"""Versioned personal sampling extension; simulator/extraction remain frozen."""
import hashlib
import json
import os
import re
import subprocess
import time
import zipfile
from pathlib import Path
import numpy as np
from .common import ROOT,read,digest,rel
from .analoggym_campaign import inside,sha,testbenches,extract,resolve_parameters,evaluate
from .windows_campaign_io import atomic_json,replace_with_retry
from .analoggym_campaign import parameter_plan as frozen_parameter_plan


def parameter_plan(entry,candidate,seed=42):
    policy=entry.get('_sampling_policy')
    if not policy or candidate<policy['start_candidate'] or entry['name']!='ldo_folded_cascode':
        return frozen_parameter_plan(entry,candidate,seed)
    centre=policy['centers'][(candidate-policy['start_candidate'])%len(policy['centers'])]
    independent={}
    rng=np.random.default_rng(np.random.SeedSequence([seed,712913,candidate,policy['seed_offset']]))
    mode=policy['mode'];span=policy['span']
    width_factor=np.exp(rng.uniform(-np.log1p(span),np.log1p(span)))
    length_factor=np.exp(rng.uniform(-np.log1p(span),np.log1p(span)))
    for key,expression in sorted(entry['definitions'].items()):
        try:evaluate(expression,{})
        except KeyError:continue  # Source aliases/matched devices remain exactly tied.
        value=centre[key]
        if key.startswith('w_') or key.startswith('l_'):
            if mode=='correlated' and key not in {'w_m10','l_m10'}:
                factor=width_factor if key.startswith('w_') else length_factor
            else:factor=np.exp(rng.uniform(-np.log1p(span),np.log1p(span)))
            value*=factor
        elif key.startswith('vb'):
            value*=np.exp(rng.uniform(-np.log1p(policy['bias_span']),np.log1p(policy['bias_span'])))
        elif key in {'m_m10','m_cfb'} and policy.get('vary_discrete',False):
            value=max(1,round(value*np.exp(rng.uniform(-np.log1p(span),np.log1p(span)))))
        # All other multipliers, feedback ratios and load parameters remain fixed.
        if key.startswith('w_'):value=min(100.,max(.42,value))
        if key.startswith('l_'):value=min(30.,max(.15,value))
        independent[key]=float(value)
    return resolve_parameters(entry['definitions'],independent)


def sampling_trace(entry,candidate):
    policy=entry.get('_sampling_policy')
    active=policy and candidate>=policy['start_candidate'] and entry['name']=='ldo_folded_cascode'
    return {'sampling_policy_version':policy['version'] if active else 'personal_sampler_v1',
            'sampling_policy_sha256':digest(policy) if active else None}

def simulate(entry, candidate, output, environment, timeout=120, precision=False):
    """Fresh directory per attempt; archive every raw file before removing scratch.

    No recursive deletion: only the files enumerated and CRC/SHA verified inside
    this one workspace-owned run directory are unlinked.
    """
    output=inside(output); params=parameter_plan(entry,candidate)
    run=inside(output/'work'/entry['name']/f'{candidate:08d}')
    archive=inside(output/'raw'/entry['name']/f'{candidate//1000:04d}'/f'{candidate:08d}.zip')
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            if z.testzip(): raise ValueError('Corrupt existing archive')
            result=json.loads(z.read('result.json'))
        if result['parameters_native'] != params: raise ValueError('Resume parameters changed')
        return result
    if run.exists():
        # Retain interrupted evidence; never reuse old waveform files.
        recovered=run.with_name(run.name+'_interrupted_'+str(time.time_ns()))
        run.replace(inside(recovered))
    run.mkdir(parents=True); archive.parent.mkdir(parents=True,exist_ok=True)
    (run/'dut.spice').write_bytes((ROOT/entry['dut']).read_bytes())
    (run/'parameters.spice').write_text('\n'.join(f'.param {k}={v:.17g}' for k,v in params.items())+'\n',encoding='ascii')
    benches=testbenches(entry, ROOT/environment['library'])
    if precision:
        benches={k:v.replace('reltol=1e-7','reltol=1e-8').replace('0 25n','0 12.5n') for k,v in benches.items()}
    started=time.perf_counter(); failure=None; values=None; runtimes={}
    temp=inside(ROOT/'.tmp/ngspice_campaign'); temp.mkdir(parents=True,exist_ok=True)
    try:
        for kind,deck in benches.items():
            (run/(kind+'.cir')).write_text(deck,encoding='ascii')
            t=time.perf_counter()
            proc=subprocess.run([str(ROOT/environment['ngspice_executable']),'-n','-b','-o',kind+'.log',kind+'.cir'],
                cwd=run,env={**os.environ,'TEMP':str(temp),'TMP':str(temp),'OMP_NUM_THREADS':'1'},
                capture_output=True,text=True,timeout=timeout,creationflags=subprocess.CREATE_NO_WINDOW)
            runtimes[kind]=time.perf_counter()-t
            (run/(kind+'_stdout.txt')).write_text(proc.stdout,encoding='utf-8')
            (run/(kind+'_stderr.txt')).write_text(proc.stderr,encoding='utf-8')
            log=(run/(kind+'.log')).read_text(errors='replace')
            if proc.returncode or re.search(r'(?im)^\s*(error|fatal)\b|timestep too small|singular matrix|out of range',log):
                raise ValueError(kind+' simulator error/convergence/model-range diagnostic')
        values=extract(entry,{p.name:p.read_bytes() for p in run.glob('*_wave.dat')})
    except (ValueError,AssertionError,OSError,subprocess.TimeoutExpired) as err:
        failure=type(err).__name__+': '+str(err)
    result={'topology':entry['name'],'candidate':candidate,'parameters_native':params,
            **sampling_trace(entry,candidate),
            'accepted':failure is None,'failure':failure,'performance':values,
            'elapsed_seconds':time.perf_counter()-started,'analysis_seconds':runtimes,
            'precision_check':precision,'archive':rel(archive),
            'artifacts_sha256':{p.name:sha(p) for p in sorted(run.iterdir()) if p.is_file()}}
    atomic_json(run/'result.json',result)
    temporary=archive.with_suffix('.zip.tmp')
    files=list(run.iterdir())
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in files: z.write(p,p.name)
    with zipfile.ZipFile(temporary) as z:
        if z.testzip(): raise ValueError('Archive verification failed')
        for p in files:
            if hashlib.sha256(z.read(p.name)).hexdigest()!=sha(p): raise ValueError('Archive SHA mismatch')
    replace_with_retry(temporary,archive)
    for p in files:
        if inside(p).parent != run or p.is_symlink(): raise ValueError('Unsafe scratch path')
        p.unlink()
    run.rmdir()
    return result

