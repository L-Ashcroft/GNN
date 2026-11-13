"""Independent AnalogGym/SKY130 campaign: real waveforms, bounded design sampling.

No upstream Python is executed. This module deliberately does not import the
historical dataset merger or its label mapping.
"""
import hashlib
import io
import json
import os
import re
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np

from .common import ROOT, read, rel, digest
from .netlist import attributes, evaluate, make_graph, feature_expressions, FEATURE_NAMES, topology_hash
from .analoggym_generation import METRICS as PILOT_METRICS, first_unity, sha

UPSTREAM = ROOT / 'third_party/AnalogGym'
AMP = UPSTREAM / 'AnalogGym/Amplifier'
LDO = UPSTREAM / 'AnalogGym/Low Dropout Regulator'
DEFAULT_OUTPUT = 'data/generated/analoggym_campaign_v1'
AMP_METRICS = PILOT_METRICS[:9] + [
    ('rising_slew_observed_10_90_V_per_s','V/s','80% of observed output transition / interpolated 10-to-90% interval'),
    ('falling_slew_observed_90_10_V_per_s','V/s','Positive 80% of observed output transition / interpolated 90-to-10% interval'),
    ('rising_settling_observed_1pct_s','s','First time after last exit of 1% observed amplitude band about final measured high output; finite window'),
    ('falling_settling_observed_1pct_s','s','First time after last exit of 1% observed amplitude band about final measured low output; finite window'),
    ('high_final_tracking_error_V','V','Time-weighted final output mean over 180..200 us minus ideal 0.5 V'),
    ('low_final_tracking_error_V','V','Time-weighted final output mean over 380..400 us minus ideal 0.3 V'),
]
LDO_METRICS = [
    ('output_light_V', 'V', 'Output at 1 mA load, VDD=1.8 V, 27 C'),
    ('output_heavy_V', 'V', 'Output at 10 mA load, VDD=1.8 V, 27 C'),
    ('input_power_light_W', 'W', '-1.8*I(VDD), including current bias fed from supply, at 1 mA'),
    ('input_power_heavy_W', 'W', '-1.8*I(VDD), including current bias fed from supply, at 10 mA'),
    ('load_regulation_V_per_A', 'V/A', 'Signed (output_10mA-output_1mA)/0.009 A'),
    ('line_regulation_V_per_V', 'V/V', 'Signed endpoint slope of output vs VDD=1.62..1.8 V at 1 mA'),
    ('supply_transfer_10Hz_dB', 'dB', '20log10(abs(Vout/Vsupply_ac)), 1 mA; not positive PSRR'),
    ('supply_transfer_1kHz_dB', 'dB', '20log10(abs(Vout/Vsupply_ac)), 1 mA'),
    ('supply_transfer_100kHz_dB', 'dB', '20log10(abs(Vout/Vsupply_ac)), 1 mA'),
    ('load_step_output_min_V', 'V', 'Minimum output after 1-to-10 mA step, 10..110 us'),
    ('load_step_output_max_V', 'V', 'Maximum output after 10-to-1 mA step, 110..210 us'),
    ('heavy_window_final_output_V', 'V', 'Time-weighted mean output over 100..109 us'),
    ('light_window_final_output_V', 'V', 'Time-weighted mean output over 200..209 us'),
]


def inside(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Path must stay inside D:/EDA: ' + str(path))
    return path


def atomic_json(path, obj):
    path = inside(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def param_definitions(path):
    lines = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.lower().startswith('.param'): lines.append(line[6:])
        elif line.startswith('+'): lines.append(line[1:])
    return {k.lower(): v.strip("'\"{}").lower() for k, v in attributes(' '.join(lines)).items()}


def resolve_parameters(definitions, overrides=None):
    values = dict(overrides or {}); pending = dict(definitions)
    while pending:
        progressed = False
        for key, expression in list(pending.items()):
            try: value = values[key] if key in values else evaluate(expression, values)
            except KeyError: continue
            if not np.isfinite(value) or value <= 0: raise ValueError('Invalid positive parameter: '+key)
            values[key] = value; del pending[key]; progressed = True
        if not progressed: raise ValueError('Cyclic or unresolved parameters: '+str(pending))
    return dict(sorted(values.items()))


def spice_graph(path):
    devices = []; formal = None; subckt = None
    for raw in path.read_text().splitlines():
        line = raw.strip().lower()
        if not line or line.startswith('*'): continue
        fields = line.split()
        if fields[0] == '.subckt':
            if formal is not None: raise ValueError('Multiple subcircuits require explicit flattening')
            subckt, formal = fields[1], fields[2:]; continue
        if fields[0] == '.ends': continue
        if fields[0].startswith('x'):
            model_i = next(i for i, t in enumerate(fields) if t.startswith('sky130_fd_pr__'))
            pdk = fields[model_i]; nets = fields[1:model_i]
            attrs = {k: v.strip("'\"{}") for k,v in attributes(' '.join(fields[model_i+1:])).items()}
            if 'nfet_' in pdk or 'pfet_' in pdk:
                model = 'nmos' if 'nfet_' in pdk else 'pmos'; ports = ['D','G','S','B']
            elif '__cap_' in pdk: model = pdk; ports = ['P','N']
            elif '__res_' in pdk: model = pdk; ports = ['P','N','B']
            else: raise ValueError('Unreviewed PDK cell: '+pdk)
            for key in ('w','l'):
                if key in attrs: attrs[key] = '('+attrs[key]+')*1e-6'
        elif fields[0][0] in 'rciv':
            model, key = {'r':('resistor','r'), 'c':('capacitor','c'),
                          'i':('isource','dc'), 'v':('vsource','dc')}[fields[0][0]]
            nets, ports, pdk = fields[1:3], ['P','N'], None
            attrs = {key:' '.join(fields[3:]).strip("'\"{}")}
        else: raise ValueError('Unparsed DUT line: '+raw)
        if len(nets) != len(ports): raise ValueError('Port arity: '+raw)
        devices.append({'name':fields[0], 'model':model, 'pdk_model':pdk,
                        'nets':nets, 'ports':ports, 'attributes':attrs, 'device_references':{}})
    if not devices or not formal: raise ValueError('Empty netlist')
    if len({d['name'] for d in devices}) != len(devices): raise ValueError('Duplicate devices')
    g = make_graph(devices)
    g.update(source_netlist=rel(path), netlist_sha256=sha(path), external_ports=formal,
             subcircuit_name=subckt, feature_names=FEATURE_NAMES,
             feature_expressions=feature_expressions(g), graph_scope='DUT; external fixtures in conditions.json')
    for d, row in zip(devices, g['feature_expressions']):
        if (d['pdk_model'] or '').startswith(('sky130_fd_pr__cap_', 'sky130_fd_pr__res_')):
            row[0] = d['attributes'].get('w'); row[1] = d['attributes'].get('l')
    g['topology_hash'] = topology_hash(g)
    return g


def catalog():
    entries = []; excluded = []
    candidates = [(p.stem, 'Amplifier', p, AMP/'design_variables'/p.name)
                  for p in sorted((AMP/'spice_netlist').iterdir())]
    candidates += [(name, 'LDO', LDO/'design_variables'/(name+'.txt'), LDO/'spice_netlist'/(name+'_vars.spice'))
                   for name in ('ldo_simple','ldo_1','ldo_2','ldo_folded_cascode')]
    candidates.append(('Basic_LDO','LDO',LDO/'spice_netlist/LDO_netlist.txt',LDO/'design_variables/LDO_variables.txt'))
    for name, family, dut, params in candidates:
        try:
            graph = spice_graph(dut); defs = param_definitions(params); baseline = resolve_parameters(defs)
            for row in graph['feature_expressions']:
                for expr in row:
                    if expr is not None: evaluate(expr, baseline)
            entries.append({'name':name, 'family':family, 'dut':rel(dut), 'parameter_source':rel(params),
                            'dut_sha256':sha(dut), 'parameter_source_sha256':sha(params),
                            'definitions':defs, 'baseline':baseline, 'graph':graph})
        except (ValueError, KeyError, StopIteration) as err:
            excluded.append({'name':name, 'source':rel(dut), 'reason':type(err).__name__+': '+str(err)})
    # These are separate evidence entries, not invented runnable topology counts.
    excluded += [
        {'name':'Voltage Reference', 'source':rel(UPSTREAM/'AnalogGym/Voltage Reference'),
         'reason':'Markdown contains four reference circuit netlists and testbenches, but explicitly omits confidential PDK imports; models are not the bundled SKY130 models. Requires a documented process port and requalification.'},
        {'name':'Phase-Locked Loop', 'source':rel(UPSTREAM/'AnalogGym/Phase-Locked Loop'),
         'reason':'Cadence/OCEAN flow and proprietary model dependencies; no validated ngspice/SKY130 deck'},
        {'name':'Charge Pump', 'source':rel(UPSTREAM/'AnalogGym/Charge Pump'),
         'reason':'p11ll_ckt/n11ll_ckt/pod33ll_ckt model dependencies are not bundled SKY130 models'},
        {'name':'Sensing Front End', 'source':rel(UPSTREAM/'AnalogGym/Sensing Front End'),
         'reason':'HSPICE/Spectre decks reference TSMC180HV/other proprietary models; no verified SKY130 port'},
    ]
    return entries, excluded


def parameter_plan(entry, candidate, seed=42):
    """Stratified local log boxes; aliases/mirror ratios remain tied exactly.

    Twenty percent narrow (x0.9..1.111), sixty percent x0.8..1.25,
    twenty percent x2/3..1.5. These are documented engineering ranges, not
    claimed official optimization bounds or a uniform whole-space sample.
    """
    defs = entry['definitions']; baseline = entry['baseline']; independent = {}
    if entry.get('sampling_centers'):
        baseline=entry['sampling_centers'][candidate % len(entry['sampling_centers'])]
    topo_seed = int(hashlib.sha256(entry['name'].encode()).hexdigest()[:8],16)
    rng = np.random.default_rng(np.random.SeedSequence([seed, topo_seed, candidate]))
    # Interleave ranges so even a short pipeline smoke run exercises wide boxes.
    band = (0,8,2,1,3,4,5,6,7,9)[candidate%10]
    span = (np.log(1/0.9) if band < 2 else np.log(1.25) if band < 8 else np.log(1.5))
    for key, expr in sorted(defs.items()):
        try: evaluate(expr, {})
        except KeyError: continue  # preserve dependent aliases, rather than breaking matched devices
        value = baseline[key]
        if key in {'cload','vcm','m_cl'}: independent[key] = value; continue
        if candidate:
            extent = min(span, np.log(1.1)) if key.startswith('vb') else span
            if key.startswith('capacitor'): extent = np.log(1.35 if entry.get('sampling_centers') else 2)
            value *= np.exp(rng.uniform(-extent, extent))
            if re.search(r'(^m_|_m_)',key): value = max(1,round(value))
        # Explicit adaptation also applies to the baseline: the upstream Tan
        # deck has 0.253 um NFET widths for which this bundle has no model bin.
        if re.search(r'(^w_|_w_)',key): value = min(100.,max(0.42,value))
        if re.search(r'(^l_|_l_)',key): value = min(30.,max(0.15,value))
        independent[key] = float(value)
    return resolve_parameters(defs, independent)


def conditions(entry):
    common = {'PDK':'SKY130 AnalogGym pinned bundle', 'corner':'tt', 'supply_voltage_V':1.8,
              'temperature_C':27, 'normalization_fitted':False}
    if entry['family'] == 'Amplifier':
        return {**common, 'load_capacitance_F':15e-9 if entry['name']=='Yan_AZ_Pin_3' else 500e-12, 'common_mode_V':.45,
                'dc_temperature_sweep_C':[-40,125,1], 'power_offset_temperature_C':25,
                'ac_frequency_Hz':[.1,1e9], 'ac_points_per_decade':100,
                'transient_levels_V':[.3,.5], 'transient_edges_s':[1e-6,201e-6+1e-9],
                'transient_input_transition_s':1e-9,
                'transient_stop_s':401e-6, 'transient_maxstep_s':25e-9}
    return {**common, 'target_output_V':1.2, 'load_current_A':[.001,.01],
            'load_capacitance_F':100e-9, 'supply_sweep_V':[1.62,1.8,.01],
            'ac_frequency_Hz':[1,1e7], 'ac_points_per_decade':100,
            'load_step_edges_s':[10e-6,110e-6], 'load_step_transition_s':100e-9,
            'transient_stop_s':210e-6, 'transient_maxstep_s':25e-9,
            'bias_voltage_sources':'explicit input parameters; supply-power label excludes separate voltage-bias sources',
            'reference_voltage_V':.3 if entry['name'] in {'Basic_LDO','ldo_1'} else 1.2}


def testbenches(entry, library):
    prefix = ('AnalogGym standalone derived measurement fixture\n'
              f'.lib "{library.as_posix()}" tt\n.include "dut.spice"\n.include "parameters.spice"\n'
              '.param mc_mm_switch=0 mc_pr_switch=0\n.temp 27\n'
              '.options reltol=1e-7 vntol=1e-10 abstol=1e-14\n')
    ctrl = '.control\nset noaskquit\nset num_threads=1\nset numdgt=15\n'
    end = 'quit\n.endc\n.end\n'
    sub = entry['graph']['subcircuit_name']
    if entry['family'] == 'Amplifier':
        acdc = f'''V1 vdd 0 1.8
Vindc opin 0 .45
Vin signal_in 0 dc .45 ac 1
Lfb opout opout_dc 1T
Cin opout_dc signal_in 1T
Xop1 0 vdd opout_dc opin opout {sub}
Cload1 opout 0 500p
Xop2 0 vdd cm2 cm1 cm3 {sub}
Cload2 cm3 0 500p
Vcmdc cm0 0 .45
Vcmac1 cm1 cm0 0 ac 1
Vcmac2 cm2 cm3 0 ac 1
Vgndpsrr gndpsrr 0 0 ac 1
Vvddpsrr vddpsrr 0 1.8 ac 1
Xop3 0 vddpsrr ppsr1 opin ppsr1 {sub}
Cload3 ppsr1 0 500p
Xop4 gndpsrr vdd npsr1 opin npsr1 {sub}
Cload4 npsr1 0 500p
Vvdddc vdddc 0 1.8
Xop5 0 vdddc vout6 opin vout6 {sub}
Cload5 vout6 0 500p
'''
        tran = f'''V1 vdd 0 1.8
Vvisr visr 0 pulse(.3 .5 1u 1n 1n 200u 1)
Xop6 0 vdd vout3 visr vout3 {sub}
Cload6 vout3 0 500p
'''
        cload=conditions(entry)['load_capacitance_F']
        acdc=acdc.replace(' 500p',f' {cload:.17g}'); tran=tran.replace(' 500p',f' {cload:.17g}')
        return {'acdc':prefix+acdc+ctrl+'dc temp -40 125 1\nwrdata dc_wave.dat v(vout6) i(vvdddc)\nset temp=27\nac dec 100 .1 1G\nwrdata ac_wave.dat v(opout) v(cm3) v(ppsr1) v(npsr1)\n'+end,
                'tran':prefix+tran+ctrl+'tran 10n 401u 0 25n\nwrdata tran_wave.dat v(visr) v(vout3)\n'+end}
    name = entry['name']; ref = conditions(entry)['reference_voltage_V']
    if name == 'ldo_simple':
        dut = f'Xdut vdd out ref vb 0 out {sub}\nVb vb 0 {{vb}}\n'
    elif name == 'ldo_folded_cascode':
        dut = f'Xdut vdd out vb2 ref vb1 0 out {sub}\nVb1 vb1 0 {{vb1}}\nVb2 vb2 0 {{vb2}}\n'
    else:
        dut = f'Xdut 0 vdd ref out fb fb ib {sub}\n'
        dut += 'Ib ib 0 {current_0_bias}\n' if name == 'Basic_LDO' else 'Ib vdd ib {current_0_bias}\n'
    cload=conditions(entry)['load_capacitance_F']
    base = prefix+f'VDD vdd 0 1.8 ac 1\nVref ref 0 {ref}\n'+dut+f'Cload out 0 {cload:.17g}\n'
    return {'acdc':base+'Iload out 0 .001\n'+ctrl+'''
dc Iload .001 .01 .009
wrdata load_wave.dat v(out) i(vdd)
alter Iload dc=.001
dc VDD 1.62 1.8 .01
wrdata line_wave.dat v(out)
alter VDD dc=1.8
ac dec 100 1 10Meg
wrdata ac_wave.dat v(out)
'''+end,
            'tran':base+'Iload out 0 pulse(.001 .01 10u 100n 100n 100u 1)\n'+ctrl+
            'tran 10n 210u 0 25n\nwrdata tran_wave.dat v(out)\n'+end}


def observed_step(t, y, start, stop, initial, final):
    delta=final-initial
    if abs(delta)<.05: raise ValueError('Insufficient observed output transition')
    keep=(t>=start)&(t<stop); t=t[keep]; y=y[keep]
    normalized=(y-initial)/delta
    times=[]
    for level in (.1,.9):
        hits=np.flatnonzero((normalized[:-1]<level)&(normalized[1:]>=level))
        if not len(hits): raise ValueError('Missing observed slew crossing')
        j=hits[0]; times.append(float(t[j]+(level-normalized[j])/(normalized[j+1]-normalized[j])*(t[j+1]-t[j])))
    if times[1]<=times[0]: raise ValueError('Invalid observed slew interval')
    outside=np.flatnonzero(np.abs(y-final)>.01*abs(delta))
    if not len(outside) or outside[-1]>=len(t)-1: raise ValueError('Observed response never settles')
    settling=float(t[outside[-1]+1]-start)
    if settling>.8*(stop-start): raise ValueError('Insufficient settled observation time')
    return .8*abs(delta)/(times[1]-times[0]),settling


def extract(entry, wave):
    def load(name, columns):
        a = np.loadtxt(io.BytesIO(wave[name]), ndmin=2)
        if a.shape[1] != columns or not np.isfinite(a).all(): raise ValueError('Invalid waveform '+name)
        if np.any(np.diff(a[:,0]) <= 0): raise ValueError('Non-monotonic sweep '+name)
        return a
    ac = load('ac_wave.dat', 12 if entry['family']=='Amplifier' else 3)
    if entry['family'] == 'Amplifier':
        dc = load('dc_wave.dat',4); tr = load('tran_wave.dat',4)
        np.testing.assert_allclose(dc[:,0],np.arange(-40,126),rtol=0,atol=1e-8)
        for a, cols in ((dc,(2,)),(tr,(2,)),(ac,(3,6,9))):
            for col in cols: np.testing.assert_array_equal(a[:,0],a[:,col])
        np.testing.assert_allclose(ac[:,0],np.logspace(-1,9,1001),rtol=1e-9)
        if tr[-1,0] < 400.9e-6: raise ValueError('Truncated transient')
        power = -float(dc[65,3])*1.8
        if power <= 0 or np.max(np.abs(dc[:,1]-.45)) > .09:
            raise ValueError('DC follower outside 20% operating window or nonpositive power')
        responses = [ac[:,j]+1j*ac[:,j+1] for j in (1,4,7,10)]
        ugf, phase = first_unity(ac[:,0],responses[0])
        def final_mean(a,b):
            y=tr[(tr[:,0]>=a)&(tr[:,0]<=b)]
            return float(np.trapezoid(y[:,3],y[:,0])/(y[-1,0]-y[0,0]))
        initial=final_mean(.1e-6,.9e-6); high=final_mean(180e-6,200e-6); low=final_mean(380e-6,400e-6)
        if abs(high-.5)>.1 or abs(low-.3)>.06: raise ValueError('Transient tracking error exceeds 20%')
        rise, rise_settle = observed_step(tr[:,0],tr[:,3],1e-6,201e-6+1e-9,initial,high)
        fall, fall_settle = observed_step(tr[:,0],tr[:,3],201e-6+1e-9,401e-6,high,low)
        values = [float(np.ptp(dc[:,1])/np.mean(dc[:,1])/165),power,float(dc[65,1]-.45),
                  *[float(20*np.log10(abs(v[0]))) for v in responses],ugf,phase,rise,fall,rise_settle,fall_settle,high-.5,low-.3]
    else:
        dc=load('load_wave.dat',4); line=load('line_wave.dat',2); tr=load('tran_wave.dat',2)
        np.testing.assert_allclose(dc[:,0],[.001,.01],rtol=1e-8)
        np.testing.assert_allclose(line[:,0],np.linspace(1.62,1.8,19),rtol=1e-8)
        np.testing.assert_allclose(ac[:,0],np.logspace(0,7,701),rtol=1e-8)
        if tr[-1,0] < 209.9e-6: raise ValueError('Truncated transient')
        if np.max(np.abs(dc[:,1]-1.2)) > .24: raise ValueError('LDO outside 20% regulation window')
        powers = -1.8*dc[:,3]
        if np.any(powers <= 0): raise ValueError('Nonpositive supply power')
        resp = ac[:,1]+1j*ac[:,2]
        transfers = [float(20*np.log10(abs(resp[np.argmin(abs(ac[:,0]-f))]))) for f in (10,1e3,1e5)]
        def window(a,b):
            keep = (tr[:,0]>=a)&(tr[:,0]<=b)
            if keep.sum()<10: raise ValueError('Insufficient transient window')
            return tr[keep]
        high=window(10e-6,110e-6); low=window(110.1e-6,210e-6)
        def average(a,b):
            y=window(a,b)
            if np.ptp(y[:,1])>.012: raise ValueError('LDO final-window ripple exceeds 1% of target; oscillatory/unsettled')
            return float(np.trapezoid(y[:,1],y[:,0])/(y[-1,0]-y[0,0]))
        values = [*map(float,dc[:,1]),*map(float,powers),float((dc[1,1]-dc[0,1])/.009),
                  float((line[-1,1]-line[0,1])/.18),*transfers,float(high[:,1].min()),float(low[:,1].max()),
                  average(100e-6,109e-6),average(200e-6,209e-6)]
        if max(abs(values[-1]-1.2),abs(values[-2]-1.2))>.24:
            raise ValueError('Transient final windows outside 20% regulation window')
    expected=len(AMP_METRICS if entry['family']=='Amplifier' else LDO_METRICS)
    if len(values)!=expected or not np.isfinite(values).all(): raise ValueError('Nonfinite or incomplete targets')
    return values


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
    os.replace(temporary,archive)
    for p in files:
        if inside(p).parent != run or p.is_symlink(): raise ValueError('Unsafe scratch path')
        p.unlink()
    run.rmdir()
    return result
