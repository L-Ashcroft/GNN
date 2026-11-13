"""Small, auditable SKY130 NMCF simulation pipeline; no RL or upstream imports.

Uses a derived testbench with explicit bias return, temperature and waveform
exports. Raw waveforms are the only source of performance labels.
"""
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, dump, read, rel, digest
from .netlist import attributes, make_graph, feature_expressions, FEATURE_NAMES, evaluate
from scripts.audit_analoggym_replay import action_schema

BASE = ROOT / 'third_party/AnalogGym/RGNN_RL'
DUT = BASE / 'simulations/AMP_NMCF_Pin_3_HSPICE_130.txt'
TOPOLOGY = 'AnalogGym_generated__Leung_NMCF_sky130_tt_v1'
METRICS = [
    ('temperature_coefficient_per_K', '1/K', 'peak-to-peak follower output / mean output / 165 K; -40..125 C'),
    ('supply_power_25C_W', 'W', '-I(VVDDdc)*1.8 at 25 C in DC follower fixture'),
    ('follower_output_offset_25C_V', 'V', 'V(vout6)-0.45 at 25 C; signed output error, not input offset'),
    ('gain_at_0p1Hz_dB', 'dB', '20*log10(abs(V(opout))) at 0.1 Hz, 1 V AC in inverting fixture, 27 C'),
    ('common_mode_transfer_at_0p1Hz_dB', 'dB', '20*log10(abs(V(cm3))) at 0.1 Hz; not generic positive CMRR'),
    ('positive_supply_transfer_at_0p1Hz_dB', 'dB', '20*log10(abs(V(ppsr1))) at 0.1 Hz; 1 V supply excitation'),
    ('negative_supply_transfer_at_0p1Hz_dB', 'dB', '20*log10(abs(V(npsr1))) at 0.1 Hz; 1 V ground excitation'),
    ('unity_gain_frequency_Hz', 'Hz', 'first descending 0 dB crossing; log-frequency interpolation'),
    ('phase_at_unity_deg', 'degree', 'unwrapped phase of V(opout) at that crossing; not automatically phase margin'),
    ('rising_slew_10_90_V_per_s', 'V/s', '0.16 V / first 10-to-90 percent output crossing interval for 0.3-to-0.5 V step'),
    ('falling_slew_90_10_V_per_s', 'V/s', 'positive 0.16 V / first 90-to-10 percent output crossing interval'),
    ('rising_settling_1pct_s', 's', 'time after rising input edge until output stays within 0.002 V of 0.5 through observed high window'),
    ('falling_settling_1pct_s', 's', 'time after falling edge until output stays within 0.002 V of 0.3 through observed low window'),
]
CONTROLS = {
    'acdc': '''.control
set noaskquit
set num_threads=1
dc temp -40 125 1
wrdata dc_wave.dat v(vout6) i(vvdddc)
set temp=27
ac dec 100 0.1 1G
wrdata ac_wave.dat v(opout) v(cm3) v(ppsr1) v(npsr1)
quit
.endc
.end
''',
    'tran': '''.control
set noaskquit
set num_threads=1
set temp=27
tran 10n 401u 0 50n
wrdata tran_wave.dat v(visr) v(vout3)
quit
.endc
.end
''',
}


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def plan_parameters(count, seed):
    names, integer, mapping, bounds = action_schema(BASE / 'AMP_NMCF.py', 'GraphAMPNMCF')
    baseline = {k.lower(): float(v) for k,v in re.findall(r'(\w+)\s*=\s*([-+0-9.eE]+)',
                (BASE / 'simulations/AMP_NMCF_vars.spice').read_text())}
    initial = np.array([baseline[mapping[n].lower()] for n in names])
    low, high = np.array(bounds['action_space_low']), np.array(bounds['action_space_high'])
    assert np.all(initial >= low) and np.all(initial <= high)
    plans = []
    for i in range(count):
        rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
        values = initial.copy() if i == 0 else np.clip(initial * np.exp(rng.uniform(-.1, .1, len(initial))), low, high)
        for n in integer: values[names.index(n)] = np.clip(np.rint(values[names.index(n)]), low[names.index(n)], high[names.index(n)])
        plans.append({mapping[n].lower(): float(v) for n,v in zip(names, values)})
    assert len({digest(p) for p in plans}) == len(plans), 'Duplicate parameter point'
    return plans


def dut_graph():
    devices = []; ports = None
    for line in DUT.read_text().splitlines():
        tokens = line.split()
        if not tokens: continue
        if tokens[0].lower() == '.subckt': ports = tokens[2:]; continue
        if tokens[0].startswith('.'): continue
        model_index = next(j for j,x in enumerate(tokens) if x.startswith('sky130_fd_pr__'))
        pdk_model = tokens[model_index]
        mos = 'fet_01v8' in pdk_model
        model = 'pmos' if 'pfet' in pdk_model else 'nmos' if mos else pdk_model
        nets = tokens[1:model_index]
        assert len(nets) == (4 if mos else 2)
        # SPICE identifiers are case-insensitive; canonicalize expressions as
        # well as parameter keys (M_C0 in the upstream MIM cell is uppercase).
        a = {k.lower():v.strip("'\"").lower() for k,v in attributes(' '.join(tokens[model_index+1:])).items()}
        for key in ('w', 'l'):
            if key in a: a[key] = '(' + a[key] + ')*1e-6'
        devices.append({'name': tokens[0], 'nets': nets, 'model': model, 'pdk_model': pdk_model,
                        'attributes': a, 'ports': ['D','G','S','B'] if mos else ['P','N'], 'device_references': {}})
    assert len(devices) == 26 and ports
    graph = make_graph(devices)
    graph.update(source_netlist=rel(DUT), netlist_sha256=sha(DUT), external_ports=ports,
                 graph_scope='one DUT subcircuit; supply, stimulus, bias and load are explicit testbench conditions',
                 feature_names=FEATURE_NAMES, feature_expressions=feature_expressions(graph))
    # MIM width and length are geometry inputs, not inferred capacitance labels.
    for d, row in zip(devices, graph['feature_expressions']):
        if d['model'].startswith('sky130_fd_pr__cap_'):
            row[0] = d['attributes']['w']; row[1] = d['attributes']['l']
    return graph


def derived_testbench(kind, library):
    filename = 'AMP_NMCF_ACDC.cir' if kind == 'acdc' else 'AMP_NMCF_Tran.cir'
    body = (BASE / 'simulations' / filename).read_text().split('.control')[0]
    result = []; inserted_library = False
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith('.include'):
            if 'mosfet_model' in line:
                if not inserted_library:
                    result.append('.lib "' + library.as_posix() + '" tt'); inserted_library = True
                continue
            if 'AMP_NMCF_Pin_3_HSPICE_130.txt' in line:
                line = '.include "dut.spice"'
            elif 'AMP_NMCF_vars.spice' in line:
                line = '.include "parameters.spice"'
            else: raise ValueError('Unreviewed include: ' + line)
        if re.match(r'^\s*Ib\w*\s', line, re.I):
            fields = line.split()
            assert fields[2].lower() == 'gnda'
            fields[2] = 'vss'  # Upstream bias-return net was otherwise floating.
            line = ' '.join(fields)
        result.append(line)
    assert inserted_library
    return '\n'.join(result) + '\n.temp 27\n.options reltol=1e-5 vntol=1e-8 abstol=1e-14\n' + CONTROLS[kind]


def first_unity(frequency, response):
    db = 20*np.log10(np.abs(response))
    crossings = np.flatnonzero((db[:-1] >= 0) & (db[1:] < 0))
    if not len(crossings): raise ValueError('No descending unity-gain crossing; do not fill a zero label')
    i = crossings[0]; alpha = db[i] / (db[i]-db[i+1])
    f = 10 ** ((1-alpha)*np.log10(frequency[i]) + alpha*np.log10(frequency[i+1]))
    phase = np.unwrap(np.angle(response))*180/np.pi
    return float(f), float((1-alpha)*phase[i]+alpha*phase[i+1])


def edge_metrics(time_values, output, start, end, rising):
    keep = (time_values >= start) & (time_values < end)
    t, y = time_values[keep], output[keep]
    assert len(t) > 10
    levels = (.32, .48) if rising else (.48, .32)
    crossing = []
    for level in levels:
        hits = np.flatnonzero((y[:-1] < level) & (y[1:] >= level) if rising else
                             (y[:-1] > level) & (y[1:] <= level))
        if not len(hits): raise ValueError('Output does not cross required slew threshold')
        j = hits[0]
        crossing.append(float(t[j] + (level-y[j])/(y[j+1]-y[j])*(t[j+1]-t[j])))
    if crossing[1] <= crossing[0]: raise ValueError('Invalid slew interval')
    target = .5 if rising else .3
    outside = np.flatnonzero(np.abs(y-target) > .002)
    if not len(outside) or outside[-1] == len(t)-1:
        raise ValueError('Settling not observable within the measurement window')
    return .16/(crossing[1]-crossing[0]), float(t[outside[-1]+1]-start)


def extract_metrics(run_dir):
    dc = np.loadtxt(run_dir/'dc_wave.dat', ndmin=2)
    ac = np.loadtxt(run_dir/'ac_wave.dat', ndmin=2)
    tran = np.loadtxt(run_dir/'tran_wave.dat', ndmin=2)
    assert dc.shape == (166, 4) and ac.shape[1] == 12 and tran.shape[1] == 4
    for values in (dc, ac, tran):
        assert np.isfinite(values).all()
    np.testing.assert_allclose(dc[:,0], dc[:,2], rtol=0, atol=0)
    for j in (3,6,9): np.testing.assert_array_equal(ac[:,0], ac[:,j])
    np.testing.assert_array_equal(tran[:,0], tran[:,2])
    at25 = np.flatnonzero(dc[:,0] == 25)
    assert len(at25) == 1 and dc[:,1].mean() > 0
    row = dc[at25[0]]
    power = -float(row[3])*1.8
    if power <= 0: raise ValueError('Nonpositive measured supply power')
    responses = [ac[:,j] + 1j*ac[:,j+1] for j in (1,4,7,10)]
    assert all(np.all(np.abs(v) > 0) for v in responses)
    f, phase = first_unity(ac[:,0], responses[0])
    rise_sr, rise_settle = edge_metrics(tran[:,0], tran[:,3], 1e-6, 201e-6, True)
    fall_sr, fall_settle = edge_metrics(tran[:,0], tran[:,3], 201e-6+1e-12, 401e-6, False)
    values = [float(np.ptp(dc[:,1])/dc[:,1].mean()/165), power, float(row[1]-.45),
              *[float(20*np.log10(abs(v[0]))) for v in responses], f, phase,
              rise_sr, fall_sr, rise_settle, fall_settle]
    assert len(values) == len(METRICS) and np.isfinite(values).all()
    return values


def simulate(run_dir, params, testbenches, executable, timeout):
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir/'dut.spice').write_bytes(DUT.read_bytes())
    (run_dir/'parameters.spice').write_text('\n'.join(f'.param {k}={v:.17g}' for k,v in sorted(params.items()))+'\n', encoding='ascii')
    temp = ROOT/'.tmp/ngspice'; temp.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); failure = None; values = None
    for kind, text in testbenches.items(): (run_dir/(kind+'.cir')).write_text(text, encoding='ascii')
    try:
        for kind in testbenches:
            proc = subprocess.run([str(executable), '-n', '-b', '-o', kind+'.log', kind+'.cir'],
                cwd=run_dir, env={**os.environ, 'TEMP':str(temp), 'TMP':str(temp)},
                capture_output=True, text=True, timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW)
            (run_dir/(kind+'_stdout.txt')).write_text(proc.stdout, encoding='utf-8')
            (run_dir/(kind+'_stderr.txt')).write_text(proc.stderr, encoding='utf-8')
            log = (run_dir/(kind+'.log')).read_text(encoding='utf-8', errors='replace')
            if proc.returncode or re.search(r'(?im)^\s*(?:error|fatal)\b|timestep too small|singular matrix', log):
                raise ValueError(kind+' simulation returned an error or convergence warning')
        values = extract_metrics(run_dir)
    except (AssertionError, ValueError, OSError, subprocess.TimeoutExpired) as error:
        failure = type(error).__name__ + ': ' + str(error)
    files = {p.name:sha(p) for p in sorted(run_dir.iterdir()) if p.is_file()}
    result = {'accepted': failure is None, 'failure': failure, 'performance': values,
              'parameters_native':params, 'elapsed_seconds':time.perf_counter()-started,
              'artifacts_sha256':files}
    dump(run_dir/'result.json', result)
    return result


def generate(output, count=1, seed=42, timeout=120):
    from .audit_sources import verify_source
    verify_source('AnalogGym')
    output = output.resolve()
    if not output.is_relative_to(ROOT): raise ValueError('Output must remain within EDA')
    if count < 1 or count > 20: raise ValueError('Pilot budget is 1..20 attempts; review resources before expanding')
    config = read(ROOT/'config/simulation_environment.json')
    executable = (ROOT/config['ngspice_executable']).resolve(strict=True)
    library = (ROOT/config['library']).resolve(strict=True)
    for path in (executable, library):
        if not path.is_relative_to(ROOT): raise ValueError('External simulation path')
    testbenches = {kind:derived_testbench(kind, library) for kind in ('acdc','tran')}
    recipe = {'version':'nmcf-waveform-v1', 'seed':seed, 'source_commit':verify_source('AnalogGym'),
              'source_dut_sha256':sha(DUT), 'ngspice_executable_sha256':sha(executable),
              'model_archive_sha256':read(ROOT/config['pdk_manifest'])['archive_sha256'],
              'testbenches_sha256':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in testbenches.items()},
              'generator_sha256':sha(Path(__file__)), 'metrics':METRICS,
              'sampling':'baseline followed by local multiplicative perturbations exp(U(-0.1,0.1)); integer multiplicity rounded; not full-domain coverage',
              'testbench_changes':['ground floating bias-source returns at vss', 'absolute model library include',
                  'DC sweep -40..125 C, power/offset at 25 C, AC/TRAN explicitly reset to 27 C',
                  'export raw waves; no upstream measurement fallbacks or absolute gain correction',
                  '100 AC points/decade, 10 ns transient output step, 50 ns maximum internal step',
                  'explicit solver tolerances; remove plots and operating-point feature export']}
    recipe_hash = digest(recipe)
    if (output/'recipe.json').exists():
        if digest(read(output/'recipe.json')) != recipe_hash:
            raise ValueError('Generation recipe changed; use a fresh output directory')
    elif output.exists() and any(output.iterdir()): raise ValueError('Nonempty output without recipe')
    output.mkdir(parents=True, exist_ok=True); dump(output/'recipe.json', recipe)
    graph = dut_graph(); dump(output/'graph.json', graph)
    conditions = {'PDK':'SKY130 bundled AnalogGym snapshot', 'corner':'tt', 'supply_voltage_V':1.8,
                  'load_capacitance_F':100e-12, 'common_mode_V':.45, 'ac_temperature_C':27,
                  'dc_temperature_sweep_C':[-40,125,1], 'dc_power_offset_temperature_C':25,
                  'transient_temperature_C':27, 'transient_input_levels_V':[.3,.5],
                  'transient_step_times_s':[1e-6,201e-6+1e-12]}
    dump(output/'conditions.json', conditions)
    dump(output/'performance_mapping.json', {'max_dimensions':16, 'shared_with_existing_datasets':False,
        'mask_policy':'missing targets only; failed measurements reject the candidate',
        'metrics':[{'slot':i,'name':m[0],'unit':m[1],'definition':m[2],
                    'semantic_id':TOPOLOGY+'::'+m[0]} for i,m in enumerate(METRICS)]})
    records = []; statuses = []
    for i, params in enumerate(plan_parameters(count, seed)):
        run_dir = output/'runs'/f'{i:06d}'
        if (run_dir/'result.json').exists():
            result = read(run_dir/'result.json')
            assert result['parameters_native'] == params
            for name, expected in result['artifacts_sha256'].items(): assert sha(run_dir/name) == expected
        elif run_dir.exists(): raise ValueError('Interrupted run preserved; choose fresh output or inspect '+str(run_dir))
        else: result = simulate(run_dir, params, testbenches, executable, timeout)
        statuses.append({'candidate':i, 'run_directory':rel(run_dir), **result})
        print('candidate', i, 'accepted' if result['accepted'] else 'QUARANTINED',
              round(result['elapsed_seconds'],3), 's', result['failure'] or '', flush=True)
        if not result['accepted']: continue
        features = [[None if exp is None else evaluate(exp, params) for exp in row] for row in graph['feature_expressions']]
        values = result['performance']; identifier = 'AnalogGym_generated:'+recipe_hash[:16]+':'+digest(params)[:16]
        records.append({'sample_id':identifier, 'source_dataset':'AnalogGym_generated',
            'topology_id':TOPOLOGY, 'circuit_family':'Amplifier', 'process':'SKY130', 'PDK':conditions['PDK'],
            'temperature':None, 'supply_voltage':1.8, 'testbench_id':recipe_hash,
            'conditions_json':json.dumps(conditions, sort_keys=True), 'graph':rel(output/'graph.json'),
            'parameter_names':list(params), 'parameters':list(params.values()),
            'parameter_units':['um' if '_w_' in k or '_l_' in k else 'A' if k=='current_0_bias' else '1' for k in params],
            'device_features':features, 'performance':values+[None]*(16-len(values)),
            'mask':[True]*len(values)+[False]*(16-len(values)),
            'semantic_ids':[TOPOLOGY+'::'+m[0] for m in METRICS]+[None]*(16-len(values)),
            'source_path':rel(run_dir), 'source_sha256':sha(run_dir/'parameters.spice'),
            'model_manifest':config['pdk_manifest'], 'generation_status':'measured_pilot_not_merged',
            'design_group':digest([TOPOLOGY,params])})
    frame = pd.DataFrame(records)
    if len(frame):
        assert frame.sample_id.is_unique and frame.design_group.is_unique
        frame.to_parquet(output/'samples.parquet', index=False, compression='zstd')
        frame[['sample_id','source_dataset','topology_id','graph','source_path','design_group']].to_parquet(output/'index.parquet', index=False)
    summary = {'attempts':count, 'accepted':len(records), 'quarantined':count-len(records),
               'topologies':1, 'recipe_hash':recipe_hash, 'output':rel(output),
               'total_simulation_seconds':sum(s['elapsed_seconds'] for s in statuses),
               'status':'pilot_success' if records else 'no_accepted_pilot_samples',
               'main_dataset_modified':False, 'normalization_fitted':False,
               'limitations':['one topology, local perturbations only', 'source-specific targets, no approved cross-source mapping',
                              'temperature varies by metric; inspect conditions', 'no training or evaluation split']}
    dump(output/'candidate_status.json', statuses); dump(output/'summary.json', summary)
    return summary
