"""Native Windows AC provenance replay with a guaranteed original-run recovery.

Only four real archived parameter rows; no invented dataset labels.  The official
testbench/PDK are preserved and copied locally.  Simulator output is audit-only.
"""
from pathlib import Path
from datetime import datetime,timezone
import ast,csv,gzip,hashlib,json,os,re,subprocess,time,zipfile
import psutil

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'data/audit/loadbench_physical_v1'
OLD=ROOT/'artifacts/experiment2_loto_v1/LB_M04__hold_SimpleOpAmp4'
CONTROL=ROOT/'monitoring/experiment2_loto_v1/LB_M04__hold_SimpleOpAmp4'
MARKER='loadbench_physical_v1_bounded_audit_window'
EXE=ROOT/'tools/Spice64/bin/ngspice_con.exe'
VARIANT='native_wrapper_units_v2'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=lambda:datetime.now(timezone.utc).isoformat()

def dump(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def identity(d,pid_key='pid',time_key='create_time'):
    try:
        p=psutil.Process(d[pid_key]);return p if abs(p.create_time()-d[time_key])<.01 else None
    except (KeyError,psutil.NoSuchProcess):return None

def pause():
    s=read(OLD/'status.json');supervisor=read(CONTROL/'status.json')
    if s['state']=='complete':return {'paused_by_audit':False,'already_complete':True,'recipe_sha256':sha(OLD/'recipe.json')}
    assert identity(s) is not None,'Original trainer not independently verified'
    assert identity(supervisor,'supervisor_pid','supervisor_create_time') is not None,'Supervisor missing'
    assert not (CONTROL/'STOP').exists() and not (OLD/'STOP').exists(),'Existing stop request retained'
    record={'requested_at':now(),'paused_by_audit':True,'original_worker':s,'original_supervisor':supervisor,
            'recipe_sha256':sha(OLD/'recipe.json'),'best_before':sha(OLD/'best.pt')}
    if (BASE/'pause_record.json').exists():
        with (BASE/'pause_history.jsonl').open('a',encoding='utf-8') as h:h.write(json.dumps(read(BASE/'pause_record.json'))+'\n')
    dump(BASE/'pause_record.json',record)
    (CONTROL/'STOP').write_text(MARKER,encoding='utf-8')
    deadline=time.monotonic()+150
    while time.monotonic()<deadline:
        state=read(OLD/'status.json')
        if identity(s) is None and state['state'] in {'paused','complete'}:
            # Wait for the supervisor to release its lock too.
            if identity(supervisor,'supervisor_pid','supervisor_create_time') is None:break
        time.sleep(2)
    else:raise TimeoutError('Checkpoint pause was not completed; no competing audit simulation launched')
    record.update(checkpointed_at=now(),saved_status=read(OLD/'status.json'),last_checkpoint_sha256=sha(OLD/'last.pt'))
    assert sha(OLD/'recipe.json')==record['recipe_sha256'];dump(BASE/'pause_record.json',record)
    return record

def resume(record):
    if not record.get('paused_by_audit'):return
    assert sha(OLD/'recipe.json')==record['recipe_sha256'],'Original recipe changed'
    # STOP files must belong to this audit (or to its original supervisor).
    for p in [CONTROL/'STOP',OLD/'STOP']:
        if p.exists():
            message=p.read_text(encoding='utf-8')
            assert message in {MARKER,'User stop or pinned source change; preserve fold checkpoint'},'External stop request retained'
            p.unlink()
    current=read(OLD/'status.json')
    if current['state']=='complete':record['resume_state']='original_completed';dump(BASE/'pause_record.json',record);return
    assert identity(current) is None,'A trainer is already running; no second trainer launched'
    assert sha(OLD/'last.pt')==record['last_checkpoint_sha256'],'Checkpoint changed during audit'
    command=[str(ROOT/'.venv/Scripts/python.exe'),'-X','utf8','-B','-u',str(ROOT/'worktrees/loto_training_v1/supervisor.py')]
    with (CONTROL/'physical_audit_resume_stdout.log').open('ab') as out,(CONTROL/'physical_audit_resume_stderr.log').open('ab') as err:
        p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    record.update(resume_requested_at=now(),supervisor_launcher_pid=p.pid,resume_state='supervisor_started_same_recipe')
    dump(BASE/'pause_record.json',record)

def source_row(folder,file,row_number=None,normal=False):
    path=folder/file;opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8-sig',newline='') as stream:
        for i,row in enumerate(csv.DictReader(stream),1):
            if row_number is not None and i!=row_number:continue
            if normal and not (20<float(row['phasemargin'])<100 and 10<float(row['dc_gain'])<100 and 1<float(row['ft'])<1e9):continue
            return path,i,row
    raise ValueError('Requested real row unavailable')

def unpack_models():
    target=BASE/'models';target.mkdir(parents=True,exist_ok=True)
    archive=ROOT/'data/raw/LOADBench/reduced_sky130_combined.zip'
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            p=(target/item.filename).resolve();assert target.resolve() in p.parents or p==target.resolve()
        z.extractall(target)
    return target/'reduced_sky130_combined/sky130.lib.spice'

def analyze(folder,raw,source_folder):
    import numpy as np
    tree=ast.parse((source_folder/'standalone_tb_extract.py').read_text(encoding='utf-8'))
    names={'parse_raw','correct_phase_series','calc_ac'}
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    scope={'np':np,'re':re,'INFINITY':9999999999999,'MINUS_INFINITY':-9999999999999}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'frozen_official_AC_functions','exec'),scope)
    wave=scope['parse_raw'](folder/'ac.raw');f=wave['frequency'].real;gain=wave['vdb(xtb_ac_vtran.out)'].real;phase=wave['ph(xtb_ac_vtran.out)'].real
    archived={k:float(raw[k]) for k in ['dc_gain','ft','phasemargin']}
    try:
        official=scope['calc_ac'](None,None,wave)
        extracted={'dc_gain':float(official[0]),'ft':float(official[1]),'phasemargin':float(official[2])}
        if not all(np.isfinite(v) for v in extracted.values()):extracted={'error':'nonfinite official output'}
    except Exception as e:extracted={'error':str(e)}
    down=np.flatnonzero((gain[:-1]>=0)&(gain[1:]<0));up=np.flatnonzero((gain[:-1]<0)&(gain[1:]>=0))
    unwrapped=np.unwrap(phase);unwrapped-=unwrapped[0];crossings=[]
    for index in down:
        a=float(gain[index]/(gain[index]-gain[index+1]));hz=float(np.exp(np.log(f[index])+a*(np.log(f[index+1])-np.log(f[index]))))
        pm=float(180+np.degrees(unwrapped[index]+a*(unwrapped[index+1]-unwrapped[index])))
        crossings.append({'frequency_Hz':hz,'phase_margin_degree':pm,'bracket_Hz':[float(f[index]),float(f[index+1])]})
    result={'archived':archived,'official_function_on_replayed_waveform':extracted,'frequency_min_Hz':float(f.min()),'frequency_max_Hz':float(f.max()),
            'points':len(f),'gain_min_dB':float(gain.min()),'gain_max_dB':float(gain.max()),'descending_unity_crossings':crossings,
            'ascending_unity_crossing_count':int(len(up)),'gain_at_1Hz_difference_dB':float(gain[0]-archived['dc_gain']),
            'no_measured_unity_crossing':not (len(down) or len(up)),
            'interpretation':'Independent crossing calculation never extrapolates and does not overwrite official labels.'}
    dump(folder/'waveform_analysis.json',result);return result

def run_case(label,source,model):
    folder=BASE/'replays_native_units'/label;folder.mkdir(parents=True,exist_ok=True)
    path,rownum,row=source;original=path.parent;tb=(original/'CombinedTestBench.cir').read_text(encoding='utf-8').split('.control',1)[0]
    tb=re.sub(r'(?im)^\s*\.lib\s+\S+\s+tt\s*$',f'.lib "{model.as_posix()}" tt',tb)
    parameters=(original/'params.cir').read_text(encoding='utf-8')+'\n* Exact archived CSV overrides in SKY130 wrapper-native micrometres.\n'
    for name,value in row.items():
        if name.lower().startswith('param_'):
            # SKY130 wrapper subcircuits use micron W/L with global scale1e-6.
            # Graph storage remains SI. The first replay's SI parameters under
            # default micron scale failed model bin selection; retain its logs.
            v=float(value)
            parameters+=f'.param {name}={v:.17g}\n'
    (folder/'params.cir').write_text(parameters,encoding='ascii')
    control='''
.control
set noaskquit
set num_threads=1
set filetype=binary
op
let offset=-(v(xtb_tran.inn)-v(xtb_tran.inp))
alter v.xtb_ac_vtran.Voffset DC = $&offset
op
ac dec 20 1 1G
write ac.raw vdb(xtb_ac_vtran.out) ph(xtb_ac_vtran.out)
quit
.endc
.end
'''
    (folder/'audit.cir').write_text(tb+'\n.option scale=1e-6\n'+control,encoding='ascii',errors='replace')
    (folder/'.spiceinit').write_bytes((original/'.spiceinit').read_bytes())
    record={'label':label,'source_path':path.relative_to(ROOT).as_posix(),'source_row':rownum,'source_sha256':sha(path),
            'original_testbench_sha256':sha(original/'CombinedTestBench.cir'),'audit_netlist_sha256':sha(folder/'audit.cir'),
            'params_sha256':sha(folder/'params.cir'),'simulator_sha256':sha(EXE),
            'changes':'Paths localized; AC-only OP/offset/1Hz..1GHz sweep; original DUT/fixture/models and physical sizing retained. CSV micron widths fed to wrapper with global scale1e-6; graph-SI parameters are not passed directly. DC range/transient omitted.',
            'replay_variant':VARIANT,'qualification':'Native-unit replay requires normal CSV controls to match; not a claim of bit-identical historical simulator/environment.',
            'scope':'Diagnostic only; original labels not replaced, no dataset sample added.'}
    dump(folder/'provenance.json',record);tick=time.monotonic()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (folder/'stdout.log').open('wb') as out,(folder/'stderr.log').open('wb') as err:
        p=subprocess.Popen([str(EXE),'-b','audit.cir'],cwd=folder,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        deadline=time.monotonic()+45
        while p.poll() is None:
            if time.monotonic()>deadline or psutil.virtual_memory().available<512*2**20:
                p.kill();p.wait();record['state']='timeout_or_memory_guard';break
            time.sleep(.25)
        else:record['state']='simulated' if p.returncode==0 and (folder/'ac.raw').exists() else 'simulator_failed'
    record.update(seconds=time.monotonic()-tick,returncode=p.returncode)
    if record['state']=='simulated':
        try:record['analysis']=analyze(folder,row,original)
        except Exception as e:record.update(state='analysis_failed',error=str(e))
    dump(folder/'result.json',record);print(label,record['state'],round(record['seconds'],2),flush=True);return record

def main():
    BASE.mkdir(parents=True,exist_ok=True);record={};results=[]
    try:
        record=pause()
        if psutil.virtual_memory().available<1.25*2**30:raise MemoryError('Not enough released RAM for the bounded replay')
        model=unpack_models();a=ROOT/'data/raw/LOADBench/subset/SimpleOpAmp130';b=ROOT/'data/raw/LOADBench/subset/SimpleOpAmp4'
        cases=[('extreme_130',source_row(a,'intermediate_prefix.csv.gz',5795)),
               ('normal_final_130',source_row(a,'FinalPOFfull.csv',normal=True)),
               ('normal_history_130',source_row(a,'intermediate_prefix.csv.gz',normal=True)),
               ('normal_final_4',source_row(b,'FinalPOFfull.csv',normal=True))]
        for label,source in cases:results.append(run_case(label,source,model))
        dump(BASE/(VARIANT+'_summary.json'),{'created_at':now(),'results':results,'original_training_recipe_sha256':record['recipe_sha256'],'new_dataset_labels_created':False})
        # Independent metadata flags and checks are cheap and run before recovery.
        if (Path(__file__).with_name('audit.py')).exists() and not (BASE/'manifest.json').exists():
            subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-X','utf8','-B',str(Path(__file__).with_name('audit.py'))],cwd=ROOT,check=True)
    finally:
        # Even an extractor/simulator failure must not strand the original run.
        if not record and (BASE/'pause_record.json').exists():
            pending=read(BASE/'pause_record.json')
            if 'resume_state' not in pending:record=pending
        resume(record)

if __name__=='__main__':main()
