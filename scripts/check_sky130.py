"""Check ngspice can load the bundled SKY130 library and simulate NMOS/PMOS."""
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import dump, read, rel


def main():
    if platform.system() != 'Windows': raise RuntimeError('Native Windows is required')
    config = read(ROOT / 'config/simulation_environment.json')
    executable = (ROOT / config['ngspice_executable']).resolve(strict=True)
    library = (ROOT / config['library']).resolve(strict=True)
    if not executable.is_relative_to(ROOT) or not library.is_relative_to(ROOT):
        raise ValueError('Simulation paths must remain within workspace')
    parent = ROOT / 'data/simulator_checks'; parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix='sky130_', dir=parent))
    # Library all.spice explicitly sets scale=1u; l/w inputs here are micrometres.
    text = f'''SKY130 TT CMOS inverter model check
.lib "{library.as_posix()}" tt
.temp 27
.options reltol=1e-6 vntol=1e-9 abstol=1e-15
VDD vdd 0 1.8
VIN in 0 0
XN out in 0 0 sky130_fd_pr__nfet_01v8 l=0.15 w=1
XP out in vdd vdd sky130_fd_pr__pfet_01v8 l=0.15 w=2
.control
set noaskquit
set num_threads=1
dc VIN 0 1.8 0.1
wrdata inverter_dc.dat v(out)
quit
.endc
.end
'''
    (run_dir / 'inverter.cir').write_text(text, encoding='ascii')
    temp = ROOT / '.tmp/ngspice'; temp.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    proc = subprocess.run([str(executable), '-n', '-b', '-o', 'ngspice.log', 'inverter.cir'],
                          cwd=run_dir, env={**os.environ, 'TEMP': str(temp), 'TMP': str(temp)},
                          capture_output=True, text=True, timeout=60,
                          creationflags=subprocess.CREATE_NO_WINDOW)
    (run_dir / 'stdout.txt').write_text(proc.stdout, encoding='utf-8')
    (run_dir / 'stderr.txt').write_text(proc.stderr, encoding='utf-8')
    log = (run_dir / 'ngspice.log').read_text(encoding='utf-8', errors='replace')
    if proc.returncode or re.search(r'(?im)^\s*(?:error|fatal)\b', log):
        raise RuntimeError('SKY130 check failed; inspect ' + rel(run_dir / 'ngspice.log'))
    values = np.loadtxt(run_dir / 'inverter_dc.dat', ndmin=2)
    assert values.shape == (19, 2) and np.isfinite(values).all()
    assert values[0, 1] > 1.6 and values[-1, 1] < 0.2
    assert np.all(np.diff(values[:, 1]) <= 1e-5)
    warnings = [line for line in log.splitlines() if 'warning' in line.lower()]
    dump(ROOT / 'reports/sky130_check.json', {
        'status': 'passed', 'pdk_root': config['pdk_root'], 'library': config['library'],
        'corner': 'tt', 'temperature_C': 27, 'supply_voltage_V': 1.8,
        'models_checked': ['sky130_fd_pr__nfet_01v8', 'sky130_fd_pr__pfet_01v8'],
        'dc_points': len(values), 'output_at_input_0V': float(values[0,1]),
        'output_at_input_1_8V': float(values[-1,1]), 'warnings': warnings,
        'elapsed_seconds': time.perf_counter()-started, 'run_directory': rel(run_dir),
        'scope': 'Library includes and a simple CMOS inverter; AnalogGym full testbench not yet validated'})
    print('SKY130 TT NMOS/PMOS passed;', len(values), 'DC points')
    print('Vout:', float(values[0,1]), 'V at Vin=0;', float(values[-1,1]), 'V at Vin=1.8')
    print('Warnings:', len(warnings), '; report: reports/sky130_check.json')


if __name__ == '__main__': main()
