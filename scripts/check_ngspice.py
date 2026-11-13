"""Native Windows ngspice check against analytic DC/AC results, no PDK needed."""
import argparse
import hashlib
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
from src.common import dump, rel

NETLIST = """EDA ngspice installation check: resistive divider with RC low-pass
V1 in 0 dc 1.8 ac 1
R1 in out 1k
R2 out 0 1k
C1 out 0 1u
.control
set noaskquit
set num_threads=1
set filetype=ascii
dc V1 0 1.8 0.1
wrdata divider_dc.dat v(out)
ac dec 20 1 10000
wrdata divider_ac.dat v(out)
quit
.endc
.end
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', default='tools/Spice64/bin/ngspice_con.exe')
    args = parser.parse_args()
    if platform.system() != 'Windows':
        raise RuntimeError('This workspace requires native Windows execution')
    executable = (ROOT / args.executable).resolve(strict=True)
    if not executable.is_relative_to(ROOT):
        raise ValueError('Use the simulator installed within this workspace')
    temp = ROOT / '.tmp/ngspice'; temp.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'TEMP': str(temp), 'TMP': str(temp)}
    flags = subprocess.CREATE_NO_WINDOW
    version = subprocess.run([str(executable), '--version'], cwd=ROOT, env=env,
                             capture_output=True, text=True, timeout=15, check=True,
                             creationflags=flags)
    match = re.search(r'ngspice-(\d+)', version.stdout + version.stderr)
    if not match: raise RuntimeError('Cannot parse ngspice version')
    major = int(match[1])
    if major < 42: raise RuntimeError('AnalogGym requires ngspice >= 42')
    parent = ROOT / 'data/simulator_checks'; parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix='ngspice_', dir=parent))
    netlist = run_dir / 'divider.cir'; netlist.write_text(NETLIST, encoding='ascii')
    started = time.perf_counter()
    result = subprocess.run([str(executable), '-n', '-b', '-o', 'ngspice.log', netlist.name],
                            cwd=run_dir, env=env, capture_output=True, text=True,
                            timeout=30, creationflags=flags)
    elapsed = time.perf_counter() - started
    (run_dir / 'stdout.txt').write_text(result.stdout, encoding='utf-8')
    (run_dir / 'stderr.txt').write_text(result.stderr, encoding='utf-8')
    log = (run_dir / 'ngspice.log').read_text(encoding='utf-8', errors='replace')
    if result.returncode or re.search(r'(?im)^\s*(?:error|fatal)\b', log):
        raise RuntimeError('Simulator failed; inspect ' + rel(run_dir / 'ngspice.log'))
    dc = np.loadtxt(run_dir / 'divider_dc.dat', ndmin=2)
    ac = np.loadtxt(run_dir / 'divider_ac.dat', ndmin=2)
    assert dc.shape == (19, 2) and ac.shape[1] == 3
    assert np.isfinite(dc).all() and np.isfinite(ac).all()
    np.testing.assert_allclose(dc[:, 1], dc[:, 0] / 2, rtol=1e-7, atol=1e-10)
    expected = 1 / (2 + 1j * 2 * np.pi * ac[:, 0] * 1e-3)
    np.testing.assert_allclose(ac[:, 1] + 1j * ac[:, 2], expected, rtol=1e-7, atol=1e-10)
    report = {'status': 'passed', 'executable': rel(executable), 'version_major': major,
              'executable_sha256': hashlib.sha256(executable.read_bytes()).hexdigest(),
              'operating_system': platform.platform(), 'native_windows': True,
              'run_directory': rel(run_dir), 'elapsed_seconds': elapsed,
              'dc_points_checked': len(dc), 'ac_points_checked': len(ac),
              'expected_output_at_1_8V': 0.9, 'actual_output_at_1_8V': float(dc[-1, 1]),
              'scope': 'Executable and analytic DC/AC correctness; SKY130/AnalogGym testbenches not yet validated'}
    dump(ROOT / 'reports/ngspice_check.json', report)
    print('ngspice', major, 'passed;', len(dc), 'DC points,', len(ac), 'AC points;',
          'Vout =', float(dc[-1, 1]), 'V at Vin = 1.8 V')
    print('Report: reports/ngspice_check.json; artifacts:', rel(run_dir))


if __name__ == '__main__': main()
