"""Run two numerical resolutions on the same qualified design per topology."""
import subprocess
import sys
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


if __name__=='__main__':
    qualified='data/generated/analoggym_qualification_current'
    names=sorted({r['topology'] for r in json.loads((ROOT/qualified/'summary.json').read_text())['results'] if r['accepted']})
    common=['--candidates','1','--workers','4','--names',*names,'--qualified',qualified]
    for output,extra in [('analoggym_standard_v4',[]),('analoggym_precision_v4',['--precision'])]:
        subprocess.run([sys.executable,'scripts/probe_analoggym_campaign.py','--output','data/generated/'+output,*extra,*common],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'scripts/check_analoggym_precision.py','--standard','data/generated/analoggym_standard_v4','--tighter','data/generated/analoggym_precision_v4'],cwd=ROOT,check=True)
