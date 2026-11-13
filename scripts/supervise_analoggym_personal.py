"""Native Windows supervisor: automatically resume only our RAM-protection stop.

User STOP, disk/attempt budgets and errors are never automatically cleared.
"""
import json
import msvcrt
from pathlib import Path
import subprocess
import sys
import time
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.windows_campaign_io import atomic_json
from scripts.generate_analoggym_personal import settings


def main():
    _,output=settings();stop_path=output/'STOP'
    lock=(output/'supervisor.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        while True:
            state=json.loads((output/'status.json').read_text())
            if state.get('complete'):
                atomic_json(output/'supervisor_status.json',{'state':'complete','accepted':state['accepted']});return
            if stop_path.exists():
                content=stop_path.read_text(encoding='utf-8')
                if not content.startswith('Resource guard:'):
                    atomic_json(output/'supervisor_status.json',{'state':'stopped_by_user','accepted':state['accepted']});return
                stable_since=None
                while stop_path.exists():
                    if stop_path.read_text(encoding='utf-8')!=content:
                        atomic_json(output/'supervisor_status.json',{'state':'stopped_by_user'});return
                    available=psutil.virtual_memory().available
                    atomic_json(output/'supervisor_status.json',{'state':'waiting_for_memory','available_bytes':available,
                        'resume_threshold_bytes':int(1.5*2**30),'required_stable_seconds':30,'accepted':state['accepted']})
                    stable_since=(stable_since or time.monotonic()) if available>=1.5*2**30 else None
                    if stable_since is not None and time.monotonic()-stable_since>=30:
                        # This is one explicitly named file in the validated campaign directory.
                        if stop_path.resolve().parent!=output.resolve():raise ValueError('Unsafe STOP path')
                        if stop_path.read_text(encoding='utf-8')==content:stop_path.unlink()
                        break
                    time.sleep(5)
            atomic_json(output/'supervisor_status.json',{'state':'running','accepted':state['accepted']})
            result=subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts/run_analoggym_personal_pipeline.py')],
                cwd=ROOT,creationflags=subprocess.CREATE_NO_WINDOW)
            state=json.loads((output/'status.json').read_text())
            if state.get('complete'):continue
            if stop_path.exists() and stop_path.read_text(encoding='utf-8').startswith('Resource guard:'):continue
            atomic_json(output/'supervisor_status.json',{'state':'checkpointed_requires_review',
                'campaign_state':state['state'],'returncode':result.returncode,'accepted':state['accepted']})
            return
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':main()
