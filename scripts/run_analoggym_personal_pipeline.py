"""Run resumable 8-worker production, then finalize independent evaluation files."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read
from scripts.generate_analoggym_personal import settings,guarded_run
from src.analoggym_personal_protocols import finalize
from src.analoggym_campaign import sha


if __name__=='__main__':
    config,output=settings()
    for name,expected in read(output/'orchestration_recipe.json')['code_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Pipeline code changed: '+name)
    assert read(output/'manifest.json')['stage']=='production'
    guarded_run()
    progress=read(output/'status.json')
    if progress['complete']:
        print('FINALIZED',finalize(output),flush=True)
    else:print('CHECKPOINTED',progress['state'],progress['accepted'],flush=True)
    from scripts.report_analoggym_personal import report
    report()
