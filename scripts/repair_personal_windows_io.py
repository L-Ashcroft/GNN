"""Record and pin an I/O-only revision; sampling policy contents stay identical."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read,digest
from src.analoggym_campaign import sha
from src.windows_campaign_io import atomic_json


def main():
    out=ROOT/'data/generated/analoggym_personal_20k_v1';backup=out/'revisions/windows_io_retry_v1'
    original=read(backup/'sampling_policy_v2.json');recipe=read(out/'recipe.json')
    for path,expected in recipe['code_sha256'].items():assert sha(ROOT/path)==expected
    for filename in ['sampling_policy_v2.json','manifest.json','orchestration_recipe.json']:
        assert sha(out/filename)==sha(backup/filename),'Review already recorded; do not repeat'
    for name in ['runner','engine']:
        path=f'src/analoggym_personal_{name}.py'
        assert sha(backup/f'analoggym_personal_{name}_before.py')==original['implementation_sha256'][path]
    db=sqlite3.connect(f'file:{(out/"ledger.sqlite").as_posix()}?mode=ro',uri=True)
    rows=list(db.execute('select topology,candidate,accepted,result_json,archive_sha256 from attempts order by topology,candidate'));db.close()
    atomic_json(backup/'preserved_ledger_hashes.json',{
        f'{r[0]}::{r[1]}':digest(list(r)) for r in rows})
    revised=read(out/'sampling_policy_v2.json')
    revised['implementation_sha256'].update({p:sha(ROOT/p) for p in [
        'src/analoggym_personal_runner.py','src/analoggym_personal_engine.py','src/windows_campaign_io.py']})
    assert digest(revised['sampling_policy'])==digest(original['sampling_policy'])
    atomic_json(out/'sampling_policy_v2.json',revised)
    manifest=read(out/'manifest.json');manifest['sampling_policy_file_sha256']=sha(out/'sampling_policy_v2.json')
    atomic_json(out/'manifest.json',manifest)
    orchestration=read(out/'orchestration_recipe.json')
    orchestration['code_sha256'].update(revised['implementation_sha256'])
    atomic_json(out/'orchestration_recipe.json',orchestration)
    atomic_json(backup/'repair_receipt.json',{'time_UTC':datetime.now(timezone.utc).isoformat(),
        'original_error':read(backup/'last_error.json'),'preserved_attempts':len(rows),
        'preserved_accepted':sum(r[2] for r in rows),'sampling_policy_digest_unchanged':digest(revised['sampling_policy']),
        'change':'Unique JSON temporary names, fsync, retry only Windows access/sharing/lock errors up to 15 seconds. Same status fields, simulation, extraction, quality rules and parameters.',
        'old_metadata_snapshots':str(backup.relative_to(ROOT))})
    print('I/O-only revision recorded;',len(rows),'ledger rows preserved')


if __name__=='__main__':main()
