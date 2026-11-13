"""Audit all accepted archives and adopt an explicit, reproducible quality gate."""
from datetime import datetime,timezone
from pathlib import Path
import json
import shutil
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read,digest
from src.analoggym_campaign import sha
from src.analoggym_personal_quality import POLICY,apply_quality
from src.windows_campaign_io import atomic_json


def main():
    output=ROOT/'data/generated/analoggym_personal_20k_v1'
    assert read(output/'status.json')['state']=='stopped_by_request'
    assert (output/'STOP').read_text(encoding='utf-8').startswith('AC quality revision maintenance:')
    if (output/'quality_policy_v1.json').exists():raise ValueError('Already adopted')
    import msvcrt
    lock=(output/'runner.lock').open('a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        backup=output/'revisions/ac_quality_v1';backup.mkdir(parents=True,exist_ok=False)
        for name in ['manifest.json','sampling_policy_v2.json','orchestration_recipe.json','status.json','dataset.json']:
            shutil.copyfile(output/name,backup/name)
        recipe=read(output/'recipe.json');entries={e['name']:e for e in recipe['entries']}
        for path,expected in recipe['code_sha256'].items():assert sha(ROOT/path)==expected
        db=sqlite3.connect(output/'ledger.sqlite');db.execute('pragma synchronous=FULL')
        db.execute('''create table if not exists quality_reviews(
            topology text,candidate integer,policy_sha256 text,original_result_json text,decision_json text,
            reviewed_at_UTC text,primary key(topology,candidate,policy_sha256))''')
        rows=list(db.execute('select topology,candidate,accepted,result_json,archive_sha256 from attempts order by topology,candidate'))
        atomic_json(backup/'ledger_hashes_before.json',{f'{r[0]}::{r[1]}':digest(list(r)) for r in rows})
        rejected=[];checked=0;now=datetime.now(timezone.utc).isoformat()
        for name,candidate,accepted,raw,archive_sha in rows:
            if not accepted:continue
            original=json.loads(raw)
            assert sha(ROOT/original['archive'])==archive_sha
            result=apply_quality(entries[name],original);checked+=1
            if not result['accepted']:
                db.execute('insert into quality_reviews values(?,?,?,?,?,?)',
                    (name,candidate,digest(POLICY),raw,json.dumps(result,allow_nan=False),now))
                db.execute('update attempts set accepted=0,result_json=? where topology=? and candidate=?',
                    (json.dumps(result,allow_nan=False),name,candidate))
                rejected.append({'topology':name,'candidate':candidate,'sample_design':entries[name]['graph']['topology_hash'],
                    'archive':original['archive'],'archive_sha256':archive_sha,'diagnostics':result['ac_quality_diagnostics']})
            if checked%500==0:print('Retrospective AC quality scan',checked,flush=True)
        db.commit()
        # Good and previously failed rows remain byte-for-byte identical in the ledger.
        after=list(db.execute('select topology,candidate,accepted,result_json,archive_sha256 from attempts order by topology,candidate'))
        excluded={(r['topology'],r['candidate']) for r in rejected}
        assert all(a==b for a,b in zip(rows,after) if (a[0],a[1]) not in excluded)
        db.close()
        policy_record={'policy':POLICY,'adopted_at_UTC':now,'base_recipe_hash':read(output/'manifest.json')['recipe_hash'],
            'retrospective_accepted_scanned':checked,'quarantined':rejected,'raw_archives_modified':False,
            'unchanged_other_ledger_rows':len(rows)-len(rejected),'goal_accepted':20000,'per_topology_goal':2000,
            'performance_definitions_or_masks_changed':False,'sampling_parameters_changed':False}
        atomic_json(output/'quality_policy_v1.json',policy_record)
        revision=read(output/'sampling_policy_v2.json')
        for path in ['src/analoggym_personal_runner.py','src/analoggym_personal_quality.py','scripts/check_personal_quality.py']:
            revision['implementation_sha256'][path]=sha(ROOT/path)
        atomic_json(output/'sampling_policy_v2.json',revision)
        manifest=read(output/'manifest.json');manifest.update(
            sampling_policy_file_sha256=sha(output/'sampling_policy_v2.json'),quality_policy_file='quality_policy_v1.json',
            quality_policy_file_sha256=sha(output/'quality_policy_v1.json'))
        atomic_json(output/'manifest.json',manifest)
        orchestration=read(output/'orchestration_recipe.json');orchestration['code_sha256'].update(revision['implementation_sha256'])
        orchestration['code_sha256']['scripts/monitor_analoggym_personal.py']=sha(ROOT/'scripts/monitor_analoggym_personal.py')
        atomic_json(output/'orchestration_recipe.json',orchestration)
        print('Accepted scanned',checked,'; quarantined',len(rejected),'; all other ledger rows unchanged')
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':main()
