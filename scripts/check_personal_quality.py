"""Full accepted-ledger AC gate check; no simulations or parameter fitting."""
from collections import Counter
from pathlib import Path
import json
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read,digest
from src.analoggym_campaign import sha
from src.analoggym_personal_quality import POLICY,apply_quality
from src.windows_campaign_io import atomic_json


def check(output):
    output=Path(output);recipe=read(output/'recipe.json');entries={e['name']:e for e in recipe['entries']}
    manifest=read(output/'manifest.json')
    assert sha(output/'quality_policy_v1.json')==manifest['quality_policy_file_sha256']
    assert read(output/'quality_policy_v1.json')['policy']==POLICY
    db=sqlite3.connect(f'file:{(output/"ledger.sqlite").as_posix()}?mode=ro',uri=True)
    db.execute('begin');rows=list(db.execute('select topology,result_json from attempts where accepted=1'));db.close()
    counts=Counter()
    for index,(name,raw) in enumerate(rows):
        assert apply_quality(entries[name],json.loads(raw))['accepted'],f'Accepted sample requires AC review: {name}'
        counts[name]+=1
        if index%1000==0:print('Full quality gate check',index,'/',len(rows),flush=True)
    if len(rows)>=manifest['target_accepted']:assert dict(counts)==manifest['quotas']
    result={'passed':True,'policy_sha256':digest(POLICY),'accepted_checked':len(rows),'topology_counts':dict(counts),
        'all_accepted_pass_ac_gate':True,'scope':'Snapshot of accepted ledger; no new simulations'}
    atomic_json(output/'quality_gate_checks.json',result);return result


if __name__=='__main__':print(json.dumps(check(ROOT/'data/generated/analoggym_personal_20k_v1'),indent=2))
