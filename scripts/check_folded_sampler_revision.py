"""Verify immutable old rows, pinned sources, and per-candidate revision trace."""
import copy
import json
from pathlib import Path
import sqlite3
import sys
import zipfile
import hashlib
import numpy as np
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read,digest
from src.analoggym_campaign import atomic_json,sha,extract
from src.analoggym_personal_engine import parameter_plan,sampling_trace


def check():
    out=ROOT/'data/generated/analoggym_personal_20k_v1';recipe=read(out/'recipe.json')
    manifest=read(out/'manifest.json');revision=read(out/'sampling_policy_v2.json')
    assert digest(recipe)==manifest['recipe_hash']==revision['base_recipe_hash']
    for path,expected in recipe['code_sha256'].items():assert sha(ROOT/path)==expected
    assert sha(out/'sampling_policy_v2.json')==manifest['sampling_policy_file_sha256']
    for path,expected in revision['implementation_sha256'].items():assert sha(ROOT/path)==expected
    old=read(out/'revisions/sampler_v1_before_v2/preserved_row_hashes.json');seen=set()
    new=0
    for shard in read(out/'dataset.json')['shards']:
        for row in pq.read_table(ROOT/shard['path']).to_pylist():
            version=row.pop('sampling_policy_version');policy_sha=row.pop('sampling_policy_sha256')
            if row['sample_id'] in old:
                assert digest(row)==old[row['sample_id']];seen.add(row['sample_id'])
                assert version=='personal_sampler_v1' and policy_sha==manifest['recipe_hash']
            elif row['topology_id']=='ldo_folded_cascode':
                entry=copy.deepcopy(next(e for e in recipe['entries'] if e['name']==row['topology_id']))
                entry['_sampling_policy']=revision['sampling_policy']
                expected=sampling_trace(entry,row['candidate'])
                assert version==expected['sampling_policy_version']
                assert policy_sha==expected['sampling_policy_sha256']
                assert dict(zip(row['parameter_names'],row['parameters']))==parameter_plan(entry,row['candidate'])
                new+=1
    assert seen==set(old)
    db=sqlite3.connect(f'file:{(out/"ledger.sqlite").as_posix()}?mode=ro',uri=True)
    boundary=revision['sampling_policy']['start_candidate']
    rows=db.execute("select result_json from attempts where topology='ldo_folded_cascode' and candidate>=?",(boundary,)).fetchall();db.close()
    entry=copy.deepcopy(next(e for e in recipe['entries'] if e['name']=='ldo_folded_cascode'))
    entry['_sampling_policy']=revision['sampling_policy']
    deep=0
    for raw, in rows:
        r=json.loads(raw)
        assert r['parameters_native']==parameter_plan(entry,r['candidate'])
        assert r['sampling_policy_sha256']==sampling_trace(entry,r['candidate'])['sampling_policy_sha256']
        if r['accepted'] and deep<3:
            with zipfile.ZipFile(ROOT/r['archive']) as z:
                assert z.testzip() is None
                for member,expected in r['artifacts_sha256'].items():
                    assert hashlib.sha256(z.read(member)).hexdigest()==expected
                values=extract(entry,{n:z.read(n) for n in z.namelist() if n.endswith('_wave.dat')})
                np.testing.assert_array_equal(values,r['performance'])
            deep+=1
    result={'passed':True,'old_rows_identical_except_added_trace_fields':len(seen),
        'base_recipe_and_frozen_code_unchanged':True,'new_folded_attempts_verified':len(rows),
        'new_folded_exported_rows_verified':new,'new_folded_raw_labels_recomputed':deep,
        'candidate_boundary':boundary}
    atomic_json(out/'sampling_revision_checks.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':check()
