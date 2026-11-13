"""Adopt a verified sampler revision without rewriting old sample identities."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import zipfile
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read,digest
from src.analoggym_campaign import atomic_json,sha


def main():
    output=ROOT/'data/generated/analoggym_personal_20k_v1';experiments=ROOT/'data/generated/analoggym_folded_optimization_v2'
    progress=read(output/'status.json');assert progress['state'] not in {'running','draining'}
    if (output/'sampling_policy_v2.json').exists():raise ValueError('Revision already adopted')
    validation=read(experiments/'validate_summary.json');stats=validation['summary']['narrow_recentered']
    assert stats['attempts']>=20 and stats['accepted']/stats['attempts']>=.8
    plan=read(experiments/'plan.json');strategy=plan['strategies']['narrow_recentered']
    db=sqlite3.connect(output/'ledger.sqlite')
    maximum=db.execute("select max(candidate) from attempts where topology='ldo_folded_cascode'").fetchone()[0]
    old_counts=dict(db.execute('select topology,count(*) from attempts where accepted=1 group by topology'));db.close()
    old_archives=[int(p.stem) for p in (output/'raw/ldo_folded_cascode').glob('*/*.zip')]
    start=max([maximum]+old_archives)+1
    backup=output/'revisions/sampler_v1_before_v2';backup.mkdir(parents=True,exist_ok=False)
    for name in ['manifest.json','status.json','dataset.json','orchestration_recipe.json']:
        shutil.copyfile(output/name,backup/name)
    preserved={}
    for shard in read(output/'dataset.json')['shards']:
        for row in pq.read_table(ROOT/shard['path']).to_pylist():
            preserved[row['sample_id']]=digest(row)
    assert len(preserved)==sum(old_counts.values())
    atomic_json(backup/'preserved_row_hashes.json',preserved)
    # A source snapshot is retained as text; legacy seven frozen modules unchanged.
    # The previous controller implementation hash remains in the backed-up manifest.
    policy={**strategy,'version':'folded_recentered_narrow_v2','start_candidate':start,
            'seed_offset':20261028,'vary_discrete':False}
    implementations=['src/analoggym_personal_engine.py','src/analoggym_personal_runner.py']
    revision={'adopted_at_UTC':datetime.now(timezone.utc).isoformat(),'topology':'ldo_folded_cascode',
        'sampling_policy':policy,'base_recipe_hash':read(output/'manifest.json')['recipe_hash'],
        'implementation_sha256':{p:sha(ROOT/p) for p in implementations},
        'centre_provenance':plan['recentred_provenance'],
        'validation_archive':str((experiments/'validate_summary.json').relative_to(ROOT)).replace('\\','/'),
        'validation_sha256':sha(experiments/'validate_summary.json'),
        'preserved_accepted_counts':old_counts,'quality_threshold_changed':False,
        'testbench_or_metric_definition_changed':False,'other_nine_samplers_changed':False,
        'provenance_note':'recipe_hash remains the immutable base simulation recipe; sampler version/hash are additional per-row and per-archive fields. Old candidates keep v1 parameters, identities and labels.'}
    atomic_json(output/'sampling_policy_v2.json',revision)
    manifest=read(output/'manifest.json');manifest.update(
        personal_controller_sha256=sha(ROOT/'scripts/generate_analoggym_personal.py'),
        sampling_policy_revision='folded_recentered_narrow_v2',sampling_policy_file='sampling_policy_v2.json',
        sampling_policy_file_sha256=sha(output/'sampling_policy_v2.json'))
    atomic_json(output/'manifest.json',manifest)
    orchestration=read(output/'orchestration_recipe.json')
    for p in implementations+['scripts/generate_analoggym_personal.py','scripts/report_analoggym_personal.py']:
        orchestration['code_sha256'][p]=sha(ROOT/p)
    atomic_json(output/'orchestration_recipe.json',orchestration)
    print('Adopted v2 starting candidate',start,'; retained',sum(old_counts.values()),'accepted samples')


if __name__=='__main__':main()
