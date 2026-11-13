"""Verify a completed finite smoke dataset can resume without new simulations."""
import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.analoggym_campaign import inside, read, sha, atomic_json


def snapshot(folder):
    files=sorted(folder.glob('raw/*/*/*.zip'))+sorted(folder.glob('samples/*/*.parquet'))+sorted(folder.glob('index/*/*.parquet'))
    hashes={str(p.relative_to(folder)).replace('\\','/'):sha(p) for p in files}
    with sqlite3.connect(f'file:{(folder/"ledger.sqlite").as_posix()}?mode=ro',uri=True) as db:
        counts=db.execute('select count(*),sum(accepted) from attempts').fetchone()
    return {'files':hashes,'attempts':counts[0],'accepted':counts[1]}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='data/generated/analoggym_campaign_smoke_v5')
    a=p.parse_args(); folder=inside(ROOT/a.output)
    if read(folder/'status.json')['state']!='complete':
        raise ValueError('This check only resumes an already completed finite smoke campaign')
    before=snapshot(folder)
    with (folder/'resume_check.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'scripts/generate_analoggym_dataset.py','run','--output',a.output],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    after=snapshot(folder)
    assert before==after, 'Resume changed designs, attempt count, raw evidence or exported Parquet'
    assert read(folder/'quality_checks.json')['passed']
    result={'passed':True,'attempts_unchanged':before['attempts'],'accepted_unchanged':before['accepted'],
            'unchanged_archive_and_parquet_files':len(before['files']),
            'scope':'Completed finite campaign resume; no new simulations or duplicate designs'}
    atomic_json(folder/'resume_checks.json',result)
    print(json.dumps(result,indent=2))
