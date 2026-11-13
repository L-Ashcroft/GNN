"""Audit exported samples against source terminals, ledger and actual raw waves."""
import argparse
import hashlib
import json
import re
import sqlite3
import sys
import zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.analoggym_campaign import inside, read, sha, atomic_json, extract, evaluate, param_definitions
from src.analoggym_campaign_runner import design_fingerprint, metadata, unit
from src.load_analoggym_generated import iter_campaign_samples


def check(output,deep_per_topology=3):
    output=inside(output); recipe=read(output/'recipe.json'); manifest=read(output/'manifest.json')
    entries={e['name']:e for e in recipe['entries']}
    ids=set(); designs=set(); counts={}; deep={}; archives=set()
    for name,e in entries.items():
        g=read(output/'topologies'/name/'graph.json'); original=(ROOT/e['dut']).read_text().lower()
        # Independent terminal oracle: MOS instance token order in source SPICE.
        for d in g['devices']:
            match=re.search(r'^'+re.escape(d['name'])+r'\s+(.+)$',original,re.M)
            if not match: raise AssertionError('Graph device absent in original netlist')
            tokens=match[1].split()
            assert tokens[:len(d['nets'])]==d['nets']
            if d['model'] in {'nmos','pmos'}: assert d['ports']==['D','G','S','B']
        assert g['num_devices']==len(g['devices'])
        assert g['num_nodes']==len({n for d in g['devices'] for n in d['nets']})
        assert g['num_edges']==sum(len(d['ports'])*(len(d['ports'])-1)//2 for d in g['devices'])
        expected=[]
        for i,d in enumerate(g['devices']):
            for a in range(len(d['ports'])):
                for b in range(a+1,len(d['ports'])):
                    expected.append((i,d['ports'][a],d['ports'][b],d['nets'][a],d['nets'][b]))
        actual=[(x['device'],*x['ports'],g['nets'][x['source']],g['nets'][x['target']]) for x in g['edges']]
        assert expected==actual
    db=sqlite3.connect(f'file:{(output/"ledger.sqlite").as_posix()}?mode=ro',uri=True)
    for row in iter_campaign_samples(output):
        name=row['topology_id']; e=entries[name]
        assert row['sample_id'] not in ids and row['design_group'] not in designs
        ids.add(row['sample_id']); designs.add(row['design_group']); counts[name]=counts.get(name,0)+1
        assert len(row['performance'])==len(row['mask'])==len(row['semantic_ids'])==16
        for v,k,s in zip(row['performance'],row['mask'],row['semantic_ids']):
            assert bool(k)==(v is not None)==(s is not None)
            if k: assert np.isfinite(v)
        assert row['recipe_hash']==manifest['recipe_hash']
        mapping=metadata(e)
        assert row['semantic_ids']==[m['semantic_id'] for m in mapping['metrics']]+[None]*(16-len(mapping['metrics']))
        assert row['parameter_units']==[unit(k) for k in row['parameter_names']]
        assert len(row['edge_features'])==row['graph']['num_edges']
        params=dict(zip(row['parameter_names'],row['parameters']))
        assert design_fingerprint(e,params)==row['design_group']
        expected=[[None if exp is None else evaluate(exp,params) for exp in r] for r in row['graph']['feature_expressions']]
        assert expected==row['device_features']
        accepted,stored,archive_sha=db.execute('select accepted,result_json,archive_sha256 from attempts where topology=? and candidate=?',(name,row['candidate'])).fetchone()
        assert accepted==1
        result=json.loads(stored)
        assert result['performance']==[v for v,k in zip(row['performance'],row['mask']) if k]
        assert result['parameters_native']==params
        assert row['source_sha256']==archive_sha
        if deep_per_topology<0 or deep.get(name,0)<deep_per_topology:
            archive=inside(ROOT/row['source_path']); assert sha(archive)==archive_sha
            with zipfile.ZipFile(archive) as z:
                assert z.testzip() is None
                for member,expected_sha in result['artifacts_sha256'].items():
                    assert hashlib.sha256(z.read(member)).hexdigest()==expected_sha
                values=extract(e,{s:z.read(s) for s in z.namelist() if s.endswith('_wave.dat')})
                np.testing.assert_array_equal(values,result['performance'])
                native={k:float(v) for k,v in re.findall(r'\.param (\w+)=(\S+)',z.read('parameters.spice').decode())}
                assert native==params
            deep[name]=deep.get(name,0)+1; archives.add(str(archive))
    ledger_counts=dict(db.execute('select topology,count(*) from attempts where accepted=1 group by topology'))
    for name,count in counts.items(): assert count<=ledger_counts[name] and count<=manifest['quotas'][name]
    state=read(output/'status.json')['state']
    if state not in {'running','draining'}:
        assert counts==ledger_counts
        assert len(ids)==read(output/'dataset.json')['sample_count']
    db.close()
    summary={'passed':True,'samples_checked':len(ids),'topologies':counts,'raw_waveforms_recomputed':deep,
             'sample_and_design_ids_unique':True,'ports_and_connectivity_checked':True,
             'mask_only_missing':True,'not_merged_with_historical':True,
             'scope':'export snapshot; running campaign may have additional unexported ledger rows'}
    atomic_json(output/'quality_checks.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',default='data/generated/analoggym_campaign_v1')
    p.add_argument('--deep-per-topology',type=int,default=3); a=p.parse_args()
    print(json.dumps(check(ROOT/a.output,a.deep_per_topology),indent=2))
