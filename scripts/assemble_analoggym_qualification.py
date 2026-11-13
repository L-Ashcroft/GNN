"""Recheck recorded runs under the current definitions and exact fixture conditions."""
import argparse
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.analoggym_campaign import catalog, read, extract, atomic_json, testbenches, rel
from src.analoggym_campaign_runner import normalize_deck


def main():
    p=argparse.ArgumentParser(); p.add_argument('inputs',nargs='+'); p.add_argument('--output',required=True)
    a=p.parse_args(); entries,excluded=catalog(); by_name={e['name']:e for e in entries}
    env=read(ROOT/'config/simulation_environment.json'); results=[]; omitted=[]
    for folder in a.inputs:
        for r in read(ROOT/folder/'summary.json')['results']:
            if not r['accepted']: results.append(r); continue
            e=by_name[r['topology']]
            try:
                with zipfile.ZipFile(ROOT/r['archive']) as z:
                    for kind,deck in testbenches(e,ROOT/env['library']).items():
                        if normalize_deck(z.read(kind+'.cir').decode())!=normalize_deck(deck):
                            raise ValueError('Superseded testbench conditions')
                    vals=extract(e,{s:z.read(s) for s in z.namelist() if s.endswith('_wave.dat')})
                    np.testing.assert_array_equal(vals,r['performance'])
                results.append(r)
            except (ValueError,AssertionError) as error:
                omitted.append({'archive':r['archive'],'reason':str(error),'topology':r['topology']})
    atomic_json(ROOT/a.output/'summary.json',{'results':results,'inputs':a.inputs,'accepted':sum(r['accepted'] for r in results),
                'superseded_or_rejected':omitted,'note':'Original diagnostics unchanged; only currently compatible verified seeds are eligible'})
    atomic_json(ROOT/a.output/'catalog.json',{'entries':entries,'excluded':excluded})
    print('accepted',sum(r['accepted'] for r in results),'topologies',len({r['topology'] for r in results if r['accepted']}),'superseded',len(omitted))


if __name__=='__main__': main()
