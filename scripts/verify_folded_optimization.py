"""Recompute trial labels and compare three new-region points at finer precision."""
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import read
from src.analoggym_campaign import atomic_json,extract
from src.analoggym_personal_engine import simulate


def main():
    out=ROOT/'data/generated/analoggym_folded_optimization_v2';plan=read(out/'plan.json')
    total=0;accepted=0
    for phase in ['screen','validate']:
        for result in read(out/(phase+'_summary.json'))['results']:
            with zipfile.ZipFile(ROOT/result['archive']) as z:
                assert z.testzip() is None
                assert json.loads(z.read('result.json'))=={k:v for k,v in result.items() if k!='strategy'}
                for member,expected in result['artifacts_sha256'].items():
                    assert hashlib.sha256(z.read(member)).hexdigest()==expected
                if result['accepted']:
                    values=extract(plan['entry'],{n:z.read(n) for n in z.namelist() if n.endswith('_wave.dat')})
                    np.testing.assert_array_equal(values,result['performance']);accepted+=1
            total+=1
    validation=read(out/'validate_summary.json')['results'];selected=[]
    for residue in range(3):
        selected.append(next(r for r in validation if r['accepted'] and r['candidate']%3==residue))
    entry=copy.deepcopy(plan['entry'])
    entry['_sampling_policy']={**plan['strategies']['narrow_recentered'],'start_candidate':0,
        'seed_offset':20261028,'version':'folded_v2_trial_narrow_recentered','vary_discrete':False}
    def precise(r):return simulate(entry,r['candidate'],out/'precision',plan['environment'],120,True)
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(precise,selected))
    comparisons=[]
    for original,finer in zip(selected,results):
        assert finer['accepted'] and finer['parameters_native']==original['parameters_native']
        a=np.array(original['performance']);b=np.array(finer['performance'])
        # Audit only: never changes production acceptance. Near-zero native values get a tiny floor.
        relative=np.abs(a-b)/np.maximum(np.abs(a),1e-8)
        comparisons.append({'candidate':original['candidate'],'maximum_relative_change':float(relative.max()),
            'per_metric_relative_change':relative.tolist(),'archive':finer['archive']})
    summary={'archives_crc_and_sha_checked':total,'accepted_labels_recomputed_exactly':accepted,
        'precision_comparisons':comparisons,'production_samples_added':0}
    atomic_json(out/'verification.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
