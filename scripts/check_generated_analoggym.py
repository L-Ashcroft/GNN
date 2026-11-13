"""Reconcile every generated pilot label with its fresh raw waveform artifacts."""
import argparse
import hashlib
import re
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.common import dump, read, digest, rel
from src.analoggym_generation import extract_metrics, METRICS, DUT, sha, plan_parameters
from src.load_analoggym_generated import iter_generated_samples


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='data/generated/analoggym_nmcf_pilot_v2'); a=p.parse_args()
    folder=(ROOT/a.output).resolve()
    assert folder.is_relative_to(ROOT)
    summary=read(folder/'summary.json'); recipe=read(folder/'recipe.json')
    statuses=read(folder/'candidate_status.json'); graph=read(folder/'graph.json')
    assert digest(recipe)==summary['recipe_hash'] and len(statuses)==summary['attempts']
    # Validate all model files, not just the manifest's claim about the ZIP.
    config=read(ROOT/'config/simulation_environment.json')
    models=read(ROOT/config['pdk_manifest'])
    for path,expected in models['files_sha256'].items(): assert sha(ROOT/path)==expected,path
    assert sha(ROOT/config['ngspice_executable'])==recipe['ngspice_executable_sha256']
    assert sha(DUT)==recipe['source_dut_sha256']==graph['netlist_sha256']
    assert graph['num_devices']==26 and graph['num_nodes']==graph['num_nets']
    assert graph['num_edges']==24*6+2
    # Pin oracle from original DUT lines, separate from the generated graph parser.
    oracle={m[1]:m[2].split() for m in re.finditer(r'(?m)^(\S+)\s+(.+?)\s+sky130_fd_pr__[np]fet_01v8\s',DUT.read_text())}
    assert len(oracle)==24
    for device in graph['devices']:
        if device['model'] in ('nmos','pmos'):
            assert device['nets']==oracle[device['name']] and device['ports']==['D','G','S','B']
    plans=plan_parameters(summary['attempts'],recipe['seed'])
    for i,s in enumerate(statuses):
        run=ROOT/s['run_directory']
        assert run.is_relative_to(folder/'runs') and s['parameters_native']==plans[i]
        for path,expected in s['artifacts_sha256'].items(): assert sha(run/path)==expected,path
        # Recipe hashes identify canonical text. The raw artifact byte hashes
        # above separately validate Windows CRLF files without ignoring changes.
        for kind in ('acdc','tran'):
            text=(run/(kind+'.cir')).read_text(encoding='ascii')
            assert hashlib.sha256(text.encode()).hexdigest()==recipe['testbenches_sha256'][kind]
        actual={k:float(v) for k,v in re.findall(r'(?m)^\.param\s+(\w+)=([-+0-9.eE]+)',(run/'parameters.spice').read_text())}
        assert actual==plans[i]
    rows=list(iter_generated_samples(a.output))
    assert len(rows)==summary['accepted'] and len({r['sample_id'] for r in rows})==len(rows)
    assert len({r['design_group'] for r in rows})==len(rows)
    for row in rows:
        values=extract_metrics(ROOT/row['source_path'])
        np.testing.assert_array_equal(row['performance'][:len(METRICS)],values)
        assert len(row['mask'])==len(row['performance'])==len(row['semantic_ids'])==16
        assert row['mask']==[True]*13+[False]*3
        assert all(v is None for v in row['performance'][13:])
        assert row['source_dataset']=='AnalogGym_generated' and row['temperature'] is None
        assert len(row['edge_features'])==146 and len(row['device_features'])==26
        assert all(len(v)==17 for v in row['edge_features'])
        assert all(np.isfinite(v) for f in row['device_features'] for v in f if v is not None)
        for d,features in zip(graph['devices'],row['device_features']):
            if d['model'] in ('nmos','pmos'): assert all(v>0 for v in features[:3])
    dump(folder/'quality_checks.json', {'status':'passed', 'attempts_checked':len(statuses),
        'accepted_rows_checked':len(rows), 'model_files_verified':len(models['files_sha256']),
        'labels_recomputed_from_waves':True,'parameter_files_reconciled':True,'mos_ports_checked':True,
        'mask_only_missingness':True,'normalization_fitted':False,
        'scope':'Pilot data integrity and named measurements; not a cross-source compatibility certification'})
    print('Generated pilot passed:',len(rows),'samples;',len(models['files_sha256']),'model files verified')


if __name__=='__main__':main()
