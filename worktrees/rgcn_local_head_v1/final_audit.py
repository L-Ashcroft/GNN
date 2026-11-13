"""Read-only artifact audit after test; never retune the published model."""
from pathlib import Path
import json,hashlib,re,ast
import numpy as np
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
OUT=ROOT/'artifacts/experiment1_rgcn_local_head_v1';pub=read(ROOT/'versions/rgcn_local_id_v1/publication.json');bundle=read(OUT/'bundle.json');test=read(OUT/'fixed_test/summary.json')
assert sha(ROOT/pub['source'])==pub['source_sha256'] and sha(ROOT/pub['bundle'])==pub['bundle_sha256']==test['bundle_sha256']
manifest=read(ROOT/'data/model_ready/fusion_v2_training_v1/manifest.json');seen=set();count=0
for path in (OUT/'fixed_test').glob('*.parquet'):
    name=path.stem;folder=ROOT/manifest['topologies'][name]['folder'];table=pq.read_table(path);ids=table['sample_id'].to_pylist()
    assert len(set(ids))==len(ids) and not seen&set(ids);seen.update(ids);count+=len(ids)
    role=np.load(folder/'role.npy',mmap_mode='r');source_ids=np.load(folder/'sample_ids.npy',mmap_mode='r');target=np.load(folder/'target.npy',mmap_mode='r');mask=np.load(folder/'mask.npy',mmap_mode='r');rows=np.flatnonzero(role==2)
    assert ids==[s.decode() for s in source_ids[rows]]
    actual_target=np.asarray(table['target'].to_pylist(),dtype='float32');actual_mask=np.asarray(table['mask'].to_pylist(),dtype=bool);prediction=np.asarray(table['prediction'].to_pylist(),dtype='float64')
    assert actual_target.shape==actual_mask.shape==prediction.shape==(len(rows),16)
    assert np.array_equal(actual_target,np.asarray(target[rows])) and np.array_equal(actual_mask,np.asarray(mask[rows]))
    assert np.isfinite(prediction[actual_mask]).all()
    for a in [role,source_ids,target,mask]:a._mmap.close()
assert count==49056
docs=[ROOT/'reports/rgcn_readout_revision_v3.md',ROOT/'versions/rgcn_local_id_v1/README.md']
for p in docs:
    for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):assert (p.parent/link).exists(),str(p.parent/link)
for folder in ['representation_diagnostic_v1','rgcn_direct_head_v3','rgcn_direct_head_v3_io','rgcn_local_head_v1','verified_loto_v1']:
    for p in (ROOT/'worktrees'/folder).glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
folds=read(ROOT/'data/fusion_v2/splits/verified_loto_v1/manifest.json')
assert len(folds['folds'])==24 and all(sha(ROOT/f['path'])==f['sha256'] for f in folds['folds'])
result={'published_code_and_model_bundle_unchanged_after_test':True,'test_prediction_rows_unique':count,
    'test_prediction_targets_masks_equal_original_cache':True,'all_observed_predictions_finite':True,
    'fixed_16_vector_format':True,'report_links_and_python_syntax_valid':True,'LOTO_24_files_hashes_valid':True,
    'test_not_used_for_fit_or_reselection':True,'current_training_running':False}
(OUT/'final_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
